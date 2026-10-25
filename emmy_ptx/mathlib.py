"""Accurate math functions, built from PTX primitives: single precision, plus
double-precision exp, log and sigmoid.

The hardware offers fast approximations (``ex2.approx``, ``lg2.approx``,
``sin.approx`` ...) whose error grows with the argument. Each function here
pairs an exact or near-exact argument reduction with a polynomial (or a
hardware primitive used only on a small safe interval), so results stay within
a few ulp over the whole float range, like CUDA's own ``expf``/``logf``/...
that the problem tolerances were calibrated against.

The polynomial coefficients are minimax fits produced by ``tools/fit_poly.py``
(which also reports each kernel's worst-case error in float32 ulps), and every
polynomial is evaluated with fused multiply-adds (Horner's rule).
"""

from __future__ import annotations

import math

from .builder import Builder, Value

LOG2E_HI = 1.44269502162933349609375  # float32(log2(e))
LOG2E_LO = 1.925963033500011e-08  # log2(e) - LOG2E_HI
LN2_HI = 0.693145751953125  # ln(2) with 15 significant bits: e * LN2_HI is exact
LN2_LO = 1.4286067653e-06  # ln(2) - LN2_HI
INF, NAN = float("inf"), float("nan")

# log(1 + f) = f + f^2 * P(f) on [sqrt(1/2) - 1, sqrt(2) - 1]   (0.76 ulp)
LOG1P = (-0.49999988079071045, 0.33333325386047363, -0.25001582503318787, 0.20001980662345886,
         -0.16609081625938416, 0.14181727170944214, -0.13242870569229126, 0.1290489137172699,
         -0.07622590661048889)
# tanh(x) = x + x^3 * P(x^2) on [0, 0.6]   (0.71 ulp)
TANH = (-0.3333333134651184, 0.13333246111869812, -0.0539521723985672, 0.021732749417424202,
        -0.008262819610536098, 0.002227049320936203)
# erf(x) = x + x * P(x^2) on [0, 1]   (0.98 ulp)
ERF = (0.12837916612625122, -0.3761262595653534, 0.1128358542919159, -0.026853814721107483,
       0.005188334733247757, -0.0008010261226445436, 7.854098657844588e-05)
# log(erfc(a)) = P(a) on [1, 4]   (0.93 ulp in erf)
LOG_ERFC = (-0.0008293656865134835, -1.1244639158248901, -0.6445020437240601, -0.09402139484882355,
            0.013422789983451366, 0.002270197495818138, -0.001978313084691763, 0.000585538859013468,
            -9.77453964878805e-05, 9.0702815214172e-06, -3.650506243957352e-07)
# sin(r) = r + r^3 * P(r^2), cos(r) = 1 + r^2 * Q(r^2) on [0, 0.8]   (0.68 / 0.85 ulp)
SIN = (-0.1666666716337204, 0.008333329111337662, -0.00019839162996504456, 2.7167186544829747e-06)
COS = (-0.5, 0.0416666679084301, -0.0013888879911974072, 2.4798757294774987e-05, -2.716033691285702e-07)

# pi/2 = PIO2_A + PIO2_B + PIO2_C to ~1e-23 (Cody-Waite reduction with FMAs)
PIO2_A, PIO2_B, PIO2_C = 1.5707963705062866, -4.371138828673793e-08, -1.7151245100058819e-15
TWO_OVER_PI = 0.6366197723675814
TRIG_FAST_LIMIT = 105615.0  # beyond this the float32 reduction loses bits; use float64
PIO2_HI_D, PIO2_LO_D = 1.5707963267948966, 6.123233995736766e-17

# float64: ln 2 = LN2_HI_D + LN2_LO_D, LN2_HI_D with 32 significant bits so k * LN2_HI_D is
# exact for |k| < 2^21 (fdlibm's split)
LN2_HI_D, LN2_LO_D = 6.93147180369123816490e-01, 1.90821492927058770002e-10
SQRT_HALF_BITS = 0x3FE6A09E667F3BCD  # float64 bits of sqrt(1/2)
DBL_MIN = 2.2250738585072014e-308  # smallest normal float64


def _c(b: Builder, x: float) -> Value:
    return b.f32_const(x)


def horner(b: Builder, coeffs, u: Value) -> Value:
    """sum(coeffs[i] * u**i) as a chain of FMAs."""
    acc = _c(b, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = b.fma(acc, u, _c(b, c))
    return acc


def _fold(b: Builder, fn, *args: Value) -> Value | None:
    """Constant-fold a call whose arguments are all known (float64 then rounded)."""
    if all(a.const is not None for a in args):
        try:
            return b.f32_const(fn(*(float(a.const) for a in args)))
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
    return None


def _keep_nan(b: Builder, x: Value, r: Value) -> Value:
    """``r``, except NaN when ``x`` is NaN (min/max would otherwise drop it)."""
    return b.select(b.fcmp("==", x, x), r, x)


# ---------------------------------------------------------------- exponential


def exp(b: Builder, x: Value) -> Value:
    """e^x = 2^j * 2^f with j = rint(x*log2 e) and f the (small) remainder.

    f is computed with two FMAs against a split log2(e), so it carries almost no
    rounding error even for |x| near 88; 2^f comes from ``ex2.approx`` on
    |f| <= 1/2, and 2^j is assembled in the exponent field. j is clamped to
    [-126, 126] so 2^j stays a normal number; larger arguments just leave a
    larger f, which ``ex2.approx`` turns into inf or a tiny value, and the
    final multiply rounds into the subnormal range correctly.
    """
    folded = _fold(b, lambda v: math.exp(v) if v < 709 else INF, x)
    if folded is not None:
        return folded
    pair = _rescale_pair(b, x)
    if pair is not None:
        return pair
    return _exp(b, x)


def _rescale_pair(b: Builder, x: Value) -> Value | None:
    """exp(x) for x = u - max(u, w): one of an online-softmax step's two
    rescaling factors, exp(u - m) and exp(w - m) with m = max(u, w).

    One of the two exponents is exactly 0 (m is u or w), so their sum is
    exactly the other exponent, and both factors come from one exponential
    of that sum: exp(x) = 1 if x == 0, else exp(x + (w - m)). The sum is the
    same value for both factors, so value numbering shares the exponential.
    Bit for bit what the two separate exponentials compute.
    """
    d = b.def_of(x)
    if d is None or d[0] != "sub.rn.f32":
        return None
    u, m = d[1]
    md = b.def_of(m)
    if md is None or md[0] != "max.f32":
        return None
    p, q = md[1]
    if u.key() == p.key():
        w = q
    elif u.key() == q.key():
        w = p
    else:
        return None
    other = b.fbin("+", x, b.fbin("-", w, m))
    return b.select(b.fcmp("==", x, _c(b, 0.0)), _c(b, 1.0), _exp(b, other))


def _exp(b: Builder, x: Value) -> Value:
    t = b.fbin("*", x, _c(b, LOG2E_HI))
    j = b.fop("cvt.rni.f32.f32", t)
    j = b.fop("min.f32", b.fop("max.f32", j, _c(b, -126.0)), _c(b, 126.0))
    f = b.fma(x, _c(b, LOG2E_HI), b.fop("neg.f32", j))
    f = b.fma(x, _c(b, LOG2E_LO), f)
    p = b.fop("ex2.approx.ftz.f32", f)
    e = b.assume(b.convert(j, "s32"), -126, 126)
    scale = b.from_bits(b.shl(b.add(e, b.int_const(127)), b.int_const(23)))
    return b.fbin("*", p, scale)


def exp_fast(b: Builder, x: Value) -> Value:
    """CUDA's ``__expf``: one ``ex2.approx`` of x*log2(e) (fast, less accurate)."""
    return b.fop("ex2.approx.ftz.f32", b.fop("mul.ftz.f32", b.in_reg(x), _c(b, LOG2E_HI)))


def sigmoid(b: Builder, x: Value) -> Value:
    """emmy's spelling: 1 / (1 + e^(0 - x))."""
    e = exp(b, b.fbin("-", _c(b, 0.0), x))
    return b.fop("rcp.rn.f32", b.fbin("+", _c(b, 1.0), e))


# ---------------------------------------------------------------- logarithm


def _split_mantissa(b: Builder, x: Value):
    """x >= 0 finite, nonzero  ->  (e, m) with x = m * 2^e, m in [sqrt(1/2), sqrt(2)).

    Subnormals are first scaled by 2^23. ``e`` is returned as an f32.
    """
    tiny = b.fcmp("<", x, _c(b, 1.1754943508222875e-38))
    xs = b.select(tiny, b.fbin("*", x, _c(b, 8388608.0)), x)
    i = b.bitop("&", b.bits(xs), b.int_const(0x7FFFFFFF))
    t = b.sub(i, b.int_const(0x3F3504F3))  # 0x3F3504F3 = bits of sqrt(1/2)
    e = b.shr(t, b.int_const(23))
    # x / 2^e: keep t's mantissa bits and re-bias them onto [sqrt(1/2), sqrt(2))
    m = b.from_bits(b.add(b.bitop("&", t, b.int_const(0x7FFFFF)), b.int_const(0x3F3504F3)))
    ef = b.fbin("-", b.convert(e, "f32"), b.select(tiny, _c(b, 23.0), _c(b, 0.0)))
    return ef, m


def log(b: Builder, x: Value) -> Value:
    """log x = e*ln2 + log1p(m - 1), with x = m * 2^e and m - 1 exact."""
    folded = _fold(b, math.log, x)
    if folded is not None:
        return folded
    ef, m = _split_mantissa(b, x)
    f = b.fbin("-", m, _c(b, 1.0))
    r = b.fma(b.fbin("*", f, f), horner(b, LOG1P, f), f)
    y = b.fma(ef, _c(b, LN2_HI), b.fma(ef, _c(b, LN2_LO), r))
    y = b.select(b.fcmp("==", x, _c(b, INF)), _c(b, INF), y)
    y = b.select(b.fcmp("==", x, _c(b, 0.0)), _c(b, -INF), y)
    return b.select(b.fcmp(">=", x, _c(b, 0.0)), y, _c(b, NAN))  # x < 0 or NaN


# ---------------------------------------------------------------- tanh, erf


def tanh(b: Builder, x: Value) -> Value:
    """Odd polynomial below 0.6; 1 - 2/(e^(2|x|) + 1) with x's sign above."""
    folded = _fold(b, math.tanh, x)
    if folded is not None:
        return folded
    a = b.fop("abs.f32", b.in_reg(x))
    x2 = b.fbin("*", x, x)
    small = b.fop("copysign.f32", x, b.fma(b.fbin("*", x, x2), horner(b, TANH, x2), x))  # tanh(-0) = -0
    e = exp(b, b.fbin("+", a, a))
    big = b.fbin("-", _c(b, 1.0), b.fbin("*", _c(b, 2.0), b.fop("rcp.rn.f32", b.fbin("+", e, _c(b, 1.0)))))
    big = b.fop("copysign.f32", x, big)
    return b.select(b.fcmp("<", a, _c(b, 0.6)), small, big)


def erf(b: Builder, x: Value) -> Value:
    """Odd polynomial below 1; 1 - exp(P(|x|)) with P ~ log(erfc) above."""
    folded = _fold(b, math.erf, x)
    if folded is not None:
        return folded
    a = b.fop("abs.f32", b.in_reg(x))
    small = b.fma(x, horner(b, ERF, b.fbin("*", x, x)), x)
    q = horner(b, LOG_ERFC, b.fop("min.f32", a, _c(b, 4.0)))  # erf(4) rounds to 1
    big = b.fop("copysign.f32", x, b.fbin("-", _c(b, 1.0), exp(b, q)))
    return _keep_nan(b, x, b.select(b.fcmp("<", a, _c(b, 1.0)), small, big))


# ---------------------------------------------------------------- sin, cos


def _reduce_half_pi(b: Builder, x: Value):
    """x = k*pi/2 + r with |r| <= ~pi/4 and the quadrant k (an int).

    Three-constant Cody-Waite reduction with FMAs for |x| < 105615; beyond
    that a float64 reduction (accurate to |x| ~ 2^40) on a rarely taken branch.
    """
    k = b.fop("cvt.rni.s32.f32", b.fbin("*", x, _c(b, TWO_OVER_PI)), type_="s32")
    nk = b.fop("neg.f32", b.convert(k, "f32"))
    r = b.fma(nk, _c(b, PIO2_A), x)
    r = b.fma(nk, _c(b, PIO2_B), r)
    r = b.fma(nk, _c(b, PIO2_C), r)
    r_var, k_var = b.variable("f32", r), b.variable("s32", k)
    fast = b.label("trig_fast")
    b.branch(fast, b.fcmp(">", b.fop("abs.f32", b.in_reg(x)), _c(b, TRIG_FAST_LIMIT)), negate=True)
    with b.guarded(), b.cse_scope():
        xd = b.convert(x, "f64")
        kd = b.fop("cvt.rni.f64.f64", b.fop("mul.rn.f64", xd, _f64(b, TWO_OVER_PI), type_="f64"), type_="f64")
        nkd = b.fop("neg.f64", kd, type_="f64")
        rd = b.fop("fma.rn.f64", nkd, _f64(b, PIO2_HI_D), xd, type_="f64")
        rd = b.fop("fma.rn.f64", nkd, _f64(b, PIO2_LO_D), rd, type_="f64")
        b.assign(r_var, b.convert(rd, "f32"))
        kq = b.fop("cvt.rzi.s64.f64", kd, type_="s64")
        b.assign(k_var, b.convert(b.bitop("&", kq, b.int_const(3, "s64")), "s32"))
    b.place(fast)
    return r_var.read(), k_var.read()


def _f64(b: Builder, v: float) -> Value:
    return b.float_const(v, "f64")


def _sincos_kernels(b: Builder, r: Value):
    r2 = b.fbin("*", r, r)
    s = b.fma(b.fbin("*", r, r2), horner(b, SIN, r2), r)
    c = b.fma(r2, horner(b, COS, r2), _c(b, 1.0))
    return s, c


def sin(b: Builder, x: Value) -> Value:
    folded = _fold(b, math.sin, x)
    if folded is not None:
        return folded
    r, k = _reduce_half_pi(b, x)
    s, c = _sincos_kernels(b, r)
    v = b.select(b.icmp("!=", b.bitop("&", k, b.int_const(1)), b.int_const(0)), c, s)
    neg = b.icmp("!=", b.bitop("&", k, b.int_const(2)), b.int_const(0))
    return b.select(neg, b.fop("neg.f32", v), v)


def cos(b: Builder, x: Value) -> Value:
    folded = _fold(b, math.cos, x)
    if folded is not None:
        return folded
    r, k = _reduce_half_pi(b, x)
    s, c = _sincos_kernels(b, r)
    v = b.select(b.icmp("!=", b.bitop("&", k, b.int_const(1)), b.int_const(0)), s, c)
    neg = b.icmp("!=", b.bitop("&", b.add(k, b.int_const(1)), b.int_const(2)), b.int_const(0))
    return b.select(neg, b.fop("neg.f32", v), v)


# ---------------------------------------------------------------- float64 exp, log


def _fold64(b: Builder, fn, x: Value) -> Value | None:
    if x.const is None:
        return None
    try:
        return _f64(b, fn(float(x.const)))
    except (OverflowError, ValueError):
        return None


def _horner64(b: Builder, coeffs, u: Value) -> Value:
    acc = _f64(b, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = b.fma(acc, u, _f64(b, c))
    return acc


def _pow2_64(b: Builder, k: Value) -> Value:
    """2^k as a float64, for an int k in the normal exponent range."""
    return b.pure("mov.b64", "f64", (b.shl(b.convert(b.add(k, b.int_const(1023)), "s64"), b.int_const(52)),))


def _exp_or_inf(v: float) -> float:
    try:
        return math.exp(v)
    except OverflowError:
        return INF


EXP64 = tuple(1.0 / math.factorial(k) for k in range(14))  # e^r to r^13: below 0.02 ulp on |r| <= ln2/2


def exp64(b: Builder, x: Value) -> Value:
    """e^x in float64: 2^j * e^r with j = rint(x * log2 e), r = x - j*ln2 by
    two FMAs against fdlibm's split ln 2, and e^r by its Taylor series.

    x is first clamped to [-746, 710], past which e^x rounds to 0 or inf, so j
    stays in [-1077, 1025]; 2^j is applied as two powers of two, so a result in
    the subnormal range is rounded once, by the last multiply.
    """
    folded = _fold64(b, _exp_or_inf, x)
    if folded is not None:
        return folded
    xc = b.fop("min.f64", b.fop("max.f64", b.in_reg(x), _f64(b, -746.0), type_="f64"), _f64(b, 710.0), type_="f64")
    j = b.fop("cvt.rni.f64.f64", b.fbin("*", xc, _f64(b, 1.0 / math.log(2.0))), type_="f64")
    nj = b.fop("neg.f64", j, type_="f64")
    r = b.fma(nj, _f64(b, LN2_HI_D), xc)
    r = b.fma(nj, _f64(b, LN2_LO_D), r)
    k = b.assume(b.fop("cvt.rzi.s32.f64", j, type_="s32"), -1077, 1025)
    half = b.shr(k, b.int_const(1))
    y = b.fbin("*", b.fbin("*", _horner64(b, EXP64, r), _pow2_64(b, half)), _pow2_64(b, b.sub(k, half)))
    return _keep_nan(b, x, y)


# log(1 + f) = 2 atanh(s), s = f / (2 + f): R(z) = sum 2 z^k / (2k + 1), z = s^2 <= 0.0295
LOG64 = tuple(2.0 / (2 * k + 1) for k in range(1, 12))


def log64(b: Builder, x: Value) -> Value:
    """log x in float64: x = m * 2^e with m in [sqrt(1/2), sqrt(2)) (subnormals
    scaled by 2^54 first), and log m = log(1 + f) = f - (f^2/2 - s(f^2/2 + R))
    with s = f / (2 + f), fdlibm's arrangement, in which the large term f is
    exact and only small corrections round."""
    folded = _fold64(b, math.log, x)
    if folded is not None:
        return folded
    tiny = b.fcmp("<", x, _f64(b, DBL_MIN))
    xs = b.select(tiny, b.fbin("*", x, _f64(b, 2.0**54)), x)
    i = b.bitop("&", b.pure("mov.b64", "s64", (b.in_reg(xs),)), b.int_const((1 << 63) - 1, "s64"))
    t = b.sub(i, b.int_const(SQRT_HALF_BITS, "s64"))
    e = b.shr(t, b.int_const(52))
    m = b.pure("mov.b64", "f64", (b.add(b.bitop("&", t, b.int_const((1 << 52) - 1, "s64")),
                                        b.int_const(SQRT_HALF_BITS, "s64")),))
    ef = b.fbin("-", b.convert(e, "f64"), b.select(tiny, _f64(b, 54.0), _f64(b, 0.0)))
    f = b.fbin("-", m, _f64(b, 1.0))
    s = b.fbin("/", f, b.fbin("+", f, _f64(b, 2.0)))
    z = b.fbin("*", s, s)
    hfsq = b.fbin("*", _f64(b, 0.5), b.fbin("*", f, f))
    r = b.fbin("*", z, _horner64(b, LOG64, z))
    log1p = b.fbin("-", f, b.fbin("-", hfsq, b.fbin("*", s, b.fbin("+", hfsq, r))))
    y = b.fma(ef, _f64(b, LN2_HI_D), b.fma(ef, _f64(b, LN2_LO_D), log1p))
    y = b.select(b.fcmp("==", x, _f64(b, INF)), _f64(b, INF), y)
    y = b.select(b.fcmp("==", x, _f64(b, 0.0)), _f64(b, -INF), y)
    return b.select(b.fcmp(">=", x, _f64(b, 0.0)), y, _f64(b, NAN))  # x < 0 or NaN


def sigmoid64(b: Builder, x: Value) -> Value:
    """emmy's spelling, 1 / (1 + e^(0 - x)), in float64."""
    e = exp64(b, b.fbin("-", _f64(b, 0.0), x))
    return b.fop("rcp.rn.f64", b.fbin("+", _f64(b, 1.0), e), type_="f64")


# ---------------------------------------------------------------- pow


def _pow_const(b: Builder, x: Value, y: float) -> Value | None:
    """x^y for a compile-time exponent with a short exact-or-nearly-exact form.

    Integer powers are products (x*x is correctly rounded, x*x*x within
    1.5 ulp), and the IEEE products already give C's special cases for
    zeros, infinities and NaN. Half powers use the correctly rounded sqrt,
    with the two inputs where C's pow and sqrt disagree (-0 and -inf) fixed up.
    """
    if y == 0:
        return _c(b, 1.0)  # pow(x, 0) is 1 even for NaN
    if y == 1:
        return x
    if y == 2:
        return b.fbin("*", x, x)
    if y == 3:
        return b.fbin("*", b.fbin("*", x, x), x)
    if y == 4:
        sq = b.fbin("*", x, x)
        return b.fbin("*", sq, sq)
    if y == -1:
        return b.fop("rcp.rn.f32", b.in_reg(x))
    if y == -2:
        return b.fop("rcp.rn.f32", b.fbin("*", x, x))
    if y in (0.5, 1.5, -0.5):
        root = b.fbin("+", b.fop("sqrt.rn.f32", b.in_reg(x)), _c(b, 0.0))  # +0 for -0, as pow gives
        r = root if y == 0.5 else b.fbin("*", x, root) if y == 1.5 else b.fop("rcp.rn.f32", root)
        minus_inf = b.fcmp("==", x, _c(b, -INF))  # pow(-inf, y) is +inf (y > 0) or +0 (y < 0)
        return b.select(minus_inf, _c(b, INF if y > 0 else 0.0), r)
    return None


def pow(b: Builder, x: Value, y: Value) -> Value:
    """C ``powf``: |x|^y = 2^(y * log2|x|) evaluated in float64, plus the C99
    special cases (zeros, infinities, NaNs, negative bases with integer y).

    float64 gives y*log2|x| the ~40 bits a correctly rounded float32 result
    needs; log2 uses the atanh series and 2^t a Taylor series, both far past
    float32 accuracy, so the final rounding to float32 is the only real error.
    """
    folded = _fold(b, lambda u, v: math.pow(u, v), x, y)
    if folded is not None:
        return folded
    if y.const is not None:
        special = _pow_const(b, x, float(y.const))
        if special is not None:
            return special
    ax = b.fop("abs.f32", b.in_reg(x))
    ef, m = _split_mantissa(b, ax)
    md = b.convert(m, "f64")
    one = _f64(b, 1.0)
    s = b.fop("div.rn.f64", b.fop("sub.rn.f64", md, one, type_="f64"), b.fop("add.rn.f64", md, one, type_="f64"), type_="f64")
    s2 = b.fop("mul.rn.f64", s, s, type_="f64")
    series = _f64(b, 1.0 / 13.0)
    for k in (11, 9, 7, 5, 3, 1):
        series = b.fop("fma.rn.f64", series, s2, _f64(b, 1.0 / k), type_="f64")
    ln_m = b.fop("mul.rn.f64", b.fop("add.rn.f64", s, s, type_="f64"), series, type_="f64")
    log2x = b.fop("fma.rn.f64", ln_m, _f64(b, 1.0 / math.log(2.0)), b.convert(ef, "f64"), type_="f64")
    t = b.fop("mul.rn.f64", b.convert(y, "f64"), log2x, type_="f64")
    t = b.fop("min.f64", b.fop("max.f64", t, _f64(b, -200.0), type_="f64"), _f64(b, 200.0), type_="f64")
    j = b.fop("cvt.rni.f64.f64", t, type_="f64")
    u = b.fop("mul.rn.f64", b.fop("sub.rn.f64", t, j, type_="f64"), _f64(b, math.log(2.0)), type_="f64")
    p = _f64(b, 1.0 / math.factorial(12))
    for k in range(11, -1, -1):
        p = b.fop("fma.rn.f64", p, u, _f64(b, 1.0 / math.factorial(k)), type_="f64")
    ji = b.assume(b.fop("cvt.rzi.s32.f64", j, type_="s32"), -200, 200)
    scale = b.pure("mov.b64", "f64", (b.shl(b.convert(b.add(ji, b.int_const(1023)), "s64"), b.int_const(52)),))
    r = b.convert(b.fop("mul.rn.f64", p, scale, type_="f64"), "f32")

    # y's integer-ness and parity decide the sign and NaN cases
    y_int = b.fcmp("==", b.fop("cvt.rni.f32.f32", b.in_reg(y)), y)
    small_y = b.fcmp("<", b.fop("abs.f32", b.in_reg(y)), _c(b, 16777216.0))
    y_odd = b.pand(b.pand(y_int, small_y),
                   b.icmp("!=", b.bitop("&", b.fop("cvt.rzi.s32.f32", b.in_reg(y), type_="s32"), b.int_const(1)), b.int_const(0)))
    y_neg = b.fcmp("<", y, _c(b, 0.0))
    # |x| == 0 or inf: the exponent's sign decides between 0 and inf
    r = b.select(b.fcmp("==", ax, _c(b, 0.0)), b.select(y_neg, _c(b, INF), _c(b, 0.0)), r)
    r = b.select(b.fcmp("==", ax, _c(b, INF)), b.select(y_neg, _c(b, 0.0), _c(b, INF)), r)
    x_signed = b.icmp("<", b.bits(x), b.int_const(0))  # includes -0 and -inf
    r = b.select(b.pand(x_signed, y_odd), b.fop("neg.f32", r), r)
    y_inf = b.fcmp("==", b.fop("abs.f32", b.in_reg(y)), _c(b, INF))
    r = b.select(b.pand(b.fcmp("==", ax, _c(b, 1.0)), y_inf), _c(b, 1.0), r)
    bad_base = b.pand(b.fcmp("<", x, _c(b, 0.0)), b.pand(b.pnot(y_int), b.pnot(y_inf)))
    r = b.select(bad_base, _c(b, NAN), r)
    r = b.select(b.por(b.fcmp("!=", x, x), b.fcmp("!=", y, y)), _c(b, NAN), r)
    r = b.select(b.fcmp("==", x, _c(b, 1.0)), _c(b, 1.0), r)
    return b.select(b.fcmp("==", y, _c(b, 0.0)), _c(b, 1.0), r)
