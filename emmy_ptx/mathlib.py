"""Accurate single-precision math functions, built from PTX primitives.

The hardware offers fast approximations (``ex2.approx``, ``lg2.approx``,
``sin.approx`` ...) whose error grows with the argument. Each function here
pairs a careful argument reduction with either a hardware primitive used only
on a small, safe interval or a fitted polynomial, so the results stay within
a few ulp over the whole float range. That matches the accuracy of CUDA's
own ``expf``/``logf``/... which the problem tolerances were calibrated on.
"""

from __future__ import annotations

from .builder import Builder, Value

LOG2E_HI = 1.44269502162933349609375  # float32(log2(e))
LOG2E_LO = 1.925963033500011e-08  # log2(e) - LOG2E_HI


def exp(b: Builder, x: Value) -> Value:
    """e^x = 2^j * 2^f with j = rint(x*log2 e) and f the (small) remainder.

    f is computed with two FMAs against a split log2(e) so that it carries
    almost no rounding error even for |x| near 88; 2^f comes from
    ``ex2.approx`` on |f| <= 1/2, and 2^j is assembled in the exponent field.
    j is clamped to [-126, 126] so 2^j stays a normal number; arguments beyond
    that just leave a larger f, which ``ex2.approx`` turns into inf or a tiny
    value, and the final multiply rounds into the subnormal range correctly.
    """
    if x.const is not None:
        import math

        try:
            return b.f32_const(math.exp(x.const))
        except OverflowError:
            return b.f32_const(float("inf"))
    t = b.fbin("*", x, b.f32_const(LOG2E_HI))
    j = b.fop("cvt.rni.f32.f32", t)
    j = b.fop("min.f32", b.fop("max.f32", j, b.f32_const(-126.0)), b.f32_const(126.0))
    f = b.fma(x, b.f32_const(LOG2E_HI), b.fop("neg.f32", j))
    f = b.fma(x, b.f32_const(LOG2E_LO), f)
    p = b.fop("ex2.approx.ftz.f32", f)
    e = b.assume(b.convert(j, "s32"), -126, 126)
    scale = b.from_bits(b.shl(b.add(e, b.int_const(127)), b.int_const(23)))
    return b.fbin("*", p, scale)


def exp_fast(b: Builder, x: Value) -> Value:
    """CUDA's ``__expf``: one ``ex2.approx`` of x*log2(e) (fast, less accurate)."""
    return b.fop("ex2.approx.ftz.f32", b.fop("mul.ftz.f32", b.in_reg(x), b.f32_const(LOG2E_HI)))


def sigmoid(b: Builder, x: Value) -> Value:
    """emmy's spelling: 1 / (1 + e^(0 - x))."""
    e = exp(b, b.fbin("-", b.f32_const(0.0), x))
    return b.fop("rcp.rn.f32", b.fbin("+", b.f32_const(1.0), e))


def _todo(name: str):
    def fn(b: Builder, *args: Value) -> Value:
        raise NotImplementedError(f"math function {name} is not implemented yet")
    return fn


log = _todo("log")
tanh = _todo("tanh")
erf = _todo("erf")
sin = _todo("sin")
cos = _todo("cos")
pow = _todo("pow")
