"""Elementwise operations (``Assign``, ``Accum`` and combine programs).

Semantics follow emmy's CUDA rendering (``op_to_expr`` plus the CUDA math
library). Basic arithmetic is IEEE rounded to nearest, so a pointwise kernel
reproduces numpy's results bit for bit; the transcendental functions come from
:mod:`emmy_ptx.mathlib`.

Result types follow emmy's ``dtype_promote`` unless the IR gives one:
integer arithmetic and bit operations on integer operands stay integer,
all-f16 operands stay f16 (we compute them in f32 and round once), everything
else is f32. An operation is computed in its result's type: an f64 result in
double precision (emmy's CUDA computes those in float too; numpy, whose
answers the problems check, does not). Of the transcendental functions only
exp, log and sigmoid have double-precision versions; the others compute an
f64 result in single precision.
"""

from __future__ import annotations

import math

import numpy as np

from . import mathlib
from .builder import NUMPY_FLOAT, Builder, Value, float_type
from .ptx import Imm

INTEGER_DTYPES = frozenset({"i16", "i32", "i64", "u8", "u16", "u32", "u64"})
# fp8 formats: a value is its byte code in an s32 register; no arithmetic runs on it, only the
# decode cast (from_f8*) out of it and the encode cast (to_f8*) into it
F8_DTYPES = frozenset({"f8e4m3", "f8e5m2"})
INTEGER_OPS = frozenset({
    "add", "subtract", "multiply", "floor_divide", "remainder", "mod",
    "left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_count",
})
_WIDTH = {"u8": 8, "i16": 16, "u16": 16, "i32": 32, "u32": 32, "i64": 64, "u64": 64}


def value_type(dtype: str) -> str:
    """The register type a value of emmy dtype ``dtype`` is computed in. The
    type carries the signedness: unsigned values divide, shift, compare and
    widen as unsigned numbers."""
    if dtype in ("f32", "bf16"):
        return "f32"
    if dtype in ("f16", "f64"):
        return dtype
    if dtype in ("i64", "u64"):
        return "s" + dtype[1:] if dtype[0] == "i" else dtype
    if dtype in ("u8", "u16", "u32"):
        return "u32"
    if dtype in INTEGER_DTYPES or dtype in F8_DTYPES or dtype == "bool":
        return "s32"
    raise ValueError(f"unsupported dtype {dtype!r}")


def promote(op: str, dtypes: list[str]) -> str:
    """emmy's ``dtype_promote`` rule for the result dtype of ``op``."""
    if dtypes and op in INTEGER_OPS and all(d in INTEGER_DTYPES for d in dtypes):
        signed = max((_WIDTH[d] for d in dtypes if d.startswith("i")), default=0)
        unsigned = max((_WIDTH[d] for d in dtypes if d.startswith("u")), default=0)
        return f"u{unsigned}" if unsigned >= signed else f"i{signed}"
    if dtypes and all(d == "f16" for d in dtypes):
        return "f16"
    return "f32"


def apply(b: Builder, op: str, args: list[Value], dtype: str) -> Value:
    """``op(*args)`` with result dtype ``dtype``."""
    if op in _F8_CASTS:
        return _F8_CASTS[op](b, args[0])
    out_type = value_type(dtype)
    if op in _COMPARE:
        # C compares the operands in their own types (integers exactly, an f64 in
        # double); the result is the 0 or 1 of the test, in the result's type
        x, y = args
        test = b.icmp(_COMPARE[op], x, y) if x.is_int and y.is_int else b.fcmp(_COMPARE[op], x, y)
        return b.convert(test, out_type)
    if out_type in INT_VALUE_TYPES and op in _INT:
        if op == "where":  # the condition keeps its own type
            args = [args[0], *(b.convert(a, out_type) for a in args[1:])]
        elif op in ("left_shift", "right_shift"):  # the shift amount is a u32 operand
            args = [b.convert(args[0], out_type), b.convert(args[1], "u32")]
        else:
            args = [b.convert(a, out_type) for a in args]
        return _narrow(b, _INT[op](b, out_type, *args), dtype)
    double = out_type == "f64" and (op in _DOUBLE or op not in _SINGLE)
    fn = _DOUBLE.get(op, _FLOAT.get(op)) if double else _FLOAT.get(op)
    if fn is None:
        raise NotImplementedError(f"elementwise op {op!r}")
    ftype = "f64" if double else "f32"
    if op in ("where",):  # the condition keeps its own type; C tests it for non-zero
        args = [args[0], *(b.convert(a, ftype) for a in args[1:])]
    else:
        args = [b.convert(a, ftype) for a in args]
    return b.convert(fn(b, *args), out_type)


def _decode_f8(fmt: str):
    """``from_f8*``: an fp8 byte code to f32. ``cvt.rn.f16x2.<fmt>x2`` widens a
    pair of codes to two halves exactly (every e4m3 and e5m2 value is an f16;
    e4m3's 0x7f/0xff come out NaN, e5m2's infinities and NaNs stay), and the
    low half is the code's value."""
    def fn(b: Builder, code: Value) -> Value:
        b.k.require("8.1")  # fp8 conversions on sm_89
        pair = b.pure("cvt.u16.u32", "u16", (b.in_reg(b.convert(code, "s32")),))
        halves = b.pure(f"cvt.rn.f16x2.{fmt}x2", "b32", (pair,))
        return b.pure("cvt.f32.f16", "f32", (b.pure("cvt.u16.u32", "u16", (halves,)),))
    return fn


def _encode_f8(fmt: str):
    """``to_f8*``: f32 to an fp8 byte code, rounding to nearest even and
    saturating to the largest finite value, NaN to 0x7f: emmy's encode (the
    ``__NV_SATFINITE`` conversion of cuda_fp8.h), one ``cvt.rn.satfinite``.
    The instruction packs two values; both are x, and the low byte is kept."""
    def fn(b: Builder, x: Value) -> Value:
        b.k.require("8.1")  # fp8 conversions on sm_89
        x = b.in_reg(b.convert(x, "f32"))
        pair = b.pure(f"cvt.rn.satfinite.{fmt}x2.f32", "u16", (x, x))
        return b.bitop("&", b.pure("cvt.u32.u16", "s32", (pair,), 0, 0xFFFF), b.int_const(0xFF))
    return fn


_F8_CASTS = {
    "from_f8e4m3": _decode_f8("e4m3"), "from_f8e5m2": _decode_f8("e5m2"),
    "to_f8e4m3": _encode_f8("e4m3"), "to_f8e5m2": _encode_f8("e5m2"),
}


def _as_float(b: Builder, p: Value) -> Value:
    """A C comparison used as a value: 1.0 or 0.0."""
    return b.select(p, b.f32_const(1.0), b.f32_const(0.0))


def _unary(opcode: str, fold):
    """One instruction, ``opcode.f32`` or ``opcode.f64`` by the operand's type;
    ``fold`` computes it on a constant (a numpy scalar of that type)."""
    def fn(b: Builder, x: Value) -> Value:
        if x.const is not None and fold is not None:
            with np.errstate(all="ignore"):
                return b.float_const(float(fold(NUMPY_FLOAT[x.type](x.const))), x.type)
        return b.fop(f"{opcode}.{x.type}", b.in_reg(x), type_=x.type)
    return fn


def _fmax(x: float, y: float) -> float:
    """emmy's ``maximum`` is CUDA's ``fmaxf``; folded as the GPU's ``max`` computes
    it: a NaN gives the other operand, and -0 is below +0 (Python's ``max``
    keeps the first of two zeros, and a NaN that comes first)."""
    if math.isnan(x) or math.isnan(y):
        return y if math.isnan(x) else x
    if x == y == 0:
        return x if math.copysign(1.0, x) > 0 else y
    return max(x, y)


def _fmin(x: float, y: float) -> float:
    """``minimum``, CUDA's ``fminf``, folded as the GPU's ``min`` (see _fmax)."""
    if math.isnan(x) or math.isnan(y):
        return y if math.isnan(x) else x
    if x == y == 0:
        return x if math.copysign(1.0, x) < 0 else y
    return min(x, y)


def _minmax(opcode: str, pick):
    def fn(b: Builder, x: Value, y: Value) -> Value:
        t = float_type(x, y)
        x, y = b.convert(x, t), b.convert(y, t)
        if x.const is not None and y.const is not None:
            return b.float_const(pick(x.const, y.const), t)
        if x.const is not None:
            x, y = y, x
        return b.pure(f"{opcode}.{t}", t, (x, y), commutative=True)
    return fn


def _sqrt(v):
    """IEEE square root of a numpy scalar (NaN below zero)."""
    return np.sqrt(v) if v >= 0 else np.nan


_FLOAT = {
    "add": lambda b, x, y: b.fbin("+", x, y),
    "subtract": lambda b, x, y: b.fbin("-", x, y),
    "multiply": lambda b, x, y: b.fbin("*", x, y),
    "divide": lambda b, x, y: b.fbin("/", x, y),
    "maximum": _minmax("max", _fmax),
    "minimum": _minmax("min", _fmin),
    "negative": _unary("neg", lambda v: -v),
    "abs": _unary("abs", abs),
    "copy": lambda b, x: x,
    "pad": lambda b, x: x,
    "square": lambda b, x: b.fbin("*", x, x),
    "reciprocal": _unary("rcp.rn", lambda v: 1 / v),
    "relu": lambda b, x: _minmax("max", _fmax)(b, x, b.f32_const(0.0)),
    "sqrt": _unary("sqrt.rn", _sqrt),
    "rsqrt": _unary("rsqrt.approx", None),
    "exp": mathlib.exp,
    "exp_fast": mathlib.exp_fast,
    "log": mathlib.log,
    "sigmoid": mathlib.sigmoid,
    "tanh": mathlib.tanh,
    "erf": mathlib.erf,
    "sin": mathlib.sin,
    "cos": mathlib.cos,
    "pow": mathlib.pow,
    "where": lambda b, c, x, y: b.select(b.truthy(c), x, y),
    "bitwise_and": lambda b, x, y: _as_float(b, b.pand(b.truthy(x), b.truthy(y))),
    "bitwise_or": lambda b, x, y: _as_float(b, b.por(b.truthy(x), b.truthy(y))),
    "bitwise_not": lambda b, x: _as_float(b, b.fcmp("==", x, b.f32_const(0.0))),
}
# the single-precision routines of _FLOAT (the rest work in either precision), and the
# double-precision versions that exist
_SINGLE = frozenset({"rsqrt", "exp", "exp_fast", "log", "sigmoid", "tanh", "erf", "sin", "cos", "pow"})
_DOUBLE = {
    "rsqrt": lambda b, x: _FLOAT["reciprocal"](b, _FLOAT["sqrt"](b, x)),
    "exp": mathlib.exp64,
    "exp_fast": mathlib.exp64,
    "log": mathlib.log64,
    "sigmoid": mathlib.sigmoid64,
}

# ---- integers: fixed-width two's-complement arithmetic of the result type (like numpy's
# arrays), signed or unsigned by the type; constants fold with the same wrap-around

INT_VALUE_TYPES = frozenset({"s32", "s64", "u32", "u64"})
_COMPARE = {"equal": "==", "not_equal": "!=", "greater": ">", "less": "<", "greater_equal": ">=", "less_equal": "<="}


def _bits(t: str) -> int:
    return 64 if t in ("s64", "u64") else 32


def _wrapped(t: str, v: int) -> int:
    """``v`` reduced to type ``t``'s width (two's complement for signed types)."""
    bits = _bits(t)
    v &= (1 << bits) - 1
    return v - (1 << bits) if t[0] == "s" and v >> (bits - 1) else v


NARROW_RANGE = {"u8": (0, 255), "u16": (0, 65535), "i8": (-128, 127), "i16": (-32768, 32767)}


def declared(b: Builder, v: Value, dtype: str, literal: bool = False) -> Value:
    """``v`` as the value a definition of emmy dtype ``dtype`` holds (a typed
    Let or Init, an accumulator), converted as C converts it. An integer
    keeps the low bits of a narrow type (C's rule for unsigned types, two's
    complement for signed ones): 255 + 1 in a u8 is 0, though its register
    holds 32 bits. A float, and a ``literal`` (which emmy writes as a float),
    truncates and saturates at the type's range, as cvt.rzi does (C leaves an
    out-of-range float undefined)."""
    out = b.convert(v, value_type(dtype))
    if dtype not in NARROW_RANGE or not out.is_int:
        return out
    if literal or v.is_float:
        lo, hi = NARROW_RANGE[dtype]
        return b.imax(b.imin(out, b.int_const(hi, out.type)), b.int_const(lo, out.type))
    return _narrow(b, out, dtype)


def _narrow(b: Builder, v: Value, dtype: str) -> Value:
    """A u8, u16, i8 or i16 result computed in 32 bits, wrapped to its width."""
    width = {"u8": 8, "u16": 16, "i8": 8, "i16": 16}.get(dtype)
    if width is None or v.type not in ("s32", "u32"):
        return v
    if v.const is not None:
        w = int(v.const) & ((1 << width) - 1)
        return b.int_const(w - (1 << width) if dtype[0] == "i" and w >> (width - 1) else w, v.type)
    if dtype[0] == "u":
        return b.pure("and.b32", v.type, (v, b.int_const((1 << width) - 1, v.type)), 0, (1 << width) - 1)
    return b.pure("bfe.s32", v.type, (v, Imm(0, "u32"), Imm(width, "u32")), -(1 << (width - 1)), (1 << (width - 1)) - 1)


def _int_op(opcode: str, fold, commutative: bool = False):
    """One instruction ``opcode`` (``{t}``: the type, ``{w}``: its width), or
    the folded constant when both operands are constants."""
    def fn(b: Builder, t: str, x: Value, y: Value) -> Value:
        if x.const is not None and y.const is not None:
            r = fold(int(x.const), int(y.const))
            if r is not None:
                return b.int_const(_wrapped(t, r), t)
        return b.pure(opcode.format(t=t, w=_bits(t)), t, (x, y), commutative=commutative)
    return fn


def _division(remainder: bool):
    """numpy's ``floor_divide`` / ``remainder`` (Python's ``//`` and ``%``: the
    quotient rounds down, the remainder takes the divisor's sign). The
    hardware's ``div``/``rem`` truncate; for signed operands of different
    signs the result moves by one step. (emmy's CUDA prints C's ``/`` and
    ``%``, which differ from numpy's answer there.)"""
    def fold(x: int, y: int):
        if y == 0:
            return None  # left to the hardware, like any division by zero
        return x % y if remainder else x // y

    def fn(b: Builder, t: str, x: Value, y: Value) -> Value:
        if x.const is not None and y.const is not None and fold(int(x.const), int(y.const)) is not None:
            return b.int_const(_wrapped(t, fold(int(x.const), int(y.const))), t)
        r = b.pure(f"rem.{t}", t, (x, y))
        q = r if remainder else b.pure(f"div.{t}", t, (x, y))
        if t[0] == "u" or (x.nonneg and y.lo is not None and y.lo > 0):
            return q  # no operand below zero: truncation is the floor
        zero = b.int_const(0, t)
        sign_differs = b.icmp("<", b.pure(f"xor.b{_bits(t)}", t, (r, y), commutative=True), zero)
        step = b.pand(b.icmp("!=", r, zero), sign_differs)
        moved = b.pure(f"add.{t}", t, (r, y)) if remainder else b.pure(f"sub.{t}", t, (q, b.int_const(1, t)))
        return b.select(step, moved, q)
    return fn


def _shift(left: bool):
    def fn(b: Builder, t: str, x: Value, y: Value) -> Value:
        if x.const is not None and y.const is not None and 0 <= y.const < _bits(t):
            return b.int_const(_wrapped(t, int(x.const) << int(y.const) if left else int(x.const) >> int(y.const)), t)
        # shr.s* shifts in copies of the sign bit (numpy's >> on signed values), shr.u* zeros
        return b.pure(f"shl.b{_bits(t)}" if left else f"shr.{t}", t, (x, y))
    return fn


def _int_unary(fn_signed, fn_unsigned=None):
    def fn(b: Builder, t: str, x: Value) -> Value:
        return (fn_unsigned or fn_signed)(b, t, x) if t[0] == "u" else fn_signed(b, t, x)
    return fn


def _iabs(b: Builder, t: str, x: Value) -> Value:
    if x.const is not None:
        return b.int_const(_wrapped(t, abs(int(x.const))), t)
    return b.pure(f"abs.{t}", t, (x,))


def _ineg(b: Builder, t: str, x: Value) -> Value:
    if x.const is not None:
        return b.int_const(_wrapped(t, -int(x.const)), t)
    return b.pure(f"sub.{t}", t, (b.int_const(0, t), x))


_INT = {
    "add": _int_op("add.{t}", lambda x, y: x + y, commutative=True),
    "subtract": _int_op("sub.{t}", lambda x, y: x - y),
    "multiply": _int_op("mul.lo.{t}", lambda x, y: x * y, commutative=True),
    "floor_divide": _division(remainder=False),
    "remainder": _division(remainder=True),
    "mod": _division(remainder=True),
    "left_shift": _shift(left=True),
    "right_shift": _shift(left=False),
    "bitwise_and": _int_op("and.b{w}", lambda x, y: x & y, commutative=True),
    "bitwise_or": _int_op("or.b{w}", lambda x, y: x | y, commutative=True),
    "bitwise_xor": _int_op("xor.b{w}", lambda x, y: x ^ y, commutative=True),
    "bitwise_count": lambda b, t, x: b.pure(f"popc.b{_bits(t)}", "s32", (x,), 0, _bits(t)),
    "maximum": _int_op("max.{t}", max, commutative=True),
    "minimum": _int_op("min.{t}", min, commutative=True),
    "abs": _int_unary(_iabs, lambda b, t, x: x),
    "negative": _ineg,
    "square": lambda b, t, x: _INT["multiply"](b, t, x, x),
    "copy": lambda b, t, x: x,
    "pad": lambda b, t, x: x,
    "where": lambda b, t, c, x, y: b.select(b.truthy(c), x, y),
}
