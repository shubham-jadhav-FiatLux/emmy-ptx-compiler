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

import numpy as np

from . import mathlib
from .builder import NUMPY_FLOAT, Builder, Value, float_type

INTEGER_DTYPES = frozenset({"i16", "i32", "i64", "u8", "u16", "u32", "u64"})
INTEGER_OPS = frozenset({
    "add", "subtract", "multiply", "floor_divide", "remainder", "mod",
    "left_shift", "right_shift", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_count",
})
_WIDTH = {"u8": 8, "i16": 16, "u16": 16, "i32": 32, "u32": 32, "i64": 64, "u64": 64}


def value_type(dtype: str) -> str:
    """The register type a value of emmy dtype ``dtype`` is computed in."""
    if dtype in ("f32", "bf16"):
        return "f32"
    if dtype in ("f16", "f64"):
        return dtype
    if dtype in ("i64", "u64"):
        return "s64"
    if dtype in INTEGER_DTYPES or dtype == "bool":
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
    out_type = value_type(dtype)
    if out_type in ("s32", "s64") and op in _INT:
        args = [b.convert(a, out_type) for a in args]
        return _INT[op](b, *args)
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


def _as_float(b: Builder, p: Value) -> Value:
    """A C comparison used as a value: 1.0 or 0.0."""
    return b.select(p, b.f32_const(1.0), b.f32_const(0.0))


def _cmp(op: str):
    return lambda b, x, y: _as_float(b, b.fcmp(op, x, y))


def _unary(opcode: str, fold):
    """One instruction, ``opcode.f32`` or ``opcode.f64`` by the operand's type;
    ``fold`` computes it on a constant (a numpy scalar of that type)."""
    def fn(b: Builder, x: Value) -> Value:
        if x.const is not None and fold is not None:
            with np.errstate(all="ignore"):
                return b.float_const(float(fold(NUMPY_FLOAT[x.type](x.const))), x.type)
        return b.fop(f"{opcode}.{x.type}", b.in_reg(x), type_=x.type)
    return fn


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
    "maximum": _minmax("max", max),
    "minimum": _minmax("min", min),
    "negative": _unary("neg", lambda v: -v),
    "abs": _unary("abs", abs),
    "copy": lambda b, x: x,
    "pad": lambda b, x: x,
    "square": lambda b, x: b.fbin("*", x, x),
    "reciprocal": _unary("rcp.rn", lambda v: 1 / v),
    "relu": lambda b, x: _minmax("max", max)(b, x, b.f32_const(0.0)),
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
    "equal": _cmp("=="),
    "not_equal": _cmp("!="),
    "greater": _cmp(">"),
    "less": _cmp("<"),
    "greater_equal": _cmp(">="),
    "less_equal": _cmp("<="),
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

_INT = {
    "add": lambda b, x, y: b.add(x, y),
    "subtract": lambda b, x, y: b.sub(x, y),
    "multiply": lambda b, x, y: b.mul(x, y),
    "floor_divide": lambda b, x, y: b.div(x, y),  # emmy prints `//` as C `/`
    "remainder": lambda b, x, y: b.rem(x, y),
    "mod": lambda b, x, y: b.rem(x, y),
    "left_shift": lambda b, x, y: b.shl(x, y),
    "right_shift": lambda b, x, y: b.shr(x, y),
    "bitwise_and": lambda b, x, y: b.bitop("&", x, y),
    "bitwise_or": lambda b, x, y: b.bitop("|", x, y),
    "bitwise_xor": lambda b, x, y: b.bitop("^", x, y),
    "bitwise_count": lambda b, x: b.pure(f"popc.b{x.type[1:]}", "s32", (x,), 0, int(x.type[1:])),
}
