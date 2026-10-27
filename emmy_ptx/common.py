"""What the lowering's modules share: emmy's dtype tables, the kernel's
buffers as the lowering sees them, affine index forms, C scopes, and a few
constants and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from emmy.compiler.ir.expr import Literal

from .builder import Value

# emmy dtype names (graph tensors) -> PTX memory types
MEM_TYPE = {
    "f32": "f32", "f16": "f16", "f64": "f64", "bf16": "bf16",
    "i32": "s32", "i64": "s64", "i16": "s16", "i8": "s8",
    "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64", "bool": "u8",
    # fp8 values travel as their byte codes (see elementwise.py: only the casts look inside)
    "f8e4m3": "u8", "f8e5m2": "u8",
}
# C type names used by Smem declarations -> emmy dtype names
C_TYPE = {
    "float": "f32", "__half": "f16", "half": "f16", "__nv_bfloat16": "bf16", "double": "f64",
    "int": "i32", "unsigned": "u32", "unsigned int": "u32", "long long": "i64",
    "unsigned long long": "u64", "unsigned char": "u8", "char": "i8", "short": "i16",
    "__nv_fp8_e4m3": "f8e4m3", "__nv_fp8_e5m2": "f8e5m2",
}
# memory type -> the value type it is computed in after loading
VALUE_TYPE = {
    "f32": "f32", "f16": "f16", "f64": "f64",
    "s32": "s32", "u32": "u32", "s64": "s64", "u64": "u64",
    "s16": "s32", "u16": "u32", "s8": "s32", "u8": "u32",
}
# memory type -> the red.add that adds a value to it (an integer add is the same for either sign)
ATOMIC_ADD = {"f32": "f32", "f64": "f64", "f16": "noftz.f16", "s32": "s32", "u32": "u32", "s64": "u64", "u64": "u64"}
# Static shared memory a kernel may declare; past it the arrays form one dynamic pool.
STATIC_SMEM_LIMIT = 48 * 1024
# Fully unroll an ``unroll`` loop only while the expanded body stays below this
# many statements: past it the PTX grows faster than ptxas can schedule it.
UNROLL_BUDGET = 4096
# Loops this short are unrolled completely even when the IR does not ask (loops
# over a shared-memory tile up to SMEM_UNROLL_TRIPS).
AUTO_UNROLL_TRIPS, SMEM_UNROLL_TRIPS = 64, 128
# Streaming multiprocessors of the evaluation GPU (RTX 5090). Only a scheduling
# hint depends on it; the generated code is correct on any GPU.
TARGET_SMS = 170
# identities of the reduction combines (emmy's ElementwiseImpl table)
IDENTITY = {"add": 0.0, "multiply": 1.0, "maximum": -1e30, "minimum": 1e30}
REDUCE_CANON = {
    "add": "add", "sum": "add", "multiply": "multiply", "prod": "multiply",
    "maximum": "maximum", "amax": "maximum", "fmax": "maximum", "minimum": "minimum", "fmin": "minimum",
}


@dataclass(frozen=True)
class GlobalBuffer:
    name: str
    pointer: Value  # generic->global converted base address
    type: str  # PTX memory type of one element
    shape: tuple[int, ...]
    readonly: bool


@dataclass(frozen=True)
class SharedArray:
    name: str
    symbol: str  # the array's PTX symbol (or the dynamic pool's)
    offset: int  # byte offset of the array inside ``symbol``
    type: str
    shape: tuple[int, ...]


@dataclass
class Affine:
    """``sum(coeff * atom) + const`` over integer values."""

    terms: dict = field(default_factory=dict)  # key -> [Value, coeff]
    const: int = 0

    def scaled(self, k: int) -> Affine:
        return Affine({key: [v, c * k] for key, (v, c) in self.terms.items()}, self.const * k)

    def plus(self, other: Affine, sign: int = 1) -> Affine:
        terms = {key: [v, c] for key, (v, c) in self.terms.items()}
        for key, (v, c) in other.terms.items():
            terms.setdefault(key, [v, 0])[1] += sign * c
        return Affine({k: t for k, t in terms.items() if t[1] != 0}, self.const + sign * other.const)


class Scope:
    """A C block scope."""

    def __init__(self, parent: Scope | None = None):
        self.parent = parent
        self.names: dict[str, object] = {}

    def lookup(self, name: str):
        scope = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None

    def rebind(self, name: str, value) -> None:
        """Point ``name`` at ``value`` in the scope that declares it."""
        scope = self
        while scope is not None:
            if name in scope.names:
                scope.names[name] = value
                return
            scope = scope.parent


class LoweringError(Exception):
    pass


def body_size(stmts) -> int:
    """Statements in a body after the full unrolling we would apply inside it."""
    total = 0
    for s in stmts:
        kind = type(s).__name__
        if kind == "Loop":
            inner = body_size(s.body)
            extent = int(s.axis.extent.as_static())
            total += inner * extent if s.unroll and inner * extent <= UNROLL_BUDGET else inner + 4
        elif kind in ("StridedLoop", "Cond"):
            total += sum(body_size(b) for b in s.nested()) + 2
        elif kind == "Tile":
            total += body_size(s.body)
        elif kind in ("WarpShuffle", "TreeHalve"):
            total += 8 * max(1, int(s.length).bit_length()) * max(1, len(s.combine_states))
        else:
            total += 1
    return max(total, 1)


def int_lit(v: int):
    return Literal(v, "int")


def lowbit(n: int) -> int:
    """The largest power of two dividing ``n``."""
    return n & -n

