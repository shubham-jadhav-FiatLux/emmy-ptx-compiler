"""An independent interpreter of emmy's Kernel IR, the oracle for the compiler's tests.

It shares no code with emmy_ptx. It runs the IR statement by statement with
numpy, all threads of a block in lockstep (a value is an array over the
block's threads), every value in the type the IR gives it, and takes the
IR's own rules from emmy: result types from ``dtype_promote``, reduction
identities from ``ElementwiseImpl``. Arithmetic is numpy's, which the
problems are checked against: float32 in float32 (IEEE, no fused
multiply-add), f64 in double, integers in fixed width (wrapping, floor
division, logical shifts of unsigned values). Transcendental functions are
computed in double and rounded once. Float ``maximum``, ``minimum`` and
``relu`` are CUDA's ``fmaxf`` and ``fminf``, which emmy's renderer emits for
them: a NaN gives the other operand, and -0 is below +0, as the GPU computes
them (numpy's own propagate a NaN and, of two zeros, return the second
operand). Cross-thread statements follow their
definitions in emmy's IR: a WarpShuffle is a butterfly over aligned groups
of lanes, a TreeHalve folds its shared slab level by level, leaves it so and
broadcasts the root, and a barrier is implicit in lockstep execution.

The subset: Tiles without rasterization or auxiliary bands, Loop,
StridedLoop, Cond, Let, Init, Assign, Accum, Select, Load and Write (global
and shared, atomic adds), Smem, Sync, WarpShuffle and TreeHalve.
"""

from __future__ import annotations

import math

import numpy as np
from emmy.compiler.ir.elementwise import ElementwiseImpl, reduce_canon
from emmy.compiler.ir.stmt.base import dtype_promote

NP = {"f32": np.float32, "f64": np.float64, "f16": np.float16, "i16": np.int16, "i32": np.int32,
      "i64": np.int64, "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64, "bool": np.float32}
C_TYPES = {"float": "f32", "double": "f64", "int": "i32", "unsigned": "u32", "unsigned int": "u32", "__half": "f16"}
WARP = 32


class Unsupported(Exception):
    """An IR form outside the interpreter's subset."""


def run(body, buffers: dict[str, np.ndarray], grid: int, block: int) -> dict[str, np.ndarray]:
    """Run ``body`` on ``grid`` blocks of ``block`` threads; ``buffers`` maps
    global names to arrays (outputs included), which the run updates."""
    for b in range(grid):
        _Block(buffers, b, block).run(body)
    return buffers


class _Block:
    def __init__(self, globals_: dict, ctaid: int, block: int):
        self.globals = globals_
        self.shared: dict[str, np.ndarray] = {}  # flat arrays
        self.shared_shape: dict[str, tuple] = {}
        self.ctaid, self.block = ctaid, block
        self.tid = np.arange(block, dtype=np.int64)
        # the thread's lane and warp, which emmy's kernels name without binding them
        self.env: dict[str, np.ndarray] = {"lane": (self.tid & 31).astype(np.int32),
                                           "warp": (self.tid >> 5).astype(np.int32)}

    # ---------------------------------------------------------------- values

    def set(self, name: str, value, mask: np.ndarray) -> None:
        value = np.broadcast_to(np.asarray(value), (self.block,))
        old = self.env.get(name)
        if old is None or old.dtype != value.dtype:
            old = np.zeros(self.block, value.dtype) if old is None else old.astype(value.dtype)
        self.env[name] = np.where(mask, value, old)

    def expr(self, e) -> np.ndarray:
        """An IR expression over every thread (C semantics for integers: / and % truncate)."""
        k = type(e).__name__
        if k == "Var":
            return self.env[e.name]
        if k == "Literal":
            v = e.value
            dtype = np.int64 if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else np.float64
            return np.full(self.block, v, dtype)
        if k == "BinaryExpr":
            a, b = self.expr(e.left), self.expr(e.right)
            op = e.op
            if op in ("/", "%") and a.dtype.kind in "iu" and b.dtype.kind in "iu":
                q = np.trunc(a / np.where(b == 0, 1, b)).astype(np.int64)
                return q if op == "/" else a - q * b
            fns = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "<": np.less,
                   "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal,
                   "!=": np.not_equal, "&&": np.logical_and, "||": np.logical_or, "&": np.bitwise_and,
                   "|": np.bitwise_or, "^": np.bitwise_xor, "<<": np.left_shift, ">>": np.right_shift}
            if op not in fns:
                raise Unsupported(f"operator {op}")
            return fns[op](a, b)
        if k == "TernaryExpr":
            return np.where(self.truthy(self.expr(e.cond)), self.expr(e.if_true), self.expr(e.if_false))
        if k == "CastExpr":
            return self.expr(e.expr).astype(NP[_dtype_name(e.dtype)])
        raise Unsupported(f"expression {k}")

    @staticmethod
    def truthy(v: np.ndarray) -> np.ndarray:
        return v.astype(bool) if v.dtype != bool else v

    def flat(self, shape, index) -> np.ndarray:
        idx = np.zeros(self.block, np.int64)
        for e, n in zip(index, shape):
            idx = idx * int(n) + self.expr(e).astype(np.int64)
        return idx

    def memory(self, name: str) -> np.ndarray:
        return self.shared[name] if name in self.shared else self.globals[name]

    # ------------------------------------------------------------ statements

    def run(self, stmts, mask: np.ndarray | None = None) -> None:
        mask = np.ones(self.block, bool) if mask is None else mask
        for s in stmts:
            getattr(self, "s_" + type(s).__name__, self.unsupported)(s, mask)

    def unsupported(self, s, mask) -> None:
        raise Unsupported(f"statement {type(s).__name__}")

    def s_Tile(self, s, mask) -> None:
        if s.raster_axes or s.aux_threads or (s.block_threads not in (None, self.block)):
            raise Unsupported("rasterized, banded or partial-block Tile")
        extents = [int(a.extent.as_static()) for a in s.axes]
        gid = self.ctaid * self.block + self.tid
        mask = mask & (gid < math.prod(extents))
        stride = 1
        for axis, n in reversed(list(zip(s.axes, extents))):
            self.set(axis.name, (gid // stride) % n, mask)
            stride *= n
        self.run(s.body, mask)

    def seed(self, loop, mask) -> None:
        if not loop.seed:
            return
        for st in loop.body:
            if type(st).__name__ == "Accum":
                identity = ElementwiseImpl(reduce_canon(st.op.name)).identity
                dtype = NP[_dtype_name(st.dtype) or "f32"]
                self.set(st.name, np.asarray(identity).astype(dtype), mask)

    def s_Loop(self, s, mask) -> None:
        self.seed(s, mask)
        for i in range(int(s.axis.extent.as_static())):
            self.set(s.axis.name, np.int64(i), mask)
            self.run(s.body, mask)

    def s_StridedLoop(self, s, mask) -> None:
        self.seed(s, mask)
        start, step = self.expr(s.start).astype(np.int64), self.expr(s.step).astype(np.int64)
        if s.end is not None:
            end = self.expr(s.end).astype(np.int64)
            self.set(f"{s.axis.name}_end", end, mask)
        else:
            end = np.full(self.block, int(s.axis.extent.as_static()), np.int64)
        k = 0
        while True:
            v = start + k * step
            active = mask & (v < end)
            if not active.any():
                break
            self.set(s.axis.name, v, active)
            self.run(s.body, active)
            k += 1

    def s_Cond(self, s, mask) -> None:
        c = self.truthy(self.expr(s.cond))
        self.run(s.body, mask & c)
        self.run(s.else_body, mask & ~c)

    def s_Let(self, s, mask) -> None:
        v = self.expr(s.value)
        if s.dtype is not None:
            v = v.astype(NP[_dtype_name(s.dtype)])
        elif v.dtype == np.int64 and np.all((v >= -2**31) & (v < 2**31)):
            v = v.astype(np.int32)  # an int, as C declares it
        self.set(s.name, v, mask)

    def s_Init(self, s, mask) -> None:
        self.set(s.name, np.asarray(s.identity).astype(NP[_dtype_name(s.dtype)]), mask)

    def s_Assign(self, s, mask) -> None:
        args = [self.env[a] for a in s.args]
        out = _dtype_name(s.dtype) or dtype_promote(s.op.name, [_numpy_name(a.dtype) for a in args])
        self.set(s.name, apply(s.op.name, args, out), mask)

    def s_Accum(self, s, mask) -> None:
        acc = self.env[s.name]
        base = self.env[s.base] if s.base is not None else acc
        op = reduce_canon(s.op.name)
        self.set(s.name, apply(op if op in _BINARY else "add", [base, self.env[s.value]], _numpy_name(acc.dtype)), mask)

    def s_Select(self, s, mask) -> None:
        branches = list(s.branches)
        out = self.env[branches[-1].value].astype(np.float32)
        for br in reversed(branches[:-1]):
            out = np.where(self.truthy(self.expr(br.select)), self.env[br.value].astype(np.float32), out)
        self.set(s.name, out, mask)

    def s_Smem(self, s, mask) -> None:
        if s.name not in self.shared:
            ctype = getattr(s, "dtype", None) or "float"
            self.shared[s.name] = np.zeros(math.prod(int(n) for n in s.extents), NP[C_TYPES.get(ctype, ctype)])
            self.shared_shape[s.name] = tuple(int(n) for n in s.extents)

    def shape_of(self, name: str):
        return self.shared_shape[name] if name in self.shared else self.globals[name].shape

    def s_Load(self, s, mask) -> None:
        mem = self.memory(s.input).reshape(-1)
        idx = self.flat(self.shape_of(s.input), s.index)
        for k, name in enumerate(s.names):
            safe = np.where(mask, idx + k, 0)
            v = mem[safe]
            if s.dtype is not None:
                v = v.astype(NP[_dtype_name(s.dtype)])
            self.set(name, v, mask)

    def s_Write(self, s, mask) -> None:
        mem = self.memory(s.output).reshape(-1)
        idx = self.flat(self.shape_of(s.output), s.index)
        for k, name in enumerate(s.values):
            v = self.env[name].astype(mem.dtype)
            if s.atomic:
                np.add.at(mem, idx[mask] + k, v[mask])
            else:
                mem[idx[mask] + k] = v[mask]

    def s_Sync(self, s, mask) -> None:
        pass  # all threads of the block run each statement together

    def combine(self, program, mask) -> None:
        for a in program:
            self.s_Assign(a, mask)

    def s_WarpShuffle(self, s, mask) -> None:
        width = int(s.length) // 2
        while width:
            partner = self.tid ^ width
            for st, sb in zip(s.state, s.state_b):
                self.set(sb, self.env[st][partner], mask)
            self.combine(s.combine_states, mask)
            width //= 2

    def s_TreeHalve(self, s, mask) -> None:
        bufs = [self.memory(b).reshape(-1) for b in s.bufs]
        t = self.env[s.tid_var].astype(np.int64)
        if s.inner is not None:
            iv, scale = self.env[s.inner[0]].astype(np.int64), int(s.inner[1])
            slot = lambda k: k * scale + iv  # noqa: E731
        else:
            slot = lambda k: k  # noqa: E731
        half = int(s.length) // 2
        while half:
            active = mask & (t < half)
            mine, theirs = np.where(active, slot(t), 0), np.where(active, slot(t + half), 0)
            for st, sb, buf in zip(s.state, s.state_b, bufs):
                self.set(st, buf[mine], active)
                self.set(sb, buf[theirs], active)
            self.combine(s.combine_states, active)
            for st, buf in zip(s.state, bufs):
                buf[mine[active]] = self.env[st][active].astype(buf.dtype)
            half //= 2
        root = slot(np.zeros(self.block, np.int64))
        for st, buf in zip(s.state, bufs):
            self.set(st, buf[np.where(mask, root, 0)], mask)


# ---------------------------------------------------------------- elementwise operations


def _dtype_name(d) -> str | None:
    return None if d is None else getattr(d, "name", d)


def _numpy_name(dtype: np.dtype) -> str:
    if np.dtype(dtype) == np.bool_:
        return "bool"
    return {np.dtype(v): k for k, v in NP.items() if k != "bool"}[np.dtype(dtype)]


def _float_fn(fn):
    """A transcendental function: computed in double, rounded once to the result type."""
    def apply_(t, *xs):
        with np.errstate(all="ignore"):
            return fn(*(x.astype(np.float64) for x in xs)).astype(t)
    return apply_


def _erf(x):
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def _fmax(a, b):
    """CUDA's fmax for floats (a NaN gives the other operand, -0 is below +0); numpy's for integers."""
    if not np.issubdtype(np.result_type(a, b), np.floating):
        return np.maximum(a, b)
    r = np.fmax(a, b)
    return np.where((a == 0) & (b == 0) & ~(np.signbit(a) & np.signbit(b)), np.abs(r), r)


def _fmin(a, b):
    """CUDA's fmin for floats, as _fmax."""
    if not np.issubdtype(np.result_type(a, b), np.floating):
        return np.minimum(a, b)
    r = np.fmin(a, b)
    return np.where((a == 0) & (b == 0) & (np.signbit(a) | np.signbit(b)), -np.abs(r), r)


_UNARY = {
    "negative": lambda t, x: np.negative(x.astype(t)),
    "abs": lambda t, x: np.abs(x.astype(t)),
    "copy": lambda t, x: x.astype(t), "pad": lambda t, x: x.astype(t),
    "square": lambda t, x: np.multiply(x.astype(t), x.astype(t)),
    "reciprocal": lambda t, x: np.divide(t(1), x.astype(t)),
    "relu": lambda t, x: _fmax(x.astype(t), t(0)),
    "sqrt": lambda t, x: np.sqrt(x.astype(t)),
    "exp": _float_fn(np.exp), "log": _float_fn(np.log), "tanh": _float_fn(np.tanh), "sin": _float_fn(np.sin),
    "cos": _float_fn(np.cos), "erf": _float_fn(_erf), "sigmoid": _float_fn(lambda x: 1 / (1 + np.exp(-x))),
    "rsqrt": _float_fn(lambda x: 1 / np.sqrt(x)),
    "bitwise_count": lambda t, x: np.bitwise_count(x).astype(t),
}
_BINARY = {
    "add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide,
    "maximum": _fmax, "minimum": _fmin, "floor_divide": np.floor_divide, "remainder": np.remainder,
    "mod": np.remainder, "left_shift": np.left_shift, "right_shift": np.right_shift,
    "bitwise_and": np.bitwise_and, "bitwise_or": np.bitwise_or, "bitwise_xor": np.bitwise_xor,
}
_COMPARE = {"equal": np.equal, "not_equal": np.not_equal, "greater": np.greater, "less": np.less,
            "greater_equal": np.greater_equal, "less_equal": np.less_equal}


def apply(op: str, args: list[np.ndarray], out: str) -> np.ndarray:
    """``op(*args)`` with result dtype ``out``, as numpy computes it."""
    t = NP[out]
    with np.errstate(all="ignore"):
        if op in _COMPARE:  # compared in the operands' common type; the result is the 0 or 1 of the test
            return _COMPARE[op](*args).astype(t)
        if op == "where":
            return np.where(args[0].astype(bool), args[1].astype(t), args[2].astype(t))
        if op in _UNARY:
            return np.asarray(_UNARY[op](t, args[0])).astype(t)
        if op in _BINARY:
            if op in ("left_shift", "right_shift"):
                return _BINARY[op](args[0].astype(t), args[1].astype(t)).astype(t)
            return _BINARY[op](args[0].astype(t), args[1].astype(t)).astype(t)
        if op == "pow":
            return _float_fn(np.power)(t, *args)
    raise Unsupported(f"elementwise op {op}")
