"""Kernel IR -> PTX lowering.

The :class:`Lowerer` walks the statement tree once, in order, and emits PTX
through the :class:`~emmy_ptx.builder.Builder`. Every statement kind has one
method, ``stmt_<Kind>``; expressions are lowered by ``expr``.

Names follow C block scoping, exactly like the CUDA C++ emmy would print for
the same IR: a Tile, loop or branch body opens a scope, a declaration shadows
outer names, and a name declared before a loop and reassigned inside it (an
accumulator, a combined state) is one mutable register.

Addresses are computed in *affine form*: a flattened index is split into a
register part and a constant part, the constant goes into the instruction's
``[base+offset]`` immediate, and the register part is shared by every access
that differs only by a constant (``x[4*a0+0] .. x[4*a0+3]`` cost one address).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field, replace

from . import elementwise
from .builder import INT_RANGE, WARP, Builder, Value, Variable, eliminate_dead_code
from .congruence import congruent_accumulators, merge_congruent
from .frontend import DEFAULT_BLOCK, KernelSource
from .ptx import SIZEOF, Imm, Kernel, Mem, Reg, Sym, Vec
from .vectorize import vectorize_loads

# emmy dtype names (graph tensors) -> PTX memory types
MEM_TYPE = {
    "f32": "f32", "f16": "f16", "f64": "f64", "bf16": "bf16",
    "i32": "s32", "i64": "s64", "i16": "s16", "i8": "s8",
    "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64", "bool": "u8",
}

# C type names used by Smem declarations -> emmy dtype names
C_TYPE = {
    "float": "f32", "__half": "f16", "half": "f16", "__nv_bfloat16": "bf16", "double": "f64",
    "int": "i32", "unsigned": "u32", "unsigned int": "u32", "long long": "i64",
    "unsigned long long": "u64", "unsigned char": "u8", "char": "i8", "short": "i16",
}

# memory type -> the value type it is computed in after loading
VALUE_TYPE = {
    "f32": "f32", "f16": "f16", "f64": "f64",
    "s32": "s32", "u32": "s32", "s64": "s64", "u64": "s64",
    "s16": "s32", "u16": "s32", "s8": "s32", "u8": "s32",
}

STATIC_SMEM_LIMIT = 48 * 1024
# Alignment (bytes) assumed for every buffer base: cudaMalloc returns 256-byte
# aligned memory and emmy's own float4 accesses rely on 16; shared arrays are
# declared .align 16 or more.
MAX_ALIGN = 16
# Fully unroll an ``unroll`` loop only while the expanded body stays below this
# many statements: past it the PTX grows faster than ptxas can schedule it.
UNROLL_BUDGET = 4096
# Loops this short are unrolled completely even when the IR does not ask.
AUTO_UNROLL_TRIPS = 64
# Iterations per trip of a rolled loop, by body size (statements).
UNROLL_TINY_BODY, UNROLL_TINY = 6, 8
UNROLL_SMALL_BODY, UNROLL_SMALL = 24, 4
UNROLL_BIG_BODY, UNROLL_BIG_CHOICES = 400, (8, 4, 2)  # tuned with tools/tune.py
# Streaming multiprocessors of the evaluation GPU (RTX 5090). Only a scheduling
# hint depends on it; the generated code is correct on any GPU.
TARGET_SMS = 170
# A reduction shared by a group of lanes is split across them only when it has
# at least this many trips (the shuffle tree costs log2(group) steps).
SPLIT_MIN_TRIPS = 16
# Associative, commutative combines a lane-split reduction may regroup.
SPLITTABLE = ("add", "multiply", "maximum", "minimum")

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


class Lowerer:
    def __init__(self, src: KernelSource):
        self.src = src
        # A grid that fits in one wave puts at most one CTA on each SM, so extra
        # registers per thread cannot cost occupancy: let ptxas spend them on
        # keeping more loads in flight (it otherwise caps registers to leave room
        # for CTAs that never come). Bigger grids keep ptxas's own balance.
        one_wave = src.launch.grid <= TARGET_SMS
        self.kernel = Kernel(src.name, tuple(_param_name(p) for p in src.params), maxntid=src.launch.block,
                             minnctapersm=1 if one_wave else None)
        self.b = Builder(self.kernel)
        self.scope = Scope()
        self.mutable = _reassigned_names(src.body)
        self.products: dict[str, tuple[Value, Value]] = {}  # SSA name -> factors of a multiply
        self._uniq = 0
        self.converged = False  # every lane of each warp is executing the current statement
        self.shared = self._layout_shared()
        self.globals = self._prologue()
        self._readonly = frozenset(n for n, g in self.globals.items() if g.readonly)

    # ------------------------------------------------------------------ driver

    def run(self) -> Kernel:
        self.body(self.src.body)
        self.b.emit("ret")
        self.kernel.flatten()
        eliminate_dead_code(self.kernel)
        vectorize_loads(self.kernel, self.b.align)
        return self.kernel

    def body(self, stmts) -> None:
        for s in _issue_loads_early(stmts, self._readonly):
            handler = getattr(self, "stmt_" + type(s).__name__, None)
            if handler is None:
                raise LoweringError(f"unsupported statement {type(s).__name__}")
            handler(s)

    @contextlib.contextmanager
    def block(self, dominated: bool = False, guarded: bool = False, divergent: bool = False):
        """A C block: a new name scope, and unless ``dominated`` (straight-line
        code such as an unrolled iteration) a new value-numbering scope.
        ``guarded`` marks a branch body (its loads must stay where they are);
        ``divergent`` a branch whose condition may differ between lanes."""
        self.scope = Scope(self.scope)
        converged = self.converged
        self.converged = converged and not divergent
        try:
            with contextlib.ExitStack() as stack:
                if not dominated:
                    stack.enter_context(self.b.cse_scope())
                if guarded:
                    stack.enter_context(self.b.guarded())
                yield
        finally:
            self.scope = self.scope.parent
            self.converged = converged

    def _pessimize(self, stmts) -> None:
        """Forget the uniformity of the variables ``stmts`` reassign: at a merge
        point or a loop head their value is not known to be uniform."""
        for name in _assigned_in(stmts):
            var = self.scope.lookup(name)
            if isinstance(var, Variable):
                var.forget()

    # ------------------------------------------------------------- set-up

    def _prologue(self) -> dict[str, GlobalBuffer]:
        launch = self.src.launch
        b = self.b
        self.tid = b.special("%tid.x", 0, launch.block - 1)
        self.ctaid = b.special("%ctaid.x", 0, launch.grid - 1)
        self.scope.names["lane"] = b.bitop("&", self.tid, b.int_const(31))
        self.scope.names["warp"] = b.uniform(b.shr(self.tid, b.int_const(5)), WARP)
        out = {}
        for name in self.src.params:
            buf = self.src.buffers.get(name)
            if buf is None:
                raise LoweringError(f"parameter {name!r} has no tensor record in the IR or meta.json")
            raw = b.pure("ld.param.u64", "s64", (Mem(Sym(_param_name(name))),))
            ptr = b.pure("cvta.to.global.u64", "s64", (raw,))
            b.note_align(ptr.op, MAX_ALIGN)
            out[name] = GlobalBuffer(name, ptr, MEM_TYPE[buf.dtype], buf.shape, name not in self.src.written)
        return out

    def _layout_shared(self) -> dict[str, SharedArray]:
        """Static ``.shared`` arrays up to 48 KiB in total, else one dynamic pool
        packed exactly like emmy's ``pack_smem`` (so ``meta.smem_bytes`` fits)."""
        decls = {}
        for s in _walk(self.src.body):
            if type(s).__name__ == "Smem":
                decls.setdefault(s.name, s)
        arrays, cursor, pool_align = {}, 0, 16
        placed = []
        for s in decls.values():
            dtype = C_TYPE.get(s.dtype)
            if dtype is None:
                raise LoweringError(f"shared array {s.name!r} has unsupported type {s.dtype!r}")
            mtype = MEM_TYPE[dtype]
            size = SIZEOF.get(mtype, 1)
            align = max(size, int(s.align or 0))
            cursor = -(-cursor // align) * align
            placed.append((s, mtype, cursor, align))
            cursor += math.prod(int(e) for e in s.extents) * size
            pool_align = max(pool_align, align)
        dynamic = cursor > STATIC_SMEM_LIMIT
        if dynamic:
            self.kernel.decls.append(f".extern .shared .align {pool_align} .b8 smem_pool[];  // {cursor} bytes")
        for s, mtype, offset, align in placed:
            shape = tuple(int(e) for e in s.extents)
            if dynamic:
                arrays[s.name] = SharedArray(s.name, "smem_pool", offset, mtype, shape)
            else:
                sym = f"smem_{s.name}"
                nbytes = math.prod(shape) * SIZEOF.get(mtype, 1)
                self.kernel.decls.append(f".shared .align {max(16, align)} .b8 {sym}[{max(nbytes, 1)}];")
                arrays[s.name] = SharedArray(s.name, sym, 0, mtype, shape)
        return arrays

    # ------------------------------------------------------------- names

    def lookup(self, name: str):
        found = self.scope.lookup(name)
        if found is None:
            raise LoweringError(f"undefined name {name!r}")
        return found

    def value(self, name: str) -> Value:
        found = self.lookup(name)
        if isinstance(found, Variable):
            return found.read()
        if isinstance(found, Value):
            return found
        raise LoweringError(f"{name!r} is not a value")

    def define(self, name: str, v: Value) -> None:
        """Declare ``name`` in the current scope (a register if it is reassigned later)."""
        if name in self.mutable:
            self.scope.names[name] = self.b.variable(v.type, v)
        else:  # C copies: a later update of an accumulator must not show through
            self.scope.names[name] = self.b.snapshot(v)

    def declare_var(self, name: str, type_: str, init: Value) -> Variable:
        var = self.b.variable(type_, init)
        self.scope.names[name] = var
        return var

    def reassign(self, name: str, v: Value) -> None:
        found = self.lookup(name)
        if not isinstance(found, Variable):
            raise LoweringError(f"cannot reassign {name!r}: it was not declared as a variable")
        self.b.assign(found, v)

    def buffer(self, name: str) -> GlobalBuffer | SharedArray:
        found = self.scope.lookup(name)
        if isinstance(found, SharedArray):
            return found
        if name in self.shared and found is None:
            return self.shared[name]
        if name in self.globals:
            return self.globals[name]
        raise LoweringError(f"unknown buffer {name!r} (not a parameter and not a shared array)")

    # ------------------------------------------------------------- expressions

    def expr(self, e) -> Value:
        kind = type(e).__name__
        b = self.b
        if kind == "Var":
            return self.value(e.name)
        if kind == "Literal":
            if isinstance(e.value, float) or e.dtype == "float":
                return b.f32_const(float(e.value))
            return b.int_const(int(e.value))
        if kind == "BinaryExpr":
            return self._binary(e.op, e.left, e.right)
        if kind == "TernaryExpr":
            p = self.pred(e.cond)
            t, f = self.expr(e.if_true), self.expr(e.if_false)
            t, f = _promote_pair(b, t, f)
            return b.select(p, t, f)
        if kind == "CastExpr":
            target = {"int": "s32", "long long": "s64", "float": "f32", "double": "f64", "__half": "f16"}.get(e.dtype)
            if target is None:
                raise LoweringError(f"unsupported cast to {e.dtype!r}")
            return b.convert(self.expr(e.expr), target)
        if kind == "FuncCallExpr":
            return self._call(e.name, [self.expr(a) for a in e.args])
        if kind == "Builtin":
            return self._builtin(e.name)
        if kind == "FlatIndex":
            return self.materialize(self.flat(self.buffer(e.buffer), e.index))
        raise LoweringError(f"unsupported expression {kind}")

    def pred(self, e) -> Value:
        return self.b.truthy(self.expr(e))

    def _binary(self, op: str, left, right) -> Value:
        b = self.b
        if op in ("&&", "||"):
            x, y = self.pred(left), self.pred(right)
            return b.pand(x, y) if op == "&&" else b.por(x, y)
        x, y = self.expr(left), self.expr(right)
        x = b.convert(x, "s32") if x.type == "pred" else x
        y = b.convert(y, "s32") if y.type == "pred" else y
        if op in ("<", "<=", ">", ">=", "==", "!="):
            if x.is_float or y.is_float:
                return b.fcmp(op, x, y)
            return b.icmp(op, x, y)
        if x.is_float or y.is_float:
            if op not in ("+", "-", "*", "/"):
                raise LoweringError(f"operator {op!r} on floating-point operands")
            return b.fbin(op, x, y)
        if op == "+":
            return b.add(x, y)
        if op == "-":
            return b.sub(x, y)
        if op == "*":
            return b.mul(x, y)
        if op in ("/", "//"):
            return b.div(x, y)
        if op == "%":
            return b.rem(x, y)
        if op in ("&", "|", "^"):
            return b.bitop(op, x, y)
        if op == "<<":
            return b.shl(x, y)
        if op == ">>":
            return b.shr(x, y)
        raise LoweringError(f"unsupported operator {op!r}")

    def _call(self, name: str, args: list[Value]) -> Value:
        if name in ("min", "max") and all(a.is_int for a in args):
            return (self.b.imin if name == "min" else self.b.imax)(*args)
        op = {"fmax": "maximum", "fmin": "minimum", "fabs": "abs", "min": "minimum", "max": "maximum"}.get(name, name)
        dtype = "f32" if any(a.is_float for a in args) else elementwise.promote(op, [_dtype_of(a) for a in args])
        return elementwise.apply(self.b, op, args, dtype)

    def _builtin(self, name: str) -> Value:
        launch = self.src.launch
        table = {
            "thread_idx.x": lambda: self.tid,
            "block_idx.x": lambda: self.ctaid,
            "block_dim.x": lambda: self.b.int_const(launch.block),
            "grid_dim.x": lambda: self.b.int_const(launch.grid),
            "warp_size": lambda: self.b.int_const(32),
        }
        if name in ("thread_idx.y", "thread_idx.z", "block_idx.y", "block_idx.z"):
            return self.b.int_const(0)
        if name in ("block_dim.y", "block_dim.z", "grid_dim.y", "grid_dim.z"):
            return self.b.int_const(1)
        if name not in table:
            raise LoweringError(f"unsupported builtin {name!r}")
        return table[name]()

    # ------------------------------------------------------------- affine indices

    def affine(self, e) -> Affine:
        """Split an integer expression into ``sum(coeff * atom) + const``."""
        kind = type(e).__name__
        if kind == "Literal" and not (isinstance(e.value, float) or e.dtype == "float"):
            return Affine(const=int(e.value))
        if kind == "BinaryExpr" and e.op in ("+", "-"):
            return self.affine(e.left).plus(self.affine(e.right), 1 if e.op == "+" else -1)
        if kind == "BinaryExpr" and e.op == "*":
            left, right = self.affine(e.left), self.affine(e.right)
            if not right.terms:
                return left.scaled(right.const)
            if not left.terms:
                return right.scaled(left.const)
        v = self.expr(e)
        if not v.is_int:
            raise LoweringError(f"non-integer index expression {e!r}")
        return self._as_affine(v)

    def materialize(self, aff: Affine, with_const: bool = True) -> Value:
        b = self.b
        acc = b.int_const(0)
        for v, c in aff.terms.values():
            acc = b.add(acc, b.mul(v, b.int_const(c)))
        return b.add(acc, b.int_const(aff.const)) if with_const else acc

    def flat(self, buf, index) -> Affine:
        """Row-major element offset of ``index`` in ``buf`` (emmy's render_index)."""
        index = tuple(index)
        if not index:
            return Affine()
        if len(index) == 1:
            return self.affine(index[0])
        shape = buf.shape
        if len(shape) != len(index):
            out = Affine()
            for e in index:
                out = out.plus(self.affine(e))
            return out
        out = Affine()
        for d, e in enumerate(index):
            out = out.plus(self.affine(e).scaled(math.prod(shape[d + 1 :])))
        return out

    def address(self, buf, index=None, flat: Affine | None = None, swizzle: str = "NONE") -> Mem:
        """Memory operand for element ``index`` of ``buf``."""
        b = self.b
        aff = flat if flat is not None else self.flat(buf, index)
        size = SIZEOF.get(buf.type, 1)
        if swizzle != "NONE" and _swizzle_mask(swizzle):
            elem = _swizzled(b, self.materialize(aff), swizzle)
            aff = Affine({elem.key() or ("swz", id(elem)): [elem, 1]}) if elem.const is None else Affine(const=elem.const)
        var = self.materialize(aff, with_const=False)
        # bytes the register part is a multiple of: every term contributes coeff*size
        # times an arbitrary integer; bases (parameters, shared arrays) are 16-aligned
        align = min([MAX_ALIGN] + [_lowbit(abs(c) * size) for _, c in aff.terms.values()])
        if isinstance(buf, SharedArray):
            disp = buf.offset + aff.const * size
            if var.const == 0:
                return Mem(Sym(buf.symbol), disp)
            base = b.pure("mov.u32", "s32", (Sym(buf.symbol),), 0, 1 << 24)
            reg = b.add(base, b.mul(var, b.int_const(size))).op
            b.note_align(reg, align)
            return Mem(reg, disp)
        disp = aff.const * size
        if var.const == 0:
            return Mem(buf.pointer.op, disp)
        if var.type == "s64":
            off = b.mul(var, b.int_const(size, "s64"))
        else:
            wide = "mul.wide.u32" if var.nonneg else "mul.wide.s32"
            off = b.pure(wide, "s64", (var, Imm(size, "u32")), var.lo * size, var.hi * size)
        reg = b.pure("add.s64", "s64", (buf.pointer, off), commutative=True).op
        b.note_align(reg, align)
        return Mem(reg, disp)

    # ------------------------------------------------------------- statements: structure

    def stmt_Tile(self, s) -> None:
        b = self.b
        extents = [int(a.extent.as_static()) for a in s.axes]
        n = math.prod(extents) if extents else 1
        cells = s.block_threads or DEFAULT_BLOCK
        if s.aux_threads:
            per = b.int_const(s.block_threads)
            gid = b.add(b.mul(self.ctaid, per), b.rem(self.tid, per))
        else:
            gid = b.add(b.mul(self.ctaid, b.int_const(self.src.launch.block)), self.tid)
        # emmy guards when cells do not tile N; the range check also covers a
        # launch that is larger than the iteration space for any other reason
        needs_guard = s.aux_threads != 0 or n % cells != 0 or gid.hi >= n
        done = b.label("tile_end")
        if needs_guard:
            b.branch(done, b.icmp(">=", gid, b.int_const(n)))
            gid = b.assume(gid, gid.lo, n - 1)  # inside the guard
        # every lane of each warp runs the body, unless the guard can split a warp
        block = self.src.launch.block
        self.converged = not s.aux_threads and (not needs_guard or (n % WARP == 0 and block % WARP == 0))
        with self.block(guarded=needs_guard):
            for name, v in zip((a.name for a in s.axes), self._decode(s, gid, extents)):
                self.scope.names[name] = v
            self.body(s.body)
        self.converged = False
        b.place(done)

    def _decode(self, s, gid: Value, extents: list[int]) -> list[Value]:
        """Per-axis coordinates of the flat id (row-major), with emmy's optional
        grouped CTA rasterization of two adjacent block axes."""
        b = self.b
        strides = [math.prod(extents[i + 1 :]) for i in range(len(extents))]
        # Lanes l of a warp hold flat ids base + l, base a multiple of the
        # largest power of two dividing both the block size and 32. So
        # (gid / stride) % extent is the same throughout each aligned group of
        # lowbit(stride) lanes, capped there: the coordinate's uniform group.
        cap = 1 if s.aux_threads else min(WARP, _lowbit(self.src.launch.block))
        coords = []
        for i, (stride, extent) in enumerate(zip(strides, extents)):
            term = b.div(gid, b.int_const(stride))
            coord = term if i == 0 else b.rem(term, b.int_const(extent))
            coords.append(b.uniform(coord, min(cap, _lowbit(stride))))
        raster = _raster_plan(s, extents)
        if raster is None:
            return coords
        gi, oi, g = raster
        ea, eb = extents[gi], extents[oi]
        mi, ni = min(gi, oi), max(gi, oi)
        sub = b.div(gid, b.int_const(strides[ni]))
        if mi != 0:
            sub = b.rem(sub, b.int_const(ea * eb))
        gsize = b.int_const(g * eb)
        if ea % g == 0:
            a = b.add(b.mul(b.div(sub, gsize), b.int_const(g)), b.rem(sub, b.int_const(g)))
            o = b.div(b.rem(sub, gsize), b.int_const(g))
        else:
            grp = b.div(sub, gsize)
            size = b.imin(b.int_const(g), b.sub(b.int_const(ea), b.mul(grp, b.int_const(g))))
            a = b.add(b.mul(grp, b.int_const(g)), b.rem(b.rem(sub, gsize), size))
            o = b.div(b.rem(sub, gsize), size)
        group = min(cap, _lowbit(strides[ni]))  # both are functions of gid / strides[ni]
        coords[gi] = b.uniform(b.assume(a, 0, ea - 1), group)
        coords[oi] = b.uniform(b.assume(o, 0, eb - 1), group)
        return coords

    def stmt_Loop(self, s) -> None:
        extent = int(s.axis.extent.as_static())
        self._seed(s)
        self._loop(s, self.b.int_const(0), self.b.int_const(1), self.b.int_const(extent))

    def stmt_StridedLoop(self, s) -> None:
        self._seed(s)
        start, step = self.expr(s.start), self.expr(s.step)
        end = self.expr(s.end) if s.end is not None else self.b.int_const(int(s.axis.extent.as_static()))
        self._loop(s, start, step, end, end_name=f"{s.axis.name}_end" if s.end is not None else None)

    def _seed(self, loop) -> None:
        """Declare-and-seed each distinct Accum of the loop's immediate body."""
        seen = set()
        for st in loop.body:
            if type(st).__name__ == "Accum" and st.name not in seen:
                seen.add(st.name)
                canon = REDUCE_CANON.get(st.op.name)
                if canon is None:
                    raise LoweringError(f"Accum {st.name!r}: op {st.op.name!r} has no identity")
                if loop.seed:
                    dtype = elementwise.value_type(st.dtype.name if st.dtype is not None else "f32")
                    self.declare_var(st.name, dtype, self.b.convert(self.b.f32_const(IDENTITY[canon]), dtype))

    def _loop(self, s, start: Value, step: Value, end: Value, end_name: str | None = None) -> None:
        """``for (v = start; v < end; v += step) body``.

        Loops are unrolled completely when the trip count is bounded and the
        expanded body fits the budget: always for ``unroll`` loops, and for any
        loop of at most AUTO_UNROLL_TRIPS trips. Other loops with a constant
        step run ``factor`` iterations per trip of a rolled loop, followed by
        the leftover iterations. The iterations stay in their original order
        (so accumulations round exactly as written); unrolling only lets the
        loads of several iterations be in flight at once.
        """
        trips = _trip_range(start, step, end)
        size = _body_size(s.body)
        s, aliases = self._merge_congruent(s)
        with self.block(dominated=True):
            if end_name:  # the hoisted bound is visible to the body
                self.scope.names[end_name] = end
            self._loop_body(s, start, step, end, trips, size)
        for member, rep_ in aliases.items():  # equal at every iteration, so equal after the loop
            self.scope.rebind(member, self.lookup(rep_))

    def _merge_congruent(self, s):
        """``s`` with provably equal accumulators sharing one representative
        (see congruence.py), and the members to alias after the loop."""
        start = {}
        for st in s.body:
            if type(st).__name__ == "Accum":
                var = self.scope.lookup(st.name)
                start[st.name] = var.last if isinstance(var, Variable) else None
        members = congruent_accumulators(s, start)
        body = merge_congruent(s.body, members) if members else None
        if body is None:
            return s, {}
        return replace(s, body=body), members

    def _loop_body(self, s, start: Value, step: Value, end: Value, trips, size: int) -> None:
        """Pick the loop's shape: split across lanes, unrolled, or rolled."""
        group = self._split_plan(s, start, step, end, trips)
        if group:
            self._split_reduction(s, start, step, end, trips[0], group)
        elif trips is not None and trips[1] * size <= UNROLL_BUDGET and (s.unroll or trips[1] <= AUTO_UNROLL_TRIPS):
            for k in range(trips[1]):
                v = self.b.add(start, self.b.mul(step, self.b.int_const(k)))
                self._guarded_iteration(s.axis.name, v, s.body, end, guard=k >= trips[0])
        else:
            factor = _unroll_factor(size, self.src.launch.grid <= TARGET_SMS, trips) if step.const is not None else 1
            self._rolled(s, start, step, end, trips, factor)

    def _split_plan(self, s, start: Value, step: Value, end: Value, trips) -> int:
        """Lanes that can share the iterations of reduction loop ``s`` (0: none).

        Emmy's tile decode often gives every lane of an aligned group the same
        reduction to compute (all lanes of a warp take the same query rows in
        attention, and each computes the same dot products). When the folded
        values are uniform over groups of G lanes, those G lanes can split the
        iterations and combine their partial results with log2(G) shuffles:
        the same reduction, regrouped, which an associative and commutative
        combine allows. The loop qualifies when every lane of the warp runs it,
        its trip count is known, its body only loads, computes and folds, and
        nothing but the fold itself reads the accumulators.
        """
        if not self.converged or trips is None or trips[0] != trips[1] or trips[0] < SPLIT_MIN_TRIPS:
            return 0
        if not s.seed or any(type(st).__name__ not in ("Load", "Assign", "Let", "Accum") for st in s.body):
            return 0
        accs = [st for st in s.body if type(st).__name__ == "Accum"]
        names = {st.name for st in accs}
        if not accs:
            return 0
        for st in accs:
            if st.base is not None or REDUCE_CANON.get(st.op.name) not in SPLITTABLE:
                return 0
            if (st.dtype.name if st.dtype is not None else "f32") != "f32":
                return 0
        loop_ug = min(start.ug, step.ug, end.ug)
        env: dict[str, int] = {}

        def of_name(n: str) -> int:
            if n in env:
                return env[n]
            if n == s.axis.name:
                return loop_ug
            bound = self.scope.lookup(n)
            return bound.ug if isinstance(bound, (Value, Variable)) else 1

        def of_expr(e) -> int:
            return min([of_name(v) for v in e.free_vars()] + [WARP])

        group = loop_ug
        for st in s.body:
            kind = type(st).__name__
            reads = set()
            if kind == "Load":
                reads = {v for e in st.index for v in e.free_vars()}
                for n in st.names:
                    env[n] = min([of_expr(e) for e in st.index] + [WARP])
            elif kind == "Assign":
                reads = set(st.args)
                env[st.name] = min([of_name(a) for a in st.args] + [WARP])
            elif kind == "Let":
                reads = set(st.value.free_vars())
                env[st.name] = of_expr(st.value)
            else:  # Accum
                reads = {st.value}
                group = min(group, of_name(st.value))
            if reads & names:
                return 0  # an accumulator is read mid-fold: not a pure reduction
        group = min(group, 1 << (trips[0].bit_length() - 1))
        return group if group >= 2 else 0

    def _split_reduction(self, s, start: Value, step: Value, end: Value, trips: int, group: int) -> None:
        """Lane l of each aligned group of ``group`` lanes folds iterations
        l, l+group, l+2*group, ...; a butterfly of shuffles then gives every
        lane of the group the whole reduction."""
        b = self.b
        lane = self.value("lane")
        rank = lane if group == WARP else b.bitop("&", lane, b.int_const(group - 1))
        if trips % group == 0:  # contiguous chunks: each lane's loads are adjacent (vectorizable)
            chunk = trips // group
            first = b.mul(rank, b.int_const(chunk))
            for j in range(chunk):
                v = b.add(start, b.mul(b.add(first, b.int_const(j)), step))
                self._guarded_iteration(s.axis.name, v, s.body, end, guard=False)
        else:  # interleaved, the last round only on the lanes it reaches
            rounds = -(-trips // group)
            for j in range(rounds):
                v = b.add(start, b.mul(b.add(rank, b.int_const(group * j)), step))
                self._guarded_iteration(s.axis.name, v, s.body, end, guard=j == rounds - 1)
        mask = self.kernel.reg("s32")
        b.emit("activemask.b32", mask, ndefs=1, effect=False)
        for name in dict.fromkeys(st.name for st in s.body if type(st).__name__ == "Accum"):
            var = self.lookup(name)
            canon = REDUCE_CANON[next(st.op.name for st in s.body if type(st).__name__ == "Accum" and st.name == name)]
            width = group // 2
            while width:
                cur = b.in_reg(var.read())
                other = self.kernel.reg(var.type)
                b.emit("shfl.sync.bfly.b32", other, cur, Imm(width, "u32"), Imm(31, "u32"), mask, ndefs=1, effect=False)
                b.assign(var, elementwise.apply(b, canon, [var.read(), Value(var.type, other)], _dtype_name(var.type)))
                width //= 2
            var.ug = group  # every lane of the group now holds the whole fold

    def _rolled(self, s, start: Value, step: Value, end: Value, trips, factor: int) -> None:
        try:
            self._rolled_loops(s, start, step, end, trips, factor)
        finally:
            self._pessimize(s.body)  # the exit merges every trip count's final values

    def _rolled_loops(self, s, start: Value, step: Value, end: Value, trips, factor: int) -> None:
        b = self.b
        name = s.axis.name
        type_ = "s64" if "s64" in (start.type, end.type, step.type) else "s32"
        exact = trips is not None and trips[0] == trips[1]
        if trips is not None and trips[1] == 0:
            return
        counter = b.variable(type_, start)
        counter.ug = min(start.ug, step.ug, end.ug)
        self._pessimize(s.body)
        if factor > 1 and (trips is None or trips[1] >= factor):
            # main loop: `factor` iterations per trip while v + (factor-1)*step < end
            span = b.int_const((factor - 1) * step.const)
            limit = b.sub(end, span)
            body_hi = limit.hi - 1 if limit.hi is not None else None
            first = b.icmp("<", start, limit) if not (exact and trips[0] >= factor) else b.pred_const(True)
            top, rest = b.label("loop_x%d" % factor), b.label("loop_rest")
            if first.const is not False:
                if first.const is None:
                    b.branch(rest, first, negate=True)
                with b.loop():
                    b.place(top)
                    with self.block():
                        # within one trip the counter is constant: let its uses be value-numbered
                        # (the body's table closes before the latch; the counter is never hoisted)
                        base = Value(type_, counter.reg, start.lo, body_hi, ug=counter.ug)
                        for k in range(factor):
                            v = b.add(base, b.int_const(k * step.const))
                            self._guarded_iteration(name, v, s.body, end, guard=False)
                    nxt = b.add(Value(type_, counter.reg, start.lo, body_hi, mutable=True), b.int_const(factor * step.const))
                    b.assign(counter, nxt)
                    b.branch(top, b.icmp("<", nxt, limit))
                b.place(rest)
            if exact:  # the leftover is the same for every thread: straight-line code
                cur = Value(type_, counter.reg, start.lo, end.hi, ug=counter.ug)
                with b.cse_scope():
                    for k in range(trips[0] % factor):
                        self._guarded_iteration(name, b.add(cur, b.int_const(k * step.const)), s.body, end, guard=False)
                return
        # one iteration per trip (also the leftover of a partially unrolled loop)
        hi = end.hi - 1 if end.hi is not None else None
        cur = Value(type_, counter.reg, start.lo, None if hi is None else hi + abs(step.hi or 0) * max(factor, 1),
                    mutable=True)
        first = b.icmp("<", cur if factor > 1 else start, end)
        if first.const is False:
            return
        top, done = b.label("loop"), b.label("loop_end")
        if first.const is None:
            b.branch(done, first, negate=True)
        with b.loop():
            b.place(top)
            with self.block():
                self.scope.names[name] = Value(type_, counter.reg, start.lo, hi, ug=counter.ug)  # constant within a trip
                self.body(s.body)
            nxt = b.add(Value(type_, counter.reg, start.lo, hi, mutable=True), step)
            b.assign(counter, nxt)
            b.branch(top, b.icmp("<", nxt, end))
        b.place(done)

    def _guarded_iteration(self, name: str, v: Value, stmts, end: Value, guard: bool) -> None:
        """One unrolled iteration; ``guard`` when only some threads still run it."""
        if not guard:
            self._iteration(name, v, stmts, end)
            return
        b = self.b
        skip = b.label("trip_skip")
        runs = b.icmp("<", v, end)
        b.branch(skip, runs, negate=True)
        with self.block(guarded=True, divergent=runs.ug < WARP):
            self._iteration(name, v, stmts, end)
        b.place(skip)
        self._pessimize(stmts)

    def _iteration(self, name: str, v: Value, stmts, end: Value) -> None:
        with self.block(dominated=True):
            if end.hi is not None:
                v = self.b.assume(v, v.lo, end.hi - 1)  # inside the body, v < end
            self.scope.names[name] = v
            self.body(stmts)

    def stmt_Cond(self, s) -> None:
        b = self.b
        p = self.pred(s.cond)
        if p.const is not None:  # decided by range analysis: no branch at all
            with self.block(dominated=True):
                self.body(s.body if p.const else s.else_body)
            return
        other = b.label("else" if s.else_body else "endif")
        b.branch(other, p, negate=True)
        divergent = p.ug < WARP
        with self.block(guarded=True, divergent=divergent):
            self.body(s.body)
        if s.else_body:
            done = b.label("endif")
            b.branch(done)
            b.place(other)
            with self.block(guarded=True, divergent=divergent):
                self.body(s.else_body)
            b.place(done)
        else:
            b.place(other)
        self._pessimize((*s.body, *s.else_body))

    # ------------------------------------------------------------- statements: values

    def stmt_Let(self, s) -> None:
        if s.dtype is None:
            v = self.expr(s.value)
        else:
            v = self.b.convert(self.expr(s.value), elementwise.value_type(s.dtype.name))
        self.define(s.name, v)

    def stmt_Init(self, s) -> None:
        dtype = elementwise.value_type(s.dtype.name)
        self.declare_var(s.name, dtype, self.b.convert(self.b.f32_const(float(s.identity)), dtype))

    def stmt_Assign(self, s) -> None:
        args = [self.value(a) for a in s.args]
        op = s.op.name
        dtype = s.dtype.name if s.dtype is not None else elementwise.promote(op, [_dtype_of(a) for a in args])
        v = elementwise.apply(self.b, op, args, dtype)
        if (op == "multiply" and v.type == "f32" and v.const is None and all(a.type == "f32" for a in args)
                and not any(a.mutable for a in args)  # its factors are still those values at the fold
                and v.key() not in [a.key() for a in args]):  # a real product (x * 1 folds away)
            self.products[v.key()] = (args[0], args[1])  # an Accum may fuse it into an FMA
        self.define(s.name, v)

    def stmt_Accum(self, s) -> None:
        b = self.b
        acc = self.lookup(s.name)
        if not isinstance(acc, Variable):
            raise LoweringError(f"accumulator {s.name!r} used before it was seeded")
        base = self.value(s.base) if s.base is not None else acc.read()
        canon = REDUCE_CANON.get(s.op.name, "add")  # emmy folds unknown combines additively
        value = self.value(s.value)
        new = None
        if canon == "add" and acc.type == "f32":  # contract a product into the fold: one FMA
            for product, addend in ((value, base), (base, value)):
                factors = self.products.get(product.key()) if product.key() is not None else None
                if factors is not None and addend.type == "f32":
                    new = b.fma(factors[0], factors[1], addend)
                    break
        if new is None:
            new = elementwise.apply(b, canon, [base, b.convert(value, acc.type)], _dtype_name(acc.type))
        b.assign(acc, new)

    def stmt_Select(self, s) -> None:
        b = self.b
        branches = list(s.branches)
        result = b.convert(self.value(branches[-1].value), "f32")
        for br in reversed(branches[:-1]):
            result = b.select(self.pred(br.select), b.convert(self.value(br.value), "f32"), result)
        self.define(s.name, result)

    # ------------------------------------------------------------- statements: memory

    def stmt_Smem(self, s) -> None:
        self.scope.names[s.name] = self.shared[s.name]

    def stmt_Load(self, s) -> None:
        buf = self.buffer(s.input)
        mtype = MEM_TYPE[s.dtype.name] if s.dtype is not None else buf.type
        if isinstance(buf, GlobalBuffer) and s.input in self.src.constants and s.is_scalar:
            value = self.src.constants[s.input]  # a 1-element buffer holds exactly this
            vtype = VALUE_TYPE[mtype]
            const = self.b.int_const(int(value), vtype) if vtype in INT_RANGE else self.b.convert(self.b.f32_const(value), vtype)
            self.define(s.names[0], const)
            return
        values = self.load(buf, mtype, s.index, len(s.names))
        for name, v in zip(s.names, values):
            self.define(name, v)

    def load(self, buf, mtype: str, index, width: int, flat: Affine | None = None) -> list[Value]:
        """``width`` consecutive elements starting at ``index``, as few vector
        loads as the width allows. Read-only global data goes through the
        non-coherent path and is value-numbered (it cannot change)."""
        b = self.b
        base = flat if flat is not None else self.flat(buf, index)
        vtype = VALUE_TYPE[mtype]
        out: list[Value] = []
        shared = isinstance(buf, SharedArray)
        space = "shared" if shared else ("global.nc" if buf.readonly else "global")
        k = 0
        while k < width:
            n = _vector_width(width - k, mtype)
            addr = self.address(buf, flat=base.plus(Affine(const=k)))
            ug = b.ug_of(addr)  # lanes that read the same address read the same value
            if n == 1 and space == "global.nc":
                out.append(b.pure(f"ld.{space}.{_ldst(mtype)}", vtype, (addr,)))
            elif n == 1:
                reg = self.kernel.reg(vtype)
                b.emit(f"ld.{space}.{_ldst(mtype)}", reg, addr, ndefs=1, effect=False, ug=ug)
                out.append(Value(vtype, reg, *_type_range(vtype), ug=ug))
            else:
                regs = tuple(self.kernel.reg(vtype) for _ in range(n))
                b.emit(f"ld.{space}.v{n}.{_ldst(mtype)}", Vec(regs), addr, ndefs=1, effect=False, ug=ug)
                out.extend(Value(vtype, r, *_type_range(vtype), ug=ug) for r in regs)
            k += n
        return out

    def stmt_Write(self, s) -> None:
        b = self.b
        buf = self.buffer(s.output)
        vtype = VALUE_TYPE[buf.type]
        values = [b.in_reg(b.convert(self.value(v), vtype)) for v in s.values]
        base = self.flat(buf, s.index)
        shared = isinstance(buf, SharedArray)
        space = "shared" if shared else "global"
        if s.atomic:
            if buf.type != "f32" and buf.type != "s32":
                raise LoweringError(f"atomic add on {buf.type} is not supported")
            addr = self.address(buf, flat=base, swizzle=s.swizzle)
            op = "red.global.add.f32" if buf.type == "f32" else "red.global.add.s32"
            b.emit(op if not shared else op.replace("global", "shared"), addr, values[0])
            return
        k = 0
        while k < len(values):
            n = _vector_width(len(values) - k, buf.type)
            addr = self.address(buf, flat=base.plus(Affine(const=k)), swizzle=s.swizzle)
            if n == 1:
                b.emit(f"st.{space}.{_ldst(buf.type)}", addr, values[k])
            else:
                b.emit(f"st.{space}.v{n}.{_ldst(buf.type)}", addr, Vec(tuple(v.op for v in values[k : k + n])))
            k += n

    def stmt_Sync(self, s) -> None:
        if s.warp:
            self.b.emit("bar.warp.sync", Imm(-1, "s32"))
        elif s.barrier_id == 0:
            self.b.emit("bar.sync", Imm(0, "u32"))
        else:
            if s.count is None:
                raise LoweringError("named barrier without a thread count")
            self.b.emit("bar.sync", Imm(s.barrier_id, "u32"), Imm(s.count, "u32"))

    def stmt_CpAsyncCopy(self, s) -> None:
        b = self.b
        smem, src = self.buffer(s.smem), self.buffer(s.src)
        if not isinstance(smem, SharedArray) or not isinstance(src, GlobalBuffer):
            raise LoweringError("cp.async must copy from a global buffer into a shared array")
        dst = self.flat(smem, s.smem_index)
        if s.lane_index is not None:
            row, col = (self.affine(e) for e in s.lane_index)
            dst = dst.plus(row.scaled(smem.shape[-1])).plus(col)
        dst_addr = self.address(smem, flat=dst, swizzle=s.swizzle)
        src_addr = self.address(src, s.src_index)
        if s.nbytes not in (4, 8, 16):
            raise LoweringError(f"cp.async of {s.nbytes} bytes")
        cache = "cg" if s.nbytes == 16 else "ca"
        b.emit(f"cp.async.{cache}.shared.global", dst_addr, src_addr, Imm(s.nbytes, "u32"))

    def stmt_CpAsyncCommit(self, s) -> None:
        self.b.emit("cp.async.commit_group")

    def stmt_CpAsyncWait(self, s) -> None:
        self.b.emit("cp.async.wait_group", Imm(int(s.group), "u32"))

    def stmt_ZeroPrologue(self, s) -> None:
        """CTA 0 zeroes ``words`` 32-bit words of another kernel's accumulator."""
        b = self.b
        buf = self.globals[s.dst]
        skip = b.label("zero_end")
        b.branch(skip, b.icmp("!=", self.ctaid, b.int_const(0)))
        words = GlobalBuffer(buf.name, buf.pointer, "u32", (s.words,), False)
        top = b.label("zero")
        counter = b.variable("s32", self.tid, lo=0, hi=s.words - 1)
        first = b.icmp("<", self.tid, b.int_const(s.words))
        if first.const is not True:
            b.branch(skip, first, negate=True)
        b.place(top)
        i = Value("s32", counter.reg, 0, s.words - 1)
        addr = self.address(words, flat=Affine({("zp",): [i, 1]}))
        b.emit("st.global.u32", addr, Imm(0, "u32"))
        nxt = b.add(i, b.int_const(self.src.launch.block))
        b.assign(counter, nxt)
        b.branch(top, b.icmp("<", nxt, b.int_const(s.words)))
        b.place(skip)

    # ------------------------------------------------------------- statements: cross-thread

    def stmt_WarpShuffle(self, s) -> None:
        b = self.b
        vtype = elementwise.value_type(s.dtype.name)
        mask = self.kernel.reg("s32")
        b.emit("activemask.b32", mask, ndefs=1, effect=False)
        lane_mask = int(s.length) // 2
        while lane_mask > 0:
            with self.block():
                for st, sb in zip(s.state, s.state_b):
                    cur = b.in_reg(b.convert(self.value(st), vtype))
                    out = self.kernel.reg(vtype)
                    b.emit("shfl.sync.bfly.b32", out, cur, Imm(lane_mask, "u32"), Imm(31, "u32"), mask, ndefs=1, effect=False)
                    self.define(sb, Value(vtype, out))
                self.merge(s.combine_states, s.state, vtype)
            lane_mask //= 2
        for st in s.state:  # the butterfly leaves each aligned group of `length` lanes agreeing
            var = self.scope.lookup(st)
            if isinstance(var, Variable):
                var.ug = max(var.ug, min(WARP, int(s.length)))

    def stmt_TreeHalve(self, s) -> None:
        b = self.b
        vtype = elementwise.value_type(s.dtype.name)
        t = self.value(s.tid_var)
        bufs = [self.buffer(name) for name in s.bufs]
        half = int(s.length) // 2
        while half > 0:
            skip = b.label("tree_skip")
            active = b.icmp("<", t, b.int_const(half))
            b.branch(skip, active, negate=True)
            with self.block(guarded=True, divergent=active.ug < WARP):
                if s.inner is not None:
                    iv, scale = self.value(s.inner[0]), b.int_const(int(s.inner[1]))
                    slot = b.add(b.mul(t, scale), iv)
                    other = b.add(b.mul(b.add(t, b.int_const(half)), scale), iv)
                else:
                    slot, other = t, b.add(t, b.int_const(half))
                for buf, st, sb in zip(bufs, s.state, s.state_b):
                    self.define(st, self.load_one(buf, slot, vtype))
                    self.define(sb, self.load_one(buf, other, vtype))
                self.merge(s.combine_states, s.state, vtype)
                for buf, st in zip(bufs, s.state):
                    self.store_one(buf, slot, self.value(st))
            b.place(skip)
            if s.barrier_id == 0:
                b.emit("bar.sync", Imm(0, "u32"))
            else:
                b.emit("bar.sync", Imm(s.barrier_id, "u32"), Imm(int(s.barrier_count), "u32"))
            half //= 2
        root = self.value(s.inner[0]) if s.inner is not None else b.int_const(0)
        for buf, st in zip(bufs, s.state):
            self.reassign(st, self.load_one(buf, root, vtype))

    def merge(self, program, state, vtype: str) -> None:
        """Run a combine program: an Assign to a state name updates that
        (innermost) variable; any other Assign declares a temporary."""
        state = set(state)
        for a in program:
            args = [self.value(x) for x in a.args]
            v = elementwise.apply(self.b, a.op.name, args, _dtype_name(vtype))
            if a.name in state:
                self.reassign(a.name, v)
            else:
                self.define(a.name, v)

    def load_one(self, buf: SharedArray, index: Value, vtype: str) -> Value:
        """One shared-memory element at an already computed index."""
        return self.b.convert(self.load(buf, buf.type, None, 1, flat=self._as_affine(index))[0], vtype)

    def store_one(self, buf: SharedArray, index: Value, v: Value) -> None:
        v = self.b.in_reg(self.b.convert(v, VALUE_TYPE[buf.type]))
        self.b.emit(f"st.shared.{_ldst(buf.type)}", self.address(buf, flat=self._as_affine(index)), v)

    def _as_affine(self, v: Value) -> Affine:
        """An integer value as an affine form, looking through ``base + c``
        definitions so the constant can become an address immediate."""
        if v.const is not None:
            return Affine(const=int(v.const))
        linear = self.b.linear_of(v)
        if linear is not None:
            terms, const = linear
            out = Affine(const=const)
            for t, c in terms:
                out = out.plus(self._as_affine(t).scaled(c))
            return out
        key = v.key()
        if key is None:  # a mutable register: its own term, never merged
            self._uniq += 1
            key = ("var", self._uniq)
        return Affine({key: [v, 1]})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _param_name(buffer: str) -> str:
    return f"p_{buffer}"


def _walk(stmts):
    for s in stmts:
        yield s
        for child in s.nested():
            yield from _walk(child)


def _reassigned_names(body) -> set[str]:
    """Names that some statement reassigns after declaring them."""
    out = set()
    for s in _walk(body):
        kind = type(s).__name__
        if kind in ("Accum", "Init"):
            out.add(s.name)
        elif kind in ("WarpShuffle", "TreeHalve"):
            out.update(s.state)
    return out


def _body_size(stmts) -> int:
    """Statements in a body after the full unrolling we would apply inside it."""
    total = 0
    for s in stmts:
        kind = type(s).__name__
        if kind == "Loop":
            inner = _body_size(s.body)
            extent = int(s.axis.extent.as_static())
            total += inner * extent if s.unroll and inner * extent <= UNROLL_BUDGET else inner + 4
        elif kind in ("StridedLoop", "Cond"):
            total += sum(_body_size(b) for b in s.nested()) + 2
        elif kind == "Tile":
            total += _body_size(s.body)
        elif kind in ("WarpShuffle", "TreeHalve"):
            total += 8 * max(1, int(s.length).bit_length()) * max(1, len(s.combine_states))
        else:
            total += 1
    return max(total, 1)


def _issue_loads_early(stmts, readonly: frozenset) -> tuple:
    """``stmts`` with every load of read-only global memory moved up past the
    statements before it that do not produce any name its index reads.

    Nothing can change read-only memory, so the load computes the same value
    earlier; issuing loads before the arithmetic between them lets their
    latencies overlap (and lets adjacent ones merge into one vector load),
    which matters most when that arithmetic contains branches.
    """
    out: list = []
    for s in stmts:
        if type(s).__name__ == "Load" and s.input in readonly:
            needs = {v for e in s.index for v in e.free_vars()}
            pos = len(out)
            while pos > 0 and type(out[pos - 1]).__name__ != "Load" and not (_produces(out[pos - 1]) & needs):
                pos -= 1
            out.insert(pos, s)
        else:
            out.append(s)
    return tuple(out)


def _produces(s) -> set[str]:
    """Names a statement declares or reassigns that later statements can see."""
    kind = type(s).__name__
    if kind == "Load":
        return set(s.names)
    if kind in ("Assign", "Let", "Init", "Select", "Accum", "Smem"):
        return {s.name}
    return _assigned_in((s,)) | ({"*"} if kind in ("WarpShuffle", "TreeHalve") else set())


def _lowbit(n: int) -> int:
    """The largest power of two dividing ``n``."""
    return n & -n


def _assigned_in(stmts) -> set[str]:
    """Names some statement in ``stmts`` (at any depth) reassigns."""
    out = set()
    for s in _walk(stmts):
        kind = type(s).__name__
        if kind == "Accum":
            out.add(s.name)
        elif kind in ("WarpShuffle", "TreeHalve"):
            out.update(s.state)
    return out


def _unroll_factor(body_size: int, one_wave: bool, trips) -> int:
    """Iterations per trip for a rolled loop: enough independent loads in
    flight to cover memory latency, without blowing up big bodies.

    A one-wave grid has too few warps to hide latency, so even big bodies
    are unrolled: by the largest of UNROLL_BIG_CHOICES that divides an exact
    trip count (no leftover iterations to expand), else the last choice."""
    if body_size <= UNROLL_TINY_BODY:
        return UNROLL_TINY
    if body_size <= UNROLL_SMALL_BODY:
        return UNROLL_SMALL
    if not one_wave or body_size > UNROLL_BIG_BODY:
        return 1
    if trips is not None and trips[0] == trips[1]:
        for f in UNROLL_BIG_CHOICES:
            if trips[0] % f == 0:
                return f
    return UNROLL_BIG_CHOICES[-1]


def _trip_range(start: Value, step: Value, end: Value):
    """(min, max) trip count of ``for (v = start; v < end; v += step)`` over the
    ranges of its bounds, or ``None`` when the step is not a known positive int."""
    if step.const is None or step.const <= 0 or None in (start.lo, start.hi, end.lo, end.hi):
        return None
    if end.const is None and start.const is None:
        return None

    def trips(a, e):
        return max(0, -(-(e - a) // step.const))

    return trips(start.hi, end.lo), trips(start.lo, end.hi)


def _raster_plan(s, extents):
    if not (s.raster_group and s.raster_axes):
        return None
    names = [a.name for a in s.axes]
    if s.raster_axes[0] not in names or s.raster_axes[1] not in names:
        return None
    mi, ni = names.index(s.raster_axes[0]), names.index(s.raster_axes[1])
    if ni != mi + 1:
        return None
    gi, oi = (mi, ni) if s.raster_orient == "m" else (ni, mi)
    g = min(int(s.raster_group), extents[gi])
    if extents[gi] > 1 and g > 1:
        return gi, oi, g
    return None


_SWIZZLE_MASK = {"B32": 1, "B64": 3, "B128": 7}


def _swizzle_mask(mode: str):
    base, _, shift = mode.partition("@")
    mask = _SWIZZLE_MASK.get(base)
    if mask is None:
        return None
    return (int(shift) if shift else 6), mask


def _swizzled(b: Builder, e: Value, mode: str) -> Value:
    """``e ^ (((e >> shift) & mask) << 3)``: the XOR swizzle of a slab index."""
    shift, mask = _swizzle_mask(mode)
    field_ = b.bitop("&", b.shr(e, b.int_const(shift)), b.int_const(mask))
    return b.bitop("^", e, b.shl(field_, b.int_const(3)))


def _promote_pair(b: Builder, x: Value, y: Value):
    if x.type == y.type:
        return x, y
    if x.is_float or y.is_float:
        t = "f32" if "f64" not in (x.type, y.type) else "f64"
        return b.convert(x, t), b.convert(y, t)
    if x.type == "pred" or y.type == "pred":
        return b.convert(x, "s32"), b.convert(y, "s32")
    return x, y


def _type_range(vtype: str):
    return INT_RANGE.get(vtype, (None, None))


def _ldst(mtype: str) -> str:
    """Type suffix of ld/st for a memory type (PTX has no .f16/.bf16 loads)."""
    return "b16" if mtype in ("f16", "bf16") else mtype


def _vector_width(remaining: int, mtype: str) -> int:
    """Widest ld/st.vN that fits ``remaining`` elements (vectors of f32 only)."""
    if mtype != "f32":
        return 1
    return 4 if remaining >= 4 else 2 if remaining >= 2 else 1


def _dtype_of(v: Value) -> str:
    return _dtype_name(v.type)


def _dtype_name(vtype: str) -> str:
    return {"s32": "i32", "s64": "i64", "pred": "f32"}.get(vtype, vtype)
