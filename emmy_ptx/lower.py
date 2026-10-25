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

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Assign, Cond, Loop
from emmy.compiler.ir.stmt.leaves import Let

from . import elementwise, mapping
from .builder import INT_RANGE, WARP, Builder, Value, Variable, eliminate_dead_code
from .congruence import congruent_accumulators, drop_needless_snapshots, merge_congruent
from .frontend import DEFAULT_BLOCK, KernelSource
from .mapping import LOOPS, LaneSlot, assigned, kind, produced, reads, walk
from .ptx import PTX_LIMIT, PTX_VERSION, SIZEOF, TARGET, Imm, Kernel, Mem, Sym, Vec
from .regroup import fully_combined, regroup
from .tensorcore import TensorCoreLowering, fragment_index, fragment_refs
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

# memory type -> the red.add that adds a value to it (an integer add is the same for either sign)
ATOMIC_ADD = {"f32": "f32", "f64": "f64", "f16": "noftz.f16", "s32": "s32", "u32": "u32", "s64": "u64", "u64": "u64"}

STATIC_SMEM_LIMIT = 48 * 1024
# Alignment (bytes) assumed for every buffer base: cudaMalloc returns 256-byte
# aligned memory and emmy's own float4 accesses rely on 16; shared arrays are
# declared .align 16 or more.
MAX_ALIGN = 16
# Fully unroll an ``unroll`` loop only while the expanded body stays below this
# many statements: past it the PTX grows faster than ptxas can schedule it.
UNROLL_BUDGET = 4096
# Loops this short are unrolled completely even when the IR does not ask (loops
# over a shared-memory tile up to SMEM_UNROLL_TRIPS).
AUTO_UNROLL_TRIPS, SMEM_UNROLL_TRIPS = 64, 128
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
# Bytes of an L1 cache line: a warp load costs one pass per line it touches.
LINE_BYTES = 128
# A split lane lowers consecutive iterations together (unroll-and-jam) only
# while their inner loops keep at most this many accumulators live.
JAM_ACCUMULATORS = 64
# Sibling loops are fused only while the fused body stays this small (statements).
FUSE_BUDGET = 512
# Registers of an SM (and so of one CTA), and the most one thread can address.
REGISTER_FILE, THREAD_REGISTERS = 65536, 256
# Shared array through which lane-blocked loops broadcast (stmt_LaneStash), when
# a block broadcasts at least LANE_BUFFER_MIN values per lane.
LANE_BUFFER, LANE_BUFFER_MIN = "smem_lanes", 8
# Lanes that split one iteration's shared part when one lane per iteration would
# read a cache line per lane (see Lowerer._key_lanes): a lane per this many bytes
# of the iteration's row, at most KEY_LANES (8 lanes x 16 bytes cover a line).
ROW_BYTES_PER_LANE, KEY_LANES = 64, 8
# A straight-line reduction split across lanes gives each lane one contiguous
# chunk while chunks are at most this long; past it the lanes take vectors side by
# side, in rounds that form a loop of their own beyond SPLIT_UNROLLED_ROUNDS.
CHUNK_TRIPS, SPLIT_UNROLLED_ROUNDS = 16, 16

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


@dataclass(frozen=True)
class Fold:
    """A loop's reduction as lanes can share it (see Lowerer._fold)."""

    group: int  # aligned lanes that fold the same values
    states: tuple[str, ...]  # the carried accumulators
    # independent merges: (states, Assigns merging a partner lane's copies, read as "<state>$b")
    merges: tuple[tuple[tuple[str, ...], tuple], ...]


@dataclass(frozen=True)
class SplitPlan:
    """How a lane-shared reduction is dealt to its lanes (see Lowerer._split_plan)."""

    fold: Fold
    group: int  # lanes the iterations are split over (a power of two)
    vector: int  # consecutive iterations a lane takes at a time
    jam: int  # consecutive iterations of a lane lowered as one body


@dataclass(frozen=True)
class Placement:
    """Where the cells of a Tile run (see Lowerer._placement): grouped by every
    coordinate but ``axis``'s, the cells of each group on aligned groups of
    ``parts[k]`` lanes in region k, the regions one after another."""

    axis: int  # the Tile axis whose coordinate varies within a group
    parts: tuple[int, ...]  # lanes of a group in each region (powers of two, at most a warp)
    groups: int  # cells / the axis's extent


class Lowerer(TensorCoreLowering):
    def __init__(self, src: KernelSource, spread: bool = False, plain: bool = False, place: bool = False,
                 strict: bool = False):
        self.src = src
        self.spread = spread  # do idle threads join the cells (_spread)?
        self.place = place  # do cells run where their lanes share work (_placement)?
        # every floating-point operation of the IR in its order, each rounded on its own: no rewrite
        # that regroups a sum (lane splits, chunks, a softmax updated once per block) and no FMA contraction
        self.strict = strict
        # the fallback lowering (see compile_file): one rule per statement, in the IR's order, with
        # none of the optional rewrites (fusion, regrouping, lane splits, unrolling the IR does not ask for)
        self.plain = plain
        self.cell_threads = 1  # threads per cell when a cell spans warps (a whole block), else 1
        self.shared_loop = False  # did lanes share a loop's work in the Tile being lowered?
        self.wants_spread = False  # would a Tile gain from spread cells (see run)?
        self.wants_place = False  # would one gain from placed cells (see run)?
        self.placed_shared = False  # did a Tile with placed cells share a loop's work?
        self.store_guard: Value | None = None  # only lanes for which this holds store (_placed_tile)
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
        self._lane_region = None  # (byte offset, rows per warp) of the lane buffer for the current blocked loop
        self._lane_bytes = 0  # size of the lane buffer
        self._stashed: dict = {}  # broadcast value -> where stmt_LaneStash put it
        self._static_smem = 0  # bytes of static shared arrays (None: a dynamic pool)
        self.shared = self._layout_shared()
        self.globals = self._prologue()
        self._readonly = frozenset(n for n, g in self.globals.items() if g.readonly)

    # ------------------------------------------------------------------ driver

    def run(self) -> Kernel:
        self.body(self.src.body)
        if not (self.spread or self.place):
            # cells that, as emmy lays them out, share no work with the lanes beside them may
            # share it with more lanes each (_spread) or laid out anew (_placement)
            if self.wants_spread:
                return Lowerer(self.src, spread=True, strict=self.strict).run()
            if self.wants_place:
                placed = Lowerer(self.src, place=True, strict=self.strict)
                kernel = placed.run()
                # each region lowers the Tile's body again: keep it while the module fits the runner
                if placed.placed_shared and len(kernel.render(PTX_VERSION, TARGET)) <= PTX_LIMIT:
                    return kernel
        self.b.emit("ret")
        if self._lane_bytes:
            self.kernel.decls.append(f".shared .align 16 .b8 {LANE_BUFFER}[{self._lane_bytes}];")
        self.kernel.flatten()
        eliminate_dead_code(self.kernel)
        if not self.plain:
            vectorize_loads(self.kernel, self.b.align)
        return self.kernel

    def body(self, stmts) -> None:
        stmts = list(stmts if self.plain else _issue_loads_early(stmts, self._readonly))
        i = 0
        while i < len(stmts):
            s = stmts[i]
            if not self.plain:
                # later loops over the same iterations join this one (exact; see mapping.py)
                s = mapping.absorb_siblings(stmts, i, self._fusable)
                if kind(s) == "StridedLoop" and self.converged:
                    s = self._regrouped(stmts, i, s)
            handler = getattr(self, "stmt_" + kind(s), None)
            if handler is None:
                raise LoweringError(f"unsupported statement {kind(s)}")
            handler(s)
            i += 1

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

    def _regrouped(self, stmts, i: int, s):
        """``s`` in chunked order when its accumulators flow straight into a full
        combine across the cooperating threads (see regroup.py), else ``s``."""
        accs = {st.name for st in s.body if type(st).__name__ == "Accum"}
        step = _int_value(s.step)
        if not accs or step is None or self.strict:  # (chunks regroup the sums)
            return s
        start = self.expr(s.start)  # the thread's rank among the cooperating threads
        if start.lo != 0 or start.hi is None or not fully_combined(stmts, i, accs, start.hi + 1):
            return s
        return regroup(s, start.hi, step, int(s.axis.extent.as_static())) or s

    def _fusable(self, s, t) -> bool:
        """May sibling loops ``s`` and ``t`` be fused? Only loops that stay loops:
        one that becomes straight-line code (unrolled completely, or split across
        lanes) already shares its loads with its siblings through value
        numbering, and fusing would only reorder it. Both must also share their
        reduction over the same lanes (or neither), so fusing cannot cost a
        split, and the fused body must stay small."""
        if _body_size(s.body) + _body_size(t.body) > FUSE_BUDGET:
            return False
        group = self._shared_by(s)
        if group != self._shared_by(t):
            return False
        return not (self._straight_line(s, group) or self._straight_line(t, group))

    def _straight_line(self, loop, group: int) -> bool:
        """Will ``loop`` (whose reduction ``group`` lanes share) be lowered without a
        loop of its own: split across those lanes, or unrolled completely?"""
        if any(kind(st) in LOOPS for st in loop.body):
            return False
        if group >= 2:
            return True
        extent = int(loop.axis.extent.as_static())
        unrolls = loop.unroll or extent <= self._auto_unroll(loop)
        return kind(loop) == "Loop" and unrolls and extent * _body_size(loop.body) <= UNROLL_BUDGET

    def _shared_by(self, s) -> int:
        """Aligned lanes that compute loop ``s``'s reduction alike (0: not a lane-shareable reduction)."""
        extent = self.b.int_const(int(s.axis.extent.as_static()))
        if kind(s) == "Loop":
            bounds = (self.b.int_const(0), self.b.int_const(1), extent)
        else:
            bounds = (self.expr(s.start), self.expr(s.step), self.expr(s.end) if s.end is not None else extent)
        fold = self._fold(s, *bounds)
        return fold.group if fold is not None else 0

    def _pessimize(self, stmts) -> None:
        """Forget the uniformity of the variables ``stmts`` reassign: at a merge
        point or a loop head their value is not known to be uniform."""
        for name in assigned(stmts):
            var = self.scope.lookup(name)
            if isinstance(var, Variable):
                var.forget()
        self.forget_fragments(stmts)

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
        for s in walk(self.src.body):
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
        self._static_smem = None if dynamic else cursor
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

    def expr(self, e, ftype: str = "f32") -> Value:
        """The value of expression ``e``; float literals in it are ``ftype``
        constants (f64 in the value of an f64 Let)."""
        kind = type(e).__name__
        b = self.b
        if kind == "Var":
            return self.value(e.name)
        if kind == "Literal":
            if isinstance(e.value, float) or e.dtype == "float":
                return b.float_const(float(e.value), ftype)
            return b.int_const(int(e.value))
        if kind == "BinaryExpr":
            return self._binary(e.op, e.left, e.right, ftype)
        if kind == "TernaryExpr":
            p = self.pred(e.cond)
            t, f = self.expr(e.if_true, ftype), self.expr(e.if_false, ftype)
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

    def _binary(self, op: str, left, right, ftype: str = "f32") -> Value:
        b = self.b
        if op in ("&&", "||"):
            x, y = self.pred(left), self.pred(right)
            return b.pand(x, y) if op == "&&" else b.por(x, y)
        x, y = self.expr(left, ftype), self.expr(right, ftype)
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
        spread = self._spread(s, n)
        plan = self._placement(s, extents) if self.place and spread == 1 else None
        if plan is not None:
            self.shared_loop = False
            self._placed_tile(s, plan, extents)
            self.placed_shared |= self.shared_loop
            return
        if s.aux_threads:
            per = b.int_const(s.block_threads)
            gid = b.add(b.mul(self.ctaid, per), b.rem(self.tid, per))
        else:
            gid = b.add(b.mul(self.ctaid, b.int_const(self.src.launch.block)), self.tid)
        # emmy guards when cells do not tile N; the range check also covers a
        # launch that is larger than the iteration space for any other reason
        lanes = n * spread
        needs_guard = s.aux_threads != 0 or (spread == 1 and n % cells != 0) or gid.hi >= lanes
        done = b.label("tile_end")
        if needs_guard:
            b.branch(done, b.icmp(">=", gid, b.int_const(lanes)))
            gid = b.assume(gid, gid.lo, lanes - 1)  # inside the guard
        # every lane of each warp runs the body, unless the guard can split a warp
        block = self.src.launch.block
        self.converged = not s.aux_threads and (not needs_guard or (lanes % WARP == 0 and block % WARP == 0))
        cell = b.shr(gid, b.int_const(spread.bit_length() - 1)) if spread > 1 else gid
        self.shared_loop, self.cell_threads = False, spread if spread > WARP else 1
        coords = self._decode(s, cell, extents, spread)
        with self.block(guarded=needs_guard):
            for name, v in zip((a.name for a in s.axes), coords):
                self.scope.names[name] = v
            self.body(s.body)
        # cells that already share their work across lanes (one query row for all, say) gain
        # nothing from more lanes each; cells that fold alone, one per lane, do (see run)
        self.wants_spread |= not self.shared_loop and self._spread(s, n, candidate=True) > 1
        plan = self._placement(s, extents) if spread == 1 else None
        if plan is not None:  # worth it where every lane group would be larger than the lanes a group has now
            alike = min((v.ug for i, v in enumerate(coords) if i != plan.axis and extents[i] > 1), default=WARP)
            self.wants_place |= min(plan.parts) > alike
        self.converged, self.cell_threads = False, 1
        b.place(done)

    def _spread(self, s, n: int, candidate: bool = False) -> int:
        """Lanes per cell of Tile ``s``: 1, unless the launch has at least twice
        as many threads as there are cells (and the kernel is lowered with
        ``spread``, or ``candidate`` asks what spreading would give).

        Some emmy schedules fold a whole row in one thread while most of the
        block idles (a 65536-element sum on one thread of 256). When nothing a
        cell does depends on which thread runs it, cell c may as well run on an
        aligned group of lanes (cell = gid / spread): they compute it alike, so
        the lane split of reductions (``_split_plan``) shares its folds among
        them and ``_dealt`` its independent loops. ``run`` lowers a kernel with
        spread cells only when, lowered without, the lanes shared no loop."""
        launch = self.src.launch
        if self.plain or not (self.spread or candidate) or s.aux_threads or not _thread_free(s.body):
            return 1
        if any(kind(x) == "Write" and x.atomic for x in walk(s.body)):
            return 1  # the lanes of a cell would each add its result
        if not any(kind(x) in LOOPS for x in walk(s.body)):
            return 1  # nothing to share
        limit = min(WARP, _lowbit(launch.block))
        room = self._static_smem is not None and self._static_smem + self._lane_bytes + 4096 <= STATIC_SMEM_LIMIT
        if n == launch.grid and _lowbit(launch.block) == launch.block and room:
            limit = launch.block  # one cell per block: it may span the warps (_merge_warps joins them)
        spread = 1
        while spread < limit and n * spread * 2 <= launch.grid * launch.block:
            spread *= 2
        return spread

    def _placement(self, s, extents: list[int]) -> Placement | None:
        """Where the cells of Tile ``s`` may run so that more lanes share work,
        or None.

        Emmy lays the cells out row-major, so lanes beside each other share a
        query row only when the row's cells fill aligned groups of lanes. With
        11 cells to a row (three output columns each, 32 columns), or cells
        rasterized lane by lane (consecutive lanes on consecutive rows), no
        group of lanes holds one row, and every lane computes its row's
        scores itself. Cells that do not depend on which thread runs them may
        run anywhere, though. Grouped by every coordinate but the fastest
        axis's, a group's cells go to aligned groups of lanes of power-of-two
        sizes: 20 = 16 + 4; 11 = 16 when there are threads enough, else 8 + 4
        with one lane idle, else 8 + 2 + 1 (_placed_tile). Worth trying only
        when some loop's work is the same along that axis."""
        launch = self.src.launch
        if self.plain or s.aux_threads or launch.block % WARP or not _thread_free(s.body):
            return None
        axis = max((i for i, e in enumerate(extents) if e > 1), default=None)
        if axis is None:
            return None
        along = mapping.dependents(s.body, {s.axes[axis].name})
        if not any(kind(x) in LOOPS and mapping.trip_estimate(x) >= SPLIT_MIN_TRIPS and not reads((x,)) & along
                   for x in walk(s.body)):
            return None  # no loop's work is the same for all cells of a group
        groups = math.prod(extents) // extents[axis]
        for parts in _decompositions(extents[axis]):
            if sum(_round_up(groups * part, WARP) for part in parts) <= launch.grid * launch.block:
                return Placement(axis, parts, groups)
        return None

    def _placed_tile(self, s, plan: Placement, extents: list[int]) -> None:
        """Tile ``s`` with its cells where ``plan`` puts them. The regions are
        whole warps, one after another: lane u of region k (P = plan.parts[k]
        lanes a group) takes group u / P, and that group's cell number
        (sum of the earlier parts) + u % P along the plan's axis. So every
        other coordinate is the same on aligned groups of P lanes, and the
        lowering of the body (lane splits and blocks) shares their work. A
        lane past the last group or the last cell computes a copy of the last
        one, so that whole groups take part, and stores nothing."""
        b = self.b
        names = [a.name for a in s.axes]
        size = extents[plan.axis]
        others = [i for i in range(len(extents)) if i != plan.axis]
        stride = {i: math.prod(extents[j] for j in others if j > i) for i in others}
        regions = [_round_up(plan.groups * part, WARP) for part in plan.parts]
        gid = b.add(b.mul(self.ctaid, b.int_const(self.src.launch.block)), self.tid)
        done = b.label("tile_end")
        if gid.hi >= sum(regions):
            b.branch(done, b.icmp(">=", gid, b.int_const(sum(regions))))
        base = first = 0
        for part, lanes in zip(plan.parts, regions):
            after = b.label("cells_end")
            if base + lanes < sum(regions):
                b.branch(after, b.icmp(">=", gid, b.int_const(base + lanes)))  # whole warps: no divergence
            self.converged = True
            with self.block(guarded=True):
                u = b.assume(b.sub(gid, b.int_const(base)), 0, lanes - 1)
                group = b.shr(u, b.int_const(part.bit_length() - 1))
                cell = b.add(b.bitop("&", u, b.int_const(part - 1)), b.int_const(first))
                own = None  # does the lane hold a cell of its own?
                clamped = lanes > plan.groups * part
                if clamped:  # lanes past the last group
                    own = b.icmp("<", group, b.int_const(plan.groups))
                    group = b.imin(group, b.int_const(plan.groups - 1))
                if first + part > size:  # lanes past the group's last cell
                    real = b.icmp("<", cell, b.int_const(size))
                    own = real if own is None else b.pand(own, real)
                    cell = b.imin(cell, b.int_const(size - 1))
                self.store_guard = own
                self.scope.names[names[plan.axis]] = cell
                for i in others:
                    coord = b.rem(b.div(group, b.int_const(stride[i])), b.int_const(extents[i]))
                    alike = part if clamped else part * _lowbit(stride[i])  # lanes holding the same coordinate
                    self.scope.names[names[i]] = b.uniform(coord, min(WARP, alike))
                self.body(s.body)
            self.store_guard = None
            b.branch(done)
            b.place(after)
            base, first = base + lanes, first + part
        self.converged = False
        b.place(done)

    def _decode(self, s, gid: Value, extents: list[int], spread: int = 1) -> list[Value]:
        """Per-axis coordinates of flat cell id ``gid`` (row-major), with emmy's
        optional grouped CTA rasterization of two adjacent block axes; each cell
        runs on ``spread`` consecutive lanes."""
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
            coords.append(b.uniform(coord, min(cap, _lowbit(stride) * spread)))
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
        group = min(cap, _lowbit(strides[ni]) * spread)  # both are functions of gid / strides[ni]
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
        if self.plain:
            return s, {}
        start = {}
        for st in s.body:
            if type(st).__name__ == "Accum":
                var = self.scope.lookup(st.name)
                start[st.name] = var.last if isinstance(var, Variable) else None
        members = congruent_accumulators(s, start)
        body = merge_congruent(s.body, members) if members else None
        if body is None:
            return s, {}
        # what fed only the dropped updates (a member's own inner loops, say) is now dead, and the
        # members' copies of the representative's arithmetic are repeats
        return replace(s, body=drop_needless_snapshots(mapping.cse(mapping.prune_dead(body)))), members

    def _loop_body(self, s, start: Value, step: Value, end: Value, trips, size: int) -> None:
        """Pick the loop's shape: split across lanes, unrolled, or rolled."""
        plan = self._split_plan(s, start, step, end, trips)
        blocked = self._blocked(s, start, step, end, trips) if plan is None else None
        dealt = self._dealt(s, start, step, end, trips) if plan is None and blocked is None else 0
        if plan is not None:
            self._split_reduction(s, start, step, end, trips[0], plan)
        elif blocked is not None:
            self._lower_blocked(blocked)
        elif dealt:
            self._deal(s, start, step, end, trips[0], dealt)
        elif trips is not None and (_indexes_fragments(s) or trips[1] * size <= UNROLL_BUDGET
                                    and (s.unroll or trips[1] <= self._auto_unroll(s))):
            for k in range(trips[1]):
                v = self.b.add(start, self.b.mul(step, self.b.int_const(k)))
                self._guarded_iteration(s.axis.name, v, s.body, end, guard=k >= trips[0])
        else:
            partly = step.const is not None and not self.plain
            factor = _unroll_factor(size, self.src.launch.grid <= TARGET_SMS, trips) if partly else 1
            self._rolled(s, start, step, end, trips, factor)

    def _auto_unroll(self, s) -> int:
        """Trip counts up to which loop ``s`` unrolls completely without being
        marked ``unroll``: more for a loop over a shared-memory tile (a matmul's
        inner product), whose loads then all take immediate offsets (measured:
        a 256x256 cp.async matmul 6.8 -> 4.8 us; unrolling every loop of up to
        128 trips instead grew some kernels five times over)."""
        if self.plain:
            return 0
        tile = any(kind(x) == "Load" and isinstance(self.scope.lookup(x.input) or self.shared.get(x.input), SharedArray)
                   for x in walk(s.body))
        return SMEM_UNROLL_TRIPS if tile else AUTO_UNROLL_TRIPS

    def _split_plan(self, s, start: Value, step: Value, end: Value, trips) -> SplitPlan | None:
        """How to split reduction loop ``s`` across the lanes that share it, or None.

        Emmy's tile decode often gives every lane of an aligned group the same
        reduction to compute (all lanes of a warp take the same query rows in
        attention, and each computes the same dot products). When the folded
        values are uniform over groups of G lanes (``_fold``), those G lanes can
        split the iterations and combine their partial results with log2(G)
        shuffles: the same reduction, regrouped, which an associative and
        commutative combine allows. The loop qualifies when every lane of the
        warp runs it and its trip count is known. A body without inner loops
        gives each lane one contiguous chunk of the iterations (their loads
        merge into vector loads).

        A reduction nested in another can be split at either level. Splitting
        the outer loop leaves each lane whole inner loops and one combine at the
        end; splitting the inner one combines once per outer iteration. Which is
        cheaper is mostly a question of coalescing: the lanes of a split loop
        read the addresses of consecutive iterations, one cache line for a loop
        along a row but a line per lane for a loop down a column. So ``s`` is
        split only when that costs fewer line accesses and shuffles than
        splitting the loops in it (the coalescing-driven choice of the loop to
        parallelize, as in PPCG and Yang et al., PLDI'10).
        """
        if self.plain or self.strict or not self.converged or trips is None or trips[0] != trips[1]:
            return None  # (strict: a split regroups the reduction)
        if trips[0] < SPLIT_MIN_TRIPS:
            return None
        fold = self._fold(s, start, step, end)
        n = trips[0]
        group = 0 if fold is None else min(fold.group, 1 << (n.bit_length() - 1))
        if group < 2:
            return None
        nested = self._nested_split_cost(s, n, group)
        deltas = self._deltas(s.body, s.axis.name, step.const)
        if nested is None and group == WARP and n >= 2 * self.cell_threads > 2 * WARP:
            group = self.cell_threads  # a cell spanning the warps of its block: all of its threads split it
        if nested is None and n <= group * CHUNK_TRIPS:  # a short straight-line body: one contiguous chunk per lane
            return SplitPlan(fold, group, n // group if n % group == 0 else 1, 1)
        if nested is None:  # a long one: lanes side by side, a vector each (coalesced)
            return SplitPlan(fold, group, _split_vector(deltas, n, group, whole=False), 1)
        if nested < n / group * _line_cost(deltas, group) + math.log2(group) * _merge_cost(fold):
            return None
        vector = _split_vector(deltas, n, group)
        return SplitPlan(fold, group, vector, self._jam_factor(s, fold, n, group, vector))

    def _fold(self, s, start: Value, step: Value, end: Value) -> Fold | None:
        """Loop ``s``'s reduction as lanes can share it, or None when it is not one.

        ``s`` must seed its accumulators and hold only values, folds and loops
        of them (mapping.is_pure_loop). Each top-level accumulation must be an
        associative, commutative fold (SPLITTABLE) or part of an online softmax
        (mapping.online_groups), and nothing else may read the accumulators
        mid-fold. ``group`` is how many aligned lanes see the same values in
        every fold. Each of ``merges`` combines a partner lane's partial state
        into one independent part of the fold: a plain fold's own combine, and
        for an online softmax the combine derived from its update
        (mapping.combine_program).
        """
        if not s.seed or not mapping.is_pure_loop(s):
            return None
        groups = mapping.online_groups(s.body)
        accs = [st for st in s.body if kind(st) == "Accum"]
        if groups is None or not accs or any(st.dtype is not None and st.dtype.name != "f32" for st in accs):
            return None
        in_group = {id(x) for g in groups for x in g.stmts}
        plain = {st.name: st for st in accs if id(st) not in in_group}
        if any(st.base is not None or REDUCE_CANON.get(st.op.name) not in SPLITTABLE for st in plain.values()):
            return None
        states = tuple(dict.fromkeys(st.name for st in accs))
        if assigned(s.body) - mapping.local_names(s.body) != set(states):
            return None  # a nested fold updates some outer name: it would need a merge of its own
        hidden = set(states) | {x.name for g in groups for x in g.stmts if kind(x) == "Assign"}
        for st in walk(s.body):  # nothing but the folds may see the running state
            own = {st.name} if kind(st) == "Accum" and id(st) not in in_group else set()
            if id(st) not in in_group and mapping.own_reads(st) & hidden - own:
                return None
        loop_ug = min(start.ug, step.ug, end.ug)
        of_name = self._uniformity(s, loop_ug)
        folded = [st.value for st in plain.values()] + [x for g in groups for x in (g.s, *(w for _, _, w in g.sums))]
        group = min([loop_ug] + [of_name(x) for x in folded])
        merges = [((st.name,), (Assign(st.name, REDUCE_CANON[st.op.name], (st.name, f"{st.name}$b"), st.dtype),))
                  for st in plain.values()]
        for g in groups:
            merges.append(((g.m, *(l for l, _, _ in g.sums)),
                           mapping.combine_program(g, lambda n: f"{n}$b", lambda n: f"{n}$c")))
        return Fold(group, states, tuple(merges))

    def _uniformity(self, s, loop_ug: int):
        """``ug(name)`` for the values one iteration of loop ``s`` computes: how
        many aligned lanes compute each alike when the loop variable is uniform
        over ``loop_ug`` lanes (a name bound outside the loop keeps its own)."""
        env: dict[str, int] = {s.axis.name: loop_ug}

        def of_name(n: str) -> int:
            if n in env:
                return env[n]
            bound = self.scope.lookup(n)
            return bound.ug if isinstance(bound, (Value, Variable)) else 1

        def of_expr(e) -> int:
            return min([of_name(v) for v in e.free_vars()] + [WARP])

        def visit(stmts) -> None:
            for st in stmts:
                k = kind(st)
                if k == "Load":
                    for n in st.names:
                        env[n] = min([of_expr(e) for e in st.index] + [WARP])
                elif k == "Assign":
                    env[st.name] = min([of_name(a) for a in st.args] + [WARP])
                elif k == "Let":
                    env[st.name] = of_expr(st.value)
                elif k == "Select":
                    env[st.name] = min([min(of_name(br.value), of_expr(br.select)) for br in st.branches] + [WARP])
                elif k == "Accum":
                    folded = [st.value] + ([st.base] if st.base is not None else [])
                    env[st.name] = min([env.get(st.name, WARP)] + [of_name(x) for x in folded])
                elif k in LOOPS:  # a nested loop: every lane runs it whole
                    env[st.axis.name] = WARP if k == "Loop" else min([of_expr(e) for e in st.exprs()] + [WARP])
                    if k == "StridedLoop" and st.end is not None:
                        env[f"{st.axis.name}_end"] = of_expr(st.end)
                    visit(st.body)
                else:  # a branch's bodies (what else there is binds nothing)
                    for child in st.nested():
                        visit(child)

        visit(s.body)
        return of_name

    def _blocked(self, s, start: Value, step: Value, end: Value, trips) -> mapping.Blocked | None:
        """Loop ``s`` dealt to the lanes that share part of each iteration, in
        blocks (mapping.blocked), or None.

        In attention every lane holding the same query rows computes the same
        scores, but folds them into its own output columns, so the loop is no
        lane-shared reduction (``_split_plan``). Instead the G lanes of each
        group take the block's iterations in turn for the scores and broadcast
        the results; G is the largest group (a power of two, at most the trip
        count) for which the shared part still holds an inner loop.

        Slots taking consecutive iterations read the shared inputs the
        iteration stride apart. With the keys stored row by row (k[t, d]
        rather than kt[d, t]), one lane per key would read a cache line per
        lane, so a slot then has several lanes (``_key_lanes``), which split
        the key's dot product and read its row coalesced."""
        if self.plain or not self.converged or trips is None or trips[0] != trips[1]:
            return None
        if None in (start.const, step.const) or not s.seed or not mapping.is_pure_loop(s):
            return None
        accs = [st for st in s.body if kind(st) == "Accum"]
        if not accs or any(st.dtype is not None and st.dtype.name != "f32" for st in accs):
            return None
        if assigned(s.body) - mapping.local_names(s.body) != {st.name for st in accs}:
            return None  # a nested fold updates some outer name
        if self.strict and mapping.online_groups(s.body) != []:
            return None  # an online softmax updated once per block of keys rounds differently
        n = trips[0]
        of_name = self._uniformity(s, min(start.ug, step.ug, end.ug))
        group = min(WARP, 1 << (n.bit_length() - 1))
        first, step_ = int(start.const), int(step.const)
        while group >= 2:
            uniform = (lambda g: lambda x: of_name(x) >= g)(group)
            plan = mapping.blocked(s, n, first, step_, group, 1, uniform)
            if plan is not None:
                share = 1 if self.strict else self._key_lanes(plan, s.axis.name, step_, group)  # (lanes split a dot)
                return mapping.blocked(s, n, first, step_, group, share, uniform) if share > 1 else plan
            group //= 2
        return None

    def _key_lanes(self, plan: mapping.Blocked, var: str, step: int, group: int) -> int:
        """Lanes per slot of a blocked loop. One while slots taking consecutive
        iterations read the shared part's inputs coalesced. When each
        iteration's inputs are a row of their own (k[t, d]), more lanes per
        slot read a row in fewer lines but merge their partial dot products
        with shuffles and leave fewer iterations per block to share the
        softmax update: a lane per 64 bytes of row was best measured (twice
        that on a full SM, where L1 bandwidth is what runs out), up to
        KEY_LANES."""
        rows = [abs(d) for _, d, _ in self._deltas(plan.shared, var, step) if d != 0]
        if all(d is not None and d < LINE_BYTES for d in rows):
            return 1
        inner = min((mapping.trip_estimate(st) for st in plan.shared if kind(st) in LOOPS), default=0)
        if inner < SPLIT_MIN_TRIPS:
            return 1
        full = self.src.launch.block // WARP * -(-self.src.launch.grid // TARGET_SMS) >= WARP  # warps per SM
        lanes = max((d for d in rows if d is not None), default=LINE_BYTES) // ROW_BYTES_PER_LANE * (2 if full else 1)
        return max(1, min(KEY_LANES, group // 2, 1 << (max(lanes, 1).bit_length() - 1)))

    def _dealt(self, s, start: Value, step: Value, end: Value, trips) -> int:
        """Lanes the iterations of loop ``s`` may be dealt to (0: none).

        A loop that carries nothing from one iteration to the next and whose
        iterations store to different places can run its iterations in any
        order, so when the lanes of an aligned group would all compute the
        same iteration alike (the lanes of a spread cell, ``_spread``), lane r
        can take iterations r, r + G, ... instead: the same stores, G times
        fewer each. Each buffer is stored by one statement whose index moves
        with the loop variable, and the loop reads no buffer it stores."""
        if self.plain or not self.converged or trips is None or trips[0] != trips[1] or trips[0] < 2:
            return 0
        if self.store_guard is not None:
            return 0  # a lane that stores nothing must not take iterations of its own
        if any(kind(x) in mapping.CROSS_THREAD or kind(x) == "Init" for x in walk(s.body)):
            return 0
        if assigned(s.body) - mapping.local_names(s.body):
            return 0  # a fold carried across the iterations
        writes = [x for x in walk(s.body) if kind(x) == "Write"]
        outputs = [w.output for w in writes]
        if not writes or len(set(outputs)) < len(outputs) or any(w.atomic for w in writes):
            return 0
        if {x.input for x in walk(s.body) if kind(x) == "Load"} & set(outputs):
            return 0
        lets = mapping.index_lets(s.body)
        if any(not mapping.flat_coefficient(w.index, self.buffer(w.output).shape, s.axis.name, lets) for w in writes):
            return 0  # an iteration's store does not move with it (or not provably)
        of_name = self._uniformity(s, min(start.ug, step.ug, end.ug))
        names = (mapping.binds(s.body) | reads(s.body)) - {st.axis.name for st in walk(s.body) if kind(st) in LOOPS}
        group = min([WARP] + [of_name(n) for n in names])
        if group == WARP and self.cell_threads > WARP:
            group = self.cell_threads  # a cell spanning its block's warps
        return group if group >= 2 else 0

    def _deal(self, s, start: Value, step: Value, end: Value, trips: int, group: int) -> None:
        """Loop ``s`` with lane r of each aligned group of ``group`` running
        iterations r, r + G, ... (``_dealt``): a loop over the rounds, and the
        last, partial round only on the lanes it reaches."""
        b, t = self.b, s.axis.name
        rounds, rest = divmod(trips, group)

        def at(index):  # Let t = start + (index * group + rank) * step
            i = BinaryExpr("+", BinaryExpr("*", index, _int_lit(group)), Var(f"{t}$rank"))
            return Let(name=t, value=BinaryExpr("+", Var(f"{t}$start"), BinaryExpr("*", i, Var(f"{t}$step"))))

        slot = LaneSlot(f"{t}$rank", group, 1)
        with self.block(dominated=True):
            self.scope.names[f"{t}$start"], self.scope.names[f"{t}$step"] = start, step
            if rounds:
                loop = Loop(axis=Axis(f"{t}$round", Dim(rounds)), body=(slot, at(Var(f"{t}$round")), *s.body),
                            unroll=s.unroll, seed=False)
                with self.block(dominated=True):
                    self._loop_body(loop, b.int_const(0), b.int_const(1), b.int_const(rounds), (rounds, rounds),
                                    _body_size(loop.body))
            if rest:
                self.body((slot, Cond(cond=BinaryExpr("<", Var(f"{t}$rank"), _int_lit(rest)),
                                      body=(at(_int_lit(rounds)), *s.body))))

    def _lower_blocked(self, plan: mapping.Blocked) -> None:
        """The blocks of a lane-blocked loop, then the lanes' partial sums merged.

        A lane reads whole query rows for its own iteration of each block, the
        same values in every block. Unrolled blocks keep them in registers
        (value numbering shares them), which pays while they take at most half
        the registers a thread of this launch may have (measured); beyond that
        the blocks run as a rolled loop that leaves those loads in place
        (ptxas still keeps what fits)."""
        self.mutable |= plan.mutable
        self.shared_loop = True
        self._lane_region = self._lane_rows(plan)
        loop = plan.loop
        try:
            if loop is not None:
                b = self.b
                n, size = int(loop.axis.extent.as_static()), _body_size(loop.body)
                registers = min(THREAD_REGISTERS, REGISTER_FILE // self.src.launch.block)
                unrolls = (loop.unroll or n <= AUTO_UNROLL_TRIPS) and n * size <= UNROLL_BUDGET
                if unrolls and plan.invariant <= registers // 2:
                    with self.block(dominated=True):
                        self._loop_body(loop, b.int_const(0), b.int_const(1), b.int_const(n), (n, n), size)
                else:
                    self._rolled(loop, b.int_const(0), b.int_const(1), b.int_const(n), (n, n), 1, hoist_loads=False)
            with self.block(dominated=True):
                self.body(plan.rest)
        finally:
            self._lane_region, self._stashed = None, {}
        for merge in plan.merges:
            self.stmt_LaneButterfly(merge)

    def _lane_rows(self, plan: mapping.Blocked):
        """``(byte offset, rows per warp)`` of a part of the lane buffer for the
        broadcasts of ``plan`` (a row of 32 words per broadcast value), or None
        when a block broadcasts too little to pay for the barriers (fewer than
        LANE_BUFFER_MIN values per lane, measured), or the buffer does not fit
        next to the static shared arrays (or the kernel has a dynamic pool,
        which its launch sizes exactly)."""
        stmts = ((plan.loop,) if plan.loop is not None else ()) + plan.rest
        rows = max((len(st.values) for st in walk(stmts) if kind(st) == "LaneStash"), default=0)
        one_block = (plan.loop,) if plan.loop is not None else plan.rest
        per_block = sum(1 for st in walk(one_block) if kind(st) == "LaneBroadcast")
        size = -(-self.src.launch.block // WARP) * rows * WARP * 4
        if per_block < LANE_BUFFER_MIN or self._static_smem is None:
            return None
        if self._static_smem + self._lane_bytes + size > STATIC_SMEM_LIMIT:
            return None
        offset = self._lane_bytes
        self._lane_bytes += size
        return offset, rows

    def _deltas(self, stmts, var: str, step: int) -> list:
        """``(runs per iteration, bytes between the addresses of consecutive
        iterations of ``var`` or None, element bytes)`` for every load in ``stmts``."""
        lets = mapping.index_lets(stmts)
        out = []
        for ld, runs in mapping.weighted_loads(stmts):
            buf = self.buffer(ld.input)
            size = SIZEOF.get(MEM_TYPE[ld.dtype.name] if ld.dtype is not None else buf.type, 1)
            c = mapping.flat_coefficient(ld.index, buf.shape, var, lets)
            out.append((runs, None if c is None else c * step * size, size))
        return out

    def _nested_split_cost(self, s, trips: int, group: int):
        """``_split_plan``'s cost of leaving ``s`` whole and splitting the
        reductions directly inside it instead (None when it has none)."""
        inner = [st for st in s.body if kind(st) in LOOPS]
        if not inner:
            return None
        cost = trips * sum(1 for st in s.body if kind(st) == "Load")  # every lane, the same line
        for st in inner:
            runs = mapping.trip_estimate(st)
            step = st.step.value if kind(st) == "StridedLoop" and kind(st.step) == "Literal" else 1
            deltas = self._deltas(st.body, st.axis.name, int(step))
            accs = len({x.name for x in st.body if kind(x) == "Accum"})
            if st.seed and accs and mapping.is_pure_loop(st):
                cost += trips * (runs / group * _line_cost(deltas, group) + math.log2(group) * 2 * accs)
            else:  # each lane runs it whole
                cost += trips * runs * sum(r for r, _, _ in deltas)
        return cost

    def _jam_factor(self, s, fold: Fold, trips: int, group: int, vector: int) -> int:
        """Consecutive iterations a split lane lowers as one body (unroll-and-jam):
        worth it when the body has inner loops, which fusing the copies merges
        (one load of what they share, independent chains, vector loads across
        the copies)."""
        inner = [st for st in s.body if kind(st) in LOOPS]
        if not inner or trips % (vector * group):
            return 1
        if any(self.scope.lookup(n) is not None for n in mapping.binds(s.body) - set(fold.states)):
            return 1  # a body name also bound outside: renaming the copies apart could capture it
        live = len(assigned(inner))
        jam = vector if vector > 1 else next((j for j in (4, 2) if (trips // group) % j == 0), 1)
        while jam > 1 and jam * live > JAM_ACCUMULATORS:
            jam //= 2
        return jam

    def _split_reduction(self, s, start: Value, step: Value, end: Value, trips: int, plan: SplitPlan) -> None:
        """Lane rank r of each aligned group folds its share of the iterations;
        a butterfly of shuffles then merges the partial states (``_merge_lanes``),
        leaving every lane of the group the whole reduction.

        When the trips divide evenly, a lane takes ``plan.vector`` consecutive
        iterations at a time, t = V*r + V*G*k + j (j < V): with V = trips/G
        one contiguous chunk per lane, with a smaller V rounds in which the
        warp's loads cover whole vectors side by side (blocked-cyclic). Each
        ``plan.jam`` consecutive iterations of a lane are lowered as one body,
        so their inner loops fuse (unroll-and-jam). Otherwise lanes take every
        G-th iteration, the last round only on the lanes it reaches. More than
        SPLIT_UNROLLED_ROUNDS rounds become a loop of their own (``_split_rounds``).
        """
        b = self.b
        group, vector = plan.group, plan.vector
        self.shared_loop = True
        rank = self._rank(group)
        rounds = trips // (vector * group)
        if rounds > SPLIT_UNROLLED_ROUNDS:
            self._split_rounds(s, start, step, end, trips, plan, rank)
        elif trips % (vector * group) == 0:
            first = b.mul(rank, b.int_const(vector))
            offsets = [vector * group * k + j for k in range(trips // (vector * group)) for j in range(vector)]
            for i in range(0, len(offsets), plan.jam):
                values = [b.add(start, b.mul(b.add(first, b.int_const(o)), step)) for o in offsets[i : i + plan.jam]]
                self._jammed(s, values, end, plan.fold.states)
        else:  # interleaved, the last round only on the lanes it reaches
            rounds = -(-trips // group)
            for j in range(rounds):
                v = b.add(start, b.mul(b.add(rank, b.int_const(group * j)), step))
                self._guarded_iteration(s.axis.name, v, s.body, end, guard=j == rounds - 1)
        self._merge_lanes(plan.fold, group)

    def _split_rounds(self, s, start: Value, step: Value, end: Value, trips: int, plan: SplitPlan, rank: Value) -> None:
        """``_split_reduction``'s rounds as a loop: round k gives lane r the
        ``plan.vector`` iterations from V*r + V*G*k (lowered as one body), and
        the iterations past the last whole round go to the lanes one each."""
        b = self.b
        group, vector, t = plan.group, plan.vector, s.axis.name
        span = vector * group
        rounds, index = trips // span, f"{t}$round"
        first = BinaryExpr("*", BinaryExpr("+", BinaryExpr("*", Var(index), _int_lit(group)), Var(f"{t}$rank")),
                           _int_lit(vector))

        def value(j):  # start + (first + j) * step
            i = BinaryExpr("+", first, _int_lit(j))
            return BinaryExpr("+", Var(f"{t}$start"), BinaryExpr("*", i, Var(f"{t}$step")))

        body = (LaneSlot(f"{t}$rank", group, 1), *(Let(name=f"{t}${j}", value=value(j)) for j in range(vector)),
                *mapping.jammed(s.body, t, vector, set(plan.fold.states)))
        loop = Loop(axis=Axis(index, Dim(rounds)), body=body, seed=False)
        with self.block(dominated=True):
            self.scope.names[f"{t}$start"], self.scope.names[f"{t}$step"] = start, step
            self._loop_body(loop, b.int_const(0), b.int_const(1), b.int_const(rounds), (rounds, rounds),
                            _body_size(loop.body))
        for j in range(-(-(trips - rounds * span) // group)):
            v = b.add(start, b.mul(b.add(rank, b.int_const(rounds * span + group * j)), step))
            self._guarded_iteration(t, v, s.body, end, guard=rounds * span + group * (j + 1) > trips)

    def _jammed(self, s, values: list[Value], end: Value, states) -> None:
        """Iterations ``values`` of loop ``s``, lowered as one body (mapping.jammed)."""
        if len(values) == 1:
            self._iteration(s.axis.name, values[0], s.body, end)
            return
        with self.block(dominated=True):
            for j, v in enumerate(values):
                self.scope.names[f"{s.axis.name}${j}"] = self.b.assume(v, v.lo, end.hi - 1) if end.hi is not None else v
            self.body(mapping.jammed(s.body, s.axis.name, len(values), set(states)))

    def _merge_lanes(self, fold: Fold, group: int) -> None:
        """A butterfly over each aligned group of ``group`` lanes per independent
        merge, one after another (which measured better than stepping them
        together): each step swaps the merge's states with the lane ``width``
        away and combines the pair with its program. A group wider than a warp
        (a cell spanning its block) then merges the warps' totals
        (``_merge_warps``)."""
        b = self.b
        mask = self.kernel.reg("s32")
        b.emit("activemask.b32", mask, ndefs=1, effect=False)
        for states, program in fold.merges:
            width = min(group, WARP) // 2
            while width:
                with self.block():
                    for name in states:
                        var = self.lookup(name)
                        other = self.kernel.reg(var.type)
                        b.emit("shfl.sync.bfly.b32", other, b.in_reg(var.read()), Imm(width, "u32"), Imm(31, "u32"),
                               mask, ndefs=1, effect=False)
                        self.define(f"{name}$b", Value(var.type, other))
                    self.merge(program, states, "f32")
                width //= 2
        if group > WARP:
            self._merge_warps(fold, group // WARP)
        for name in fold.states:
            self.lookup(name).ug = min(group, WARP)  # every lane of the group now holds the whole fold

    def _merge_warps(self, fold: Fold, warps: int) -> None:
        """The second stage of a fold split over the warps of a block (a cell
        spanning them, ``_spread``): each warp's total, which the butterfly left
        in all its lanes, goes to shared memory, and after one barrier every
        thread combines the ``warps`` totals pairwise as a TreeHalve would, so
        all threads hold the same result. A barrier before the stores lets an
        enclosing loop reuse the slots."""
        b = self.b
        states = fold.states
        offset = self._lane_bytes
        self._lane_bytes += warps * len(states) * 4
        base = b.add(b.pure("mov.u32", "s32", (Sym(LANE_BUFFER),), 0, 1 << 24), b.int_const(offset))
        mine = b.in_reg(b.add(base, b.mul(self.value("warp"), b.int_const(len(states) * 4)))).op
        b.emit("bar.sync", Imm(0, "u32"))
        for c, name in enumerate(states):
            b.emit("st.shared.f32", Mem(mine, c * 4), b.in_reg(self.lookup(name).read()))
        b.emit("bar.sync", Imm(0, "u32"))
        totals = b.in_reg(base).op
        for part, program in fold.merges:
            level = []
            for w in range(warps):
                row = []
                for name in part:
                    out = self.kernel.reg("f32")
                    addr = Mem(totals, (w * len(states) + states.index(name)) * 4)
                    b.emit("ld.shared.f32", out, addr, ndefs=1, effect=False)
                    row.append(Value("f32", out))
                level.append(row)
            half = warps // 2
            while half:
                for i in range(half):
                    with self.block():
                        for name, v in zip(part, level[i]):
                            self.define(name, v)
                        for name, v in zip(part, level[i + half]):
                            self.define(f"{name}$b", v)
                        self.merge(program, part, "f32")
                        level[i] = [self.value(name) for name in part]
                half //= 2
            for name, v in zip(part, level[0]):
                self.reassign(name, v)

    def _rolled(self, s, start: Value, step: Value, end: Value, trips, factor: int, hoist_loads: bool = True) -> None:
        try:
            self._rolled_loops(s, start, step, end, trips, factor, hoist_loads)
        finally:
            self._pessimize(s.body)  # the exit merges every trip count's final values

    def _rolled_loops(self, s, start: Value, step: Value, end: Value, trips, factor: int, hoist_loads: bool) -> None:
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
                with b.loop(hoist_loads):
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
        with b.loop(hoist_loads):
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
            vtype = elementwise.value_type(s.dtype.name)
            v = self.b.convert(self.expr(s.value, "f64" if vtype == "f64" else "f32"), vtype)
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
                and all(a.key() is not None for a in args)  # a variable read only with its version
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
        if canon == "add" and acc.type == "f32" and not self.strict:  # contract a product into the fold: one FMA
            for product, addend in ((value, base), (base, value)):
                factors = self.products.get(product.key()) if product.key() is not None else None
                # a factor read from a variable must still hold (alpha * l fuses into l's own update)
                if factors is not None and addend.type == "f32" and all(b.current(f) for f in factors):
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
            if vtype in INT_RANGE:
                const = self.b.int_const(int(value), vtype)
            else:
                const = self.b.convert(self.b.float_const(value, vtype), vtype)
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
        guard = self.store_guard
        if s.atomic:
            if buf.type not in ATOMIC_ADD:
                raise LoweringError(f"atomic add on {buf.type} is not supported")
            addr = self.address(buf, flat=base, swizzle=s.swizzle)
            b.emit(f"red.{space}.add.{ATOMIC_ADD[buf.type]}", addr, values[0], guard=guard)
            return
        k = 0
        while k < len(values):
            n = _vector_width(len(values) - k, buf.type)
            addr = self.address(buf, flat=base.plus(Affine(const=k)), swizzle=s.swizzle)
            if n == 1:
                b.emit(f"st.{space}.{_ldst(buf.type)}", addr, values[k], guard=guard)
            else:
                b.emit(f"st.{space}.v{n}.{_ldst(buf.type)}", addr, Vec(tuple(v.op for v in values[k : k + n])),
                       guard=guard)
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
        self._butterfly(s.state, s.state_b, s.combine_states, int(s.length), 1, elementwise.value_type(s.dtype.name))

    def stmt_LaneButterfly(self, s) -> None:
        self._butterfly(s.state, s.state_b, s.combine_states, s.group, s.low, "f32")

    def _butterfly(self, state, state_b, program, length: int, low: int, vtype: str) -> None:
        """For lane distances ``length/2`` down to ``low``: swap the state with
        the lane that far away and combine the pair with ``program``."""
        b = self.b
        mask = self.kernel.reg("s32")
        b.emit("activemask.b32", mask, ndefs=1, effect=False)
        lane_mask = length // 2
        while lane_mask >= low:
            with self.block():
                for st, sb in zip(state, state_b):
                    cur = b.in_reg(b.convert(self.value(st), vtype))
                    self.define(sb, self._shuffle_xor(cur, lane_mask, mask))
                self.merge(program, state, vtype)
            lane_mask //= 2
        for st in state:  # the butterfly leaves each aligned group of `length` lanes agreeing
            var = self.scope.lookup(st)
            if isinstance(var, Variable):
                var.ug = max(var.ug, min(WARP, length))

    def _shuffle_xor(self, v: Value, lane_mask: int, mask) -> Value:
        """``v`` of the lane ``lane_mask`` away (``shfl.sync.bfly``, which moves
        32 bits): a 64-bit value moves as its two halves, a 16-bit one as f32."""
        b = self.b
        size = SIZEOF.get(v.type, 4)
        if size == 2:
            return b.convert(self._shuffle_xor(b.convert(v, "f32"), lane_mask, mask), v.type)
        out = self.kernel.reg(v.type)
        if size == 4:
            b.emit("shfl.sync.bfly.b32", out, v, Imm(lane_mask, "u32"), Imm(31, "u32"), mask, ndefs=1, effect=False)
            return Value(v.type, out)
        halves = tuple(self.kernel.reg("s32") for _ in range(2))
        moved = tuple(self.kernel.reg("s32") for _ in range(2))
        b.emit("mov.b64", Vec(halves), v, ndefs=1, effect=False)
        for src, dst in zip(halves, moved):
            b.emit("shfl.sync.bfly.b32", dst, src, Imm(lane_mask, "u32"), Imm(31, "u32"), mask, ndefs=1, effect=False)
        b.emit("mov.b64", out, Vec(moved), ndefs=1, effect=False)
        return Value(v.type, out)

    def stmt_LaneSlot(self, s) -> None:
        """``(lane % group) / share``, alike on the ``share`` lanes of a slot."""
        b = self.b
        rank = self._rank(s.group)
        slot = b.shr(rank, b.int_const(s.share.bit_length() - 1)) if s.share > 1 else rank
        self.define(s.name, b.uniform(slot, min(s.share, WARP)))

    def _rank(self, group: int) -> Value:
        """This thread's rank in its aligned group of ``group`` threads (a group
        wider than a warp: a cell spanning its block's warps)."""
        b = self.b
        if group > WARP:
            return b.bitop("&", self.tid, b.int_const(group - 1))
        lane = self.value("lane")
        return lane if group == WARP else b.bitop("&", lane, b.int_const(group - 1))

    def stmt_LaneStash(self, s) -> None:
        """Store ``values`` for the LaneBroadcasts that follow, when the blocked
        loop has a part of the lane buffer (``_lower_blocked``): one word per
        slot of this lane's group, in the warp's own rows, between warp
        barriers (the warp's reads of the previous block must be done before
        the writes, and the writes before the reads)."""
        region = self._lane_region
        self._stashed = {}
        if region is None:
            return  # the broadcasts shuffle
        b = self.b
        offset, rows = region
        mask = b.pure("activemask.b32", "s32", ())
        lane, warp = self.value("lane"), self.value("warp")
        buffer = b.pure("mov.u32", "s32", (Sym(LANE_BUFFER),), 0, 1 << 24)
        base = b.add(buffer, b.add(b.int_const(offset), b.mul(warp, b.int_const(rows * WARP * 4))))
        # the group's slot 0, then this lane's slot
        first = b.add(base, b.mul(b.bitop("&", lane, b.int_const(WARP - s.group)), b.int_const(4)))
        slot = b.bitop("&", lane, b.int_const(s.group - 1))
        if s.share > 1:
            slot = b.shr(slot, b.int_const(s.share.bit_length() - 1))
        mine = b.in_reg(b.add(first, b.mul(slot, b.int_const(4)))).op
        b.emit("bar.warp.sync", mask)
        for row, name in enumerate(s.values):
            b.emit("st.shared.f32", Mem(mine, row * WARP * 4), b.in_reg(b.convert(self.value(name), "f32")))
        b.emit("bar.warp.sync", mask)
        self._stashed = {name: (first, row, s.share) for row, name in enumerate(s.values)}

    def stmt_LaneBroadcast(self, s) -> None:
        """``name`` = ``value`` of lane ``lane`` in this lane's aligned group of
        ``group``: a load of its stash (stmt_LaneStash), else ``shfl.sync.idx``
        with the group as the shuffle segment. A shared load costs far less
        (measured: 32 broadcasts, 4 us against 68 us of shuffles)."""
        b = self.b
        stashed = self._stashed.get(s.value)
        if stashed is not None:  # one shared load (ptxas merges neighbouring slots into vector loads)
            first, row, share = stashed
            out = self.kernel.reg("f32")
            addr = Mem(b.in_reg(first).op, row * WARP * 4 + s.lane // share * 4)
            b.emit("ld.shared.f32", out, addr, ndefs=1, effect=False)
            self.define(s.name, b.uniform(Value("f32", out), s.group))
            return
        v = self.value(s.value)
        mask = b.pure("activemask.b32", "s32", ())  # the code runs converged: one mask serves every shuffle
        out = self.kernel.reg(v.type)
        segment = ((WARP - s.group) << 8) | (WARP - 1)
        b.emit("shfl.sync.idx.b32", out, b.in_reg(v), Imm(s.lane, "u32"), Imm(segment, "u32"), mask,
               ndefs=1, effect=False)
        self.define(s.name, b.uniform(Value(v.type, out), s.group))

    def stmt_TreeHalve(self, s) -> None:
        """emmy's halving tree over the partials in shared memory, computed by
        every thread on its own. After the barrier that precedes the tree (the
        partials are written), each thread loads the ``length`` partials of its
        tree and combines them pairwise exactly as the tree would (slot i with
        slot i + h, for h = length/2 .. 1), so it holds the root's value without
        the tree's barrier per step, its branches and its stores. One barrier at
        the end keeps the ordering the tree's last barrier gave the code after
        it; the partials themselves are not read again (the root is the tree's
        only result)."""
        b = self.b
        vtype = elementwise.value_type(s.dtype.name)
        bufs = [self.buffer(name) for name in s.bufs]
        length = int(s.length)
        if s.inner is not None:
            iv, scale = self.value(s.inner[0]), int(s.inner[1])
            slots = [b.add(b.int_const(k * scale), iv) for k in range(length)]
        else:
            slots = [b.int_const(k) for k in range(length)]
        level = [[self.load_one(buf, slot, vtype) for buf in bufs] for slot in slots]
        half = length // 2
        while half:
            for i in range(half):
                with self.block():  # the tree's temporaries shadow the outer state
                    for name, v in zip((*s.state, *s.state_b), (*level[i], *level[i + half])):
                        self.define(name, v)
                    self.merge(s.combine_states, s.state, vtype)
                    level[i] = [self.value(st) for st in s.state]
            half //= 2
        if s.barrier_id == 0:
            b.emit("bar.sync", Imm(0, "u32"))
        else:
            b.emit("bar.sync", Imm(s.barrier_id, "u32"), Imm(int(s.barrier_count), "u32"))
        for st, v in zip(s.state, level[0]):
            self.reassign(st, v)

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


def _thread_free(body) -> bool:
    """Does what the statements compute not depend on which thread runs them?
    No cross-thread statement, shared memory or thread index."""
    for s in walk(body):
        k = kind(s)
        if k in mapping.CROSS_THREAD or k in ("Smem", "ZeroPrologue"):
            return False
        if any(_names_thread(e) for e in getattr(s, "exprs", lambda: ())()):
            return False
    return not reads(body) & {"lane", "warp"}


def _names_thread(e) -> bool:
    """Does expression ``e`` read a thread or block index (a Builtin)?"""
    if kind(e) == "Builtin":
        return True
    return any(_names_thread(c) for c in vars(e).values() if hasattr(c, "free_vars")) or any(
        _names_thread(c) for v in vars(e).values() if isinstance(v, tuple) for c in v if hasattr(c, "free_vars"))


def _reassigned_names(body) -> set[str]:
    """Names that some statement reassigns after declaring them."""
    out = set()
    for s in walk(body):
        kind = type(s).__name__
        if kind in ("Accum", "Init"):
            out.add(s.name)
        elif kind in ("WarpShuffle", "TreeHalve"):
            out.update(s.state)
    return out


def _indexes_fragments(loop) -> bool:
    """Does a fragment reference in ``loop`` take the loop variable as its index
    (``_c0_[_r0]``)? Such a loop is unrolled, whatever its size: a register
    array has no runtime index."""
    return any(fragment_index(r) == loop.axis.name for x in walk(loop.body) for r in fragment_refs(x))


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
            needs, defines = {v for e in s.index for v in e.free_vars()}, set(s.names)
            pos = len(out)
            while pos > 0 and type(out[pos - 1]).__name__ != "Load":
                prev = out[pos - 1]
                if produced(prev) & (needs | defines) or reads((prev,)) & defines:
                    break  # a true, anti or output dependence
                pos -= 1
            out.insert(pos, s)
        else:
            out.append(s)
    return tuple(out)


def _int_lit(v: int):
    return Literal(v, "int")


def _int_value(e):
    """The value of an integer literal expression, else None."""
    if type(e).__name__ == "Literal" and not isinstance(e.value, float):
        return int(e.value)
    return None


def _lowbit(n: int) -> int:
    """The largest power of two dividing ``n``."""
    return n & -n


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _decompositions(n: int):
    """Aligned lane groups for n cells (powers of two up to a warp), most
    sharing first: past the whole warps, the rest in one padded group; the
    largest power of two below it and one padded group; exactly n lanes."""
    whole, rest = (WARP,) * (n // WARP), n % WARP
    if not rest:
        yield whole
        return
    yield whole + (_pow2_ceil(rest),)
    big = 1 << (rest.bit_length() - 1)
    if big < rest:
        two = (big, _pow2_ceil(rest - big))
        yield whole + two
        exact = tuple(1 << i for i in reversed(range(rest.bit_length())) if rest >> i & 1)
        if exact != two:
            yield whole + exact


def _lines(group: int, stride) -> float:
    """Cache lines one warp load touches when consecutive lanes read ``stride``
    bytes apart (None: unknown, so assume a line per lane)."""
    if stride is None:
        return group
    if stride == 0:
        return 1
    return min(group, -(-group * abs(stride) // LINE_BYTES))


def _line_cost(deltas, group: int) -> float:
    """Line accesses per iteration of a loop split over ``group`` lanes that
    take consecutive iterations (``deltas`` as from Lowerer._deltas)."""
    return sum(runs * _lines(group, stride) for runs, stride, _ in deltas)


def _merge_cost(fold: Fold) -> int:
    """Instructions in one butterfly step of ``fold``'s lane merges (an exp counts as eight)."""
    return len(fold.states) + sum(8 if a.op.name == "exp" else 1 for _, program in fold.merges for a in program)


def _split_vector(deltas, trips: int, group: int, whole: bool = True) -> int:
    """Consecutive iterations a split lane takes at a time: 4 or 2 when that turns
    a contiguous f32 load into one vector load per lane without spreading another
    load over more lines, else 1 (the lanes interleave single iterations).
    ``whole``: the rounds of ``group`` such vectors must cover the trips exactly."""
    moving = [(stride, size) for _, stride, size in deltas if stride != 0]
    if not any(stride == size == 4 for stride, size in moving):
        return 1
    if any(stride is not None and abs(stride) != size and abs(stride) < LINE_BYTES for stride, size in moving):
        return 1
    return next((v for v in (4, 2) if not whole or trips % (v * group) == 0), 1)


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
