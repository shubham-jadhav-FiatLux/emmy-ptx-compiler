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

The rules are grouped by topic. This module holds the driver, names,
expressions, addresses, loops and the statements' own lowering. Three mixins
of :class:`Lowerer` hold the rest, sharing its state:
:mod:`~emmy_ptx.cells` decides where a Tile's cells run,
:mod:`~emmy_ptx.lanes` how lanes share a loop's work, and
:mod:`~emmy_ptx.tensorcore` lowers the tensor-core statements.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import replace

from . import elementwise, mapping
from .builder import INT_RANGE, WARP, Builder, Value, Variable, eliminate_dead_code
from .cells import CellLayout
from .common import (ATOMIC_ADD, AUTO_UNROLL_TRIPS, C_TYPE, IDENTITY, MEM_TYPE, REDUCE_CANON, SMEM_UNROLL_TRIPS,
                     STATIC_SMEM_LIMIT, TARGET_SMS, UNROLL_BUDGET, VALUE_TYPE, Affine, GlobalBuffer, LoweringError,
                     Scope, SharedArray, body_size, lowbit)
from .congruence import congruent_accumulators, drop_needless_snapshots, merge_congruent
from .frontend import KernelSource
from .lanes import LANE_BUFFER, LaneSharing
from .mapping import LOOPS, assigned, kind, produced, reads, walk
from .ptx import PTX_LIMIT, PTX_VERSION, SIZEOF, TARGET, Imm, Kernel, Mem, Sym, Vec
from .regroup import fully_combined, regroup
from .tensorcore import TensorCoreLowering, fragment_index, fragment_refs
from .vectorize import vectorize_loads


# Alignment (bytes) assumed for every buffer base: cudaMalloc returns 256-byte
# aligned memory and emmy's own float4 accesses rely on 16; shared arrays are
# declared .align 16 or more.
MAX_ALIGN = 16
# Iterations per trip of a rolled loop, by body size (statements).
UNROLL_TINY_BODY, UNROLL_TINY = 6, 8
UNROLL_SMALL_BODY, UNROLL_SMALL = 24, 4
UNROLL_BIG_BODY, UNROLL_BIG_CHOICES = 400, (8, 4, 2)  # tuned with tools/tune.py
# Sibling loops are fused only while the fused body stays this small (statements).
FUSE_BUDGET = 512


class Lowerer(CellLayout, LaneSharing, TensorCoreLowering):
    """Lowers one kernel; ``run()`` returns its :class:`~emmy_ptx.ptx.Kernel`.

    ``plain`` and ``strict`` select the fallback and the bit-exact lowerings
    (see compile_file); ``spread`` and ``place`` are set by ``run`` itself when
    it lowers a kernel again with cells re-mapped."""

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
        if body_size(s.body) + body_size(t.body) > FUSE_BUDGET:
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
        return kind(loop) == "Loop" and unrolls and extent * body_size(loop.body) <= UNROLL_BUDGET


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
        align = min([MAX_ALIGN] + [lowbit(abs(c) * size) for _, c in aff.terms.values()])
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
        size = body_size(s.body)
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


def _int_value(e):
    """The value of an integer literal expression, else None."""
    if type(e).__name__ == "Literal" and not isinstance(e.value, float):
        return int(e.value)
    return None


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
