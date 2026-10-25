"""Tensor cores: emmy's warp-level ``mma.sync`` statements (the m16n8k16 path).

For f16 matmuls and attention, emmy computes 16x8 output tiles on the tensor
cores. A warp holds each operand tile in per-lane registers (``RegFragment``),
lane l owning fixed elements (PTX ISA, "Matrix Fragments for mma.m16n8k16"):
for the f32 C (accumulator) tile, rows g = l/4 and g + 8 and columns 2t and
2t + 1 (t = l % 4); the A (16x16) and B (16x8) tiles pack two 16-bit elements
into each 32-bit register in the same lane map. The statements:

* ``LdmatrixLoad`` fills an A or B fragment from a (swizzled) shared-memory
  slab with ``ldmatrix``, each lane supplying one 16-byte row address (the
  ``.trans`` form turns a row-major K x N slab into the column-major B
  operand), or straight from global memory with the same lane map, clamped at
  a masked tile's edge and zero-filled past a masked K;
* ``MmaSyncPtx`` is one ``mma.sync.aligned.m16n8k16.row.col``, f16 or bf16
  operands, accumulating into the C fragment in place;
* ``FragmentApply`` computes an elementwise op per C element (a softmax's
  scale, subtraction and exp; a mask from the element's coordinates; a bias
  read at them); ``FragmentRowReduce`` folds a lane's two rows across
  fragments and then across the four lanes that share a row (two shuffles);
  ``FragmentRepack`` turns two C fragments into an A operand (``cvt.rn.f16x2``:
  the P of P @ V); ``FragmentPromote`` folds an f16 accumulator into its f32
  shadow;
* ``RegStore`` stores a C fragment lane by lane, each row pair as one
  vector store, through an optional fused epilogue and boundary guards.

Fragment registers are ordinary mutable registers. A fragment indexed by a
loop variable (``_c0_[_r0]``) is resolved once the loop is unrolled, which
``Lowerer._loop_body`` does for such loops whatever their size.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from . import elementwise
from .builder import Value, Variable
from .mapping import kind, walk
from .ptx import Imm, Vec

M16N8K16 = (16, 8, 16)
# element i of a lane's C fragment: row g + 8 * C_ROW[i], column 2t + C_COL[i]
C_ROW, C_COL = (0, 0, 1, 1), (0, 1, 0, 1)
# FragmentApply operand kinds, as emmy spells them
FRAG, ROW, UNIFORM, COORD, GMEM = "frag", "row", "uniform", "coord", "gmem"
# FragmentApply's element coordinates and RegStore's element offsets, as emmy names them
FRAG_ROW, FRAG_COL, ELEM_ROW, ELEM_COL = "__frow", "__fcol", "__M__", "__N__"
_REF = re.compile(r"^(\w+)(?:\[(\w+)\])?$")


class TensorCoreError(Exception):
    pass


@dataclass
class Fragment:
    """A RegFragment: ``count`` members of ``n`` registers each."""

    role: str  # "a", "b" or "c"
    dtype: str  # element dtype: f16 or bf16 operands, an f32 or f16 accumulator
    members: list[list[Variable]]
    arrayed: bool  # declared with a count, so referenced as name[i]


def fragment_refs(s) -> list[str]:
    """The fragments (``name`` or ``name[i]``) a statement reads or writes."""
    k = kind(s)
    if k == "LdmatrixLoad":
        return [s.frag] + ([s.pair_frag] if s.pair_frag is not None else [])
    if k == "MmaSyncPtx":
        return [s.c_frag, s.a_frag, s.b_frag]
    if k == "RegStore":
        return [s.frag, *s.extra_frags]
    if k == "FragmentApply":
        return [s.out, *(a for a, kd in zip(s.args, s.kinds) if kd == FRAG)]
    if k == "FragmentRowReduce":
        return list(s.frags)
    if k == "FragmentRepack":
        return [s.frag, *s.srcs]
    if k == "FragmentPromote":
        return [s.dst, s.src]
    return []


def fragment_index(ref: str) -> str | None:
    """The index of a fragment reference (``"_r0"`` of ``"_c0_[_r0]"``), if any."""
    m = _REF.match(ref)
    return m.group(2) if m else None


def fragments_written(stmts) -> set[str]:
    """The fragments (their declared names) the statements write, at any depth."""
    out = set()
    for s in walk(stmts):
        k = kind(s)
        if k == "LdmatrixLoad":
            refs = [s.frag] + ([s.pair_frag] if s.pair_frag is not None else [])
        elif k == "MmaSyncPtx":
            refs = [s.c_frag]
        elif k in ("FragmentApply", "FragmentRepack"):
            refs = [s.out if k == "FragmentApply" else s.frag]
        elif k == "FragmentPromote":
            refs = [s.dst, s.src]
        else:
            continue
        out.update(_REF.match(r).group(1) for r in refs)
    return out


class TensorCoreLowering:
    """The rules for the fragment statements, mixed into :class:`~emmy_ptx.lower.Lowerer`."""

    def fragment(self, ref: str) -> list[Variable]:
        """The registers of fragment reference ``name`` or ``name[i]`` (i a
        literal or a variable that is constant here, an unrolled loop's)."""
        frag = self.declared(ref)
        index = fragment_index(ref)
        if index is None:
            return frag.members[0]
        i = int(index) if index.isdigit() else self.value(index).const
        if i is None:
            raise TensorCoreError(f"fragment {ref!r}: index {index!r} is not known at compile time")
        return frag.members[int(i)]

    def declared(self, ref: str) -> Fragment:
        """The RegFragment a reference names."""
        m = _REF.match(ref)
        frag = self.scope.lookup(m.group(1)) if m else None
        if not isinstance(frag, Fragment):
            raise TensorCoreError(f"{ref!r} is not a declared fragment")
        return frag

    def forget_fragments(self, stmts) -> None:
        """At a merge point or a loop head: the fragments ``stmts`` write no longer hold what they did."""
        for name in fragments_written(stmts):
            frag = self.scope.lookup(name)
            if isinstance(frag, Fragment):
                for member in frag.members:
                    for var in member:
                        var.forget()

    # ------------------------------------------------------------- declaration

    def stmt_RegFragment(self, s) -> None:
        """Registers for ``count`` fragments: 4 (A), 2 (B) or 4 f32 (C; 2 packed for an f16
        accumulator); an accumulator starts at zero."""
        if tuple(s.shape) != M16N8K16:
            raise TensorCoreError(f"RegFragment {s.name}: shape {s.shape} (only m16n8k16 is supported)")
        b = self.b
        dtype = s.dtype.name
        wide = s.role == "c" and dtype == "f32"
        n = s.nregs if s.nregs is not None else {"a": 4, "b": 2, "c": 4 if wide else 2}[s.role]
        if n == 0:
            raise TensorCoreError(f"RegFragment {s.name}: block-scale registers are not supported")
        vtype = "f32" if wide else "s32"
        zero = (b.f32_const(0.0) if wide else b.int_const(0)) if s.role == "c" else None
        members = [[b.variable(vtype, zero) for _ in range(n)] for _ in range(s.count)]
        self.scope.names[s.name] = Fragment(s.role, dtype, members, s.count > 1)

    # ------------------------------------------------------------- operands

    def stmt_LdmatrixLoad(self, s) -> None:
        """One ``ldmatrix`` from a shared slab, else the global-memory loader."""
        if s.fragment_layout != "m16n8k16" or s.byte_slab or s.scale_buffer is not None:
            raise TensorCoreError("LdmatrixLoad: only 16-bit m16n8k16 operands are supported")
        buf = self.buffer(s.src_buffer)
        frag = self.fragment(s.frag)
        if not s.staged:
            self._fragment_from_global(s, buf, frag)
            return
        if not hasattr(buf, "symbol"):  # a SharedArray
            raise TensorCoreError(f"ldmatrix reads shared memory, not {s.src_buffer!r}")
        b = self.b
        lane = self.value("lane")
        ldm = s.ldm or buf.shape[-1]
        low16, low8 = b.bitop("&", lane, b.int_const(15)), b.bitop("&", lane, b.int_const(7))
        half = b.shr(lane, b.int_const(4))  # lanes 16-31 address the second pair of 8x8 matrices
        bit3 = b.bitop("&", b.shr(lane, b.int_const(3)), b.int_const(1))

        def row(r: Value, col8: Value) -> Value:  # the lane's row address: r * ldm + 8 * col8
            return b.add(b.mul(r, b.int_const(ldm)), b.mul(col8, b.int_const(8)))

        if s.pair_frag is not None:  # two B fragments side by side in one x4
            dests = frag[:2] + self.fragment(s.pair_frag)[:2]
            if s.b_trans:  # an N x K slab: N rows lane % 8 (+ 8 for the second fragment), K halves (lane / 8) % 2
                op, elem = "x4", row(b.add(low8, b.mul(half, b.int_const(8))), bit3)
            else:  # a K x N slab: K rows lane % 16, transposed; the second fragment 8 columns on
                op, elem = "x4.trans", row(low16, half)
        elif s.role == "a":  # 16 x 16: rows lane % 16, K halves lane / 16
            op, elem, dests = "x4", row(low16, half), frag[:4]
        elif s.b_trans:
            op, elem, dests = "x2", row(low8, bit3), frag[:2]
        else:
            op, elem, dests = "x2.trans", row(low16, b.int_const(0)), frag[:2]
        addr = self.address(buf, flat=self.flat(buf, s.src_index).plus(self._as_affine(elem)), swizzle=s.swizzle)
        b.emit(f"ldmatrix.sync.aligned.m8n8.{op}.shared.b16", Vec(tuple(v.reg for v in dests)), addr, ndefs=1,
               effect=False)
        for v in dests:
            b.overwritten(v)

    def _fragment_from_global(self, s, buf, frag: list[Variable]) -> None:
        """emmy's ``emmy_mma_load_{a,b}_gmem*``: each lane reads its own elements
        of the A (16x16) or B (16x8) tile, in the lane map ``ldmatrix`` would give,
        and packs them in pairs. A masked tile clamps the lane's row (A) or
        column (B) into range (the copy is never stored); elements past a masked
        K are zero, since the mma sums them."""
        b = self.b
        lane = self.value("lane")
        g, t2 = b.shr(lane, b.int_const(2)), b.mul(b.bitop("&", lane, b.int_const(3)), b.int_const(2))
        ldm = s.ldm or buf.shape[-1]
        base = self.flat(buf, s.src_index)

        def left(guard):  # elements in range from the tile base, or None
            return None if guard is None else b.sub(self.expr(guard[1]), self.expr(guard[0]))

        lanes_left, k_left = left(s.gmem_guard), left(s.k_zero)

        def clamp(x: Value) -> Value:
            if lanes_left is None:
                return x
            return b.imin(x, b.imax(b.sub(lanes_left, b.int_const(1)), b.int_const(0)))

        pairs = []  # per register: ((offset, k) of the low half, (offset, k) of the high half)
        if s.role == "a":
            for i in range(4):
                r, k = clamp(b.add(g, b.int_const(8 * (i & 1)))), b.add(t2, b.int_const(8 * (i >> 1)))
                at = b.add(b.mul(r, b.int_const(ldm)), k)
                pairs.append(((at, k), (b.add(at, b.int_const(1)), b.add(k, b.int_const(1)))))
        else:
            n = clamp(g)
            for i in range(2):
                k = b.add(t2, b.int_const(8 * i))
                k1 = b.add(k, b.int_const(1))
                if s.b_trans:  # N x K: the pair is contiguous in row n
                    at = b.add(b.mul(n, b.int_const(ldm)), k)
                    pairs.append(((at, k), (b.add(at, b.int_const(1)), k1)))
                else:
                    pairs.append(((b.add(b.mul(k, b.int_const(ldm)), n), k), (b.add(b.mul(k1, b.int_const(ldm)), n), k1)))
        for var, halves in zip(frag, pairs):
            parts = []
            for at, k in halves:
                v = self.load(buf, buf.type, None, 1, flat=base.plus(self._as_affine(at)))[0]
                v = self._half(v, buf.type, self.declared(s.frag).dtype)
                if k_left is not None:
                    v = b.select(b.icmp("<", k, k_left), v, Value("f16", Imm(0, "f16")))
                parts.append(b.in_reg(v))
            b.assign(var, b.pure("mov.b32", "s32", (Vec((parts[0].op, parts[1].op)),)))

    def _half(self, v: Value, src: str, dst: str) -> Value:
        """A loaded element as the fragment's 16-bit type."""
        if src == dst:
            return v
        if dst == "f16":
            return self.b.convert(v, "f16")
        raise TensorCoreError(f"a {src} element cannot fill a {dst} fragment")

    # ------------------------------------------------------------- the mma

    def stmt_MmaSyncPtx(self, s) -> None:
        """``d = a @ b + c`` on one 16x8 tile, ``d`` the accumulator itself."""
        if tuple(s.shape) != M16N8K16 or s.block_scaled or s.b_row_major or s.ab_dtype not in ("f16", "bf16"):
            raise TensorCoreError(f"MmaSyncPtx: m{s.shape} {s.ab_dtype} is not supported")
        c, a, bf = (self.fragment(r) for r in (s.c_frag, s.a_frag, s.b_frag))
        if s.c_dtype == "f32":
            op = f"mma.sync.aligned.m16n8k16.row.col.f32.{s.ab_dtype}.{s.ab_dtype}.f32"
        elif s.c_dtype == "f16" and s.ab_dtype == "f16":
            op = "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        else:
            raise TensorCoreError(f"MmaSyncPtx: a {s.c_dtype} accumulator for {s.ab_dtype} is not supported")

        def regs(f):
            return Vec(tuple(v.reg for v in f))

        self.b.emit(op, regs(c), regs(a), regs(bf), regs(c), ndefs=1, effect=False)
        for v in c:
            self.b.overwritten(v)

    # ------------------------------------------------------------- C-fragment arithmetic

    def stmt_FragmentApply(self, s) -> None:
        """``out[i] = post(op(args at element i))`` for the four elements."""
        if s.layout.n_elems != 4 or tuple(s.layout.elem_row) != C_ROW:
            raise TensorCoreError("FragmentApply: only the m16n8 C layout is supported")
        b = self.b
        values = []
        for i in range(4):
            with self.block(dominated=True):
                if any(k in (COORD, GMEM) for k in s.kinds):
                    self._bind_coordinates(s.row_base, s.col_base, i)
                args = [self._operand(a, k, i) for a, k in zip(s.args, s.kinds)]
                v = elementwise.apply(b, s.op.name, args, "f32")
                for post in s.post:
                    v = elementwise.apply(b, post.name, [v], "f32")
                values.append(v)
        if s.in_place:
            for var, v in zip(self.fragment(s.out), values):
                b.assign(var, v)
        else:
            self.scope.names[s.out] = Fragment("c", "f32", [[b.variable("f32", v) for v in values]], False)

    def _lane_coordinates(self) -> tuple[Value, Value]:
        """``(g, 2t)``: the lane's first row and column within a C tile."""
        b = self.b
        lane = self.value("lane")
        return b.shr(lane, b.int_const(2)), b.mul(b.bitop("&", lane, b.int_const(3)), b.int_const(2))

    def _bind_coordinates(self, row_base, col_base, i: int) -> None:
        """Element i's absolute coordinates under emmy's names for a COORD or GMEM operand."""
        b = self.b
        g, t2 = self._lane_coordinates()
        if row_base is not None:
            self.scope.names[FRAG_ROW] = b.add(self.expr(row_base), b.add(g, b.int_const(8 * C_ROW[i])))
        if col_base is not None:
            self.scope.names[FRAG_COL] = b.add(self.expr(col_base), b.add(t2, b.int_const(C_COL[i])))

    def _operand(self, arg, kind_: str, i: int) -> Value:
        if kind_ == FRAG:
            return self.fragment(arg)[i].read()
        if kind_ == ROW:  # one scalar per row of the lane: (row g, row g + 8)
            return self.value(arg[C_ROW[i]])
        if kind_ == UNIFORM:
            return self.value(arg) if isinstance(arg, str) else self.expr(arg)
        if kind_ == COORD:
            return self.pred(arg)
        if kind_ == GMEM:
            name, index = arg
            buf = self.buffer(name)
            return self.b.convert(self.load(buf, buf.type, index, 1)[0], "f32")
        raise TensorCoreError(f"FragmentApply: operand kind {kind_!r}")

    def stmt_FragmentRowReduce(self, s) -> None:
        """Per row of the lane, the fold of its elements in all ``frags`` (in
        order), then across the lanes that hold the row's other columns."""
        b = self.b
        frags = [self.fragment(f) for f in s.frags]
        rows = []
        for r in (0, 1):
            parts = [f[i].read() for f in frags for i in range(4) if C_ROW[i] == r]
            acc = parts[0]
            for p in parts[1:]:
                acc = elementwise.apply(b, s.op.name, [acc, p], "f32")
            rows.append(acc)
        for distance in s.layout.reduce_xors:
            for r in (0, 1):
                other = self._shuffle_xor(b.in_reg(rows[r]), distance, Imm(-1, "s32"))  # a tile's warp is whole
                rows[r] = elementwise.apply(b, s.op.name, [rows[r], other], "f32")
        self.define(s.top, rows[0])
        self.define(s.bot, rows[1])

    def stmt_FragmentRepack(self, s) -> None:
        """Two k-adjacent C fragments as one A operand: their lane maps agree,
        so each register packs two neighbouring elements (low half = column 2t)."""
        if s.fragment_layout != "m16n8k16" or len(s.srcs) != 2 or s.ab_dtype not in ("f16", "bf16"):
            raise TensorCoreError("FragmentRepack: only two m16n8k16 C fragments into f16 or bf16 are supported")
        c0, c1 = (self.fragment(x) for x in s.srcs)
        for var, (hi, lo) in zip(self.fragment(s.frag), ((c0[1], c0[0]), (c0[3], c0[2]), (c1[1], c1[0]), (c1[3], c1[2]))):
            self.b.assign(var, self.b.pure(f"cvt.rn.{s.ab_dtype}x2.f32", "s32", (hi.read(), lo.read())))

    def stmt_FragmentPromote(self, s) -> None:
        """Add an f16 accumulator's packed halves into its f32 shadow, then zero it."""
        b = self.b
        dst, src = self.fragment(s.dst), self.fragment(s.src)
        for i, h in enumerate(src):
            halves = (self.kernel.reg("f16"), self.kernel.reg("f16"))
            b.emit("mov.b32", Vec(halves), h.read(), ndefs=1, effect=False)
            for j, half in enumerate(halves):
                acc = dst[2 * i + j]
                b.assign(acc, b.fbin("+", acc.read(), b.convert(Value("f16", half), "f32")))
            b.assign(h, b.int_const(0))

    # ------------------------------------------------------------- the store

    def stmt_RegStore(self, s) -> None:
        """Each lane stores its four C elements (through the epilogue, if any):
        a row's two neighbouring columns as one 8-byte f32 or 4-byte f16x2
        store, unless a column guard or a transposed output splits them."""
        if s.fragment_layout != "m16n8k16" or s.volta_interleaved:
            raise TensorCoreError("RegStore: only m16n8k16 fragments are supported")
        b = self.b
        buf = self.buffer(s.dst_buffer)
        if buf.type not in ("f32", "f16"):
            raise TensorCoreError(f"RegStore into a {buf.type} buffer is not supported")
        frags = [self.fragment(s.frag), *(self.fragment(x) for x in s.extra_frags)]
        shape = buf.shape
        ldm = s.ldm or (math.prod(shape[s.row_dim + 1 :]) if s.row_dim is not None else shape[-1])
        ldn = math.prod(shape[s.col_dim + 1 :]) if s.col_dim is not None else 1
        base = self.flat(buf, s.dst_index)
        g, t2 = self._lane_coordinates()
        rows = [b.add(g, b.int_const(8 * C_ROW[i])) for i in range(4)]
        cols = [b.add(t2, b.int_const(C_COL[i])) for i in range(4)]
        values = [self._store_value(s, frags, i, rows[i], cols[i]) for i in range(4)]
        preds: list[Value | None] = [None] * 4
        for guard, coords in ((s.m_guard, rows), (s.n_guard, cols)):
            if guard is None:
                continue
            start, bound = self.expr(guard[0]), self.expr(guard[1])
            for i in range(4):
                p = b.icmp("<", b.add(start, coords[i]), bound)
                preds[i] = p if preds[i] is None else b.pand(preds[i], p)
        values = [b.in_reg(b.convert(v, buf.type)) for v in values]
        offsets = [b.add(b.mul(rows[i], b.int_const(ldm)), b.mul(cols[i], b.int_const(ldn))) for i in range(4)]
        space = "shared" if hasattr(buf, "symbol") else "global"
        even = ldm % 2 == 0 and base.const % 2 == 0 and all(c % 2 == 0 for _, c in base.terms.values())
        paired = (ldn == 1 and s.n_guard is None and even and buf.type in ("f32", "f16")
                  and not (s.atomic and buf.type == "f32"))
        for i in ((0, 2) if paired else range(4)):
            addr = self.address(buf, flat=base.plus(self._as_affine(offsets[i])), swizzle=s.swizzle)
            if not paired:
                op = (f"red.{space}.add.{'noftz.f16' if buf.type == 'f16' else buf.type}" if s.atomic
                      else f"st.{space}.{'b16' if buf.type == 'f16' else buf.type}")
                b.emit(op, addr, values[i], guard=preds[i])
            elif buf.type == "f32":
                b.emit(f"st.{space}.v2.f32", addr, Vec((values[i].op, values[i + 1].op)), guard=preds[i])
            else:  # two halves in one 32-bit word, low half the first column
                packed = b.pure("mov.b32", "s32", (Vec((values[i].op, values[i + 1].op)),))
                op = f"red.{space}.add.noftz.f16x2" if s.atomic else f"st.{space}.b32"
                b.emit(op, addr, b.in_reg(packed), guard=preds[i])

    def _store_value(self, s, frags, i: int, row: Value, col: Value) -> Value:
        """Element i's stored value (f32): the fragment's, or the fused epilogue's
        over it, its leading parameters bound to the fragments' element i and
        emmy's offset names to the element's row and column in the tile."""
        if s.epilogue is None:
            return frags[0][i].read()
        epi = s.epilogue
        with self.block(dominated=True):
            for param, frag in zip(epi.params, frags):
                self.scope.names[param] = frag[i].read()
            self.scope.names[ELEM_ROW], self.scope.names[ELEM_COL] = row, col
            self.body(tuple(epi.body))
            return self.value(epi.results[0])
