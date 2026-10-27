"""Typed values and an instruction builder that folds, simplifies and reuses.

The lowering never writes PTX opcodes for arithmetic directly; it calls the
:class:`Builder`, which

* folds constants (``a0 * 4 + 0`` with ``a0`` known becomes an immediate),
* tracks a proven integer range ``[lo, hi]`` for every int value, and uses it
  to pick cheaper instructions: unsigned division, shifts and masks for powers
  of two, dropping ``x % n`` when ``x < n``, and deciding comparisons outright,
* value-numbers pure instructions (common-subexpression elimination) within
  dominance scopes that the lowering opens around branches and loop bodies,
* widens integer arithmetic to 64 bits only when a range could overflow 32.

Integer semantics are C's (``/`` truncates, ``%`` is the C remainder), which
is what Emmy's index expressions mean. Float arithmetic is IEEE with
round-to-nearest (``.rn``), in single precision, or double when an operand is
f64: spelled that way, ptxas may not contract a multiply and an add into one
FMA, so results match numpy bit for bit.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np

from .ptx import Block, Imm, Instr, Kernel, Label, LabelRef, Mem, Reg, Vec

INT_RANGE = {"s32": (-(2**31), 2**31 - 1), "s64": (-(2**63), 2**63 - 1), "u32": (0, 2**32 - 1), "u64": (0, 2**64 - 1)}
WARP = 32
NUMPY_FLOAT = {"f32": np.float32, "f64": np.float64}


@dataclass(frozen=True)
class Value:
    """An operand with a type and, for integers, a proven inclusive range.

    ``type`` is one of ``s32``, ``s64`` (C ``int`` / ``long long``), ``f32``,
    ``f16``, ``f64`` or ``pred``. ``mutable`` marks a read of a register that is
    reassigned elsewhere (an accumulator or a loop counter): such reads are
    never value-numbered.

    ``ug`` (uniform group) is the largest aligned group of lanes, 1 to 32,
    within which every lane of a warp is guaranteed to hold the same value:
    32 for constants and block-level indices, 1 for anything lane-dependent.
    """

    type: str
    op: Reg | Imm
    lo: int | None = None
    hi: int | None = None
    mutable: bool = False
    ug: int = 1
    version: int | None = None  # for a read of a variable: which write it sees

    def __post_init__(self):
        if isinstance(self.op, Imm):
            object.__setattr__(self, "ug", WARP)

    @property
    def const(self):
        return self.op.value if isinstance(self.op, Imm) else None

    @property
    def is_int(self) -> bool:
        return self.type in INT_RANGE

    @property
    def is_float(self) -> bool:
        return self.type in ("f16", "f32", "f64")

    @property
    def nonneg(self) -> bool:
        return self.lo is not None and self.lo >= 0

    def key(self):
        """Identity of the value for value numbering (``None``: never reuse).

        A read of a variable is identified by the write it sees: two reads
        with no write (or merge point) between them are the same value."""
        if self.mutable:
            return None if self.version is None else ("var", self.op.name, self.version)
        if isinstance(self.op, Imm):
            value = self.op.value
            # a float by its bits: +0 and -0 are equal numbers but different values (1/x tells them apart)
            return ("imm", self.op.type, value.hex() if isinstance(value, float) else value)
        return ("reg", self.op.name)


@dataclass
class Variable:
    """A mutable register: an accumulator, a reassigned state or a loop counter."""

    reg: Reg
    type: str
    lo: int | None = None
    hi: int | None = None
    ug: int = 1  # uniformity of the value it holds now (see Value.ug)
    last: object = None  # value-numbering key of the value it holds now, when known
    version: int = 0  # bumped by every write and every merge point

    def read(self) -> Value:
        lo, hi = (self.lo, self.hi) if self.type in INT_RANGE else (None, None)
        if self.type in INT_RANGE and lo is None:
            lo, hi = INT_RANGE[self.type]
        return Value(self.type, self.reg, lo, hi, mutable=True, ug=self.ug, version=self.version)

    def forget(self) -> None:
        """A merge point or loop head: the value held is no longer known."""
        self.ug, self.last, self.version = 1, None, self.version + 1


def _is_pow2(n) -> bool:
    return isinstance(n, int) and n > 0 and n & (n - 1) == 0


def _cdiv(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _crem(a: int, b: int) -> int:
    """C remainder (takes the sign of the dividend)."""
    return a - _cdiv(a, b) * b


def _f32(x) -> float:
    return float(np.float32(x))


def _operand_key(x):
    if isinstance(x, Value):
        return x.key()
    if isinstance(x, Mem):
        base = ("reg", x.base.name) if isinstance(x.base, Reg) else ("sym", x.base.name)
        return ("mem", base, x.offset)
    return ("raw", str(x))


@dataclass
class _Loop:
    """An open loop: where its invariant instructions go, and how many may."""

    preheader: list  # instruction list spliced in just before the loop's first trip
    table: int  # index of the value-numbering table live at the preheader
    guards: int  # conditional nesting depth when the loop opened
    budget: int  # instructions this preheader may still receive (register pressure)
    loads: bool = True  # whether loads may be hoisted here


class Builder:
    """Emits instructions into a :class:`~emmy_ptx.ptx.Kernel` (see module doc)."""

    # Per loop, at most this many invariant instructions are hoisted into its
    # preheader: each hoisted value occupies a register for the whole loop.
    HOIST_BUDGET = 96

    def __init__(self, kernel: Kernel):
        self.k = kernel
        self._tables: list[dict] = [{}]
        self._loops: list[_Loop] = []
        self._level: dict[str, int] = {}  # register -> loop depth of its definition
        self._guards = 0
        self._linear: dict = {}  # value key -> (terms, const): value = sum(coeff * term) + const
        self._ug: dict[str, int] = {}  # register -> uniform group of its value (see Value.ug)
        self.align: dict[str, int] = {}  # address register -> known byte alignment
        self._defs: dict = {}  # value key -> (opcode, operands) of the instruction that made it
        self._vars: dict[str, Variable] = {}  # register -> the variable it holds

    # -- dominance scopes for value numbering --------------------------------

    @contextlib.contextmanager
    def cse_scope(self):
        """Values numbered inside the block are forgotten when it closes."""
        self._tables.append({})
        try:
            yield
        finally:
            self._tables.pop()

    @contextlib.contextmanager
    def loop(self, hoist_loads: bool = True):
        """The body of a rolled loop (open it after the zero-trip check).

        Loop-invariant code motion: a pure instruction whose operands were all
        defined before an enclosing loop is emitted once, in that loop's
        preheader, instead of once per trip. Loads qualify only from read-only
        memory and only when no branch inside the loop guards them, so a
        hoisted load never runs where the original would not have; with
        ``hoist_loads`` False they stay in the loop (a value per trip is
        cheaper than a register held for the whole loop when there are many).

        The preheader follows the zero-trip check, so it dominates the loop
        body but not the code after the loop: hoisted values are numbered in
        a table of their own that closes with the loop.
        """
        pre = Block()
        self.k.emit(pre)
        self._tables.append({})
        self._loops.append(_Loop(pre.items, len(self._tables) - 1, self._guards, self.HOIST_BUDGET, hoist_loads))
        try:
            yield
        finally:
            self._loops.pop()
            self._tables.pop()

    @contextlib.contextmanager
    def guarded(self):
        """Code that only runs under a condition (a branch body)."""
        self._guards += 1
        try:
            yield
        finally:
            self._guards -= 1

    def _lookup(self, key):
        for table in reversed(self._tables):
            if key in table:
                return table[key]
        return None

    def _depth_of(self, x) -> int:
        """Loop depth at which operand ``x`` becomes available."""
        if isinstance(x, Value):
            if x.mutable:
                return len(self._loops)
            x = x.op
        if isinstance(x, Vec):
            return max((self._depth_of(r) for r in x.items), default=0)
        if isinstance(x, Mem):
            x = x.base
        return self._level.get(x.name, 0) if isinstance(x, Reg) else 0

    def _hoist_target(self, op: str, args: tuple) -> _Loop | None:
        """The outermost open loop whose preheader can hold this instruction."""
        depth = max((self._depth_of(a) for a in args), default=0)
        for frame in self._loops[depth:]:
            if frame.budget > 0 and (not op.startswith("ld.") or frame.loads and frame.guards == self._guards):
                return frame
        return None

    def pure(self, op: str, type_: str, args: tuple, lo=None, hi=None, commutative=False) -> Value:
        """Emit, or reuse, a side-effect-free instruction with one result."""
        keys = tuple(_operand_key(a) for a in args)
        # a variable's register in a vector operand may hold something else next time
        changing = any(isinstance(a, Vec) and any(isinstance(r, Reg) and r.name in self._vars for r in a.items)
                       for a in args)
        if commutative:
            order = sorted(range(len(args)), key=lambda i: repr(keys[i]))
            args, keys = tuple(args[i] for i in order), tuple(keys[i] for i in order)
            # PTX reads best with the immediate last
            if len(args) == 2 and isinstance(args[0], Value) and args[0].const is not None:
                args, keys = args[::-1], keys[::-1]
        key = None if changing or None in keys else (op, keys)
        if key is not None:
            hit = self._lookup(key)
            if hit is not None:
                return hit
        reg = self.k.reg(type_)
        instr = Instr(op, (reg, *(a.op if isinstance(a, Value) else a for a in args)))
        ug = min((self.ug_of(a) for a in args), default=WARP)
        self._ug[reg.name] = ug
        target = self._hoist_target(op, args)
        if target is not None:
            target.preheader.append(instr)
            target.budget -= 1
            self._level[reg.name] = self._loops.index(target)
            table = self._tables[target.table]
        else:
            self.k.emit(instr)
            self._level[reg.name] = len(self._loops)
            table = self._tables[-1]
        if type_ in INT_RANGE and lo is None:
            lo, hi = INT_RANGE[type_]
        v = Value(type_, reg, lo, hi, ug=ug)
        if key is not None:
            table[key] = v
            self._defs[v.key()] = (op, args)
        return v

    def def_of(self, v: Value):
        """``(opcode, operands)`` of the pure instruction that computed ``v``."""
        return self._defs.get(v.key()) if v.key() is not None else None

    def ug_of(self, x) -> int:
        """Uniform group of an operand (a Value, an address, a symbol, a special register)."""
        if isinstance(x, Value):
            return x.ug
        if isinstance(x, Vec):
            return min((self.ug_of(r) for r in x.items), default=WARP)
        if isinstance(x, Mem):
            x = x.base
        if isinstance(x, Reg):
            if x.name.startswith("%ctaid") or x.name.startswith("%nctaid") or x.name.startswith("%ntid"):
                return WARP
            return self._ug.get(x.name, 1)
        return WARP  # immediates, symbols, labels

    def uniform(self, v: Value, ug: int) -> Value:
        """``v``, known (from the mathematics of its computation) to be uniform
        within aligned groups of ``ug`` lanes."""
        if v.const is not None:
            return v
        if isinstance(v.op, Reg):
            self._ug[v.op.name] = ug
        return Value(v.type, v.op, v.lo, v.hi, v.mutable, ug, v.version)

    def emit(self, op: str, *args, ndefs: int = 0, effect: bool = True, guard: Value | None = None, negate=False,
             ug: int = 1):
        """Emit an instruction verbatim (stores, barriers, async copies ...);
        ``ug`` is the uniformity of whatever it defines."""
        g = None
        if guard is not None:
            if guard.const is not None:
                if bool(guard.const) == negate:
                    return  # statically never executes
            else:
                g = guard.op
        ops = tuple(a.op if isinstance(a, Value) else a for a in args)
        instr = Instr(op, ops, ndefs=ndefs, effect=effect, guard=g, negate=negate)
        for r in instr.defs():
            self._level[r.name] = len(self._loops)
            self._ug[r.name] = ug
        self.k.emit(instr)

    # -- constants and registers ------------------------------------------------

    def int_const(self, v: int, type_: str | None = None) -> Value:
        v = int(v)
        if type_ is None:
            type_ = "s32" if INT_RANGE["s32"][0] <= v <= INT_RANGE["s32"][1] else "s64"
        v = _wrap_int(v, type_)
        return Value(type_, Imm(v, type_), v, v)

    def f32_const(self, x: float) -> Value:
        return Value("f32", Imm(_f32(x), "f32"))

    def float_const(self, x: float, type_: str) -> Value:
        """``x`` as an f64 constant, or rounded to an f32 one for any other type."""
        return Value("f64", Imm(float(x), "f64")) if type_ == "f64" else self.f32_const(x)

    def pred_const(self, b: bool) -> Value:
        return Value("pred", Imm(bool(b), "pred"))

    def const_of(self, x, type_: str) -> Value:
        """The number ``x`` (an IR initializer) as a constant of value type
        ``type_``, rounded once: f64 keeps every bit, f16 rounds straight from
        the double, and an integer is exact (a float out of its range saturates,
        as ``cvt.rzi`` does)."""
        if type_ in INT_RANGE:
            lo, hi = INT_RANGE[type_]
            if isinstance(x, float):
                x = 0 if x != x else min(max(x, lo), hi)  # (cvt.rzi turns NaN into 0)
            return self.int_const(min(max(int(x), lo), hi), type_)
        if type_ == "f16":
            with np.errstate(over="ignore"):  # (past the f16 range: an infinity, as C converts it)
                return self.convert(self.f32_const(float(np.float16(x))), "f16")  # exact: every f16 is an f32
        return self.float_const(float(x), type_)

    def special(self, name: str, lo: int, hi: int) -> Value:
        """Read a special register such as ``%tid.x`` (range from the launch)."""
        if lo == hi:
            return self.int_const(lo)
        return self.pure("mov.u32", "s32", (Reg(name, "b32"),), lo, hi)

    def snapshot(self, v: Value) -> Value:
        """``v`` as an immutable value: a copy when it is a read of a register
        that will be reassigned."""
        if not v.mutable:
            return v
        reg = self.k.reg(v.type)  # a real copy: never value-numbered back onto the variable
        self.k.emit(Instr(f"mov.{_mov_type(v.type)}", (reg, v.op)))
        self._ug[reg.name] = v.ug
        self._level[reg.name] = len(self._loops)
        return Value(v.type, reg, v.lo, v.hi, ug=v.ug)

    def note_align(self, reg: Reg, align: int) -> None:
        """Record that the address in ``reg`` is a multiple of ``align`` bytes."""
        self.align[reg.name] = align

    def assume(self, v: Value, lo: int, hi: int) -> Value:
        """``v`` with a range the caller knows from the mathematics (e.g. a clamp
        done in floating point before the conversion to int)."""
        if v.const is not None:
            return v
        return Value(v.type, v.op, max(lo, v.lo), min(hi, v.hi), v.mutable, v.ug, v.version)

    def in_reg(self, v: Value) -> Value:
        """``v`` in a register (stores and some instructions take no immediates)."""
        if isinstance(v.op, Reg):
            return v
        return self.pure(f"mov.{_mov_type(v.type)}", v.type, (v,), v.lo, v.hi)

    # -- mutable variables ------------------------------------------------------

    def variable(self, type_: str, init: Value | None = None, lo=None, hi=None) -> Variable:
        var = Variable(self.k.reg(type_), type_, lo, hi)
        self._level[var.reg.name] = 1 << 30  # reassigned: invariant in no loop
        self._vars[var.reg.name] = var
        if init is not None:
            self.assign(var, init)
        return var

    def current(self, v: Value) -> bool:
        """Does ``v`` still hold what was read? Always for an immutable value; a
        read of a variable only until the next write to it (or merge point)."""
        if not v.mutable:
            return True
        var = self._vars.get(v.op.name)
        return var is not None and v.version is not None and var.version == v.version

    def overwritten(self, var: Variable) -> None:
        """The instruction just emitted wrote ``var``'s register itself (an
        ``mma`` accumulating in place): what it held is gone."""
        var.ug, var.last = 1, None
        var.version += 1

    def assign(self, var: Variable, v: Value) -> None:
        v = self.convert(v, var.type)
        var.ug = v.ug
        var.last = v.key()
        var.version += 1
        self.k.emit(Instr(f"mov.{_mov_type(var.type)}", (var.reg, v.op)))

    # -- conversions -------------------------------------------------------------

    def convert(self, v: Value, type_: str) -> Value:
        """C conversion of ``v`` to ``type_``."""
        if v.type == type_:
            return v
        if type_ == "pred":
            return self.truthy(v)
        if v.type == "pred":
            if type_ in INT_RANGE:
                return self.select(v, self.int_const(1, type_), self.int_const(0, type_))
            one, zero = self.f32_const(1.0), self.f32_const(0.0)
            return self.convert(self.select(v, one, zero), type_)
        if v.is_int and type_ in INT_RANGE:  # C's conversion: the value modulo the width
            if v.const is not None:
                return self.int_const(v.const, type_)
            lo, hi = INT_RANGE[type_]
            keep = v.lo is not None and lo <= v.lo and v.hi <= hi
            rlo, rhi = (v.lo, v.hi) if keep else (None, None)
            wide, to_wide = v.type in ("s64", "u64"), type_ in ("s64", "u64")
            if wide == to_wide:  # the same bits, read with the other sign
                return self.pure(f"mov.b{64 if wide else 32}", type_, (v,), rlo, rhi)
            if to_wide:  # sign- or zero-extended, as the source's own sign says
                return self.pure("cvt.s64.s32" if v.type == "s32" else "cvt.u64.u32", type_, (v,), rlo, rhi)
            return self.pure("cvt.u32.u64", type_, (v,), rlo, rhi)  # the low half
        if v.is_int and type_ in ("f32", "f64"):
            if v.const is not None:
                return self.float_const(float(v.const), type_)
            return self.pure(f"cvt.rn.{type_}.{v.type}", type_, (v,))
        if v.is_float and type_ in INT_RANGE:
            if v.type == "f16":
                v = self.convert(v, "f32")
            return self.pure(f"cvt.rzi.{type_}.{v.type}", type_, (v,))
        if v.is_float and type_ in ("f16", "f32", "f64"):
            if v.const is not None and v.type in NUMPY_FLOAT and type_ in NUMPY_FLOAT:
                return self.float_const(v.const, type_)
            rnd = ".rn" if _float_rank(type_) < _float_rank(v.type) else ""
            return self.pure(f"cvt{rnd}.{type_}.{v.type}", type_, (self.in_reg(v),))
        raise TypeError(f"cannot convert {v.type} to {type_}")

    def truthy(self, v: Value) -> Value:
        """C truthiness: non-zero (NaN counts as true)."""
        if v.type == "pred":
            return v
        if v.const is not None:
            return self.pred_const(v.const != 0)
        if v.is_int:
            return self.icmp("!=", v, self.int_const(0, v.type))
        return self.pure(f"setp.neu.{v.type}", "pred", (v, self._fzero(v.type)))

    def _fzero(self, type_: str) -> Value:
        return self.f32_const(0.0) if type_ == "f32" else self.convert(self.f32_const(0.0), type_)

    # -- integer arithmetic ---------------------------------------------------------

    def _widen(self, a: Value, b: Value, lo: int, hi: int):
        """Common integer type for ``a op b``; 64-bit if the range needs it."""
        t = "s64" if {a.type, b.type} & {"s64", "u64"} else "s32"
        if t == "s32" and not (INT_RANGE["s32"][0] <= lo and hi <= INT_RANGE["s32"][1]):
            t = "s64"
        return self.convert(a, t), self.convert(b, t), t

    def add(self, a: Value, b: Value) -> Value:
        if a.const is not None and b.const is not None:
            return self.int_const(a.const + b.const)
        if a.const == 0:
            return b
        if b.const == 0:
            return a
        lo, hi = a.lo + b.lo, a.hi + b.hi
        a, b, t = self._widen(a, b, lo, hi)
        v = self.pure(f"add.{t}", t, (a, b), lo, hi, commutative=True)
        self._note_linear(v, [(a, 1), (b, 1)])
        return v

    def sub(self, a: Value, b: Value) -> Value:
        if a.const is not None and b.const is not None:
            return self.int_const(a.const - b.const)
        if b.const == 0:
            return a
        if a.key() is not None and a.key() == b.key():
            return self.int_const(0)
        lo, hi = a.lo - b.hi, a.hi - b.lo
        a, b, t = self._widen(a, b, lo, hi)
        v = self.pure(f"sub.{t}", t, (a, b), lo, hi)
        self._note_linear(v, [(a, 1), (b, -1)])
        return v

    def _note_linear(self, v: Value, parts) -> None:
        """Remember how ``v`` was built as ``sum(coeff * part)`` (constants folded
        into the constant term), so addresses can split off the constant into
        an immediate offset and see the factors that prove alignment."""
        if v.key() is None or v.const is not None:
            return
        terms, const = [], 0
        for part, coeff in parts:
            if part.const is not None:
                const += coeff * int(part.const)
            elif part.key() is None:
                return  # a reassigned register: not a fixed quantity
            else:
                terms.append((part, coeff))
        self._linear[v.key()] = (terms, const)

    def linear_of(self, v: Value):
        """``(terms, const)`` with v = sum(c * t) + const, if ``v`` was built that way."""
        return self._linear.get(v.key()) if v.key() is not None else None

    def mul(self, a: Value, b: Value) -> Value:
        if a.const is not None and b.const is not None:
            return self.int_const(a.const * b.const)
        if b.const is None and a.const is not None:
            a, b = b, a
        if b.const == 0:
            return self.int_const(0)
        if b.const == 1:
            return a
        corners = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        lo, hi = min(corners), max(corners)
        if _is_pow2(b.const):
            a2, _, t = self._widen(a, b, lo, hi)
            v = self.pure(f"shl.b{t[1:]}", t, (a2, Imm(b.const.bit_length() - 1, "u32")), lo, hi)
        else:
            a2, b2, t = self._widen(a, b, lo, hi)
            v = self.pure(f"mul.lo.{t}", t, (a2, b2), lo, hi, commutative=True)
        if b.const is not None:
            self._note_linear(v, [(a2, int(b.const))])
        return v

    def div(self, a: Value, b: Value) -> Value:
        """C ``/`` on integers (truncation)."""
        if a.const is not None and b.const is not None and b.const != 0:
            return self.int_const(_cdiv(a.const, b.const))
        if b.const == 1:
            return a
        a, b, t = self._widen(a, b, min(a.lo, -a.hi), max(a.hi, -a.lo))  # |a / b| <= |a|
        if a.nonneg and b.lo > 0:
            lo, hi = a.lo // b.hi, a.hi // b.lo
            if lo == hi:
                return self.int_const(lo, t)
            if _is_pow2(b.const):
                return self.pure(f"shr.u{t[1:]}", t, (a, Imm(b.const.bit_length() - 1, "u32")), lo, hi)
            return self.pure(f"div.u{t[1:]}", t, (a, b), lo, hi)
        bound = max(abs(a.lo), abs(a.hi))
        return self.pure(f"div.{t}", t, (a, b), -bound, bound)

    def rem(self, a: Value, b: Value) -> Value:
        """C ``%`` on integers (sign of the dividend)."""
        if a.const is not None and b.const is not None and b.const != 0:
            return self.int_const(_crem(a.const, b.const))
        if b.const in (1, -1):
            return self.int_const(0, a.type)
        a, b, t = self._widen(a, b, a.lo, a.hi)
        if a.nonneg and b.lo > 0:
            if a.hi < b.lo:
                return a
            hi = min(a.hi, b.hi - 1)
            if _is_pow2(b.const):
                return self.pure(f"and.b{t[1:]}", t, (a, Imm(b.const - 1, t)), 0, hi)
            return self.pure(f"rem.u{t[1:]}", t, (a, b), 0, hi)
        mag = max(abs(b.lo), abs(b.hi)) - 1
        return self.pure(f"rem.{t}", t, (a, b), -mag if not a.nonneg else 0, mag)

    def shl(self, a: Value, b: Value) -> Value:
        if b.const is not None:
            return self.mul(a, self.int_const(1 << b.const, a.type))
        return self.pure(f"shl.b{a.type[1:]}", a.type, (a, self.convert(b, "s32")))

    def shr(self, a: Value, b: Value) -> Value:
        if a.const is not None and b.const is not None:
            return self.int_const(a.const >> b.const)
        if b.const is not None:
            sign = "u" if a.nonneg else "s"  # arithmetic shift == floor division, like Python's >>
            return self.pure(f"shr.{sign}{a.type[1:]}", a.type, (a, Imm(b.const, "u32")), a.lo >> b.const, a.hi >> b.const)
        return self.pure(f"shr.{a.type}", a.type, (a, self.convert(b, "s32")))

    def bitop(self, op: str, a: Value, b: Value) -> Value:
        """``&``, ``|`` or ``^`` on integers."""
        fold = {"&": lambda x, y: x & y, "|": lambda x, y: x | y, "^": lambda x, y: x ^ y}[op]
        if a.const is not None and b.const is not None:
            return self.int_const(fold(a.const, b.const))
        if b.const is None and a.const is not None:
            a, b = b, a
        if op == "&" and a.nonneg and b.const is not None and _is_pow2(b.const + 1) and a.hi <= b.const:
            return a  # the mask keeps every bit the value can have
        if op != "&" and b.const == 0:
            return a
        a, b, t = self._widen(a, b, min(a.lo, b.lo), max(a.hi, b.hi))
        lo = hi = None
        if op == "&" and (a.nonneg or b.nonneg):  # x & m lies in [0, m] for m >= 0
            lo, hi = 0, min(v.hi for v in (a, b) if v.nonneg)
        elif a.nonneg and b.nonneg:
            lo, hi = 0, (1 << max(a.hi, b.hi).bit_length()) - 1
        name = {"&": "and", "|": "or", "^": "xor"}[op]
        return self.pure(f"{name}.b{t[1:]}", t, (a, b), lo, hi, commutative=True)

    def imin(self, a: Value, b: Value) -> Value:
        if a.hi <= b.lo:
            return a
        if b.hi <= a.lo:
            return b
        a, b, t = self._widen(a, b, min(a.lo, b.lo), max(a.hi, b.hi))
        return self.pure(f"min.{t}", t, (a, b), min(a.lo, b.lo), min(a.hi, b.hi), commutative=True)

    def imax(self, a: Value, b: Value) -> Value:
        if a.lo >= b.hi:
            return a
        if b.lo >= a.hi:
            return b
        a, b, t = self._widen(a, b, min(a.lo, b.lo), max(a.hi, b.hi))
        return self.pure(f"max.{t}", t, (a, b), max(a.lo, b.lo), max(a.hi, b.hi), commutative=True)

    def icmp(self, op: str, a: Value, b: Value) -> Value:
        """Integer comparison -> predicate, decided statically when ranges allow."""
        decided = _decide(op, a.lo, a.hi, b.lo, b.hi)
        if decided is not None:
            return self.pred_const(decided)
        if a.type == b.type:  # one type: one comparison of that type (unsigned or signed)
            return self.pure(f"setp.{_CMP[op]}.{a.type}", "pred", (a, b))
        a, b, t = self._widen(a, b, min(a.lo, b.lo), max(a.hi, b.hi))
        return self.pure(f"setp.{_CMP[op]}.{t}", "pred", (a, b))

    # -- predicates ---------------------------------------------------------------

    def pnot(self, p: Value) -> Value:
        if p.const is not None:
            return self.pred_const(not p.const)
        return self.pure("not.pred", "pred", (p,))

    def pand(self, a: Value, b: Value) -> Value:
        for x, y in ((a, b), (b, a)):
            if x.const is not None:
                return y if x.const else self.pred_const(False)
        return self.pure("and.pred", "pred", (a, b), commutative=True)

    def por(self, a: Value, b: Value) -> Value:
        for x, y in ((a, b), (b, a)):
            if x.const is not None:
                return self.pred_const(True) if x.const else y
        return self.pure("or.pred", "pred", (a, b), commutative=True)

    def select(self, p: Value, a: Value, b: Value) -> Value:
        """``p ? a : b``. Integer operands share a width, floats a precision."""
        if p.const is not None:
            return a if p.const else b
        if a.key() is not None and a.key() == b.key():
            return a
        if a.is_int and b.is_int and a.type == b.type:  # (integers of one type stay that type)
            t, lo, hi = a.type, min(a.lo, b.lo), max(a.hi, b.hi)
        elif a.is_int and b.is_int:
            a, b, t = self._widen(a, b, min(a.lo, b.lo), max(a.hi, b.hi))
            lo, hi = min(a.lo, b.lo), max(a.hi, b.hi)
        else:
            t = a.type if a.type == b.type else float_type(a, b)
            a, b, lo, hi = self.convert(a, t), self.convert(b, t), None, None
        return self.pure(f"selp.{_mov_type(t)}", t, (a, b, p), lo, hi)

    # -- float arithmetic (IEEE, round to nearest) ------------------------------

    def fbin(self, op: str, a: Value, b: Value) -> Value:
        """``+ - * /`` on f32 (f64 when an operand is f64), folded when both
        operands are constants."""
        t = float_type(a, b)
        a, b = self.convert(a, t), self.convert(b, t)
        if a.const is not None and b.const is not None:
            x, y = NUMPY_FLOAT[t](a.const), NUMPY_FLOAT[t](b.const)
            with np.errstate(all="ignore"):
                r = {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[op]
            return self.float_const(float(r), t)
        # identities exact for every x, NaN and -0 included (x + 0 is not: -0 + 0 = +0)
        if (op == "*" and b.const == 1.0) or (op == "/" and b.const == 1.0) or (op == "-" and _is_plus_zero(b)):
            return a
        if op == "/" and _exact_inverse(b.const, t) is not None:
            # x / 2^k == x * 2^-k exactly (both scale by a power of two, rounding alike)
            return self.fbin("*", a, self.float_const(_exact_inverse(b.const, t), t))
        if op == "*" and a.const == 1.0:
            return b
        name = {"+": "add.rn", "-": "sub.rn", "*": "mul.rn", "/": "div.rn"}[op]
        return self.pure(f"{name}.{t}", t, (a, b), commutative=op in "+*")

    def fma(self, a: Value, b: Value, c: Value) -> Value:
        t = float_type(a, b, c)
        return self.pure(f"fma.rn.{t}", t, tuple(self.convert(x, t) for x in (a, b, c)))

    def fop(self, op: str, *args, type_: str = "f32") -> Value:
        """Any other pure instruction, e.g. ``fop("sqrt.rn.f32", x)``."""
        return self.pure(op, type_, args)

    def bits(self, x: Value) -> Value:
        """The IEEE bit pattern of an f32 as an int."""
        return self.pure("mov.b32", "s32", (self.convert(x, "f32"),))

    def from_bits(self, i: Value) -> Value:
        """Reinterpret an int's bits as an f32."""
        if i.const is not None:
            return self.f32_const(float(np.array([i.const & 0xFFFFFFFF], np.uint32).view(np.float32)[0]))
        return self.pure("mov.b32", "f32", (i,))

    def fcmp(self, op: str, a: Value, b: Value) -> Value:
        """C float comparison -> predicate (ordered, except ``!=``), in f32
        unless an operand is f64."""
        t = float_type(a, b)
        a, b = self.convert(a, t), self.convert(b, t)
        if a.const is not None and b.const is not None:
            x, y = a.const, b.const
            return self.pred_const({"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y, "==": x == y, "!=": x != y}[op])
        code = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "neu"}[op]
        return self.pure(f"setp.{code}.{t}", "pred", (a, b))

    # -- control flow ---------------------------------------------------------------

    def label(self, hint: str = "L") -> str:
        return self.k.label(hint)

    def place(self, label: str) -> None:
        self.k.emit(Label(label))

    def branch(self, label: str, when: Value | None = None, negate: bool = False) -> None:
        """``bra label``, optionally predicated on ``when`` (or its negation)."""
        self.emit("bra", LabelRef(label), guard=when, negate=negate)


_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}


def _decide(op, alo, ahi, blo, bhi):
    """Statically decide ``a op b`` from ranges, or ``None``."""
    if None in (alo, ahi, blo, bhi):
        return None
    if op == "<":
        return True if ahi < blo else False if alo >= bhi else None
    if op == "<=":
        return True if ahi <= blo else False if alo > bhi else None
    if op == ">":
        return True if alo > bhi else False if ahi <= blo else None
    if op == ">=":
        return True if alo >= bhi else False if ahi < blo else None
    if op == "==":
        if alo == ahi == blo == bhi:
            return True
        return False if ahi < blo or bhi < alo else None
    if op == "!=":
        d = _decide("==", alo, ahi, blo, bhi)
        return None if d is None else not d
    return None


def _exact_inverse(c, type_: str = "f32") -> float | None:
    """1/c when c is a power of two whose inverse is a normal float of
    ``type_``, else None."""
    if c is None or c == 0 or not math.isfinite(c):
        return None
    m, e = math.frexp(abs(c))
    low, high = (-1021, 1023) if type_ == "f64" else (-125, 127)
    if m != 0.5 or not low <= 1 - e <= high:
        return None
    return math.copysign(2.0 ** (1 - e), c)


def float_type(*values: Value) -> str:
    """The type float arithmetic on ``values`` is done in: f64 if any is, else f32."""
    return "f64" if any(v.type == "f64" for v in values) else "f32"


def _is_plus_zero(v: Value) -> bool:
    return v.const is not None and v.const == 0.0 and math.copysign(1.0, v.const) > 0


def _mov_type(t: str) -> str:
    return {"s32": "b32", "s64": "b64", "u32": "b32", "u64": "b64", "f16": "b16"}.get(t, t)


def _wrap_int(v: int, type_: str) -> int:
    """``v`` as integer type ``type_`` holds it: modulo its width, two's
    complement for the signed types."""
    if type_ not in INT_RANGE:
        return v
    bits = 64 if type_ in ("s64", "u64") else 32
    v &= (1 << bits) - 1
    return v - (1 << bits) if type_[0] == "s" and v >> (bits - 1) else v


def _float_rank(t: str) -> int:
    return {"f16": 0, "f32": 1, "f64": 2}[t]


def eliminate_dead_code(kernel: Kernel) -> None:
    """Delete pure instructions whose results are never read, to a fixpoint.

    Flow-insensitive and therefore safe for registers written more than once:
    an instruction goes only if *no* instruction anywhere reads what it writes.
    """
    while True:
        used = {r.name for item in kernel.body if isinstance(item, Instr) for r in item.uses()}
        keep = [
            item
            for item in kernel.body
            if not isinstance(item, Instr) or item.effect or not item.defs() or any(r.name in used for r in item.defs())
        ]
        if len(keep) == len(kernel.body):
            return
        kernel.body[:] = keep
