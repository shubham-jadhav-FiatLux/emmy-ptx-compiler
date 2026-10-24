"""PTX data model: operands, instructions and module text.

This module knows nothing about Emmy. A kernel is a flat list of instructions
and labels over typed *virtual* registers; :meth:`Kernel.render` prints the
module text that ``ptxas`` consumes. ptxas performs the real register
allocation, so registers here are unlimited and never reused.

Register classes
    Values live in one of six register classes. Integers use bit-typed
    registers (``.b32``/``.b64``) so the same register can feed signed,
    unsigned and bitwise instructions; floats use ``.f32``/``.f64``; halves use
    ``.b16``; comparisons produce ``.pred``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# register class -> (declared PTX type, name prefix)
_REG_CLASSES: dict[str, tuple[str, str]] = {
    "pred": ("pred", "%p"),
    "b16": ("b16", "%h"),
    "b32": ("b32", "%r"),
    "b64": ("b64", "%rd"),
    "f32": ("f32", "%f"),
    "f64": ("f64", "%fd"),
}

# PTX value type -> register class that holds it
REG_CLASS: dict[str, str] = {
    "pred": "pred",
    "b16": "b16", "u16": "b16", "s16": "b16", "f16": "b16",
    "b32": "b32", "u32": "b32", "s32": "b32",
    "b64": "b64", "u64": "b64", "s64": "b64",
    "f32": "f32",
    "f64": "f64",
}

# size in bytes of every PTX value type
SIZEOF: dict[str, int] = {
    "b16": 2, "u16": 2, "s16": 2, "f16": 2,
    "b32": 4, "u32": 4, "s32": 4, "f32": 4,
    "b64": 8, "u64": 8, "s64": 8, "f64": 8,
}


# ---------------------------------------------------------------------------
# Operands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reg:
    """A virtual register, e.g. ``%r7``."""

    name: str
    cls: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Imm:
    """An immediate operand, spelled for PTX type ``type``.

    Floats are printed as exact bit patterns (``0f3F800000`` is 1.0) so no
    decimal round trip can perturb a constant.
    """

    value: int | float | bool
    type: str

    def __str__(self) -> str:
        if self.type == "f32":
            return "0f%08X" % struct.unpack("<I", struct.pack("<f", self.value))[0]
        if self.type == "f64":
            return "0d%016X" % struct.unpack("<Q", struct.pack("<d", self.value))[0]
        if self.type == "pred":
            return "1" if self.value else "0"
        return str(int(self.value))


@dataclass(frozen=True)
class Sym:
    """A module-level symbol (a shared array or a kernel parameter)."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Mem:
    """A memory operand ``[base+offset]``; ``base`` is a register or a symbol."""

    base: Reg | Sym
    offset: int = 0

    def __str__(self) -> str:
        if self.offset == 0:
            return f"[{self.base}]"
        return f"[{self.base}+{self.offset}]"  # ptxas reads "+-4" as a negative offset


@dataclass(frozen=True)
class Vec:
    """A vector operand ``{a, b, ...}`` of ``ld/st.v2/.v4``."""

    items: tuple

    def __str__(self) -> str:
        return "{" + ", ".join(str(x) for x in self.items) + "}"


@dataclass(frozen=True)
class LabelRef:
    name: str

    def __str__(self) -> str:
        return self.name


Operand = Reg | Imm | Sym | Mem | Vec | LabelRef


def registers_in(operand: Operand) -> tuple[Reg, ...]:
    """Every register an operand mentions (a memory base, vector items)."""
    if isinstance(operand, Reg):
        return (operand,)
    if isinstance(operand, Mem):
        return (operand.base,) if isinstance(operand.base, Reg) else ()
    if isinstance(operand, Vec):
        return tuple(r for item in operand.items for r in registers_in(item))
    return ()


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


@dataclass
class Instr:
    """One PTX instruction.

    ``args`` are the operands in printed order; the first ``ndefs`` of them are
    written by the instruction. ``effect`` marks instructions that must never
    be deleted even when nothing reads their result (stores, barriers,
    branches). ``guard`` is an optional predicate, e.g. ``Reg`` with
    ``negate=True`` prints ``@!%p1``.
    """

    op: str
    args: tuple = ()
    ndefs: int = 1
    effect: bool = False
    guard: Reg | None = None
    negate: bool = False

    def defs(self) -> tuple[Reg, ...]:
        return tuple(r for a in self.args[: self.ndefs] for r in registers_in(a))

    def uses(self) -> tuple[Reg, ...]:
        used = tuple(r for a in self.args[self.ndefs :] for r in registers_in(a))
        return used + ((self.guard,) if self.guard is not None else ())

    def render(self) -> str:
        guard = f"@{'!' if self.negate else ''}{self.guard} " if self.guard is not None else ""
        if not self.args:
            return f"\t{guard}{self.op};"
        return f"\t{guard}{self.op} \t{', '.join(str(a) for a in self.args)};"


@dataclass
class Label:
    name: str

    def render(self) -> str:
        return f"{self.name}:"


@dataclass
class Comment:
    text: str

    def render(self) -> str:
        return f"\t// {self.text}"


@dataclass
class Block:
    """A slot in the instruction stream filled after the fact (a loop preheader)."""

    items: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kernel / module
# ---------------------------------------------------------------------------


@dataclass
class Kernel:
    """A ``.entry`` function plus the module-scope declarations it needs."""

    name: str
    params: tuple[str, ...]
    maxntid: int | None = None
    minnctapersm: int | None = None
    body: list = field(default_factory=list)
    decls: list[str] = field(default_factory=list)  # module-scope lines (shared arrays)
    _counts: dict[str, int] = field(default_factory=dict)
    _labels: int = 0

    def reg(self, ptx_type: str) -> Reg:
        cls = REG_CLASS[ptx_type]
        n = self._counts.get(cls, 0) + 1
        self._counts[cls] = n
        return Reg(f"{_REG_CLASSES[cls][1]}{n}", cls)

    def label(self, hint: str = "L") -> str:
        self._labels += 1
        return f"${hint}_{self._labels}"

    def emit(self, item) -> None:
        self.body.append(item)

    def flatten(self) -> None:
        """Splice every :class:`Block` into the instruction stream."""
        self.body = [x for item in self.body for x in (item.items if isinstance(item, Block) else (item,))]

    def render(self, version: str, target: str) -> str:
        self.flatten()
        out = [
            "//",
            f"// {self.name}: generated by emmy-ptx-compiler",
            "//",
            "",
            f".version {version}",
            f".target {target}",
            ".address_size 64",
            "",
        ]
        if self.decls:
            out.extend(self.decls)
            out.append("")
        params = ",\n".join(f"\t.param .u64 {p}" for p in self.params)
        out.append(f".visible .entry {self.name}(" + (f"\n{params}\n" if params else "") + ")")
        if self.maxntid:
            out.append(f".maxntid {self.maxntid}, 1, 1")
        if self.minnctapersm:
            out.append(f".minnctapersm {self.minnctapersm}")
        out.append("{")
        for cls, (ptx_type, prefix) in _REG_CLASSES.items():
            if cls in self._counts:
                out.append(f"\t.reg .{ptx_type} \t{prefix}<{self._counts[cls] + 1}>;")
        out.append("")
        out.extend(item.render() for item in self.body)
        out.append("}")
        return "\n".join(out) + "\n"
