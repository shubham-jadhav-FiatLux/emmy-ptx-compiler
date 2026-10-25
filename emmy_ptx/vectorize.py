"""Merge runs of adjacent scalar loads into vector loads (``ld.v2`` / ``ld.v4``).

After unrolling, a lane often loads ``x[i], x[i+1], x[i+2], x[i+3]`` with
separate instructions: one shared base register and the immediates
``+0, +4, +8, +12``. A vector load fetches them in one instruction when the
first address is aligned to the vector's size, which the lowering proves and
records per address register (coefficients of the index times the element
size, over a base that is at least 16-byte aligned).

Within a basic block the merged load takes the place of the earliest load of
the run; the later ones are deleted. That only moves loads earlier, and never
past a branch or a label. It never moves a load past a store, atomic, barrier
or async copy either, except for read-only (``.nc``) data, which nothing in
the kernel can change.
"""

from __future__ import annotations

from .ptx import Instr, Label, Mem, Reg, Sym, Vec

_WRITES = ("st.", "red.", "atom.", "bar.", "barrier.", "cp.async", "membar", "fence")
_SCALAR32 = (".f32", ".b32", ".s32", ".u32")


# State spaces whose loads are merged. Shared memory is left alone: measured on
# the evaluation GPU, wide shared loads cost more (bank conflicts, issue order)
# than they save in emmy's matmul tiles (m02: 3.5 -> 4.8 us with them).
SPACES = ("global",)


def vectorize_loads(kernel, align: dict[str, int], sym_align: int = 16) -> int:
    """Rewrite ``kernel.body`` in place; returns how many loads were merged away."""
    body = kernel.body
    replace: dict[int, Instr] = {}
    delete: set[int] = set()
    groups: dict[tuple, list] = {}

    def base_align(base) -> int:
        return sym_align if isinstance(base, Sym) else align.get(base.name, 4)

    def flush(key) -> None:
        loads = groups.pop(key)
        op, base = key[0], loads[0][2].base
        by_offset = {}
        for idx, dst, mem in loads:
            by_offset.setdefault(mem.offset, (idx, dst))
        if len(by_offset) != len(loads):
            return  # repeated addresses (not value-numbered): leave them alone
        a = base_align(base)
        offsets = sorted(by_offset)
        taken = set()
        for o in offsets:
            if o in taken:
                continue
            for width in (4, 2):
                run = [o + 4 * k for k in range(width)]
                if all(r in by_offset and r not in taken for r in run) and o % (4 * width) == 0 and a % (4 * width) == 0:
                    members = [by_offset[r] for r in run]
                    first = min(idx for idx, _ in members)
                    vec_op = op[: op.rindex(".")] + f".v{width}" + op[op.rindex(".") :]
                    replace[first] = Instr(vec_op, (Vec(tuple(dst for _, dst in members)), Mem(base, o)), effect=False)
                    delete.update(idx for idx, _ in members if idx != first)
                    taken.update(run)
                    break

    for idx, item in enumerate(body):
        if isinstance(item, Label) or (isinstance(item, Instr) and (item.op.startswith("bra") or item.op == "ret")):
            for key in list(groups):
                flush(key)
            continue
        if not isinstance(item, Instr):
            continue
        op = item.op
        if (op.startswith("ld.") and ".v" not in op and item.guard is None and op.endswith(_SCALAR32)
                and op.split(".")[1] in SPACES and isinstance(item.args[0], Reg) and isinstance(item.args[1], Mem)):
            mem = item.args[1]
            groups.setdefault((op, mem.base.name), []).append((idx, item.args[0], mem))
            continue
        if op.startswith(_WRITES):
            for key in [k for k in groups if ".nc." not in k[0]]:
                flush(key)
    for key in list(groups):
        flush(key)
    if not replace:
        return 0
    kernel.body = [replace.get(i, item) for i, item in enumerate(body) if i not in delete]
    return len(delete)
