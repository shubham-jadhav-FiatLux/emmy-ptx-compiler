"""``compile.py --report``: what the lowering decided for a kernel, and what PTX it gave.

The lowering notes each decision as it makes it (``Lowerer.remark``): the
shape it gave every loop (split across lanes, lane-blocked, dealt, unrolled,
rolled), fused and merged loops and accumulators, cells re-mapped onto lanes,
the one-barrier trees, FMA contraction, vector loads and launch hints. A
decision repeated in unrolled code is listed once with its count. The last
line profiles the PTX: static instruction counts by kind, the ones that show
where a kernel spends its issue slots and memory traffic. Before the
decisions come the planner's plans (planner.py): the loops whose repeated work
pays to share, and at which scope. Only the report runs the planner, so it
cannot change the PTX.
"""

from __future__ import annotations

import re
from collections import Counter

from . import planner

# (singular, plural, test on the opcode) in the order they are printed
_KINDS = (
    ("FMA", "FMAs", lambda op: op.startswith("fma.") and _is_float(op)),
    ("other float op", "other float ops",
     lambda op: re.match(r"(add|sub|mul|div|max|min|neg|abs|cvt)\.", op) and _is_float(op)),
    ("special function", "special functions", lambda op: re.match(r"(ex2|lg2|rcp|sqrt|rsqrt|sin|cos|tanh)\.", op)),
    ("global load", "global loads", lambda op: op.startswith("ld.global")),
    ("shared load", "shared loads", lambda op: op.startswith("ld.shared")),
    ("global store", "global stores", lambda op: op.startswith("st.global")),
    ("shared store", "shared stores", lambda op: op.startswith("st.shared")),
    ("atomic", "atomics", lambda op: op.startswith(("red.", "atom."))),
    ("async copy", "async copies", lambda op: op.startswith("cp.async") and "commit" not in op and "wait" not in op),
    ("ldmatrix", "ldmatrix", lambda op: op.startswith("ldmatrix")),
    ("mma", "mma", lambda op: op.startswith("mma.")),
    ("shuffle", "shuffles", lambda op: op.startswith("shfl.")),
    ("barrier", "barriers", lambda op: op.startswith("bar.")),
    ("branch", "branches", lambda op: op == "bra" or op.startswith("bra.")),
)


def _is_float(op: str) -> bool:
    return bool(re.search(r"\.(f16|f16x2|bf16|bf16x2|f32|f64)\b", op))


def profile(ptx: str) -> str:
    """Static instruction counts of a PTX module, by kind (vector accesses noted)."""
    counts, vectors = Counter(), Counter()
    lines = 0
    for line in ptx.splitlines():
        text = line.strip()
        if not text or text.startswith(("//", ".", "{", "}", "$")) or text.endswith(":"):
            continue
        lines += 1
        op = re.sub(r"^@!?%p\d+\s+", "", text).split(None, 1)[0]
        for one, _, test in _KINDS:
            if test(op):
                counts[one] += 1
                if ".v2." in op or ".v4." in op:
                    vectors[one] += 1
                break
    parts = []
    for one, many, _ in _KINDS:
        if counts[one]:
            vec = f" ({vectors[one]} vector)" if vectors[one] else ""
            parts.append(f"{counts[one]} {one if counts[one] == 1 else many}{vec}")
    return f"{lines} instructions: " + ", ".join(parts)


def plan_remarks(src) -> list[str]:
    """The planner's plan for every loop whose repeated work pays to share."""
    try:
        return [p.describe() for p in planner.plans(src.body, src.launch.block, src.launch.grid) if p.best != "none"]
    except Exception as e:  # noqa: BLE001 - a report must not stop the kernel from compiling
        return [f"plan: none ({type(e).__name__}: {e})"]


def report(src, kernel, ptx: str, mode: str) -> str:
    """The report for one kernel: launch, plans, decisions, PTX profile."""
    launch = src.launch
    smem = f", {launch.smem_bytes} B of dynamic shared memory" if launch.smem_bytes else ""
    blocks = "1 block" if launch.grid == 1 else f"{launch.grid} blocks"
    out = [f"{kernel.name}: {blocks} x {launch.block} threads{smem}; lowered {mode}"]
    remarks = plan_remarks(src) + kernel.remarks
    counts = Counter(remarks)
    for text in dict.fromkeys(remarks):  # first appearance order, each once
        out.append(f"  {text}" + (f"  (x{counts[text]})" if counts[text] > 1 else ""))
    out.append(f"  PTX: {profile(ptx)}")
    return "\n".join(out)
