"""emmy_ptx: a compiler from Emmy Kernel IR to NVIDIA PTX.

Pipeline::

    frontend.load      IR json + meta.json  ->  KernelSource
    lower.Lowerer      statements           ->  ptx.Kernel   (via builder.Builder)
    ptx.Kernel.render  instructions         ->  PTX text
"""

from __future__ import annotations

import sys
import traceback

from .frontend import load
from .lower import Lowerer
from .ptx import PTX_LIMIT, PTX_VERSION, TARGET
from .report import report as _report

# compile.py --explore: the PTX an autotuner's exploratory candidate should take. ptxas time grows
# with the module (about 6 s for 1.1 MB); past this size the kernel is lowered again with its
# complete unrolling scaled down, each step a quarter of the last, until it fits or no smaller
# budget can change what is unrolled. A target, not a bound: what is not optional unrolling stays.
EXPLORE_PTX_CHARS = 250_000
EXPANSIONS = (1.0, 0.25, 0.0625, 0.0)


def compile_file(ir_path: str, plain: bool = False, strict: bool = False, report: bool = False,
                 bulk_copy: bool = False, explore: bool = False) -> str:
    """Compile one ``*.ir.json`` file and return the PTX module text.

    Should the optimizing lowering fail (an IR shape one of its rewrites did
    not foresee) or give a module longer than the runner accepts, the kernel
    is lowered again plainly (``Lowerer(plain=True)``): one rule per
    statement, in the IR's order, without the optional rewrites. This is a
    fallback for a lowering that fails, not a check of the code: wrong code
    that lowers without an error is not caught here (the tests do that). ``plain``
    lowers it that way from the start (``compile.py --plain``). ``strict``
    keeps only the rewrites that leave every floating-point result bit for bit
    as the IR's own order and rounding give it (``compile.py --strict``).
    ``report`` writes what the lowering decided to stderr (report.py);
    ``bulk_copy`` hands contiguous cp.async fills to the copy engine (bulk.py);
    ``explore`` aims the module at a size ptxas assembles fast
    (EXPLORE_PTX_CHARS), at some cost in speed: the policy for an autotuner's
    exploratory trials, where assembly time counts as much as the kernel's.
    ``--report`` says when the size cannot be reached."""
    src = load(ir_path)
    mode = "strictly" if strict else "optimized"
    if not plain:
        try:
            best = None
            for expansion in EXPANSIONS if explore else (1.0,):
                kernel = Lowerer(src, strict=strict, bulk_copy=bulk_copy, expansion=expansion).run()
                ptx = kernel.render(PTX_VERSION, TARGET)
                if best is None or len(ptx) < 0.9 * len(best[1]) or len(ptx) <= EXPLORE_PTX_CHARS < len(best[1]):
                    best = kernel, ptx, expansion  # (a module barely smaller is not worth its rolled loops)
                if len(best[1]) <= EXPLORE_PTX_CHARS or not kernel.budgeted:
                    break  # small enough, or no smaller budget can change what is unrolled
            kernel, ptx, expansion = best
            if explore and (expansion < 1 or len(ptx) > EXPLORE_PTX_CHARS):
                scaled = f"complete unrolling scaled by {expansion}" if expansion < 1 else "the unrolling kept"
                kernel.remarks.append(
                    f"compile budget (--explore): {scaled} to keep the module under {EXPLORE_PTX_CHARS} characters"
                    if len(ptx) <= EXPLORE_PTX_CHARS else
                    f"compile budget (--explore): target not met: {scaled}, {len(ptx)} characters remain, more than "
                    f"{EXPLORE_PTX_CHARS} (less optional unrolling does not make the module 10% smaller)")
            if len(ptx) <= PTX_LIMIT:
                if report:
                    print(_report(src, kernel, ptx, mode), file=sys.stderr)
                return ptx
            print(f"warning: {src.name}: {len(ptx)} characters of PTX, more than the runner accepts", file=sys.stderr)
        except Exception:  # noqa: BLE001 - whatever failed, the plain lowering may still succeed
            traceback.print_exc(file=sys.stderr)
        print(f"warning: {src.name}: lowering it again without the optional rewrites", file=sys.stderr)
        mode = "plainly (the fallback)"
    else:
        mode = "plainly" + (" and strictly" if strict else "")
    kernel = Lowerer(src, plain=True, strict=strict).run()
    ptx = kernel.render(PTX_VERSION, TARGET)
    if len(ptx) > PTX_LIMIT:
        print(f"error: {src.name}: even the plain lowering gives {len(ptx)} characters of PTX, more than the "
              f"runner's {PTX_LIMIT}", file=sys.stderr)
    if report:
        print(_report(src, kernel, ptx, mode), file=sys.stderr)
    return ptx
