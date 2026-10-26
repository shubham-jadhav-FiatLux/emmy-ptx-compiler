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


def compile_file(ir_path: str, plain: bool = False, strict: bool = False, report: bool = False,
                 bulk_copy: bool = False) -> str:
    """Compile one ``*.ir.json`` file and return the PTX module text.

    Should the optimizing lowering fail (an IR shape one of its rewrites did
    not foresee) or give a module longer than the runner accepts, the kernel
    is lowered again plainly (``Lowerer(plain=True)``): one rule per
    statement, in the IR's order, without the optional rewrites. ``plain``
    lowers it that way from the start (``compile.py --plain``). ``strict``
    keeps only the rewrites that leave every floating-point result bit for bit
    as the IR's own order and rounding give it (``compile.py --strict``).
    ``report`` writes what the lowering decided to stderr (report.py);
    ``bulk_copy`` hands contiguous cp.async fills to the copy engine (bulk.py)."""
    src = load(ir_path)
    mode = "strictly" if strict else "optimized"
    if not plain:
        try:
            kernel = Lowerer(src, strict=strict, bulk_copy=bulk_copy).run()
            ptx = kernel.render(PTX_VERSION, TARGET)
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
    if report:
        print(_report(src, kernel, ptx, mode), file=sys.stderr)
    return ptx
