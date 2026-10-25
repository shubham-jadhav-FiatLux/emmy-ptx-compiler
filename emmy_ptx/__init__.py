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

# PTX ISA 8.0 (CUDA 12.0) for sm_89: every problem assembles for the RTX 4090
# it was scheduled on, and ptxas compiles it unchanged for newer GPUs.
PTX_VERSION = "8.0"
TARGET = "sm_89"
# The longest PTX module (characters) the evaluation runner accepts.
PTX_LIMIT = 2_000_000


def compile_file(ir_path: str, plain: bool = False) -> str:
    """Compile one ``*.ir.json`` file and return the PTX module text.

    Should the optimizing lowering fail (an IR shape one of its rewrites did
    not foresee) or give a module longer than the runner accepts, the kernel
    is lowered again plainly (``Lowerer(plain=True)``): one rule per
    statement, in the IR's order, without the optional rewrites. ``plain``
    lowers it that way from the start (``compile.py --plain``)."""
    src = load(ir_path)
    if not plain:
        try:
            ptx = Lowerer(src).run().render(PTX_VERSION, TARGET)
            if len(ptx) <= PTX_LIMIT:
                return ptx
            print(f"warning: {src.name}: {len(ptx)} characters of PTX, more than the runner accepts", file=sys.stderr)
        except Exception:  # noqa: BLE001 - whatever failed, the plain lowering may still succeed
            traceback.print_exc(file=sys.stderr)
        print(f"warning: {src.name}: lowering it again without the optional rewrites", file=sys.stderr)
    return Lowerer(src, plain=True).run().render(PTX_VERSION, TARGET)
