"""emmy_ptx: a compiler from Emmy Kernel IR to NVIDIA PTX.

Pipeline::

    frontend.load      IR json + meta.json  ->  KernelSource
    lower.Lowerer      statements           ->  ptx.Kernel   (via builder.Builder)
    ptx.Kernel.render  instructions         ->  PTX text
"""

from __future__ import annotations

from .frontend import load
from .lower import Lowerer

# PTX ISA 8.0 (CUDA 12.0) for sm_89: every problem assembles for the RTX 4090
# it was scheduled on, and ptxas compiles it unchanged for newer GPUs.
PTX_VERSION = "8.0"
TARGET = "sm_89"


def compile_file(ir_path: str) -> str:
    """Compile one ``*.ir.json`` file and return the PTX module text."""
    return Lowerer(load(ir_path)).run().render(PTX_VERSION, TARGET)
