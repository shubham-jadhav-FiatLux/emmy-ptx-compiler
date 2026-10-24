#!/usr/bin/env python3
"""Compile one Emmy Kernel-IR file to PTX.

    python compile.py path/to/<kernel>.ir.json > out.ptx

PTX goes to stdout; diagnostics go to stderr. ``meta.json`` must sit next to
the IR file (it names the kernel parameters in launch order).
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emmy_ptx import compile_file  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py <kernel>.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(argv[1])
    except Exception:  # report everything; stdout must stay clean
        traceback.print_exc(file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
