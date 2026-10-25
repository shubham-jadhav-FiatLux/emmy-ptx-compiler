#!/usr/bin/env python3
"""Compile one Emmy Kernel-IR file to PTX.

    python compile.py path/to/<kernel>.ir.json > out.ptx
    python compile.py --plain path/to/<kernel>.ir.json > out.ptx

PTX goes to stdout; diagnostics go to stderr. ``meta.json`` must sit next to
the IR file (it names the kernel parameters in launch order). ``--plain``
gives the fallback lowering, without the optional rewrites (for testing it).
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emmy_ptx import compile_file  # noqa: E402


def main(argv: list[str]) -> int:
    paths = [a for a in argv[1:] if a != "--plain"]
    if len(paths) != 1:
        print("usage: python compile.py [--plain] <kernel>.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(paths[0], plain="--plain" in argv[1:])
    except Exception:  # report everything; stdout must stay clean
        traceback.print_exc(file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
