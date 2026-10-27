#!/usr/bin/env python3
"""Compile one Emmy Kernel-IR file to PTX.

    python compile.py [--strict] [--plain] [--report] [--bulk-copy] [--explore] path/to/<kernel>.ir.json > out.ptx

PTX goes to stdout; diagnostics go to stderr. ``meta.json`` must sit next to
the IR file (it names the kernel parameters in launch order).

``--strict`` turns off the rewrites that change floating-point results within
rounding (sums regrouped across lanes, an online softmax updated once per
block, FMA contraction of ``acc += a * b``): every result is then bit for bit
what the IR's own order of operations gives. ``--plain`` gives the fallback
lowering, without any optional rewrite (for testing it). ``--report`` writes
to stderr what the lowering decided for the kernel and a profile of its PTX.
``--bulk-copy`` hands the shared-memory fills that copy contiguous runs to the
copy engine (``cp.async.bulk`` completing on mbarriers, sm_90 and later).
``--explore`` keeps the module small enough for ptxas to assemble quickly
(a compile budget for an autotuner's exploratory trials), unrolling less.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emmy_ptx import compile_file  # noqa: E402


FLAGS = ("--strict", "--plain", "--report", "--bulk-copy", "--explore")


def main(argv: list[str]) -> int:
    flags = {a for a in argv[1:] if a.startswith("--")}
    paths = [a for a in argv[1:] if not a.startswith("--")]
    if len(paths) != 1 or flags - set(FLAGS):
        print("usage: python compile.py [--strict] [--plain] [--report] [--bulk-copy] [--explore] <kernel>.ir.json"
              " > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(paths[0], plain="--plain" in flags, strict="--strict" in flags, report="--report" in flags,
                           bulk_copy="--bulk-copy" in flags, explore="--explore" in flags)
    except Exception:  # report everything; stdout must stay clean
        traceback.print_exc(file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
