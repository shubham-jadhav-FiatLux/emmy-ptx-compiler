#!/usr/bin/env python3
"""What compiling each kernel costs: cold compile.py time, PTX size, and ptxas time.

    python tools/compile_cost.py PROBLEMS_DIR [...] [--ptxas PATH] [--flags=--explore] [--jobs N] [--json out.json]

Each kernel is compiled by a fresh ``python compile.py`` process (the
evaluator's way: interpreter start, imports, lowering), then assembled with a
local ptxas for sm_120 (the runner's GPU) at -O3. ptxas is not part of the
compiler; it is timed because an autotuner pays for it on every candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite import COMPILER, problem_dirs  # noqa: E402


def cost(ir: Path, ptxas: str, flags: list[str]) -> dict:
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(COMPILER), *flags, str(ir)], capture_output=True, text=True)
    gen = time.perf_counter() - t0
    row = {"id": ir.parent.name, "kernel": ir.stem.removesuffix(".ir"), "generate_s": gen, "ptx_chars": len(r.stdout)}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "k.ptx"
        src.write_text(r.stdout)
        t0 = time.perf_counter()
        a = subprocess.run([ptxas, "-arch=sm_120", "-O3", "-v", str(src), "-o", str(Path(td) / "k.cubin")],
                           capture_output=True, text=True)
        row["ptxas_s"] = time.perf_counter() - t0
    regs = re.findall(r"Used (\d+) registers", a.stderr)
    row["registers"] = max(map(int, regs)) if regs else None
    row["ok"] = r.returncode == 0 and a.returncode == 0
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ptxas", default=shutil.which("ptxas") or "ptxas")
    ap.add_argument("--flags", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()
    irs = [ir for d in problem_dirs(args.paths) for ir in sorted(d.glob("*.ir.json"))]
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda ir: cost(ir, args.ptxas, shlex.split(args.flags)), irs))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    gen = sorted(r["generate_s"] for r in rows)
    asm = sorted(r["ptxas_s"] for r in rows)
    print(f"{len(rows)} kernels, {sum(not r['ok'] for r in rows)} failed")
    for name, v in (("compile.py", gen), ("ptxas", asm)):
        print(f"  {name:10} median {statistics.median(v):.3f} s  p95 {v[int(0.95 * len(v))]:.3f} s  max {v[-1]:.3f} s  "
              f"total {sum(v):.1f} s")
    print("  largest modules:")
    for r in sorted(rows, key=lambda r: -r["ptx_chars"])[:8]:
        print(f"    {r['id']:32} {r['ptx_chars']:>9} chars  compile.py {r['generate_s']:.2f} s  ptxas {r['ptxas_s']:.2f} s  "
              f"{r['registers']} regs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
