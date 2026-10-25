#!/usr/bin/env python3
"""Put measured kernel times next to the hardware's limits and a CPU baseline.

    python tools/roofline.py BENCH.json PROBLEMS_DIR [...] [--cpu]

BENCH.json is written by ``tools/ref_bench.py --json`` (our time and nvcc's
for each problem). For every problem this prints the useful work (matmul
FLOPs counted by torch's FlopCounterMode on the problem's own expression,
and the bytes of every argument), the throughput each kernel achieved as a
fraction of the RTX 5090's peaks, and optionally the time PyTorch takes for
the same expression on this machine's CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PEAK_FLOPS = 104.8e12  # RTX 5090 FP32, boost clock
PEAK_BYTES = 1.792e12  # RTX 5090 GDDR7 bandwidth
SIZE = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i32": 4, "i64": 8}


def problem_dirs(paths):
    out = {}
    for p in map(Path, paths):
        dirs = [p] if (p / "meta.json").exists() else [d for d in p.iterdir() if (d / "meta.json").exists()]
        out.update({d.name: d for d in dirs})
    return out


def work(meta: dict) -> tuple[float, float]:
    """(matmul FLOPs, bytes moved by the arguments) of a problem."""
    nbytes = sum(math.prod(a["shape"]) * SIZE.get(a["dtype"], 4) for k in meta["kernels"] for a in k["args"])
    flops = 0.0
    code = meta.get("code")
    if code:
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        stmts = [s.strip() for s in code.split(";")]
        env = {"torch": torch}
        with FlopCounterMode(display=False) as fc:
            for s in stmts[:-1]:
                exec(s, env)
            eval(stmts[-1], env)
        flops = float(fc.get_total_flops())
    return flops, nbytes


def cpu_time(code: str) -> float:
    import torch

    stmts = [s.strip() for s in code.split(";")]
    env = {"torch": torch}
    for s in stmts[:-1]:
        exec(s, env)
    expr = compile(stmts[-1], "<expr>", "eval")
    eval(expr, env)  # warm up
    reps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 0.2:
        eval(expr, env)
        reps += 1
    return (time.perf_counter() - t0) / reps * 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--cpu", action="store_true", help="also time PyTorch on this CPU")
    args = ap.parse_args()
    dirs = problem_dirs(args.paths)
    rows = [r for r in json.loads(Path(args.bench).read_text()) if "error" not in r and r["id"] in dirs]
    print(f"{'problem':<26}{'ours us':>9}{'nvcc us':>9}{'speedup':>8}{'GFLOP':>8}{'MB':>8}"
          f"{'ours %peak':>11}{'nvcc %peak':>11}" + (f"{'cpu us':>10}{'vs cpu':>8}" if args.cpu else ""))
    for r in sorted(rows, key=lambda r: r["id"]):
        meta = json.loads((dirs[r["id"]] / "meta.json").read_text())
        flops, nbytes = work(meta)

        def peak(us):  # the binding resource's utilisation
            s = us * 1e-6
            return 100 * max(flops / s / PEAK_FLOPS, nbytes / s / PEAK_BYTES)

        line = (f"{r['id']:<26}{r['ours_us']:>9.1f}{r['ref_us']:>9.1f}{r['ref_us'] / r['ours_us']:>7.2f}x"
                f"{flops / 1e9:>8.2f}{nbytes / 1e6:>8.2f}{peak(r['ours_us']):>10.1f}%{peak(r['ref_us']):>10.1f}%")
        if args.cpu and meta.get("code"):
            c = cpu_time(meta["code"])
            line += f"{c:>10.1f}{c / r['ours_us']:>7.1f}x"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
