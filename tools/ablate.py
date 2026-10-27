#!/usr/bin/env python3
"""What each optional rewrite contributes to speed: switch it off, measure.

    python tools/ablate.py PROBLEMS_DIR [...] [--rewrites split,blocking,...] [--jobs 4] [--json out.json]

For every rewrite of ``emmy_ptx.lower.REWRITES`` (and the plain lowering,
every rewrite off at once) each kernel is compiled with that rewrite switched
off (``EMMY_PTX_DISABLE``). A problem whose PTX does not change runs the very
same code, so its ratio is exactly 1; only the problems whose PTX changes are
raced on the runner, full build against ablated build, in the order
full, ablated, ablated, full, each run the median of 20 launches and each side
its best of two runs. The table gives, per rewrite, how many problems it
changes, the geometric mean of ablated / full time over those and over all
problems (unchanged ones at 1), the same per family (the schedule variants of
one expression count once), and the problem that loses the most without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ref_bench import launch  # noqa: E402
from run_suite import COMPILER, RUNNER, problem_dirs  # noqa: E402

from emmy_ptx.lower import REWRITES  # noqa: E402

PLAIN = "plain"


def compile_ptx(ir: Path, config: str | None) -> str:
    env = dict(os.environ)
    env.pop("EMMY_PTX_DISABLE", None)
    args = [sys.executable, str(COMPILER), str(ir)]
    if config == PLAIN:
        args.insert(2, "--plain")
    elif config:
        env["EMMY_PTX_DISABLE"] = config
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{ir.name} [{config}]: {r.stderr.strip()[-300:]}")
    return r.stdout


def family(pid: str) -> str:
    """Schedule variants of one expression (``x_sum__reduce-r4``...) form one family."""
    return pid.split("__")[0]


def category(pid: str) -> str:
    knobs = {"x_amax": "reductions", "x_attn": "attention", "x_mm": "matmul", "x_mm_rect": "matmul",
             "x_rms": "fused", "x_rows": "reductions", "x_softmax": "fused", "x_sum": "reductions"}
    if pid.startswith("x_"):
        return knobs.get(family(pid), "other")
    head = pid.lstrip("h23")[:1] if pid.startswith("h") else pid[:1]
    return {"a": "attention", "f": "fused", "k": "chained", "m": "matmul", "p": "pointwise",
            "r": "reductions"}.get(head, "other")


def run_problem(pdir: Path, ptx: list[str], reps: int) -> tuple[float, bool]:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    produced, total, ok = {}, 0.0, True
    for k, p in zip(meta["kernels"], ptx):
        _, res = launch(RUNNER, p, k, produced, reps)
        total += float(res.get("median_us") or 0)
        for n, o in res["outputs"].items():
            produced[n] = o["values"]
            if n in expected:
                got = np.array(o["values"], np.float32).reshape(expected[n].shape)
                ok &= bool(np.allclose(got, expected[n], rtol=meta["rtol"], atol=meta["atol"]))
    return total, ok


def geomean(xs) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rewrites", default=",".join([*REWRITES, PLAIN]))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    configs = [c for c in args.rewrites.split(",") if c]
    dirs = problem_dirs(args.paths)
    kernels = {d: [d / f"{k['entry']}.ir.json" for k in json.loads((d / "meta.json").read_text())["kernels"]]
               for d in dirs}

    def build(job):
        d, config = job
        try:
            return d, config, [compile_ptx(ir, config) for ir in kernels[d]]
        except RuntimeError as e:
            return d, config, str(e)

    with ThreadPoolExecutor(8) as pool:
        built = list(pool.map(build, [(d, c) for d in dirs for c in [None, *configs]]))
    ptx = {(d, c): p for d, c, p in built}
    digest = {key: hashlib.md5("".join(p).encode()).hexdigest() if isinstance(p, list) else None
              for key, p in ptx.items()}
    jobs = [(d, c) for c in configs for d in dirs
            if isinstance(ptx[(d, c)], list) and isinstance(ptx[(d, None)], list) and digest[(d, c)] != digest[(d, None)]]
    print(f"{len(dirs)} problems; {len(jobs)} (problem, rewrite) pairs whose PTX changes", flush=True)

    def race(job):
        d, c = job
        try:
            times = {"full": [], "off": []}
            ok = True
            for side in ("full", "off", "off", "full"):
                t, good = run_problem(d, ptx[(d, None if side == "full" else c)], args.reps)
                times[side].append(t)
                ok &= good
            return {"id": d.name, "rewrite": c, "full_us": min(times["full"]), "off_us": min(times["off"]), "ok": ok}
        except Exception as e:  # noqa: BLE001
            return {"id": d.name, "rewrite": c, "error": str(e)[:200]}

    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(race, jobs))
    if args.json:
        Path(args.json).write_text(json.dumps({"problems": [d.name for d in dirs], "rows": rows}, indent=1))

    ids = [d.name for d in dirs]
    print(f"\n{'rewrite':12} {'changes':>7} {'off/full (changed)':>19} {'(all)':>7} {'(families)':>11}  "
          f"{'by category (changed)':40}  most lost without it")
    for c in configs:
        mine = [r for r in rows if r["rewrite"] == c and "error" not in r and r["full_us"] > 0]
        ratio = {r["id"]: r["off_us"] / r["full_us"] for r in mine}
        all_ratios = [ratio.get(i, 1.0) for i in ids]
        fam = defaultdict(list)
        for i in ids:
            fam[family(i)].append(math.log(ratio.get(i, 1.0)))
        fam_mean = math.exp(sum(sum(v) / len(v) for v in fam.values()) / len(fam))
        cats = defaultdict(list)
        for i, x in ratio.items():
            cats[category(i)].append(x)
        by_cat = " ".join(f"{k} {geomean(v):.2f}" for k, v in sorted(cats.items()))
        worst = max(ratio.items(), key=lambda kv: kv[1], default=None)
        wrong = [r["id"] for r in mine if not r["ok"]]
        print(f"{c:12} {len(mine):7} {geomean(ratio.values()):19.3f} {geomean(all_ratios):7.3f} {fam_mean:11.3f}  "
              f"{by_cat:40}  {f'{worst[0]} {worst[1]:.2f}x' if worst else ''}"
              f"{'  WRONG: ' + ','.join(wrong) if wrong else ''}")
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"{len(errors)} errors, e.g. {errors[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
