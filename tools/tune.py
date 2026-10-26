#!/usr/bin/env python3
"""Measure compiler heuristics against each other on real problems (offline tuning).

    python tools/tune.py PROBLEM_DIR [...] --variant NAME:module.ATTR=value[;module.ATTR=value] [...]

Every variant overrides some of the compiler's tuning constants (for example
``lower.FUSE_BUDGET=1024`` or ``builder.Builder.HOIST_BUDGET=48``), compiles
each problem in-process, and times it on the runner (best of a few runs of
20 timed launches). A ``base`` variant with no overrides is always measured.
The table prints each variant's time relative to the base and flags wrong
results, so a heuristic is only kept when it is faster and still correct.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import emmy_ptx  # noqa: E402
from run_suite import RUNNER, post, problem_dirs  # noqa: E402


def parse_variant(spec: str):
    name, _, body = spec.partition(":")
    overrides = []
    for item in filter(None, body.split(";")):
        target, _, value = item.partition("=")
        module, _, attr = target.rpartition(".")
        overrides.append((module, attr, eval(value)))  # noqa: S307 - developer tool
    return name, overrides


@contextlib.contextmanager
def applied(overrides):
    saved = []
    for module, attr, value in overrides:
        owner = importlib.import_module(f"emmy_ptx.{module.split('.')[0]}")
        for part in module.split(".")[1:]:
            owner = getattr(owner, part)
        saved.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, value)
    try:
        yield
    finally:
        for owner, attr, value in reversed(saved):
            setattr(owner, attr, value)


def measure(pdir: Path, server: str, reps: int) -> tuple[float, bool]:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    total, ok, produced = 0.0, True, {}
    for k in meta["kernels"]:
        ptx = emmy_ptx.compile_file(str(pdir / f"{k['entry']}.ir.json"))
        c = post(server, "/compile", {"ptx": ptx})
        if not c.get("ok"):
            return math.inf, False
        args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            args.append(a)
        need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
        best = math.inf
        for _ in range(reps):
            r = post(server, "/run", {"cubin_id": c["cubin_id"], "entry": k["entry"], "grid": k["grid"],
                                      "block": k["block"], "smem_bytes": k["smem_bytes"],
                                      "zero_outputs": k["zero_outputs"], "args": args, "warmup": 3, "iters": 20,
                                      "max_return": max(4096, need)})
            if not r.get("ok"):
                return math.inf, False
            best = min(best, float(r["median_us"]))
            for name, out in r["outputs"].items():
                produced[name] = out["values"]
                if name in expected:
                    got = np.array(out["values"], np.float32).reshape(expected[name].shape)
                    ok &= bool(np.allclose(got, expected[name], rtol=meta["rtol"], atol=meta["atol"]))
        total += best
    return total, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--server", default=RUNNER)
    args = ap.parse_args()
    variants = [("base", [])] + [parse_variant(v) for v in args.variant]
    dirs = problem_dirs(args.paths)
    print(f"{'problem':<26}" + "".join(f"{name:>14}" for name, _ in variants))
    logs = {name: [] for name, _ in variants}
    for d in dirs:
        cells, base = [], None
        for name, overrides in variants:
            with applied(overrides):
                t, ok = measure(d, args.server, args.reps)
            base = t if base is None else base
            logs[name].append(t / base if base and math.isfinite(t) else math.nan)
            cells.append(f"{t:>9.1f}" + (f" {t / base:>4.2f}" if name != "base" else "     ") + ("" if ok else "!"))
        print(f"{d.name:<26}" + "".join(f"{c:>14}" for c in cells), flush=True)
    print(f"{'geomean vs base':<26}" + "".join(
        f"{math.exp(np.nanmean(np.log(v))):>14.3f}" if name != "base" else f"{'':>14}" for name, v in logs.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
