#!/usr/bin/env python3
"""Compile problem folders with compile.py, run them on the runner, check them.

    python tools/run_suite.py PROBLEMS_DIR [PROBLEMS_DIR ...] [--jobs 4] [--bench] [--strict] [--plain] [--bulk-copy]

Each argument is a problem folder (with meta.json) or a directory of them.
Every kernel is compiled exactly the way the evaluator does it (a fresh
``python compile.py <ir>`` process) and launched with meta.json's geometry;
outputs are compared with expected.npz under the problem's rtol/atol. The
table also shows how close each result came to its tolerance ("margin": the
largest |got-ref| / (atol + rtol*|ref|), so <= 1 passes) and the worst error
in ulps, which says far more than PASS/FAIL about numerical health.
``--strict``, ``--plain`` and ``--bulk-copy`` are passed on to compile.py.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "compile.py"
RUNNER = "http://runner.riftstack.ai:48701"


def post(server: str, path: str, body: dict) -> dict:
    for attempt in range(120):
        try:
            r = requests.post(f"{server}{path}", json=body, timeout=(10, 300))
            r.raise_for_status()
            out = r.json()
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2 + attempt % 5)
            continue
        if out.get("stage") != "busy":
            return out
        time.sleep(2)
    return {"ok": False, "stage": "busy", "stderr": "runner stayed busy"}


def ulp_error(got: np.ndarray, ref: np.ndarray) -> float:
    """Largest distance in float32 ulps (finite values only)."""
    g, r = got.astype(np.float32).ravel(), ref.astype(np.float32).ravel()
    ok = np.isfinite(g) & np.isfinite(r)
    if not ok.any():
        return 0.0

    def ordered(x):
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)

    return float(np.max(np.abs(ordered(g[ok]) - ordered(r[ok]))))


def run_problem(pdir: Path, server: str, bench: bool, flags: tuple = ()) -> dict:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz")) if (pdir / "expected.npz").exists() else {}
    row = {"id": pdir.name, "ok": True, "us": 0.0, "regs": [], "compile_s": 0.0, "margin": 0.0, "ulp": 0.0, "err": ""}
    produced: dict[str, list] = {}
    for k in meta["kernels"]:
        ir = pdir / f"{k['entry']}.ir.json"
        t0 = time.time()
        cmd = [sys.executable, str(COMPILER), *flags, str(ir)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        row["compile_s"] += time.time() - t0
        if r.returncode != 0:
            return {**row, "ok": False, "err": "compile: " + (r.stderr.strip().splitlines() or ["?"])[-1][:160]}
        c = post(server, "/compile", {"ptx": r.stdout})
        if not c.get("ok"):
            return {**row, "ok": False, "err": "ptxas: " + c.get("stderr", "")[:300]}
        row["regs"].append(c.get("registers", {}).get(k["entry"]))
        args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            args.append(a)
        need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
        res = post(server, "/run", {
            "cubin_id": c["cubin_id"], "entry": k["entry"], "grid": k["grid"], "block": k["block"],
            "smem_bytes": k["smem_bytes"], "zero_outputs": k["zero_outputs"], "args": args,
            "warmup": 3 if bench else 0, "iters": 20 if bench else 1, "max_return": max(4096, need),
        })
        if not res.get("ok"):
            return {**row, "ok": False, "err": f"run ({res.get('stage')}): " + res.get("stderr", "")[:300]}
        row["us"] += float(res.get("median_us") or 0.0)
        for name, out in res["outputs"].items():
            produced[name] = out["values"]
            if name not in expected:
                continue
            ref = expected[name]
            got = np.array(out["values"], dtype=np.float32).reshape(ref.shape)
            tol = meta["atol"] + meta["rtol"] * np.abs(ref.astype(np.float64))
            diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            bad = ~np.isfinite(got) & np.isfinite(ref)
            margin = float(np.nanmax(np.where(tol > 0, diff / np.maximum(tol, 1e-300), np.where(diff > 0, np.inf, 0.0))))
            row["margin"] = max(row["margin"], margin if not bad.any() else math.inf)
            row["ulp"] = max(row["ulp"], ulp_error(got, ref))
            if not np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"]):
                row["ok"] = False
                worst = int(np.nanargmax(diff)) if diff.size else 0
                row["err"] = (f"{name}: max abs err {float(np.nanmax(diff)):.3e} at {np.unravel_index(worst, ref.shape)} "
                              f"(got {got.ravel()[worst]:.6g}, want {ref.ravel()[worst]:.6g})")
    return row


def problem_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "meta.json").exists():
            out.append(p)
        else:
            out.extend(sorted(d for d in p.iterdir() if (d / "meta.json").exists()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--server", default=RUNNER)
    ap.add_argument("--bench", action="store_true", help="time 20 launches after 3 warm-ups")
    ap.add_argument("--json", help="also write the rows to this file")
    ap.add_argument("--strict", action="store_true", help="compile.py --strict: no rewrite that changes rounding")
    ap.add_argument("--plain", action="store_true", help="compile.py --plain: the fallback lowering")
    ap.add_argument("--bulk-copy", action="store_true", help="compile.py --bulk-copy: fills through the copy engine")
    args = ap.parse_args()

    dirs = problem_dirs(args.paths)
    with ThreadPoolExecutor(args.jobs) as pool:
        flags = tuple(f for f, on in (("--strict", args.strict), ("--plain", args.plain),
                                      ("--bulk-copy", args.bulk_copy)) if on)
        rows = list(pool.map(lambda d: _safe(d, args.server, args.bench, flags), dirs))
    print(f"{'problem':<34} {'result':<6} {'us':>9} {'regs':>8} {'margin':>8} {'ulp':>8} {'compile':>8}")
    for r in rows:
        regs = "/".join(str(x) for x in r["regs"]) or "-"
        print(f"{r['id']:<34} {'PASS' if r['ok'] else 'FAIL':<6} {r['us']:>9.1f} {regs:>8} "
              f"{r['margin']:>8.3f} {r['ulp']:>8.0f} {r['compile_s']:>7.2f}s")
        if r["err"]:
            print(f"    {r['err']}")
    passed = sum(r["ok"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0 if passed == len(rows) else 1


def _safe(d: Path, server: str, bench: bool, flags: tuple) -> dict:
    try:
        return run_problem(d, server, bench, flags)
    except Exception as e:  # noqa: BLE001 - one broken problem must not stop the suite
        return {"id": d.name, "ok": False, "us": 0.0, "regs": [], "compile_s": 0.0, "margin": 0.0, "ulp": 0.0,
                "err": f"harness: {type(e).__name__}: {e}"}


if __name__ == "__main__":
    sys.exit(main())
