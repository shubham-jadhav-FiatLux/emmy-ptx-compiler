#!/usr/bin/env python3
"""Race our PTX against emmy's reference CUDA (nvcc) on the runner.

    python tools/ref_bench.py PROBLEMS_DIR [...] [--nvcc PATH] [--jobs 4]

For every problem folder that ships a ``*.ref.cu`` (the open set and folders
made by tools/make_corpus.py), the reference kernel is compiled to PTX with a
local nvcc (development use only; the compiler itself never calls it), both
versions are launched with identical arguments and 20 timed iterations, and
the table shows both times, both register counts and whether each output
matches expected.npz. This is how we find where our code generation lags.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_suite import COMPILER, RUNNER, post, problem_dirs  # noqa: E402


def nvcc_ptx(nvcc: str, cu: Path) -> str:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ref.ptx"
        r = subprocess.run([nvcc, "-ptx", "-arch=sm_89", "-std=c++17", "-allow-unsupported-compiler", "-O3",
                            str(cu), "-o", str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-400:])
        # the runner's ptxas (CUDA 12.9) reads PTX ISA <= 8.8; these kernels use nothing newer
        return re.sub(r"^\.version \d+\.\d+", ".version 8.8", out.read_text(), count=1, flags=re.M)


def launch(server: str, ptx: str, k: dict, produced: dict) -> tuple[dict, dict]:
    c = post(server, "/compile", {"ptx": ptx})
    if not c.get("ok"):
        raise RuntimeError("ptxas: " + c.get("stderr", "")[:300])
    args = []
    for a in k["args"]:
        a = dict(a)
        if a.pop("intermediate", False):
            a["data"] = produced[a["name"]]
        args.append(a)
    need = max((int(np.prod(a["shape"])) for a in k["args"] if a.get("output")), default=1)
    r = post(server, "/run", {"cubin_id": c["cubin_id"], "entry": k["entry"], "grid": k["grid"], "block": k["block"],
                              "smem_bytes": k["smem_bytes"], "zero_outputs": k["zero_outputs"], "args": args,
                              "warmup": 3, "iters": 20, "max_return": max(4096, need)})
    if not r.get("ok"):
        raise RuntimeError(f"run: {r.get('stderr', '')[:300]}")
    return c, r


def bench(pdir: Path, nvcc: str, server: str) -> dict:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    row = {"id": pdir.name, "ours_us": 0.0, "ref_us": 0.0, "ours_regs": [], "ref_regs": [], "ours_ok": True, "ref_ok": True}
    for side in ("ours", "ref"):
        produced: dict = {}
        for k in meta["kernels"]:
            if side == "ours":
                r = subprocess.run([sys.executable, str(COMPILER), str(pdir / f"{k['entry']}.ir.json")],
                                   capture_output=True, text=True, check=True)
                ptx = r.stdout
            else:
                ptx = nvcc_ptx(nvcc, pdir / f"{k['entry']}.ref.cu")
            c, res = launch(server, ptx, k, produced)
            row[f"{side}_regs"].append(c.get("registers", {}).get(k["entry"]))
            row[f"{side}_us"] += float(res.get("median_us") or 0)
            for name, out in res["outputs"].items():
                produced[name] = out["values"]
                if name in expected:
                    got = np.array(out["values"], np.float32).reshape(expected[name].shape)
                    row[f"{side}_ok"] &= bool(np.allclose(got, expected[name], rtol=meta["rtol"], atol=meta["atol"]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "nvcc")
    ap.add_argument("--server", default=RUNNER)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()
    dirs = [d for d in problem_dirs(args.paths) if list(d.glob("*.ref.cu"))]

    def safe(d):
        try:
            return bench(d, args.nvcc, args.server)
        except Exception as e:  # noqa: BLE001
            return {"id": d.name, "error": str(e)[:200]}

    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(safe, dirs))
    print(f"{'problem':<28} {'ours us':>9} {'nvcc us':>9} {'ratio':>6} {'regs ours/nvcc':>15}  correct ours/nvcc")
    for r in rows:
        if "error" in r:
            print(f"{r['id']:<28} error: {r['error']}")
            continue
        ratio = r["ours_us"] / r["ref_us"] if r["ref_us"] else float("nan")
        regs = f"{'/'.join(map(str, r['ours_regs']))} / {'/'.join(map(str, r['ref_regs']))}"
        print(f"{r['id']:<28} {r['ours_us']:>9.1f} {r['ref_us']:>9.1f} {ratio:>6.2f} {regs:>15}  "
              f"{'ok' if r['ours_ok'] else 'WRONG'}/{'ok' if r['ref_ok'] else 'WRONG'}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
