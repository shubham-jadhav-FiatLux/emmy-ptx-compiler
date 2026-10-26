#!/usr/bin/env python3
"""Do two ways of compiling give bit-identical outputs on the runner?

    python tools/bitwise.py PROBLEMS_DIR [...] [--a "--strict"] [--b "--plain --strict"] [--json OUT]

Each problem is compiled with compile.py under both flag sets (by default
``--strict`` and ``--plain --strict``), launched, and its outputs compared
bit for bit. A problem whose kernels add floats atomically is counted apart:
the order of float atomics, and so their rounding, varies from run to run.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_suite import COMPILER, RUNNER, post, problem_dirs  # noqa: E402


def outputs(pdir: Path, flags: list[str], server: str) -> dict:
    """Every output of the problem's kernels, run in meta.json's order."""
    meta = json.loads((pdir / "meta.json").read_text())
    produced: dict[str, list] = {}
    for k in meta["kernels"]:
        cmd = [sys.executable, str(COMPILER), *flags, str(pdir / f"{k['entry']}.ir.json")]
        ptx = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            args.append(a)
        need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
        r = post(server, "/run", {"ptx": ptx, "entry": k["entry"], "grid": k["grid"], "block": k["block"],
                                  "smem_bytes": k["smem_bytes"], "zero_outputs": k["zero_outputs"], "args": args,
                                  "warmup": 0, "iters": 1, "max_return": max(4096, need)})
        if not r.get("ok"):
            raise RuntimeError(f"{k['entry']}: {r.get('stage')}: {str(r.get('stderr'))[:200]}")
        produced.update((name, o["values"]) for name, o in r["outputs"].items())
    return produced


def compare(pdir: Path, a: list[str], b: list[str], server: str) -> dict:
    meta = json.loads((pdir / "meta.json").read_text())
    atomic = any(k["zero_outputs"] for k in meta["kernels"]) or any(
        "atomic=True" in (pdir / f"{k['entry']}.ir.json").read_text() for k in meta["kernels"])
    try:
        x, y = outputs(pdir, a, server), outputs(pdir, b, server)
    except Exception as e:  # noqa: BLE001 - one broken problem must not stop the comparison
        return {"id": pdir.name, "error": str(e)}
    return {"id": pdir.name, "same": all(json.dumps(x[n]) == json.dumps(y.get(n)) for n in x), "atomic": atomic}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--a", default="--strict", help="compile.py flags of the first build")
    ap.add_argument("--b", default="--plain --strict", help="compile.py flags of the second build")
    ap.add_argument("--server", default=RUNNER)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--json", help="also write the rows to this file")
    args = ap.parse_args()
    a, b = args.a.split(), args.b.split()
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda d: compare(d, a, b, args.server), problem_dirs(args.paths)))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    differ = [r for r in rows if "error" in r or (not r["same"] and not r["atomic"])]
    for r in differ:
        print(r)
    print(f"{len(rows)} problems: {sum(r.get('same', False) for r in rows)} bit-identical, "
          f"{sum('same' in r and not r['same'] and not r['atomic'] for r in rows)} differ without float atomics, "
          f"{sum('same' in r and not r['same'] and r['atomic'] for r in rows)} differ with them, "
          f"{sum('error' in r for r in rows)} errors")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
