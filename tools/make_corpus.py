#!/usr/bin/env python3
"""Build problem folders from tools/corpus_specs.py with the organisers' generator.

    python tools/make_corpus.py --hackathon PATH/emmy-hackathon-2026 [--out corpus] [--jobs 4] [--only ID ...]

Runs ``tools/generate.py`` of the hackathon repository once per spec, several
at a time (each emmy trace takes ~20 s and ~400 MB). Specs whose folder
already exists are skipped unless ``--force``. Specs emmy cannot trace are
reported and skipped: that is useful information too.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hackathon", required=True, help="checkout of emmy-hackathon-2026 (for tools/generate.py)")
    ap.add_argument("--specs", default=str(HERE / "corpus_specs.py"))
    ap.add_argument("--emmy", default=shutil.which("emmy") or str(Path(sys.executable).with_name("emmy")))
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("corpus_specs", args.specs)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ids = [s["id"] for s in mod.OPEN if not args.only or s["id"] in args.only]
    out = Path(args.out)
    todo = [i for i in ids if args.force or not (out / i / "meta.json").exists()]
    generator = Path(args.hackathon) / "tools" / "generate.py"

    def one(pid: str) -> tuple[str, bool, str]:
        r = subprocess.run(
            [sys.executable, str(generator), "--emmy", args.emmy, "--out", str(out), "--only", pid, args.specs],
            capture_output=True, text=True,
        )
        line = next((ln.strip() for ln in r.stdout.splitlines() if pid in ln), r.stderr.strip()[-200:])
        return pid, r.returncode == 0, line

    print(f"{len(todo)} to generate ({len(ids) - len(todo)} already present)")
    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for pid, ok, line in pool.map(one, todo):
            print(("  " if ok else "! ") + line, flush=True)
            if not ok:
                failed.append(pid)
    print(f"done: {len(todo) - len(failed)} generated, {len(failed)} failed {failed if failed else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
