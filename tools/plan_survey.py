#!/usr/bin/env python3
"""The planner's choices (emmy_ptx/planner.py) against what the lowering does, over problem sets.

    python tools/plan_survey.py PROBLEMS_DIR [...] [--json out.json]

For every loop of every kernel the planner names the scope it pays to share
the loop's repeated work at (warp, block, cluster, or none); the lowering's
remarks say what its recognisers did with that loop: split it across lanes or
blocked it (warp), spread its cell over a block or merged warps (block), ran
it in a cluster (cluster), or nothing. The table counts loops by both; the
loops with the most saving left to the planner's estimate and nothing done
are listed: work the separate recognisers do not reach.

The two agree on a loop when the planner shares it exactly when the lowering
does, and also when the lowering shares a loop the planner would not because
of a loop inside it that the planner shares (lane blocking shares attention's
key loop for the scores inside it): both are counted, apart.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_suite import problem_dirs  # noqa: E402

SCOPES = ("none", "warp", "block", "cluster")
DONE = (*SCOPES, "idle")


def survey(ir: str) -> list[dict]:
    from emmy_ptx import planner
    from emmy_ptx.frontend import load
    from emmy_ptx.lower import Lowerer

    src = load(ir)
    remarks = Lowerer(src).run().remarks
    tile_level, idle = "none", False
    for r in remarks:
        if r.startswith("clusters of"):
            tile_level = "cluster"
        elif m := re.match(r"cells: idle threads join them, (\d+) lanes per cell", r):
            tile_level, idle = ("block" if int(m.group(1)) > 32 else "warp"), True
        elif r.startswith("cells: each row's cells moved"):
            tile_level = "warp"
    def shared(loop: str) -> str:  # what the lowering did with ``loop`` itself
        for r in remarks:
            if m := re.match(rf"loop {re.escape(loop)} \(\d+ trips\): reduction split over (\d+) lanes", r):
                return "block" if int(m.group(1)) > 32 else "warp"
            if re.match(rf"loop {re.escape(loop)} \(\d+ trips\): (lane-blocked|independent iterations dealt)", r):
                return "warp"
        return "none"

    rows, seen = [], set()
    for p in planner.plans(src.body, src.launch.block, src.launch.grid):
        if (p.loop, p.key) in seen:
            continue
        seen.add((p.loop, p.key))
        # a loop is shared if it or a loop around it is split, blocked or dealt across lanes
        done = next((d for d in map(shared, (p.loop, *reversed(p.outer))) if d != "none"), "none")
        if done == "none" and tile_level != "none":
            done = tile_level  # a cell-level re-mapping (placement, spreading, a cluster) shares the Tile's loops
        if p.best == "none" and done != "none" and idle:
            done = "idle"  # idle threads joined the cell: spare lanes put to work, not repeated work removed
        rows.append({"problem": Path(ir).parent.name, "kernel": Path(ir).name, "loop": p.loop, "outer": list(p.outer),
                     "planned": p.best, "done": done, "saving": p.net.get(p.best, 0.0), "cost": p.cost,
                     "lanes": p.lanes, "repeats": p.repeats})
    return rows


def for_inner_work(row: dict, rows: list[dict]) -> bool:
    """Does the lowering share loop ``row``, which the planner would not, for a
    loop inside it that the planner shares?"""
    return row["planned"] == "none" and row["done"] not in ("none", "idle") and any(
        r["kernel"] == row["kernel"] and row["loop"] in r["outer"] and r["planned"] != "none" for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    irs = [str(ir) for d in problem_dirs(args.paths) for ir in sorted(d.glob("*.ir.json"))]
    with ProcessPoolExecutor(8) as pool:
        rows = [r for part in pool.map(survey, irs) for r in part]
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    table = Counter((r["planned"], r["done"]) for r in rows)
    print(f"{len(irs)} kernels, {len(rows)} loops")
    print(f"{'planned / done':>16} " + " ".join(f"{s:>8}" for s in DONE))
    for p in SCOPES:
        print(f"{p:>16} " + " ".join(f"{table[(p, d)]:>8}" for d in DONE))
    agree = sum(v for (p, d), v in table.items() if (p == "none") == (d in ("none", "idle")))
    inner = sum(for_inner_work(r, rows) for r in rows)
    print(f"share or not as the planner says: {agree} of {len(rows)} loops (idle: cells spread over spare threads), "
          f"and {inner} more the lowering shares for a loop inside them that the planner shares: {agree + inner}")
    lanes = Counter(r["lanes"] for r in rows if r["planned"] != "none")
    print(f"repeating lanes in a warp, where sharing pays: {dict(lanes)}")
    left = sorted((r for r in rows if r["planned"] != "none" and r["done"] == "none"), key=lambda r: -r["saving"])
    print("largest estimated savings left unshared:")
    for r in left[:12]:
        print(f"  {r['problem']:36} loop {r['loop']:8} {r['planned']:8} saves ~{r['saving']:.0f} of {r['cost']:.0f} "
              f"instructions a warp ({r['repeats']} cells repeat it, lanes {r['lanes']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
