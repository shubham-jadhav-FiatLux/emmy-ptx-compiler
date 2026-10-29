#!/usr/bin/env python3
"""How much machine code a loop can run from the runner's instruction cache.

    python tools/icache.py [--threads 256] [--json out.json]

One block runs a loop whose body is N fused multiply-adds in eight
independent chains (16 bytes of machine code each, so the body is about N/64
KB), as many trips as make every warp execute about two million instructions.
While the body fits the cache, the time per instruction stays at the issue
rate; past it every trip fetches its code again and the time rises. On the
RTX 5090 a body of up to 112 KB runs at 0.70 ns an instruction a warp, one of
128 KB at 0.84 and one of 160 KB or more at 1.50 (docs/results/
icache_2026-10-29.json). Code that runs once, such as copies of a loop, does
not pay this: the sizes where it matters are those of a loop's own body.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite import RUNNER, post  # noqa: E402

SIZES = (256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384)
WORK = 2_000_000  # instructions every warp executes, whatever the body's size


def ptx(n: int) -> str:
    """A loop of ``n`` FMAs over eight registers, its trip count read from memory."""
    body = "\n".join(f"\tfma.rn.f32 %f{i % 8 + 1}, %f{i % 8 + 1}, %fa, %fb;" for i in range(n))
    return f""".version 8.0
.target sm_89
.address_size 64
.visible .entry icache(.param .u64 p_in, .param .u64 p_out)
{{
\t.reg .f32 %f<9>; .reg .f32 %fa; .reg .f32 %fb; .reg .f32 %fr;
\t.reg .u32 %r<5>; .reg .u64 %rd<6>; .reg .pred %p;
\tld.param.u64 %rd1, [p_in]; cvta.to.global.u64 %rd2, %rd1;
\tld.param.u64 %rd3, [p_out]; cvta.to.global.u64 %rd4, %rd3;
\tld.global.f32 %fa, [%rd2]; ld.global.f32 %fb, [%rd2+4]; ld.global.f32 %fr, [%rd2+8];
\tcvt.rzi.u32.f32 %r1, %fr;
\tmov.u32 %r2, %tid.x; cvt.rn.f32.u32 %f1, %r2;
\tmov.f32 %f2, %f1; mov.f32 %f3, %f1; mov.f32 %f4, %f1; mov.f32 %f5, %f1;
\tmov.f32 %f6, %f1; mov.f32 %f7, %f1; mov.f32 %f8, %f1;
\tmov.u32 %r3, 0;
$loop:
{body}
\tadd.u32 %r3, %r3, 1; setp.lt.u32 %p, %r3, %r1; @%p bra $loop;
\tadd.f32 %f1, %f1, %f2; add.f32 %f3, %f3, %f4; add.f32 %f5, %f5, %f6; add.f32 %f7, %f7, %f8;
\tadd.f32 %f1, %f1, %f3; add.f32 %f5, %f5, %f7; add.f32 %f1, %f1, %f5;
\tmul.wide.u32 %rd5, %r2, 4; add.s64 %rd5, %rd4, %rd5; st.global.f32 [%rd5], %f1;
\tret;
}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = []
    for n in SIZES:
        trips = max(1, WORK // n)
        c = post(RUNNER, "/compile", {"ptx": ptx(n)})
        if not c.get("ok"):
            raise RuntimeError("ptxas: " + c.get("stderr", "")[:300])
        best = None
        for _ in range(2):  # the runner's clocks are not locked: the best of two runs
            r = post(RUNNER, "/run", {
                "cubin_id": c["cubin_id"], "entry": "icache", "grid": [1, 1, 1], "block": [args.threads, 1, 1],
                "smem_bytes": 0, "zero_outputs": [], "warmup": 2, "iters": 10, "max_return": 16,
                "args": [dict(name="in", dtype="f32", shape=[4], data=[1.0000001, 1e-7, float(trips), 0.0]),
                         dict(name="out", dtype="f32", shape=[args.threads], output=True)]})
            if not r.get("ok"):
                raise RuntimeError("run: " + r.get("stderr", "")[:300])
            best = r["median_us"] if best is None else min(best, r["median_us"])
        ns = best * 1000 / (n * trips)
        rows.append({"fmas": n, "body_kb": n * 16 / 1024, "trips": trips, "median_us": best, "ns_per_instruction": ns})
        print(f"{n:6d} FMAs  {n * 16 / 1024:6.0f} KB  {trips:5d} trips  {best:9.1f} us  {ns:.3f} ns an instruction a warp",
              flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
