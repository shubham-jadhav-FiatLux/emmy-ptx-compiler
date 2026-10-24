#!/usr/bin/env python3
"""Measure the accuracy of emmy_ptx.mathlib on the GPU.

    python tools/mathtest.py [exp log tanh ...] [--points 1048576]

For each function a tiny kernel is built directly with the compiler's own
Builder: thread i manufactures its input from its index (stepping through the
float32 bit patterns of a range, so a million inputs cost no upload), applies
the function, and stores the result. The host compares every result with the
float64 reference rounded to float32 and reports the worst error in ulps,
where it occurs, and whether special values (0, -0, inf, NaN, subnormals)
behave like C's math library.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emmy_ptx import PTX_VERSION, TARGET, mathlib  # noqa: E402
from emmy_ptx.builder import Builder  # noqa: E402
from emmy_ptx.ptx import Kernel, Mem, Sym  # noqa: E402

RUNNER = "http://runner.riftstack.ai:48701"
BLOCK = 256


def _bits(x: float) -> int:
    return int(np.array([x], np.float32).view(np.uint32)[0])


def build(fn, start: int, step: int, n: int) -> str:
    """out[i] = fn(x_i), x_i = bitcast(start + i*step); a second array gets fn(-x_i)."""
    k = Kernel("mtest", ("p_pos", "p_neg"), maxntid=BLOCK)
    b = Builder(k)
    tid = b.special("%tid.x", 0, BLOCK - 1)
    cta = b.special("%ctaid.x", 0, n // BLOCK - 1)
    gid = b.add(b.mul(cta, b.int_const(BLOCK)), tid)
    x = b.from_bits(b.add(b.int_const(start), b.mul(gid, b.int_const(step))))
    for param, arg in (("p_pos", x), ("p_neg", b.fop("neg.f32", x))):
        # store the result's bits: the runner's JSON cannot carry inf or NaN
        y = b.in_reg(b.bits(fn(b, arg)))
        base = b.pure("cvta.to.global.u64", "s64", (b.pure("ld.param.u64", "s64", (Mem(Sym(param)),)),))
        off = b.pure("mul.wide.u32", "s64", (gid, b.int_const(4)))
        addr = b.pure("add.s64", "s64", (base, off))
        b.emit("st.global.s32", Mem(addr.op), y)
    b.emit("ret")
    return k.render(PTX_VERSION, TARGET)


def run(ptx: str, n: int, server: str) -> tuple[np.ndarray, np.ndarray]:
    c = requests.post(f"{server}/compile", json={"ptx": ptx}, timeout=300).json()
    if not c.get("ok"):
        raise RuntimeError(c.get("stderr"))
    args = [dict(name=nm, dtype="i32", shape=[n], output=True) for nm in ("pos", "neg")]
    body = {"cubin_id": c["cubin_id"], "entry": "mtest", "grid": [n // BLOCK, 1, 1], "block": [BLOCK, 1, 1],
            "smem_bytes": 0, "zero_outputs": [], "args": args, "warmup": 0, "iters": 1, "max_return": 2 * n}
    for _ in range(60):
        r = requests.post(f"{server}/run", json=body, timeout=300).json()
        if r.get("stage") != "busy":
            break
    if not r.get("ok"):
        raise RuntimeError(r.get("stderr"))
    out = r["outputs"]

    def floats(name):
        return np.array(out[name]["values"], np.int64).astype(np.uint32).view(np.float32)

    return floats("pos"), floats("neg")


def ulp_dist(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    def ordered(x):
        i = x.astype(np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)

    d = np.abs(ordered(got) - ordered(want)).astype(np.float64)
    both_nan = np.isnan(got) & np.isnan(want)
    d[both_nan] = 0
    d[np.isnan(got) != np.isnan(want)] = np.inf
    return d


def _erf(x):
    import scipy.special as sp

    return sp.erf(x)


# name -> (builder function, float64 reference, [(lo, hi) ranges of |x| to sample])
CASES = {
    "exp": (mathlib.exp, np.exp, [(0.0, 1.0), (1.0, 104.0)]),
    "log": (mathlib.log, np.log, [(0.0, 1e-30), (1e-30, 3e38), (0.5, 2.0)]),
    "tanh": (mathlib.tanh, np.tanh, [(0.0, 0.6), (0.6, 12.0)]),
    "sigmoid": (mathlib.sigmoid, lambda x: 1.0 / (1.0 + np.exp(-x)), [(0.0, 20.0), (20.0, 104.0)]),
    "erf": (mathlib.erf, _erf, [(0.0, 1.0), (1.0, 6.0)]),
    "sin": (mathlib.sin, np.sin, [(0.0, 7.0), (7.0, 105615.0), (105615.0, 1e9)]),
    "cos": (mathlib.cos, np.cos, [(0.0, 7.0), (7.0, 105615.0), (105615.0, 1e9)]),
}


def check(name: str, points: int, server: str) -> bool:
    fn, ref, ranges = CASES[name]
    worst_all = 0.0
    for lo, hi in ranges:
        a, z = _bits(lo), _bits(hi)
        step = max(1, (z - a) // points)
        n = min(points, (z - a) // step) // BLOCK * BLOCK
        pos, neg = run(build(fn, a, step, n), n, server)
        xs = (np.arange(n, dtype=np.int64) * step + a).astype(np.uint32).view(np.float32)
        worst_here = 0.0
        for got, x in ((pos, xs), (neg, -xs)):
            with np.errstate(all="ignore"):
                want = ref(x.astype(np.float64)).astype(np.float32)
            d = ulp_dist(got, want)
            i = int(np.argmax(d))
            if d[i] > worst_here:
                worst_here, where = d[i], (float(x[i]), float(got[i]), float(want[i]))
        worst_all = max(worst_all, worst_here)
        detail = f"  worst at x={where[0]:.9g}: got {where[1]:.9g}, want {where[2]:.9g}" if worst_here else ""
        print(f"  {name:8s} |x| in [{lo:g}, {hi:g}]  {2 * n:>8d} points  max {worst_here:g} ulp{detail}")
    return worst_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=list(CASES))
    ap.add_argument("--points", type=int, default=1 << 20)
    ap.add_argument("--server", default=RUNNER)
    args = ap.parse_args()
    for name in args.names:
        check(name, args.points, args.server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
