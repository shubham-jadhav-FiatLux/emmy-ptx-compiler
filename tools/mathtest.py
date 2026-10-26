#!/usr/bin/env python3
"""Measure the accuracy of emmy_ptx.mathlib on the GPU.

    python tools/mathtest.py [exp log tanh ...] [--points 1048576]

For each function a tiny kernel is built directly with the compiler's own
Builder: thread i manufactures its input from its index (stepping through the
float32 bit patterns of a range, so a million inputs cost no upload), applies
the function, and stores the result. The host compares every result with the
float64 reference rounded to float32 and reports the worst error in ulps,
where it occurs, and whether special values (0, -0, inf, NaN, subnormals)
behave like C's math library. The float64 functions (``exp64`` ...) step
through float64 bit patterns and are compared with an x87 extended-precision
reference rounded to float64; their results travel as two 32-bit words (the
runner's JSON carries an i64 as a double, which would round its low bits).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emmy_ptx import PTX_VERSION, TARGET, mathlib  # noqa: E402
from emmy_ptx.builder import Builder  # noqa: E402
from emmy_ptx.ptx import Kernel, Mem, Sym, Vec  # noqa: E402

RUNNER = "http://runner.riftstack.ai:48701"
BLOCK = 256


def _bits(x: float, double: bool = False) -> int:
    if double:
        return int(np.array([x], np.float64).view(np.uint64)[0])
    return int(np.array([x], np.float32).view(np.uint32)[0])


def build(fn, start: int, step: int, n: int, double: bool = False) -> str:
    """out[i] = fn(x_i), x_i = bitcast(start + i*step); a second array gets fn(-x_i)."""
    k = Kernel("mtest", ("p_pos", "p_neg"), maxntid=BLOCK)
    b = Builder(k)
    tid = b.special("%tid.x", 0, BLOCK - 1)
    cta = b.special("%ctaid.x", 0, n // BLOCK - 1)
    gid = b.add(b.mul(cta, b.int_const(BLOCK)), tid)
    if double:
        bits = b.add(b.int_const(start, "s64"), b.mul(b.convert(gid, "s64"), b.int_const(step, "s64")))
        x = b.pure("mov.b64", "f64", (bits,))
        cases = (("p_pos", x), ("p_neg", b.fop("neg.f64", x, type_="f64")))
    else:
        x = b.from_bits(b.add(b.int_const(start), b.mul(gid, b.int_const(step))))
        cases = (("p_pos", x), ("p_neg", b.fop("neg.f32", x)))
    for param, arg in cases:
        # store the result's bits: the runner's JSON cannot carry inf or NaN
        r = b.in_reg(fn(b, arg))
        base = b.pure("cvta.to.global.u64", "s64", (b.pure("ld.param.u64", "s64", (Mem(Sym(param)),)),))
        off = b.pure("mul.wide.u32", "s64", (gid, b.int_const(8 if double else 4)))
        addr = b.pure("add.s64", "s64", (base, off))
        if double:
            words = Vec((k.reg("s32"), k.reg("s32")))
            b.emit("mov.b64", words, r, ndefs=1, effect=False)
            b.emit("st.global.v2.b32", Mem(addr.op), words)
        else:
            b.emit("st.global.s32", Mem(addr.op), b.in_reg(b.bits(r)))
    b.emit("ret")
    return k.render(PTX_VERSION, TARGET)


def run(ptx: str, n: int, server: str, double: bool = False) -> tuple[np.ndarray, np.ndarray]:
    c = requests.post(f"{server}/compile", json={"ptx": ptx}, timeout=300).json()
    if not c.get("ok"):
        raise RuntimeError(c.get("stderr"))
    words = 2 * n if double else n
    args = [dict(name=nm, dtype="i32", shape=[words], output=True) for nm in ("pos", "neg")]
    body = {"cubin_id": c["cubin_id"], "entry": "mtest", "grid": [n // BLOCK, 1, 1], "block": [BLOCK, 1, 1],
            "smem_bytes": 0, "zero_outputs": [], "args": args, "warmup": 0, "iters": 1, "max_return": 2 * words}
    for _ in range(60):
        r = requests.post(f"{server}/run", json=body, timeout=300).json()
        if r.get("stage") != "busy":
            break
    if not r.get("ok"):
        raise RuntimeError(r.get("stderr") or str(r)[:300])
    out = r["outputs"]

    def floats(name):
        words = np.array(out[name]["values"], np.int64).astype(np.uint32)
        return words.view(np.float64) if double else words.view(np.float32)

    return floats("pos"), floats("neg")


def ulp_dist(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    def ordered(x):
        if x.dtype == np.float64:  # as exact Python ints: the int64 difference may overflow
            i = x.view(np.int64).astype(object)
            return np.where(i < 0, -(i & 0x7FFFFFFFFFFFFFFF), i)
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
    # float64, against long double
    "exp64": (mathlib.exp64, np.exp, [(0.0, 1.0), (1.0, 708.0), (708.0, 746.0)]),
    "log64": (mathlib.log64, np.log, [(0.0, 1e-300), (1e-300, 1e308), (0.5, 2.0)]),
    # (below -709.78 emmy's spelling overflows e^-x to inf and gives 0, as numpy's float64 does)
    "sigmoid64": (mathlib.sigmoid64, lambda x: 1 / (1 + np.exp(-x)), [(0.0, 40.0), (40.0, 709.0)]),
}


def check(name: str, points: int, server: str) -> bool:
    fn, ref, ranges = CASES[name]
    double = name.endswith("64")
    if double:  # two words a result, and the runner returns about 2M values at most
        points = min(points, 1 << 19)
    worst_all = 0.0
    for lo, hi in ranges:
        a, z = _bits(lo, double), _bits(hi, double)
        step = max(1, (z - a) // points)
        n = min(points, (z - a) // step) // BLOCK * BLOCK
        pos, neg = run(build(fn, a, step, n, double), n, server, double)
        if double:
            xs = (np.arange(n, dtype=np.int64) * step + a).view(np.float64)
        else:
            xs = (np.arange(n, dtype=np.int64) * step + a).astype(np.uint32).view(np.float32)
        worst_here = 0.0
        for got, x in ((pos, xs), (neg, -xs)):
            with np.errstate(all="ignore"):
                if double:
                    want = ref(x.astype(np.longdouble)).astype(np.float64)
                else:
                    want = ref(x.astype(np.float64)).astype(np.float32)
            d = ulp_dist(got, want)
            i = int(np.argmax(d))
            if d[i] > worst_here:
                worst_here, where = d[i], (float(x[i]), float(got[i]), float(want[i]))
        worst_all = max(worst_all, worst_here)
        digits = 17 if double else 9
        detail = (f"  worst at x={where[0]:.{digits}g}: got {where[1]:.{digits}g}, want {where[2]:.{digits}g}"
                  if worst_here else "")
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
