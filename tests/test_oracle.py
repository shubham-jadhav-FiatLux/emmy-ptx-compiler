"""The compiler against an independent interpreter (tests/oracle.py).

First the interpreter itself is checked on small kernels against hand-written
numpy. Then randomly generated kernels, in three families that reach the
compiler's rewrites, are compiled optimized, strictly and plainly, run on the
GPU (set EMMY_RUNNER) and compared with the interpreter:

* typed pointwise expressions over float32, int32 and uint32 inputs, with
  constants such as -0.0 and 2^24 + 1 (every lowering must match exactly);
* reductions that several lanes share, folded with add, maximum or minimum
  (strict exactly; the others reassociate sums, so within rounding);
* cooperative reductions: a strided loop over the block's threads, then a warp
  butterfly and a tree over the warps whose combine may differ from the
  per-lane fold, sometimes with a read of the tree's shared memory afterwards
  (exactly where nothing sums, else within rounding);
* attention as emmy fuses it (an online softmax over keys, the lanes of a row
  sharing its scores, so lane blocking applies), with odd key counts and
  masks (within the rounding of exp, in every lowering).
"""

from __future__ import annotations

import os

import numpy as np
import oracle
import pytest
import requests
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, F64, I32, U32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, StridedLoop, Write
from emmy.compiler.ir.stmt.leaves import Init, Let

from emmy_ptx import PTX_VERSION, TARGET
from emmy_ptx.frontend import Buffer, KernelSource, Launch
from emmy_ptx.lower import Lowerer

RUNNER = os.environ.get("EMMY_RUNNER")
gpu = pytest.mark.skipif(not RUNNER, reason="set EMMY_RUNNER to run GPU tests")
MODES = {"optimized": {}, "strict": {"strict": True}, "plain": {"plain": True}}
DTYPE = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64", np.dtype(np.int32): "i32",
         np.dtype(np.uint32): "u32"}


def I(v):  # noqa: E743 - an int literal
    return Literal(v, "int")


def B(op, a, b):
    return BinaryExpr(op, a, b)


# ---------------------------------------------------------------- the interpreter itself


def test_oracle_tree_leaves_its_slab_and_broadcasts_the_root():
    t = Var("t")
    warps = B(">>", t, I(5))
    body = [Tile(axes=(Axis("t", Dim(128)),), block_threads=128, body=(
        Smem("partials", (4,), "float"), Init("s", 0.0, dtype=F32), Let(name="v", value=B("+", warps, I(1))),
        Cond(B("==", B("&", t, I(31)), I(0)), (Write("partials", (warps,), "v"),)), Sync(),
        TreeHalve(bufs=("partials",), state=("s",), state_b=("sb",),
                  combine_states=(Assign("s", "add", ("s", "sb")),), length=4, tid_var="t"),
        Load(name="slot", input="partials", index=(B("&", t, I(3)),)),
        Write("out", (I(0), t), "s"), Write("out", (I(1), t), "slot"),
    ))]
    out = oracle.run(body, {"out": np.zeros((2, 128), np.float32)}, 1, 128)["out"]
    assert np.array_equal(out[0], np.full(128, 10, np.float32))
    assert np.array_equal(out[1], np.tile(np.array([10, 6, 3, 4], np.float32), 32))


def test_oracle_butterfly_takes_the_maximum_of_per_lane_sums():
    x = np.zeros(128, np.float32)
    x[:4] = 1
    t = Var("t")
    body = [Tile(axes=(Axis("t", Dim(32)),), block_threads=32, body=(
        StridedLoop(axis=Axis("i", Dim(128)), start=t, step=I(32), body=(
            Load(name="v", input="x", index=(Var("i"),)), Accum("s", "v"))),
        WarpShuffle(state=("s",), state_b=("sb",), combine_states=(Assign("s", "maximum", ("s", "sb")),), length=32),
        Write("out", (t,), "s"),
    ))]
    out = oracle.run(body, {"x": x, "out": np.zeros(32, np.float32)}, 1, 32)["out"]
    assert np.array_equal(out, np.ones(32, np.float32))


def test_oracle_integers_are_numpys():
    a = np.array([-7, 7, 2**31 - 1, -2**31], np.int32)
    u = np.array([0xFFFFFFFF, 7, 0x80000000, 3], np.uint32)
    i = Var("i")
    body = [Tile(axes=(Axis("i", Dim(4)),), body=(
        Load(name="a", input="xa", index=(i,), dtype=I32), Load(name="u", input="xu", index=(i,), dtype=U32),
        Let(name="three", value=I(3)), Let(name="one", value=I(1)),
        Assign("fd", "floor_divide", ("a", "three"), dtype=I32), Assign("wrap", "add", ("a", "one"), dtype=I32),
        Assign("sh", "right_shift", ("u", "one"), dtype=U32), Assign("eq", "equal", ("a", "a")),
        Write("out", (I(0), i), "fd"), Write("out", (I(1), i), "wrap"), Write("out", (I(2), i), "sh"),
        Write("out", (I(3), i), "eq"),
    ))]
    out = oracle.run(body, {"xa": a, "xu": u, "out": np.zeros((4, 4), np.float64)}, 1, 4)["out"]
    with np.errstate(over="ignore"):
        want = [a.astype(np.int64) // 3, (a + np.int32(1)).astype(np.int64), u.astype(np.int64) >> 1, np.ones(4)]
    assert np.array_equal(out, np.array(want, np.float64))


# ---------------------------------------------------------------- random kernels


def pointwise(rng):
    """A random typed expression over f32, i32 and u32 inputs; every value it makes is written out (as f64)."""
    n = 256
    xs = rng.standard_normal(n).astype(np.float32) * 10
    xs[:6] = [0.0, -0.0, 3e30, -3e30, 1e-40, 16777217.0]
    inputs = {"xf": xs, "yf": rng.standard_normal(n).astype(np.float32),
              "xi": rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32),
              "yi": np.where(rng.random(n) < 0.5, rng.integers(-50, 50, n), 16777217).astype(np.int32),
              "xu": rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)}
    inputs["yi"][inputs["yi"] == 0] = 3
    i = Var("i")
    stmts = [Load(name="f0", input="xf", index=(i,)), Load(name="f1", input="yf", index=(i,)),
             Load(name="i0", input="xi", index=(i,), dtype=I32), Load(name="i1", input="yi", index=(i,), dtype=I32),
             Load(name="u0", input="xu", index=(i,), dtype=U32),
             Let(name="nz", value=Literal(-0.0), dtype=F32), Let(name="big", value=I(16777217)),
             Let(name="two", value=I(2))]
    names = {"f": ["f0", "f1", "nz"], "i": ["i0", "i1", "big", "two"], "u": ["u0"]}
    ops = [("f", "add", 2), ("f", "subtract", 2), ("f", "multiply", 2), ("f", "maximum", 2), ("f", "minimum", 2),
           ("f", "abs", 1), ("f", "negative", 1), ("f", "reciprocal", 1), ("i", "add", 2), ("i", "multiply", 2),
           ("i", "floor_divide", 2), ("i", "remainder", 2), ("i", "bitwise_xor", 2), ("i", "maximum", 2),
           ("u", "right_shift", 2), ("u", "floor_divide", 2), ("u", "remainder", 2)]
    made = []
    for k in range(8):
        kind, op, arity = ops[rng.integers(len(ops))]
        if kind == "u":
            args = ("u0", "two")
        elif op in ("floor_divide", "remainder"):
            args = (str(rng.choice(names["i"][:3])), "i1")  # a divisor that is never 0
        else:
            args = tuple(str(rng.choice(names[kind])) for _ in range(arity))
        dtype = {"f": F32, "i": I32, "u": U32}[kind]
        name = f"v{k}"
        stmts.append(Assign(name, op, args, dtype=dtype))
        names[kind].append(name)
        made.append(name)
        if rng.random() < 0.4:  # a comparison of two values of one type, kept as its 0/1
            kind2 = str(rng.choice(["f", "i"]))
            cmp_op = str(rng.choice(["equal", "not_equal", "less", "greater_equal"]))
            stmts.append(Assign(f"c{k}", cmp_op, tuple(str(rng.choice(names[kind2])) for _ in range(2))))
            stmts.append(Assign(f"w{k}", "where", (f"c{k}", "f0", "f1"), dtype=F32))
            made += [f"c{k}", f"w{k}"]
    stmts += [Write("out", (I(j), i), name) for j, name in enumerate(made)]
    body = [Tile(axes=(Axis("i", Dim(n)),), body=tuple(stmts))]
    return body, inputs, {"out": np.zeros((len(made), n), np.float64)}, 1, n, "exact"


def shared_reduction(rng):
    """Rows whose fold `cols` lanes share (a lane split), folded with add, maximum or minimum."""
    cols = int(rng.choice([4, 8, 16, 32]))
    rows, trips = 256 // cols, int(rng.choice([24, 50, 64, 97]))
    op = str(rng.choice(["add", "maximum", "minimum"]))
    inputs = {"x": rng.standard_normal((rows, trips)).astype(np.float32),
              "y": rng.standard_normal(trips).astype(np.float32), "z": rng.standard_normal(cols).astype(np.float32)}
    body = [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(cols))), body=(
        Loop(axis=Axis("k", Dim(trips)), body=(
            Load(name="xv", input="x", index=(Var("r"), Var("k"))), Load(name="yv", input="y", index=(Var("k"),)),
            Assign("p", "multiply", ("xv", "yv")), Accum("acc", "p", op=op))),
        Load(name="zc", input="z", index=(Var("c"),)), Assign("o", "add", ("acc", "zc")),
        Write("out", (Var("r"), Var("c")), "o"),
    ))]
    return body, inputs, {"out": np.zeros((rows, cols), np.float64)}, 1, 256, "exact" if op != "add" else "sums"


def cooperative(rng):
    """A strided loop over the block's threads, then a butterfly and a tree over the warps; the
    cross-thread combine may differ from the per-thread fold, and the tree's slab may be read after."""
    block = int(rng.choice([32, 64, 128, 256]))
    n = block * 4 * int(rng.integers(1, 5))
    fold, combine = str(rng.choice(["add", "maximum"])), str(rng.choice(["add", "maximum", "minimum"]))
    peek = block > 32 and rng.random() < 0.5
    inputs = {"x": rng.standard_normal(n).astype(np.float32)}
    t = Var("t")
    stmts = [StridedLoop(axis=Axis("i", Dim(n)), start=t, step=I(block), body=(
                 Load(name="v", input="x", index=(Var("i"),)), Accum("s", "v", op=fold))),
             WarpShuffle(state=("s",), state_b=("sb",), combine_states=(Assign("s", combine, ("s", "sb")),),
                         length=min(block, 32))]
    outputs = {"out": np.zeros(block, np.float64)}
    if block > 32:
        warps = block // 32
        stmts += [Let(name="warp", value=B(">>", t, I(5))), Smem("slab", (warps,), "float"),
                  Cond(B("==", B("&", t, I(31)), I(0)), (Write("slab", (Var("warp"),), "s"),)), Sync(),
                  TreeHalve(bufs=("slab",), state=("s",), state_b=("sb",),
                            combine_states=(Assign("s", combine, ("s", "sb")),), length=warps, tid_var="warp")]
        if peek:
            stmts += [Load(name="left", input="slab", index=(B("&", t, I(warps - 1)),)), Write("peek", (t,), "left")]
            outputs["peek"] = np.zeros(block, np.float64)
    stmts.append(Write("out", (t,), "s"))
    body = [Tile(axes=(Axis("t", Dim(block)),), block_threads=block, body=tuple(stmts))]
    return body, inputs, outputs, 1, block, "exact" if "add" not in (fold, combine) else "sums"


def attention(rng):
    """softmax(q @ kt * scale + mask) @ v, emmy's online-softmax form: each lane holds one query row and one
    output column, so the lanes of a row compute the same scores (lane blocking shares them)."""
    rows, cols = int(rng.choice([4, 8, 16])), int(rng.choice([4, 8, 16, 32]))
    d, keys = int(rng.choice([8, 16, 32])), int(rng.choice([16, 33, 44, 64, 100]))
    mask = np.where(rng.random(keys) < 0.2, -1e4, 0.0).astype(np.float32)
    inputs = {"q": rng.standard_normal((rows, d)).astype(np.float32),
              "kt": rng.standard_normal((d, keys)).astype(np.float32),
              "v": rng.standard_normal((keys, cols)).astype(np.float32), "mask": mask}
    scale = float(np.float32(1 / np.sqrt(d)))
    loop = Loop(axis=Axis("t", Dim(keys)), body=(
        Loop(axis=Axis("k", Dim(d)), body=(
            Load(name="qk", input="q", index=(Var("r"), Var("k"))), Load(name="kk", input="kt", index=(Var("k"), Var("t"))),
            Assign("p", "multiply", ("qk", "kk")), Accum("dot", "p"))),
        Load(name="mk", input="mask", index=(Var("t"),)), Let(name="scale", value=Literal(scale), dtype=F32),
        Assign("s0", "multiply", ("dot", "scale")), Assign("s", "add", ("s0", "mk")),
        Let(name="one", value=Literal(1.0), dtype=F32),
        Assign("gn", "maximum", ("s", "m")), Assign("dg", "subtract", ("m", "gn")), Assign("alpha", "exp", ("dg",)),
        Assign("dgo", "subtract", ("s", "gn")), Assign("beta", "exp", ("dgo",)),
        Assign("la", "multiply", ("alpha", "l")), Assign("lb", "multiply", ("beta", "one")), Accum("l", "lb", base="la"),
        Assign("oa", "multiply", ("o", "alpha")), Load(name="vv", input="v", index=(Var("t"), Var("c"))),
        Assign("ob", "multiply", ("vv", "beta")), Accum("o", "ob", base="oa"), Accum("m", "s", op="maximum"),
    ))
    body = [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(cols))), body=(
        loop, Assign("inv", "reciprocal", ("l",)), Assign("res", "multiply", ("o", "inv")),
        Write("out", (Var("r"), Var("c")), "res"),
    ))]
    return body, inputs, {"out": np.zeros((rows, cols), np.float64)}, 1, rows * cols, "approx"


def compile_and_run(body, inputs, outputs, grid, block, options) -> dict:
    buffers = {n: Buffer(n, a.shape, DTYPE[a.dtype]) for n, a in inputs.items()}
    buffers.update({n: Buffer(n, a.shape, "f64") for n, a in outputs.items()})
    src = KernelSource("k_random", tuple(body), tuple(inputs) + tuple(outputs), buffers, Launch(grid, block, 0),
                       frozenset(outputs))
    ptx = Lowerer(src, **options).run().render(PTX_VERSION, TARGET)
    c = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=120).json()
    assert c.get("ok"), c.get("stderr")
    args = [dict(name=n, dtype=buffers[n].dtype.replace("u32", "i32"), shape=list(a.shape),
                 data=(a.view(np.int32) if a.dtype == np.uint32 else a).ravel().tolist()) for n, a in inputs.items()]
    args += [dict(name=n, dtype="f64", shape=list(a.shape), output=True) for n, a in outputs.items()]
    r = requests.post(f"{RUNNER}/run", json={
        "cubin_id": c["cubin_id"], "entry": "k_random", "grid": [grid, 1, 1], "block": [block, 1, 1],
        "smem_bytes": 0, "zero_outputs": [], "args": args, "warmup": 0, "iters": 1, "max_return": 1 << 16},
        timeout=120).json()
    assert r.get("ok"), r.get("stderr")
    # (the runner returns infinities and NaN as null)
    return {n: np.array([np.nan if v is None else v for v in r["outputs"][n]["values"]], np.float64).reshape(a.shape)
            for n, a in outputs.items()}


def agree(got: np.ndarray, want: np.ndarray, kind: str) -> bool:
    want = np.where(np.isfinite(want), want, np.nan)  # what the runner can return
    if kind == "exact":
        same = (got == want) | (np.isnan(got) & np.isnan(want))
        return bool(np.all(same))
    tol = 1e-4 if kind == "approx" else 1e-5  # (approx: exp within its 2 ulp, then sums)
    return bool(np.allclose(got, want, rtol=tol, atol=tol / 10, equal_nan=True))


@gpu
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("family,seed", [(f, s) for f in ("pointwise", "shared_reduction", "cooperative", "attention")
                                         for s in range(6)])
def test_random_kernels_match_the_independent_interpreter(family, seed, mode):
    rng = np.random.default_rng(1000 * seed + len(family))
    body, inputs, outputs, grid, block, kind = globals()[family](rng)
    want = oracle.run(body, {**{k: v.copy() for k, v in inputs.items()}, **{k: v.copy() for k, v in outputs.items()}},
                      grid, block)
    got = compile_and_run(body, inputs, outputs, grid, block, MODES[mode])
    if mode == "strict" and kind == "sums":
        kind = "exact"  # the IR's own order of operations: bit for bit
    for name in outputs:
        assert agree(got[name], want[name], kind), (family, seed, mode, name)
