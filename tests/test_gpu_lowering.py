"""Hand-built Kernel IR for statement forms the generated problems rarely use,
run on the GPU and checked against a numpy model of their C semantics.

These need the runner: set EMMY_RUNNER (e.g. http://runner.riftstack.ai:48701)
to enable them.
"""

import os

import numpy as np
import pytest
import requests

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Select, SelectBranch, StridedLoop, Write
from emmy.compiler.ir.stmt.leaves import Init, Let

from emmy_ptx import PTX_VERSION, TARGET
from emmy_ptx.frontend import Buffer, KernelSource, Launch
from emmy_ptx.lower import Lowerer

RUNNER = os.environ.get("EMMY_RUNNER")
pytestmark = pytest.mark.skipif(not RUNNER, reason="set EMMY_RUNNER to run GPU tests")


def I(v):  # noqa: E743 - an int literal
    return Literal(v, "int")


def B(op, a, b):
    return BinaryExpr(op, a, b)


def run(body, inputs: dict, outputs: dict, grid: int, block: int) -> dict:
    """Compile ``body`` as a kernel and run it; inputs/outputs map names to arrays/shapes."""
    buffers = {n: Buffer(n, a.shape, "f32" if a.dtype == np.float32 else "i32") for n, a in inputs.items()}
    buffers.update({n: Buffer(n, s, "f32") for n, s in outputs.items()})
    params = tuple(inputs) + tuple(outputs)
    src = KernelSource("k_test", tuple(body), params, buffers, Launch(grid, block, 0), frozenset(outputs))
    ptx = Lowerer(src).run().render(PTX_VERSION, TARGET)
    c = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=120).json()
    assert c.get("ok"), c.get("stderr")
    args = [dict(name=n, dtype=buffers[n].dtype, shape=list(a.shape), data=a.ravel().tolist()) for n, a in inputs.items()]
    args += [dict(name=n, dtype="f32", shape=list(s), output=True) for n, s in outputs.items()]
    r = requests.post(f"{RUNNER}/run", json={
        "cubin_id": c["cubin_id"], "entry": "k_test", "grid": [grid, 1, 1], "block": [block, 1, 1], "smem_bytes": 0,
        "zero_outputs": [], "args": args, "warmup": 0, "iters": 1, "max_return": 1 << 16}, timeout=120).json()
    assert r.get("ok"), r.get("stderr")
    return {n: np.array(r["outputs"][n]["values"], np.float32).reshape(s) for n, s in outputs.items()}


rng = np.random.default_rng(7)


def test_cond_with_else_branch():
    x = rng.standard_normal(256).astype(np.float32)
    body = [Tile(axes=(Axis("a0", Dim(256)),), body=(
        Load(name="x0", input="x", index=(Var("a0"),)),
        Let(name="two", value=Literal(2.0), dtype=F32),
        Cond(cond=B("<", Var("a0"), I(100)),
             body=(Assign("v", "multiply", ("x0", "two")), Write("out", (Var("a0"),), "v")),
             else_body=(Assign("w", "add", ("x0", "two")), Write("out", (Var("a0"),), "w"))),
    ))]
    got = run(body, {"x": x}, {"out": (256,)}, 1, 256)["out"]
    want = np.where(np.arange(256) < 100, x * np.float32(2), x + np.float32(2))
    assert np.array_equal(got, want)


def test_strided_loop_with_runtime_end_and_end_name():
    x = rng.standard_normal((64, 64)).astype(np.float32)
    a1 = Axis("a1", Dim(64))
    body = [Tile(axes=(Axis("a0", Dim(64)),), body=(
        StridedLoop(axis=a1, start=I(0), step=I(1), end=B("+", Var("a0"), I(1)), body=(
            Load(name="in0", input="x", index=(Var("a0"), Var("a1"))),
            Accum("acc", "in0"),
            # the hoisted bound is visible as a1_end: count the trips
            Load(name="one", input="ones", index=(B("-", Var("a1_end"), Var("a0")),)),
            Accum("cnt", "one"),
        )),
        Write("out", (Var("a0"),), "acc"),
        Write("trips", (Var("a0"),), "cnt"),
    ))]
    got = run(body, {"x": x, "ones": np.ones(4, np.float32)}, {"out": (64,), "trips": (64,)}, 1, 64)
    want = np.array([x[r, : r + 1].astype(np.float64).sum() for r in range(64)], np.float32)
    assert np.allclose(got["out"], want, rtol=1e-5, atol=1e-5)
    assert np.array_equal(got["trips"], np.arange(1, 65, dtype=np.float32))


def test_select_let_cast_and_ternary_index():
    x = rng.standard_normal(200).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)
    a0 = Var("a0")
    body = [Tile(axes=(Axis("a0", Dim(100)),), body=(
        Let(name="j", value=TernaryExpr(B("<", a0, I(50)), B("*", a0, I(2)), B("-", I(199), a0))),
        Load(name="xj", input="x", index=(Var("j"),)),
        Load(name="y0", input="y", index=(a0,)),
        Select("s", (SelectBranch("xj", B("<", a0, I(30))), SelectBranch("y0", B(">=", a0, I(30))))),
        Let(name="half", value=CastExpr("int", B("/", CastExpr("float", a0), Literal(2.0)))),
        Load(name="yh", input="y", index=(Var("half"),)),
        Assign("r", "add", ("s", "yh")),
        Write("out", (a0,), "r"),
    ))]
    got = run(body, {"x": x, "y": y}, {"out": (100,)}, 1, 128)["out"]
    i = np.arange(100)
    j = np.where(i < 50, 2 * i, 199 - i)
    want = np.where(i < 30, x[j], y) + y[(i / 2.0).astype(int)]
    assert np.array_equal(got, want.astype(np.float32))


@pytest.mark.parametrize("named", [False, True])
def test_segmented_tree_halve_with_inner_slot(named):
    """slot = seg*4 + inner: every thread ends with the sum over segments of its inner slot."""
    x = rng.standard_normal(128).astype(np.float32)
    t = Var("t")
    sync = Sync(barrier_id=1, count=128) if named else Sync()
    body = [Tile(axes=(Axis("t", Dim(128)),), block_threads=128, body=(
        Load(name="x0", input="x", index=(t,)),
        Smem("buf", (128,)),
        Write("buf", (t,), "x0"),
        sync,
        Let(name="seg", value=B("/", t, I(4))),
        Let(name="inn", value=B("%", t, I(4))),
        Init("acc", 0.0, dtype=F32),
        TreeHalve(bufs=("buf",), state=("acc",), state_b=("acc__o",),
                  combine_states=(Assign("acc", "add", ("acc", "acc__o")),), length=32, tid_var="seg",
                  barrier_id=1 if named else 0, barrier_count=128 if named else None, inner=("inn", 4)),
        Write("out", (t,), "acc"),
    ))]
    got = run(body, {"x": x}, {"out": (128,)}, 1, 128)["out"]
    want = x.reshape(32, 4).astype(np.float64).sum(0)[np.arange(128) % 4]
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_segmented_warp_shuffle():
    x = rng.standard_normal(64).astype(np.float32)
    t = Var("t")
    body = [Tile(axes=(Axis("t", Dim(64)),), block_threads=64, body=(
        Load(name="x0", input="x", index=(t,)),
        Init("acc", 0.0, dtype=F32),
        Accum("acc", "x0"),
        WarpShuffle(state=("acc",), state_b=("acc__o",), combine_states=(Assign("acc", "add", ("acc", "acc__o")),), length=8),
        Write("out", (t,), "acc"),
    ))]
    got = run(body, {"x": x}, {"out": (64,)}, 1, 64)["out"]
    want = np.repeat(x.reshape(8, 8).astype(np.float64).sum(1), 8)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cols,trips", [(16, 50), (32, 64), (32, 33), (8, 24)])
def test_reduction_shared_by_lanes_is_split_across_them(cols, trips):
    """Every lane of a `cols`-lane group needs the same dot product of row r;
    the compiler splits its iterations across the group (including an uneven
    last round) and combines the partials with shuffles."""
    rows = 256 // cols
    x = rng.standard_normal((rows, trips)).astype(np.float32)
    y = rng.standard_normal(trips).astype(np.float32)
    z = rng.standard_normal(cols).astype(np.float32)
    k = Axis("k", Dim(trips))
    body = [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(cols))), body=(
        Loop(axis=k, body=(
            Load(name="xv", input="x", index=(Var("r"), Var("k"))),
            Load(name="yv", input="y", index=(Var("k"),)),
            Assign("p", "multiply", ("xv", "yv")),
            Accum("acc", "p"),
            Accum("big", "xv", op="maximum"),
        )),
        Load(name="zc", input="z", index=(Var("c"),)),
        Assign("o", "add", ("acc", "zc")),
        Assign("o2", "add", ("o", "big")),
        Write("out", (Var("r"), Var("c")), "o2"),
    ))]
    got = run(body, {"x": x, "y": y, "z": z}, {"out": (rows, cols)}, 1, 256)["out"]
    want = (x.astype(np.float64) @ y)[:, None] + z[None, :] + x.max(1)[:, None]
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_integer_elementwise_ops():
    a = rng.integers(-1000, 1000, 256).astype(np.int32)
    b = rng.integers(1, 50, 256).astype(np.int32)
    body = [Tile(axes=(Axis("a0", Dim(256)),), body=(
        Load(name="p", input="a", index=(Var("a0"),)),
        Load(name="q", input="b", index=(Var("a0"),)),
        Assign("s", "floor_divide", ("p", "q")),
        Assign("m", "remainder", ("p", "q")),
        Assign("x", "bitwise_xor", ("p", "q")),
        Assign("u", "add", ("s", "m")),
        Assign("v", "add", ("u", "x")),
        Write("out", (Var("a0"),), "v"),
    ))]
    got = run(body, {"a": a, "b": b}, {"out": (256,)}, 1, 256)["out"]
    s = np.trunc(a / b).astype(np.int64)  # C semantics, as emmy prints `//`
    m = a - s * b
    want = (s + m + (a ^ b)).astype(np.float32)
    assert np.array_equal(got, want)
