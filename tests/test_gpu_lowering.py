"""Hand-built Kernel IR for statement forms the generated problems rarely use,
run on the GPU and checked against a numpy model of their C semantics.

These need the runner: set EMMY_RUNNER (e.g. http://runner.riftstack.ai:48701)
to enable them. Each kernel runs twice: lowered as usual, and with the
fallback lowering (``Lowerer(plain=True)``).
"""

import os
import sys

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
PLAIN = False  # lower with the fallback lowering (set per test by `lowering`)


@pytest.fixture(autouse=True, params=[False, True], ids=["optimized", "plain"])
def lowering(request, monkeypatch):
    """Run each test once per lowering."""
    monkeypatch.setattr(sys.modules[__name__], "PLAIN", request.param)


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
    ptx = Lowerer(src, plain=PLAIN).run().render(PTX_VERSION, TARGET)
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


@pytest.mark.parametrize("cols", [256, 96])
def test_online_softmax_over_matmul_rows_is_split_across_lanes(cols):
    """softmax(x @ w)'s normalizer, the way emmy fuses it: every lane of a warp
    needs its row's running max and sum over all matmul columns. The compiler
    splits the column loop across the lanes (each lane computes whole dot
    products, reading w coalesced), merges the online-softmax states with the
    derived combine, and computes the two identical copies of the row's state
    (per output column, as emmy emits them) only once."""
    rows, k = 8, 64
    x = rng.standard_normal((rows, k)).astype(np.float32)
    w = rng.standard_normal((k, cols)).astype(np.float32)

    def normalizer(c):
        j, kk = Axis(f"j{c}", Dim(cols)), Axis(f"k{c}", Dim(k))
        return Loop(axis=j, body=(
            Loop(axis=kk, body=(
                Load(name=f"xv{c}", input="x", index=(Var("r"), Var(kk.name))),
                Load(name=f"wv{c}", input="w", index=(Var(kk.name), Var(j.name))),
                Assign(f"p{c}", "multiply", (f"xv{c}", f"wv{c}")),
                Accum(f"s{c}", f"p{c}"),
            )),
            Let(name=f"one{c}", value=Literal(1.0), dtype=F32),
            Assign(f"gn{c}", "maximum", (f"s{c}", f"m{c}")),
            Assign(f"dg{c}", "subtract", (f"m{c}", f"gn{c}")),
            Assign(f"alpha{c}", "exp", (f"dg{c}",)),
            Assign(f"dgo{c}", "subtract", (f"s{c}", f"gn{c}")),
            Assign(f"beta{c}", "exp", (f"dgo{c}",)),
            Assign(f"sa{c}", "multiply", (f"alpha{c}", f"l{c}")),
            Assign(f"sb{c}", "multiply", (f"beta{c}", f"one{c}")),
            Accum(f"l{c}", f"sb{c}", base=f"sa{c}"),
            Accum(f"m{c}", f"s{c}", op="maximum"),
        ))

    body = [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(32))), body=(
        normalizer(0),
        Write("out_m", (Var("r"), Var("c")), "m0"),
        normalizer(1),
        Assign("lsum", "add", ("l0", "l1")),
        Write("out_l", (Var("r"), Var("c")), "lsum"),
    ))]
    got = run(body, {"x": x, "w": w}, {"out_m": (rows, 32), "out_l": (rows, 32)}, 1, 256)
    s = x.astype(np.float64) @ w.astype(np.float64)
    m = s.max(1)
    l = np.exp(s - m[:, None]).sum(1)
    assert np.allclose(got["out_m"], np.repeat(m[:, None], 32, 1), rtol=1e-5, atol=1e-5)
    assert np.allclose(got["out_l"], np.repeat(2 * l[:, None], 32, 1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("keys", [44, 64])
def test_attention_is_blocked_across_the_lanes_of_a_row(keys):
    """softmax(q @ kt + mask) @ v as emmy fuses it: each lane holds one query
    row and one output column, so the 8 lanes of a row compute the same
    scores. Lane blocking computes each score on one lane and broadcasts its
    weight (with 44 keys the last block is partial, and its idle lanes must
    add nothing); masked keys fill whole blocks too."""
    rows, d, cols = 8, 16, 8
    q = rng.standard_normal((rows, d)).astype(np.float32)
    kt = rng.standard_normal((d, keys)).astype(np.float32)
    v = rng.standard_normal((keys, cols)).astype(np.float32)
    mask = np.zeros(keys, np.float32)
    mask[[3, 9, 16, 17, 18, 19, 20, 21, 22, 23, 40]] = -1e4  # keys 16..23: one whole block
    t, k = Axis("t", Dim(keys)), Axis("k", Dim(d))
    attention = Loop(axis=t, body=(
        Loop(axis=k, body=(
            Load(name="qk", input="q", index=(Var("r"), Var("k"))),
            Load(name="kk", input="kt", index=(Var("k"), Var("t"))),
            Assign("p", "multiply", ("qk", "kk")),
            Accum("dot", "p"),
        )),
        Load(name="mk", input="mask", index=(Var("t"),)),
        Let(name="scale", value=Literal(0.25), dtype=F32),
        Assign("s0", "multiply", ("dot", "scale")),
        Assign("s", "add", ("s0", "mk")),
        Let(name="one", value=Literal(1.0), dtype=F32),
        Assign("gn", "maximum", ("s", "m")),
        Assign("dg", "subtract", ("m", "gn")),
        Assign("alpha", "exp", ("dg",)),
        Assign("dgo", "subtract", ("s", "gn")),
        Assign("beta", "exp", ("dgo",)),
        Assign("la", "multiply", ("alpha", "l")),
        Assign("lb", "multiply", ("beta", "one")),
        Accum("l", "lb", base="la"),
        Assign("oa", "multiply", ("o", "alpha")),
        Load(name="vv", input="v", index=(Var("t"), Var("c"))),
        Assign("ob", "multiply", ("vv", "beta")),
        Accum("o", "ob", base="oa"),
        Accum("m", "s", op="maximum"),
    ))
    body = [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(cols))), body=(
        attention,
        Assign("inv", "reciprocal", ("l",)),
        Assign("res", "multiply", ("o", "inv")),
        Write("out", (Var("r"), Var("c")), "res"),
    ))]
    got = run(body, {"q": q, "kt": kt, "v": v, "mask": mask}, {"out": (rows, cols)}, 1, rows * cols)["out"]
    s = (q.astype(np.float64) @ kt) * 0.25 + mask
    p = np.exp(s - s.max(1, keepdims=True))
    want = (p / p.sum(1, keepdims=True)) @ v
    assert np.allclose(got, want, rtol=1e-4, atol=1e-5)


def test_few_cells_spread_over_idle_threads():
    """Emmy's thread-serial schedules: 4 cells (rows) on 256 threads, each
    folding a 3001-element row and then storing the row divided by its sum.
    Each cell runs on 32 lanes: the row sum is split across them (rolled
    rounds of 4-element vectors, then the odd tail) and the stores are dealt."""
    rows, n = 4, 3001
    x = rng.standard_normal((rows, n)).astype(np.float32) + 2
    body = [Tile(axes=(Axis("r", Dim(rows)),), body=(
        Loop(axis=Axis("k", Dim(n)), body=(
            Load(name="xv", input="x", index=(Var("r"), Var("k"))),
            Accum("total", "xv"),
        )),
        Write("sums", (Var("r"),), "total"),
        Assign("inv", "reciprocal", ("total",)),
        Loop(axis=Axis("j", Dim(n)), body=(
            Load(name="xj", input="x", index=(Var("r"), Var("j"))),
            Assign("yj", "multiply", ("xj", "inv")),
            Write("out", (Var("r"), Var("j")), "yj"),
        )),
    ))]
    got = run(body, {"x": x}, {"sums": (rows,), "out": (rows, n)}, 1, 256)
    total = x.astype(np.float64).sum(1)
    assert np.allclose(got["sums"], total, rtol=1e-5)
    assert np.allclose(got["out"], x / total[:, None], rtol=1e-5, atol=1e-7)


def test_one_cell_max_on_a_whole_block():
    n = 70000
    x = rng.standard_normal(n).astype(np.float32)
    body = [Tile(axes=(), body=(
        Loop(axis=Axis("k", Dim(n)), body=(
            Load(name="xv", input="x", index=(Var("k"),)),
            Accum("top", "xv", op="maximum"),
        )),
        Write("out", (), "top"),
    ))]
    assert run(body, {"x": x}, {"out": (1,)}, 1, 256)["out"][0] == x.max()


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
