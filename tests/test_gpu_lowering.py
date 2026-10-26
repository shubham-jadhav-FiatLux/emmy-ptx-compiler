"""Hand-built Kernel IR for statement forms the generated problems rarely use,
run on the GPU and checked against a numpy model of their C semantics.

These need the runner: set EMMY_RUNNER (e.g. http://runner.riftstack.ai:48701)
to enable them. Each kernel runs three times: lowered as usual, strictly
(``Lowerer(strict=True)``: no rewrite that changes rounding) and with the
fallback lowering (``Lowerer(plain=True)``).
"""

import os
import sys

import numpy as np
import pytest
import requests
from kernels import rasterized_row_scale, shared_sum_added_atomically, three_column_attention

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32, F64
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (FragmentApply, FragmentPromote, LdmatrixLoad, MmaSyncPtx, RegFragment, RegStore,
                                       Smem, Sync, Tile, TreeHalve, WarpShuffle)
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Select, SelectBranch, StridedLoop, Write
from emmy.compiler.ir.stmt.leaves import Init, Let

from emmy_ptx import PTX_VERSION, TARGET
from emmy_ptx.frontend import Buffer, KernelSource, Launch
from emmy_ptx.lower import Lowerer

RUNNER = os.environ.get("EMMY_RUNNER")
pytestmark = pytest.mark.skipif(not RUNNER, reason="set EMMY_RUNNER to run GPU tests")
LOWERING = {}  # Lowerer options, set per test by `lowering`


@pytest.fixture(autouse=True, params=[{}, {"strict": True}, {"plain": True}], ids=["optimized", "strict", "plain"])
def lowering(request, monkeypatch):
    """Run each test once per lowering."""
    monkeypatch.setattr(sys.modules[__name__], "LOWERING", request.param)


def I(v):  # noqa: E743 - an int literal
    return Literal(v, "int")


def B(op, a, b):
    return BinaryExpr(op, a, b)


DTYPE = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64", np.dtype(np.float16): "f16",
         np.dtype(np.int32): "i32"}


def run(body, inputs: dict, outputs: dict, grid: int, block: int, out_dtype: str = "f32") -> dict:
    """Compile ``body`` as a kernel and run it; inputs/outputs map names to arrays/shapes."""
    buffers = {n: Buffer(n, a.shape, DTYPE[a.dtype]) for n, a in inputs.items()}
    buffers.update({n: Buffer(n, s, out_dtype) for n, s in outputs.items()})
    params = tuple(inputs) + tuple(outputs)
    src = KernelSource("k_test", tuple(body), params, buffers, Launch(grid, block, 0), frozenset(outputs))
    ptx = Lowerer(src, **LOWERING).run().render(PTX_VERSION, TARGET)
    c = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=120).json()
    assert c.get("ok"), c.get("stderr")
    args = [dict(name=n, dtype=buffers[n].dtype, shape=list(a.shape), data=a.ravel().tolist()) for n, a in inputs.items()]
    args += [dict(name=n, dtype=out_dtype, shape=list(s), output=True) for n, s in outputs.items()]
    r = requests.post(f"{RUNNER}/run", json={
        "cubin_id": c["cubin_id"], "entry": "k_test", "grid": [grid, 1, 1], "block": [block, 1, 1], "smem_bytes": 0,
        "zero_outputs": [], "args": args, "warmup": 0, "iters": 1, "max_return": 1 << 16}, timeout=120).json()
    assert r.get("ok"), r.get("stderr")
    numpy_type = {"f64": np.float64, "f16": np.float16}.get(out_dtype, np.float32)
    return {n: np.array(r["outputs"][n]["values"], numpy_type).reshape(s) for n, s in outputs.items()}


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


def test_float64_results_are_computed_in_double_precision():
    """The IEEE operations bit for bit like numpy's float64 (a float literal
    of an f64 Let included); exp, log and sigmoid within 2 ulp."""
    n = 256
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    ops = [("add", "x", "y"), ("subtract", "x", "y"), ("multiply", "x", "y"), ("divide", "x", "y"),
           ("abs", "y"), ("maximum", "x", "y"), ("reciprocal", "y"), ("multiply", "x", "tenth"),
           ("exp", "x"), ("log", "v0"), ("sigmoid", "x"), ("sqrt", "v4")]
    stmts = [Load(name="x", input="xs", index=(Var("i"),)), Load(name="y", input="ys", index=(Var("i"),)),
             Let(name="tenth", value=Literal(0.1), dtype=F64)]
    stmts += [Assign(f"v{k}", op, args, dtype=F64) for k, (op, *args) in enumerate(ops)]
    stmts += [Write("out", (I(k), Var("i")), f"v{k}") for k in range(len(ops))]
    body = [Tile(axes=(Axis("i", Dim(n)),), body=tuple(stmts))]
    got = run(body, {"xs": x, "ys": y}, {"out": (len(ops), n)}, 1, n, out_dtype="f64")["out"]
    exact = [x + y, x - y, x * y, x / y, abs(y), np.maximum(x, y), 1 / y, x * 0.1]
    assert np.array_equal(got[:8], np.array(exact))
    with np.errstate(invalid="ignore"):  # log(x + y) is NaN where x + y < 0
        want = np.array([np.exp(x), np.log(x + y), 1 / (1 + np.exp(-x)), np.sqrt(abs(y))])
    close = np.abs(got[8:] - want) <= 2 * np.spacing(np.abs(want))
    assert np.all(close | (np.isnan(got[8:]) & np.isnan(want)))


def test_float64_warp_sum_and_atomic_add():
    """A float64 butterfly moves both halves of each value; lane 0 of each warp
    then adds its warp's sum into one float64 output atomically."""
    x = rng.standard_normal(256)
    t = Var("t")
    body = [Tile(axes=(Axis("t", Dim(256)),), body=(
        Load(name="x0", input="x", index=(t,)),
        Init("acc", 0.0, dtype=F64),
        Accum("acc", "x0", dtype=F64),
        WarpShuffle(state=("acc",), state_b=("acc__o",), length=32, dtype=F64,
                    combine_states=(Assign("acc", "add", ("acc", "acc__o"), dtype=F64),)),
        Cond(cond=B("==", B("%", t, I(32)), I(0)), body=(Write("out", (I(0),), "acc", atomic=True),)),
    ))]
    got = run(body, {"x": x}, {"out": (1,)}, 1, 256, out_dtype="f64")["out"][0]
    assert abs(got - x.sum()) <= 1e-13 * np.abs(x).sum()


def test_cells_of_three_columns_are_placed_to_share_their_rows():
    """Attention with 11 cells of three columns to a row: placed 8 + 4 lanes a
    row (one lane of each 4 computes but does not store), each part blocks
    the row's keys."""
    rows, d, keys = 16, 16, 64
    q = rng.standard_normal((rows, d)).astype(np.float32)
    kt = rng.standard_normal((d, keys)).astype(np.float32)
    v = rng.standard_normal((keys, 32)).astype(np.float32)
    got = run(three_column_attention(rows, d, keys), {"q": q, "kt": kt, "v": v}, {"out": (rows, 32)}, 1, 192)["out"]
    s = q.astype(np.float64) @ kt
    p = np.exp(s - s.max(1, keepdims=True))
    assert np.allclose(got, (p / p.sum(1, keepdims=True)) @ v, rtol=1e-4, atol=1e-5)


def test_placed_lanes_that_only_help_add_nothing():
    """5 rows of 11 cells on 128 threads: 16 lanes a row, and the 5 lanes past
    each row's cells and the 16 past the last row must not add their copies."""
    x = rng.standard_normal((5, 64)).astype(np.float32)
    w = rng.standard_normal(11).astype(np.float32)
    got = run(shared_sum_added_atomically(5, 64, 11), {"x": x, "w": w}, {"out": (5,)}, 1, 128)["out"]
    assert np.allclose(got, x.astype(np.float64).sum(1) * w.astype(np.float64).sum(), rtol=1e-4, atol=1e-4)


def test_rasterized_cells_are_placed_to_share_their_rows():
    """Cells rasterized four rows at a time, so a row's lanes are four apart:
    placed 16 + 8 lanes a row (24 cells, exactly the 384 threads)."""
    x = rng.standard_normal((16, 24)).astype(np.float32)
    got = run(rasterized_row_scale(16, 24, 24, 4), {"x": x}, {"out": (16, 24)}, 3, 128)["out"]
    assert np.allclose(got, x * (x.astype(np.float64) ** 2).sum(1, keepdims=True), rtol=1e-5, atol=1e-5)


def _one_warp_mma(acc_dtype, tail: tuple) -> list:
    """One warp computing a 16x8 tile on the tensor cores: A (16x16) and B stored
    N x K (8x16) staged in shared memory by plain stores, ldmatrix x4 for A and
    x2 for B (no .trans: an N x K slab is already the mma's column-major B),
    one mma into ``acc``, then ``tail``."""
    lane, shape = Var("_lane"), (16, 8, 16)

    def at(cols):  # lane l stages row l / (16 / cols), ``cols`` columns from (l % (16 / cols)) * cols
        per = 16 // cols
        return lambda j: (B("/", lane, I(per)), B("+", B("*", B("%", lane, I(per)), I(cols)), I(j)))

    fill = []
    for name, src, cols in (("a_s", "a", 8), ("b_s", "bt", 4)):
        for j in range(cols):
            fill += [Load(name=f"{name}{j}", input=src, index=at(cols)(j)), Write(name, at(cols)(j), f"{name}{j}")]
    return [Tile(axes=(Axis("_lane", Dim(32)),), body=(
        Smem("a_s", (16, 16), "__half", 16), Smem("b_s", (8, 16), "__half", 16), *fill, Sync(),
        RegFragment("fa", "a", shape, F16), RegFragment("fb", "b", shape, F16), RegFragment("acc", "c", shape, acc_dtype),
        LdmatrixLoad("fa", "a_s", (I(0), I(0)), "a", ldm=16),
        LdmatrixLoad("fb", "b_s", (I(0), I(0)), "b", ldm=16, b_trans=True),
        MmaSyncPtx("acc", "fa", "fb", shape, c_dtype=acc_dtype.name),
        *tail,
    ))]


def _tile_inputs():
    a = (rng.standard_normal((16, 16)) / 4).astype(np.float16)
    bt = (rng.standard_normal((8, 16)) / 4).astype(np.float16)
    return a, bt, a.astype(np.float64) @ bt.astype(np.float64).T


def test_tensor_core_tile_with_coordinate_mask_bias_and_f16_accumulator():
    """An f16-accumulate mma folded into an f32 fragment, a mask from each
    element's coordinates (column > row becomes -1) and a bias read at them."""
    a, bt, prod = _tile_inputs()
    bias = rng.standard_normal((16, 8)).astype(np.float32)
    shape = (16, 8, 16)
    tail = (
        RegFragment("c", "c", shape, F32),
        FragmentPromote("c", "acc"),
        FragmentApply("c", ElementwiseImpl("where"), (B(">", Var("__fcol"), Var("__frow")), Literal(-1.0), "c"),
                      ("coord", "uniform", "frag"), in_place=True, row_base=I(0), col_base=I(0)),
        FragmentApply("c", ElementwiseImpl("add"), ("c", ("bias", (Var("__frow"), Var("__fcol")))), ("frag", "gmem"),
                      in_place=True, row_base=I(0), col_base=I(0)),
        RegStore("out", (I(0), I(0)), "c", shape),
    )
    got = run(_one_warp_mma(F16, tail), {"a": a, "bt": bt, "bias": bias}, {"out": (16, 8)}, 1, 32)["out"]
    rows, cols = np.indices((16, 8))
    want = np.where(cols > rows, -1.0, prod.astype(np.float16)) + bias
    assert np.allclose(got, want, rtol=2e-3, atol=2e-3)


def test_tensor_core_tile_added_atomically_as_f16_pairs():
    """The f32 tile stored twice into an f16 output with atomic adds (packed
    f16x2 pairs, the split-K finish)."""
    a, bt, prod = _tile_inputs()
    shape = (16, 8, 16)
    tail = (RegStore("out", (I(0), I(0)), "acc", shape, atomic=True), RegStore("out", (I(0), I(0)), "acc", shape, atomic=True))
    got = run(_one_warp_mma(F32, tail), {"a": a, "bt": bt}, {"out": (16, 8)}, 1, 32, out_dtype="f16")["out"]
    assert np.allclose(got.astype(np.float64), 2 * prod, rtol=5e-3, atol=5e-3)


def test_division_by_a_constant_is_correctly_rounded():
    """emmy spells x / c as x * reciprocal(c). Outside loops the optimized
    lowering computes the correctly rounded quotient that numpy's x / c gives
    (pointwise problems compare with rtol = 0); the strict and plain lowerings
    keep the IR's multiply by RN(1/c)."""
    n = 4096
    # (no quotient overflows: the runner returns infinities as null)
    x = np.concatenate([rng.standard_normal(n - 6), [0.0, -0.0, 1e-40, -3e37, 3e37, 1e30]]).astype(np.float32)
    consts = [3.0, 7.0, -0.1, 0.5]
    stmts = [Load(name="x", input="xs", index=(Var("i"),))]
    for k, c in enumerate(consts):
        stmts += [Let(name=f"c{k}", value=Literal(c), dtype=F32), Assign(f"r{k}", "reciprocal", (f"c{k}",)),
                  Assign(f"q{k}", "multiply", ("x", f"r{k}")), Write("out", (I(k), Var("i")), f"q{k}")]
    body = [Tile(axes=(Axis("i", Dim(n)),), body=tuple(stmts))]
    got = run(body, {"xs": x}, {"out": (len(consts), n)}, n // 256, 256)["out"]
    for k, c in enumerate(consts):
        c32 = np.float32(c)
        want = x / c32 if not LOWERING else x * np.float32(1 / c32)
        assert np.array_equal(got[k], want), (c, LOWERING)


def test_fp8_decode_and_encode_match_emmys_definitions():
    """from_f8* / to_f8* against emmy's own reference tables (dtype.decode_f8,
    encode_f8): all 256 codes of each format decoded, and float32 values of
    every kind encoded (rounding ties, the largest finite values, saturation,
    subnormals, infinities and NaN). The runner has no byte buffers, so the
    codes travel four to an int32."""
    from emmy.compiler.dtype import F8E4M3, F8E5M2, decode_f8, encode_f8

    n = 1024
    xs = np.concatenate([rng.standard_normal(n // 2) * 100, rng.standard_normal(n // 4) * 0.01,
                         np.ldexp(rng.integers(8, 32, n // 4 - 16), -rng.integers(0, 24, n // 4 - 16)),
                         [0.0, -0.0, 448.0, 464.0, 480.0, -500.0, 57344.0, 61440.0, 1e30, -1e30,
                          np.inf, -np.inf, np.nan, 2.0 ** -9, 2.0 ** -10, 2.0 ** -17]]).astype(np.float32)
    codes = np.arange(256, dtype=np.uint8)
    for fmt, dt in (("f8e4m3", F8E4M3), ("f8e5m2", F8E5M2)):
        body = [Tile(axes=(Axis("i", Dim(n)),), body=(
            Load(name="x", input="xs", index=(Var("i"),)),
            Assign("c", f"to_{fmt}", ("x",), dtype=dt),
            Write("enc", (Var("i"),), "c"),
            Cond(cond=B("<", Var("i"), I(256)), body=(
                Load(name="k", input="codes", index=(Var("i"),)),
                Assign("v", f"from_{fmt}", ("k",), dtype=F32),
                Write("dec", (Var("i"),), "v"),
            )),
        ))]
        buffers = {"xs": Buffer("xs", (n,), "f32"), "codes": Buffer("codes", (256,), fmt),
                   "enc": Buffer("enc", (n,), fmt), "dec": Buffer("dec", (256,), "f32")}
        src = KernelSource("k_test", tuple(body), ("xs", "codes", "enc", "dec"), buffers, Launch(n // 256, 256, 0),
                           frozenset({"enc", "dec"}))
        ptx = Lowerer(src, **LOWERING).run().render(PTX_VERSION, TARGET)
        c = requests.post(f"{RUNNER}/compile", json={"ptx": ptx}, timeout=120).json()
        assert c.get("ok"), c.get("stderr")
        args = [dict(name="xs", dtype="f32", shape=[n], data=[float(v) if np.isfinite(v) else str(v) for v in xs]),
                dict(name="codes", dtype="i32", shape=[64], data=codes.view(np.int32).tolist()),
                dict(name="enc", dtype="i32", shape=[n // 4], output=True),
                dict(name="dec", dtype="f32", shape=[256], output=True)]
        r = requests.post(f"{RUNNER}/run", json={
            "cubin_id": c["cubin_id"], "entry": "k_test", "grid": [n // 256, 1, 1], "block": [256, 1, 1], "smem_bytes": 0,
            "zero_outputs": [], "args": args, "warmup": 0, "iters": 1, "max_return": 1 << 16}, timeout=120).json()
        assert r.get("ok"), r.get("stderr")
        enc = np.array(r["outputs"]["enc"]["values"], np.int64).astype(np.int32).view(np.uint8)
        assert np.array_equal(enc, encode_f8(xs, fmt)), fmt
        dec = np.array([np.nan if v is None else v for v in r["outputs"]["dec"]["values"]], np.float32)
        want = decode_f8(codes, fmt)
        finite = np.isfinite(want)
        assert np.array_equal(dec[finite], want[finite]), fmt  # (the runner returns inf and NaN as null)
