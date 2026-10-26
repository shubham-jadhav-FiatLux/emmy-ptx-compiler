"""Hand-built Kernel IR shared by the tests: Tiles whose cells the lowering places
anew (Lowerer._placement), a loop whose work neighbouring blocks repeat
(cluster.py), and a staged matmul whose cells coarsen.py merges."""

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync, Tile, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, StridedLoop, Write
from emmy.compiler.ir.stmt.leaves import Init, Let


def I(v):  # noqa: E743 - an int literal
    return Literal(v, "int")


def B(op, a, b):
    return BinaryExpr(op, a, b)


def three_column_attention(rows: int, d: int, keys: int, cols: int = 32) -> list:
    """softmax(q @ kt) @ v the way emmy tiles some attention kernels: a cell
    (r, c) computes output columns 3c .. 3c+2 of row r, so 11 cells cover a
    row of 32 columns (the last one has two; the v column wraps, the store is
    guarded)."""
    cells = -(-cols // 3)

    def col(j):
        return B("+", B("*", Var("c"), I(3)), I(j))

    step = [
        Loop(axis=Axis("k", Dim(d)), body=(
            Load(name="qk", input="q", index=(Var("r"), Var("k"))),
            Load(name="kk", input="kt", index=(Var("k"), Var("t"))),
            Assign("p", "multiply", ("qk", "kk")),
            Accum("dot", "p"),
        )),
        Let(name="one", value=Literal(1.0), dtype=F32),
        Assign("gn", "maximum", ("dot", "m")),
        Assign("dg", "subtract", ("m", "gn")),
        Assign("alpha", "exp", ("dg",)),
        Assign("dgo", "subtract", ("dot", "gn")),
        Assign("beta", "exp", ("dgo",)),
        Assign("la", "multiply", ("alpha", "l")),
        Assign("lb", "multiply", ("beta", "one")),
        Accum("l", "lb", base="la"),
    ]
    for j in range(3):
        step += [
            Assign(f"oa{j}", "multiply", (f"o{j}", "alpha")),
            Load(name=f"vv{j}", input="v", index=(Var("t"), B("%", col(j), I(cols)))),
            Assign(f"ob{j}", "multiply", (f"vv{j}", "beta")),
            Accum(f"o{j}", f"ob{j}", base=f"oa{j}"),
        ]
    step.append(Accum("m", "dot", op="maximum"))
    tail = [Assign("inv", "reciprocal", ("l",))]
    for j in range(3):
        tail += [Assign(f"res{j}", "multiply", (f"o{j}", "inv")),
                 Cond(cond=B("<", col(j), I(cols)), body=(Write("out", (Var("r"), col(j)), f"res{j}"),))]
    attention = Loop(axis=Axis("t", Dim(keys)), body=tuple(step))
    return [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(cells))), body=(attention, *tail))]


def shared_sum_added_atomically(rows: int, n: int, cells: int) -> list:
    """Cell (r, c) adds sum(x[r, :]) * w[c] to out[r] atomically: every cell
    of a row needs the same sum."""
    return [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(cells))), body=(
        Loop(axis=Axis("k", Dim(n)), body=(
            Load(name="xv", input="x", index=(Var("r"), Var("k"))),
            Accum("total", "xv"),
        )),
        Load(name="wc", input="w", index=(Var("c"),)),
        Assign("y", "multiply", ("total", "wc")),
        Write("out", (Var("r"),), "y", atomic=True),
    ))]


def rasterized_row_scale(rows: int, n: int, cells: int, group: int) -> list:
    """out[r, c] = x[r, c] * sum(x[r, :]^2), with emmy's grouped rasterization of
    the two axes: consecutive cells walk down ``group`` rows before the next
    column, so the lanes of a row are ``group`` apart."""
    return [Tile(axes=(Axis("r", Dim(rows)), Axis("c", Dim(cells))), raster_axes=("r", "c"), raster_group=group,
                 raster_orient="m", body=(
        Loop(axis=Axis("k", Dim(n)), body=(
            Load(name="xk", input="x", index=(Var("r"), Var("k"))),
            Assign("sq", "multiply", ("xk", "xk")),
            Accum("ss", "sq"),
        )),
        Load(name="xc", input="x", index=(Var("r"), Var("c"))),
        Assign("y", "multiply", ("xc", "ss")),
        Write("out", (Var("r"), Var("c")), "y"),
    ))]


def column_recomputing_mlp(rows: int = 2, k: int = 128, hidden: int = 128, cols: int = 16, lanes: int = 8) -> list:
    """relu(x @ w1) @ w2 the way emmy fuses it when every block computes one
    output: the ``lanes`` threads of block (row, col) split the hidden units,
    and every block recomputes all hidden units of its row (the same for each
    of the row's ``cols`` blocks). Launch: rows * cols blocks of ``lanes``."""
    co = Var("a2_co")
    loop = StridedLoop(axis=Axis("a2", Dim(hidden)), start=co, step=I(lanes), body=(
        Loop(axis=Axis("a3", Dim(k)), body=(
            Load(name="xv", input="x", index=(Var("a0"), Var("a3"))),
            Load(name="wv", input="w1", index=(Var("a3"), Var("a2"))),
            Assign("p", "multiply", ("wv", "xv")),
            Accum("acc0", "p"),
        )),
        Assign("h", "relu", ("acc0",)),
        Load(name="w2v", input="w2", index=(Var("a2"), Var("a1"))),
        Assign("q", "multiply", ("w2v", "h")),
        Accum("acc1", "q"),
    ))
    return [Tile(axes=(Axis("a0", Dim(rows)), Axis("a1", Dim(cols)), Axis("a2_co", Dim(lanes))), body=(
        loop,
        WarpShuffle(state=("acc1",), state_b=("acc1__o",), combine_states=(Assign("acc1", "add", ("acc1", "acc1__o")),),
                    length=lanes),
        Cond(cond=B("==", co, I(0)), body=(Write("out", (Var("a0"), Var("a1")), "acc1"),)),
    ))]


def staged_matmul(rows: int = 32, cols: int = 64, k: int = 64, stage: int = 32) -> list:
    """a @ b the way emmy schedules an f32 matmul on 512 threads: cells of
    2 x 2 outputs (a0_u: 16 cell rows, a1_u: 32 cell columns), the tiles of a
    and b staged in shared memory by cp.async (8 and 16 bytes a thread), and
    an unrolled inner loop over each stage. One block: rows x cols outputs."""
    t = B("+", B("*", Var("a0_u"), I(32)), Var("a1_u"))  # the flat thread
    r, c = (B("+", B("*", Var(axis), I(2)), I(j)) for axis, j in (("a0_u", 0), ("a1_u", 0)))
    acc = {(i, j): f"acc{i}{j}" for i in range(2) for j in range(2)}

    def row(i):
        return B("+", B("*", Var("a0_u"), I(2)), I(i))

    def col(j):
        return B("+", B("*", Var("a1_u"), I(2)), I(j))

    ki = Var("ki")
    compute = [Load(name=f"av{i}", input="a_s", index=(row(i), ki)) for i in range(2)]
    compute += [Load(name=f"bv{j}", input="b_s", index=(ki, col(j))) for j in range(2)]
    for (i, j), name in acc.items():
        compute += [Assign(f"p{i}{j}", "multiply", (f"av{i}", f"bv{j}")), Accum(name, f"p{i}{j}")]
    ks = Var("ks")
    step = (
        CpAsyncCopy("a_s", (B("/", t, I(16)), B("*", B("%", t, I(16)), I(2))), "a",
                    (B("/", t, I(16)), B("+", B("*", ks, I(stage)), B("*", B("%", t, I(16)), I(2)))), nbytes=8),
        CpAsyncCopy("b_s", (B("/", t, I(16)), B("*", B("%", t, I(16)), I(4))), "b",
                    (B("+", B("*", ks, I(stage)), B("/", t, I(16))), B("*", B("%", t, I(16)), I(4))), nbytes=16),
        CpAsyncCommit(), CpAsyncWait(0), Sync(),
        Loop(axis=Axis("ki", Dim(stage)), body=tuple(compute), unroll=True, seed=False),
        Sync(),
    )
    del r, c
    return [Tile(axes=(Axis("a0_u", Dim(rows // 2)), Axis("a1_u", Dim(cols // 2))), body=(
        *(Init(name, 0.0, dtype=F32) for name in acc.values()),
        Smem("a_s", (rows, stage), "float", 16), Smem("b_s", (stage, cols), "float", 16),
        Loop(axis=Axis("ks", Dim(k // stage)), body=step),
        *(Write("out", (row(i), col(j)), name) for (i, j), name in acc.items()),
    ))]
