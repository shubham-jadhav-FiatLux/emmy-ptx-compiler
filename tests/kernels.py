"""Hand-built Kernel IR shared by the tests: Tiles whose cells the lowering places
anew (Lowerer._placement)."""

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Write
from emmy.compiler.ir.stmt.leaves import Let


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
