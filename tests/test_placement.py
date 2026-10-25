"""Where the lowering puts a Tile's cells (Lowerer._placement)."""

import pytest
from kernels import three_column_attention

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, Write
from emmy_ptx.frontend import Buffer, KernelSource, Launch
from emmy_ptx.lower import Lowerer, Placement, _decompositions


def source(body, shapes: dict, outputs: tuple, grid: int, block: int) -> KernelSource:
    buffers = {n: Buffer(n, s, "f32") for n, s in shapes.items()}
    return KernelSource("k_test", tuple(body), tuple(shapes), buffers, Launch(grid, block, 0), frozenset(outputs))


def attention(rows: int, block: int) -> KernelSource:
    shapes = {"q": (rows, 16), "kt": (16, 64), "v": (64, 32), "out": (rows, 32)}
    return source(three_column_attention(rows, 16, 64), shapes, ("out",), 1, block)


def test_decompositions_put_most_cells_on_one_group_first():
    assert list(_decompositions(11)) == [(16,), (8, 4), (8, 2, 1)]
    assert list(_decompositions(20)) == [(32,), (16, 4)]
    assert list(_decompositions(64)) == [(32, 32)]
    assert list(_decompositions(70)) == [(32, 32, 8), (32, 32, 4, 2)]


@pytest.mark.parametrize("block,parts", [(512, (16,)), (384, (8, 4)), (352, (8, 2, 1))])
def test_the_launch_decides_how_many_lanes_a_row_gets(block, parts):
    # 32 rows of 11 cells: 16 lanes a row take 512 threads, 8 + 4 take 384, exactly 11 (8 + 2 + 1) 352
    src = attention(32, block)
    assert Lowerer(src)._placement(src.body[0], [32, 11]) == Placement(1, parts, 32)


def test_cells_are_placed_only_where_their_lanes_then_share_work():
    src = attention(16, 192)
    first = Lowerer(src)
    first.body(src.body)
    assert not first.shared_loop and first.wants_place  # as emmy lays them out, no lanes share a row
    placed = Lowerer(src, place=True)
    placed.run()
    assert placed.placed_shared


def test_rows_that_only_pairs_of_lanes_share_get_larger_groups():
    """14 cells to a row (42 columns): as emmy lays them out, aligned pairs of
    lanes hold one row and share its scores; 16 lanes a row share more."""
    shapes = {"q": (16, 16), "kt": (16, 64), "v": (64, 42), "out": (16, 42)}
    src = source(three_column_attention(16, 16, 64, cols=42), shapes, ("out",), 1, 256)
    first = Lowerer(src)
    first.body(src.body)
    assert first.shared_loop and first.wants_place
    assert first._placement(src.body[0], [16, 14]) == Placement(1, (16,), 16)


def test_cells_whose_loops_all_depend_on_the_axis_stay_put():
    """A matmul cell (r, c) folds row r of x with column c of y: no loop is the
    same for the cells of a row."""
    body = [Tile(axes=(Axis("r", Dim(16)), Axis("c", Dim(11))), body=(
        Loop(axis=Axis("k", Dim(64)), body=(
            Load(name="xv", input="x", index=(Var("r"), Var("k"))),
            Load(name="yv", input="y", index=(Var("k"), Var("c"))),
            Assign("p", "multiply", ("xv", "yv")),
            Accum("acc", "p"),
        )),
        Write("out", (Var("r"), Var("c")), "acc"),
    ))]
    src = source(body, {"x": (16, 64), "y": (64, 11), "out": (16, 11)}, ("out",), 1, 192)
    assert Lowerer(src)._placement(src.body[0], [16, 11]) is None
