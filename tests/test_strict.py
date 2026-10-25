"""``Lowerer(strict=True)``: no rewrite that changes a floating-point result."""

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, Write
from emmy_ptx import PTX_VERSION, TARGET
from emmy_ptx.frontend import Buffer, KernelSource, Launch
from emmy_ptx.lower import Lowerer


def shared_dot_products() -> KernelSource:
    """Row r of x dotted with y, needed alike by the 16 lanes (columns) of row r."""
    body = [Tile(axes=(Axis("r", Dim(16)), Axis("c", Dim(16))), body=(
        Loop(axis=Axis("k", Dim(64)), body=(
            Load(name="xv", input="x", index=(Var("r"), Var("k"))),
            Load(name="yv", input="y", index=(Var("k"),)),
            Assign("p", "multiply", ("xv", "yv")),
            Accum("acc", "p"),
        )),
        Write("out", (Var("r"), Var("c")), "acc"),
    ))]
    shapes = {"x": (16, 64), "y": (64,), "out": (16, 16)}
    buffers = {n: Buffer(n, s, "f32") for n, s in shapes.items()}
    return KernelSource("k", tuple(body), tuple(shapes), buffers, Launch(1, 256, 0), frozenset({"out"}))


def test_strict_keeps_the_reduction_whole_and_each_operation_rounded():
    src = shared_dot_products()
    fast = Lowerer(src).run().render(PTX_VERSION, TARGET)
    strict = Lowerer(src, strict=True).run().render(PTX_VERSION, TARGET)
    assert "shfl.sync.bfly" in fast and "fma.rn.f32" in fast  # split across the row's lanes, contracted
    assert "shfl.sync.bfly" not in strict and "fma.rn.f32" not in strict
    assert "mul.rn.f32" in strict and "add.rn.f32" in strict  # each lane: products and sums in the IR's order
