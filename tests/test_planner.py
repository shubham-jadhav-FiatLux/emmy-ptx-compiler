"""The planner's dependences (emmy_ptx/planner.py) follow each value in order."""

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Write
from emmy.compiler.ir.stmt.leaves import Init, Let

from emmy_ptx.planner import plans


def keys(*body, axes=(("t", 32),)) -> dict[str, tuple]:
    tile = Tile(axes=tuple(Axis(n, Dim(e)) for n, e in axes), block_threads=32, body=body)
    return {p.loop: p.key for p in plans((tile,), 32, 1)}


def test_an_accumulator_updated_from_lane_data_carries_its_lane():
    """s starts at 0, then folds x[t]: a loop folding s * w[j] depends on t,
    though s's definition (its Init) reads nothing (an independent review's case)."""
    found = keys(Init("s", 0.0, dtype=F32),
                 Loop(axis=Axis("k", Dim(1)), seed=False, body=(Load(name="xv", input="x", index=(Var("t"),)),
                                                                 Accum("s", "xv"))),
                 Loop(axis=Axis("j", Dim(32)), body=(Load(name="wv", input="w", index=(Var("j"),)),
                                                      Assign("p", "multiply", ("s", "wv")), Accum("r", "p"))),
                 Write("out", (Var("t"),), "r"))
    assert found["j"] == ("t",)


def test_a_value_from_an_earlier_trip_reaches_the_statements_before_its_update():
    """In each trip y = s * w[j] reads the s the trip before left, which folds x[t]."""
    found = keys(Init("s", 0.0, dtype=F32),
                 Loop(axis=Axis("j", Dim(8)), seed=False, body=(
                     Load(name="wv", input="w", index=(Var("j"),)), Assign("y", "multiply", ("s", "wv")),
                     Accum("r", "y"), Load(name="xv", input="x", index=(Var("t"),)), Accum("s", "xv"))),
                 Write("out", (Var("t"),), "r"))
    assert found["j"] == ("t",)


def test_an_accumulator_a_loop_seeds_starts_afresh():
    """Attention's form: the score loop k seeds its own sum on every key, so it
    depends on the row only, though the key loop around it folds per-column
    values into its own accumulator."""
    score = Loop(axis=Axis("k", Dim(16)), body=(Load(name="qv", input="q", index=(Var("r"), Var("k"))),
                                                Load(name="kv", input="kt", index=(Var("k"), Var("key"))),
                                                Assign("qk", "multiply", ("qv", "kv")), Accum("dot", "qk")))
    keyloop = Loop(axis=Axis("key", Dim(8)), seed=False, body=(
        score, Load(name="vv", input="v", index=(Var("key"), Var("c"))), Assign("pv", "multiply", ("dot", "vv")),
        Accum("o", "pv")))
    found = keys(Init("o", 0.0, dtype=F32), keyloop, Write("out", (Var("r"), Var("c")), "o"),
                 axes=(("r", 4), ("c", 8)))
    assert found["k"] == ("r",) and found["key"] == ("r", "c")


def test_a_value_set_under_a_condition_depends_on_it():
    """w is 1, plus 2 where t < 4: it depends on t through the test alone."""
    found = keys(Init("w", 1.0, dtype=F32), Let(name="two", value=Literal(2.0), dtype=F32),
                 Cond(cond=BinaryExpr("<", Var("t"), Literal(4, "int")), body=(Accum("w", "two"),)),
                 Loop(axis=Axis("j", Dim(8)), body=(Load(name="xv", input="x", index=(Var("j"),)),
                                                     Assign("y", "multiply", ("xv", "w")), Accum("r", "y"))),
                 Write("out", (Var("t"),), "r"))
    assert found["j"] == ("t",)
