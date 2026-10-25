"""Reduction mapping rewrites (mapping.py): fusion, pruning, the online-softmax
combine, unroll-and-jam renaming and address strides."""

import math
import random

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop, Write
from emmy.compiler.ir.stmt.leaves import Let

from emmy_ptx import mapping

K = Axis("k", Dim(64))


def dot(acc, row, axis=K):
    """``acc = sum_k x[row, k] * w[k]``: a pure reduction loop."""
    return Loop(axis=axis, body=(
        Load(name=f"x_{acc}", input="x", index=(Literal(row, "int"), Var(axis.name))),
        Load(name=f"w_{acc}", input="w", index=(Var(axis.name),)),
        Assign(f"p_{acc}", "multiply", (f"x_{acc}", f"w_{acc}")),
        Accum(acc, f"p_{acc}"),
    ))


def fused(stmts):
    stmts = list(stmts)
    mapping.absorb_siblings(stmts, 0, lambda s, t: True)
    return stmts


def test_independent_siblings_fuse_past_what_lies_between():
    out = fused([dot("a", 0), Write("o", (Literal(0, "int"),), "a"), dot("b", 1, Axis("k2", Dim(64)))])
    assert [type(s).__name__ for s in out] == ["Loop", "Write"]
    loop = out[0]
    assert [s.name for s in loop.body if type(s).__name__ == "Accum"] == ["a", "b"]
    assert "k2" not in mapping.reads(loop.body)  # the absorbed loop now reads the fused loop's variable


def test_a_loop_reading_an_earlier_result_stays_put():
    second = Loop(axis=K, body=(Load(name="y", input="x", index=(Var("k"),)), Assign("q", "multiply", ("y", "a")),
                                Accum("b", "q")))
    assert len(fused([dot("a", 0), second])) == 2


def test_no_fusion_across_a_barrier_or_a_store_to_loaded_memory():
    assert len(fused([dot("a", 0), Sync(), dot("b", 1)])) == 3
    assert len(fused([dot("a", 0), Write("x", (Literal(0, "int"),), "a"), dot("b", 1)])) == 3


def test_different_iteration_spaces_never_fuse():
    assert len(fused([dot("a", 0), dot("b", 1, Axis("k", Dim(32)))])) == 2


def test_prune_drops_what_only_dead_code_reads():
    body = (dot("dead", 0), Assign("u", "multiply", ("dead", "dead")), dot("live", 1), Accum("total", "live"))
    kept = mapping.prune_dead(body)
    assert [type(s).__name__ for s in kept] == ["Loop", "Accum"]
    assert kept[0].body[-1].name == "live"


def online_softmax_update():
    """emmy's online softmax over scores ``s`` (one step of a loop body)."""
    return (
        Let(name="one", value=Literal(1.0)),
        Assign("gn", "maximum", ("s", "m")),
        Assign("dg", "subtract", ("m", "gn")),
        Assign("alpha", "exp", ("dg",)),
        Assign("dgo", "subtract", ("s", "gn")),
        Assign("beta", "exp", ("dgo",)),
        Assign("sa", "multiply", ("alpha", "l")),
        Assign("sb", "multiply", ("beta", "one")),
        Accum("l", "sb", base="sa"),
        Accum("m", "s", op="maximum"),
    )


OPS = {"maximum": max, "subtract": lambda a, b: a - b, "exp": math.exp, "multiply": lambda a, b: a * b,
       "add": lambda a, b: a + b}


def step(stmts, env):
    """Interpret Let/Assign/Accum statements over Python floats."""
    for s in stmts:
        kind = type(s).__name__
        if kind == "Let":
            env[s.name] = s.value.value
        elif kind == "Assign":
            env[s.name] = OPS[s.op.name](*(env[a] for a in s.args))
        elif kind == "Accum":
            base = s.base if s.base is not None else s.name
            env[s.name] = OPS[{"add": "add", "maximum": "maximum"}[s.op.name]](env[base], env[s.value])
    return env


def fold(scores):
    env = {"m": -1e30, "l": 0.0}
    for x in scores:
        env = step(online_softmax_update(), {**env, "s": x})
    return env


def test_online_softmax_state_is_recognised():
    (group,) = mapping.online_groups(online_softmax_update())
    assert (group.m, group.s, group.sums) == ("m", "s", (("l", "sb", "one"),))


def test_derived_combine_merges_two_partial_states_exactly():
    random.seed(3)
    scores = [random.gauss(0, 3) for _ in range(40)]
    (group,) = mapping.online_groups(online_softmax_update())
    program = mapping.combine_program(group, lambda n: n + "$b", lambda n: n + "$c")
    for cut in (0, 1, 17, 40):
        a, b = fold(scores[:cut]), fold(scores[cut:])
        merged = step(program, {"m": a["m"], "l": a["l"], "m$b": b["m"], "l$b": b["l"]})
        top = max(scores)
        assert merged["m"] == top
        assert math.isclose(merged["l"], sum(math.exp(x - top) for x in scores), rel_tol=1e-12)


def test_an_unrecognised_rescaled_sum_blocks_the_split():
    body = (Assign("sa", "multiply", ("l", "c")), Accum("l", "v", base="sa"))
    assert mapping.online_groups(body) is None


def test_jammed_copies_are_renamed_apart_but_share_the_state():
    body = (dot("s", 0), Accum("total", "s"))
    copies = mapping.jammed(body, "j", 2, {"total"})
    accs = [s.name for s in copies if type(s).__name__ == "Accum"]
    assert accs == ["total", "total"]
    inner = [s for s in copies if type(s).__name__ == "Loop"]
    assert [s.body[-1].name for s in inner] == ["s$0", "s$1"]
    assert [s.axis.name for s in inner] == ["k$0", "k$1"]


def test_flat_coefficients_follow_the_row_major_offset():
    index = (Var("a2"), Var("a1"))
    assert mapping.flat_coefficient(index, (256, 512), "a1", {}) == 1
    assert mapping.flat_coefficient(index, (256, 512), "a2", {}) == 512
    lets = {"j": BinaryExpr("*", Var("a1"), Literal(3, "int"))}
    assert mapping.flat_coefficient((Var("j"),), (999,), "a1", lets) == 3
    wrapped = (BinaryExpr("%", Var("a1"), Literal(7, "int")),)
    assert mapping.flat_coefficient(wrapped, (7,), "a1", {}) is None
