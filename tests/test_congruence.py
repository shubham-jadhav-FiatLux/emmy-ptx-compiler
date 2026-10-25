"""The congruence analysis merges exactly the provably equal accumulators."""

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop

from emmy_ptx.congruence import congruent_accumulators, merge_congruent

K = Axis("k", Dim(64))
START = ("imm", "f32", -1e30)


def softmax_cells(columns):
    """Online-softmax state of one row for several output columns."""
    body = [Load(name="s", input="scores", index=(Var("k"),))]
    for c in columns:
        body += [
            Assign(f"m_new{c}", "maximum", (f"m{c}", "s")),
            Assign(f"d{c}", "subtract", (f"m{c}", f"m_new{c}")),
            Assign(f"alpha{c}", "exp", (f"d{c}",)),
            Assign(f"rescaled{c}", "multiply", (f"l{c}", f"alpha{c}")),
            Load(name=f"v{c}", input="v", index=(Var("k"), Literal(c, "int"))),
            Assign(f"wv{c}", "multiply", (f"v{c}", f"alpha{c}")),
            Accum(f"l{c}", "s", base=f"rescaled{c}"),
            Accum(f"acc{c}", f"wv{c}"),
            Accum(f"m{c}", "s", op="maximum"),
        ]
    return Loop(axis=K, body=tuple(body))


def starts(loop, value=START):
    return {st.name: value for st in loop.body if type(st).__name__ == "Accum"}


def test_equal_states_merge_and_column_sums_stay_apart():
    loop = softmax_cells([0, 1, 2])
    members = congruent_accumulators(loop, starts(loop))
    # max and sum depend only on the row's scores; the weighted sums read other columns of v
    assert members == {"l1": "l0", "l2": "l0", "m1": "m0", "m2": "m0"}


def test_different_start_values_never_merge():
    loop = softmax_cells([0, 1])
    s = starts(loop)
    s["m1"] = ("imm", "f32", 0.0)
    assert "m1" not in congruent_accumulators(loop, s)


def test_unknown_start_values_never_merge():
    loop = softmax_cells([0, 1])
    assert congruent_accumulators(loop, starts(loop, None)) == {}


def test_same_shape_on_different_data_is_not_congruent():
    body = (
        Load(name="a", input="x", index=(Var("k"),)),
        Load(name="b", input="y", index=(Var("k"),)),
        Accum("sa", "a"),
        Accum("sb", "b"),
    )
    loop = Loop(axis=K, body=body)
    assert congruent_accumulators(loop, starts(loop, ("imm", "f32", 0.0))) == {}


def test_merge_reads_one_snapshot_and_drops_member_updates():
    loop = softmax_cells([0, 1])
    members = congruent_accumulators(loop, starts(loop))
    new = merge_congruent(loop.body, members)
    accums = [st.name for st in new if type(st).__name__ == "Accum"]
    assert accums == ["l0", "acc0", "m0", "acc1"]  # l1 and m1 are no longer updated
    lets = [st.name for st in new if type(st).__name__ == "Let"]
    assert sorted(lets) == ["l0$top", "m0$top"]
    reads = {a for st in new if type(st).__name__ == "Assign" for a in st.args}
    assert "m1" not in reads and "l1" not in reads
