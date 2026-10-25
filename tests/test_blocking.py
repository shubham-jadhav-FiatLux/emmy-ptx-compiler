"""Lane blocking (mapping.blocked): run by a group of lanes in lockstep, the
rewritten loop computes what each lane computes running the original alone.

A small interpreter executes the IR over Python floats. ``run`` drives all
lanes of a group statement by statement, so the cross-lane statements of the
rewrite (WarpShuffle butterflies, LaneBroadcast) see every lane's values at
the same point, as on the GPU.
"""

import math
import random

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Load, Loop
from emmy.compiler.ir.stmt.leaves import Let

from emmy_ptx import mapping

D = 8  # dot-product length
OPS = {"maximum": max, "subtract": lambda a, b: a - b, "exp": math.exp, "multiply": lambda a, b: a * b,
       "add": lambda a, b: a + b, "sigmoid": lambda a: 1 / (1 + math.exp(-a))}
IDENTITY = {"add": 0.0, "maximum": -1e30, "multiply": 1.0}


def run(stmts, lanes: list[dict], arrays: dict) -> None:
    """Execute ``stmts`` on every lane (an environment each), in lockstep."""
    for s in stmts:
        kind = type(s).__name__
        if kind == "Loop":
            if s.seed:
                for name, op in {x.name: x.op.name for x in s.body if type(x).__name__ == "Accum"}.items():
                    for env in lanes:
                        env[name] = IDENTITY[op]
            for i in range(int(s.axis.extent.as_static())):
                for env in lanes:
                    env[s.axis.name] = i
                run(s.body, lanes, arrays)
        elif kind == "WarpShuffle":
            width = s.length // 2
            while width:
                before = [dict(env) for env in lanes]
                for i, env in enumerate(lanes):
                    for st, sb in zip(s.state, s.state_b):
                        env[sb] = before[i ^ width][st]
                    run(s.combine_states, [env], arrays)
                width //= 2
        elif kind == "LaneBroadcast":
            before = [dict(env) for env in lanes]
            for i, env in enumerate(lanes):
                env[s.name] = before[(i & ~(s.group - 1)) + s.lane][s.value]
        else:
            for env in lanes:
                step(s, env, arrays)


def step(s, env: dict, arrays: dict) -> None:
    kind = type(s).__name__
    if kind == "Let":
        env[s.name] = s.value.eval(env)
    elif kind == "Assign":
        env[s.name] = OPS[s.op.name](*(env[a] for a in s.args))
    elif kind == "Accum":
        env[s.name] = OPS[s.op.name](env[s.base if s.base is not None else s.name], env[s.value])
    elif kind == "Load":
        env[s.names[0]] = float(arrays[s.input][tuple(e.eval(env) for e in s.index)])
    else:
        raise AssertionError(f"unexpected {kind}")


def scores(t: str = "t"):
    """``dot = sum_k q[k] * kt[k, t]``: the work every lane of a row group repeats."""
    return Loop(axis=Axis("k", Dim(D)), body=(
        Load(name="qk", input="q", index=(Var("k"),)),
        Load(name="kk", input="kt", index=(Var("k"), Var(t))),
        Assign("p", "multiply", ("qk", "kk")),
        Accum("dot", "p"),
    ))


def attention(n: int):
    """One output column of softmax(q . kt) @ v, as emmy's online softmax."""
    return Loop(axis=Axis("t", Dim(n)), body=(
        scores(),
        Assign("s", "multiply", ("dot", "scale")),
        Let(name="one", value=Literal(1.0)),
        Assign("gn", "maximum", ("m", "s")),
        Assign("dg", "subtract", ("m", "gn")),
        Assign("alpha", "exp", ("dg",)),
        Assign("dgo", "subtract", ("s", "gn")),
        Assign("beta", "exp", ("dgo",)),
        Assign("la", "multiply", ("l", "alpha")),
        Assign("lb", "multiply", ("one", "beta")),
        Accum("l", "lb", base="la"),
        Assign("oa", "multiply", ("o", "alpha")),
        Load(name="vv", input="v", index=(Var("t"), Var("col"))),
        Assign("ob", "multiply", ("vv", "beta")),
        Accum("o", "ob", base="oa"),
        Accum("m", "s", op="maximum"),
    ))


def sigmoid_attention(n: int):
    """One output column of sigmoid(q . kt) @ v: a plain fold of broadcast scores."""
    return Loop(axis=Axis("t", Dim(n)), body=(
        scores(),
        Assign("g", "sigmoid", ("dot",)),
        Load(name="vv", input="v", index=(Var("t"), Var("col"))),
        Assign("ob", "multiply", ("vv", "g")),
        Accum("o", "ob"),
    ))


PER_LANE = {"vv", "ob"}  # what depends on the lane's own column


def inputs(n: int, lanes: int) -> dict:
    rng = np.random.default_rng(n)
    return {"q": rng.standard_normal(D), "kt": rng.standard_normal((D, n)) * 2, "v": rng.standard_normal((n, lanes))}


def alone(loop, arrays: dict, col: int, states: dict) -> dict:
    env = {"col": col, "scale": 0.7, **states}
    run((loop,), [env], arrays)
    return env


def together(plan: mapping.Blocked, arrays: dict, lanes: int, states: dict) -> list[dict]:
    envs = [{"col": i, "lane": i, "scale": 0.7, **states} for i in range(lanes)]
    run(((plan.loop,) if plan.loop is not None else ()) + plan.rest + plan.merges, envs, arrays)
    return envs


SOFTMAX = {"m": -1e30, "l": 0.0, "o": 0.0}


def test_blocked_online_softmax_matches_each_lane_alone():
    for n, group in ((32, 8), (21, 8), (5, 8), (64, 32)):  # whole blocks, a partial last block, one partial block
        loop, arrays = attention(n), inputs(n, group)
        plan = mapping.blocked(loop, n, 0, 1, group, lambda name: name not in PER_LANE)
        assert plan is not None and len(plan.merges) == 1  # l is kept per lane, then added up
        for lane, env in enumerate(together(plan, arrays, group, SOFTMAX)):
            want = alone(loop, arrays, lane, SOFTMAX)
            assert env["m"] == want["m"]
            assert math.isclose(env["l"], want["l"], rel_tol=1e-12)
            assert math.isclose(env["o"], want["o"], rel_tol=1e-12, abs_tol=1e-12)


def test_blocked_plain_fold_is_bit_identical():
    n, group = 24, 8
    loop, arrays = sigmoid_attention(n), inputs(n, group)
    plan = mapping.blocked(loop, n, 0, 1, group, lambda name: name not in PER_LANE)
    assert plan is not None and not plan.merges
    for lane, env in enumerate(together(plan, arrays, group, {"o": 0.0})):
        assert env["o"] == alone(loop, arrays, lane, {"o": 0.0})["o"]  # same values, same order


def test_blocking_needs_shared_work_and_shared_scores():
    n = 16
    flat = Loop(axis=Axis("t", Dim(n)), body=tuple(s for s in attention(n).body if type(s).__name__ != "Loop")
                + (Load(name="dot", input="w", index=(Var("t"),)),))
    assert mapping.blocked(flat, n, 0, 1, 8, lambda name: name not in PER_LANE) is None  # no inner loop to share
    # scores that differ between lanes cannot be computed by one lane for the others
    assert mapping.blocked(attention(n), n, 0, 1, 8, lambda name: name not in PER_LANE | {"qk", "p", "dot", "s"}) is None


def test_cse_merges_repeats_until_an_operand_changes():
    body = (
        Assign("a", "multiply", ("x", "y")),
        Assign("b", "multiply", ("x", "y")),
        Accum("x", "b"),
        Assign("c", "multiply", ("x", "y")),
        Assign("d", "add", ("b", "c")),
    )
    out = mapping.cse(body)
    assert [s.name for s in out] == ["a", "x", "c", "d"]
    assert out[1].value == "a" and out[3].args == ("a", "c")


def test_random_scores_with_masked_keys():
    """-inf scores (a causal mask) leave the blocked fold exact, even for a whole block."""
    random.seed(5)
    n, group = 40, 8
    loop = attention(n)
    arrays = inputs(n, group)
    arrays["kt"][:, 8:16] = 0.0
    arrays["q"] = np.full(D, 1.0)
    body = list(loop.body)
    body.insert(2, Assign("s", "add", ("s0", "mask")))
    body[1] = Assign("s0", "multiply", ("dot", "scale"))
    body.insert(1, Load(name="mask", input="mask", index=(Var("t"),)))
    masked = Loop(axis=loop.axis, body=tuple(body))
    arrays["mask"] = np.array([-math.inf if 8 <= t < 16 or random.random() < 0.3 else 0.0 for t in range(n)])
    arrays["mask"][0] = 0.0
    plan = mapping.blocked(masked, n, 0, 1, group, lambda name: name not in PER_LANE)
    assert plan is not None
    for lane, env in enumerate(together(plan, arrays, group, SOFTMAX)):
        want = alone(masked, arrays, lane, SOFTMAX)
        assert math.isclose(env["l"], want["l"], rel_tol=1e-12)
        assert math.isclose(env["o"], want["o"], rel_tol=1e-12, abs_tol=1e-12)
