"""Cooperative reductions are regrouped only when the partials are fully combined."""

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, StridedLoop, Write

from emmy_ptx.regroup import VECTOR, fully_combined, regroup

N, B = 4096, 128
COMBINE = (Assign("acc", "add", ("acc", "acc__o")),)


def sum_loop(extra=()):
    body = (Load(name="x0", input="x", index=(Var("a"),)), *extra, Accum("acc", "x0"))
    return StridedLoop(axis=Axis("a", Dim(N)), start=Var("t"), step=Literal(B, "int"), body=body)


def full_combine():
    return [
        WarpShuffle(state=("acc",), state_b=("acc__o",), combine_states=COMBINE, length=32),
        Smem("acc_smem", (B // 32,)),
        Cond(cond=BinaryExpr("==", Var("lane"), Literal(0, "int")), body=(Write("acc_smem", (Var("warp"),), "acc"),)),
        Sync(),
        TreeHalve(bufs=("acc_smem",), state=("acc",), state_b=("acc__o",), combine_states=COMBINE, length=B // 32,
                  tid_var="warp"),
    ]


def test_full_combine_is_recognised():
    stmts = [sum_loop(), *full_combine()]
    assert fully_combined(stmts, 0, {"acc"}, B)


def test_partials_used_without_a_combine_are_not_regrouped():
    stmts = [sum_loop(), Write("out", (Var("t"),), "acc")]  # each thread's partial is observed
    assert not fully_combined(stmts, 0, {"acc"}, B)


def test_a_combine_over_fewer_threads_is_not_enough():
    stmts = [sum_loop(), *full_combine()[:1]]  # the warp butterfly alone: 32 of 128 threads
    assert not fully_combined(stmts, 0, {"acc"}, B)


def test_rewrite_folds_contiguous_chunks_of_every_element_once():
    loop = regroup(sum_loop(), B - 1, B, N)
    assert loop is not None and int(loop.axis.extent.as_static()) == N // (VECTOR * B)
    loads = [s for s in loop.body if type(s).__name__ == "Load"]
    assert len(loads) == VECTOR and len({s.names[0] for s in loads}) == VECTOR  # independent copies
    covered = []
    for t in range(B):
        for k in range(N // (VECTOR * B)):
            for s in loads:
                covered.append(s.index[0].eval({"t": t, f"a$chunk": k}))
    assert sorted(covered) == list(range(N))


def test_reading_an_accumulator_mid_fold_blocks_the_rewrite():
    peek = (Assign("y", "multiply", ("x0", "acc")),)
    assert regroup(sum_loop(peek), B - 1, B, N) is None


def test_extent_must_split_into_whole_chunks():
    loop = StridedLoop(axis=Axis("a", Dim(N + 4)), start=Var("t"), step=Literal(B, "int"), body=sum_loop().body)
    assert regroup(loop, B - 1, B, N + 4) is None


def unrolled_loop(copies=2):
    """emmy's coop/rR form: copy j reads x[a + j*B] into an accumulator of its own."""
    body = []
    for j in range(copies):
        index = Var("a") if j == 0 else BinaryExpr("+", Var("a"), Literal(j * B, "int"))
        body += [Load(name=f"x{j}", input="x", index=(index,)), Accum(f"acc{j}", f"x{j}")]
    return StridedLoop(axis=Axis("a", Dim(N)), start=Var("t"), step=Literal(copies * B, "int"), body=tuple(body))


def test_unrolled_partials_folded_before_the_combine_count_as_combined():
    merge = [Accum("acc0", "acc1")]
    add = (Assign("acc0", "add", ("acc0", "acc0__o")),)
    combine = [WarpShuffle(state=("acc0",), state_b=("acc0__o",), combine_states=add, length=32), *full_combine()[1:2]]
    tree = TreeHalve(bufs=("acc_smem",), state=("acc0",), state_b=("acc0__o",), combine_states=add, length=B // 32,
                     tid_var="warp")
    stmts = [unrolled_loop(), *merge, combine[0], combine[1], Sync(), tree]
    assert fully_combined(stmts, 0, {"acc0", "acc1"}, B)
    assert not fully_combined([unrolled_loop(), *combine], 0, {"acc0", "acc1"}, B)  # acc1 never joins


def test_unrolled_loop_regroups_into_chunks_dealt_to_its_copies():
    for copies in (2, 4):
        loop = regroup(unrolled_loop(copies), B - 1, copies * B, N)
        assert loop is not None
        loads = [s for s in loop.body if type(s).__name__ == "Load"]
        folds = [s.name for s in loop.body if type(s).__name__ == "Accum"]
        assert sorted(set(folds)) == [f"acc{j}" for j in range(copies)]  # still one chain per copy
        covered = sorted(s.index[0].eval({"t": t, "a$chunk": k}) for t in range(B)
                         for k in range(N // (VECTOR * B)) for s in loads)
        assert covered == list(range(N))
