"""Cooperative reductions are regrouped only when the partials are fully combined
and every element keeps its value."""

import numpy as np
import pytest
from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, StridedLoop, Write
from kernels import regroup_hazards

from emmy_ptx.frontend import Buffer, KernelSource, Launch
from emmy_ptx.lower import Lowerer
from emmy_ptx.regroup import VECTOR, combine_refusal, outer_values, regroup

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
    assert combine_refusal(stmts, 0, {"acc"}, B) is None


def test_partials_used_without_a_combine_are_not_regrouped():
    stmts = [sum_loop(), Write("out", (Var("t"),), "acc")]  # each thread's partial is observed
    assert combine_refusal(stmts, 0, {"acc"}, B) is not None


def test_a_combine_over_fewer_threads_is_not_enough():
    stmts = [sum_loop(), *full_combine()[:1]]  # the warp butterfly alone: 32 of 128 threads
    assert combine_refusal(stmts, 0, {"acc"}, B) is not None


def test_a_combine_unlike_the_fold_is_refused():
    """A maximum of partial sums is not a sum: moving elements between the
    threads would change every partial, and so the maximum."""
    biggest = (Assign("acc", "maximum", ("acc", "acc__o")),)
    stmts = [sum_loop(), WarpShuffle(state=("acc",), state_b=("acc__o",), combine_states=biggest, length=32)]
    assert "maximum" in combine_refusal(stmts, 0, {"acc"}, 32)
    assert combine_refusal([sum_loop(), *full_combine()[:1]], 0, {"acc"}, 32) is None  # the sum of sums is fine


def test_a_combine_that_reads_the_partials_otherwise_is_refused():
    scaled = (Assign("w", "multiply", ("acc", "acc__o")), Assign("acc", "add", ("acc", "acc__o")))
    stmts = [sum_loop(), WarpShuffle(state=("acc",), state_b=("acc__o",), combine_states=scaled, length=32)]
    assert combine_refusal(stmts, 0, {"acc"}, 32) is not None


def test_a_combine_in_another_dtype_is_refused():
    from emmy.compiler.dtype import F64
    stmts = [sum_loop(), WarpShuffle(state=("acc",), state_b=("acc__o",), combine_states=COMBINE, length=32, dtype=F64)]
    assert "f64" in combine_refusal(stmts, 0, {"acc"}, 32)


def test_warp_partials_written_anywhere_but_the_tree_are_refused():
    stmts = [sum_loop(), *full_combine()]
    stmts[3] = Cond(cond=BinaryExpr("==", Var("lane"), Literal(0, "int")),
                    body=(Write("acc_smem", (Var("warp"),), "acc"), Write("out", (Var("warp"),), "acc")))
    assert combine_refusal(stmts, 0, {"acc"}, B) is not None
    # and a tree whose shared partials something else reads is not private
    assert combine_refusal([sum_loop(), *full_combine()], 0, {"acc"}, B, lambda tree: "read later") == "read later"


def test_rewrite_folds_contiguous_chunks_of_every_element_once():
    loop = regroup(sum_loop(), B - 1, B, N)
    assert loop is not None and int(loop.axis.extent.as_static()) == N // (VECTOR * B)
    loads = [s for s in loop.body if type(s).__name__ == "Load"]
    assert len(loads) == VECTOR and len({s.names[0] for s in loads}) == VECTOR  # independent copies
    covered = []
    for t in range(B):
        for k in range(N // (VECTOR * B)):
            for s in loads:
                covered.append(s.index[0].eval({"t": t, "a$chunk": k}))
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
    hand_off = Cond(cond=BinaryExpr("==", Var("lane"), Literal(0, "int")),
                    body=(Write("acc_smem", (Var("warp"),), "acc0"),))
    combine = [WarpShuffle(state=("acc0",), state_b=("acc0__o",), combine_states=add, length=32), *full_combine()[1:2]]
    tree = TreeHalve(bufs=("acc_smem",), state=("acc0",), state_b=("acc0__o",), combine_states=add, length=B // 32,
                     tid_var="warp")
    stmts = [unrolled_loop(), *merge, combine[0], combine[1], hand_off, Sync(), tree]
    assert combine_refusal(stmts, 0, {"acc0", "acc1"}, B) is None
    assert combine_refusal([unrolled_loop(), *combine], 0, {"acc0", "acc1"}, B) is not None  # acc1 never joins


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


def test_every_warp_hands_its_partial_to_the_tree_once():
    """Lane 0 of each warp writes the warp's partial to slot ``warp`` of the
    array the tree folds it from, then a barrier. Written by one thread only,
    to one slot or another array, or with no barrier, a warp's partial may
    stay out of the total, which regrouping, moving elements between the
    warps, would change."""
    def handed(cond=BinaryExpr("==", Var("lane"), Literal(0, "int")), index=Var("warp"), array="acc_smem",
               barrier=True, builtin=lambda name: True):
        stmts = [sum_loop(), *full_combine()]
        stmts[3] = Cond(cond=cond, body=(Write(array, (index,), "acc"),))
        if not barrier:
            del stmts[4]
        return combine_refusal(stmts, 0, {"acc"}, B, builtin=builtin)

    assert handed() is None
    assert "lane == 0" in handed(cond=BinaryExpr("==", Var("t"), Literal(0, "int")))
    assert handed(index=Literal(0, "int")) is not None
    assert "does not fold acc" in handed(array="other_smem")
    assert "barrier" in handed(barrier=False)
    assert "bound" in handed(builtin=lambda name: name != "warp")  # the kernel bound warp to a value of its own


def test_copies_that_compute_differently_are_not_regrouped():
    """Element j of a chunk goes to copy j mod R, wherever the stride put it,
    so each copy must compute what the first does, its names and offset aside."""
    def squared(loop, op1="multiply"):
        x0, acc0, x1, acc1 = loop.body
        body = (x0, Assign("y0", "multiply", ("x0", "x0")), Accum("acc0", "y0"),
                x1, Assign("y1", op1, ("x1", "x1")), Accum("acc1", "y1"))
        return StridedLoop(axis=loop.axis, start=loop.start, step=loop.step, body=body)

    assert regroup(squared(unrolled_loop()), B - 1, 2 * B, N) is not None
    assert regroup(squared(unrolled_loop(), op1="add"), B - 1, 2 * B, N) is None  # x0 * x0 but x1 + x1


def test_outer_values_are_what_the_body_reads_from_outside():
    loop = StridedLoop(axis=Axis("a", Dim(N)), start=Var("t"), step=Literal(B, "int"), body=(
        Load(name="x0", input="x", index=(Var("r"), Var("a"))), Assign("y", "multiply", ("x0", "w")),
        Accum("acc", "y")))
    assert outer_values(loop) == {"r", "w"}


def remarks(body, inputs: dict, shape, threads: int, grid: int = 1) -> list[str]:
    """What the lowering decided for kernel ``body``: inputs and output ``out`` are float32."""
    buffers = {n: Buffer(n, a.shape, "f32") for n, a in inputs.items()}
    buffers["out"] = Buffer("out", shape, "f32")
    src = KernelSource("k_test", tuple(body), (*inputs, "out"), buffers, Launch(grid, threads, 0), frozenset({"out"}))
    return Lowerer(src).run().remarks


@pytest.mark.parametrize("case", list(regroup_hazards()))
def test_a_sum_whose_elements_would_change_value_is_not_regrouped(case):
    """The loop must start at each thread's own rank, read nothing from outside
    that differs between the threads, load memory no thread writes, have
    copies that compute alike and hand every warp's partial to the tree
    (tests/kernels.py; tests/test_gpu_lowering.py runs these on the GPU)."""
    body, inputs, shape, threads, _ = regroup_hazards()[case]
    decided = remarks(body, inputs, shape, threads)
    assert not any("contiguous chunks" in r for r in decided), decided


def test_a_sum_that_keeps_every_value_is_still_regrouped():
    """emmy's form: each block of 128 threads strides its row (the row, the
    block's coordinate, is the same for all of them), then the butterfly, the
    hand-off and the tree."""
    row = Load(name="x0", input="x", index=(Var("r"), Var("a")))
    loop = StridedLoop(axis=Axis("a", Dim(N)), start=Var("t"), step=Literal(B, "int"), body=(row, Accum("acc", "x0")))
    body = [Tile(axes=(Axis("r", Dim(2)), Axis("t", Dim(B))), block_threads=B, body=(
        loop, *full_combine(), Write("out", (Var("r"), Var("t")), "acc")))]
    decided = remarks(body, {"x": np.zeros((2, N), np.float32)}, (2, B), B, grid=2)
    assert any("128 cooperating threads take contiguous chunks" in r for r in decided), decided
