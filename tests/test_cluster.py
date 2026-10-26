"""When cluster.py shares a loop's work across neighbouring blocks, and when it leaves it alone."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernels import column_recomputing_mlp  # noqa: E402

from emmy_ptx import cluster  # noqa: E402


def test_a_loop_repeated_across_blocks_is_split_over_a_cluster():
    plan = cluster.plan(column_recomputing_mlp()[0], 32, 8)
    assert plan is not None and (plan.values, plan.slots) == (1, 2)  # h passes on; 16 iterations / 8 blocks
    assert [type(s).__name__ for s in plan.body] == ["Loop", "ClusterSync", "Loop", "ClusterSync", "WarpShuffle", "Cond"]
    compute, _, gather, _ = plan.body[:4]
    assert [type(s).__name__ for s in compute.body] == ["Let", "Loop", "Assign", "ClusterStash"]
    assert [type(s).__name__ for s in gather.body] == ["Let", "ClusterLoad", "Load", "Assign", "Accum"]


def test_small_or_uneven_work_stays_in_each_block():
    assert cluster.plan(column_recomputing_mlp(k=32)[0], 32, 8) is None  # 16 x 32 shared iterations: too few
    assert cluster.plan(column_recomputing_mlp(hidden=120)[0], 32, 8) is None  # 15 iterations a thread
    assert cluster.plan(column_recomputing_mlp(cols=12)[0], 24, 8) is None  # a cluster would span two rows
    assert cluster.plan(column_recomputing_mlp()[0], 16, 16) is None  # threads that are not one cell each
