"""Which matmul Tiles coarsen.py gives two cells a thread, and how."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernels import staged_matmul  # noqa: E402

from emmy_ptx import coarsen  # noqa: E402


def kinds(stmts):
    return [type(s).__name__ for s in stmts]


def test_pairs_of_cells_share_one_inner_loop():
    body = coarsen.plan(staged_matmul()[0], 1, 512)
    assert body is not None
    assert kinds(body) == ["Init"] * 8 + ["Smem", "Smem", "Loop", "Cond"]  # each accumulator twice; one epilogue
    step = body[10].body
    assert kinds(step) == ["CpAsyncCopy", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "Sync", "Cond", "Sync"]
    lets, both = step[5].body[:2], step[5].body[2]
    assert kinds(lets) == ["Let", "Let"] and len(both.body) == 2 * 12  # both cells' loads and folds, one loop
    assert {s.name for s in both.body if type(s).__name__ == "Accum"} == {f"acc{i}{j}__m{m}" for i in range(2)
                                                                         for j in range(2) for m in range(2)}


def test_small_blocks_and_other_shapes_are_left_alone():
    assert coarsen.plan(staged_matmul()[0], 2, 256) is None  # too few threads a block
    assert coarsen.plan(staged_matmul(rows=64, cols=32)[0], 1, 512) is None  # 16 cell columns: under a warp
