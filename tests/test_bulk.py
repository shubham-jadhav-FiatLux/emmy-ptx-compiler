"""The copy engine's plan (bulk.py): which cp.async fills are contiguous runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import replace  # noqa: E402

from test_gpu_lowering import pipelined_fills  # noqa: E402

from emmy_ptx import bulk  # noqa: E402

GLOBALS = {"x": ((128, 64), 4), "out": ((128, 64), 4), "out2": ((4, 512), 4)}
SHARED = {"s": ((64, 64), 4, 0), "u": ((1024,), 4, 16384)}


def test_a_contiguous_stage_is_one_run_and_a_scattered_fill_none():
    tile = pipelined_fills()[0]
    plan = bulk.plan(tile, 1, 512, GLOBALS, SHARED)
    runs = sorted(plan.values())
    assert runs == [512, 512]  # the prologue's fill and the loop's, each one whole stage
    assert all(s.smem == "s" for s in plan)


def swizzled(stmts) -> tuple:
    """Every cp.async fill of ``stmts`` with a swizzled destination."""
    out = []
    for s in stmts:
        if type(s).__name__ == "CpAsyncCopy":
            s = replace(s, swizzle="B128")
        elif hasattr(s, "body"):
            s = replace(s, body=swizzled(s.body))
        out.append(s)
    return tuple(out)


def test_a_swizzled_or_unaligned_fill_is_left_alone():
    tile = pipelined_fills()[0]
    assert bulk.plan(replace(tile, body=swizzled(tile.body)), 1, 512, GLOBALS, SHARED) == {}
    shifted = {"s": ((64, 64), 4, 4), "u": SHARED["u"]}  # the tile 4 bytes into its symbol
    assert bulk.plan(tile, 1, 512, GLOBALS, shifted) == {}


def test_a_launch_with_more_threads_than_cells_is_left_alone():
    assert bulk.plan(pipelined_fills()[0], 2, 512, GLOBALS, SHARED) == {}
