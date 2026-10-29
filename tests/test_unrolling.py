"""A complete unrolling copies every loop inside it. Copies of a loop that
stays rolled gain no straight-line code, so they are kept only while the
kernel stays within COPY_BUDGET instructions (Lowerer.run)."""

import pytest
from kernels import short_loop_around_a_long_one

from emmy_ptx.frontend import Buffer, KernelSource, Launch
from emmy_ptx.lower import Lowerer


def lowered(hidden: int, k: int = 200, cols: int = 32, **options):
    buffers = {"x": Buffer("x", (hidden, k), "f32"), "y": Buffer("y", (k, cols), "f32"),
               "w": Buffer("w", (hidden, cols), "f32"), "out": Buffer("out", (cols,), "f32")}
    src = KernelSource("k_test", tuple(short_loop_around_a_long_one(hidden, k, cols)), tuple(buffers), buffers,
                       Launch(1, cols, 0), frozenset({"out"}))
    return Lowerer(src, **options).run()


def test_copies_of_a_rolled_loop_stay_in_a_small_kernel():
    kernel = lowered(16)  # 16 copies of the 200-trip loop: about 1,300 instructions
    assert kernel.copies and "loop a: unrolled (16 trips)" in kernel.remarks
    assert not any(r.startswith("copies of a rolled loop undone") for r in kernel.remarks)


@pytest.mark.parametrize("options", [{}, {"strict": True}], ids=["optimized", "strict"])
def test_copies_in_a_large_kernel_are_undone(options):
    kernel = lowered(64, **options)  # 64 copies: past 4,096 instructions
    assert not kernel.copies and any(r.startswith("loop a: rolled") for r in kernel.remarks)
    assert any(r.startswith("copies of a rolled loop undone") for r in kernel.remarks)


def test_without_copies_the_short_loop_is_rolled():
    kernel = lowered(16, copy_rolled=False)
    assert not kernel.copies and any(r.startswith("loop a: rolled") for r in kernel.remarks)
