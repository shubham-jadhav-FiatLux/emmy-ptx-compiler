"""Bigger register tiles for emmy's SIMT matmuls.

Emmy's f32 matmul schedules stage tiles of A and B in shared memory with
cp.async and give each thread a small block of outputs, a cell (2 x 2 for a
512³ matmul on blocks of 512 threads). Every multiply-add then needs its own
shared-memory load: per step of k, a warp loads 64 values of B and two
broadcast values of A for 128 multiply-adds, and shared memory, not the FMA
units, sets the pace (m01's arithmetic alone takes 12.9 of its 15.0 µs).

Two cells that hold the same columns share their B values. So the threads of
the first half of the block each compute two vertically neighbouring cells,
rows 2p and 2p+1 of the cell grid, with one loop whose body does both cells'
work for each step of k: B is loaded once for both (the second, identical
loads disappear inside ptxas). The other half of the block only helps fill the
tiles. Measured on m01: the arithmetic 12.9 → 10.9 µs, the kernel 15.0 → 13.0
µs. Every cell's sum runs in the same order as before, so the results are bit
for bit the same.
"""

from __future__ import annotations

import math

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Let

from .mapping import LOOPS, binds, is_pure_loop, kind, reads, renamed, walk

FILL = ("Smem", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "Sync")
MIN_BLOCK = 512  # threads a block must have (half of them then compute: at least 8 warps)


def plan(tile, grid: int, block: int) -> tuple | None:
    """Tile ``tile``'s body with pairs of cells computed by one thread, or None."""
    extents = [int(a.extent.as_static()) for a in tile.axes]
    if tile.aux_threads or math.prod(extents) != grid * block or block < MIN_BLOCK:
        return None
    # exactly two thread axes, rows then columns, the columns one warp wide or a multiple of it
    inner = [(a.name, e) for a, e, s in zip(tile.axes, extents, _strides(extents)) if s < block]
    if len(inner) != 2 or inner[1][1] % 32 or inner[0][1] % 2 or inner[0][1] * inner[1][1] != block:
        return None
    rows = inner[0][0]
    body = list(tile.body)
    loops = [i for i, s in enumerate(body) if kind(s) in LOOPS]
    if len(loops) != 1:
        return None
    i = loops[0]
    head, k_loop, tail = body[:i], body[i], body[i + 1:]
    inits = [s for s in head if kind(s) == "Init"]
    if any(kind(s) not in FILL + ("Init",) for s in head) or not inits:
        return None
    accumulators = {s.name for s in inits}
    computes = [s for s in k_loop.body if kind(s) in LOOPS]
    if len(computes) != 1 or any(kind(s) not in FILL for s in k_loop.body if kind(s) not in LOOPS):
        return None
    compute = computes[0]
    if not is_pure_loop(compute) or compute.seed:
        return None
    folded = {s.name for s in walk(compute.body) if kind(s) == "Accum"}
    if not folded or not folded <= accumulators or rows not in reads([compute]):
        return None
    if any(kind(s) in FILL or kind(s) in LOOPS for s in walk(tail)):
        return None  # the epilogue: values and stores only

    def cell(j: int, stmts) -> tuple:
        """``stmts`` for cell row 2p + j: the row and every name they bind renamed."""
        names = (binds(stmts) | accumulators) - {compute.axis.name}
        mapping = {n: f"{n}__m{j}" for n in names}
        mapping[rows] = f"{rows}__m{j}"
        return tuple(renamed(s, mapping) for s in stmts)

    def row(j: int) -> Let:
        return Let(name=f"{rows}__m{j}", value=BinaryExpr("+", BinaryExpr("*", Literal(2, "int"), Var(rows)),
                                                       Literal(j, "int")))

    half = BinaryExpr("<", Var(rows), Literal(inner[0][1] // 2, "int"))
    both = Loop(axis=compute.axis, unroll=compute.unroll, seed=False,
                body=cell(0, compute.body) + cell(1, compute.body))
    k_body = tuple(Cond(cond=half, body=(row(0), row(1), both)) if s is compute else s for s in k_loop.body)
    new_head = tuple(x for s in head for x in ((*cell(0, (s,)), *cell(1, (s,))) if kind(s) == "Init" else (s,)))
    epilogue = (Cond(cond=half, body=(row(0), row(1), *cell(0, tail), *cell(1, tail))),) if tail else ()
    return new_head + (_with_body(k_loop, k_body),) + epilogue


def _strides(extents: list[int]) -> list[int]:
    out, stride = [], math.prod(extents)
    for e in extents:
        stride //= e
        out.append(stride)
    return out


def _with_body(loop, body: tuple):
    from dataclasses import replace
    return replace(loop, body=body)
