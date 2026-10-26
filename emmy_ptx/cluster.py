"""Thread-block clusters for work emmy's schedule repeats across blocks.

Some emmy schedules give each block one output column and have every block
recompute a value that does not depend on the column. ``gelu(x @ w1) @ w2``,
fused into one kernel of 4096 blocks of 8 threads, recomputes the whole hidden
layer once for each of the 128 output columns. No lane of a block can share
that work, but the blocks that hold neighbouring columns can, through
distributed shared memory (Hopper and later):

* the kernel is launched in clusters of ``CLUSTER`` neighbouring blocks
  (``.reqnctapercluster``, no change to the launch itself);
* block r of a cluster computes the column-free part of its threads' loop
  iterations ``[r K/C, (r+1) K/C)`` and stashes the results in its own
  shared memory;
* after one cluster barrier every block reads all K results, each from the
  block that computed it (``mapa`` + ``ld.shared::cluster``), and runs the
  part that depends on its column; a second barrier keeps every block's
  shared memory alive until the others have read it.

Every value is computed exactly as before and folded in the same order, so
results are bit for bit the same (the rewrite is allowed under ``--strict``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Loop
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.leaves import Let

from .mapping import LOOPS, dependents, is_pure_loop, kind, own_reads, produced, trip_estimate, walk

# blocks a cluster: 8 is the portable maximum. Measured on the fused MLP (403 us without): 8 blocks
# 205 us, 4 blocks 349 us, 2 blocks 656 us (clusters of tiny blocks cost scheduling).
CLUSTER = 8
MIN_SHARED_WORK = 2048  # inner-loop iterations a thread's shared parts take, at least, to pay
MAX_VALUES = 4  # values a shared part may pass on
RANK = "_cluster_rank"  # the block's rank in its cluster (bound by the lowering)
STASH = "cluster_stash"  # the shared array the results go through


@dataclass(frozen=True)
class ClusterSync(Stmt):
    """``barrier.cluster.arrive`` + ``barrier.cluster.wait``: every thread of every block of the cluster."""


@dataclass(frozen=True)
class ClusterStash(Stmt):
    """Store ``values`` (f32) at place ``slot`` of this thread's part of the stash."""

    values: tuple[str, ...]
    slot: object  # an index expression

    def deps(self) -> tuple[str, ...]:
        return self.values

    def exprs(self) -> tuple:
        return (self.slot,)


@dataclass(frozen=True)
class ClusterLoad(Stmt):
    """``names`` = the values this thread's twin in block ``rank`` of the cluster stashed at ``slot``."""

    names: tuple[str, ...]
    rank: object  # index expressions
    slot: object

    def defines(self) -> tuple[str, ...]:
        return self.names

    def exprs(self) -> tuple:
        return (self.rank, self.slot)


@dataclass(frozen=True)
class Plan:
    body: tuple  # the Tile's new body
    values: int  # values each iteration stashes
    slots: int  # iterations each block computes for a thread (K / CLUSTER)


def plan(tile, grid: int, block: int) -> Plan | None:
    """The Tile body rewritten to share a loop's column-free work across a
    cluster, or None."""
    extents = [int(a.extent.as_static()) for a in tile.axes]
    n = math.prod(extents)
    if tile.aux_threads or tile.raster_axes or n != grid * block or grid % CLUSTER:
        return None
    # the coordinate that steps from one block to the next (its stride is exactly the block). Every
    # other axis must lie inside a block or across whole blocks, so that thread t of each block of a
    # cluster holds the same thread coordinates (its twins compute the same cells' shared parts)
    stride, column, threads = n, None, {}
    for axis, extent in zip(tile.axes, extents):
        stride //= extent
        if stride < block:
            if block % (stride * extent):
                return None
            threads[axis.name] = extent
        elif stride % block:
            return None
        elif stride == block and extent % CLUSTER == 0:
            column = axis.name
    if column is None:
        return None
    body = list(tile.body)
    for i, s in enumerate(body):
        found = _split(s, column, threads) if kind(s) in LOOPS else None
        if found is not None:
            new, values, slots = found
            return Plan(tuple(body[:i]) + new + tuple(body[i + 1:]), values, slots)
    return None


def _split(loop, column: str, threads: dict):
    """Loop ``loop`` as (cluster phases, values, slots), or None."""
    if not is_pure_loop(loop) or loop.unroll:
        return None
    extent = int(loop.axis.extent.as_static())
    if kind(loop) == "Loop":
        start, step, trips = Literal(0, "int"), 1, extent
    else:
        # each thread's iterations start + step*k: the start is a thread coordinate below the step
        if loop.end is not None or type(loop.step).__name__ != "Literal":
            return None
        step = int(loop.step.value)
        start = loop.start
        if type(start).__name__ != "Var" or threads.get(start.name) != step or extent % step:
            return None
        trips = extent // step
    if trips % CLUSTER:
        return None
    dep = dependents(loop.body, {column})
    own = [s for s in loop.body if (produced(s) - {"*"}) & dep]
    shared = [s for s in loop.body if s not in own]
    if not own or not shared or any(kind(s) == "Accum" for s in shared):
        return None  # nothing to share, nothing column-specific, or a fold carried across iterations
    made = set().union(*(produced(s) for s in shared))
    values = sorted(made & set().union(*(own_reads(t) for s in own for t in walk([s]))))
    if not values or len(values) > MAX_VALUES or not all(_f32(v, shared) for v in values):
        return None
    work = trips * sum(trip_estimate(t) for t in walk(shared) if kind(t) in LOOPS)
    if work < MIN_SHARED_WORK:
        return None
    slots = trips // CLUSTER
    var = loop.axis.name

    def index(k):  # the loop variable of iteration k of this thread
        return BinaryExpr("+", start, BinaryExpr("*", Literal(step, "int"), k))

    mine = BinaryExpr("+", BinaryExpr("*", Var(RANK), Literal(slots, "int")), Var("_cluster_k"))
    compute = Loop(axis=Axis("_cluster_k", Dim(slots)), seed=False, body=(
        Let(name=var, value=index(mine)), *shared, ClusterStash(tuple(values), Var("_cluster_k"))))
    k = Var("_cluster_j")
    gather = Loop(axis=Axis("_cluster_j", Dim(trips)), seed=loop.seed, body=(
        Let(name=var, value=index(k)),
        ClusterLoad(tuple(values), BinaryExpr("/", k, Literal(slots, "int")), BinaryExpr("%", k, Literal(slots, "int"))),
        *own))
    return (compute, ClusterSync(), gather, ClusterSync()), len(values), slots


def _f32(name: str, stmts) -> bool:
    """Is ``name`` an f32 value the statements compute (an Assign or a fold)?"""
    for s in walk(stmts):
        if kind(s) in ("Assign", "Accum") and s.name == name:
            return s.dtype is None or s.dtype.name == "f32"
    return False
