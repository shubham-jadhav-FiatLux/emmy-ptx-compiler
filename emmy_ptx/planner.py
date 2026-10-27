"""One analysis of repeated work at every GPU scope, and the scope it pays to share it at.

Emmy's schedules give every cell (a Tile's unit of work) its own copy of a
computation, also when many cells compute the very same thing. The lowering's
recognisers each remove one form of that repetition: lanes split a reduction
or block a loop (4.2, 4.3), cells move or spread (4.4, 4.5), clusters share
work across blocks (4.8). This module looks at the repetition itself, the
same way at every scope.

A loop of a Tile's body depends on some of the Tile's axes: those its loads,
bounds and arithmetic read, directly or through the values defined before it.
Cells that differ only in the other axes compute the same instance of the
loop. Emmy lays the cells out row-major (the last axis fastest), so the
cells of one instance sit at regular gid strides, not always next to each
other: an aligned run of lanes (what the uniformity analysis of 4.1 sees) or
a strided set of them. For each scope the plan counts how many cells of one
scope instance compute each loop instance:

* warp: the 32 lanes of a warp;
* block: the threads of a block;
* cluster: 8 neighbouring blocks.

It weighs, in instructions a warp issues, the work sharing removes against
what it costs at that scope. Lanes of a warp run in lockstep, so a warp
whose lanes repeat a loop of cost C issues C whether one lane needs it or
all; shared among the R cells that compute it, each lane does C / R, a
saving of C (1 - 1/R) per warp. The cost: shuffles to merge a warp's partial
results; shared memory and a barrier for a block; cluster barriers and
distributed shared memory for a cluster. A barrier is paid once for all the
instances one exchange can carry (those of the enclosing loops), so its cost
is spread over them, as cluster.py does. The scope with the largest net
saving is the one to share at. `--report` prints
the plan beside what the recognisers did, and `tools/plan_survey.py` compares
the two over the problem sets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mapping import CROSS_THREAD, LOOPS, binds, kind, reads, trip_estimate, walk

WARP, CLUSTER = 32, 8
# rough costs, in instructions
TRANSCENDENTAL = {"exp": 16, "log": 16, "tanh": 20, "sigmoid": 18, "erf": 20, "sin": 24, "cos": 24, "pow": 40,
                  "divide": 4, "sqrt": 4, "rsqrt": 2, "reciprocal": 4}
SHUFFLE, SHARED_ACCESS, BARRIER, CLUSTER_BARRIER = 2, 4, 32, 1000


@dataclass(frozen=True)
class LoopPlan:  # (net savings and costs in instructions per warp, per instance of the loop)
    loop: str  # the loop's axis name
    key: tuple[str, ...]  # the Tile axes its work depends on
    repeats: int  # cells computing each instance, over the whole launch
    per_scope: dict  # scope -> cells of one scope instance computing the same loop instance
    lanes: str  # how a warp's repeating lanes lie: "aligned" (a power-of-two run), "run" (another length), "strided"
    cost: float  # the loop's work, in instructions per instance
    net: dict  # scope -> estimated instructions saved per scope instance (negative: not worth it)
    outer: tuple = ()  # the loops around it, outermost first

    @property
    def best(self) -> str:
        scope = max(self.net, key=self.net.get, default=None)
        return scope if scope is not None and self.net[scope] > 0 else "none"

    def describe(self) -> str:
        where = ", ".join(f"{self.per_scope[s]} a {s}" for s in ("warp", "block", "cluster"))
        lanes = {"aligned": "an aligned run of lanes", "run": "a run of lanes no power of two long",
                 "strided": "a strided set of lanes"}[self.lanes]
        key = ", ".join(self.key) or "no cell axis"
        return (f"plan: loop {self.loop} depends on {key}; each instance is computed by {self.repeats} cells "
                f"({where}; in a warp, {lanes}); cost about {self.cost:.0f} instructions; "
                f"best shared at: {self.best}" + (f", saving about {self.net[self.best]:.0f} a warp"
                                                  if self.best != "none" else ""))


def plan_tile(tile, block: int, grid: int) -> list[LoopPlan]:
    """The plan for every loop of Tile ``tile``, at any depth. (A rasterized
    Tile renumbers its blocks, not the cells inside one, so its warp and block
    figures hold as they are; its cluster figure assumes emmy's plain order.)"""
    if tile.aux_threads:
        return []
    axes = [a.name for a in tile.axes]
    extents = [int(a.extent.as_static()) for a in tile.axes]
    deps: dict[str, frozenset] = {a: frozenset({a}) for a in axes}
    out: list[LoopPlan] = []
    _visit(tile.body, deps, axes, extents, math.prod(extents), block, grid, out)
    return out


def _depends(names, deps) -> frozenset:
    return frozenset().union(*(deps.get(n, frozenset()) for n in names))


def _visit(stmts, deps: dict, axes, extents, cells: int, block: int, grid: int, out: list,
           batch: float = 1.0, outer: tuple = ()) -> None:
    """Plan each loop of ``stmts`` and those nested in it; ``deps`` maps every
    name bound so far to the Tile axes its value depends on, ``batch`` is the
    trips of the loops around ``stmts``."""
    for s in stmts:
        k = kind(s)
        if k in LOOPS:
            inside = binds((s,))
            key = _depends(reads((s,)) - inside, deps)
            out.append(_plan_loop(s, sorted(key, key=axes.index), axes, extents, cells, block, grid, batch, outer))
            inner = dict(deps)
            bounds = [x for x in (getattr(s, "start", None), getattr(s, "end", None)) if x is not None]
            inner[s.axis.name] = _depends({v for e in bounds for v in e.free_vars()}, deps)
            _visit(s.body, inner, axes, extents, cells, block, grid, out, batch * trip_estimate(s),
                   outer + (s.axis.name,))
        elif k == "Cond":
            _visit(s.body, dict(deps), axes, extents, cells, block, grid, out, batch, outer)
            _visit(s.else_body, dict(deps), axes, extents, cells, block, grid, out, batch, outer)
        for name in binds((s,)) - set(deps):  # what a statement defines depends on what it reads
            deps[name] = _depends(reads((s,)), deps)


def _plan_loop(loop, key: list[str], axes, extents, cells: int, block: int, grid: int, batch: float,
               outer: tuple = ()) -> LoopPlan:
    repeats = math.prod(e for a, e in zip(axes, extents) if a not in key)
    keyed = [i for i, a in enumerate(axes) if a in key]
    per_scope, lanes = {}, "aligned"
    for scope, size in (("warp", min(WARP, block)), ("block", block), ("cluster", block * CLUSTER)):
        n = min(size, cells, block * grid)
        gid = np.arange(n)
        coords, stride = [], 1
        for e in reversed(extents):
            coords.append((gid // stride) % e)
            stride *= e
        coords = coords[::-1]
        ident = np.zeros(n, np.int64)
        for i in keyed:
            ident = ident * extents[i] + coords[i]
        distinct = len(np.unique(ident))
        per_scope[scope] = max(1, n // max(distinct, 1))
        if scope == "warp" and per_scope[scope] > 1:  # how do lane 0's partners lie?
            mates = np.flatnonzero(ident == ident[0])
            if not np.array_equal(mates, np.arange(len(mates))):
                lanes = "strided"
            elif len(mates) & (len(mates) - 1):
                lanes = "run"
    cost = _cost(loop.body) * trip_estimate(loop)
    states = len({x.name for x in loop.body if kind(x) == "Accum"}) or 1
    comm = {
        "warp": states * math.log2(max(per_scope["warp"], 1)) * SHUFFLE,
        "block": states * (SHARED_ACCESS + math.log2(max(per_scope["block"], 1))) + BARRIER / batch,
        "cluster": states * 2 * SHARED_ACCESS + 2 * CLUSTER_BARRIER / batch,
    }
    net = {scope: cost * (1 - 1 / r) - comm[scope] for scope, r in per_scope.items() if r > 1}
    if any(kind(x) in CROSS_THREAD or kind(x) in ("Smem", "Sync") for x in walk(loop.body)):
        net = {}  # lanes, fragments or shared tiles already cooperate in it: not one cell's own work
    return LoopPlan(loop.axis.name, tuple(key), repeats, per_scope, lanes, cost, net, outer)


def _cost(stmts) -> float:
    """Instructions one pass over ``stmts`` takes, roughly (nested loops times their trips)."""
    total = 0.0
    for s in stmts:
        k = kind(s)
        if k in LOOPS:
            total += _cost(s.body) * trip_estimate(s)
        elif k == "Assign":
            total += TRANSCENDENTAL.get(s.op.name, 1)
        elif k in ("Load", "Accum", "Let", "Write"):
            total += 1
        elif k == "Cond":
            total += 1 + _cost(s.body) + _cost(s.else_body)
    return total


def plans(body, block: int, grid: int) -> list[LoopPlan]:
    """The plans of every Tile in a kernel body."""
    return [p for s in walk(body) if kind(s) == "Tile" for p in plan_tile(s, block, grid)]
