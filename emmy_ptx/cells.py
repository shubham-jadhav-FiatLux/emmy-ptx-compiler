"""Where a Tile's cells run (DESIGN sections 3, 4.4 and 4.5).

``stmt_Tile`` gives every thread its cell: the flat thread id decoded
row-major into the Tile's coordinates (``_decode``, with emmy's grouped CTA
rasterization), each coordinate marked with how many neighbouring lanes hold
it alike (the uniform groups of section 4.1). Two re-mappings let lanes share
more work: spread cells give a cell several lanes when threads would idle
(``_spread``), and placement moves cells so that the cells of a row fill
aligned groups of lanes (``_placement``, ``_placed_tile``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import mapping
from .builder import WARP, Value
from .common import STATIC_SMEM_LIMIT, lowbit
from .frontend import DEFAULT_BLOCK
from .lanes import SPLIT_MIN_TRIPS
from .mapping import LOOPS, kind, reads, walk


@dataclass(frozen=True)
class Placement:
    """Where the cells of a Tile run (see Lowerer._placement): grouped by every
    coordinate but ``axis``'s, the cells of each group on aligned groups of
    ``parts[k]`` lanes in region k, the regions one after another."""

    axis: int  # the Tile axis whose coordinate varies within a group
    parts: tuple[int, ...]  # lanes of a group in each region (powers of two, at most a warp)
    groups: int  # cells / the axis's extent


class CellLayout:
    """The cell-mapping rules of :class:`~emmy_ptx.lower.Lowerer` (a mixin)."""

    def stmt_Tile(self, s) -> None:
        """The Tile's body once per cell, each thread taking cell ``gid`` (or
        ``gid / spread``, or the one placement gives it), guarded where the
        launch has more threads than cells."""
        b = self.b
        extents = [int(a.extent.as_static()) for a in s.axes]
        n = math.prod(extents) if extents else 1
        cells = s.block_threads or DEFAULT_BLOCK
        spread = self._spread(s, n)
        if spread > 1:
            self.remark(f"cells: idle threads join them, {spread} lanes per cell")
        plan = self._placement(s, extents) if self.place and spread == 1 else None
        if plan is not None:
            self.remark(f"cells: each row's cells moved onto aligned groups of "
                        f"{' + '.join(map(str, plan.parts))} lanes, so they share its work")
            self.shared_loop = False
            self._placed_tile(s, plan, extents)
            self.placed_shared |= self.shared_loop
            return
        if s.aux_threads:
            per = b.int_const(s.block_threads)
            gid = b.add(b.mul(self.ctaid, per), b.rem(self.tid, per))
        else:
            gid = b.add(b.mul(self.ctaid, b.int_const(self.src.launch.block)), self.tid)
        # emmy guards when cells do not tile N; the range check also covers a
        # launch that is larger than the iteration space for any other reason
        lanes = n * spread
        needs_guard = s.aux_threads != 0 or (spread == 1 and n % cells != 0) or gid.hi >= lanes
        done = b.label("tile_end")
        if needs_guard:
            b.branch(done, b.icmp(">=", gid, b.int_const(lanes)))
            gid = b.assume(gid, gid.lo, lanes - 1)  # inside the guard
        # every lane of each warp runs the body, unless the guard can split a warp
        block = self.src.launch.block
        self.converged = not s.aux_threads and (not needs_guard or (lanes % WARP == 0 and block % WARP == 0))
        cell = b.shr(gid, b.int_const(spread.bit_length() - 1)) if spread > 1 else gid
        self.shared_loop, self.cell_threads = False, spread if spread > WARP else 1
        # the spread lanes of a cell compute its results alike: one of them stores them
        self.cell_writer = b.icmp("==", b.bitop("&", gid, b.int_const(spread - 1)), b.int_const(0)) if spread > 1 else None
        coords = self._decode(s, cell, extents, spread)
        with self.block(guarded=needs_guard):
            for name, v in zip((a.name for a in s.axes), coords):
                self.scope.names[name] = v
            copy_engine = spread == 1 and not needs_guard and self._bulk_begin(s)
            shared = self._cluster_body(s) if spread == 1 and not needs_guard else None
            if shared is None and spread == 1 and not needs_guard:
                shared = self._coarsened_body(s)
            self.body(shared if shared is not None else s.body)
            if copy_engine:
                self._bulk_end()
        # cells that already share their work across lanes (one query row for all, say) gain
        # nothing from more lanes each; cells that fold alone, one per lane, do (see run)
        self.wants_spread |= not self.shared_loop and self._spread(s, n, candidate=True) > 1
        plan = self._placement(s, extents) if spread == 1 else None
        if plan is not None:  # worth it where every lane group would be larger than the lanes a group has now
            alike = min((v.ug for i, v in enumerate(coords) if i != plan.axis and extents[i] > 1), default=WARP)
            self.wants_place |= min(plan.parts) > alike
        self.converged, self.cell_threads, self.cell_writer = False, 1, None
        b.place(done)

    def _spread(self, s, n: int, candidate: bool = False) -> int:
        """Lanes per cell of Tile ``s``: 1, unless the launch has at least twice
        as many threads as there are cells (and the kernel is lowered with
        ``spread``, or ``candidate`` asks what spreading would give).

        Some emmy schedules fold a whole row in one thread while most of the
        block idles (a 65536-element sum on one thread of 256). When nothing a
        cell does depends on which thread runs it, cell c may as well run on an
        aligned group of lanes (cell = gid / spread): they compute it alike, so
        the lane split of reductions (``_split_plan``) shares its folds among
        them and ``_dealt`` its independent loops. ``run`` lowers a kernel with
        spread cells only when, lowered without, the lanes shared no loop."""
        launch = self.src.launch
        if self.plain or not (self.spread or candidate) or s.aux_threads or not _thread_free(s.body):
            return 1
        if any(kind(x) == "Write" and x.atomic for x in walk(s.body)):
            return 1  # the lanes of a cell would each add its result
        if not any(kind(x) in LOOPS for x in walk(s.body)):
            return 1  # nothing to share
        limit = min(WARP, lowbit(launch.block))
        room = self._static_smem is not None and self._static_smem + self._lane_bytes + 4096 <= STATIC_SMEM_LIMIT
        if n == launch.grid and lowbit(launch.block) == launch.block and room:
            limit = launch.block  # one cell per block: it may span the warps (_merge_warps joins them)
        spread = 1
        while spread < limit and n * spread * 2 <= launch.grid * launch.block:
            spread *= 2
        return spread

    def _placement(self, s, extents: list[int]) -> Placement | None:
        """Where the cells of Tile ``s`` may run so that more lanes share work,
        or None.

        Emmy lays the cells out row-major, so lanes beside each other share a
        query row only when the row's cells fill aligned groups of lanes. With
        11 cells to a row (three output columns each, 32 columns), or cells
        rasterized lane by lane (consecutive lanes on consecutive rows), no
        group of lanes holds one row, and every lane computes its row's
        scores itself. Cells that do not depend on which thread runs them may
        run anywhere, though. Grouped by every coordinate but the fastest
        axis's, a group's cells go to aligned groups of lanes of power-of-two
        sizes: 20 = 16 + 4; 11 = 16 when there are threads enough, else 8 + 4
        with one lane idle, else 8 + 2 + 1 (_placed_tile). Worth trying only
        when some loop's work is the same along that axis."""
        launch = self.src.launch
        if self.plain or s.aux_threads or launch.block % WARP or not _thread_free(s.body):
            return None
        axis = max((i for i, e in enumerate(extents) if e > 1), default=None)
        if axis is None:
            return None
        along = mapping.dependents(s.body, {s.axes[axis].name})
        if not any(kind(x) in LOOPS and mapping.trip_estimate(x) >= SPLIT_MIN_TRIPS and not reads((x,)) & along
                   for x in walk(s.body)):
            return None  # no loop's work is the same for all cells of a group
        groups = math.prod(extents) // extents[axis]
        for parts in _decompositions(extents[axis]):
            if sum(_round_up(groups * part, WARP) for part in parts) <= launch.grid * launch.block:
                return Placement(axis, parts, groups)
        return None

    def _placed_tile(self, s, plan: Placement, extents: list[int]) -> None:
        """Tile ``s`` with its cells where ``plan`` puts them. The regions are
        whole warps, one after another: lane u of region k (P = plan.parts[k]
        lanes a group) takes group u / P, and that group's cell number
        (sum of the earlier parts) + u % P along the plan's axis. So every
        other coordinate is the same on aligned groups of P lanes, and the
        lowering of the body (lane splits and blocks) shares their work. A
        lane past the last group or the last cell computes a copy of the last
        one, so that whole groups take part, and stores nothing."""
        b = self.b
        names = [a.name for a in s.axes]
        size = extents[plan.axis]
        others = [i for i in range(len(extents)) if i != plan.axis]
        stride = {i: math.prod(extents[j] for j in others if j > i) for i in others}
        regions = [_round_up(plan.groups * part, WARP) for part in plan.parts]
        gid = b.add(b.mul(self.ctaid, b.int_const(self.src.launch.block)), self.tid)
        done = b.label("tile_end")
        if gid.hi >= sum(regions):
            b.branch(done, b.icmp(">=", gid, b.int_const(sum(regions))))
        base = first = 0
        for part, lanes in zip(plan.parts, regions):
            after = b.label("cells_end")
            if base + lanes < sum(regions):
                b.branch(after, b.icmp(">=", gid, b.int_const(base + lanes)))  # whole warps: no divergence
            self.converged = True
            with self.block(guarded=True):
                u = b.assume(b.sub(gid, b.int_const(base)), 0, lanes - 1)
                group = b.shr(u, b.int_const(part.bit_length() - 1))
                cell = b.add(b.bitop("&", u, b.int_const(part - 1)), b.int_const(first))
                own = None  # does the lane hold a cell of its own?
                clamped = lanes > plan.groups * part
                if clamped:  # lanes past the last group
                    own = b.icmp("<", group, b.int_const(plan.groups))
                    group = b.imin(group, b.int_const(plan.groups - 1))
                if first + part > size:  # lanes past the group's last cell
                    real = b.icmp("<", cell, b.int_const(size))
                    own = real if own is None else b.pand(own, real)
                    cell = b.imin(cell, b.int_const(size - 1))
                self.store_guard = own
                self.scope.names[names[plan.axis]] = cell
                for i in others:
                    coord = b.rem(b.div(group, b.int_const(stride[i])), b.int_const(extents[i]))
                    alike = part if clamped else part * lowbit(stride[i])  # lanes holding the same coordinate
                    self.scope.names[names[i]] = b.uniform(coord, min(WARP, alike))
                self.body(s.body)
            self.store_guard = None
            b.branch(done)
            b.place(after)
            base, first = base + lanes, first + part
        self.converged = False
        b.place(done)

    def _decode(self, s, gid: Value, extents: list[int], spread: int = 1) -> list[Value]:
        """Per-axis coordinates of flat cell id ``gid`` (row-major), with emmy's
        optional grouped CTA rasterization of two adjacent block axes; each cell
        runs on ``spread`` consecutive lanes."""
        b = self.b
        strides = [math.prod(extents[i + 1 :]) for i in range(len(extents))]
        # Lanes l of a warp hold flat ids base + l, base a multiple of the
        # largest power of two dividing both the block size and 32. So
        # (gid / stride) % extent is the same throughout each aligned group of
        # lowbit(stride) lanes, capped there: the coordinate's uniform group.
        cap = 1 if s.aux_threads else min(WARP, lowbit(self.src.launch.block))
        coords = []
        for i, (stride, extent) in enumerate(zip(strides, extents)):
            term = b.div(gid, b.int_const(stride))
            coord = term if i == 0 else b.rem(term, b.int_const(extent))
            coords.append(b.uniform(coord, min(cap, lowbit(stride) * spread)))
        raster = _raster_plan(s, extents)
        if raster is None:
            return coords
        gi, oi, g = raster
        ea, eb = extents[gi], extents[oi]
        mi, ni = min(gi, oi), max(gi, oi)
        sub = b.div(gid, b.int_const(strides[ni]))
        if mi != 0:
            sub = b.rem(sub, b.int_const(ea * eb))
        gsize = b.int_const(g * eb)
        if ea % g == 0:
            a = b.add(b.mul(b.div(sub, gsize), b.int_const(g)), b.rem(sub, b.int_const(g)))
            o = b.div(b.rem(sub, gsize), b.int_const(g))
        else:
            grp = b.div(sub, gsize)
            size = b.imin(b.int_const(g), b.sub(b.int_const(ea), b.mul(grp, b.int_const(g))))
            a = b.add(b.mul(grp, b.int_const(g)), b.rem(b.rem(sub, gsize), size))
            o = b.div(b.rem(sub, gsize), size)
        group = min(cap, lowbit(strides[ni]) * spread)  # both are functions of gid / strides[ni]
        coords[gi] = b.uniform(b.assume(a, 0, ea - 1), group)
        coords[oi] = b.uniform(b.assume(o, 0, eb - 1), group)
        return coords


def _thread_free(body) -> bool:
    """Does what the statements compute not depend on which thread runs them?
    No cross-thread statement, shared memory or thread index."""
    for s in walk(body):
        k = kind(s)
        if k in mapping.CROSS_THREAD or k in ("Smem", "ZeroPrologue"):
            return False
        if any(_names_thread(e) for e in getattr(s, "exprs", lambda: ())()):
            return False
    return not reads(body) & {"lane", "warp"}


def _names_thread(e) -> bool:
    """Does expression ``e`` read a thread or block index (a Builtin)?"""
    if kind(e) == "Builtin":
        return True
    return any(_names_thread(c) for c in vars(e).values() if hasattr(c, "free_vars")) or any(
        _names_thread(c) for v in vars(e).values() if isinstance(v, tuple) for c in v if hasattr(c, "free_vars"))


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _decompositions(n: int):
    """Aligned lane groups for n cells (powers of two up to a warp), most
    sharing first: past the whole warps, the rest in one padded group; the
    largest power of two below it and one padded group; exactly n lanes."""
    whole, rest = (WARP,) * (n // WARP), n % WARP
    if not rest:
        yield whole
        return
    yield whole + (_pow2_ceil(rest),)
    big = 1 << (rest.bit_length() - 1)
    if big < rest:
        two = (big, _pow2_ceil(rest - big))
        yield whole + two
        exact = tuple(1 << i for i in reversed(range(rest.bit_length())) if rest >> i & 1)
        if exact != two:
            yield whole + exact


def _raster_plan(s, extents):
    if not (s.raster_group and s.raster_axes):
        return None
    names = [a.name for a in s.axes]
    if s.raster_axes[0] not in names or s.raster_axes[1] not in names:
        return None
    mi, ni = names.index(s.raster_axes[0]), names.index(s.raster_axes[1])
    if ni != mi + 1:
        return None
    gi, oi = (mi, ni) if s.raster_orient == "m" else (ni, mi)
    g = min(int(s.raster_group), extents[gi])
    if extents[gi] > 1 and g > 1:
        return gi, oi, g
    return None

