"""Lanes that compute the same thing share it (DESIGN section 4).

Emmy's schedules often give a group of lanes the same reduction, or the same
part of each iteration (a query row's attention scores). Before lowering a
loop as it stands, ``Lowerer._loop_body`` asks here whether its lanes can
share it:

* ``_split_plan``: a reduction whose folded values are uniform over G lanes
  is split across them and merged with a butterfly (``_split_reduction``);
* ``_blocked``: lane blocking, for loops whose iterations share only part of
  their work (``_lower_blocked``, with the rewrite in ``mapping.blocked``);
* ``_dealt``: independent iterations dealt out to the lanes (``_deal``).

The statements that lane blocking rewrites a loop into (``LaneSlot``,
``LaneStash``, ``LaneBroadcast``, ``LaneButterfly``) are lowered here too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Var
from emmy.compiler.ir.stmt import Assign, Cond, Loop
from emmy.compiler.ir.stmt.leaves import Let

from . import mapping
from .builder import WARP, Value, Variable
from .common import (AUTO_UNROLL_TRIPS, MEM_TYPE, REDUCE_CANON, STATIC_SMEM_LIMIT, TARGET_SMS, UNROLL_BUDGET,
                     body_size, int_lit)
from .mapping import LOOPS, LaneSlot, assigned, kind, reads, walk
from .ptx import SIZEOF, Imm, Mem, Sym

# A reduction shared by a group of lanes is split across them only when it has
# at least this many trips (the shuffle tree costs log2(group) steps).
SPLIT_MIN_TRIPS = 16
# Associative, commutative combines a lane-split reduction may regroup.
SPLITTABLE = ("add", "multiply", "maximum", "minimum")
# Bytes of an L1 cache line: a warp load costs one pass per line it touches.
LINE_BYTES = 128
# A split lane lowers consecutive iterations together (unroll-and-jam) only
# while their inner loops keep at most this many accumulators live.
JAM_ACCUMULATORS = 64
# Registers of an SM (and so of one CTA), and the most one thread can address.
REGISTER_FILE, THREAD_REGISTERS = 65536, 256
# Shared array through which lane-blocked loops broadcast (stmt_LaneStash), when
# a block broadcasts at least LANE_BUFFER_MIN values per lane.
LANE_BUFFER, LANE_BUFFER_MIN = "smem_lanes", 8
# Lanes that split one iteration's shared part when one lane per iteration would
# read a cache line per lane (see Lowerer._key_lanes): a lane per this many bytes
# of the iteration's row, at most KEY_LANES (8 lanes x 16 bytes cover a line).
ROW_BYTES_PER_LANE, KEY_LANES = 64, 8
# A straight-line reduction split across lanes gives each lane one contiguous
# chunk while chunks are at most this long; past it the lanes take vectors side by
# side, in rounds that form a loop of their own beyond SPLIT_UNROLLED_ROUNDS.
CHUNK_TRIPS, SPLIT_UNROLLED_ROUNDS = 16, 16


@dataclass(frozen=True)
class Fold:
    """A loop's reduction as lanes can share it (see Lowerer._fold)."""

    group: int  # aligned lanes that fold the same values
    states: tuple[str, ...]  # the carried accumulators
    # independent merges: (states, Assigns merging a partner lane's copies, read as "<state>$b")
    merges: tuple[tuple[tuple[str, ...], tuple], ...]


@dataclass(frozen=True)
class SplitPlan:
    """How a lane-shared reduction is dealt to its lanes (see Lowerer._split_plan)."""

    fold: Fold
    group: int  # lanes the iterations are split over (a power of two)
    vector: int  # consecutive iterations a lane takes at a time
    jam: int  # consecutive iterations of a lane lowered as one body


class LaneSharing:
    """The lane-sharing rules of :class:`~emmy_ptx.lower.Lowerer` (a mixin)."""

    def _shared_by(self, s) -> int:
        """Aligned lanes that compute loop ``s``'s reduction alike (0: not a lane-shareable reduction)."""
        extent = self.b.int_const(int(s.axis.extent.as_static()))
        if kind(s) == "Loop":
            bounds = (self.b.int_const(0), self.b.int_const(1), extent)
        else:
            bounds = (self.expr(s.start), self.expr(s.step), self.expr(s.end) if s.end is not None else extent)
        fold = self._fold(s, *bounds)
        return fold.group if fold is not None else 0

    def _split_plan(self, s, start: Value, step: Value, end: Value, trips) -> SplitPlan | None:
        """How to split reduction loop ``s`` across the lanes that share it, or None.

        Emmy's tile decode often gives every lane of an aligned group the same
        reduction to compute (all lanes of a warp take the same query rows in
        attention, and each computes the same dot products). When the folded
        values are uniform over groups of G lanes (``_fold``), those G lanes can
        split the iterations and combine their partial results with log2(G)
        shuffles: the same reduction, regrouped, which an associative and
        commutative combine allows. The loop qualifies when every lane of the
        warp runs it and its trip count is known. A body without inner loops
        gives each lane one contiguous chunk of the iterations (their loads
        merge into vector loads).

        A reduction nested in another can be split at either level. Splitting
        the outer loop leaves each lane whole inner loops and one combine at the
        end; splitting the inner one combines once per outer iteration. Which is
        cheaper is mostly a question of coalescing: the lanes of a split loop
        read the addresses of consecutive iterations, one cache line for a loop
        along a row but a line per lane for a loop down a column. So ``s`` is
        split only when that costs fewer line accesses and shuffles than
        splitting the loops in it (the coalescing-driven choice of the loop to
        parallelize, as in PPCG and Yang et al., PLDI'10).
        """
        if self.plain or self.strict or self.off("split") or not self.converged or trips is None or trips[0] != trips[1]:
            return None  # (strict: a split regroups the reduction)
        if trips[0] < SPLIT_MIN_TRIPS:
            return None
        fold = self._fold(s, start, step, end)
        n = trips[0]
        group = 0 if fold is None else min(fold.group, 1 << (n.bit_length() - 1))
        if group < 2:
            return None
        nested = self._nested_split_cost(s, n, group)
        deltas = self._deltas(s.body, s.axis.name, step.const)
        if nested is None and group == WARP and n >= 2 * self.cell_threads > 2 * WARP:
            group = self.cell_threads  # a cell spanning the warps of its block: all of its threads split it
        if nested is None and n <= group * CHUNK_TRIPS:  # a short straight-line body: one contiguous chunk per lane
            return SplitPlan(fold, group, n // group if n % group == 0 else 1, 1)
        if nested is None:  # a long one: lanes side by side, a vector each (coalesced)
            return SplitPlan(fold, group, _split_vector(deltas, n, group, whole=False), 1)
        if nested < n / group * _line_cost(deltas, group) + math.log2(group) * _merge_cost(fold):
            return None
        vector = _split_vector(deltas, n, group)
        return SplitPlan(fold, group, vector, self._jam_factor(s, fold, n, group, vector))

    def _fold(self, s, start: Value, step: Value, end: Value) -> Fold | None:
        """Loop ``s``'s reduction as lanes can share it, or None when it is not one.

        ``s`` must seed its accumulators and hold only values, folds and loops
        of them (mapping.is_pure_loop). Each top-level accumulation must be an
        associative, commutative fold (SPLITTABLE) or part of an online softmax
        (mapping.online_groups), and nothing else may read the accumulators
        mid-fold. ``group`` is how many aligned lanes see the same values in
        every fold. Each of ``merges`` combines a partner lane's partial state
        into one independent part of the fold: a plain fold's own combine, and
        for an online softmax the combine derived from its update
        (mapping.combine_program).
        """
        if not s.seed or not mapping.is_pure_loop(s):
            return None
        groups = mapping.online_groups(s.body)
        accs = [st for st in s.body if kind(st) == "Accum"]
        if groups is None or not accs or any(st.dtype is not None and st.dtype.name != "f32" for st in accs):
            return None
        in_group = {id(x) for g in groups for x in g.stmts}
        plain = {st.name: st for st in accs if id(st) not in in_group}
        if any(st.base is not None or REDUCE_CANON.get(st.op.name) not in SPLITTABLE for st in plain.values()):
            return None
        states = tuple(dict.fromkeys(st.name for st in accs))
        if assigned(s.body) - mapping.local_names(s.body) != set(states):
            return None  # a nested fold updates some outer name: it would need a merge of its own
        hidden = set(states) | {x.name for g in groups for x in g.stmts if kind(x) == "Assign"}
        for st in walk(s.body):  # nothing but the folds may see the running state
            own = {st.name} if kind(st) == "Accum" and id(st) not in in_group else set()
            if id(st) not in in_group and mapping.own_reads(st) & hidden - own:
                return None
        loop_ug = min(start.ug, step.ug, end.ug)
        of_name = self._uniformity(s, loop_ug)
        folded = [st.value for st in plain.values()] + [x for g in groups for x in (g.s, *(w for _, _, w in g.sums))]
        group = min([loop_ug] + [of_name(x) for x in folded])
        merges = [((st.name,), (Assign(st.name, REDUCE_CANON[st.op.name], (st.name, f"{st.name}$b"), st.dtype),))
                  for st in plain.values()]
        for g in groups:
            merges.append(((g.m, *(l for l, _, _ in g.sums)),
                           mapping.combine_program(g, lambda n: f"{n}$b", lambda n: f"{n}$c")))
        return Fold(group, states, tuple(merges))

    def _uniformity(self, s, loop_ug: int):
        """``ug(name)`` for the values one iteration of loop ``s`` computes: how
        many aligned lanes compute each alike when the loop variable is uniform
        over ``loop_ug`` lanes (a name bound outside the loop keeps its own)."""
        env: dict[str, int] = {s.axis.name: loop_ug}

        def of_name(n: str) -> int:
            if n in env:
                return env[n]
            bound = self.scope.lookup(n)
            return bound.ug if isinstance(bound, (Value, Variable)) else 1

        def of_expr(e) -> int:
            return min([of_name(v) for v in e.free_vars()] + [WARP])

        def visit(stmts) -> None:
            for st in stmts:
                k = kind(st)
                if k == "Load":
                    for n in st.names:
                        env[n] = min([of_expr(e) for e in st.index] + [WARP])
                elif k == "Assign":
                    env[st.name] = min([of_name(a) for a in st.args] + [WARP])
                elif k == "Let":
                    env[st.name] = of_expr(st.value)
                elif k == "Select":
                    env[st.name] = min([min(of_name(br.value), of_expr(br.select)) for br in st.branches] + [WARP])
                elif k == "Accum":
                    folded = [st.value] + ([st.base] if st.base is not None else [])
                    env[st.name] = min([env.get(st.name, WARP)] + [of_name(x) for x in folded])
                elif k in LOOPS:  # a nested loop: every lane runs it whole
                    env[st.axis.name] = WARP if k == "Loop" else min([of_expr(e) for e in st.exprs()] + [WARP])
                    if k == "StridedLoop" and st.end is not None:
                        env[f"{st.axis.name}_end"] = of_expr(st.end)
                    visit(st.body)
                else:  # a branch's bodies (what else there is binds nothing)
                    for child in st.nested():
                        visit(child)

        visit(s.body)
        return of_name

    def _blocked(self, s, start: Value, step: Value, end: Value, trips) -> mapping.Blocked | None:
        """Loop ``s`` dealt to the lanes that share part of each iteration, in
        blocks (mapping.blocked), or None.

        In attention every lane holding the same query rows computes the same
        scores, but folds them into its own output columns, so the loop is no
        lane-shared reduction (``_split_plan``). Instead the G lanes of each
        group take the block's iterations in turn for the scores and broadcast
        the results; G is the largest group (a power of two, at most the trip
        count) for which the shared part still holds an inner loop.

        Slots taking consecutive iterations read the shared inputs the
        iteration stride apart. With the keys stored row by row (k[t, d]
        rather than kt[d, t]), one lane per key would read a cache line per
        lane, so a slot then has several lanes (``_key_lanes``), which split
        the key's dot product and read its row coalesced."""
        if self.plain or self.off("blocking") or not self.converged or trips is None or trips[0] != trips[1]:
            return None
        if None in (start.const, step.const) or not s.seed or not mapping.is_pure_loop(s):
            return None
        accs = [st for st in s.body if kind(st) == "Accum"]
        if not accs or any(st.dtype is not None and st.dtype.name != "f32" for st in accs):
            return None
        if assigned(s.body) - mapping.local_names(s.body) != {st.name for st in accs}:
            return None  # a nested fold updates some outer name
        if self.strict and mapping.online_groups(s.body) != []:
            return None  # an online softmax updated once per block of keys rounds differently
        n = trips[0]
        of_name = self._uniformity(s, min(start.ug, step.ug, end.ug))
        group = min(WARP, 1 << (n.bit_length() - 1))
        first, step_ = int(start.const), int(step.const)
        while group >= 2:
            uniform = (lambda g: lambda x: of_name(x) >= g)(group)
            plan = mapping.blocked(s, n, first, step_, group, 1, uniform)
            if plan is not None:
                share = 1 if self.strict else self._key_lanes(plan, s.axis.name, step_, group)  # (lanes split a dot)
                plan = mapping.blocked(s, n, first, step_, group, share, uniform) if share > 1 else plan
                if plan is not None:
                    online = "; the online softmax updated once per block" if mapping.online_groups(s.body) else ""
                    self.remark(f"loop {s.axis.name} ({n} trips): lane-blocked, groups of {group} lanes taking "
                                f"{group // share} iterations a block, {share} lane{'s' if share > 1 else ''} each{online}")
                return plan
            group //= 2
        return None

    def _key_lanes(self, plan: mapping.Blocked, var: str, step: int, group: int) -> int:
        """Lanes per slot of a blocked loop. One while slots taking consecutive
        iterations read the shared part's inputs coalesced. When each
        iteration's inputs are a row of their own (k[t, d]), more lanes per
        slot read a row in fewer lines but merge their partial dot products
        with shuffles and leave fewer iterations per block to share the
        softmax update: a lane per 64 bytes of row was best measured (twice
        that on a full SM, where L1 bandwidth is what runs out), up to
        KEY_LANES."""
        rows = [abs(d) for _, d, _ in self._deltas(plan.shared, var, step) if d != 0]
        if all(d is not None and d < LINE_BYTES for d in rows):
            return 1
        inner = min((mapping.trip_estimate(st) for st in plan.shared if kind(st) in LOOPS), default=0)
        if inner < SPLIT_MIN_TRIPS:
            return 1
        full = self.src.launch.block // WARP * -(-self.src.launch.grid // TARGET_SMS) >= WARP  # warps per SM
        lanes = max((d for d in rows if d is not None), default=LINE_BYTES) // ROW_BYTES_PER_LANE * (2 if full else 1)
        return max(1, min(KEY_LANES, group // 2, 1 << (max(lanes, 1).bit_length() - 1)))

    def _lower_blocked(self, plan: mapping.Blocked) -> None:
        """The blocks of a lane-blocked loop, then the lanes' partial sums merged.

        A lane reads whole query rows for its own iteration of each block, the
        same values in every block. Unrolled blocks keep them in registers
        (value numbering shares them), which pays while they take at most half
        the registers a thread of this launch may have (measured); beyond that
        the blocks run as a rolled loop that leaves those loads in place
        (ptxas still keeps what fits)."""
        self.mutable |= plan.mutable
        self.shared_loop = True
        self._lane_region = self._lane_rows(plan)
        loop = plan.loop
        try:
            if loop is not None:
                b = self.b
                n, size = int(loop.axis.extent.as_static()), body_size(loop.body)
                registers = min(THREAD_REGISTERS, REGISTER_FILE // self.src.launch.block)
                unrolls = (loop.unroll or n <= AUTO_UNROLL_TRIPS) and n * size <= UNROLL_BUDGET
                if unrolls and plan.invariant <= registers // 2:
                    with self.block(dominated=True):
                        self._loop_body(loop, b.int_const(0), b.int_const(1), b.int_const(n), (n, n), size)
                else:
                    self._rolled(loop, b.int_const(0), b.int_const(1), b.int_const(n), (n, n), 1, hoist_loads=False)
            with self.block(dominated=True):
                self.body(plan.rest)
        finally:
            self._lane_region, self._stashed = None, {}
        for merge in plan.merges:
            self.stmt_LaneButterfly(merge)

    def _lane_rows(self, plan: mapping.Blocked):
        """``(byte offset, rows per warp)`` of a part of the lane buffer for the
        broadcasts of ``plan`` (a row of 32 words per broadcast value), or None
        when a block broadcasts too little to pay for the barriers (fewer than
        LANE_BUFFER_MIN values per lane, measured), or the buffer does not fit
        next to the static shared arrays (or the kernel has a dynamic pool,
        which its launch sizes exactly)."""
        stmts = ((plan.loop,) if plan.loop is not None else ()) + plan.rest
        rows = max((len(st.values) for st in walk(stmts) if kind(st) == "LaneStash"), default=0)
        one_block = (plan.loop,) if plan.loop is not None else plan.rest
        per_block = sum(1 for st in walk(one_block) if kind(st) == "LaneBroadcast")
        size = -(-self.src.launch.block // WARP) * rows * WARP * 4
        if per_block < LANE_BUFFER_MIN or self._static_smem is None:
            return None
        if self._static_smem + self._lane_bytes + size > STATIC_SMEM_LIMIT:
            return None
        offset = self._lane_bytes
        self._lane_bytes += size
        return offset, rows

    def _dealt(self, s, start: Value, step: Value, end: Value, trips) -> int:
        """Lanes the iterations of loop ``s`` may be dealt to (0: none).

        A loop that carries nothing from one iteration to the next and whose
        iterations store to different places can run its iterations in any
        order, so when the lanes of an aligned group would all compute the
        same iteration alike (the lanes of a spread cell, ``_spread``), lane r
        can take iterations r, r + G, ... instead: the same stores, G times
        fewer each. Each buffer is stored by one statement whose index moves
        with the loop variable, and the loop reads no buffer it stores."""
        if self.plain or self.off("dealing") or not self.converged or trips is None or trips[0] != trips[1] or trips[0] < 2:
            return 0
        if self.store_guard is not None:
            return 0  # a lane that stores nothing must not take iterations of its own
        if any(kind(x) in mapping.CROSS_THREAD or kind(x) == "Init" for x in walk(s.body)):
            return 0
        if assigned(s.body) - mapping.local_names(s.body):
            return 0  # a fold carried across the iterations
        writes = [x for x in walk(s.body) if kind(x) == "Write"]
        outputs = [w.output for w in writes]
        if not writes or len(set(outputs)) < len(outputs) or any(w.atomic for w in writes):
            return 0
        if {x.input for x in walk(s.body) if kind(x) == "Load"} & set(outputs):
            return 0
        lets = mapping.index_lets(s.body)
        if any(not mapping.flat_coefficient(w.index, self.buffer(w.output).shape, s.axis.name, lets) for w in writes):
            return 0  # an iteration's store does not move with it (or not provably)
        of_name = self._uniformity(s, min(start.ug, step.ug, end.ug))
        names = (mapping.binds(s.body) | reads(s.body)) - {st.axis.name for st in walk(s.body) if kind(st) in LOOPS}
        group = min([WARP] + [of_name(n) for n in names])
        if group == WARP and self.cell_threads > WARP:
            group = self.cell_threads  # a cell spanning its block's warps
        return group if group >= 2 else 0

    def _deal(self, s, start: Value, step: Value, end: Value, trips: int, group: int) -> None:
        """Loop ``s`` with lane r of each aligned group of ``group`` running
        iterations r, r + G, ... (``_dealt``): a loop over the rounds, and the
        last, partial round only on the lanes it reaches."""
        b, t = self.b, s.axis.name
        rounds, rest = divmod(trips, group)

        def at(index):  # Let t = start + (index * group + rank) * step
            i = BinaryExpr("+", BinaryExpr("*", index, int_lit(group)), Var(f"{t}$rank"))
            return Let(name=t, value=BinaryExpr("+", Var(f"{t}$start"), BinaryExpr("*", i, Var(f"{t}$step"))))

        slot = LaneSlot(f"{t}$rank", group, 1)
        writer, self.cell_writer = self.cell_writer, None  # every lane stores the iterations it takes
        with self.block(dominated=True):
            self.scope.names[f"{t}$start"], self.scope.names[f"{t}$step"] = start, step
            if rounds:
                loop = Loop(axis=Axis(f"{t}$round", Dim(rounds)), body=(slot, at(Var(f"{t}$round")), *s.body),
                            unroll=s.unroll, seed=False)
                with self.block(dominated=True):
                    self._loop_body(loop, b.int_const(0), b.int_const(1), b.int_const(rounds), (rounds, rounds),
                                    body_size(loop.body))
            if rest:
                self.body((slot, Cond(cond=BinaryExpr("<", Var(f"{t}$rank"), int_lit(rest)),
                                      body=(at(int_lit(rounds)), *s.body))))
        self.cell_writer = writer

    def _deltas(self, stmts, var: str, step: int) -> list:
        """``(runs per iteration, bytes between the addresses of consecutive
        iterations of ``var`` or None, element bytes)`` for every load in ``stmts``."""
        lets = mapping.index_lets(stmts)
        out = []
        for ld, runs in mapping.weighted_loads(stmts):
            buf = self.buffer(ld.input)
            size = SIZEOF.get(MEM_TYPE[ld.dtype.name] if ld.dtype is not None else buf.type, 1)
            c = mapping.flat_coefficient(ld.index, buf.shape, var, lets)
            out.append((runs, None if c is None else c * step * size, size))
        return out

    def _nested_split_cost(self, s, trips: int, group: int):
        """``_split_plan``'s cost of leaving ``s`` whole and splitting the
        reductions directly inside it instead (None when it has none)."""
        inner = [st for st in s.body if kind(st) in LOOPS]
        if not inner:
            return None
        cost = trips * sum(1 for st in s.body if kind(st) == "Load")  # every lane, the same line
        for st in inner:
            runs = mapping.trip_estimate(st)
            step = st.step.value if kind(st) == "StridedLoop" and kind(st.step) == "Literal" else 1
            deltas = self._deltas(st.body, st.axis.name, int(step))
            accs = len({x.name for x in st.body if kind(x) == "Accum"})
            if st.seed and accs and mapping.is_pure_loop(st):
                cost += trips * (runs / group * _line_cost(deltas, group) + math.log2(group) * 2 * accs)
            else:  # each lane runs it whole
                cost += trips * runs * sum(r for r, _, _ in deltas)
        return cost

    def _jam_factor(self, s, fold: Fold, trips: int, group: int, vector: int) -> int:
        """Consecutive iterations a split lane lowers as one body (unroll-and-jam):
        worth it when the body has inner loops, which fusing the copies merges
        (one load of what they share, independent chains, vector loads across
        the copies)."""
        inner = [st for st in s.body if kind(st) in LOOPS]
        if not inner or trips % (vector * group):
            return 1
        if any(self.scope.lookup(n) is not None for n in mapping.binds(s.body) - set(fold.states)):
            return 1  # a body name also bound outside: renaming the copies apart could capture it
        live = len(assigned(inner))
        jam = vector if vector > 1 else next((j for j in (4, 2) if (trips // group) % j == 0), 1)
        while jam > 1 and jam * live > JAM_ACCUMULATORS:
            jam //= 2
        return jam

    def _split_reduction(self, s, start: Value, step: Value, end: Value, trips: int, plan: SplitPlan) -> None:
        """Lane rank r of each aligned group folds its share of the iterations;
        a butterfly of shuffles then merges the partial states (``_merge_lanes``),
        leaving every lane of the group the whole reduction.

        When the trips divide evenly, a lane takes ``plan.vector`` consecutive
        iterations at a time, t = V*r + V*G*k + j (j < V): with V = trips/G
        one contiguous chunk per lane, with a smaller V rounds in which the
        warp's loads cover whole vectors side by side (blocked-cyclic). Each
        ``plan.jam`` consecutive iterations of a lane are lowered as one body,
        so their inner loops fuse (unroll-and-jam). Otherwise lanes take every
        G-th iteration, the last round only on the lanes it reaches. More than
        SPLIT_UNROLLED_ROUNDS rounds become a loop of their own (``_split_rounds``).
        """
        b = self.b
        group, vector = plan.group, plan.vector
        self.shared_loop = True
        rank = self._rank(group)
        rounds = trips // (vector * group)
        if rounds > SPLIT_UNROLLED_ROUNDS:
            self._split_rounds(s, start, step, end, trips, plan, rank)
        elif trips % (vector * group) == 0:
            first = b.mul(rank, b.int_const(vector))
            offsets = [vector * group * k + j for k in range(trips // (vector * group)) for j in range(vector)]
            for i in range(0, len(offsets), plan.jam):
                values = [b.add(start, b.mul(b.add(first, b.int_const(o)), step)) for o in offsets[i : i + plan.jam]]
                self._jammed(s, values, end, plan.fold.states)
        else:  # interleaved, the last round only on the lanes it reaches
            rounds = -(-trips // group)
            for j in range(rounds):
                v = b.add(start, b.mul(b.add(rank, b.int_const(group * j)), step))
                self._guarded_iteration(s.axis.name, v, s.body, end, guard=j == rounds - 1)
        self._merge_lanes(plan.fold, group)

    def _split_rounds(self, s, start: Value, step: Value, end: Value, trips: int, plan: SplitPlan, rank: Value) -> None:
        """``_split_reduction``'s rounds as a loop: round k gives lane r the
        ``plan.vector`` iterations from V*r + V*G*k (lowered as one body), and
        the iterations past the last whole round go to the lanes one each."""
        b = self.b
        group, vector, t = plan.group, plan.vector, s.axis.name
        span = vector * group
        rounds, index = trips // span, f"{t}$round"
        first = BinaryExpr("*", BinaryExpr("+", BinaryExpr("*", Var(index), int_lit(group)), Var(f"{t}$rank")),
                           int_lit(vector))

        def value(j):  # start + (first + j) * step
            i = BinaryExpr("+", first, int_lit(j))
            return BinaryExpr("+", Var(f"{t}$start"), BinaryExpr("*", i, Var(f"{t}$step")))

        body = (LaneSlot(f"{t}$rank", group, 1), *(Let(name=f"{t}${j}", value=value(j)) for j in range(vector)),
                *mapping.jammed(s.body, t, vector, set(plan.fold.states)))
        loop = Loop(axis=Axis(index, Dim(rounds)), body=body, seed=False)
        with self.block(dominated=True):
            self.scope.names[f"{t}$start"], self.scope.names[f"{t}$step"] = start, step
            self._loop_body(loop, b.int_const(0), b.int_const(1), b.int_const(rounds), (rounds, rounds),
                            body_size(loop.body))
        for j in range(-(-(trips - rounds * span) // group)):
            v = b.add(start, b.mul(b.add(rank, b.int_const(rounds * span + group * j)), step))
            self._guarded_iteration(t, v, s.body, end, guard=rounds * span + group * (j + 1) > trips)

    def _jammed(self, s, values: list[Value], end: Value, states) -> None:
        """Iterations ``values`` of loop ``s``, lowered as one body (mapping.jammed)."""
        if len(values) == 1:
            self._iteration(s.axis.name, values[0], s.body, end)
            return
        with self.block(dominated=True):
            for j, v in enumerate(values):
                self.scope.names[f"{s.axis.name}${j}"] = self.b.assume(v, v.lo, end.hi - 1) if end.hi is not None else v
            self.body(mapping.jammed(s.body, s.axis.name, len(values), set(states)))

    def _merge_lanes(self, fold: Fold, group: int) -> None:
        """A butterfly over each aligned group of ``group`` lanes per independent
        merge, one after another (which measured better than stepping them
        together): each step swaps the merge's states with the lane ``width``
        away and combines the pair with its program. A group wider than a warp
        (a cell spanning its block) then merges the warps' totals
        (``_merge_warps``)."""
        b = self.b
        mask = self.kernel.reg("s32")
        b.emit("activemask.b32", mask, ndefs=1, effect=False)
        for states, program in fold.merges:
            width = min(group, WARP) // 2
            while width:
                with self.block():
                    for name in states:
                        var = self.lookup(name)
                        other = self.kernel.reg(var.type)
                        b.emit("shfl.sync.bfly.b32", other, b.in_reg(var.read()), Imm(width, "u32"), Imm(31, "u32"),
                               mask, ndefs=1, effect=False)
                        self.define(f"{name}$b", Value(var.type, other))
                    self.merge(program, states, "f32")
                width //= 2
        if group > WARP:
            self._merge_warps(fold, group // WARP)
        for name in fold.states:
            self.lookup(name).ug = min(group, WARP)  # every lane of the group now holds the whole fold

    def _merge_warps(self, fold: Fold, warps: int) -> None:
        """The second stage of a fold split over the warps of a block (a cell
        spanning them, ``_spread``): each warp's total, which the butterfly left
        in all its lanes, goes to shared memory, and after one barrier every
        thread combines the ``warps`` totals pairwise as a TreeHalve would, so
        all threads hold the same result. A barrier before the stores lets an
        enclosing loop reuse the slots."""
        b = self.b
        states = fold.states
        offset = self._lane_bytes
        self._lane_bytes += warps * len(states) * 4
        base = b.add(b.pure("mov.u32", "s32", (Sym(LANE_BUFFER),), 0, 1 << 24), b.int_const(offset))
        mine = b.in_reg(b.add(base, b.mul(self.value("warp"), b.int_const(len(states) * 4)))).op
        first = b.icmp("==", b.bitop("&", self.tid, b.int_const(WARP - 1)), b.int_const(0))  # one writer a warp
        b.emit("bar.sync", Imm(0, "u32"))
        for c, name in enumerate(states):
            b.emit("st.shared.f32", Mem(mine, c * 4), b.in_reg(self.lookup(name).read()), guard=first)
        b.emit("bar.sync", Imm(0, "u32"))
        totals = b.in_reg(base).op
        for part, program in fold.merges:
            level = []
            for w in range(warps):
                row = []
                for name in part:
                    out = self.kernel.reg("f32")
                    addr = Mem(totals, (w * len(states) + states.index(name)) * 4)
                    b.emit("ld.shared.f32", out, addr, ndefs=1, effect=False)
                    row.append(Value("f32", out))
                level.append(row)
            half = warps // 2
            while half:
                for i in range(half):
                    with self.block():
                        for name, v in zip(part, level[i]):
                            self.define(name, v)
                        for name, v in zip(part, level[i + half]):
                            self.define(f"{name}$b", v)
                        self.merge(program, part, "f32")
                        level[i] = [self.value(name) for name in part]
                half //= 2
            for name, v in zip(part, level[0]):
                self.reassign(name, v)

    def stmt_LaneSlot(self, s) -> None:
        """``(lane % group) / share``, alike on the ``share`` lanes of a slot."""
        b = self.b
        rank = self._rank(s.group)
        slot = b.shr(rank, b.int_const(s.share.bit_length() - 1)) if s.share > 1 else rank
        self.define(s.name, b.uniform(slot, min(s.share, WARP)))

    def _rank(self, group: int) -> Value:
        """This thread's rank in its aligned group of ``group`` threads (a group
        wider than a warp: a cell spanning its block's warps)."""
        b = self.b
        if group > WARP:
            return b.bitop("&", self.tid, b.int_const(group - 1))
        lane = self.value("lane")
        return lane if group == WARP else b.bitop("&", lane, b.int_const(group - 1))

    def stmt_LaneStash(self, s) -> None:
        """Store ``values`` for the LaneBroadcasts that follow, when the blocked
        loop has a part of the lane buffer (``_lower_blocked``): one word per
        slot of this lane's group, in the warp's own rows, between warp
        barriers (the warp's reads of the previous block must be done before
        the writes, and the writes before the reads)."""
        region = self._lane_region
        self._stashed = {}
        if region is None:
            return  # the broadcasts shuffle
        b = self.b
        offset, rows = region
        mask = b.pure("activemask.b32", "s32", ())
        lane, warp = self.value("lane"), self.value("warp")
        buffer = b.pure("mov.u32", "s32", (Sym(LANE_BUFFER),), 0, 1 << 24)
        base = b.add(buffer, b.add(b.int_const(offset), b.mul(warp, b.int_const(rows * WARP * 4))))
        # the group's slot 0, then this lane's slot
        first = b.add(base, b.mul(b.bitop("&", lane, b.int_const(WARP - s.group)), b.int_const(4)))
        slot = b.bitop("&", lane, b.int_const(s.group - 1))
        if s.share > 1:
            slot = b.shr(slot, b.int_const(s.share.bit_length() - 1))
        mine = b.in_reg(b.add(first, b.mul(slot, b.int_const(4)))).op
        b.emit("bar.warp.sync", mask)
        for row, name in enumerate(s.values):
            b.emit("st.shared.f32", Mem(mine, row * WARP * 4), b.in_reg(b.convert(self.value(name), "f32")))
        b.emit("bar.warp.sync", mask)
        self._stashed = {name: (first, row, s.share) for row, name in enumerate(s.values)}

    def stmt_LaneBroadcast(self, s) -> None:
        """``name`` = ``value`` of lane ``lane`` in this lane's aligned group of
        ``group``: a load of its stash (stmt_LaneStash), else ``shfl.sync.idx``
        with the group as the shuffle segment. A shared load costs far less
        (measured: 32 broadcasts, 4 us against 68 us of shuffles)."""
        b = self.b
        stashed = self._stashed.get(s.value)
        if stashed is not None:  # one shared load (ptxas merges neighbouring slots into vector loads)
            first, row, share = stashed
            out = self.kernel.reg("f32")
            addr = Mem(b.in_reg(first).op, row * WARP * 4 + s.lane // share * 4)
            b.emit("ld.shared.f32", out, addr, ndefs=1, effect=False)
            self.define(s.name, b.uniform(Value("f32", out), s.group))
            return
        v = self.value(s.value)
        mask = b.pure("activemask.b32", "s32", ())  # the code runs converged: one mask serves every shuffle
        out = self.kernel.reg(v.type)
        segment = ((WARP - s.group) << 8) | (WARP - 1)
        b.emit("shfl.sync.idx.b32", out, b.in_reg(v), Imm(s.lane, "u32"), Imm(segment, "u32"), mask,
               ndefs=1, effect=False)
        self.define(s.name, b.uniform(Value(v.type, out), s.group))

    def stmt_LaneButterfly(self, s) -> None:
        self._butterfly(s.state, s.state_b, s.combine_states, s.group, s.low, "f32")


def _lines(group: int, stride) -> float:
    """Cache lines one warp load touches when consecutive lanes read ``stride``
    bytes apart (None: unknown, so assume a line per lane)."""
    if stride is None:
        return group
    if stride == 0:
        return 1
    return min(group, -(-group * abs(stride) // LINE_BYTES))


def _line_cost(deltas, group: int) -> float:
    """Line accesses per iteration of a loop split over ``group`` lanes that
    take consecutive iterations (``deltas`` as from Lowerer._deltas)."""
    return sum(runs * _lines(group, stride) for runs, stride, _ in deltas)


def _merge_cost(fold: Fold) -> int:
    """Instructions in one butterfly step of ``fold``'s lane merges (an exp counts as eight)."""
    return len(fold.states) + sum(8 if a.op.name == "exp" else 1 for _, program in fold.merges for a in program)


def _split_vector(deltas, trips: int, group: int, whole: bool = True) -> int:
    """Consecutive iterations a split lane takes at a time: 4 or 2 when that turns
    a contiguous f32 load into one vector load per lane without spreading another
    load over more lines, else 1 (the lanes interleave single iterations).
    ``whole``: the rounds of ``group`` such vectors must cover the trips exactly."""
    moving = [(stride, size) for _, stride, size in deltas if stride != 0]
    if not any(stride == size == 4 for stride, size in moving):
        return 1
    if any(stride is not None and abs(stride) != size and abs(stride) < LINE_BYTES for stride, size in moving):
        return 1
    return next((v for v in (4, 2) if not whole or trips % (v * group) == 0), 1)

