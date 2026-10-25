"""Reduction mapping: which loop a lane-shared reduction is split over, how its
iterations are dealt to the lanes, and what the lanes then share.

The lowering splits a reduction that every lane of an aligned group would
compute alike across those lanes (``Lowerer._split_plan``). This module holds
the IR rewrites and analyses that plan relies on. None of them changes what a
kernel computes.

* :func:`absorb_siblings` is horizontal loop fusion (Kennedy & McKinley,
  LCPC'93). A later loop over the same iterations that is independent of a
  loop, and of everything between the two, joins it. Every accumulator still
  folds the same values in the same order, so the result is bit for bit the
  same, but a value both loops read is now loaded once per iteration and their
  folds become independent dependency chains.
* :func:`prune_dead` drops the statements of a loop body nothing reads
  (backward liveness). Merging congruent accumulators leaves their inputs dead.
* :func:`online_groups` and :func:`combine_program`: an online softmax
  (Milakov & Gimelshein, 2018) keeps a running max ``m`` and sums ``l`` that
  are rescaled whenever ``m`` grows. Its state is a monoid: folding one element
  ``(s, w)`` into ``(m, l)`` is the same as combining with the one-element
  state ``(m, l) = (s, w)``. So the combine of two partial states is the
  loop's own update with the element replaced by the other state, which is
  how :func:`combine_program` derives it (combiner synthesis, Farzan & Nicolet
  PLDI'17, for this one monoid).
* :func:`weighted_loads` and :func:`flat_coefficient` tell how far apart the
  addresses of consecutive iterations are: the input of the coalescing cost
  model, since consecutive lanes should read consecutive addresses (Yang et
  al. PLDI'10; PPCG, Verdoolaege et al. TACO'13).
* :func:`jammed` is unroll-and-jam (Callahan, Carr & Kennedy, PLDI'90): copies
  of a loop body for consecutive iterations, renamed apart, so that fusing
  merges the inner loops of the copies.
* :func:`blocked` is lane blocking, the work partitioning of FlashAttention-2
  (Dao, ICLR'24) done by the compiler. When only part of each iteration is
  the same on all lanes of a group (in attention, the scores of a query row;
  each lane then folds them into its own output columns), the lanes take the
  iterations of a block in turn for that part and send each other its
  results, so the shared part runs once per iteration, not once per lane.
  When each iteration's inputs are a row of their own, a few lanes share an
  iteration (a slot) to read it coalesced; the lowering may send the results
  through shared memory (LaneStash) rather than shuffles.
* :func:`cse` removes repeated arithmetic from a loop body, so the analyses
  above see that two merged cells share one online-softmax update.
* :func:`dependents` finds the names that depend on a Tile coordinate: the
  work the cells along that axis have in common is everything else (where
  the lowering may move cells so that lanes share it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt import Accum, Assign, Loop
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.leaves import Let

LOOPS = ("Loop", "StridedLoop")
# what a pure loop may contain: values and folds, nothing another thread or the output sees
PURE = ("Load", "Assign", "Let", "Accum", "Select", *LOOPS)
CROSS_THREAD = ("Sync", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "WarpShuffle", "TreeHalve", "LaneBroadcast",
                "LaneButterfly", "LaneStash")
_BUTTERFLIES = ("WarpShuffle", "TreeHalve", "LaneButterfly")  # statements that reassign their ``state``
_EFFECTS = ("Write", "ZeroPrologue", "Init", "Smem", *CROSS_THREAD)
_MAX = ("maximum", "amax", "fmax")
_ADD = ("add", "sum")
# what :func:`renamed` can rename inside
_RENAMABLE = ("Load", "Assign", "Let", "Accum", "Init", "Select", "Write", "Cond", "LaneBroadcast", "LaneSlot",
              "LaneStash", *LOOPS)


def kind(s) -> str:
    return type(s).__name__


def walk(stmts):
    """Every statement of ``stmts``, nested ones included, in order."""
    for s in stmts:
        yield s
        for child in s.nested():
            yield from walk(child)


# ---------------------------------------------------------------- names


def own_reads(s) -> set[str]:
    """Names statement ``s`` itself reads (not counting the bodies nested in it)."""
    k = kind(s)
    out: set[str] = set()
    if k == "Assign":
        out.update(s.args)
    elif k == "Accum":
        out.update(x for x in (s.name, s.value, s.base) if x is not None)
    elif k == "Write":
        out.update(s.values)
    elif k == "Select":
        out.update(br.value for br in s.branches)
    elif k in _BUTTERFLIES:
        out.update(s.state)
        out.update(x for a in s.combine_states for x in a.args)
    elif k == "LaneBroadcast":
        out.add(s.value)
    elif k == "LaneStash":
        out.update(s.values)
    for e in getattr(s, "exprs", lambda: ())():
        out.update(e.free_vars())
    return out


def reads(stmts) -> set[str]:
    """Names the statements read anywhere, loop bounds and indices included."""
    return {n for s in walk(stmts) for n in own_reads(s)}


def assigned(stmts) -> set[str]:
    """Names the statements (at any depth) reassign: accumulators and cross-thread states."""
    out: set[str] = set()
    for s in walk(stmts):
        if kind(s) == "Accum":
            out.add(s.name)
        elif kind(s) in _BUTTERFLIES:
            out.update(s.state)
    return out


def produced(s) -> set[str]:
    """Names statement ``s`` declares or reassigns that later statements see. A
    cross-thread statement counts as producing everything (``"*"``)."""
    k = kind(s)
    if k == "Load":
        return set(s.names)
    if k in ("Assign", "Let", "Init", "Select", "Accum", "Smem", "LaneBroadcast", "LaneSlot"):
        return {s.name}
    return assigned((s,)) | ({"*"} if k in _BUTTERFLIES else set())


def binds(stmts) -> set[str]:
    """Every name the statements bind at any depth: values, accumulators, loop variables."""
    out: set[str] = set()
    for s in walk(stmts):
        k = kind(s)
        if k == "Load":
            out.update(s.names)
        elif k in ("Assign", "Let", "Init", "Select", "Accum", "LaneBroadcast", "LaneSlot"):
            out.add(s.name)
        elif k in LOOPS:
            out.add(s.axis.name)
            if k == "StridedLoop" and s.end is not None:
                out.add(f"{s.axis.name}_end")
    return out


def written(stmts) -> set[str]:
    """Buffers the statements store to (global outputs and shared arrays)."""
    out: set[str] = set()
    for s in walk(stmts):
        if kind(s) == "Write":
            out.add(s.output)
        elif kind(s) == "CpAsyncCopy":
            out.add(s.smem)
    return out


def renamed(s, mapping: dict[str, str]):
    """Statement ``s`` with every occurrence of the names in ``mapping``, bound or
    read, renamed (alpha-renaming: the new names must be unused)."""
    if not mapping:
        return s
    sub = {old: Var(new) for old, new in mapping.items()}

    def n(x):
        return mapping.get(x, x)

    def e(x):
        return x.substitute(sub) if hasattr(x, "substitute") else x

    def body(stmts):
        return tuple(renamed(x, mapping) for x in stmts)

    k = kind(s)
    if k == "Load":
        return replace(s, names=tuple(n(x) for x in s.names), index=tuple(e(i) for i in s.index))
    if k == "Assign":
        return replace(s, name=n(s.name), args=tuple(n(a) for a in s.args))
    if k == "Let":
        return replace(s, name=n(s.name), value=e(s.value))
    if k == "Accum":
        return replace(s, name=n(s.name), value=n(s.value), base=None if s.base is None else n(s.base),
                       axes=tuple(n(a) for a in s.axes))
    if k == "Init":
        return replace(s, name=n(s.name))
    if k == "LaneBroadcast":
        return replace(s, name=n(s.name), value=n(s.value))
    if k == "LaneSlot":
        return replace(s, name=n(s.name))
    if k == "LaneStash":
        return replace(s, values=tuple(n(v) for v in s.values))
    if k == "Select":
        return replace(s, name=n(s.name),
                       branches=tuple(replace(br, value=n(br.value), select=e(br.select)) for br in s.branches))
    if k == "Write":
        return replace(s, index=tuple(e(i) for i in s.index), values=tuple(n(v) for v in s.values))
    if k == "Cond":
        return replace(s, cond=e(s.cond), body=body(s.body), else_body=body(s.else_body))
    if k == "Loop":
        return replace(s, axis=replace(s.axis, name=n(s.axis.name)), body=body(s.body))
    if k == "StridedLoop":
        return replace(s, axis=replace(s.axis, name=n(s.axis.name)), start=e(s.start), step=e(s.step),
                       end=None if s.end is None else e(s.end), body=body(s.body))
    raise ValueError(f"cannot rename inside {k}")


# ---------------------------------------------------------------- fusion


def is_pure_loop(s) -> bool:
    """A loop of values and folds only: nothing it does is seen before its accumulators are read."""
    return kind(s) in LOOPS and all(kind(x) in PURE for x in walk(s.body))


def same_space(s, t) -> bool:
    """Do loops ``s`` and ``t`` run the same iterations, and seed alike?"""
    if kind(s) != kind(t) or s.seed != t.seed or s.axis.extent != t.axis.extent:
        return False
    return kind(s) == "Loop" or (s.start, s.step, s.end) == (t.start, t.step, t.end)


def absorb_siblings(stmts: list, i: int, compatible):
    """Fuse into loop ``stmts[i]`` every later loop of ``stmts`` that runs the same
    iterations and is independent of it and of the statements in between;
    the absorbed loops are removed from ``stmts``. ``compatible(s, t)`` may
    refuse a pair. Returns the loop now at ``stmts[i]``."""
    s = stmts[i]
    if not is_pure_loop(s):
        return s
    j = i + 1
    while j < len(stmts):
        t = stmts[j]
        if (is_pure_loop(t) and same_space(s, t) and _movable(t, stmts[i + 1 : j]) and _disjoint(s, t)
                and compatible(s, t)):
            s = stmts[i] = _fuse(s, t)
            del stmts[j]
        else:
            j += 1
    return s


def _movable(t, between) -> bool:
    """Can pure loop ``t`` run before ``between``? Not if one of those statements
    produces a name ``t`` reads or folds, reads a name ``t`` folds, stores to
    memory ``t`` loads, or talks to other threads."""
    t_reads, t_folds = reads((t,)), assigned((t,))
    t_loads = {x.input for x in walk((t,)) if kind(x) == "Load"}
    for u in between:
        if any(kind(x) in CROSS_THREAD for x in walk((u,))):
            return False
        if produced(u) & (t_reads | t_folds) or reads((u,)) & t_folds or written((u,)) & t_loads:
            return False
    return True


def _disjoint(s, t) -> bool:
    """Can the bodies of ``s`` and ``t`` share one iteration, ``s``'s part first?
    Neither may read what the other produces, their top-level names must
    differ, and ``t`` must not already bind or read the name its loop
    variable is renamed to."""
    s_out = set().union(*(produced(u) for u in s.body))
    t_out = set().union(*(produced(u) for u in t.body))
    if s.axis.name != t.axis.name and s.axis.name in binds(t.body) | reads(t.body):
        return False
    return not (s_out & t_out or s_out & reads(t.body) or t_out & reads(s.body))


def _fuse(s, t):
    rename = {t.axis.name: s.axis.name}
    if kind(t) == "StridedLoop" and t.end is not None:
        rename[f"{t.axis.name}_end"] = f"{s.axis.name}_end"
    rename = {a: b for a, b in rename.items() if a != b}
    return replace(s, body=(*s.body, *(renamed(u, rename) for u in t.body)), unroll=s.unroll and t.unroll)


def prune_dead(body) -> tuple:
    """``body`` (a loop body) without the statements nothing reads, by backward
    liveness over one iteration. A statement stays when it has an effect or
    updates a name declared outside the body (the loop's carried state)."""
    local = local_names(body)
    live: set[str] = set()
    keep = []
    for s in reversed(tuple(body)):
        effect = any(kind(x) in _EFFECTS for x in walk((s,)))
        if effect or assigned((s,)) - local or produced(s) & live:
            keep.append(s)
            live |= reads((s,))
    return tuple(reversed(keep))


def local_names(body) -> set[str]:
    """Names declared inside ``body``: values, and the accumulators that loops
    nested in it seed (a fold of an outer accumulator updates a name from outside)."""
    out: set[str] = set()
    for s in walk(body):
        k = kind(s)
        if k == "Load":
            out.update(s.names)
        elif k in ("Assign", "Let", "Init", "Select", "LaneBroadcast", "LaneSlot"):
            out.add(s.name)
        elif k in LOOPS and s.seed:
            out.update(x.name for x in s.body if kind(x) == "Accum")
    return out


def dependents(stmts, roots: set[str]) -> set[str]:
    """``roots`` and every name the statements bind whose value may depend on
    one of them: through what a statement reads, a loop's bounds (its
    variable) or a branch's condition (everything the branch binds). Repeated
    to a fixpoint, since an accumulator carries a dependence around its loop."""
    dep = set(roots)
    while True:
        size = len(dep)
        for s in walk(stmts):
            if not own_reads(s) & dep:
                continue
            k = kind(s)
            dep |= produced(s) - {"*"}
            if k in LOOPS:
                dep.add(s.axis.name)
                if k == "StridedLoop" and s.end is not None:
                    dep.add(f"{s.axis.name}_end")
            elif k == "Cond":
                for child in s.nested():
                    dep |= binds(child) | assigned(child)
        if len(dep) == size:
            return dep


def cse(body) -> tuple:
    """``body`` (a loop body) without the top-level Assigns, Lets and Selects that
    repeat an earlier one's computation; their readers read the earlier name
    (common-subexpression elimination). A repeat counts only while the names
    it reads still hold what they held for the earlier statement."""
    if any(kind(x) not in _RENAMABLE for x in walk(body)):
        return tuple(body)
    version: dict[str, int] = {}
    seen: dict = {}
    alias: dict[str, str] = {}
    out = []
    for s in body:
        s = renamed(s, alias)
        k, key = kind(s), None
        if k == "Assign":
            key = (s.op.name, tuple((a, version.get(a, 0)) for a in s.args), _dtype(s))
        elif k == "Let":
            key = (s.value, _dtype(s), tuple((v, version.get(v, 0)) for v in sorted(s.value.free_vars())))
        elif k == "Select":
            key = (s.branches, tuple((v, version.get(v, 0)) for v in sorted(own_reads(s))))
        if key is not None and key in seen:
            alias[s.name] = seen[key]
            continue
        if key is not None:
            seen[key] = s.name
        for name in assigned((s,)):
            version[name] = version.get(name, 0) + 1
        out.append(s)
    return tuple(out)


def _dtype(s):
    return s.dtype.name if s.dtype is not None else None


# ---------------------------------------------------------------- online softmax


@dataclass(frozen=True)
class OnlineGroup:
    """An online-softmax state (module doc): the running max ``m`` of the
    scores ``s``, and sums ``l <- l*e^(m_old - m) + w*e^(s - m)``."""

    m: str
    s: str
    sums: tuple[tuple[str, str, str], ...]  # (l, its addend's name, the weight w in the addend)
    stmts: tuple  # the update's statements in body order, the max's own fold last


def online_groups(body) -> list[OnlineGroup] | None:
    """The online-softmax states among the top-level accumulations of ``body``
    (a loop body), or None when an accumulation with a ``base`` is no such sum."""
    assigns = {s.name: s for s in body if kind(s) == "Assign"}
    pos = {id(s): i for i, s in enumerate(body)}
    accums = [s for s in body if kind(s) == "Accum"]
    maxes = {s.name: s for s in accums if s.op.name in _MAX and s.base is None}
    found: dict[str, list] = {}
    for acc in accums:
        if acc.base is None:
            continue
        match = _rescaled_sum(acc, assigns, maxes)
        if match is None:
            return None
        found.setdefault(match[0], []).append((acc, match[1], match[2]))
    groups = []
    for m, sums in found.items():
        top = maxes[m]
        parts = {id(x): x for acc, update, _ in sums for x in (*update, acc)}
        if any(pos[k] > pos[id(top)] for k, x in parts.items() if m in own_reads(x)):
            return None  # the update must read the max before the max moves
        if len({acc.value for acc, _, _ in sums}) != len(sums):
            return None  # two sums sharing one addend
        stmts = tuple(sorted(parts.values(), key=lambda x: pos[id(x)])) + (top,)
        groups.append(OnlineGroup(m, top.value, tuple((acc.name, acc.value, w) for acc, _, w in sums), stmts))
    return groups


def _rescaled_sum(acc, assigns, maxes):
    """For ``acc <- base + addend``: ``(m, update, w)`` when base = alpha*acc and
    addend = beta*w with alpha = e^(m - max(s, m)), beta = e^(s - max(s, m)), and
    ``m <- max(m, s)`` a running max; ``update`` lists the Assigns. Else None."""
    sa, sb = assigns.get(acc.base), assigns.get(acc.value)
    if acc.op.name not in _ADD or not _is(sa, "multiply", 2) or not _is(sb, "multiply", 2):
        return None
    alpha = assigns.get(_other(sa, acc.name))
    if not _is(alpha, "exp", 1):
        return None
    dg = assigns.get(alpha.args[0])
    if not _is(dg, "subtract", 2) or dg.args[0] not in maxes:
        return None
    m, gn = dg.args[0], assigns.get(dg.args[1])
    s = maxes[m].value
    if not _is(gn, "maximum", 2) or sorted(gn.args) != sorted((s, m)):
        return None
    for beta_name, w in (sb.args, sb.args[::-1]):
        beta = assigns.get(beta_name)
        if not _is(beta, "exp", 1):
            continue
        dg_o = assigns.get(beta.args[0])
        gn_o = assigns.get(dg_o.args[1]) if _is(dg_o, "subtract", 2) else None
        if dg_o is not None and dg_o.args[0] == s and _is(gn_o, "maximum", 2) and sorted(gn_o.args) == sorted((s, m)):
            update = tuple({id(x): x for x in (gn, dg, alpha, gn_o, dg_o, beta, sa, sb)}.values())
            return m, update, w
    return None


def _is(s, op: str, arity: int) -> bool:
    return s is not None and kind(s) == "Assign" and s.op.name == op and len(s.args) == arity


def _other(assign, name: str):
    """The operand of a binary Assign other than ``name`` (None unless ``name`` is one)."""
    a, b = assign.args
    return b if a == name else a if b == name else None


def combine_program(group: OnlineGroup, partner, fresh) -> tuple:
    """Assigns merging a partial state into ``group``'s (module doc): the group's
    own update with the element (s, w) replaced by the partner state, i.e.
    ``s`` -> partner(m) and, in each sum's addend, ``w`` -> partner(l).
    ``partner(name)`` names the partner's copy of a state; ``fresh(name)``
    renames the update's temporaries."""
    temps = {x.name: fresh(x.name) for x in group.stmts if kind(x) == "Assign"}
    rename = {**temps, group.s: partner(group.m)}
    weight = {addend: (l, w) for l, addend, w in group.sums}
    out = []
    for x in group.stmts:
        if kind(x) == "Assign":
            args = x.args
            if x.name in weight:  # beta * w: the partner's sum takes the weight's place
                l, w = weight[x.name]
                args = tuple(partner(l) if a == w else a for a in args)
            out.append(Assign(temps[x.name], x.op, tuple(rename.get(a, a) for a in args), x.dtype))
        else:  # the accumulation ``name = op(base or name, value)``, now of the partner's part
            base = x.base if x.base is not None else x.name
            out.append(Assign(x.name, x.op, (rename.get(base, base), rename.get(x.value, x.value)), x.dtype))
    return tuple(out)


# ---------------------------------------------------------------- address strides


def weighted_loads(stmts, weight: float = 1.0):
    """``(Load, runs per run of stmts)`` for every load in ``stmts``: the trip
    counts of the loops around it, multiplied."""
    for s in stmts:
        if kind(s) == "Load":
            yield s, weight
        elif kind(s) in LOOPS:
            yield from weighted_loads(s.body, weight * trip_estimate(s))
        else:
            for child in s.nested():
                yield from weighted_loads(child, weight)


def trip_estimate(loop) -> float:
    """Iterations of ``loop`` (a strided loop's extent over its literal step)."""
    extent = int(loop.axis.extent.as_static())
    if kind(loop) == "StridedLoop":
        step = loop.step.value if kind(loop.step) == "Literal" else 1
        return max(1.0, extent / max(1, int(step)))
    return float(extent)


def index_lets(stmts) -> dict:
    """The integer index expressions the statements bind with ``Let``."""
    return {s.name: s.value for s in walk(stmts) if kind(s) == "Let" and s.dtype is None}


def coefficient(e, var: str, lets: dict) -> int | None:
    """How much integer expression ``e`` grows when ``var`` grows by one, if ``e``
    is affine in ``var`` (None otherwise). ``lets`` expands bound index names."""
    k = kind(e)
    if k == "Var":
        if e.name == var:
            return 1
        return coefficient(lets[e.name], var, lets) if e.name in lets else 0
    if k == "Literal":
        return 0
    if k == "BinaryExpr" and e.op in ("+", "-", "*"):
        a, b = coefficient(e.left, var, lets), coefficient(e.right, var, lets)
        if e.op != "*":
            return None if a is None or b is None else a + b if e.op == "+" else a - b
        if a == 0 and b == 0:
            return 0
        for c, factor in ((a, e.right), (b, e.left)):
            if c and kind(factor) == "Literal" and isinstance(factor.value, int):
                return c * factor.value
        return None
    return None if _depends(e, var, lets) else 0


def _depends(e, var: str, lets: dict) -> bool:
    return any(v == var or (v in lets and _depends(lets[v], var, lets)) for v in e.free_vars())


def flat_coefficient(index, shape, var: str, lets: dict) -> int | None:
    """:func:`coefficient` of the row-major element offset of ``index`` in a
    buffer of ``shape`` (emmy's render_index: a single index is used as is)."""
    cs = [coefficient(e, var, lets) for e in index]
    if None in cs:
        return None
    if len(index) == 1:
        return cs[0]
    if len(shape) != len(index):
        return sum(cs)
    return sum(c * math.prod(shape[d + 1 :]) for d, c in enumerate(cs))


# ---------------------------------------------------------------- unroll-and-jam


def jammed(body, var: str, copies: int, keep: set[str]) -> tuple:
    """``copies`` copies of a loop body, for consecutive iterations, renamed apart
    (module doc): copy ``j`` reads the loop variable as ``var$j`` and suffixes
    every name the body binds with ``$j``, except the names in ``keep`` (the
    loop's carried accumulators, which the copies update in turn)."""
    local = binds(body) - keep
    out = []
    for j in range(copies):
        mapping = {n: f"{n}${j}" for n in local}
        mapping[var] = f"{var}${j}"
        out.extend(renamed(s, mapping) for s in body)
    return tuple(out)


# ---------------------------------------------------------------- lane blocking
# The statements below exist only inside loops that :func:`blocked` rewrote.


@dataclass(frozen=True)
class LaneSlot(Stmt):
    """``name`` = the slot of this lane in its aligned group of ``group`` lanes,
    slots being runs of ``share`` lanes: ``(lane % group) / share``."""

    name: str
    group: int
    share: int

    def defines(self) -> tuple[str, ...]:
        return (self.name,)


@dataclass(frozen=True)
class LaneBroadcast(Stmt):
    """``name = value`` as lane ``lane`` of each aligned group of ``group`` lanes holds it."""

    name: str
    value: str
    lane: int
    group: int

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)


@dataclass(frozen=True)
class LaneStash(Stmt):
    """``values`` are about to be broadcast within each aligned group of
    ``group`` lanes (slots of ``share`` lanes holding each alike): the lowering
    may store them to shared memory, so that each LaneBroadcast of them that
    follows is one shared load instead of a shuffle."""

    values: tuple[str, ...]
    group: int
    share: int

    def deps(self) -> tuple[str, ...]:
        return self.values


@dataclass(frozen=True)
class LaneButterfly(Stmt):
    """WarpShuffle's butterfly over the lane distances ``group/2`` down to
    ``low`` only: runs of ``low`` lanes already hold the state alike."""

    state: tuple[str, ...]
    state_b: tuple[str, ...]
    combine_states: tuple
    group: int
    low: int

    def deps(self) -> tuple[str, ...]:
        return self.state


@dataclass(frozen=True)
class Blocked:
    """A loop dealt to the lanes of a group in blocks (:func:`blocked`)."""

    loop: object  # a Loop over the whole blocks, or None
    rest: tuple  # the last, partial block as straight-line statements
    merges: tuple  # LaneButterflies adding up the lanes' partial sums after the loop
    mutable: frozenset  # names the new statements reassign
    shared: tuple  # the shared part of an iteration (what one slot computes for the others)
    invariant: float  # elements the shared part loads alike in every block (a query row, say)


def blocked(loop, trips: int, start: int, step: int, group: int, share: int, uniform) -> Blocked | None:
    """Loop ``loop`` (iterations ``start + i*step`` for ``i < trips``) dealt to
    aligned groups of ``group`` lanes in blocks of ``group / share``
    iterations, or None when its body does not have that shape.
    ``uniform(name)`` says whether all lanes of a group compute a body value
    alike in a given iteration.

    The body splits in two. The *shared* part computes values the lanes agree
    on without reading the loop's carried state, and must hold the expensive
    work (an inner loop). The *own* part is the rest: the values that differ
    between lanes and the folds. The group's lanes form slots of ``share``
    lanes, and in each block slot r computes the shared part of the block's
    iteration r only (its lanes split the inner loops between them, like any
    reduction lanes share). Then, for each iteration j of the block, every lane
    receives from slot j the shared values the own part reads (one
    LaneBroadcast each) and runs the own part, so plain folds fold the same
    values in the same order as before.

    An online softmax (:func:`online_groups`) is folded once per block: a
    butterfly over the slots' scores gives the block's max, each slot weighs
    its own score against the new max, and each sum is rescaled once. A sum
    whose weights the lanes share keeps one partial sum per slot (``merges``
    adds them up after the loop); a sum with per-lane weights folds the weight
    of every iteration of the block, times that iteration's broadcast beta.
    """
    body = tuple(loop.body)
    groups = online_groups(body)
    if groups is None:
        return None
    in_group = {id(x) for g in groups for x in g.stmts}
    temps = {x.name for g in groups for x in g.stmts if kind(x) == "Assign"}
    softmax_states = {g.m for g in groups} | {l for g in groups for l, _, _ in g.sums}
    states = {s.name for s in body if kind(s) == "Accum"}
    rest = [s for s in body if id(s) not in in_group]
    shared = [s for s in rest if kind(s) in PURE and kind(s) != "Accum" and not reads((s,)) & (states | temps)
              and all(uniform(n) for n in produced(s))]
    shared_ids = {id(s) for s in shared}
    from_shared = set().union(*(produced(s) for s in shared))
    bound = binds(body)
    varying, pooled = [], []
    for g in groups:
        if g.s not in from_shared:
            return None
        for l, addend, w in g.sums:
            if not uniform(w):
                varying.append((g, l, addend, w))
            elif w in bound and w not in from_shared:
                return None
            else:
                pooled.append(l)
    plain = [s for s in rest if kind(s) == "Accum"]
    if not varying and not plain:
        return None  # all lanes fold the same values: a lane-split reduction instead
    # the own part: what the per-lane weights and the plain folds need (Lets are recomputed, never sent)
    own = _needed([s for s in rest if id(s) not in shared_ids or kind(s) == "Let"],
                  {w for *_, w in varying} | {s.name for s in plain})
    for s in own:
        if reads((s,)) & (temps | softmax_states):
            return None  # the per-iteration softmax values no longer exist
    wanted = reads(own)
    sent = [n for s in shared if kind(s) != "Let" for n in _names(s) if n in wanted]
    work = _needed(shared, {g.s for g in groups} | {w for g in groups for l, _, w in g.sums if l in pooled} | set(sent))
    if not any(kind(s) in LOOPS for s in work):
        return None  # nothing worth sharing
    t = loop.axis.name
    slots = group // share
    copies = binds(own) - states
    assigns = {x.name: x for g in groups for x in g.stmts if kind(x) == "Assign"}

    def iteration(i):
        return BinaryExpr("+", _int(start), BinaryExpr("*", i, _int(step)))

    def block(index, count: int) -> list:
        first = BinaryExpr("*", index, _int(slots))
        mine = BinaryExpr("+", first, Var(f"{t}$slot"))
        out = [Let(f"{t}${j}", iteration(BinaryExpr("+", first, _int(j)))) for j in range(count)]
        out.append(LaneSlot(f"{t}$slot", group, share))
        valid = None
        if count < slots:  # slots past the last iteration redo it and drop their scores
            valid = BinaryExpr("<", mine, _int(trips))
            mine = TernaryExpr(valid, mine, _int(trips - 1))
        out.append(Let(t, iteration(mine)))
        out.extend(work)
        broadcast = list(sent)
        betas = {}
        for g in groups:
            score = g.s
            if valid is not None:
                score = f"{g.s}$in"
                out.append(Let(score, TernaryExpr(valid, Var(g.s), Literal(-math.inf)), F32))
            update, names = _block_update(g, score, group, share, pooled)
            out.extend(update)
            betas[g.m] = names
            for _, l, addend, w in (v for v in varying if v[0] is g):  # each iteration's beta, for its weights
                beta = names[_other(assigns[addend], w)]
                if beta not in broadcast:
                    broadcast.append(beta)
        if broadcast:
            out.append(LaneStash(tuple(broadcast), group, share))
        for j in range(count):
            out.extend(LaneBroadcast(f"{x}${j}", x, j * share, group) for x in broadcast)
            rename = {**{n: f"{n}${j}" for n in copies}, t: f"{t}${j}", **{x: f"{x}${j}" for x in broadcast}}
            out.extend(renamed(s, rename) for s in own)
            for g, l, addend, w in varying:
                sb = assigns[addend]
                beta = _other(sb, w)
                args = tuple(rename.get(a, a) if a == w else f"{betas[g.m][beta]}${j}" for a in sb.args)
                out.append(Assign(f"{addend}${j}", sb.op, args, sb.dtype))
                out.append(Accum(l, f"{addend}${j}", "add", dtype=_sum_of(g, l).dtype))
        return out

    full, left = divmod(trips, slots)
    index = f"{t}$blk"
    main = None
    if full:
        main = Loop(axis=Axis(index, Dim(full)), body=tuple(block(Var(index), slots)), unroll=loop.unroll, seed=False)
    tail = tuple(block(_int(full), left)) if left else ()
    merges = tuple(LaneButterfly((l,), (f"{l}$b",), (Assign(l, "add", (l, f"{l}$b"), F32),), group, share)
                   for l in pooled)
    lets = index_lets(work)
    invariant = sum(runs * len(ld.names) for ld, runs in weighted_loads(work)
                    if all(coefficient(e, t, lets) == 0 for e in ld.index))
    return Blocked(main, tail, merges, frozenset(f"{g.m}$blk" for g in groups), tuple(work), invariant)


def _block_update(g: OnlineGroup, score: str, group: int, share: int, pooled) -> tuple[list, dict]:
    """One block's update of online softmax ``g`` (see :func:`blocked`): the
    group's own update with the max taken over the block (a butterfly of the
    slots' ``score``) and each slot's beta weighing its own score. The sums in
    ``pooled`` fold that beta as the IR does; the others are only rescaled.
    Returns the statements and the block's name for each update value."""
    top = g.stmts[-1]
    peak = f"{g.m}$blk"
    names = {x.name: f"{x.name}$blk" for x in g.stmts if kind(x) == "Assign"}
    out = [Let(peak, Var(score), F32),
           LaneButterfly((peak,), (f"{peak}$b",), (Assign(peak, "maximum", (peak, f"{peak}$b"), F32),), group, share)]
    per_iteration = set()  # the products of the sums that are only rescaled here
    for l, addend, _ in g.sums:
        if l not in pooled:
            per_iteration |= {_sum_of(g, l).base, addend}
    for x in g.stmts:
        if kind(x) != "Assign" or x.name in per_iteration:
            continue
        args = x.args
        if x.op.name in _MAX and set(args) == {g.s, g.m}:  # the new max, over the block
            args = tuple(peak if a == g.s else a for a in args)
        out.append(Assign(names[x.name], x.op, tuple(score if a == g.s else names.get(a, a) for a in args), x.dtype))
    assigns = {x.name: x for x in g.stmts if kind(x) == "Assign"}
    for l, addend, _ in g.sums:
        acc = _sum_of(g, l)
        if l in pooled:
            out.append(replace(acc, base=names[acc.base], value=names[addend]))
        else:  # l * alpha, once per block
            out.append(Accum(l, names[_other(assigns[acc.base], l)], "multiply", dtype=acc.dtype))
    out.append(replace(top, value=peak))
    return out, names


def _sum_of(g: OnlineGroup, l: str):
    return next(x for x in g.stmts if kind(x) == "Accum" and x.name == l)


def _needed(stmts, roots) -> list:
    """The statements of ``stmts`` that ``roots`` (names) depend on, in order."""
    need, keep = set(roots), []
    for s in reversed(stmts):
        if produced(s) & need:
            keep.append(s)
            need |= reads((s,))
    return keep[::-1]


def _names(s) -> list:
    """What statement ``s`` produces, in a stable order."""
    return list(s.names) if kind(s) == "Load" else sorted(produced(s))


def _int(v: int):
    return Literal(v, "int")
