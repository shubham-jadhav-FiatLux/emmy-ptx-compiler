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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt import Assign

LOOPS = ("Loop", "StridedLoop")
# what a pure loop may contain: values and folds, nothing another thread or the output sees
PURE = ("Load", "Assign", "Let", "Accum", "Select", *LOOPS)
CROSS_THREAD = ("Sync", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "WarpShuffle", "TreeHalve")
_EFFECTS = ("Write", "ZeroPrologue", "Init", "Smem", *CROSS_THREAD)
_MAX = ("maximum", "amax", "fmax")
_ADD = ("add", "sum")


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
    elif k in ("WarpShuffle", "TreeHalve"):
        out.update(s.state)
        out.update(x for a in s.combine_states for x in a.args)
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
        elif kind(s) in ("WarpShuffle", "TreeHalve"):
            out.update(s.state)
    return out


def produced(s) -> set[str]:
    """Names statement ``s`` declares or reassigns that later statements see. A
    cross-thread statement counts as producing everything (``"*"``)."""
    k = kind(s)
    if k == "Load":
        return set(s.names)
    if k in ("Assign", "Let", "Init", "Select", "Accum", "Smem"):
        return {s.name}
    return assigned((s,)) | ({"*"} if k in ("WarpShuffle", "TreeHalve") else set())


def binds(stmts) -> set[str]:
    """Every name the statements bind at any depth: values, accumulators, loop variables."""
    out: set[str] = set()
    for s in walk(stmts):
        k = kind(s)
        if k == "Load":
            out.update(s.names)
        elif k in ("Assign", "Let", "Init", "Select", "Accum"):
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
        elif k in ("Assign", "Let", "Init", "Select"):
            out.add(s.name)
        elif k in LOOPS and s.seed:
            out.update(x.name for x in s.body if kind(x) == "Accum")
    return out


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
        if any(pos[k] > pos[id(top)] for k in parts):
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
