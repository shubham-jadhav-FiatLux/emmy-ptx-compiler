"""Regroup a cooperative reduction so each thread reads contiguous chunks.

Emmy's cooperative reductions stride the threads of a group over the data,

    for a in t .. N : B          (t = the thread's rank among the B threads)
        ... load x[..., a] ...
        acc <- acc (+) value
    WarpShuffle(acc) ; [ smem, Sync, TreeHalve over the warps ]

and then combine every thread's partial into the group's total. Each
thread's partial is never observed on its own, only the total, and the
combine is associative and commutative, so which elements a thread folds
does not matter as long as every element is folded exactly once. ``regroup``
rewrites the loop so thread t folds the contiguous chunks

    V*t + V*B*k + j     for j in 0..V-1, k = 0, 1, ...

which turns V scalar loads into one vector load per chunk (more bytes in
flight per instruction, the same coalescing across the threads).

Emmy may also unroll the loop R times ("coop/r2"): the step is R*B, and copy
j of the body reads ``x[a + j*B]`` into an accumulator of its own, the R
accumulators being folded together right before the combine. Each copy is then
the same fold over its own elements, so the rewrite deals the chunk's
elements to the copies in turn (element j of a chunk to copy j mod R), which
keeps R independent chains.

It only applies when the statements right after the loop are exactly such a
full combine of the loop's accumulators across all B threads, with the loop's
own fold (``combine_refusal`` checks the whole combine program and says why
not), the body is a pure fold (loads, arithmetic, accumulations nothing else
reads), every load reads the loop variable plus a multiple of B, and V*B
divides N, so no thread needs a partial chunk.

Keeping each element once is not enough: the element must also keep its
value. Its new thread computes it, and its new copy, so the copies must
compute alike (``_same_copy``), and the lowering checks that nothing the
body reads from outside (``outer_values``) differs between the threads, that
the threads are exactly the groups the combine spans, and that the loads read
memory no thread writes (``Lowerer._ownership_refusal``). A value that does
not change in the loop may still change with the thread: ``x[t, a]``, or a
weight ``t``.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import StridedLoop

from .mapping import binds, own_reads, reads, renamed

VECTOR = 4  # elements per chunk: one 16-byte load of f32
_FOLDS = ("add", "sum", "maximum", "amax", "minimum", "multiply", "prod")


def _int(e):
    return e.value if type(e).__name__ == "Literal" and not isinstance(e.value, float) else None


def outer_values(loop) -> set[str]:
    """The names ``loop``'s body reads from outside it: whatever it reads
    before binding it, but its own loop variable and accumulators."""
    bound = {loop.axis.name} | {s.name for s in loop.body if type(s).__name__ == "Accum"}
    out: set[str] = set()
    for s in loop.body:
        out |= own_reads(s) - bound
        bound |= binds((s,))
    return out


def combine_refusal(stmts, i: int, accs: set[str], width: int, private=lambda tree: None,
                    builtin=lambda name: True) -> str | None:
    """Why the statements after ``stmts[i]`` are not a full combine of ``accs``
    across all ``width`` threads of the group (a warp butterfly, then a tree
    over the warps) that folds each accumulator with its own fold, or None
    when they are. Some accumulators may first be folded into the others (an
    R-times unrolled loop's partials, with the loop's own combine).

    Moving elements between threads is legal only if no partial is observed
    on its own: each partial must flow straight into combines that apply the
    loop's own associative fold to it and to its partner (a maximum of partial
    sums is not a sum), in the same dtype, and the per-warp partials may reach
    shared memory only as the tree's input, read by nothing but the tree
    (``private(tree)`` says why not, or None). And every warp's partial must
    reach the tree, once: lane 0 of each warp writes it to slot ``warp`` of
    the array the tree folds it from, then a barrier (``builtin(name)``: do
    ``lane`` and ``warp`` still mean the thread's lane and warp index?)."""
    loop = stmts[i]
    folds = {s.name: s for s in loop.body if type(s).__name__ == "Accum"}
    ops = {n: _canon(s.op.name) for n, s in folds.items()}
    rest = list(stmts[i + 1 :])
    merged = set()
    while rest and type(rest[0]).__name__ == "Accum" and rest[0].base is None:
        fold = rest[0]
        if fold.name not in accs or fold.value not in accs - merged - {fold.name} or ops[fold.name] != _canon(fold.op.name):
            break
        if ops[fold.value] != _canon(fold.op.name):
            return f"{fold.value} is folded into {fold.name} with {fold.op.name}, but the loop folds it with {ops[fold.value]}"
        merged.add(fold.value)
        rest = rest[1:]
    if merged & reads(rest):
        return "a partial folded away is still read"
    live = accs - merged
    if not rest or type(rest[0]).__name__ != "WarpShuffle":
        return "the loop's partials do not flow straight into a warp butterfly"
    shuffle = rest[0]
    if int(shuffle.length) != min(width, 32):
        return f"the butterfly spans {int(shuffle.length)} lanes, not the {min(width, 32)} cooperating threads"
    reason = _same_combine(shuffle, live, folds, ops, "butterfly")
    if reason is not None or width <= 32:
        return reason
    if not (builtin("lane") and builtin("warp")) or any(
            type(s).__name__ == "Smem" and s.name in ("lane", "warp") for s in rest):
        return "lane or warp is bound to something other than the thread's lane and warp index"
    handed: dict[str, str] = {}  # warp partial -> the shared array lane 0 of each warp writes it to
    synced = False  # a barrier since those writes
    for s in rest[1:]:
        kind = type(s).__name__
        if kind == "TreeHalve":
            if not (int(s.length) * 32 == width and s.tid_var == "warp" and s.inner is None):
                return "the tree over the warps does not span the cooperating threads"
            reason = _same_combine(s, live, folds, ops, "tree")
            if reason is not None:
                return reason
            slot = dict(zip(s.state, s.bufs))
            lost = sorted(n for n in live if handed.get(n) != slot[n])
            if lost:
                return f"the tree does not fold {', '.join(lost)} from where lane 0 of every warp wrote it"
            return private(s) if synced else "no barrier between the warps' writes and the tree"
        if kind == "Cond":
            if not _is_lane_zero(s.cond) or s.else_body:
                return "the warp partials are written under a condition other than lane == 0"
            for x in s.body:
                if not (type(x).__name__ == "Write" and len(x.values) == 1 and x.values[0] in live - set(handed)
                        and tuple(x.index) == (Var("warp"),) and not x.atomic and x.swizzle == "NONE"):
                    return "something other than each warp's partials, once each to slot warp, is written there"
                handed[x.values[0]] = x.output
            synced = False
        elif kind == "Sync":
            synced = synced or not (s.warp or s.count is not None)  # (a warp's or a partial barrier: not all)
        elif kind != "Smem":
            return f"a {kind} sits between the butterfly and the tree"
    return "the warp partials are not combined by a tree over the warps"


def _is_lane_zero(e) -> bool:
    """Is condition ``e`` the test ``lane == 0``?"""
    return type(e).__name__ == "BinaryExpr" and e.op == "==" and any(
        x == Var("lane") and _int(y) == 0 for x, y in ((e.left, e.right), (e.right, e.left)))


def _canon(op: str) -> str:
    return {"sum": "add", "prod": "multiply", "amax": "maximum"}.get(op, op)


def _same_combine(stmt, live: set[str], folds: dict, ops: dict, what: str) -> str | None:
    """Why ``stmt`` (a WarpShuffle or TreeHalve) does not combine each of
    ``live`` with its own fold, ``acc = fold(acc, partner)``, or None."""
    partner = dict(zip(stmt.state, stmt.state_b))
    if live - set(partner):
        return f"the {what} does not combine {sorted(live - set(partner))}"
    ours = live | {partner[n] for n in live}
    seen = set()
    for a in stmt.combine_states:
        if a.name in live:
            if a.name in seen or set(a.args) != {a.name, partner[a.name]} or len(a.args) != 2:
                return f"the {what} computes {a.name} from {', '.join(a.args)}, not from {a.name} and its partner"
            if _canon(a.op.name) != ops[a.name]:
                return f"the {what} combines {a.name} with {a.op.name}, but the loop folds it with {ops[a.name]}"
            seen.add(a.name)
        elif set(a.args) & ours:
            return f"the {what} reads {sorted(set(a.args) & ours)} for {a.name}, which the loop's fold does not"
    if live - seen:
        return f"the {what} does not update {sorted(live - seen)}"
    dtype = getattr(stmt.dtype, "name", "f32")
    for n in live:
        fold_dtype = getattr(folds[n].dtype, "name", None) or "f32"
        if fold_dtype != dtype:
            return f"the {what} combines {n} in {dtype}, the loop folds it in {fold_dtype}"
    return None


def regroup(loop, start_hi: int, step: int, extent: int):
    """The chunked form of cooperative reduction ``loop`` (see module doc), or None.

    ``start_hi`` is the largest rank of the cooperating threads (they start at
    0..start_hi), ``step`` their number, ``extent`` the loop bound."""
    if type(loop).__name__ != "StridedLoop" or loop.end is not None or not loop.seed:
        return None
    threads = start_hi + 1
    copies = step // threads
    if step % threads or extent % (VECTOR * threads) != 0 or extent // (VECTOR * threads) < 1:
        return None
    a = loop.axis.name
    accs = set()
    for s in loop.body:
        kind = type(s).__name__
        if kind not in ("Load", "Assign", "Let", "Accum"):
            return None
        if kind == "Accum":
            if s.base is not None or s.op.name not in _FOLDS:
                return None
            accs.add(s.name)
    group: dict[str, int] = {}  # body name -> the copy (offset / threads) whose element it belongs to
    parts: list[list] = [[] for _ in range(copies)]
    for s in loop.body:  # pure fold: nothing reads an accumulator but its own Accum
        kind = type(s).__name__
        used = set(s.args) if kind == "Assign" else {s.value} if kind == "Accum" else set()
        if kind in ("Load", "Let"):
            used = {v for e in (s.index if kind == "Load" else (s.value,)) for v in e.free_vars()}
        if used & accs:
            return None
        if kind == "Load":
            offset = _offset(s.index[-1], a) if s.index else None
            if offset is None or offset % threads or not 0 <= offset // threads < copies:
                return None  # not the loop variable plus a copy's offset
            if any(a in e.free_vars() for e in s.index[:-1]):
                return None
            g = offset // threads
        else:
            owners = {group[n] for n in used if n in group}
            if len(owners) != 1:
                return None  # every value must belong to one copy's element
            g = owners.pop()
        for n in (s.names if kind == "Load" else (s.name,)):
            if group.setdefault(n, g) != g:
                return None  # an accumulator two copies fold into
        parts[g].append(s)
    if not all(_same_copy(parts[0], part, a) for part in parts[1:]):
        return None  # element j of a chunk goes to copy j mod R, wherever it was: each must compute it alike
    k = f"{a}$chunk"
    chunk = BinaryExpr("+", BinaryExpr("*", Literal(VECTOR, "int"), loop.start),
                       BinaryExpr("*", Literal(VECTOR * threads, "int"), Var(k)))
    body = []
    for j in range(VECTOR):
        g = j % copies
        element = BinaryExpr("+", chunk, Literal(j - g * threads, "int"))  # copy g reads a + g*threads
        local = {}  # each copy's own SSA names, so the copies stay independent
        for s in parts[g]:
            body.append(_copy(s, a, element, local, f"${j}", accs))
    return StridedLoop(axis=Axis(k, Dim(extent // (VECTOR * threads))), start=Literal(0, "int"),
                       step=Literal(1, "int"), body=tuple(body), unroll=loop.unroll, seed=loop.seed)


def _same_copy(first: list, other: list, a: str) -> bool:
    """Does the copy with statements ``other`` compute what copy ``first``
    computes, name for name, each reading the loop variable plus its own offset?"""
    if len(first) != len(other):
        return False
    names: dict[str, str] = {}  # other's names -> first's
    for s, t in zip(first, other):
        ours, theirs = (x.names if type(x).__name__ == "Load" else (x.name,) for x in (s, t))
        if type(s) is not type(t) or len(ours) != len(theirs):
            return False
        names.update(zip(theirs, ours))
        if renamed(_at_loop_variable(t, a), names) != _at_loop_variable(s, a):
            return False
    return True


def _at_loop_variable(s, a: str):
    """``s``, a Load reading the loop variable itself instead of its copy's offset from it."""
    return replace(s, index=(*s.index[:-1], Var(a))) if type(s).__name__ == "Load" else s


def _offset(e, a: str):
    """``c`` when expression ``e`` is the loop variable ``a`` plus the literal ``c``, else None."""
    if e == Var(a):
        return 0
    if type(e).__name__ == "BinaryExpr" and e.op == "+":
        for x, y in ((e.left, e.right), (e.right, e.left)):
            if x == Var(a) and _int(y) is not None:
                return _int(y)
    return None


def _copy(s, a: str, value, local: dict, suffix: str, accs: set[str]):
    """``s`` with the loop variable replaced by ``value`` and the SSA names the
    copy defines renamed with ``suffix`` (``local`` collects the renaming)."""
    sub = {a: value, **{old: Var(new) for old, new in local.items()}}
    kind = type(s).__name__
    if kind == "Load":
        names = tuple(n + suffix for n in s.names)
        local.update(zip(s.names, names))
        return replace(s, names=names, index=tuple(e.substitute(sub) for e in s.index))
    if kind == "Let":
        out = replace(s, name=s.name + suffix, value=s.value.substitute(sub))
        local[s.name] = s.name + suffix
        return out
    if kind == "Assign":
        out = replace(s, name=s.name + suffix, args=tuple(local.get(x, x) for x in s.args))
        local[s.name] = s.name + suffix
        return out
    return replace(s, value=local.get(s.value, s.value))  # Accum: the accumulator itself is shared
