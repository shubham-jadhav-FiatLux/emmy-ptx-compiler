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
full combine of the loop's accumulators across all B threads, the body is a
pure fold (loads, arithmetic, accumulations nothing else reads), every load
reads the loop variable plus a multiple of B, and V*B divides N, so no thread
needs a partial chunk.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Assign, Load, StridedLoop
from emmy.compiler.ir.stmt.leaves import Let

from .mapping import reads

VECTOR = 4  # elements per chunk: one 16-byte load of f32
_FOLDS = ("add", "sum", "maximum", "amax", "minimum", "multiply", "prod")


def _int(e):
    return e.value if type(e).__name__ == "Literal" and not isinstance(e.value, float) else None


def fully_combined(stmts, i: int, accs: set[str], width: int) -> bool:
    """Do the statements after ``stmts[i]`` combine ``accs`` across all ``width``
    threads of the group (a warp butterfly, then a tree over the warps)? Some
    of the accumulators may first be folded into the others (an R-times
    unrolled loop's partials, with the loop's own combine)."""
    ops = {s.name: s.op.name for s in stmts[i].body if type(s).__name__ == "Accum"}
    rest = list(stmts[i + 1 :])
    merged = set()
    while rest and type(rest[0]).__name__ == "Accum" and rest[0].base is None:
        fold = rest[0]
        if fold.name not in accs or fold.value not in accs - merged - {fold.name} or ops[fold.name] != fold.op.name:
            break
        if ops[fold.value] != fold.op.name:
            return False
        merged.add(fold.value)
        rest = rest[1:]
    if merged & reads(rest):
        return False  # a partial folded away is still read
    if not rest or type(rest[0]).__name__ != "WarpShuffle":
        return False
    shuffle = rest[0]
    if not accs - merged <= set(shuffle.state) or int(shuffle.length) != min(width, 32):
        return False
    if width <= 32:
        return True
    for s in rest[1:]:
        kind = type(s).__name__
        if kind == "TreeHalve":
            whole = int(s.length) * 32 == width and s.tid_var == "warp" and s.inner is None
            return accs - merged <= set(s.state) and whole
        if kind not in ("Smem", "Sync", "Cond"):
            return False
        if kind == "Cond" and not all(type(x).__name__ == "Write" for x in (*s.body, *s.else_body)):
            return False
    return False


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
