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

It only applies when the statements right after the loop are exactly such a
full combine of the loop's accumulators across all B threads, the body is a
pure fold (loads, arithmetic, accumulations nothing else reads), every load
is contiguous in the loop variable, and V*B divides N, so no thread needs a
partial chunk.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Assign, Load, StridedLoop
from emmy.compiler.ir.stmt.leaves import Let

VECTOR = 4  # elements per chunk: one 16-byte load of f32
_FOLDS = ("add", "sum", "maximum", "amax", "minimum", "multiply", "prod")


def _int(e):
    return e.value if type(e).__name__ == "Literal" and not isinstance(e.value, float) else None


def fully_combined(stmts, i: int, accs: set[str], width: int) -> bool:
    """Do the statements after ``stmts[i]`` combine ``accs`` across all ``width``
    threads of the group (a warp butterfly, then a tree over the warps)?"""
    rest = list(stmts[i + 1 :])
    if not rest or type(rest[0]).__name__ != "WarpShuffle":
        return False
    shuffle = rest[0]
    if not accs <= set(shuffle.state) or int(shuffle.length) != min(width, 32):
        return False
    if width <= 32:
        return True
    for s in rest[1:]:
        kind = type(s).__name__
        if kind == "TreeHalve":
            return accs <= set(s.state) and int(s.length) * 32 == width and s.tid_var == "warp" and s.inner is None
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
    if start_hi != step - 1 or extent % (VECTOR * step) != 0 or extent // (VECTOR * step) < 1:
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
    for s in loop.body:  # pure fold: nothing reads an accumulator but its own Accum
        kind = type(s).__name__
        reads = set(s.args) if kind == "Assign" else {s.value} if kind == "Accum" else set()
        if kind in ("Load", "Let"):
            reads = {v for e in (s.index if kind == "Load" else (s.value,)) for v in e.free_vars()}
        if reads & accs:
            return None
        if kind == "Load" and not (s.index and s.index[-1] == Var(a) and all(a not in e.free_vars() for e in s.index[:-1])):
            return None  # not contiguous in the loop variable
    k = f"{a}$chunk"
    chunk = BinaryExpr("+", BinaryExpr("*", Literal(VECTOR, "int"), loop.start),
                       BinaryExpr("*", Literal(VECTOR * step, "int"), Var(k)))
    body = []
    for j in range(VECTOR):
        element = BinaryExpr("+", chunk, Literal(j, "int"))
        local = {}  # each copy's own SSA names, so the copies stay independent
        for s in loop.body:
            body.append(_copy(s, a, element, local, f"${j}", accs))
    return StridedLoop(axis=Axis(k, Dim(extent // (VECTOR * step))), start=Literal(0, "int"), step=Literal(1, "int"),
                       body=tuple(body), unroll=loop.unroll, seed=loop.seed)


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
