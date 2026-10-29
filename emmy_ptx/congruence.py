"""Loop-carried accumulators that always hold the same value.

Emmy keeps some state redundantly. In attention, each output column a thread
handles carries its own running max and running sum of the online softmax,
yet the columns of one query row see the same scores, so their max and sum
agree at every iteration.

``congruent_accumulators`` proves such equalities with optimistic partition
refinement, the idea behind global value numbering:

1. assume that accumulators of the same kind starting from the same value
   are equal (one class per kind and start value);
2. evaluate one iteration of the loop body symbolically under that
   assumption: every value becomes a structural term (loads by buffer and
   index terms, operations by operator and operand terms, a class's value at
   the top of the iteration by the class). Each distinct term is numbered
   once and its users hold that number, so comparing two terms costs the same
   however deep they are (a chain of ``x + x`` would otherwise double the
   work at every step);
3. split every class whose members end the iteration with different terms,
   and repeat until no class splits.

What is left holds by induction over the iterations: equal at the start, and
if equal at the top of an iteration then equal at its end, because the terms
are identical computations on identical inputs.

``merge_congruent`` then rewrites the body so each class's representative is
read through one snapshot taken at the top of the iteration, and the other
members' updates are dropped (they compute the same value). The duplicated
arithmetic that fed them becomes identical code, which value numbering in
the builder removes.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.stmt.leaves import Let

from .mapping import reads

_SUPPORTED = ("Load", "Assign", "Let", "Init", "Accum", "Loop", "StridedLoop", "Select", "Write", "Cond")
_REDUCE = {"add": "add", "sum": "add", "multiply": "multiply", "prod": "multiply",
           "maximum": "maximum", "amax": "maximum", "fmax": "maximum", "minimum": "minimum", "fmin": "minimum"}


def _walk(stmts):
    for s in stmts:
        yield s
        for child in s.nested():
            yield from _walk(child)


def _dtype(s) -> str:
    return s.dtype.name if getattr(s, "dtype", None) is not None else "f32"


class _Eval:
    """Symbolic evaluation of one loop iteration (see module doc)."""

    def __init__(self, loopvars: dict[str, tuple], classes: dict[str, int], carried: set[str], written: set[str],
                 fresh=None, terms: dict | None = None):
        self.env: dict[str, tuple] = {}
        self.loopvars = loopvars  # loop variable -> its iteration term, outermost first
        self.classes = classes  # candidate accumulator -> class
        self.carried = carried  # every name the loop reassigns (non-candidates change too)
        self.written = written  # buffers stored inside the loop: their loads are fresh each time
        self.next: dict[str, tuple] = {}
        self._fresh = fresh if fresh is not None else itertools.count()  # shared: fresh terms never collide
        self._terms = terms if terms is not None else {}  # shared: a term's parts -> its number

    def fresh(self) -> tuple:
        return ("fresh", next(self._fresh))

    def term(self, *parts) -> tuple:
        """The term made of ``parts`` (operands among them, already terms), by its number."""
        return ("term", self._terms.setdefault(parts, len(self._terms)))

    def name(self, n: str) -> tuple:
        if n in self.env:
            return self.env[n]
        if n in self.classes:
            return ("head", self.classes[n])
        if n in self.loopvars:
            return self.loopvars[n]
        if n in self.carried:
            return ("carried", n)
        return ("outer", n)

    def expr(self, e) -> tuple:
        kind = type(e).__name__
        if kind == "Var":
            return self.name(e.name)
        if kind == "Literal":
            return ("lit", repr(e.value), e.dtype)
        if kind == "BinaryExpr":
            return self.term("bin", e.op, self.expr(e.left), self.expr(e.right))
        if kind == "TernaryExpr":
            return self.term("tern", self.expr(e.cond), self.expr(e.if_true), self.expr(e.if_false))
        if kind == "CastExpr":
            return self.term("cast", e.dtype, self.expr(e.expr))
        if kind == "FuncCallExpr":
            return self.term("call", e.name, tuple(self.expr(a) for a in e.args))
        if kind == "Builtin":
            return ("builtin", e.name)
        return self.fresh()

    def body(self, stmts) -> None:
        for s in stmts:
            kind = type(s).__name__
            if kind == "Load":
                idx = tuple(self.expr(e) for e in s.index)
                for k, n in enumerate(s.names):
                    fresh = s.input in self.written
                    self.env[n] = (self.fresh() if fresh else
                                   self.term("load", s.input, _dtype(s), idx, k, len(s.names)))
            elif kind == "Assign":
                self.env[s.name] = self.term("op", s.op.name, _dtype(s), tuple(self.name(a) for a in s.args))
            elif kind == "Let":
                self.env[s.name] = self.term("let", _dtype(s), self.expr(s.value))
            elif kind == "Init":
                self.env[s.name] = ("init", repr(s.identity), _dtype(s))
            elif kind == "Accum":
                base = self.name(s.base if s.base is not None else s.name)
                term = self.term("fold", _REDUCE.get(s.op.name, "add"), _dtype(s), base, self.name(s.value))
                if s.name in self.classes:
                    self.next[s.name] = term
                self.env[s.name] = term
            elif kind == "Select":
                self.env[s.name] = self.term("select", tuple((self.name(b.value), self.expr(b.select))
                                                             for b in s.branches))
            elif kind in ("Loop", "StridedLoop"):
                self.loop(s)
            elif kind == "Cond":  # names it declares are local; what it reassigns becomes unknown
                for n in _reassigned(s.body) | _reassigned(s.else_body):
                    self.env[n] = self.fresh()
            # Write: no value

    def loop(self, s) -> None:
        """An inner loop's accumulators: a structural term of the whole fold."""
        own = list(dict.fromkeys(st.name for st in s.body if type(st).__name__ == "Accum"))
        if any(type(st).__name__ not in _SUPPORTED for st in _walk(s.body)):
            for n in _reassigned(s.body):
                self.env[n] = self.fresh()
            return
        loopvars = {**self.loopvars, s.axis.name: ("iter", len(self.loopvars))}
        inner = _Eval(loopvars, self.classes, self.carried, self.written, self._fresh, self._terms)
        inner.env = dict(self.env)
        for i, n in enumerate(own):
            inner.env[n] = ("nhead", i)
        inner.body(s.body)
        if type(s).__name__ == "StridedLoop":
            bounds = ("strided", self.expr(s.start), self.expr(s.step), self.expr(s.end) if s.end is not None else None,
                      str(s.axis.extent))
        else:
            bounds = ("range", str(s.axis.extent))
        seeds = tuple(("identity", _REDUCE.get(_op_of(s.body, n), "add")) if s.seed else self.name(n) for n in own)
        signature = self.term("loop", bounds, seeds, tuple(inner.env.get(n) for n in own))
        for i, n in enumerate(own):
            self.env[n] = self.term("after", signature, i)
        for n in _reassigned(s.body) - set(own):
            self.env[n] = self.fresh()


def _op_of(body, name: str) -> str:
    return next(st.op.name for st in body if type(st).__name__ == "Accum" and st.name == name)


def _reassigned(stmts) -> set[str]:
    out = set()
    for s in _walk(stmts):
        kind = type(s).__name__
        if kind == "Accum":
            out.add(s.name)
        elif kind in ("WarpShuffle", "TreeHalve"):
            out.update(s.state)
    return out


def _written(stmts) -> set[str]:
    out = set()
    for s in _walk(stmts):
        kind = type(s).__name__
        if kind == "Write":
            out.add(s.output)
        elif kind == "CpAsyncCopy":
            out.add(s.smem)
    return out


def congruent_accumulators(loop, start_values: dict[str, object]) -> dict[str, str]:
    """``member -> representative`` for accumulators of ``loop`` that provably
    hold equal values at every iteration.

    ``start_values`` maps each accumulator the loop updates to an identity of
    its value when the loop starts (None when unknown). Only accumulators
    updated by exactly one Accum, at the top level of the body, take part.
    """
    body = loop.body
    if any(type(s).__name__ not in _SUPPORTED for s in _walk(body)):
        return {}
    tops = [s for s in body if type(s).__name__ == "Accum"]
    counts: dict[str, int] = {}
    for s in _walk(body):
        if type(s).__name__ == "Accum":
            counts[s.name] = counts.get(s.name, 0) + 1
    order = list(dict.fromkeys(s.name for s in tops))
    candidates = [n for n in order if counts[n] == 1 and start_values.get(n) is not None]
    if len(candidates) < 2:
        return {}
    kind = {s.name: (_REDUCE.get(s.op.name, "add"), _dtype(s), s.base is None) for s in tops}
    groups: list[list[str]] = []
    by_start: dict = {}
    for n in candidates:
        by_start.setdefault((kind[n], start_values[n]), []).append(n)
    groups = [g for g in by_start.values()]
    carried = _reassigned(body)
    written = _written(body)
    while True:
        classes = {n: i for i, g in enumerate(groups) for n in g}
        ev = _Eval({loop.axis.name: ("iter", 0)}, classes, carried, written)
        ev.body(body)
        split: list[list[str]] = []
        for g in groups:
            parts: dict = {}
            for n in g:
                parts.setdefault(ev.next.get(n, ("missing", n)), []).append(n)
            split.extend(parts.values())
        if len(split) == len(groups):
            break
        groups = split
    return {n: g[0] for g in groups if len(g) > 1 for n in g[1:]}


def merge_congruent(body, members: dict[str, str]):
    """Rewrite ``body`` so each class shares its representative (see module
    doc). Returns None when a name involved is redeclared inside the body,
    which would make the renaming ambiguous."""
    reps = list(dict.fromkeys(members.values()))
    involved = set(members) | set(reps)
    for s in _walk(body):
        kind = type(s).__name__
        defined = set(s.names) if kind == "Load" else {s.name} if kind in ("Assign", "Let", "Init", "Select") else set()
        if defined & involved or (kind in ("WarpShuffle", "TreeHalve") and set(s.state) & involved):
            return None
    head = {r: f"{r}$top" for r in reps}
    out = [Let(name=head[r], value=Var(r)) for r in reps]
    rename = {r: head[r] for r in reps}
    rename.update({m: head[members[m]] for m in members})
    for s in body:
        kind = type(s).__name__
        if kind == "Accum" and s.name in members:
            rename[s.name] = members[s.name]  # from here on the member reads the updated representative
            continue
        out.append(_renamed(s, rename))
        if kind == "Accum" and s.name in rename and s.name in reps:
            del rename[s.name]  # the representative now holds its updated value
    return tuple(out)


def drop_needless_snapshots(body) -> tuple:
    """``body`` without the snapshots ``Let r$top = r`` (from merge_congruent)
    whose every read comes before ``r``'s own update: up to there ``r`` still
    holds the snapshot's value, so those reads can name ``r`` itself. (A
    snapshot some read needs after the update stays, and keeps all its reads,
    so equal computations of members and representative stay equal.)"""
    body = tuple(body)
    first = {}
    for i, s in enumerate(body):
        if type(s).__name__ == "Accum":
            first.setdefault(s.name, i)
    drop = {}
    for s in body:
        if type(s).__name__ == "Let" and s.name.endswith("$top") and type(s.value).__name__ == "Var":
            r = s.value.name
            readers = [i for i, x in enumerate(body) if s.name in reads((x,))]
            if all(i <= first.get(r, -1) for i in readers):
                drop[s.name] = r
    if not drop:
        return body
    return tuple(_renamed(s, drop) for s in body if not (type(s).__name__ == "Let" and s.name in drop))


def _renamed(s, rename: dict[str, str]):
    """Statement ``s`` with reads of the names in ``rename`` redirected."""
    if not rename:
        return s
    sub = {old: Var(new) for old, new in rename.items()}
    kind = type(s).__name__

    def n(x):
        return rename.get(x, x)

    def e(x):
        return x.substitute(sub) if x is not None else None

    if kind == "Load":
        return replace(s, index=tuple(e(i) for i in s.index))
    if kind == "Assign":
        return replace(s, args=tuple(n(a) for a in s.args))
    if kind == "Let":
        return replace(s, value=e(s.value))
    if kind == "Accum":
        return replace(s, value=n(s.value), base=n(s.base) if s.base is not None else None)
    if kind == "Select":
        return replace(s, branches=tuple(replace(b, value=n(b.value), select=e(b.select)) for b in s.branches))
    if kind == "Write":
        return replace(s, index=tuple(e(i) for i in s.index), values=tuple(n(v) for v in s.values))
    if kind == "Cond":
        return replace(s, cond=e(s.cond), body=tuple(_renamed(x, rename) for x in s.body),
                       else_body=tuple(_renamed(x, rename) for x in s.else_body))
    if kind in ("Loop", "StridedLoop"):
        changes = {"body": tuple(_renamed(x, rename) for x in s.body)}
        if kind == "StridedLoop":
            changes.update(start=e(s.start), step=e(s.step), end=e(s.end))
        return replace(s, **changes)
    return s  # Init and statements without reads
