"""The copy engine for emmy's shared-memory fills (``compile.py --bulk-copy``).

Emmy fills a shared tile cooperatively: every thread issues a 4-16 byte
``cp.async``, the threads commit groups of copies, and ``cp.async.wait_group N``
waits until at most N groups are in flight. Where the lanes of an aligned group
copy one contiguous run (consecutive source and destination addresses,
``nbytes`` apart), the run's first lane can hand the whole run to the copy
engine (the TMA unit) with one ``cp.async.bulk``, which needs no tensor map;
the copy completes on an mbarrier instead of a commit group.

Whether a copy is such a run is decided here by evaluating its addresses for
every thread of the launch and every iteration of the loops around it, so a
fill that merely looks contiguous keeps its ``cp.async``. The Lowerer then
turns commits into an mbarrier arrival that expects the group's bytes, and
waits into waits on those mbarriers (see ``Lowerer.stmt_CpAsyncWait``).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

SLOTS = 8  # mbarriers, used round robin by consecutive commit groups (a power of two)
MBARRIERS = "bulk_mbarriers"  # their shared array
MAX_ITERATIONS = 4096  # loop iterations around a copy that are checked one by one


class Unsupported(Exception):
    """An expression the evaluator does not handle: the copy stays a cp.async."""


def plan(tile, grid: int, block: int, globals_: dict, shared: dict) -> dict:
    """``{CpAsyncCopy: lanes}``: the copies of Tile ``tile`` that aligned groups
    of ``lanes`` threads may hand to the copy engine as one run each.

    Only a Tile whose threads each take one cell in emmy's plain row-major
    decode qualifies (every thread then runs every copy outside a branch).
    ``globals_`` maps buffer names to (shape, element bytes), ``shared`` to
    (shape, element bytes, byte offset in its symbol)."""
    extents = [int(a.extent.as_static()) for a in tile.axes]
    n = math.prod(extents)
    if tile.aux_threads or n != grid * block or block % 32:
        return {}
    gid = np.arange(n, dtype=np.int64)
    env, stride = {}, n
    for axis, extent in zip(tile.axes, extents):
        stride //= extent
        env[axis.name] = (gid // stride) % extent
        # grouped rasterization permutes which block-level coordinates a CTA gets: the threads'
        # addresses are then checked for the same set of coordinates, in another order
        if axis.name in (tile.raster_axes or ()) and stride % block:
            return {}
    found: dict = {}
    _walk(tile.body, env, [], found, globals_, shared, block)
    return {s: lanes for s, lanes in found.items() if lanes >= 2}


def _walk(stmts, env: dict, loops: list, found: dict, globals_: dict, shared: dict, block: int) -> None:
    stmts = list(stmts)
    for i, s in enumerate(stmts):
        kind = type(s).__name__
        if kind == "CpAsyncCopy":
            # its group must be committed in this same statement list (so the
            # Lowerer counts the group's bytes in program order)
            if any(type(t).__name__ == "CpAsyncCommit" for t in stmts[i + 1:]):
                lanes = _run_lanes(s, env, loops, globals_, shared, block)
                found[s] = min(found.get(s, lanes), lanes)
            else:
                found[s] = 0
        elif kind in ("Loop", "StridedLoop"):
            values = _iterations(s, env)
            if values is None:
                _forbid(s.body, found)
            else:
                _walk(s.body, env, loops + [(s.axis.name, values)], found, globals_, shared, block)
        elif hasattr(s, "body"):
            _forbid(s.body, found)  # a branch (or anything else with a body): not every thread copies


def _forbid(stmts, found: dict) -> None:
    for s in stmts:
        if type(s).__name__ == "CpAsyncCopy":
            found[s] = 0
        if hasattr(s, "body"):
            _forbid(s.body, found)


def _iterations(loop, env: dict) -> list | None:
    """The loop variable's values when they are the same for every thread."""
    try:
        if type(loop).__name__ == "Loop":
            return list(range(int(loop.axis.extent.as_static())))
        start, step = _scalar(loop.start, env), _scalar(loop.step, env)
        end = _scalar(loop.end, env) if loop.end is not None else int(loop.axis.extent.as_static())
    except Unsupported:
        return None
    return list(range(start, end, step)) if step > 0 else None


def _scalar(e, env: dict) -> int:
    v = np.asarray(_eval(e, env))
    if v.ndim and not (v == v.flat[0]).all():
        raise Unsupported("differs between threads")
    return int(v.flat[0]) if v.ndim else int(v)


def _run_lanes(s, env: dict, loops: list, globals_: dict, shared: dict, block: int) -> int:
    """The largest aligned group of threads (a power of two, at least 2, within
    a block) that copies one contiguous, 16-byte aligned run in every
    iteration, or 0."""
    if s.swizzle != "NONE" or s.nbytes not in (4, 8, 16) or s.smem not in shared or s.src not in globals_:
        return 0
    if math.prod(len(v) for _, v in loops) > MAX_ITERATIONS:
        return 0
    sshape, selem, soffset = shared[s.smem]
    gshape, gelem = globals_[s.src]
    best = 1 << (len(env[next(iter(env))]).bit_length() - 1)  # a run may span warps, up to the whole launch
    try:
        for combo in itertools.product(*(values for _, values in loops)):
            e = dict(env)
            e.update({name: v for (name, _), v in zip(loops, combo)})
            dst = _flat(sshape, s.smem_index, e)
            if s.lane_index is not None:
                row, col = (_eval(x, e) for x in s.lane_index)
                dst = dst + row * sshape[-1] + col
            dst = soffset + selem * np.broadcast_to(dst, e[next(iter(env))].shape)
            src = gelem * np.broadcast_to(_flat(gshape, s.src_index, e), dst.shape)
            best = min(best, _contiguous(dst, src, s.nbytes, block))
            if best < 2:
                return 0
    except Unsupported:
        return 0
    return best


def _contiguous(dst: np.ndarray, src: np.ndarray, nbytes: int, block: int) -> int:
    lanes = block & -block  # runs stay inside a block (a power of two dividing it)
    while lanes >= 2:
        if len(dst) % lanes == 0 and lanes * nbytes % 16 == 0:
            d, g = dst.reshape(-1, lanes), src.reshape(-1, lanes)
            step = nbytes * np.arange(lanes)
            if ((d == d[:, :1] + step).all() and (g == g[:, :1] + step).all()
                    and (d[:, 0] % 16 == 0).all() and (g[:, 0] % 16 == 0).all()):
                return lanes
        lanes //= 2
    return 0


def _flat(shape, index, env: dict):
    """Row-major element offset of ``index`` (emmy's render_index)."""
    out, stride = 0, 1
    for extent, e in zip(reversed(shape), reversed(tuple(index))):
        out = out + _eval(e, env) * stride
        stride *= int(extent)
    return out


def _eval(e, env: dict):
    """An integer index expression over numpy arrays (C semantics)."""
    kind = type(e).__name__
    if kind == "Var":
        if e.name not in env:
            raise Unsupported(e.name)
        return env[e.name]
    if kind == "Literal":
        if isinstance(e.value, float) and not float(e.value).is_integer():
            raise Unsupported("float literal")
        return int(e.value)
    if kind == "CastExpr":
        return _eval(e.expr, env)
    if kind == "TernaryExpr":
        return np.where(_eval(e.cond, env), _eval(e.if_true, env), _eval(e.if_false, env))
    if kind == "BinaryExpr":
        a, b = _eval(e.left, env), _eval(e.right, env)
        op = e.op
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op in ("/", "%"):
            a, b = np.asarray(a), np.asarray(b)
            if (b == 0).any():
                raise Unsupported("division by zero")
            q = np.abs(a) // np.abs(b) * np.sign(a) * np.sign(b)  # C truncates toward zero
            return q if op == "/" else a - q * b
        compare = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
                   "==": np.equal, "!=": np.not_equal, "&&": np.logical_and, "||": np.logical_or}
        if op in compare:
            return compare[op](a, b)
    raise Unsupported(kind)
