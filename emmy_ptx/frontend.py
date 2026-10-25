"""Front end: read a Kernel-IR file and the launch descriptor beside it.

The IR is deserialized with emmy's ``Graph.from_dict``, the only piece of emmy
this compiler uses. ``meta.json`` (same folder) gives the kernel's parameter
order and the launch geometry. If it is missing, both are derived from the IR
the way emmy's CUDA backend derives them (with a warning on stderr).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from emmy.compiler.graph import Graph

# The thread count emmy gives a Tile that has no cooperative block_threads.
DEFAULT_BLOCK = 256


@dataclass(frozen=True)
class Buffer:
    """A global tensor the kernel reads or writes."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # emmy dtype name: f32, f16, bf16, f64, i32, i64, ...


@dataclass(frozen=True)
class Launch:
    grid: int  # CTAs along x
    block: int  # threads per CTA along x
    smem_bytes: int  # dynamic shared memory requested at launch


@dataclass(frozen=True)
class KernelSource:
    """Everything the back end needs about one kernel."""

    name: str
    body: tuple  # emmy Kernel-IR statements
    params: tuple[str, ...]  # buffer names in launch-argument order
    buffers: dict[str, Buffer]
    launch: Launch
    written: frozenset[str]  # global buffers this kernel stores to
    constants: dict = field(default_factory=dict)  # scalar ConstantOp buffers -> value (compile-time constants)


def load(ir_path: str | Path) -> KernelSource:
    ir_path = Path(ir_path)
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    kernels = [(nid, node) for nid, node in graph.nodes.items() if type(node.op).__name__ == "KernelOp"]
    if len(kernels) != 1:
        raise ValueError(f"{ir_path.name}: expected exactly one KernelOp, found {len(kernels)}")
    node_id, node = kernels[0]
    op = node.op
    body = tuple(op.body)

    buffers: dict[str, Buffer] = {}
    for n in graph.nodes.values():
        for t in n.outputs:
            shape = tuple(int(d.as_static()) for d in t.shape)
            buffers[t.name] = Buffer(t.name, shape, t.dtype.name)

    # A scalar ConstantOp is part of the program: its value is in the graph (the
    # runner passes the same value in a 1-element buffer, which stays a parameter).
    constants = {}
    for n in graph.nodes.values():
        value = getattr(n.op, "value", None) if type(n.op).__name__ == "ConstantOp" else None
        if (isinstance(value, (int, float)) and not isinstance(value, bool) and getattr(n.op, "source_path", None) is None
                and len(n.outputs) == 1 and math.prod(int(d.as_static()) for d in n.outputs[0].shape) == 1):
            constants[n.outputs[0].name] = value

    shared = {s.name for s in _walk(body) if type(s).__name__ == "Smem"}
    written = frozenset(_written_buffers(body) - shared)

    meta = _find_meta(ir_path, op.name, node_id)
    if meta is not None:
        params = tuple(meta["arg_order"])
        launch = Launch(int(meta["grid"][0]), int(meta["block"][0]), int(meta.get("smem_bytes", 0)))
        for arg in meta.get("args", ()):
            if arg["name"] not in buffers:
                buffers[arg["name"]] = Buffer(arg["name"], tuple(arg["shape"]), arg["dtype"])
    else:
        print(f"warning: no meta.json entry for {op.name}; deriving parameters and launch from the IR",
              file=sys.stderr)
        params = _signature_order(body, shared)
        launch = _derived_launch(body)
    return KernelSource(op.name, body, params, buffers, launch, written, constants)


def _walk(stmts):
    for s in stmts:
        yield s
        for child in s.nested():
            yield from _walk(child)


def _written_buffers(body) -> set[str]:
    out = set()
    for s in _walk(body):
        kind = type(s).__name__
        if kind == "Write":
            out.add(s.output)
        elif kind == "RegStore":
            out.add(s.dst_buffer)
        elif kind == "ZeroPrologue":
            out.add(s.dst)
    return out


def _find_meta(ir_path: Path, entry: str, node_id: str) -> dict | None:
    meta_path = ir_path.with_name("meta.json")
    if not meta_path.exists():
        return None
    kernels = json.loads(meta_path.read_text()).get("kernels", [])
    for key, want in (("entry", entry), ("node", node_id)):
        for k in kernels:
            if k.get(key) == want:
                return k
    return kernels[0] if len(kernels) == 1 else None


def _signature_order(body, shared: set[str]) -> tuple[str, ...]:
    """Emmy's signature rule: inputs by first read, then outputs by first write."""
    reads, writes = {}, {}
    for s in _walk(body):
        kind = type(s).__name__
        if kind == "Load" and s.input not in shared:
            reads.setdefault(s.input, None)
        elif kind == "CpAsyncCopy":
            reads.setdefault(s.src, None)
        elif kind == "Write" and s.output not in shared:
            writes.setdefault(s.output, None)
    return tuple(reads) + tuple(w for w in writes if w not in reads)


def _derived_launch(body) -> Launch:
    tiles = [s for s in _walk(body) if type(s).__name__ == "Tile"]
    if not tiles:
        return Launch(1, DEFAULT_BLOCK, 0)
    tile = tiles[0]
    cells = tile.block_threads or DEFAULT_BLOCK
    n = math.prod(int(a.extent.as_static()) for a in tile.axes)
    return Launch(max(1, -(-n // cells)), cells + tile.aux_threads, 0)
