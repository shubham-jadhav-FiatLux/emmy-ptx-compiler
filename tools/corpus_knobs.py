"""The same core expressions forced through many emmy schedules (EMMY_KNOBS).

A hidden problem may pin a schedule the open set never uses; each schedule
family produces different Kernel IR (thread counts, transposed cooperative
trees, register partials, cross-CTA splits finalized by atomics or by a
second kernel, raster orders, staging depths, warp tiles). Build with
``tools/make_corpus.py --specs tools/corpus_knobs.py``; combinations emmy
rejects for a shape are simply reported and skipped.
"""

import numpy as np

f64, f32 = np.float64, np.float32


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _c(fn):
    return lambda *a: [np.asarray(fn(*(x.astype(f64) for x in a))).astype(f32)]


BASE = {
    "sum": ("torch.randn(65536).sum()", _c(lambda x: x.sum()), 1e-3, 1e-3),
    "rows": ("torch.randn(64, 512).sum(-1)", _c(lambda x: x.sum(-1)), 1e-4, 1e-4),
    "amax": ("torch.randn(96, 384).amax(-1)", _c(lambda x: x.max(-1)), 1e-4, 1e-4),
    "softmax": ("torch.softmax(torch.randn(64, 1024), dim=-1)", _c(_softmax), 1e-3, 1e-5),
    "rms": ("x = torch.randn(32, 2048); w = torch.randn(2048); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
            _c(lambda x, w: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * w), 1e-3, 1e-4),
    "mm": ("torch.randn(256, 256) @ torch.randn(256, 256)", _c(lambda a, b: a @ b), 1e-3, 1e-3),
    "mm_rect": ("torch.randn(128, 512) @ torch.randn(512, 192)", _c(lambda a, b: a @ b), 1e-3, 1e-3),
    "attn": ("q = torch.randn(128, 64); kt = torch.randn(64, 128); v = torch.randn(128, 64); "
             "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
             _c(lambda q, kt, v: _softmax((q @ kt) / 8.0) @ v), 1e-3, 1e-4),
}

REDUCE_KNOBS = [
    "WORK=t32,REDUCE=coop", "WORK=t64,REDUCE=coop", "WORK=t256,REDUCE=coop", "WORK=t512,REDUCE=coop",
    "WORK=t1024,REDUCE=coop", "WORK=t128,REDUCE=coop-t", "WORK=t64,REDUCE=coop-t", "WORK=t128,REDUCE=coop/r2",
    "WORK=t128,REDUCE=coop/r4", "WORK=t128,REDUCE=g4a/coop", "WORK=t128,REDUCE=g4k/coop", "REDUCE=g8a",
    "REDUCE=g4k", "REDUCE=r4", "WORK=t32,REDUCE=coop/r2",
]
MATMUL_KNOBS = [
    "RASTER=gm8", "RASTER=gn4", "RASTER=gn8", "STAGE=d1/smem", "STAGE=d2/smem", "STAGE=d2/smem-async",
    "STAGE=d4/smem-async", "REDUCE=g4a", "REDUCE=g4k", "REDUCE=g2k", "WORK=w2x2", "WORK=w2x2+p1", "WORK=w4x2",
    "TILE=f4x4", "TILE=f2x2", "TILE=f8x4", "WORK=t16x16",
]
ATTN_KNOBS = ["RASTER=gm8", "RASTER=gn4", "WORK=t64", "WORK=t256", "STAGE=d2/smem-async"]


def _id(base, knobs):
    tag = knobs.lower().replace("=", "-").replace(",", "_").replace("/", "-").replace("+", "p")
    return f"x_{base}__{tag}"


OPEN = []
for base in ("sum", "rows", "amax", "softmax", "rms"):
    code, ref, rtol, atol = BASE[base]
    OPEN += [dict(id=_id(base, k), code=code, ref=ref, rtol=rtol, atol=atol, knobs=k) for k in REDUCE_KNOBS]
for base in ("mm", "mm_rect"):
    code, ref, rtol, atol = BASE[base]
    OPEN += [dict(id=_id(base, k), code=code, ref=ref, rtol=rtol, atol=atol, knobs=k) for k in MATMUL_KNOBS]
code, ref, rtol, atol = BASE["attn"]
OPEN += [dict(id=_id("attn", k), code=code, ref=ref, rtol=rtol, atol=atol, knobs=k) for k in ATTN_KNOBS]
