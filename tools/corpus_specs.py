"""A wider problem set than the open one, in the organisers' spec format.

The hidden evaluation set has the same five categories as the open set with
"different shapes, sizes and op mixes". This file is our stand-in for it:
sizes that do not tile evenly, 2-D/3-D shapes, broadcasting, every
elementwise op emmy can trace, max/min reductions, layernorm and
log-softmax, odd matmul shapes, and attention at other lengths.

Generate problem folders with the organisers' tool (see tools/make_corpus.py):

    python <hackathon>/tools/generate.py --emmy <emmy> --out corpus tools/corpus_specs.py

Each spec: ``id``, ``code`` (the torch expression emmy traces), ``ref`` (numpy,
taking the graph inputs in creation order), ``rtol``/``atol`` and optional
``knobs``. Tolerances are the organisers' where an open problem sets one, and
deliberately tight (1e-6) for transcendental pointwise ops.
"""

import math

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"

f64 = np.float64
f32 = np.float32
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _c(fn):
    """Reference computed in float64, rounded once to float32."""
    return lambda *a: [np.asarray(fn(*(x.astype(f64) for x in a))).astype(f32)]


def _e(fn):
    """Reference computed in float32 (exact IEEE ops, compare with rtol=0)."""
    return lambda *a: [np.asarray(fn(*a)).astype(f32)]


def pw(id, code, ref, rtol=1e-6, atol=1e-6):
    return dict(id=f"p_{id}", code=code, ref=ref, rtol=rtol, atol=atol)


def red(id, code, ref, rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS):
    return dict(id=f"r_{id}", code=code, ref=ref, rtol=rtol, atol=atol, knobs=knobs)


def mm(id, m, k, n):
    return dict(id=f"m_{id}", code=f"torch.randn({m}, {k}) @ torch.randn({k}, {n})",
                ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3)


def attn(id, nq, nk, d, scale):
    code = (f"q = torch.randn({nq}, {d}); kt = torch.randn({d}, {nk}); v = torch.randn({nk}, {d}); "
            f"torch.softmax(q @ kt / {scale}, dim=-1) @ v")
    return dict(id=f"a_{id}", code=code, ref=_c(lambda q, kt, v: _softmax((q @ kt) / scale) @ v), rtol=1e-3, atol=1e-4)


OPEN = [
    # ---- category 1: pointwise ----------------------------------------------------
    pw("add_100", "torch.randn(100) + torch.randn(100)", _e(lambda a, b: a + b), 0, 0),
    pw("mul_1000", "torch.randn(1000) * torch.randn(1000)", _e(lambda a, b: a * b), 0, 0),
    pw("sub_3000", "torch.randn(3000) - torch.randn(3000)", _e(lambda a, b: a - b), 0, 0),
    pw("div_4096", "torch.randn(4096) / torch.randn(4096)", _e(lambda a, b: a / b), 0, 0),
    pw("bcast_row", "x = torch.randn(64, 128); b = torch.randn(128); x + b", _e(lambda x, b: x + b), 0, 0),
    pw("bcast_col", "x = torch.randn(32, 96); s = torch.randn(32, 1); x * s", _e(lambda x, s: x * s), 0, 0),
    pw("neg", "torch.neg(torch.randn(2048))", _e(lambda x: -x), 0, 0),
    pw("abs", "torch.abs(torch.randn(2048))", _e(np.abs), 0, 0),
    pw("relu_5000", "torch.relu(torch.randn(5000))", _e(lambda x: np.maximum(x, 0)), 0, 0),
    pw("square_sum", "x = torch.randn(2048); y = torch.randn(2048); x * x + y * y",
       _e(lambda x, y: x * x + y * y), 0, 0),
    pw("diff_sq", "x = torch.randn(1536); y = torch.randn(1536); (x - y) * (x + y)",
       _e(lambda x, y: (x - y) * (x + y)), 0, 0),
    pw("max2", "torch.maximum(torch.randn(2048), torch.randn(2048))", _e(np.maximum), 0, 0),
    pw("min2", "torch.minimum(torch.randn(2048), torch.randn(2048))", _e(np.minimum), 0, 0),
    pw("clamp", "torch.clamp(torch.randn(4096), -0.5, 0.5)", _e(lambda x: np.clip(x, f32(-0.5), f32(0.5))), 0, 0),
    pw("leaky", "x = torch.randn(4096); torch.where(x > 0, x, x * 0.1)",
       _e(lambda x: np.where(x > 0, x, x * f32(0.1))), 0, 0),
    pw("recip", "x = torch.randn(4096); 1.0 / (x * x + 1.0)", _c(lambda x: 1.0 / (x * x + 1.0))),
    pw("affine2d", "torch.randn(256, 1024) * 2.0 + torch.randn(256, 1024)",
       _e(lambda a, b: a * f32(2.0) + b), 1e-6, 1e-6),
    pw("big_1m", "torch.randn(1048576) + 1.0", _e(lambda x: x + f32(1.0)), 0, 0),
    pw("sigmoid", "torch.sigmoid(torch.randn(4096))", _c(lambda x: 1.0 / (1.0 + np.exp(-x)))),
    pw("tanh", "torch.tanh(torch.randn(4096))", _c(np.tanh)),
    pw("exp", "torch.exp(torch.randn(4096))", _c(np.exp)),
    pw("exp_neg_sq", "x = torch.randn(4096); torch.exp(-x * x)", _c(lambda x: np.exp(-x * x))),
    pw("log", "torch.log(torch.abs(torch.randn(4096)) + 1.0)", _c(lambda x: np.log(np.abs(x) + 1.0))),
    pw("log_small", "torch.log(torch.abs(torch.randn(4096)))", _c(lambda x: np.log(np.abs(x)))),
    pw("sqrt", "torch.sqrt(torch.abs(torch.randn(4096)))", _c(lambda x: np.sqrt(np.abs(x)))),
    pw("rsqrt", "torch.rsqrt(torch.abs(torch.randn(4096)) + 1.0)", _c(lambda x: 1.0 / np.sqrt(np.abs(x) + 1.0))),
    pw("sin", "torch.sin(torch.randn(4096))", _c(np.sin)),
    pw("cos", "torch.cos(torch.randn(4096))", _c(np.cos)),
    # the product is rounded to float32 before the sine, on the GPU as in torch
    pw("sin_wide", "torch.sin(torch.randn(4096) * 100.0)",
       lambda x: [np.sin((x * f32(100.0)).astype(f64)).astype(f32)]),
    pw("erf", "torch.erf(torch.randn(4096))", _c(_erf)),
    pw("gelu", "torch.nn.functional.gelu(torch.randn(4096))", _c(lambda x: 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0))))),
    pw("gelu_tanh", "torch.nn.functional.gelu(torch.randn(4096), approximate='tanh')",
       _c(lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))))),
    pw("silu", "torch.nn.functional.silu(torch.randn(4096))", _c(lambda x: x / (1.0 + np.exp(-x)))),
    pw("softplus", "torch.nn.functional.softplus(torch.randn(4096))", _c(lambda x: np.log1p(np.exp(x)))),
    pw("swiglu", "x = torch.randn(4096); y = torch.randn(4096); torch.nn.functional.silu(x) * y",
       _c(lambda x, y: x / (1.0 + np.exp(-x)) * y)),
    pw("pow_sq", "torch.randn(4096) ** 2", _e(lambda x: x * x), 1e-6, 1e-6),
    pw("pow_cube", "torch.randn(4096) ** 3", _c(lambda x: x ** 3)),
    pw("pow_frac", "torch.abs(torch.randn(4096)) ** 1.5", _c(lambda x: np.abs(x) ** 1.5)),
    pw("hardsig", "x = torch.randn(4096); torch.clamp(x / 6.0 + 0.5, 0.0, 1.0)", _c(lambda x: np.clip(x / 6.0 + 0.5, 0.0, 1.0))),
    pw("elu", "x = torch.randn(4096); torch.where(x > 0, x, torch.exp(x) - 1.0)", _c(lambda x: np.where(x > 0, x, np.expm1(x)))),
    pw("mix3", "x = torch.randn(64, 256); y = torch.randn(64, 256); torch.tanh(x) * torch.sigmoid(y) + x",
       _c(lambda x, y: np.tanh(x) / (1.0 + np.exp(-y)) + x)),

    # ---- category 2: reductions ------------------------------------------------------
    red("sum_1000", "torch.randn(1000).sum()", _c(lambda x: x.sum())),
    red("sum_100k", "torch.randn(100000).sum()", _c(lambda x: x.sum()), 1e-3, 1e-3),
    red("sum_4097", "torch.randn(4097).sum()", _c(lambda x: x.sum())),
    red("mean_5000", "torch.randn(5000).mean()", _c(lambda x: x.mean())),
    red("max_4096", "torch.randn(4096).amax()", _c(lambda x: x.max())),
    red("min_rows", "torch.randn(48, 3000).amin(-1)", _c(lambda x: x.min(-1))),
    red("amax_rows", "torch.randn(32, 512).amax(-1)", _c(lambda x: x.max(-1))),
    red("sumsq", "x = torch.randn(8192); (x * x).sum()", _c(lambda x: (x * x).sum())),
    red("abs_rows", "torch.abs(torch.randn(64, 300)).sum(-1)", _c(lambda x: np.abs(x).sum(-1))),
    red("rows_1000", "torch.randn(64, 1000).sum(-1)", _c(lambda x: x.sum(-1))),
    red("rows_mean", "torch.randn(128, 256).mean(-1)", _c(lambda x: x.mean(-1))),
    red("rows_100x300", "torch.randn(100, 300).sum(-1)", _c(lambda x: x.sum(-1))),
    red("rows_7", "torch.randn(256, 7).sum(-1)", _c(lambda x: x.sum(-1))),
    red("cols", "torch.randn(256, 64).sum(0)", _c(lambda x: x.sum(0))),
    red("rows_3d", "torch.randn(8, 16, 64).sum(-1)", _c(lambda x: x.sum(-1))),
    red("rows_keep", "torch.randn(32, 128).sum(-1, keepdim=True)", _c(lambda x: x.sum(-1, keepdims=True))),
    red("sum_default", "torch.randn(65536).sum()", _c(lambda x: x.sum()), 1e-3, 1e-3, knobs=""),
    red("rows_default", "torch.randn(512, 512).sum(-1)", _c(lambda x: x.sum(-1)), knobs=""),
    red("max_default", "torch.randn(128, 1024).amax(-1)", _c(lambda x: x.max(-1)), knobs=""),

    # ---- category 3: matmul -------------------------------------------------------------
    mm("64", 64, 64, 64),
    mm("128", 128, 128, 128),
    mm("256x512x128", 256, 512, 128),
    mm("1024", 1024, 1024, 1024),
    mm("512x64x512", 512, 64, 512),
    mm("tall", 2048, 64, 64),
    mm("wide", 32, 128, 2048),
    mm("deep_k", 64, 4096, 64),
    mm("odd", 100, 200, 50),
    mm("odd2", 33, 65, 17),
    mm("row", 1, 512, 512),
    dict(id="m_bias", code="x = torch.randn(128, 256); w = torch.randn(256, 128); b = torch.randn(128); x @ w + b",
         ref=_c(lambda x, w, b: x @ w + b), rtol=1e-3, atol=1e-3),
    dict(id="m_relu", code="x = torch.randn(128, 256); w = torch.randn(256, 64); torch.relu(x @ w)",
         ref=_c(lambda x, w: np.maximum(x @ w, 0)), rtol=1e-3, atol=1e-3),
    dict(id="m_bmm", code="torch.randn(4, 64, 128) @ torch.randn(4, 128, 64)",
         ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3),

    # ---- category 4: fused ----------------------------------------------------------------
    dict(id="f_rms_4096", code="x = torch.randn(32, 4096); w = torch.randn(4096); "
                                "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=_c(lambda x, w: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * w), rtol=1e-3, atol=1e-4),
    dict(id="f_rms_rsqrt", code="x = torch.randn(128, 768); w = torch.randn(768); "
                                 "x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=_c(lambda x, w: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * w), rtol=1e-3, atol=1e-4),
    dict(id="f_rms_odd", code="x = torch.randn(7, 300); w = torch.randn(300); "
                               "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=_c(lambda x, w: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * w), rtol=1e-3, atol=1e-4),
    dict(id="f_layernorm", code="torch.nn.functional.layer_norm(torch.randn(64, 512), (512,))",
         ref=_c(lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)),
         rtol=1e-3, atol=1e-4),
    dict(id="f_layernorm_w", code="x = torch.randn(32, 256); w = torch.randn(256); b = torch.randn(256); "
                                   "torch.nn.functional.layer_norm(x, (256,), w, b)",
         ref=_c(lambda x, w, b: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * w + b),
         rtol=1e-3, atol=1e-4),
    dict(id="f_center", code="x = torch.randn(64, 384); x - x.mean(-1, keepdim=True)",
         ref=_c(lambda x: x - x.mean(-1, keepdims=True)), rtol=1e-3, atol=1e-4),
    dict(id="f_softmax_1000", code="torch.softmax(torch.randn(128, 1000), dim=-1)",
         ref=_c(_softmax), rtol=1e-3, atol=1e-5),
    dict(id="f_softmax_4096", code="torch.softmax(torch.randn(16, 4096), dim=-1)",
         ref=_c(_softmax), rtol=1e-3, atol=1e-5),
    dict(id="f_softmax_50", code="torch.softmax(torch.randn(64, 50), dim=-1)",
         ref=_c(_softmax), rtol=1e-3, atol=1e-5),
    dict(id="f_softmax_scaled", code="torch.softmax(torch.randn(64, 256) * 0.5, dim=-1)",
         ref=_c(lambda x: _softmax(x * 0.5)), rtol=1e-3, atol=1e-5),
    dict(id="f_softmax_col", code="torch.softmax(torch.randn(128, 64), dim=0)",
         ref=_c(lambda x: _softmax(x, axis=0)), rtol=1e-3, atol=1e-5),
    dict(id="f_logsoftmax", code="x = torch.randn(64, 256); x - torch.log(torch.exp(x).sum(-1, keepdim=True))",
         ref=_c(lambda x: x - np.log(np.exp(x).sum(-1, keepdims=True))), rtol=1e-3, atol=1e-4),
    dict(id="f_l2_1024", code="x = torch.randn(32, 1024); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
         ref=_c(lambda x: x / np.sqrt((x ** 2).sum(-1, keepdims=True))), rtol=1e-3, atol=1e-4),
    # (an assigned name is a graph input: `x = torch.abs(...)` would feed raw normals)
    dict(id="f_normsum", code="x = torch.randn(64, 200); torch.abs(x) / torch.abs(x).sum(-1, keepdim=True)",
         ref=_c(lambda x: np.abs(x) / np.abs(x).sum(-1, keepdims=True)), rtol=1e-3, atol=1e-4),
    dict(id="f_maxshift", code="x = torch.randn(64, 256); torch.exp(x - x.amax(-1, keepdim=True))",
         ref=_c(lambda x: np.exp(x - x.max(-1, keepdims=True))), rtol=1e-3, atol=1e-5),
    dict(id="f_gelu_mean", code="x = torch.randn(64, 512); torch.nn.functional.gelu(x).mean(-1)",
         ref=_c(lambda x: (0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))).mean(-1)), rtol=1e-3, atol=1e-4),

    # ---- category 5: attention ---------------------------------------------------------------
    attn("128x64", 128, 128, 64, 8.0),
    attn("256x32", 256, 256, 32, 5.656854),
    attn("64x128", 64, 64, 128, 11.313708),
    attn("q64_k256", 64, 256, 32, 5.656854),
    attn("q256_k64", 256, 64, 64, 8.0),
    attn("512x64", 512, 512, 64, 8.0),
    attn("odd", 100, 100, 48, 6.928203),
]
