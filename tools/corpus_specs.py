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

# ---- round 2: chains of kernels, large sizes, more op mixes ---------------------------------
OPEN += [
    # several kernels per problem (intermediates chained by the runner)
    dict(id="k_mlp", code="x = torch.randn(64, 128); w1 = torch.randn(128, 256); w2 = torch.randn(256, 64); "
                          "torch.relu(x @ w1) @ w2",
         ref=_c(lambda x, w1, w2: np.maximum(x @ w1, 0) @ w2), rtol=1e-3, atol=1e-3),
    dict(id="k_mm_softmax", code="x = torch.randn(128, 256); w = torch.randn(256, 512); torch.softmax(x @ w, dim=-1)",
         ref=_c(lambda x, w: _softmax(x @ w)), rtol=1e-3, atol=1e-5),
    dict(id="k_mm_rowsum", code="x = torch.randn(64, 512); w = torch.randn(512, 256); (x @ w).sum(-1)",
         ref=_c(lambda x, w: (x @ w).sum(-1)), rtol=1e-3, atol=1e-3),
    dict(id="k_center_global", code="x = torch.randn(65536); x - x.mean()",
         ref=_c(lambda x: x - x.mean()), rtol=1e-4, atol=1e-4),
    dict(id="k_center_global_coop", code="x = torch.randn(8192); x - x.mean()",
         ref=_c(lambda x: x - x.mean()), rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="k_div_total", code="x = torch.randn(4096); x / (x * x).sum()",
         ref=_c(lambda x: x / (x * x).sum()), rtol=1e-4, atol=1e-6),
    dict(id="k_ln_mm", code="x = torch.randn(64, 256); w = torch.randn(256, 128); "
                            "torch.nn.functional.layer_norm(x, (256,)) @ w",
         ref=_c(lambda x, w: ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)) @ w),
         rtol=1e-3, atol=1e-3),
    dict(id="k_gelu_mlp", code="x = torch.randn(32, 128); w1 = torch.randn(128, 512); w2 = torch.randn(512, 128); "
                               "torch.nn.functional.gelu(x @ w1) @ w2",
         ref=_c(lambda x, w1, w2: (0.5 * (x @ w1) * (1.0 + _erf((x @ w1) / np.sqrt(2.0)))) @ w2), rtol=1e-3, atol=1e-3),
    dict(id="k_softmax_then_mm", code="x = torch.randn(128, 256); w = torch.randn(256, 64); torch.softmax(x, dim=-1) @ w",
         ref=_c(lambda x, w: _softmax(x) @ w), rtol=1e-3, atol=1e-4),

    # large sizes
    pw("big_2m", "torch.randn(2048, 1024) * torch.randn(2048, 1024)", _e(lambda a, b: a * b), 0, 0),
    red("sum_1m", "torch.randn(1048576).sum()", _c(lambda x: x.sum()), 1e-3, 1e-3),
    red("rows_big", "torch.randn(1024, 4096).sum(-1)", _c(lambda x: x.sum(-1)), 1e-4, 1e-3),
    mm("2048", 2048, 2048, 2048),
    dict(id="f_softmax_big", code="torch.softmax(torch.randn(1024, 4096), dim=-1)",
         ref=_c(_softmax), rtol=1e-3, atol=1e-5),
    attn("1024x64", 1024, 1024, 64, 8.0),

    # pointwise mixes
    pw("expm1", "torch.exp(torch.randn(4096)) - 1.0", lambda x: [(np.exp(x.astype(f64)).astype(f32) - f32(1.0))], 1e-6, 1e-6),
    pw("mish", "x = torch.randn(4096); x * torch.tanh(torch.nn.functional.softplus(x))",
       _c(lambda x: x * np.tanh(np.log1p(np.exp(x))))),
    pw("hypot", "x = torch.randn(4096); y = torch.randn(4096); torch.sqrt(x * x + y * y)",
       _e(lambda x, y: np.sqrt(x * x + y * y)), 0, 0),
    pw("absdiff", "x = torch.randn(4096); y = torch.randn(4096); torch.where(x > y, x - y, y - x)",
       _e(lambda x, y: np.where(x > y, x - y, y - x)), 0, 0),
    pw("relu_sq", "torch.relu(torch.randn(4096)) ** 2", _e(lambda x: np.maximum(x, 0) * np.maximum(x, 0)), 1e-6, 1e-6),
    pw("sincos", "x = torch.randn(4096); y = torch.randn(4096); torch.cos(x) * torch.sin(y)",
       _c(lambda x, y: np.cos(x) * np.sin(y))),
    pw("exp_sin", "torch.exp(torch.sin(torch.randn(4096)))", _c(lambda x: np.exp(np.sin(x)))),
    pw("manual_sigmoid", "x = torch.randn(4096); 1.0 / (1.0 + torch.exp(-x))", _c(lambda x: 1.0 / (1.0 + np.exp(-x)))),
    pw("dsigmoid", "x = torch.randn(4096); torch.sigmoid(x) * (1.0 - torch.sigmoid(x))",
       _c(lambda x: 1.0 / (1.0 + np.exp(-x)) * (1.0 - 1.0 / (1.0 + np.exp(-x))))),
    pw("log2", "torch.log2(torch.abs(torch.randn(4096)) + 0.5)", _c(lambda x: np.log2(np.abs(x) + 0.5))),
    pw("exp2", "torch.exp2(torch.randn(4096))", _c(np.exp2)),
    pw("floor", "torch.floor(torch.randn(4096) * 4.0)", _e(lambda x: np.floor(x * f32(4.0))), 0, 0),
    pw("sign", "torch.sign(torch.randn(4096))", _e(np.sign), 0, 0),
    pw("bcast_3d", "x = torch.randn(8, 16, 32); y = torch.randn(16, 1); x * y + 1.0",
       _e(lambda x, y: x * y + f32(1.0)), 1e-6, 1e-6),
    pw("scale_shift_2d", "torch.randn(300, 7) * 0.5 + 0.25", _e(lambda x: x * f32(0.5) + f32(0.25)), 0, 0),

    # reductions
    red("cols_mean", "torch.randn(64, 512).mean(0)", _c(lambda x: x.mean(0))),
    red("mid_axis", "torch.randn(32, 64, 16).sum(1)", _c(lambda x: x.sum(1))),
    red("prod", "torch.prod(torch.randn(256) * 0.01 + 1.0)", lambda x: [np.prod((x * f32(0.01) + f32(1.0)).astype(f64)).astype(f32)]),
    red("var", "torch.var(torch.randn(64, 512), -1)", _c(lambda x: x.var(-1, ddof=1)), 1e-3, 1e-4),
    red("std", "torch.std(torch.randn(64, 512), -1)", _c(lambda x: x.std(-1, ddof=1)), 1e-3, 1e-4),
    red("norm", "torch.norm(torch.randn(64, 512), dim=-1)", _c(lambda x: np.sqrt((x * x).sum(-1))), 1e-3, 1e-4),
    red("logsumexp", "torch.logsumexp(torch.randn(64, 512), -1)",
        _c(lambda x: np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1)) + x.max(-1)), 1e-3, 1e-4),
    red("short_rows", "torch.randn(1000, 10).sum(-1)", _c(lambda x: x.sum(-1))),
    red("long_rows", "torch.randn(10, 10000).sum(-1)", _c(lambda x: x.sum(-1)), 1e-3, 1e-3),
    red("rowdot", "x = torch.randn(128, 256); y = torch.randn(128, 256); (x * y).sum(-1)", _c(lambda x, y: (x * y).sum(-1))),
    red("absmax", "torch.randn(64, 777).abs().amax(-1)", _c(lambda x: np.abs(x).max(-1))),
    red("cols_max", "torch.randn(500, 64).amax(0)", _c(lambda x: x.max(0))),

    # matmul variants
    dict(id="m_batched_2d", code="torch.randn(4, 32, 64) @ torch.randn(64, 128)",
         ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3),
    dict(id="m_bmm_small", code="torch.randn(8, 16, 32) @ torch.randn(8, 32, 16)",
         ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3),
    dict(id="m_scaled", code="x = torch.randn(96, 160); w = torch.randn(160, 80); (x @ w) * 0.5",
         ref=_c(lambda x, w: (x @ w) * 0.5), rtol=1e-3, atol=1e-3),
    mm("48x96x80", 48, 96, 80),
    mm("1000x100x10", 1000, 100, 10),
    mm("7x1000x3", 7, 1000, 3),

    # fused variants
    dict(id="f_softmax_row1", code="torch.softmax(torch.randn(1, 10000), dim=-1)", ref=_c(_softmax), rtol=1e-3, atol=1e-6),
    dict(id="f_softmax_tiny_rows", code="torch.softmax(torch.randn(10000, 8), dim=-1)", ref=_c(_softmax), rtol=1e-3, atol=1e-5),
    dict(id="f_softmax_3d_mid", code="torch.softmax(torch.randn(8, 64, 32), dim=1)",
         ref=_c(lambda x: _softmax(x, axis=1)), rtol=1e-3, atol=1e-5),
    dict(id="f_layernorm_odd", code="torch.nn.functional.layer_norm(torch.randn(37, 1000), (1000,))",
         ref=_c(lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)), rtol=1e-3, atol=1e-4),
    dict(id="f_rms_row1", code="x = torch.randn(1, 8192); w = torch.randn(8192); "
                                "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=_c(lambda x, w: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5) * w), rtol=1e-3, atol=1e-4),
    dict(id="f_normalize", code="torch.nn.functional.normalize(torch.randn(64, 300), dim=-1)",
         ref=_c(lambda x: x / np.maximum(np.sqrt((x * x).sum(-1, keepdims=True)), 1e-12)), rtol=1e-3, atol=1e-4),
    dict(id="f_x_softmax", code="x = torch.randn(64, 256); x * torch.softmax(x, -1)",
         ref=_c(lambda x: x * _softmax(x)), rtol=1e-3, atol=1e-5),
    # Not included: ((x - x.mean(-1, keepdim=True)) ** 2).mean(-1). emmy 0.3.11 split-K's the
    # outer mean but computes the inner mean over the first half-row only, so its own nvcc
    # reference fails too (checked with tools/ref_bench.py); our PTX reproduces that IR exactly.

    # attention variants
    attn("q64_k1024", 64, 1024, 64, 8.0),
    attn("512x32", 512, 512, 32, 5.656854),
    dict(id="a_batched", code="q = torch.randn(4, 64, 32); kt = torch.randn(4, 32, 64); v = torch.randn(4, 64, 32); "
                              "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=_c(lambda q, kt, v: _softmax((q @ kt) / 5.656854) @ v), rtol=1e-3, atol=1e-4),
    dict(id="a_mulscale", code="q = torch.randn(128, 64); kt = torch.randn(64, 128); v = torch.randn(128, 64); "
                               "torch.softmax((q @ kt) * 0.125, dim=-1) @ v",
         ref=_c(lambda q, kt, v: _softmax((q @ kt) * 0.125) @ v), rtol=1e-3, atol=1e-4),
    dict(id="a_noscale", code="q = torch.randn(64, 16); kt = torch.randn(16, 64); v = torch.randn(64, 16); "
                              "torch.softmax(q @ kt, dim=-1) @ v",
         ref=_c(lambda q, kt, v: _softmax(q @ kt) @ v), rtol=1e-3, atol=1e-4),
]


def _causal(q, k, v, scale):
    s = q @ np.swapaxes(k, -1, -2) / scale
    n = s.shape[-1]
    s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
    return _softmax(s) @ v


def sdpa(id, shape, causal=False):
    b, h, n, d = shape
    t = f"torch.randn({b}, {h}, {n}, {d})"
    code = (f"q = {t}; k = {t}; v = {t}; "
            f"torch.nn.functional.scaled_dot_product_attention(q, k, v{', is_causal=True' if causal else ''})")
    if causal:
        ref = _c(lambda q, k, v: _causal(q, k, v, np.sqrt(d)))
    else:
        ref = _c(lambda q, k, v: _softmax(q @ np.swapaxes(k, -1, -2) / np.sqrt(d)) @ v)
    return dict(id=f"a_{id}", code=code, ref=ref, rtol=1e-3, atol=1e-4)


# ---- round 3: attention in odd shapes (the new optimizations all fire here) ------------------
OPEN += [
    attn("d24", 96, 96, 24, 4.898979),
    attn("d40", 64, 160, 40, 6.324555),
    attn("d80", 128, 128, 80, 8.944272),
    attn("d96", 48, 200, 96, 9.797959),
    attn("n384", 384, 384, 32, 5.656854),
    attn("q1", 1, 256, 64, 8.0),
    attn("k16", 128, 16, 64, 8.0),
    sdpa("sdpa_b2h2", (2, 2, 100, 32)),
    sdpa("sdpa_h8", (1, 8, 64, 16)),
    sdpa("sdpa_causal_odd", (1, 2, 96, 48), causal=True),
    sdpa("sdpa_causal_long", (1, 1, 256, 64), causal=True),
    sdpa("sdpa", (1, 1, 128, 64)),
    sdpa("sdpa_causal", (1, 1, 128, 64), causal=True),
    sdpa("sdpa_heads", (2, 4, 64, 32)),
    # inputs are numbered by first use in the expression: q, kt, mask, v
    dict(id="a_masked", code="q = torch.randn(64, 32); kt = torch.randn(32, 64); m = torch.randn(64, 64); "
                              "v = torch.randn(64, 32); torch.softmax(q @ kt / 5.656854 + m, dim=-1) @ v",
         ref=_c(lambda q, kt, m, v: _softmax((q @ kt) / 5.656854 + m) @ v), rtol=1e-3, atol=1e-4),
    dict(id="a_relu_attn", code="q = torch.randn(64, 32); kt = torch.randn(32, 128); v = torch.randn(128, 32); "
                                 "torch.relu(q @ kt) @ v",
         ref=_c(lambda q, kt, v: np.maximum(q @ kt, 0) @ v), rtol=1e-3, atol=1e-3),
    dict(id="a_sigmoid_attn", code="q = torch.randn(64, 32); kt = torch.randn(32, 64); v = torch.randn(64, 32); "
                                    "torch.sigmoid(q @ kt / 5.656854) @ v",
         ref=_c(lambda q, kt, v: (1.0 / (1.0 + np.exp(-(q @ kt) / 5.656854))) @ v), rtol=1e-3, atol=1e-4),
]


# ---- round 4: attention tilings that leave no aligned group of lanes one row (cell placement) ----
def attn_v(id, nq, nk, d, dv, batch=None):
    """softmax(q @ kt / sqrt(d)) @ v with v of width dv, batched when ``batch``."""
    b = f"{batch}, " if batch else ""
    scale = round(math.sqrt(d), 6)
    code = (f"q = torch.randn({b}{nq}, {d}); kt = torch.randn({b}{d}, {nk}); v = torch.randn({b}{nk}, {dv}); "
            f"torch.softmax(q @ kt / {scale}, dim=-1) @ v")
    return dict(id=f"a_{id}", code=code, ref=_c(lambda q, kt, v: _softmax((q @ kt) / scale) @ v), rtol=1e-3, atol=1e-4)


def relu_attn(id, nq, nk, d, dv):
    code = f"q = torch.randn({nq}, {d}); kt = torch.randn({d}, {nk}); v = torch.randn({nk}, {dv}); torch.relu(q @ kt) @ v"
    return dict(id=f"a_{id}", code=code, ref=_c(lambda q, kt, v: np.maximum(q @ kt, 0) @ v), rtol=1e-3, atol=1e-3)


def sigmoid_attn(id, nq, nk, d, dv):
    scale = round(math.sqrt(d), 6)
    code = (f"q = torch.randn({nq}, {d}); kt = torch.randn({d}, {nk}); v = torch.randn({nk}, {dv}); "
            f"torch.sigmoid(q @ kt / {scale}) @ v")
    return dict(id=f"a_{id}", code=code, ref=_c(lambda q, kt, v: 1 / (1 + np.exp(-(q @ kt) / scale)) @ v),
                rtol=1e-3, atol=1e-4)


OPEN += [
    attn_v("b2_64x32", 64, 64, 32, 32, batch=2),
    attn_v("b3_32x32", 32, 48, 32, 32, batch=3),
    attn_v("b4_64x48", 64, 64, 48, 48, batch=4),
    attn_v("b2_128x32", 128, 128, 32, 32, batch=2),
    attn_v("b8_32x32", 32, 32, 32, 32, batch=8),
    attn_v("d40_k96", 64, 96, 40, 40),
    attn_v("d48", 64, 160, 48, 48),
    attn_v("d56", 64, 128, 56, 56),
    attn_v("d72", 48, 128, 72, 72),
    attn_v("d88", 32, 128, 88, 88),
    attn_v("d40_q128", 128, 160, 40, 40),
    attn_v("dv24", 64, 128, 32, 24),
    attn_v("dv40", 64, 128, 64, 40),
    relu_attn("relu_64x32", 64, 64, 32, 32),
    relu_attn("relu_32x48", 32, 128, 32, 48),
    relu_attn("relu_128x32", 128, 256, 32, 32),
    sigmoid_attn("sig_64x48", 64, 64, 48, 48),
    sigmoid_attn("sig_96x32", 96, 128, 32, 32),
]


# ---- round 5: float16 on the tensor cores (emmy's mma.sync schedules) -----------------------
def _h(fn):
    """Reference computed in float64, rounded once to float16."""
    return lambda *a: [np.asarray(fn(*(x.astype(f64) for x in a))).astype(np.float16)]


H = "dtype=torch.float16"


def mm16(id, m, k, n):
    return dict(id=f"m_f16_{id}", code=f"torch.randn({m}, {k}, {H}) @ torch.randn({k}, {n}, {H})",
                ref=_h(lambda a, b: a @ b), rtol=1e-2, atol=1e-2)


def sdpa16(id, shape, causal=False):
    b, h, n, d = shape
    t = f"torch.randn({b}, {h}, {n}, {d}, {H})"
    code = (f"q = {t}; k = {t}; v = {t}; "
            f"torch.nn.functional.scaled_dot_product_attention(q, k, v{', is_causal=True' if causal else ''})")
    if causal:
        ref = _h(lambda q, k, v: _causal(q, k, v, np.sqrt(d)))
    else:
        ref = _h(lambda q, k, v: _softmax(q @ np.swapaxes(k, -1, -2) / np.sqrt(d)) @ v)
    return dict(id=f"a_f16_{id}", code=code, ref=ref, rtol=2e-2, atol=2e-2)


def attn16(id, nq, nk, d, scale):
    code = (f"q = torch.randn({nq}, {d}, {H}); kt = torch.randn({d}, {nk}, {H}); v = torch.randn({nk}, {d}, {H}); "
            f"torch.softmax(q @ kt / {scale}, dim=-1) @ v")
    return dict(id=f"a_f16_{id}", code=code, ref=_h(lambda q, kt, v: _softmax((q @ kt) / scale) @ v),
                rtol=2e-2, atol=2e-2)


OPEN += [
    mm16("128", 128, 128, 128),
    mm16("256", 256, 256, 256),
    mm16("64x512x128", 64, 512, 128),
    mm16("512x64x512", 512, 64, 512),
    mm16("96x200x72", 96, 200, 72),
    mm16("33x77x55", 33, 77, 55),
    mm16("1x256x128", 1, 256, 128),
    mm16("200x40x24", 200, 40, 24),
    dict(id="m_f16_bmm", code=f"torch.randn(4, 64, 96, {H}) @ torch.randn(4, 96, 80, {H})",
         ref=_h(lambda a, b: a @ b), rtol=1e-2, atol=1e-2),
    dict(id="m_f16_bt", code=f"x = torch.randn(128, 96, {H}); w = torch.randn(64, 96, {H}); x @ w.t()",
         ref=_h(lambda x, w: x @ w.T), rtol=1e-2, atol=1e-2),
    dict(id="m_f16_scale", code=f"x = torch.randn(64, 128, {H}); w = torch.randn(128, 96, {H}); (x @ w) * 0.125",
         ref=_h(lambda x, w: (x @ w) * 0.125), rtol=1e-2, atol=1e-2),
    dict(id="f_f16_linear_relu", code=f"x = torch.randn(128, 128, {H}); w = torch.randn(128, 64, {H}); "
                                      f"b = torch.randn(64, {H}); torch.relu(x @ w + b)",
         ref=_h(lambda x, w, b: np.maximum(x @ w + b, 0)), rtol=1e-2, atol=1e-2),
    dict(id="f_f16_linear_gelu", code=f"x = torch.randn(64, 256, {H}); w = torch.randn(256, 128, {H}); "
                                      f"torch.nn.functional.gelu(x @ w)",
         ref=_h(lambda x, w: 0.5 * (x @ w) * (1 + _erf((x @ w) / math.sqrt(2)))), rtol=1e-2, atol=1e-2),
    dict(id="f_f16_residual", code=f"x = torch.randn(96, 64, {H}); w = torch.randn(64, 64, {H}); x @ w + x",
         ref=_h(lambda x, w: x @ w + x), rtol=1e-2, atol=1e-2),
    attn16("64x128", 64, 128, 64, 8.0),
    attn16("128", 128, 128, 64, 8.0),
    attn16("d32", 64, 256, 32, 5.656854),
    sdpa16("sdpa", (1, 1, 128, 64)),
    sdpa16("sdpa_causal", (1, 1, 128, 64), causal=True),
    sdpa16("sdpa_heads", (2, 4, 64, 32)),
    # not included: a causal (1, 2, 96, 48) sdpa, whose two kernels the generator lists consumer first
]


def mlp(act: str, m: int, k: int, h: int, n: int):
    """``act(x @ w1) @ w2`` fused into one kernel: emmy recomputes the hidden layer for every
    output column, which the compiler shares across a cluster of blocks (cluster.py)."""
    torch_act = {"gelu": "torch.nn.functional.gelu", "relu": "torch.relu", "tanh": "torch.tanh",
                 "silu": "torch.nn.functional.silu"}[act]
    np_act = {"gelu": lambda z: 0.5 * z * (1.0 + _erf(z / np.sqrt(2.0))), "relu": lambda z: np.maximum(z, 0),
              "tanh": np.tanh, "silu": lambda z: z / (1.0 + np.exp(-z))}[act]
    code = (f"x = torch.randn({m}, {k}); w1 = torch.randn({k}, {h}); w2 = torch.randn({h}, {n}); "
            f"{torch_act}(x @ w1) @ w2")
    return dict(id=f"k_mlp_{act}_{m}x{k}x{h}x{n}", code=code, ref=_c(lambda x, w1, w2: np_act(x @ w1) @ w2),
                rtol=1e-3, atol=1e-3)


# ---- round 6: fused MLPs, whose hidden layer emmy recomputes for every output column ----------
OPEN += [
    mlp("gelu", 64, 256, 1024, 128),
    mlp("relu", 32, 128, 512, 64),
    mlp("tanh", 16, 64, 256, 256),
    mlp("silu", 32, 256, 512, 128),
    mlp("gelu", 8, 512, 2048, 64),
]
