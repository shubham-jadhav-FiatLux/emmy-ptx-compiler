"""A third held-out problem set, written on 27 Oct after the correctness fixes of that day were frozen.

Same purpose and conventions as ``corpus_holdout.py``: the hidden set's five
categories, with shapes and op mixes that no earlier set has. It also aims at
the edges those fixes cover: sines and cosines of large arguments, divisions
by constants and means checked at rtol = 0, minima, maxima and clamps, and
norms over odd lengths. It was generated and run once, after the fixes, to
estimate how the compiler does on problems it was not developed against.

    python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --specs tools/corpus_holdout3.py --out holdout3
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_holdout import _c, _e, _erf, _softmax, attn, f32, f64, fused, mm, pw, red, sdpa  # noqa: E402


def _causal(q, k, v, scale):
    s = q @ np.swapaxes(k, -1, -2) / scale
    n = s.shape[-1]
    return _softmax(np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)) @ v


def _gelu(x):
    return 0.5 * x * (1 + _erf(x / np.sqrt(2.0)))


def _f32_then(fn, scale):
    """The reference of fn(x * scale) where torch forms x * scale in float32 first."""
    return lambda x: [np.asarray(fn((x * f32(scale)).astype(f64))).astype(f32)]


def again(spec):
    """Round-3 ids: h3 instead of h, so all rounds can live side by side."""
    return dict(spec, id="h3" + spec["id"][1:])


OPEN = [again(s) for s in [
    # ---- category 1: pointwise -------------------------------------------
    pw("sin_1e6", "torch.sin(torch.randn(4096) * 1000000.0)", _f32_then(np.sin, 1e6), 1e-5, 1e-6),
    pw("cos_1e30", "torch.cos(torch.randn(2048) * 1e30)", _f32_then(np.cos, 1e30), 1e-5, 1e-6),
    pw("div_7_3", "torch.randn(3333) / 7.3", _e(lambda x: x / f32(7.3)), 0, 0),
    pw("div_neg_0_3", "torch.randn(64, 70) / -0.3", _e(lambda x: x / f32(-0.3)), 0, 0),
    pw("clamp_2d", "torch.clamp(torch.randn(64, 70), -0.5, 0.5)", _e(lambda x: np.clip(x, f32(-0.5), f32(0.5))), 0, 0),
    pw("min_times_max", "x = torch.randn(2500); y = torch.randn(2500); torch.minimum(x, y) * torch.maximum(x, y)",
       _e(lambda x, y: np.minimum(x, y) * np.maximum(x, y)), 0, 0),
    pw("abs_diff_3d", "torch.abs(torch.randn(5, 7, 9) - torch.randn(5, 7, 9))", _e(lambda a, b: np.abs(a - b)), 0, 0),
    pw("square_5000", "x = torch.randn(5000); x * x", _e(lambda x: x * x), 0, 0),
    pw("sigmoid_1023", "torch.sigmoid(torch.randn(1023))", _c(lambda x: 1 / (1 + np.exp(-x)))),
    pw("exp_neg_abs", "torch.exp(-torch.abs(torch.randn(6000)))", _c(lambda x: np.exp(-np.abs(x)))),

    # ---- category 2: reductions ------------------------------------------
    red("sum_77777", "torch.randn(77777).sum()", _c(lambda x: x.sum()), 1e-3, 1e-3),
    red("amax_rows_300x129", "torch.randn(300, 129).amax(-1)", _c(lambda x: x.max(-1))),
    red("amin_cols_64x500", "torch.randn(64, 500).amin(0)", _c(lambda x: x.min(0))),
    red("mean_rows_3x50000", "torch.randn(3, 50000).mean(-1)", _c(lambda x: x.mean(-1))),
    red("sum_mid_16x123x8", "torch.randn(16, 123, 8).sum(1)", _c(lambda x: x.sum(1))),
    red("var_rows_32x400", "x = torch.randn(32, 400); ((x - x.mean(-1, keepdim=True)) ** 2).mean(-1)",
        _c(lambda x: ((x - x.mean(-1, keepdims=True)) ** 2).mean(-1))),
    red("norm_rows_50x777", "x = torch.randn(50, 777); torch.sqrt((x * x).sum(-1))",
        _c(lambda x: np.sqrt((x * x).sum(-1))), 1e-3, 1e-3),
    red("mean_exact_1024x3", "torch.randn(1024, 3).mean(-1)", _c(lambda x: x.mean(-1))),

    # ---- category 3: matmul ----------------------------------------------
    mm("200x300x100", 200, 300, 100),
    mm("17x1000x23", 17, 1000, 23),
    mm("96x96x96", 96, 96, 96),
    mm("1024x32x256", 1024, 32, 256),
    dict(id="hm_bmm_6x40x70x30", code="torch.randn(6, 40, 70) @ torch.randn(6, 70, 30)",
         ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3),
    dict(id="hm_xwt_64x128x96", code="x = torch.randn(64, 128); w = torch.randn(96, 128); x @ w.t()",
         ref=_c(lambda x, w: x @ w.T), rtol=1e-3, atol=1e-3),
    dict(id="hm_scale_shift", code="(torch.randn(120, 80) @ torch.randn(80, 60)) * 0.5 + 1.0",
         ref=_c(lambda a, b: (a @ b) * 0.5 + 1.0), rtol=1e-3, atol=1e-3),

    # ---- category 4: fused -----------------------------------------------
    fused("layernorm_100x300",
          "x = torch.randn(100, 300); w = torch.randn(300); b = torch.randn(300); "
          "torch.nn.functional.layer_norm(x, (300,), w, b)",
          _c(lambda x, w, b: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * w + b)),
    fused("softmax_7x777", "torch.softmax(torch.randn(7, 777), dim=-1)", _c(_softmax), atol=1e-5),
    fused("rms_45x333", "x = torch.randn(45, 333); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)",
          _c(lambda x: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6))),
    fused("gelu_mm_bias", "x = torch.randn(48, 80); w = torch.randn(80, 112); b = torch.randn(112); "
                          "torch.nn.functional.gelu(x @ w + b)",
          _c(lambda x, w, b: _gelu(x @ w + b)), 1e-3, 1e-3),
    fused("mm_softmax", "x = torch.randn(40, 96); w = torch.randn(96, 200); torch.softmax(x @ w, dim=-1)",
          _c(lambda x, w: _softmax(x @ w)), 1e-3, 1e-5),
    fused("half_relu_residual", "x = torch.randn(72, 72); w = torch.randn(72, 72); x + 0.5 * torch.relu(x @ w)",
          _c(lambda x, w: x + 0.5 * np.maximum(x @ w, 0)), 1e-3, 1e-3),
    fused("center_div3", "x = torch.randn(60, 333); (x - x.mean(-1, keepdim=True)) / 3.0",
          _c(lambda x: (x - x.mean(-1, keepdims=True)) / 3.0)),
    fused("mm_rowmax", "x = torch.randn(64, 100); w = torch.randn(100, 50); (x @ w).amax(-1)",
          _c(lambda x, w: (x @ w).max(-1)), 1e-3, 1e-3),

    # ---- category 5: attention-like --------------------------------------
    attn("96x96x48", 96, 96, 48, 6.928203),
    attn("33x77x40", 33, 77, 40, 6.324555),
    attn("512x128x32", 512, 128, 32, 5.656854),
    attn("1x1000x64", 1, 1000, 64, 8.0),
    sdpa("sdpa_2x3x50x24", (2, 3, 50, 24)),
    sdpa("sdpa_1x1x128x128", (1, 1, 128, 128)),
    dict(id="ha_sdpa_causal_1x4x64x32",
         code="q = torch.randn(1, 4, 64, 32); k = torch.randn(1, 4, 64, 32); v = torch.randn(1, 4, 64, 32); "
              "torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)",
         ref=_c(lambda q, k, v: _causal(q, k, v, np.sqrt(32))), rtol=1e-3, atol=1e-4),
]]
