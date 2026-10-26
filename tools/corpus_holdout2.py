"""A second held-out problem set, written on 26 Oct after every optimization in DESIGN.md.

Same purpose and conventions as ``corpus_holdout.py`` (the hidden set's five
categories, the open set's tolerances and reduction schedule), with shapes and
op mixes that neither that set nor the corpus has: prime and odd sizes, single
rows and single queries, middle-axis and column reductions, matrix-vector
products, batched matmuls against a shared matrix, fused epilogues around a
matmul, causal and multi-head attention, and a few float16 problems.

    python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --specs tools/corpus_holdout2.py --out holdout2
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_holdout import _c, _e, _erf, _softmax, attn, f32, f64, fused, mm, pw, red, sdpa  # noqa: E402


def _h(fn):
    """Reference computed in float64, rounded once to float16."""
    return lambda *a: [np.asarray(fn(*(x.astype(f64) for x in a))).astype(np.float16)]


def _causal(q, k, v, scale):
    s = q @ np.swapaxes(k, -1, -2) / scale
    n = s.shape[-1]
    return _softmax(np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)) @ v


def _gelu(x):
    return 0.5 * x * (1 + _erf(x / np.sqrt(2.0)))


H = "dtype=torch.float16"


def again(spec):
    """Round-2 ids: h2 instead of h, so both rounds can live side by side."""
    return dict(spec, id="h2" + spec["id"][1:])


OPEN = [again(s) for s in [
    # ---- category 1: pointwise -------------------------------------------
    pw("add_4097", "torch.randn(4097) + torch.randn(4097)", _e(lambda a, b: a + b), 0, 0),
    pw("mul_add", "x = torch.randn(3000); y = torch.randn(3000); z = torch.randn(3000); x * y + z",
       _e(lambda x, y, z: x * y + z), 0, 0),
    pw("div_const", "torch.randn(5000) / 3.0", _e(lambda x: x / f32(3.0)), 0, 0),
    pw("div_const_1m", "torch.randn(1048576) / 7.0", _e(lambda x: x / f32(7.0)), 0, 0),
    pw("neg_abs", "torch.abs(torch.randn(2222)) * -1.0", _e(lambda x: np.abs(x) * f32(-1.0)), 0, 0),
    pw("where_pick", "x = torch.randn(1500); y = torch.randn(1500); torch.where(x > 0, x, y)",
       _e(lambda x, y: np.where(x > 0, x, y)), 0, 0),
    pw("outer_32x64", "torch.randn(32, 1) * torch.randn(1, 64)", _e(lambda a, b: a * b), 0, 0),
    pw("sigmoid_gate", "x = torch.randn(4096); y = torch.randn(4096); torch.sigmoid(x) * y",
       _c(lambda x, y: y / (1 + np.exp(-x)))),
    pw("tanh_3d", "torch.tanh(torch.randn(8, 16, 33))", _c(np.tanh)),
    pw("log_abs_2", "torch.log(torch.abs(torch.randn(3001)) + 2.0)", _c(lambda x: np.log(np.abs(x) + 2.0))),
    pw("erf_999", "torch.erf(torch.randn(999))", _c(_erf)),
    pw("silu_3d", "x = torch.randn(4, 100, 10); x * torch.sigmoid(x)", _c(lambda x: x / (1 + np.exp(-x)))),
    pw("big_1m", "torch.randn(1048576) + 1.0", _e(lambda x: x + f32(1.0))),
    pw("exp_scaled", "torch.exp(torch.randn(7000) * 0.5)", _c(lambda x: np.exp(x * 0.5))),
    pw("bcast_3d", "torch.randn(6, 1, 40) + torch.randn(1, 30, 40)", _e(lambda a, b: a + b), 0, 0),

    # ---- category 2: reductions ------------------------------------------
    red("sum_100003", "torch.randn(100003).sum()", _c(lambda x: x.sum()), 1e-3, 1e-3),
    red("sum_1m", "torch.randn(1048576).sum()", _c(lambda x: x.sum()), 1e-3, 1e-3),
    red("mean_rows_17x9999", "torch.randn(17, 9999).mean(-1)", _c(lambda x: x.mean(-1))),
    red("amax_rows_1000x33", "torch.randn(1000, 33).amax(-1)", _c(lambda x: x.max(-1))),
    red("cols_1000x8", "torch.randn(1000, 8).sum(0)", _c(lambda x: x.sum(0)), 1e-3, 1e-3),
    red("mid_axis_8x300x16", "torch.randn(8, 300, 16).sum(1)", _c(lambda x: x.sum(1))),
    red("mean_keep_3d", "torch.randn(4, 50, 120).mean(-1, keepdim=True)", _c(lambda x: x.mean(-1, keepdims=True))),
    red("sumsq_rows_64x1000", "x = torch.randn(64, 1000); (x * x).sum(-1)", _c(lambda x: (x * x).sum(-1)), 1e-3, 1e-3),
    red("absmax_40x700", "torch.abs(torch.randn(40, 700)).amax(-1)", _c(lambda x: np.abs(x).max(-1))),
    red("sum_exp_32x256", "torch.exp(torch.randn(32, 256)).sum(-1)", _c(lambda x: np.exp(x).sum(-1)), 1e-3, 1e-3),
    red("mean_all_300x300", "torch.randn(300, 300).mean()", _c(lambda x: x.mean())),
    red("row_1x65536", "torch.randn(1, 65536).sum(-1)", _c(lambda x: x.sum(-1)), 1e-3, 1e-3),
    red("dot_8192", "x = torch.randn(8192); y = torch.randn(8192); (x * y).sum()", _c(lambda x, y: (x * y).sum()),
        1e-3, 1e-3),

    # ---- category 3: matmul ----------------------------------------------
    mm("256", 256, 256, 256),
    mm("384x192x96", 384, 192, 96),
    mm("33x65x129", 33, 65, 129),
    mm("640x64x640", 640, 64, 640),
    mm("128x2048x128", 128, 2048, 128),
    mm("1536x16x48", 1536, 16, 48),
    mm("768x768x64", 768, 768, 64),
    mm("1x512x1", 1, 512, 1),
    dict(id="hm_bmm_8x32x64x32", code="torch.randn(8, 32, 64) @ torch.randn(8, 64, 32)",
         ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3),
    dict(id="hm_bmm_shared_w", code="torch.randn(3, 40, 50) @ torch.randn(50, 60)",
         ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3),
    dict(id="hm_mv_500x300", code="torch.randn(500, 300) @ torch.randn(300)",
         ref=_c(lambda a, b: a @ b), rtol=1e-3, atol=1e-3),
    dict(id="hm_plus_one", code="(torch.randn(200, 120) @ torch.randn(120, 90)) + 1.0",
         ref=_c(lambda a, b: a @ b + 1.0), rtol=1e-3, atol=1e-3),

    # ---- category 4: fused -----------------------------------------------
    fused("rms_256x256", "x = torch.randn(256, 256); w = torch.randn(256); "
                         "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
          _c(lambda x, w: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * w)),
    fused("softmax_1x4096", "torch.softmax(torch.randn(1, 4096), dim=-1)", _c(_softmax), atol=1e-5),
    fused("softmax_512x10", "torch.softmax(torch.randn(512, 10), dim=-1)", _c(_softmax), atol=1e-5),
    fused("softmax_3d", "torch.softmax(torch.randn(4, 16, 64), dim=-1)", _c(_softmax), atol=1e-5),
    fused("logsoftmax_48x300", "x = torch.randn(48, 300); x - torch.log(torch.exp(x).sum(-1, keepdim=True))",
          _c(lambda x: x - np.log(np.exp(x).sum(-1, keepdims=True)))),
    fused("l2_64x1024", "x = torch.randn(64, 1024); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
          _c(lambda x: x / np.sqrt((x ** 2).sum(-1, keepdims=True)))),
    fused("absmax_scale", "x = torch.randn(32, 600); x / torch.abs(x).amax(-1, keepdim=True)",
          _c(lambda x: x / np.abs(x).max(-1, keepdims=True))),
    fused("bias_gelu_mm", "x = torch.randn(64, 96); w = torch.randn(96, 160); b = torch.randn(160); "
                          "torch.nn.functional.gelu(x @ w + b)",
          _c(lambda x, w, b: _gelu(x @ w + b)), 1e-3, 1e-3),
    fused("relu_mm_rowsum", "x = torch.randn(64, 128); w = torch.randn(128, 96); torch.relu(x @ w).sum(-1)",
          _c(lambda x, w: np.maximum(x @ w, 0).sum(-1)), 1e-3, 1e-3),
    fused("residual_mm", "x = torch.randn(96, 96); w = torch.randn(96, 96); x + torch.relu(x @ w)",
          _c(lambda x, w: x + np.maximum(x @ w, 0)), 1e-3, 1e-3),
    fused("softmax_mm", "x = torch.randn(64, 128); w = torch.randn(128, 256); torch.softmax(x @ w, dim=-1)",
          _c(lambda x, w: _softmax(x @ w)), 1e-3, 1e-5),
    fused("rows_center_scale", "x = torch.randn(80, 250); (x - x.mean(-1, keepdim=True)) * 2.0",
          _c(lambda x: (x - x.mean(-1, keepdims=True)) * 2.0)),

    # ---- category 5: attention-like --------------------------------------
    attn("72x144x24", 72, 144, 24, 4.898979),
    attn("256x256x64", 256, 256, 64, 8.0),
    attn("40x200x80", 40, 200, 80, 8.944272),
    attn("1x512x64", 1, 512, 64, 8.0),
    attn("128x32x32", 128, 32, 32, 5.656854),
    attn("300x300x16", 300, 300, 16, 4.0),
    sdpa("sdpa_1x1x256x64", (1, 1, 256, 64)),
    sdpa("sdpa_2x2x96x32", (2, 2, 96, 32)),
    sdpa("sdpa_1x8x64x16", (1, 8, 64, 16)),
    dict(id="ha_sdpa_causal_1x2x80x32",
         code="q = torch.randn(1, 2, 80, 32); k = torch.randn(1, 2, 80, 32); v = torch.randn(1, 2, 80, 32); "
              "torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)",
         ref=_c(lambda q, k, v: _causal(q, k, v, np.sqrt(32))), rtol=1e-3, atol=1e-4),
    dict(id="ha_batched_3x48x24", code="q = torch.randn(3, 48, 24); kt = torch.randn(3, 24, 48); v = torch.randn(3, 48, 24); "
                                       "torch.softmax(q @ kt / 4.898979, dim=-1) @ v",
         ref=_c(lambda q, kt, v: _softmax((q @ kt) / 4.898979) @ v), rtol=1e-3, atol=1e-4),

    # ---- float16 (tensor cores where emmy schedules them) ----------------
    dict(id="hm_f16_96", code=f"torch.randn(96, 96, {H}) @ torch.randn(96, 96, {H})",
         ref=_h(lambda a, b: a @ b), rtol=1e-2, atol=1e-2),
    dict(id="hm_f16_160x64x48", code=f"torch.randn(160, 64, {H}) @ torch.randn(64, 48, {H})",
         ref=_h(lambda a, b: a @ b), rtol=1e-2, atol=1e-2),
    dict(id="ha_f16_64x64x32", code=f"q = torch.randn(64, 32, {H}); kt = torch.randn(32, 64, {H}); v = torch.randn(64, 32, {H}); "
                                    "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=_h(lambda q, kt, v: _softmax((q @ kt) / 5.656854) @ v), rtol=2e-2, atol=2e-2),
    dict(id="hp_f16_gelu", code=f"torch.nn.functional.gelu(torch.randn(3000, {H}))", ref=_h(_gelu), rtol=1e-2, atol=1e-2),
]]
