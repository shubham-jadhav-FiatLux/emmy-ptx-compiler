#!/usr/bin/env python3
"""Fit the polynomial kernels of emmy_ptx/mathlib.py and measure their error.

    python tools/fit_poly.py            # fit everything, print coefficients + error report

Each kernel approximates a function on a reduced interval (the argument
reduction in mathlib.py brings every input there). We fit by Lawson's
algorithm, an iteratively reweighted least-squares scheme that converges to
the minimax (equal-ripple) polynomial for the *relative* error of the final
result, then round the coefficients to float32 and evaluate the polynomial the
way the GPU will (Horner steps as fused multiply-adds, each rounded to float32)
on a dense grid, reporting the worst error in float32 ulps against float64.
"""

from __future__ import annotations

import math

import numpy as np

f32 = np.float32


def lawson(basis, target, weight, x, iters=60):
    """Minimax-ish fit: minimise max |weight * (basis @ c - target)| over x."""
    a = basis(x)
    w = np.ones_like(x)
    c = None
    for _ in range(iters):
        sw = np.sqrt(w) * weight
        c, *_ = np.linalg.lstsq(a * sw[:, None], target * sw, rcond=None)
        err = np.abs((a @ c - target) * weight)
        w = w * (err + 1e-300)
        w /= w.sum()
    return c


def fma32(a, b, c):
    """float32 fused multiply-add (exact product, one rounding)."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(f32)


def horner32(coeffs, u):
    """sum(coeffs[i] * u**i) evaluated as the GPU does: c_n, then fma(acc, u, c_i)."""
    acc = np.full_like(u, coeffs[-1], dtype=f32)
    for c in reversed(coeffs[:-1]):
        acc = fma32(acc, u, np.full_like(u, c, dtype=f32))
    return acc


def ulps(got32, true64):
    """|got - true| in units of float32 ulp of the true value."""
    t = np.abs(true64)
    expo = np.floor(np.log2(np.maximum(t, 2.0 ** -126)))
    ulp = 2.0 ** (expo - 23)
    return np.abs(got32.astype(np.float64) - true64) / ulp


def report(name, coeffs, worst):
    print(f"{name}: degree {len(coeffs) - 1}, max error {worst:.3f} ulp")
    print("    (" + ", ".join(f"{float(f32(c))!r}" for c in coeffs) + ")")


def fit_log1p(degree=8):
    """log(1+f) = f + f^2 * P(f) for f in [sqrt(1/2)-1, sqrt(2)-1]."""
    lo, hi = math.sqrt(0.5) - 1, math.sqrt(2) - 1
    x = np.linspace(lo, hi, 20001)
    x = x[np.abs(x) > 1e-6]
    true = np.log1p(x)
    target = (true - x) / x ** 2
    c = lawson(lambda t: np.vander(t, degree + 1, increasing=True), target, x ** 2 / np.abs(true), x)
    c32 = [f32(v) for v in c]
    f = np.linspace(lo, hi, 400001).astype(f32)
    f = f[f != 0]
    p = horner32(c32, f)
    r = fma32((f * f).astype(f32), p, f)
    return c32, float(np.max(ulps(r, np.log1p(f.astype(np.float64)))))


def fit_odd(fn, xmax, degree, name):
    """fn(x) = x + x^3 * P(x^2) on [0, xmax] (tanh, sin)."""
    x = np.linspace(1e-4, xmax, 20001)
    true = fn(x)
    target = (true - x) / x ** 3
    c = lawson(lambda t: np.vander(t * t, degree + 1, increasing=True), target, x ** 3 / np.abs(true), x)
    c32 = [f32(v) for v in c]
    xs = np.linspace(1e-6, xmax, 400001).astype(f32)
    x2 = (xs * xs).astype(f32)
    r = fma32((xs * x2).astype(f32), horner32(c32, x2), xs)
    return c32, float(np.max(ulps(r, fn(xs.astype(np.float64)))))


def fit_scaled_odd(fn, xmax, degree):
    """fn(x) = x + x * P(x^2) on [0, xmax] (erf); the final FMA adds x exactly."""
    x = np.linspace(1e-4, xmax, 20001)
    true = fn(x)
    c = lawson(lambda t: np.vander(t * t, degree + 1, increasing=True), true / x - 1, x / np.abs(true), x)
    c32 = [f32(v) for v in c]
    xs = np.linspace(1e-6, xmax, 400001).astype(f32)
    r = fma32(xs, horner32(c32, (xs * xs).astype(f32)), xs)
    return c32, float(np.max(ulps(r, fn(xs.astype(np.float64)))))


def fit_cos(xmax, degree):
    """cos(r) = 1 + r^2 * P(r^2) on [0, xmax]."""
    x = np.linspace(1e-4, xmax, 20001)
    true = np.cos(x)
    target = (true - 1) / x ** 2
    c = lawson(lambda t: np.vander(t * t, degree + 1, increasing=True), target, x ** 2 / np.abs(true), x)
    c32 = [f32(v) for v in c]
    xs = np.linspace(1e-6, xmax, 400001).astype(f32)
    x2 = (xs * xs).astype(f32)
    r = fma32(x2, horner32(c32, x2), np.ones_like(xs))
    return c32, float(np.max(ulps(r, np.cos(xs.astype(np.float64)))))


def fit_log_erfc(lo, hi, degree):
    """log(erfc(a)) = P(a) on [lo, hi]; erf = 1 - exp(P(a)) there."""
    import scipy.special as sp  # only needed when refitting

    x = np.linspace(lo, hi, 20001)
    true = np.log(sp.erfc(x))
    c = lawson(lambda t: np.vander(t, degree + 1, increasing=True), true, np.ones_like(x), x)
    c32 = [f32(v) for v in c]
    xs = np.linspace(lo, hi, 400001).astype(f32)
    q = horner32(c32, xs).astype(np.float64)
    erf = (1.0 - np.exp(q)).astype(f32)  # exp itself is accurate to ~2 ulp on the GPU
    return c32, float(np.max(ulps(erf, sp.erf(xs.astype(np.float64)))))


if __name__ == "__main__":
    report("log1p   P(f)        log1p(f) = f + f^2 P(f)", *fit_log1p(8))
    report("tanh    P(x^2)      x in [0, 0.6]", *fit_odd(np.tanh, 0.6, 5, "tanh"))
    report("sin     P(r^2)      r in [0, pi/4]", *fit_odd(np.sin, math.pi / 4, 3, "sin"))
    report("cos     P(r^2)      r in [0, pi/4]", *fit_cos(math.pi / 4, 4))
    import scipy.special as sp

    report("erf     P(x^2)      x in [0, 1]", *fit_scaled_odd(sp.erf, 1.0, 6))
    report("logerfc P(a)        a in [1, 4]", *fit_log_erfc(1.0, 4.0, 10))
