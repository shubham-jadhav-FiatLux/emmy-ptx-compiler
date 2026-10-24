# Design

This document explains how `emmy_ptx` turns one Emmy Kernel-IR file into a
PTX module, and why each design decision was made.

```
*.ir.json + meta.json
      │  frontend.load          Graph.from_dict, buffer shapes/dtypes, launch geometry
      ▼
 KernelSource
      │  lower.Lowerer          one rule per IR statement, C block scoping
      │      └─ builder.Builder folding · ranges · value numbering · LICM
      │      └─ elementwise / mathlib
      ▼
 ptx.Kernel (instructions over virtual registers)
      │  dead-code elimination, Kernel.render
      ▼
 PTX text  ──► ptxas (runner) ──► SASS
```

## 1. Front end (`frontend.py`)

The IR is deserialized with emmy's `Graph.from_dict`, the only part of emmy we
use. The graph holds one `KernelOp` plus the `InputOp`/`ConstantOp` nodes that
feed it; every node's `outputs` give the shape and dtype of a buffer.
`meta.json` beside the IR fixes what the IR cannot: the parameter order
(`arg_order`; it differs from the IR's input order on some problems) and the
launch (grid, block, dynamic shared memory). If `meta.json` is missing, the
order and launch are derived the way emmy's CUDA backend derives them.

## 2. Values, registers and the builder (`builder.py`, `ptx.py`)

PTX registers are virtual and unlimited, so we never allocate registers:
ptxas does. Every computed quantity is a `Value`: a register or an immediate,
a type (`s32`, `s64`, `f32`, `f16`, `f64`, `pred`), and for integers a
*proven range* `[lo, hi]`.

The lowering never spells arithmetic opcodes itself; it asks the `Builder`,
which applies five transformations as it emits:

| Transformation | Example | Why it matters |
|---|---|---|
| Constant folding | `a0*4 + 1` with `a0 = 3` → `13` | unrolled loops become constants |
| Range analysis | `%tid.x ∈ [0, block-1]` from the launch; `gid / 128` on `[0, 1023]` → `shr`; `x % 512` with `x < 512` → `x`; `i < 1024` with `i ≤ 1023` → `true` | no slow division, no needless guard, no dead branch |
| Strength reduction | `x * 8` → `shl`, `x % 64` → `and` (only when `x ≥ 0`: C semantics) | 1-cycle ops instead of multi-instruction sequences |
| Value numbering (CSE) | the same address for `x[4*a0+k]`, `k = 0..3` | one address computation per run of accesses |
| Loop-invariant code motion | `q[row, k]` loaded once before the key loop of attention | per-trip work shrinks |

Integer semantics are C's (`/` truncates, `%` keeps the dividend's sign),
because emmy's index expressions *are* C expressions. When a range could
overflow 32 bits the arithmetic is widened to 64 bits automatically.

**Scoping of value numbering.** A table of available values is pushed for
every region that does not dominate what follows it (a branch body, a loop
body) and popped at its end, so a value is reused only where its definition
is guaranteed to have executed. Reassigned registers (accumulators, loop
counters) are never numbered, except a loop counter inside one trip of its
own body, where it is constant.

**Loop-invariant code motion.** A rolled loop opens a *preheader* placed after
its zero-trip check. A pure instruction whose operands were all defined before
the loop is emitted there instead of in the body. Loads move only when they
read memory the kernel never writes and no branch inside the loop guards them,
so a hoisted load never executes where the original would not have. Hoisted
values live in a numbering table that closes with the loop: the preheader
dominates the loop body but not the code after it (the zero-trip check can
skip it). A per-loop budget bounds the extra register pressure.

**Dead-code elimination** runs once at the end: any pure instruction whose
result is never read is deleted, repeated to a fixpoint.

## 3. Lowering (`lower.py`)

`Lowerer.body` dispatches each statement to `stmt_<Kind>`. Names follow C
block scoping exactly as the CUDA C++ emmy would print: a `Tile`, loop or
branch body is a block; declarations shadow; a name declared before a loop
and reassigned inside it (an `Accum`, a combined state) is one mutable
register. A pre-pass finds every name that is ever reassigned, so its
declaration allocates a register up front.

| Statement | Lowering |
|---|---|
| `Tile` | `gid = ctaid·block + tid` (or the aux-band form), guard `gid < N` unless the range proves it, row-major decode of the axes, optional grouped CTA rasterization |
| `Loop`, `StridedLoop` | seed the immediate body's accumulators when `seed`; full unroll when the trip count is bounded and small, else a rolled loop running 4-8 iterations per trip plus the leftover; `<v>_end` bound hoisted |
| `Cond` | predicated branch, or no branch at all when the predicate is decided by ranges |
| `Load` / `Write` | affine address, `ld/st[.v2/.v4]`, `ld.global.nc` for read-only buffers, `red.global.add` for atomic writes |
| `Assign`, `Accum`, `Init`, `Let`, `Select` | elementwise semantics below; an `Accum` add whose base or value is a product becomes one FMA |
| `Smem` | static `.shared` arrays up to 48 KiB in total, else one `.extern .shared` pool packed exactly like emmy's `pack_smem` |
| `Sync` | `bar.sync 0`, `bar.sync id, count`, `bar.warp.sync -1` |
| `WarpShuffle` | `activemask` + `shfl.sync.bfly` butterfly, the combine program run per step |
| `TreeHalve` | unrolled shared-memory halving tree with barriers, then the broadcast |
| `CpAsyncCopy/Commit/Wait` | `cp.async.{cg,ca}.shared.global`, `commit_group`, `wait_group N` |
| `ZeroPrologue` | CTA 0 zeroes the delegated accumulator |

**Affine addresses.** An index expression is split into
`Σ coeff·atom + const`; the register part is materialized once and the
constant goes into the instruction's `[base+offset]` immediate. Global
addresses widen the register part with `mul.wide.u32` (`.s32` when it may be
negative) and add the parameter base; shared addresses are 32-bit.

**Loops.** Iterations always run in their original order, so accumulations
round exactly as the IR specifies; unrolling only lets several iterations'
loads be in flight. The trip-count range comes from the ranges of `start` and
`end`: the reduction loop `for a0 = tid; a0 < 1024; a0 += 128` is exactly 8
trips for every thread, so it becomes 8 straight-line iterations with no
compare at all.

## 4. Numerics (`elementwise.py`, `mathlib.py`)

* **Basic arithmetic is IEEE single, round to nearest** (`add.rn`, `mul.rn`,
  `div.rn`, `sqrt.rn`, `rcp.rn`). Written with `.rn`, ptxas may not contract a
  multiply and an add into an FMA, so pointwise kernels match numpy's float32
  results bit for bit (several problems demand `rtol = atol = 0`; nvcc's own
  build of the reference CUDA fails some of them because it contracts).
* **Reductions contract on purpose**: `acc += a*b` becomes `fma`, as nvcc
  does: faster and more accurate, and those problems carry a tolerance.
* **Transcendentals** are built from an exact or near-exact argument
  reduction plus either a polynomial or a hardware approximation used only on
  a small interval:

  | function | method | measured on the GPU |
  |---|---|---|
  | `exp` | `x·log2 e = j + f` with a two-FMA split of `log2 e`; `ex2.approx` on `|f| ≤ ½`; `2^j` in the exponent field | ≤ 2 ulp |
  | `log` | `x = m·2^e`, `m ∈ [√½, √2)`; `log1p(m-1)` polynomial | ≤ 1 ulp, subnormals included |
  | `tanh` | odd polynomial below 0.6, `1 - 2/(e^{2|x|}+1)` above | ≤ 2 ulp |
  | `erf` | odd polynomial below 1, `1 - exp(P(|x|))` with `P ≈ log erfc` above | ≤ 1 ulp |
  | `sin`, `cos` | three-constant Cody-Waite reduction by π/2 with FMAs; float64 reduction beyond 105615 | ≤ 1 ulp up to 1e9 |
  | `pow` | `2^(y·log2|x|)` in float64 (atanh series, Taylor 2^t) plus the C99 special cases | correctly rounded in practice |
  | `sigmoid` | emmy's spelling `1/(1+exp(-x))` | ≤ 3 ulp |

  The polynomial coefficients are minimax fits produced by
  `tools/fit_poly.py`, which simulates the GPU's FMA evaluation to report each
  kernel's worst ulp error; `tools/mathtest.py` measures the finished routines
  on the GPU over about a million inputs per range.

## 5. Launch hints

`.maxntid` carries the block size (like CUDA's `__launch_bounds__`), so ptxas
never uses more registers than a block can launch with. When the whole grid
fits in one wave (at most one CTA per SM on the 170-SM RTX 5090) the module
also says `.minnctapersm 1`: extra registers then cannot cost occupancy, and
ptxas uses them to keep more loads in flight, which makes the latency-bound
attention kernels 25-40% faster.

## 6. Validation

* `tools/run_suite.py` compiles problem folders exactly as the evaluator does
  and checks them on the runner, reporting how much of the tolerance each
  result used and its worst ulp error.
* `tools/corpus_specs.py` + `tools/make_corpus.py` regenerate a wider problem
  set with the organisers' own generator: odd sizes, broadcasting, every
  traceable elementwise op, max/min reductions, layernorm, chained kernels,
  odd and batched matmuls, attention at other lengths and through
  `scaled_dot_product_attention` (causal and multi-head).
* `tools/ref_bench.py` races our PTX against nvcc's build of emmy's reference
  CUDA on identical launches.
* `tests/` holds unit tests of the builder and end-to-end compile checks
  (entry name, parameter order, ptxas acceptance).

## 7. Not supported

The statements TASK.md marks as outside the scope (tensor-core fragments,
TMA/mbarrier, `SetMaxNReg`, `Carry`/`Pre` recurrences) raise a clear error.
Emmy only produces them for tensor-core schedules (for example half-precision
matmul) and warp-specialized pipelines.
