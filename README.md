# emmy-ptx-compiler

A compiler from Emmy's Kernel IR (`*.ir.json`) straight to NVIDIA PTX, tuned on
the hackathon's RTX 5090 runner.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.12+, emmy-ml 0.3.11, numpy
python compile.py path/to/<kernel>.ir.json > out.ptx            # PTX on stdout, diagnostics on stderr
python compile.py --strict path/to/<kernel>.ir.json > out.ptx   # no rewrite that changes rounding
python compile.py --plain path/to/<kernel>.ir.json > out.ptx    # the fallback lowering alone
python compile.py --report path/to/<kernel>.ir.json > out.ptx   # what the compiler decided, on stderr
```

`meta.json`, next to the IR, supplies the parameter order and the launch. With
the organisers' checker: `python submit.py <problem> --compiler compile.py`.
The IR is loaded with emmy's `Graph.from_dict`. Everything after that is this
project's own code: the lowering of every statement, the index arithmetic, the
math library and the PTX text. [`examples/`](examples/README.md) walks through
the PTX of a reduction, an attention kernel, a tensor-core matmul and a fused
MLP that runs in thread-block clusters.

## What to look for

**Versatility**
* Every Kernel-IR statement emmy emits for this GPU:
  * structure: `Tile` (aux bands, CTA rasterization), `Loop`/`StridedLoop`
    (runtime bounds), `Cond`;
  * values: `Let`, `Assign`, `Accum`, `Init`, `Select`;
  * memory: `Load`/`Write` (vector, atomic), `Smem` (static arrays or a
    dynamic pool), `Sync` (named barriers);
  * the `cp.async` copy engine (4/8/16 bytes, swizzled, lane-indexed,
    double-buffered), `ZeroPrologue`;
  * the bulk copy engine (`--bulk-copy`): fills that copy contiguous runs go
    to the TMA unit as `cp.async.bulk`, completing on mbarriers. It is correct
    on all 68 problems that fill shared memory, and off by default because
    it was never faster here (DESIGN 3);
  * cross-lane: `WarpShuffle`, `TreeHalve`;
  * tensor cores: the eight statements of emmy's `mma.sync` path (DESIGN 6).
* Data types:
  * f32;
  * f16, on the tensor cores and elsewhere (each operation computed in f32
    and rounded once to f16);
  * f64 in true double precision, with double exp, log and sigmoid;
  * 8- to 64-bit integers with C semantics, and bool;
  * fp8 (e4m3 and e5m2): loads, stores and emmy's encode and decode casts,
    with sm_89's `cvt.rn.satfinite`, bit for bit as emmy defines them.
* Every elementwise op emmy's CUDA renderer can emit, including the integer
  bit operations (fp4 conversions aside).
* Multi-kernel problems: split-K partials, and chains passing intermediates.

**Performance, from the GPU's structure**
* A per-lane uniformity analysis finds the work that neighbouring lanes
  repeat, and shares it among them (DESIGN 4):
  * reductions split across lanes, choosing the loop by a coalescing cost
    model;
  * attention blocked across lanes, the FlashAttention-2 work split derived
    from the IR;
  * idle threads joining cells;
  * cells moved where lanes can share them;
  * block reductions with one barrier;
  * work that neighbouring blocks repeat, shared through thread-block
    clusters and distributed shared memory (DESIGN 4.8);
  * SIMT matmuls with two cells a thread, so a step of k loads B once for
    both (m01 15.9 → 12.9 µs, bit for bit the same; DESIGN 4.9).
* Memory:
  * warp loads counted in 128-byte lines;
  * shared-memory broadcasts instead of shuffles (17x cheaper, measured);
  * 128-bit vector accesses, and `ld.global.nc` for read-only data;
  * swizzled `ldmatrix` slabs.
* Registers: unrolling bounded by the per-thread register budget, and launch
  hints (`.maxntid`, `.minnctapersm`) for one-wave grids.
* Result: 0.51x the time of nvcc's build of emmy's own CUDA over 321
  problems, attention at 0.16x and fused MLPs down to 0.005x.

**Design**
* One rule per statement. Each loop-body rewrite is either exact, bit for
  bit, or a regrouping of an associative fold.
* A builder that knows the range of every integer, value-numbers, and hoists
  invariant work.
* `--strict` keeps only the rewrites that leave every bit of every result as
  the IR's own order of operations gives it; checked on 371 problems.
* `--plain`: if an optimization ever fails, the kernel is lowered again
  without the optional rewrites. That fallback is tested on every problem.
* `--report` shows what the compiler did to any kernel: the shape it gave
  each loop (split across lanes, lane-blocked, unrolled...), the cells it
  moved, the trees it folded, and a profile of the PTX.

**Novelty**
* The merge of two online-softmax states derived from the IR's own update,
  not hard-coded.
* Lane blocking and cell placement, which re-map emmy's schedules onto
  warps.
* Thread-block clusters (Hopper and Blackwell) for work emmy repeats across
  blocks: a fused MLP 2.2x faster, bit for bit the same.
* Congruent-accumulator merging.
* One `exp` for both factors of an online-softmax step.
* Accurate math routines: minimax polynomials with exact argument
  reductions, measured at ≤ 1 to 3 ulp over about a million inputs per range.
* Division by a constant correctly rounded with one double multiply: emmy's
  `x * reciprocal(3.0)` can be an ulp off numpy's `x / 3.0`, which nvcc's
  build fails at `rtol = 0` (DESIGN 7).

**Testing**
* 209 tests: unit, compile and GPU tests, each kernel lowered three ways,
  plus a lane-level interpreter for lane blocking.
* 473 problems, generated with the organisers' tool: the open set, a wider
  corpus of 205 (20 of them f16), a schedule sweep of 106, and two held-out
  sets (81, then 66) written after the compiler.
* Tools for an nvcc race (`tools/ref_bench.py`), math accuracy
  (`tools/mathtest.py`), and bit-for-bit equality of two builds
  (`tools/bitwise.py`).

## Results

Measured on the evaluation runner (RTX 5090, sm_120), with each problem's own
launch from `meta.json`:

| | |
|---|---|
| Open problem set | **15 / 15** pass (all five categories); the four pointwise problems are bit-exact |
| Wider corpus (`tools/corpus_specs.py`, generated with the organisers' tool) | **205 / 205** pass |
| Tensor cores (float16: emmy's `mma.sync` matmul, flash-attention and SDPA schedules) | **20 / 20** of the corpus's f16 problems pass; 0.77x the time of nvcc's build (geometric mean of 19); one is slower, a batched 4x64x96 matmul at 6.6 µs against 4.6 (DESIGN 8) |
| Schedule sweep (`tools/corpus_knobs.py`: 106 EMMY_KNOBS variants, incl. split-K over two kernels) | **106 / 106** pass |
| Held-out set (`tools/corpus_holdout.py`: 81 new problems written after the compiler, to estimate the hidden set) | **81 / 81** pass on first run; worst error 10.5% of the tolerance |
| Second held-out set (`tools/corpus_holdout2.py`: 66 problems written on 26 Oct, after every optimization) | **64 / 66** pass. The other two fail for any compiler, nvcc's build included: the generator lists a consumer kernel before its producer, and emmy traces `A @ v` into a 500x300 output. The set found one bug: emmy's `x * reciprocal(3.0)` is an ulp off numpy's `x / 3.0` at `rtol = 0`, so it is now computed as the correctly rounded quotient (nvcc's build still fails it; `--strict` and `--plain` keep emmy's multiply, 62 / 66) |
| Fallback lowering (`compile.py --plain`: what runs if an optimization fails) | **469 / 473** pass: all but the two generator faults and the two `rtol = 0` divisions, where it keeps emmy's multiply by the reciprocal |
| Strict mode (`compile.py --strict`: no rewrite that changes rounding) | **469 / 473** pass, the same four aside. Checked bit for bit on 402 problems: the 371 without float atomics give outputs identical to the plain lowering's (so every rewrite it keeps is exact) |
| Accuracy against numpy's answer (the check the evaluation runs) | worst error **14% of the tolerance** (median 0.13%); numpy's own float32 has median 0.12%, nvcc's build 0.18% and fails 2 bit-exact problems |
| Fused MLPs (`act(x @ w1) @ w2` in one kernel, which emmy's schedules recompute the hidden layer for) | the five of corpus round 6 (gelu, relu, tanh, silu) at **0.005x to 0.08x** the time of nvcc's build (13x to 187x faster, through lane blocking); `k_gelu_mlp` 389 → 173 µs through thread-block clusters (nvcc: 408 µs) |
| Speed against nvcc's build of emmy's reference CUDA (321 problems, same launches) | **0.51x the time** (geometric mean): chained 0.13x, attention 0.16x, fused 0.53x, reductions 0.65x, matmul 0.88x, pointwise 1.00x (launch-bound); total 4.0 ms against 158.2 ms |
| Math library accuracy (against correctly rounded results) | exp, tanh ≤ 2 ulp · log, erf, sin, cos ≤ 1 ulp (sin/cos through 1e9) · float64 exp, log ≤ 1 ulp, sigmoid ≤ 2 ulp |
| Tests | 209 (87 run hand-built IR on the GPU); compile and GPU tests run each kernel optimized, strict and plain |
| Compile time | under 2 s per kernel (mostly importing emmy) |

The same PTX compiles byte for byte under Python 3.12 and 3.14.

## How it works

```
*.ir.json + meta.json ─► frontend ─► lowering ──────────► builder ─► PTX text ─► ptxas
                                     one rule per IR      folding · ranges · CSE · LICM
                                     statement, plus      affine addresses · vector loads
                                     loop-body rewrites
```

* **One lowering rule per IR statement** (`emmy_ptx/lower.py`, with the cell
  mapping in `cells.py` and lane sharing in `lanes.py`), with names in
  C block scoping, exactly as in the CUDA C++ emmy would print.
* **A builder that knows what every integer can be** (`emmy_ptx/builder.py`):
  proven ranges turn divisions into shifts and remove guards the launch
  already guarantees; constants fold; pure instructions are value-numbered
  and loop-invariant ones hoisted; addresses are affine, so `x[4*a0 + k]` is
  one register plus immediates, and adjacent loads become vector loads.
* **Lanes that compute the same thing share it.** A uniformity analysis knows,
  for every value, how many neighbouring lanes hold it alike. Emmy's schedules
  often give many lanes the same work; the compiler splits it among them:
  * *reductions* that a group of lanes would each compute whole are split
    across the group and merged with a butterfly, choosing which loop to split
    by how its loads coalesce (attention dot products, softmax of a matmul);
  * *lane blocking* for attention: the lanes holding a query row take turns
    scoring keys (a whole dot product each), update the online softmax once per
    block of keys, and pass the weights to each other through shared memory,
    the work split of FlashAttention-2, derived from the IR;
  * *idle threads*: when a launch has far more threads than cells (emmy
    schedules that fold a whole row in one thread), each cell gets a group of
    lanes, up to a whole block, and its reductions and loops are split among
    them;
  * *cell placement*: when emmy's layout gives a row's cells no aligned
    group of lanes, or only small ones (11 or 14 cells of three columns to a
    row, cells rasterized lane by lane), the compiler moves the cells. Each
    row's cells go onto aligned lane groups of power-of-two sizes (11 = 8 + 4,
    with one lane that computes but stores nothing), so blocking shares the
    row's scores again (up to 4.9x faster on such kernels);
  * *block reductions with one barrier*: every thread folds the warps'
    partials itself, in the order emmy's halving tree would, instead of a
    barrier per tree step;
  * *blocks that compute the same thing* (`emmy_ptx/cluster.py`): where every
    block recomputes work that does not depend on its own output column (a
    fused MLP's hidden layer), the kernel runs in clusters of 8 blocks. Each
    block computes an eighth of that work and reads the rest from its
    neighbours' shared memory (`mapa`, `ld.shared::cluster`).
* **Loop-body rewrites** (`emmy_ptx/mapping.py`, `congruence.py`): provably
  equal accumulators are merged, repeated arithmetic removed, independent
  sibling loops fused, and the online softmax's merge of two partial states is
  derived from the IR's own update.
* **Tensor cores** (`emmy_ptx/tensorcore.py`). Emmy's f16 matmul and attention
  schedules hold their tiles in per-lane register fragments. These are filled
  by `ldmatrix` from swizzled shared memory or straight from global memory,
  multiplied with `mma.sync.m16n8k16`, put through the softmax's row
  reductions and repacked into the next mma's operand, and stored by fused
  epilogues. The compiler lowers each statement to the PTX instruction it
  names, unrolling every loop that indexes a fragment.
* **A fallback.** If the optimizing lowering fails on an IR shape it did not
  foresee, or its module is longer than the runner accepts, the kernel is
  lowered again plainly: one rule per statement, without the optional
  rewrites. That lowering is slower but passes every problem above too.
* **Numerics.** Elementwise arithmetic is IEEE with explicit round-to-nearest,
  in the precision of each result (an f64 result in double, which emmy's own
  CUDA does not do), so pointwise kernels reproduce numpy bit for bit; a
  division by a constant, which emmy spells as a multiply by its reciprocal,
  is computed as the correctly rounded quotient. fp8 values are byte codes
  that only emmy's encode and decode casts touch.
  `exp`, `log`, `tanh`, `erf`, `sin`, `cos` and `pow` are accurate routines
  built from minimax polynomials and exact argument reductions (`exp`, `log`
  and `sigmoid` also in double precision). Splitting a
  reduction regroups its sum (well within the tolerances); folds that are not
  split keep the IR's order. `--strict` turns off every rewrite that changes
  rounding (lane splits, softmax updates per block, FMA contraction, the
  quotient in place of emmy's multiply by a reciprocal).

[docs/DESIGN.md](docs/DESIGN.md) explains each part, with the measurements behind
the choices.

## Layout

```
compile.py             command-line entry point (PTX to stdout, errors to stderr)
emmy_ptx/
  frontend.py          IR + meta.json -> KernelSource
  lower.py             the Lowerer: driver, names, expressions, addresses, loops, statements
  cells.py             where a Tile's cells run: decode, rasterization, spread cells, placement
  lanes.py             lanes sharing a loop's work: split reductions, lane blocking, dealt loops
  mapping.py           loop-body analyses and rewrites (fusion, online softmax, lane blocking)
  congruence.py        accumulators that always hold the same value
  regroup.py           cooperative reductions in contiguous chunks
  tensorcore.py        tensor-core fragments: ldmatrix, mma.sync, fragment softmax, fragment stores
  common.py            what those share: dtype tables, buffers, affine forms, scopes
  builder.py           typed values, folding, ranges, value numbering, LICM, DCE
  vectorize.py         adjacent scalar loads -> ld.v2 / ld.v4
  elementwise.py       Assign/Accum operations and emmy's dtype rules
  mathlib.py           exp, log, tanh, erf, sin, cos, pow, sigmoid (exp, log, sigmoid also in float64)
  ptx.py               registers, operands, instructions, module text
  report.py            compile.py --report: the lowering's decisions and a PTX profile
  bulk.py              compile.py --bulk-copy: which cp.async fills the copy engine can take
  cluster.py           work neighbouring blocks repeat, shared across a thread-block cluster
  coarsen.py           two matmul cells a thread, sharing their shared-memory loads
tests/                 unit tests (builder, rewrites, cell placement, strict mode, a lane-level IR
                       interpreter) and GPU tests
tools/                 corpus generation, GPU suite runner, nvcc race, tuning, math accuracy
examples/              four problems with their PTX, walked through in examples/README.md
docs/DESIGN.md
requirements.txt
```

## Testing

```bash
# unit + compile tests; set EMMY_RUNNER=http://runner.riftstack.ai:48701 to also run the GPU tests
python -m pytest tests

# compile, run and check problem folders on the runner (margin = share of the tolerance used);
# --strict, --plain and --bulk-copy are passed on to compile.py
python tools/run_suite.py <problems-dir> [--bench] [--strict] [--plain] [--bulk-copy]

# build the wider corpus with the organisers' generator (needs torch)
python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --out corpus
python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --specs tools/corpus_knobs.py --out corpus_knobs
python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --specs tools/corpus_holdout2.py --out holdout2

# race against nvcc's build of the reference CUDA (development only)
python tools/ref_bench.py <problems-dir> --nvcc <nvcc>

# do two builds give bit-identical outputs? (default: --strict against --plain --strict)
python tools/bitwise.py <problems-dir>

# accuracy of the math library on the GPU
python tools/mathtest.py
```
