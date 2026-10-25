# emmy-ptx-compiler

A compiler from Emmy's Kernel IR (`*.ir.json`) straight to NVIDIA PTX.

```
python compile.py path/to/<kernel>.ir.json > out.ptx
```

The IR is loaded with emmy's `Graph.from_dict`. Everything after that (the
lowering of every statement, the index arithmetic, the math library and the
PTX text) is this project's own code. `meta.json`, next to the IR, supplies
the parameter order and the launch geometry. Requirements: Python 3.12+,
`emmy-ml==0.3.11`, numpy.

## Results

Measured on the evaluation runner (RTX 5090, sm_120), with each problem's own
launch from `meta.json`:

| | |
|---|---|
| Open problem set | **15 / 15** pass (all five categories); the four pointwise problems are bit-exact |
| Wider corpus (`tools/corpus_specs.py`, generated with the organisers' tool) | **162 / 162** pass |
| Schedule sweep (`tools/corpus_knobs.py`: 106 EMMY_KNOBS variants, incl. split-K over two kernels) | **106 / 106** pass |
| Held-out set (`tools/corpus_holdout.py`: 81 new problems written after the compiler, to estimate the hidden set) | **81 / 81** pass on first run; worst error 10.5% of the tolerance |
| Accuracy against numpy's answer (the check the evaluation runs) | worst error **14% of the tolerance** (median 0.13%); numpy's own float32 has median 0.12%, nvcc's build 0.18% and fails 2 bit-exact problems |
| Speed against nvcc's build of emmy's reference CUDA (281 problems, same launches) | **0.56x the time** (geometric mean): attention 0.12x, chained 0.28x, fused 0.52x, reductions 0.65x, matmul 0.87x, pointwise 1.00x (launch-bound); total 3.3 ms against 78.1 ms |
| Math library accuracy (against correctly rounded float64) | exp, tanh ≤ 2 ulp · log, erf, sin, cos ≤ 1 ulp (sin/cos through 1e9) |
| Tests | 80 (17 run hand-built IR on the GPU) |
| Compile time | under 2 s per kernel (mostly importing emmy) |

The same PTX compiles byte for byte under Python 3.12 and 3.14.

## How it works

```
*.ir.json + meta.json ─► frontend ─► lowering ──────────► builder ─► PTX text ─► ptxas
                                     one rule per IR      folding · ranges · CSE · LICM
                                     statement, plus      affine addresses · vector loads
                                     loop-body rewrites
```

* **One lowering rule per IR statement** (`emmy_ptx/lower.py`), with names in
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
  * *block reductions with one barrier*: every thread folds the warps'
    partials itself, in the order emmy's halving tree would, instead of a
    barrier per tree step.
* **Loop-body rewrites** (`emmy_ptx/mapping.py`, `congruence.py`): provably
  equal accumulators are merged, repeated arithmetic removed, independent
  sibling loops fused, and the online softmax's merge of two partial states is
  derived from the IR's own update.
* **Numerics.** Elementwise arithmetic is IEEE single with explicit
  round-to-nearest, so pointwise kernels reproduce numpy bit for bit.
  `exp`, `log`, `tanh`, `erf`, `sin`, `cos` and `pow` are accurate routines
  built from minimax polynomials and exact argument reductions. Splitting a
  reduction regroups its sum (well within the tolerances); folds that are not
  split keep the IR's order.

[docs/DESIGN.md](docs/DESIGN.md) explains each part, with the measurements behind
the choices.

## Layout

```
compile.py             command-line entry point (PTX to stdout, errors to stderr)
emmy_ptx/
  frontend.py          IR + meta.json -> KernelSource
  lower.py             Kernel IR statements -> instructions; how each loop is lowered
  mapping.py           loop-body analyses and rewrites (fusion, online softmax, lane blocking)
  congruence.py        accumulators that always hold the same value
  regroup.py           cooperative reductions in contiguous chunks
  builder.py           typed values, folding, ranges, value numbering, LICM, DCE
  vectorize.py         adjacent scalar loads -> ld.v2 / ld.v4
  elementwise.py       Assign/Accum operations and emmy's dtype rules
  mathlib.py           exp, log, tanh, erf, sin, cos, pow, sigmoid
  ptx.py               registers, operands, instructions, module text
tests/                 unit tests (builder, rewrites, a lane-level IR interpreter) and GPU tests
tools/                 corpus generation, GPU suite runner, nvcc race, tuning, math accuracy
docs/DESIGN.md
```

## Testing

```bash
# unit + compile tests; set EMMY_RUNNER=http://runner.riftstack.ai:48701 to also run the GPU tests
python -m pytest tests

# compile, run and check problem folders on the runner (margin = share of the tolerance used)
python tools/run_suite.py <problems-dir> [--bench]

# build the wider corpus with the organisers' generator (needs torch)
python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --out corpus
python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --specs tools/corpus_knobs.py --out corpus_knobs

# race against nvcc's build of the reference CUDA (development only)
python tools/ref_bench.py <problems-dir> --nvcc <nvcc>

# accuracy of the math library on the GPU
python tools/mathtest.py
```
