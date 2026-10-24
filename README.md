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

Measured on the evaluation runner (RTX 5090, sm_120):

| | |
|---|---|
| Open problem set | **15 / 15** pass (all five categories); the four pointwise problems are bit-exact |
| Wider corpus (`tools/corpus_specs.py`, generated with the organisers' tool) | **145 / 145** pass |
| Schedule sweep (`tools/corpus_knobs.py`: 106 EMMY_KNOBS variants incl. split-K over two kernels) | **106 / 106** pass |
| Hand-built IR for rarely generated statements (`tests/test_gpu_lowering.py`) | 7 / 7 pass |
| Speed vs nvcc's build of emmy's reference CUDA, same launches (158 problems) | **0.96x the time** (geometric mean): attention 0.77x, reductions 0.92x, matmul 0.97x, pointwise 1.08x |
| Math library accuracy (vs correctly rounded float64) | exp, tanh ≤ 2 ulp · log, erf, sin, cos ≤ 1 ulp (sin/cos through 1e9) |
| Compile time | < 1 s per kernel (mostly importing emmy); largest PTX 200 KB |

The same PTX compiles byte for byte under Python 3.12 and 3.14.

## How it works

```
*.ir.json + meta.json ─► frontend ─► lowering ──► builder ─► PTX text ─► ptxas
                          load the     one rule     folding · ranges · CSE
                          graph        per IR       · loop-invariant code motion
                                       statement    · affine addresses
```

* **One lowering rule per IR statement** (`emmy_ptx/lower.py`): `Tile`,
  `Loop`/`StridedLoop`, `Cond`, `Load`/`Write` (vector, atomic, shared),
  `Assign`/`Accum`/`Init`/`Let`/`Select`, `Smem` (static arrays or one dynamic
  pool), `Sync`, `WarpShuffle`, `TreeHalve`, the `cp.async` family and
  `ZeroPrologue`. Names follow C block scoping, exactly as in the CUDA C++
  emmy would print for the same IR.
* **A builder that knows what every integer can be** (`emmy_ptx/builder.py`).
  Thread and block indices get their ranges from the launch, and every
  integer value carries a proven `[lo, hi]`. Division by a power of two
  becomes a shift, `x % n` disappears when `x < n`, and guards and comparisons
  that the ranges decide are removed. Constants are folded, pure instructions
  are value-numbered within dominance scopes, and loop-invariant work moves
  to loop preheaders.
* **Addresses in affine form.** `x[4*a0 + k]` for `k = 0..3` is one address
  register plus `[reg+4k]` immediates, and `ld/st.v2/.v4` are emitted where
  emmy vectorized.
* **Loops** whose trip counts the ranges bound are unrolled completely; long
  loops run 4-8 iterations per trip, always in their original order, so sums
  round exactly as written while several loads are in flight.
* **Numerics.** Elementwise arithmetic is IEEE single with explicit
  round-to-nearest (`add.rn`, `mul.rn`, `div.rn`, `sqrt.rn`), so pointwise
  kernels reproduce numpy bit for bit. Reductions fuse `acc += a*b` into FMA.
  `exp`, `log`, `tanh`, `erf`, `sin`, `cos` and `pow` are accurate routines
  built from minimax polynomials (`tools/fit_poly.py`) and exact argument
  reductions.

[docs/DESIGN.md](docs/DESIGN.md) explains each part in detail.

## Layout

```
compile.py             command-line entry point (PTX to stdout, errors to stderr)
emmy_ptx/
  frontend.py          IR + meta.json -> KernelSource
  lower.py             Kernel IR statements -> instructions
  builder.py           typed values, folding, ranges, value numbering, LICM, DCE
  elementwise.py       Assign/Accum operations and emmy's dtype rules
  mathlib.py           exp, log, tanh, erf, sin, cos, pow, sigmoid
  ptx.py               registers, operands, instructions, module text
tests/                 builder unit tests, end-to-end compile checks (pytest)
tools/                 corpus generation, GPU suite runner, nvcc race, math accuracy
docs/DESIGN.md
```

## Testing

```bash
# unit + compile tests (set EMMY_PROBLEMS to problem folders; PTXAS to assemble too)
python -m pytest tests

# compile, run and check problem folders on the runner
python tools/run_suite.py <problems-dir> [--bench]

# build the wider corpus with the organisers' generator (needs torch)
python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --out corpus
python tools/make_corpus.py --hackathon <emmy-hackathon-2026> --specs tools/corpus_knobs.py --out corpus_knobs

# race against nvcc's build of the reference CUDA (development only)
python tools/ref_bench.py <problems-dir> --nvcc <nvcc>

# accuracy of the math library on the GPU
python tools/mathtest.py
```
