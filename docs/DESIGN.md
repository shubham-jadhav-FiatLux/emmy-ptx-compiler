# Design

This document explains how `emmy_ptx` turns one Emmy Kernel-IR file into a
PTX module, and why each design decision was made. Measurements are on the
evaluation runner (RTX 5090), each problem with its own launch.

```
*.ir.json + meta.json
      │  frontend.load          Graph.from_dict, buffer shapes/dtypes, launch geometry
      ▼
 KernelSource
      │  lower.Lowerer          one rule per IR statement, C block scoping;
      │                         per loop: split across lanes, lane-blocked, dealt, unrolled or rolled
      │      ├─ mapping / congruence / regroup   loop-body analyses and IR-to-IR rewrites
      │      ├─ builder.Builder                  folding · ranges · uniformity · value numbering · LICM
      │      └─ elementwise / mathlib
      ▼
 ptx.Kernel (instructions over virtual registers)
      │  dead-code elimination, vector-load merging, Kernel.render
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
A `ConstantOp` is a one-element buffer whose value the IR already knows, so a
scalar load of it becomes that constant (and, for example, `pow(x, 2.0)`
becomes `x*x`).

## 2. Values, registers and the builder (`builder.py`, `ptx.py`, `vectorize.py`)

PTX registers are virtual and unlimited, so we never allocate registers:
ptxas does. Every computed quantity is a `Value`: a register or an immediate,
a type (`s32`, `s64`, `f32`, `f16`, `f64`, `pred`), for integers a *proven
range* `[lo, hi]`, and a *uniform group* (section 4).

The lowering never spells arithmetic opcodes itself; it asks the `Builder`,
which applies these transformations as it emits:

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
is guaranteed to have executed. A read of a reassigned register (an
accumulator) is numbered by the write it sees: two reads with no write
between them are the same value, which lets `alpha * l` fuse into `l`'s own
update as an FMA.

**Loop-invariant code motion.** A rolled loop opens a *preheader* placed after
its zero-trip check. A pure instruction whose operands were all defined before
the loop is emitted there instead of in the body. Loads move only when they
read memory the kernel never writes and no branch inside the loop guards them,
so a hoisted load never executes where the original would not have. A per-loop
budget bounds the extra register pressure, and a loop may refuse loads
altogether (lane blocking, section 4.3, where a lane would otherwise hold whole
query rows for the whole loop).

**Affine addresses and vector loads.** An index expression is split into
`Σ coeff·atom + const`; the register part is materialized once and the
constant goes into the instruction's `[base+offset]` immediate. The lowering
records each address register's proven alignment (coefficients times the
element size over a 16-byte aligned base), and after lowering `vectorize.py`
merges runs of adjacent scalar global loads into `ld.v2`/`ld.v4`, moving loads
only earlier and never past a store or barrier (unless the data is read-only).

**Dead-code elimination** runs at the end to a fixpoint.

## 3. Lowering (`lower.py`)

`Lowerer.body` dispatches each statement to `stmt_<Kind>`. Names follow C
block scoping exactly as the CUDA C++ emmy would print: a `Tile`, loop or
branch body is a block; declarations shadow; a name declared before a loop
and reassigned inside it (an `Accum`, a combined state) is one mutable
register.

| Statement | Lowering |
|---|---|
| `Tile` | `gid = ctaid·block + tid` (or the aux-band form), guard `gid < N` unless the range proves it, row-major decode of the axes, optional grouped CTA rasterization; with idle threads, cells spread over lanes (4.4) |
| `Loop`, `StridedLoop` | seed the immediate body's accumulators when `seed`, then one of the shapes below |
| `Cond` | predicated branch, or no branch at all when the predicate is decided by ranges |
| `Load` / `Write` | affine address, `ld/st[.v2/.v4]`, `ld.global.nc` for read-only buffers, `red.global.add` for atomic writes |
| `Assign`, `Accum`, `Init`, `Let`, `Select` | elementwise semantics (section 6); an `Accum` add whose base or value is a product becomes one FMA |
| `Smem` | static `.shared` arrays up to 48 KiB in total, else one `.extern .shared` pool packed exactly like emmy's `pack_smem` |
| `Sync` | `bar.sync 0`, `bar.sync id, count`, `bar.warp.sync -1` |
| `WarpShuffle` | `activemask` + `shfl.sync.bfly` butterfly, the combine program run per step |
| `TreeHalve` | every thread folds the partials itself, pairing them exactly as the halving tree would, after the barrier that precedes it: one barrier instead of one per step (4.6) |
| `CpAsyncCopy/Commit/Wait` | `cp.async.{cg,ca}.shared.global`, `commit_group`, `wait_group N` |
| `ZeroPrologue` | CTA 0 zeroes the delegated accumulator |

**The shape of a loop** (`Lowerer._loop_body`), tried in this order:

1. *split across lanes* (4.2): a reduction that a group of lanes would each
   compute whole;
2. *lane-blocked* (4.3): an iteration whose expensive part the lanes share but
   whose folds differ per lane (attention);
3. *dealt* (4.4): independent iterations that a group of lanes would each run;
4. *unrolled*: bounded and small (a loop over a shared-memory tile up to 128
   trips, so all its loads take immediate offsets: a 256³ `cp.async` matmul
   6.8 → 4.8 µs), or marked `unroll`;
5. *rolled*: 4-8 iterations per trip plus the leftover (in one-wave grids even
   large bodies, by a divisor of the trip count).

Unrolled and rolled loops run iterations in their original order, so their
sums round exactly as the IR specifies. Read-only loads are issued as early as
their indices allow (`_issue_loads_early`), so their latencies overlap the
arithmetic between them.

## 4. Lanes that compute the same thing

Emmy's schedules often have many lanes compute the same thing: all the lanes
that hold one query row compute that row's attention scores, every lane of a
warp computes the normaliser of a softmax row, or one thread folds a whole
row while 255 others idle. Most of this section is about finding that sharing
and splitting the work instead.

### 4.1 Uniform groups

Every value carries `ug`: the largest aligned group of lanes (1 to 32) that is
guaranteed to hold it alike. Lane `l` of a warp holds flat id `base + l` with
`base` a multiple of 32 (when the block size is), so the tile coordinate
`(gid / stride) % extent` is the same throughout each aligned group of
`lowbit(stride)` lanes. Constants and block indices are uniform over 32;
an instruction's result is uniform over the smallest group of its operands;
a load from a uniform address is uniform; a butterfly leaves its group
agreeing. `Lowerer._uniformity` extends this over one iteration of a loop body
("if the loop variable is the same on G lanes, which values are?").

### 4.2 Splitting a reduction across the lanes that share it

When a loop's folds are uniform over G lanes (`_fold`), those G lanes split
its iterations and merge their partial states with log2(G) butterfly steps:
the same reduction, regrouped, which an associative and commutative combine
allows. Each lane's partial is never observed on its own. `_split_plan`
decides how:

* **Which loop.** A reduction nested in another can be split at either level.
  Splitting the outer loop leaves each lane whole inner loops and one merge at
  the end; splitting the inner one merges once per outer iteration. The choice
  is mostly coalescing: lanes of a split loop read the addresses of
  consecutive iterations, one cache line for a loop along a row, a line per
  lane for a loop down a column. A cost model counts 128-byte lines per warp
  load from the affine stride of each load (`mapping.flat_coefficient`) plus
  the butterflies, and splits where it is cheaper (the coalescing-driven choice
  of PPCG, Verdoolaege et al. TACO'13, and Yang et al. PLDI'10).
* **How iterations are dealt.** A short straight-line body gives each lane one
  contiguous chunk (one vector load). Otherwise lane r takes the V consecutive
  iterations from `V*r + V*G*k` (blocked-cyclic, V = 4 when that turns loads
  into 128-bit vector loads without spreading others), and past 16 rounds the
  rounds become a loop of their own.
* **Unroll-and-jam.** Several consecutive iterations of a lane are lowered as
  one body (`mapping.jammed`), so their inner loops fuse: a value they share
  is loaded once and the folds become independent chains (Callahan, Carr and
  Kennedy, PLDI'90).
* **Merging an online softmax.** An online softmax (Milakov and Gimelshein,
  2018) keeps a running max `m` and sums `l` rescaled whenever `m` grows. Its
  state is a monoid: folding one element `(s, w)` is combining with the
  one-element state `(m, l) = (s, w)`. So the merge of two partial states is the
  IR's own update with the element replaced by the partner's state
  (`mapping.combine_program`), derived rather than hard-coded (combiner
  synthesis, Farzan and Nicolet PLDI'17, for this one monoid).

`softmax(x @ w)` with 512 columns: the column loop is split across the warp
with coalesced loads of `w`, ten per-row loops fused, four columns jammed:
22,360 µs → 210 µs (nvcc: 58,694 µs).

### 4.3 Lane blocking (attention)

In attention every lane holding the same query rows computes the same scores
for every key, but folds them into its own output columns, so the key loop is
no shared reduction. `mapping.blocked` rewrites it IR to IR:

* The body splits into a **shared part** (values the G lanes agree on, not
  reading the loop's state: the dot products and the scale) and an **own part**
  (what differs per lane: the loads of `v` and the folds).
* In each block of keys, the G lanes form slots and slot r computes the shared
  part of the block's key r only. Then, for each key j of the block, every lane
  receives slot j's results and runs its own part for key j (the work split of
  FlashAttention-2, Dao, ICLR'24).
* The online softmax is updated **once per block**: a butterfly over the slots'
  scores gives the block's max, each slot weighs its own key against the new
  max, and every sum is rescaled once per block instead of once per key. The
  normaliser `l` stays one partial per slot, added up after the loop. Folds
  outside a softmax (sigmoid or relu attention) fold the same values in the same
  order as before, bit for bit.
* A partial last block lets idle slots redo the last key with score `-inf`, so
  they add nothing.
* **Slots.** One lane per key reads `kt[d, t]` coalesced (consecutive keys,
  consecutive addresses). With keys stored row by row (`k[t, d]`) a lane per key
  would read a line per lane, so a slot then has S lanes, which split the key's
  dot product and read its row coalesced. More lanes per slot cost butterflies
  and leave fewer keys per block to share the softmax update; a lane per 64
  bytes of row was best measured on 6 of 7 such kernels (twice that on a full
  SM).
* **Broadcasts through shared memory.** The lanes receive each key's weight
  from its slot. A shuffle (`shfl.sync.idx`) costs about 17 times a
  shared-memory broadcast (microbenchmark: 16 warps each broadcasting 32 values
  256 times took 68 µs as shuffles and 4 µs through shared memory, over the
  same arithmetic), so each block stores its weights to a small per-warp
  array between two warp barriers and reads them back with loads that ptxas
  merges into 128-bit reads. Blocks that broadcast fewer than 8 values per lane
  keep the shuffles (no gain measured).
* Unrolled blocks keep the query rows in registers; that pays while they take
  at most half the registers a thread may have, else the blocks run as a rolled
  loop that reloads them (ptxas hoists what fits).

The rewrite is checked by a small interpreter that runs the rewritten loop on
a group of lanes in lockstep and compares with each lane running the original
alone (`tests/test_blocking.py`), and on the GPU. On 37 attention problems:
geometric mean 0.44x the previous time (total 2015 → 597 µs), e.g.
`a_q64_k1024` 646 → 102 µs, `a_1024x64` 285 → 35 µs, the open `a01` 15 → 7.2 µs.

### 4.4 Idle threads

Some emmy schedules fold a whole row in one thread while most of the block
idles: a 65,536-element sum on one thread of 256, 32 rows of 2048 on 256
threads. When a Tile has at least twice as many threads as cells and nothing
a cell does depends on which thread runs it (no cross-thread statement, shared
memory, atomic or thread index), cell c runs on an aligned group of up to 32
lanes (`cell = gid / spread`). The lanes compute it alike, so the lane split of
4.2 shares its reductions, and loops whose iterations are independent (no
carried state, each buffer stored once per iteration at an index that moves with
it, nothing read that the loop stores) are **dealt**: lane r runs iterations
r, r + G, .... A kernel is lowered with spread cells only if, lowered without,
no loop shared work across lanes; a single-query attention, whose cells already
share their scores, got slower with it.

When each block holds exactly one cell, the cell takes the whole block: its
reductions are dealt over all the block's threads and merged in two stages, a
butterfly within each warp and then the warps' totals through shared memory
(4.6).

`x_sum` (one cell, 256 threads) 328 → 6.8 µs; `x_rms` 121 → 13 µs;
`x_softmax` 144 → 21 µs.

### 4.5 Cooperative reductions in contiguous chunks (`regroup.py`)

Emmy's cooperative reductions stride the B threads of a group over the data
and then combine the partials across all of them. Each thread's partial is
never observed alone, so thread t may fold the contiguous chunks
`V*t + V*B*k + j` instead, which turns V scalar loads into one vector load. It
applies only when the statements right after the loop are exactly such a full
combine. Emmy may also unroll the loop R times ("coop/r2": copy j reads
`x[a + j*B]` into an accumulator of its own, all folded together before the
combine); the chunks' elements are then dealt to the copies in turn, which
keeps R independent chains (a 65,536-element sum on one warp: 25 → 8.9 µs, nvcc
17 µs).

### 4.6 Block reductions with one barrier

Emmy combines a block's partials with a halving tree in shared memory: for
h = W/2 ... 1, the threads below h combine slot i with slot i + h, then a
barrier, and finally every thread reads slot 0. Each step costs a barrier and a
divergent branch. Since every thread can read all W partials once they are
written, each thread instead folds them itself, pairing them exactly as the
tree would, so it computes the very value slot 0 would hold, bit for bit;
one barrier at the end keeps the ordering the tree's last one gave the code
after it. The same scheme merges a cell that spans a block's warps (4.4): each
warp's butterfly total goes to shared memory, and after one barrier every
thread folds the W totals.

## 5. Loop-body rewrites (`mapping.py`, `congruence.py`)

* **Congruent accumulators.** In attention each output column a thread
  handles carries its own running max and sum, yet the columns of one row see
  the same scores. Optimistic partition refinement (the idea of global value
  numbering, Alpern, Wegman and Zadeck POPL'88) proves which accumulators
  always hold the same value; one representative is kept and the others alias
  it after the loop.
* **Common subexpressions** (`mapping.cse`): the merged cells' copies of the
  representative's arithmetic disappear, so the analyses see one online
  softmax update per row.
* **Horizontal fusion** (Kennedy and McKinley, LCPC'93): a later loop over the
  same iterations that is independent of a loop and of everything between them
  joins it. Every fold still sees the same values in the same order, so results
  are bit for bit the same, while a value both loops read is loaded once. Only
  loops that stay loops are fused: fusing loops that become straight-line code
  only reorders it and moved ptxas's schedule by ±10% either way.
* **Pruning**: statements nothing reads (backward liveness over an iteration).
* **One exponential per online-softmax step**: the two rescaling factors
  `exp(m - max(m, s))` and `exp(s - max(m, s))` have one exponent exactly 0,
  so both come from one `exp` of their sum (`mathlib._rescale_pair`).

## 6. Numerics (`elementwise.py`, `mathlib.py`)

* **Basic arithmetic is IEEE single, round to nearest** (`add.rn`, `mul.rn`,
  `div.rn`, `sqrt.rn`, `rcp.rn`). Written with `.rn`, ptxas may not contract a
  multiply and an add into an FMA, so pointwise kernels match numpy's float32
  results bit for bit (several problems demand `rtol = atol = 0`; nvcc's own
  build of the reference CUDA fails some of them because it contracts).
* **Reductions contract on purpose**: `acc += a*b` becomes `fma`, as nvcc
  does: faster and more accurate, and those problems carry a tolerance.
* **Regrouped sums.** Splitting a reduction across lanes, or updating an
  online softmax once per block, changes the order of a sum, never what is
  summed. Against numpy's answers (float64 for reductions, matmul and
  attention) our largest error over 283 problems is 14% of the tolerance and
  the median 0.13%; numpy's own float32 computation has a median of 0.12%.
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
  | `pow` | `2^(y·log2|x|)` in float64 (atanh series, Taylor 2^t) plus the C99 special cases; small constant exponents as products | correctly rounded in practice |
  | `sigmoid` | emmy's spelling `1/(1+exp(-x))` | ≤ 3 ulp |

  The polynomial coefficients are minimax fits produced by
  `tools/fit_poly.py`, which simulates the GPU's FMA evaluation to report each
  kernel's worst ulp error; `tools/mathtest.py` measures the finished routines
  on the GPU over about a million inputs per range.

## 7. Launch hints

`.maxntid` carries the block size (like CUDA's `__launch_bounds__`), so ptxas
never uses more registers than a block can launch with. When the whole grid
fits in one wave (at most one CTA per SM on the 170-SM RTX 5090) the module
also says `.minnctapersm 1`: extra registers then cannot cost occupancy, and
ptxas uses them to keep more loads in flight, which makes the latency-bound
attention kernels 25-40% faster.

## 8. Validation

* `tools/run_suite.py` compiles problem folders exactly as the evaluator does
  and checks them on the runner, reporting how much of the tolerance each
  result used and its worst ulp error.
* `tools/corpus_specs.py` + `tools/make_corpus.py` regenerate a wider problem
  set with the organisers' own generator: odd sizes, broadcasting, every
  traceable elementwise op, max/min reductions, layernorm, chained kernels,
  odd and batched matmuls, attention at other lengths and through
  `scaled_dot_product_attention` (causal and multi-head);
  `tools/corpus_knobs.py` forces the same expressions through many emmy
  schedules.
* `tools/ref_bench.py` races our PTX against nvcc's build of emmy's reference
  CUDA on identical launches. Timings move by about 2 µs from run to run, so
  every scheduling decision above was kept only after an A/B comparison
  (best of 5 runs) on the problems it affects.
* `tests/` holds unit tests of the builder and of the rewrites, the lane-level
  interpreter for lane blocking, end-to-end compile checks, and hand-built IR
  run on the GPU against numpy models (`EMMY_RUNNER` enables them).

## 9. Not supported

The statements TASK.md marks as outside the scope (tensor-core fragments,
TMA/mbarrier, `SetMaxNReg`, `Carry`/`Pre` recurrences) raise a clear error.
Emmy only produces them for tensor-core schedules (for example half-precision
matmul) and warp-specialized pipelines.
