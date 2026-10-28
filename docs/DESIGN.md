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
      │                         (should that fail: lowered again plainly, without the optional rewrites, §3)
      │      ├─ cells                            where a Tile's cells run (decode, spread, placement)
      │      ├─ lanes                            lanes sharing a loop's work (splits, blocking, dealing)
      │      ├─ mapping / congruence / regroup   loop-body analyses and IR-to-IR rewrites
      │      ├─ tensorcore                       the mma.sync fragment statements
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

## 3. Lowering (`lower.py`, `cells.py`)

`Lowerer.body` dispatches each statement to `stmt_<Kind>`. Names follow C
block scoping exactly as the CUDA C++ emmy would print: a `Tile`, loop or
branch body is a block; declarations shadow; a name declared before a loop
and reassigned inside it (an `Accum`, a combined state) is one mutable
register.

| Statement | Lowering |
|---|---|
| `Tile` | `gid = ctaid·block + tid` (or the aux-band form), guard `gid < N` unless the range proves it, row-major decode of the axes, optional grouped CTA rasterization; with idle threads, cells spread over lanes (4.4); cells placed anew where no aligned lanes share a row (4.5) |
| `Loop`, `StridedLoop` | seed the immediate body's accumulators when `seed`, then one of the shapes below |
| `Cond` | predicated branch, or no branch at all when the predicate is decided by ranges |
| `Load` / `Write` | affine address, `ld/st[.v2/.v4]`, `ld.global.nc` for read-only buffers, `red.global.add` for atomic writes |
| `Assign`, `Accum`, `Init`, `Let`, `Select` | elementwise semantics (section 7); an `Accum` add whose base or value is a product becomes one FMA |
| `Smem` | static `.shared` arrays up to 48 KiB in total, else one `.extern .shared` pool packed exactly like emmy's `pack_smem` |
| `Sync` | `bar.sync 0`, `bar.sync id, count`, `bar.warp.sync -1` |
| `WarpShuffle` | `activemask` + `shfl.sync.bfly` butterfly, the combine program run per step |
| `TreeHalve` | emmy's tree as written (each level folds in shared memory, then a barrier), or, when nothing can observe the shared slab afterwards, every thread folds the partials itself, pairing them exactly as the tree would: one barrier instead of one per level (4.7) |
| `CpAsyncCopy/Commit/Wait` | `cp.async.{cg,ca}.shared.global`, `commit_group`, `wait_group N` |
| `ZeroPrologue` | CTA 0 zeroes the delegated accumulator |
| `RegFragment`, `LdmatrixLoad`, `MmaSyncPtx`, `Fragment*`, `RegStore` | tensor cores (section 6) |

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

**The plain lowering, a fallback.** Every rewrite in this document is
optional: each applies only when its preconditions hold, and each has been
checked on hundreds of problems. Still, the hidden set may hold an IR shape
that no rewrite foresaw. So `compile_file` catches any failure of the
optimizing lowering, and also a module longer than the runner's 2,000,000
characters. It prints the traceback to stderr and lowers the kernel again
with `Lowerer(plain=True)`. That lowering is one rule per statement, in the
IR's order, with loops unrolled only when the IR asks. None of the optional
rewrites run: no early loads, fusion, regrouping, congruent accumulators,
lane splits or blocks, dealt loops, spread cells, partial unrolling or
vector-load merging. The builder's local work (folding, ranges, value
numbering, LICM) stays. `compile.py --plain` forces this lowering, so it is
tested like the main one: 469 of the 473 problems pass with it, and so does
every GPU test. The other four are the two generator faults no compiler
passes (9) and the two divisions by a constant checked at `rtol = 0`, which
it computes as emmy's multiply by the reciprocal (7).
It is slower, because lanes no longer share work (the open set's 64×64
attention takes 81 µs instead of 9), but it computes the same results.

**The copy engine (`--bulk-copy`, `bulk.py`).** Emmy fills shared tiles with
per-thread `cp.async` copies of 4 to 16 bytes, committed in groups and awaited
with `cp.async.wait_group N`. With `--bulk-copy`, a fill in which each aligned
group of lanes copies one contiguous run (consecutive, 16-byte aligned source
and destination addresses) is handed to the copy engine instead. The run's
first lane issues one `cp.async.bulk`, the TMA unit's one-dimensional form,
which needs no tensor map, and it completes on an mbarrier.

* `bulk.plan` proves each run by evaluating the copy's addresses with numpy
  for every thread of the launch and every iteration of the loops around it.
  Anything else keeps its `cp.async` (swizzled tensor-core tiles, copies in
  branches, groups committed in another loop), and both kinds can share a
  group.
* A commit becomes thread 0's `mbarrier.arrive.expect_tx` for the group's
  bytes, on one of 8 mbarriers used round robin. A wait waits in order for the
  groups it covers: group g is mbarrier g % 8 in phase g / 8 % 2.
* The Tile ends by waiting for every group still in flight, since a block
  must not exit while the engine writes its shared memory.
* The commit and wait counters live in registers that only emitted
  instructions touch. The builder's value numbering would otherwise take them
  for constants inside rolled loops. Such modules say `.target sm_90`.

All 68 problems that fill shared memory pass with it, but it is off by
default: on this GPU it never beat `cp.async` for emmy's fills (geometric mean
1.18x the time over 58; runs of 2-4 KB ran 1.2-1.5x slower; m01 15.9 → 27.1
µs). The engine pays off for large boxes, and their two-dimensional form needs
a tensor map built on the host, which the runner's API cannot pass.

## 4. Lanes that compute the same thing (`lanes.py`, `cells.py`)

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
(4.7).

`x_sum` (one cell, 256 threads) 328 → 6.8 µs; `x_rms` 121 → 13 µs;
`x_softmax` 144 → 21 µs.

### 4.5 Cells placed anew

Lanes share work only when the cells that need it sit in an aligned group of
lanes. Emmy lays a Tile's cells out row-major, and some of its tilings defeat
that:
* A row of 11 cells (three output columns each, for 32 columns) fills no
  power-of-two group of lanes at all.
* A row of 14 cells fills only aligned pairs (`lowbit(14) = 2`).
* Grouped rasterization applied to the thread axes puts consecutive cells on
  consecutive rows, so a row's lanes sit 8 apart.

So each row's scores are computed by every lane, or by every pair of lanes.

When a cell does not depend on which thread runs it (as in 4.4, except that
atomic stores are allowed, since each cell still runs once), the compiler
chooses where each cell runs (`Lowerer._placement`):

* **Groups.** Cells are grouped by every coordinate except the fastest axis's
  (a query row).
* **Lane groups.** Each group gets aligned groups of lanes whose sizes are
  powers of two: 20 cells as 16 + 4 lanes; 11 cells as 16 when the launch has
  threads enough, else 8 + 4 with one lane idle, else exactly 8 + 2 + 1.
* **Regions.** Each size forms a region of whole warps, so no warp diverges.
  Lane u of a region takes group u / P and cell u % P of that region's share
  (`_placed_tile`). The lowering then sees each row uniform over P lanes, and
  lane blocking (4.3) or a lane split (4.2) shares its work.
* **Idle lanes.** A lane past a group's last cell, or past the last group,
  computes a copy of the last cell so that whole groups take part, but its
  stores are predicated off.

Placement is tried only when two things hold. Some loop's work must be the
same for all cells of a group (`mapping.dependents`). And every lane group it
would form must be larger than the group of lanes that holds a row as emmy
lays the cells out. It is kept only if a loop is then shared and the module,
with one copy of the body per region, still fits the runner's limit.

The region with the smallest groups sets the time, because the lanes holding
a row's remainder must still score all of its keys. Thirteen of the 493
kernels are placed, all faster; nothing else changes:

| problem | as emmy lays it out | placed |
|---|---|---|
| `a_d40` (d = 40, rasterized lane by lane) | 41.7 µs | 19.1 µs |
| `a_batched` (11 cells of three columns a row) | 19.1 µs | 10.9 µs |
| `a_d56` (14 cells a row, in pairs) | 31.3 µs | 8.8 µs |
| `a_dv40` | 35.3 µs | 10.7 µs |
| `a_dv24` (12 cells a row, rasterized) | 33.4 µs | 6.8 µs |
| `a_d88` (44 cells a row, in fours) | 31.3 µs | 14.9 µs |
| `k_mlp`, `relu(x @ w1) @ w2` fused (22 cells a row) | 125.6 µs | 76.4 µs |
| `ha_48x80x24` (held-out set) | 10.8 µs | 6.8 µs |
| `k_mlp_gelu_8x512x2048x64`, a fused MLP of corpus round 6 (one block) | 20815 µs | 388 µs |

The other four (`a_relu_attn`, `a_relu_32x48`, `a_b8_32x32` and
`a_d40_q128`) gain 1.2 to 1.8x. The corpus holds 18 attention problems
written to probe these tilings (`tools/corpus_specs.py`, round 4).

### 4.6 Cooperative reductions in contiguous chunks (`regroup.py`)

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

Moving elements between threads is legal only if no partial is observed on
its own, so `combine_refusal` checks the whole combine program before the
rewrite: each partial must be combined with its own fold and its partner
(`acc = fold(acc, acc_b)`; a maximum of per-lane sums is refused, since the
elements must then stay with their lanes), in the fold's dtype, with nothing
else in the program reading it; the warp partials may reach shared memory
only as the input of the tree over the warps, and that tree's slab must be
private to it (4.7). `--report` prints the reason for each refusal, e.g.
`loop i: kept strided, not regrouped: the butterfly combines s with maximum,
but the loop folds it with add`.

### 4.7 Block reductions with one barrier

Emmy combines a block's partials with a halving tree in shared memory: for
h = W/2 ... 1, the threads below h combine slot i with slot i + h, then a
barrier, and finally every thread reads slot 0. Each step costs a barrier and a
divergent branch. Since every thread can read all W partials once they are
written, each thread instead folds them itself, pairing them exactly as the
tree would, so it computes the very value slot 0 would hold, bit for bit;
one barrier at the end keeps the ordering the tree's last one gave the code
after it. The same scheme merges a cell that spans a block's warps (4.4): lane
0 of each warp stores the warp's butterfly total (one writer per slot), and
after one barrier every thread folds the W totals.

The tree also leaves a state in shared memory: the root in slot 0 and partial
combinations in the others. The one-barrier form does not, so it is used only
when nothing can observe that state (`Lowerer._tree_refusal`): no other
statement of the kernel reads the slab, and the tree runs once (it is in no
loop, so no later run reads what an earlier one left). Otherwise, and always
in the plain lowering, the tree runs as emmy writes it, with its stores and a
barrier per level. All 155 trees of the corpus qualify.

### 4.8 Work repeated across blocks: thread-block clusters (`cluster.py`)

Lane sharing stops at the edge of a block. Some emmy schedules give every
block one output and have it recompute what its neighbouring blocks compute
too. `gelu(x @ w1) @ w2`, fused into one kernel of 4096 blocks of 8 threads,
recomputes the 512 hidden units of a row once for each of that row's 128
output columns. Hopper and Blackwell let neighbouring blocks share work
through distributed shared memory:

* The module declares `.reqnctapercluster 8, 1, 1`, so the unchanged launch
  runs in clusters of 8 neighbouring blocks: neighbouring columns of one row.
* `mapping.dependents` splits the loop into a shared part, which does not
  depend on the column (the dot products and the gelu), and an own part, which
  does (the column's weights and its fold).
* Block r of a cluster runs the shared part of its threads' iterations
  `[r K/8, (r+1) K/8)` and stores the results in its shared memory. After one
  cluster barrier, every block runs the own part of all K iterations, reading
  each result from the block that computed it (`mapa` +
  `ld.shared::cluster`). A second barrier keeps each block's shared memory
  alive until the others have read it.

Every value is computed as before and folded in the same order, so the result
is bit for bit the same, under `--strict` too. The fused MLP runs in 173 µs
instead of 389 (nvcc's build: 408).

* **Cluster size.** 8 blocks is the portable maximum, and it measured best:
  4 blocks took 349 µs and 2 blocks 656 µs, because scheduling clusters of
  8-thread blocks costs more than small clusters save.
* **When it applies.** A loop is split only when its shared part takes at
  least 2048 inner iterations per thread. One of the 493 kernels has this
  shape.
* **Fused MLPs emmy tiles instead** (corpus round 6, with gelu, relu, tanh and
  silu) share their hidden layer within blocks by lane blocking (4.3). They
  run at 0.01 to 0.08 of nvcc's time.
* **Attention is left alone.** Its column blocks (two to eight per row) also
  repeat their scores, but sharing them would take a cluster barrier for each
  block of keys, about 0.4 µs apiece, which costs about what it saves.

### 4.9 Two cells a thread for SIMT matmuls (`coarsen.py`)

Emmy's f32 matmul schedules stage tiles of A and B in shared memory with
`cp.async`, and give each thread a small cell of outputs: 2 x 2 for a 512³
matmul on blocks of 512 threads. Each multiply-add then costs its own
shared-memory load. Per step of k, a warp loads 64 values of B and two
broadcast values of A for 128 multiply-adds, so shared memory sets the pace,
not the FMA units. Measured on m01, a hand-written split of the kernel: the
copies alone take 8.5 µs, emmy's arithmetic alone 12.9 µs, and the measured
peak rate allows 4.1 µs for its million multiply-adds per SM.

Two cells that hold the same columns share their B values. So:

* In a Tile whose cells are rows x columns of a block of at least 512
  threads, with the cell columns a multiple of a warp, the threads of the
  first half of the block each compute two neighbouring cells, rows 2p and
  2p+1.
* Both cells' work goes into one inner loop, so each step of k loads B once
  for both. The second, identical loads disappear inside ptxas, which removes
  repeated shared loads within a basic block.
* The other half of the block only fills the tiles.

Each cell's sum keeps its order, so the results are bit for bit the same.
Four kernels have this shape:

| problem | before | two cells a thread |
|---|---|---|
| m01_matmul_512 (open set) | 15.9 µs | 12.9 µs |
| x_mm__raster-gn8 | 6.6 µs | 4.9 µs |
| x_mm_rect__reduce-g4a | 7.1 µs | 4.9 µs |
| hm_512x128x256 | 6.9 µs | 6.9 µs |

Blocks of 256 threads gained nothing, and one ran slower (8.8 → 10.8 µs): only
four warps would be left to compute, too few to hide the shared-memory
latency.

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

## 6. Tensor cores (`tensorcore.py`)

For f16 matmuls and attention, emmy schedules the tensor cores the way CUTLASS
does on Ampere and later: a warp holds each operand tile in per-lane registers
and multiplies 16x8x16 tiles with `mma.sync`. Lane l owns fixed elements of
each tile (rows `l/4` and `l/4 + 8`, columns `2(l%4)` and `2(l%4) + 1` of the
f32 accumulator; the 16-bit operands two to a register in the same map). The
IR spells the computation out in eight statements, each lowered to the PTX it
names:

| Statement | Lowering |
|---|---|
| `RegFragment` | 4 (A), 2 (B) or 4 f32 (C) registers a lane, a family of them when arrayed; an accumulator starts at zero |
| `LdmatrixLoad` | `ldmatrix.sync.aligned.m8n8.x4`/`.x2` (`.trans` for a K x N slab) from shared memory, each lane addressing one 16-byte row through the slab's XOR swizzle, the same one its `cp.async` fill used; two B fragments in one `x4` where emmy pairs them. From global memory: the same lane map, element by element, the lane's row clamped at a masked tile's edge and K elements past a masked extent read as zero |
| `MmaSyncPtx` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (or bf16, or the f16-accumulate form), accumulating in place |
| `FragmentApply` | the op per C element: a softmax's scale, subtraction and exp, a mask from the element's coordinates, a bias read at them |
| `FragmentRowReduce` | each lane folds its two rows across the chunk's fragments, then two `shfl.sync.bfly` steps across the four lanes that hold a row |
| `FragmentRepack` | `cvt.rn.f16x2.f32`: two C fragments become an A operand without leaving the registers (the P of P @ V) |
| `FragmentPromote` | an f16 accumulator's halves added into its f32 shadow, which is then zeroed |
| `RegStore` | per lane, a row's two columns as one `st.v2.f32` or packed `f16x2` store (`red.add.noftz.f16x2` when split-K adds them), through the fused epilogue and the row and column guards |

A fragment is an array of registers, so a loop that indexes one by its
variable is always unrolled. The fragment statements count as cross-thread,
because a fragment's elements are spread over its warp's lanes, so none of the
rewrites of sections 4 and 5 moves, deals or places them.

Twenty f16 problems were generated with the organisers' tool, and all of them
pass. They cover matmuls of odd and batched shapes, `x @ w.T`, fused
bias/relu/gelu epilogues, flash attention, and causal and multi-head SDPA. On
19 of them we take 0.77x the time of nvcc's build of emmy's reference CUDA
(geometric mean; the causal SDPA 10.8 µs against 133.6). One is slower, a
batched 4x64x96 matmul (6.6 µs against 4.6), for the reason given in section
8. Two GPU
tests build tensor-core IR by hand, for the forms the problems do not use: an
f16 accumulator, a coordinate mask, a bias read and atomic f16x2 stores.

## 7. Numerics (`elementwise.py`, `mathlib.py`)

* **Basic arithmetic is IEEE, round to nearest** (`add.rn`, `mul.rn`,
  `div.rn`, `sqrt.rn`, `rcp.rn`), in the precision of each operation's result
  type. Written with `.rn`, ptxas may not contract a multiply and an add into
  an FMA, so pointwise kernels match numpy bit for bit (several problems
  demand `rtol = atol = 0`; nvcc's own build of the reference CUDA fails some
  of them because it contracts).
* **Float64.** Emmy's CUDA backend has no double arithmetic: it declares every
  loaded value and temporary `float`, so its build of an f64 problem computes
  in single precision. Numpy, whose answers the problems are checked against,
  computes in double. We compute every f64 result in double precision
  (`add.rn.f64` …), and keep float literals and constant buffers at full
  precision, so f64 pointwise kernels match numpy bit for bit. `exp`, `log`
  and `sigmoid` have double-precision versions; `tanh`, `erf`, `sin`, `cos`
  and `pow` compute an f64 result in single precision. Reductions follow the
  IR's types, and emmy gives an f64 sum an f32 accumulator, f32 shuffles and
  a `float` shared array, so those results carry f32 accuracy whatever the
  backend. A 64-bit shuffle moves as two 32-bit halves, and a split-K f64
  matmul adds with `red.global.add.f64`. On float64 problems generated with
  the organisers' tool and read back at full precision (scale and shift,
  division, `where`, `relu`/`max`, `rsqrt`, `exp`, `log`, `sigmoid`, softmax,
  layernorm, matmul), every pointwise result is bit-exact or within 2 ulp.
  (The evaluator reads outputs back as float32, so for it any f64 tolerance
  below float32 precision is out of reach in any case.)
* **fp8.** An `f8e4m3` or `f8e5m2` value is its byte code in an integer
  register (loaded and stored like `u8`); no arithmetic runs on it, only
  emmy's casts. `from_f8*` widens a code with `cvt.rn.f16x2.e4m3x2` (or
  `e5m2x2`): every fp8 value is exactly an f16, e4m3's 0x7f and 0xff come out
  NaN, e5m2 keeps its infinities. `to_f8*` is one `cvt.rn.satfinite`: round to
  nearest even, saturate to the largest finite value, NaN to 0x7f, which is
  emmy's encode (cuda_fp8.h's `__NV_SATFINITE`). A GPU test checks all 256
  codes of each format and a thousand values of every kind against emmy's own
  `decode_f8` / `encode_f8`, bit for bit. These conversions need PTX ISA 8.1
  on sm_89, so only modules that use them say `.version 8.1`.
* **Reductions contract on purpose**: `acc += a*b` becomes `fma`, as nvcc
  does: faster and more accurate, and those problems carry a tolerance.
* **Constants keep their type and their sign.** Constants are value-numbered
  by their bit patterns, so `+0` and `-0` stay different values (`1/x` tells
  them apart), in the builder and in the IR-level CSE alike. An `Init`, a
  typed `Let` of a literal and a reduction's seed are built in their own type,
  rounded once: an f64 constant keeps all 53 bits, an integer one all its
  bits.
* **Integers** follow numpy's fixed-width arithmetic in the result's type:
  `add`, `multiply` and the rest wrap at the width; comparisons compare the
  operands in their own types, exactly at any magnitude (never through
  float32, where 16,777,216 equals 16,777,217); `where`, `maximum`, `minimum`
  and `abs` with an integer result stay integers. `floor_divide` and
  `remainder` round down, like numpy's `//` and `%` (and torch's), which the
  problems are checked against; emmy's CUDA prints C's `/` and `%`, which
  truncate, so the two differ for negative operands. Unsigned values (u8 to
  u64) keep their sign: logical right shifts, unsigned division, remainder
  and comparison, and zero-extension when widened. u8, u16 and i16 results
  wrap to their width.
* **Signed zeros, NaN and the policy for rewrites.** Every rewrite is in one
  of three classes: exact (bit for bit the IR's result: CSE, fusion,
  congruent accumulators, unrolling, vector loads, the one-barrier tree, cell
  moves, clusters, coarsening), reassociating within rounding (lane splits,
  regrouping, per-block softmax updates, FMA contraction; off under
  `--strict`), or recovering a source operation (the division below; off under
  `--strict` and `--plain`).
* **Division by a constant, correctly rounded, where the IR records one.**
  Emmy's `split_invariant_divides` (emmy/compiler/ir/stmt/normalize.py) turns
  `x / 3.0` into `x * reciprocal(3.0)`, and `x * RN(1/3)` can be an ulp from
  the correctly rounded `x / 3` that numpy computes, which fails a pointwise
  problem checked with `rtol = atol = 0` (emmy's own CUDA fails it too; found
  by the second held-out set, §9). The IR's meaning is the product, the
  source's the quotient, so the rewrite applies only where emmy's graph
  records that the divisor came from a source division: a ConstantOp named
  after a division node (`div_c1`: the tracer names a scalar argument
  `<node>_c<k>`, trace/torch.py) or a mean's element count (`<out>_count`,
  decomposition/090_mean.py). In all 493 kernels every reciprocal of a constant
  is one of these; a reciprocal the IR asks for in its own right keeps its
  literal meaning (a GPU test checks both). Outside loops, where it is cheap,
  the optimizing lowering then computes the true quotient
  (`Lowerer._quotient`), without a division instruction: `div.rn.f32` is a subroutine with a slow path, and its call
  kept ptxas from moving loads across it (a row normalization ran 3.2 →
  4.5 µs). In double precision, `x * RN(1/c)` is within 2⁻⁵² of `x / c`; and
  `x / c` is never closer than 2⁻⁴⁹ (relative) to a float32 rounding boundary,
  because a boundary has an odd 25-bit significand that `x` and `c` cannot
  produce exactly. So rounding the double product to float32 gives
  `RN(x / c)` for every `x`, specials included; a test checks it on 4 million
  bit patterns per constant. It costs one double multiply per element: free
  at the usual sizes, 2 µs on a million-element division (6.8 µs, against
  4.6 µs for the multiply that fails). A power-of-two `c` stays a multiply,
  which is exact; `--strict` and `--plain` keep the IR's multiply.
* **Regrouped sums.** Splitting a reduction across lanes, or updating an
  online softmax once per block, changes the order of a sum, never what is
  summed. Against numpy's answers (float64 for reductions, matmul and
  attention) our largest error over the 471 passing problems (commit 8b76863)
  is 26% of the tolerance, on `k_mlp_gelu_64x256x1024x128`, whose 1024-term
  sums emmy accumulates in float32; in the IR's own order (`--strict`) the
  same problem uses 38%. The median is 0.14%. These are observations on the
  problems' inputs, not error bounds.
* **sin and cos of any float.** Below |x| = 105615 the argument is reduced by
  a three-part Cody-Waite step with FMAs. Beyond it, on a branch the usual
  inputs never take, the reduction is exact (Payne-Hanek): |x| = M · 2^(E-150)
  for its 24-bit significand M, the product of M with the first 192 bits of
  2/π is formed in 32-bit words with a 64-bit carry, and its 64 bits around
  the binary point give the quadrant and the fraction, which rounds to the
  nearest quadrant and becomes r = f · π/2 in double. The bits of 2/π past
  the 192nd move the product by less than 2⁻⁶⁴, far below how close any float
  comes to a multiple of π/2. Checked against mpmath (relative error of r
  2.2 · 10⁻¹³ over 2¹⁷..2¹²⁷) and on the GPU: sin and cos are within 1 ulp on
  2 million inputs in each of [0, 7], [7, 105615], [105615, 10⁹] and
  [10⁹, 3.4 · 10³⁸]. (The former double-precision reduction lost all accuracy
  past about 2⁴⁰: `sin(1e20)` gave 4.3 · 10³⁰.)
* **`--strict`: results bit for bit as the IR orders its operations.**
  `compile.py --strict` (`Lowerer(strict=True)`) turns off the rewrites that
  change a result within rounding:
  * the lane split of a reduction (4.2), whose merge regroups the sum;
  * regrouping into contiguous chunks (4.6);
  * lane blocking of an online softmax, which updates its state once per block
    of keys (4.3);
  * lane blocking that gives a key several lanes, which split its dot product;
  * FMA contraction of `acc += a*b`;
  * the correctly rounded division by a constant (it is more accurate than the
    IR's multiply by the reciprocal, but not the IR's own operations).

  Everything else stays, because it does not change a single bit:
  * fusion, congruent accumulators and CSE;
  * unrolling and early loads;
  * vector loads and stores;
  * the one-barrier tree (4.7);
  * dealt loops, spread cells and cell placement (4.4, 4.5);
  * thread-block clusters and two cells a thread (4.8, 4.9);
  * lane blocking of plain folds;
  * the tensor cores, whose order of operations is the IR's own.

  That claim is checked on the GPU (`tools/bitwise.py`). Of the 402 problems,
  the 371 whose kernels add no floats atomically give bit-identical outputs
  under `--strict` and under the plain lowering (`--plain --strict`, which
  runs the IR as written). The other 31 finish with float atomics, whose order varies
  from run to run. Strictness costs speed where lanes shared a reduction: the
  open set's 64×64 attention takes 85 µs instead of 9. It passes the same
  469 of 473 problems as the plain lowering.
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
  | `exp` (f64) | `x = j·ln2 + r` (fdlibm's split `ln2`, two FMAs); Taylor series of `e^r` to `r^13`; `2^j` as two exact factors, so subnormal results round once | ≤ 1 ulp |
  | `log` (f64) | `x = m·2^e`, `m ∈ [√½, √2)`; fdlibm's `log(1+f) = f - (f²/2 - s(f²/2 + R))`, `s = f/(2+f)`, `R` the `atanh` series | ≤ 1 ulp, subnormals included |
  | `sigmoid` (f64) | `1/(1+exp(-x))` with the f64 `exp` | ≤ 2 ulp |

  The polynomial coefficients are minimax fits produced by
  `tools/fit_poly.py`, which simulates the GPU's FMA evaluation to report each
  kernel's worst ulp error; `tools/mathtest.py` measures the finished routines
  on the GPU over about a million inputs per range (the f64 ones against an
  extended-precision reference).

## 8. Launch hints

`.maxntid` carries the block size (like CUDA's `__launch_bounds__`), so ptxas
never uses more registers than a block can launch with. When the whole grid
fits in one wave (at most one CTA per SM on the 170-SM RTX 5090) the module
also says `.minnctapersm 1`: extra registers then cannot cost occupancy, and
ptxas uses them to keep more loads in flight.

Measured on the 161 of 326 problems whose code the hint changes: without it
they take 1.20x as long (geometric mean), attention kernels up to 4.5x. One
problem is faster without it: `m_f16_bmm` (four 64x96 by 96x80 f16 matmuls,
16 blocks of 8 warps). There ptxas spends 162 registers on issuing all 144
fragment loads of the unrolled K loop before the first `mma`, and the kernel
takes 6.6 µs against 4.6 with its default of 66 registers. A rule that
excluded blocks reading many fragments from global memory did not hold on 8
new f16 matmuls: the hint was neutral on seven and 1.42x faster on the eighth.
So the hint stays for every one-wave grid. Capping registers (`.maxnreg 128`)
instead was 1.25x slower on the 72 kernels that use more than 128.

## 9. Validation

* `tools/run_suite.py` compiles problem folders exactly as the evaluator does
  and checks them on the runner, reporting how much of the tolerance each
  result used and its worst ulp error.
* `tools/corpus_specs.py` + `tools/make_corpus.py` regenerate a wider problem
  set with the organisers' own generator: odd sizes, broadcasting, every
  traceable elementwise op, max/min reductions, layernorm, chained kernels,
  odd and batched matmuls, attention at other lengths and through
  `scaled_dot_product_attention` (causal and multi-head), and attention in
  tilings that give no aligned group of lanes one row (4.5);
  `tools/corpus_knobs.py` forces the same expressions through many emmy
  schedules.
* `tools/corpus_holdout.py` is a held-out set written after the compiler, in the style of
  the open expressions (new shapes and op mixes in all five categories, the open set's
  tolerances and reduction schedule), to estimate the hidden set: 81 of 81 pass on first
  run, the largest error 10.5% of its tolerance. Launching the kernels that use shared
  memory or barriers eight times each gives bit-identical outputs (apart from atomic
  split-K sums). That is evidence, not a proof of race freedom (Compute Sanitizer
  cannot run on the evaluation runner); where lanes would store the same value to the
  same place (a spread cell's outputs, a warp's total), one lane now stores it.
* `tools/corpus_holdout2.py` is a second held-out set, written on 26 Oct after every
  optimization above: prime and odd sizes, single rows and queries, middle-axis and column
  reductions, matrix-vector and batched products, fused epilogues around a matmul, causal and
  multi-head attention, float16. 64 of 66 pass; the other two no compiler can pass (the
  generator lists a consumer kernel first; emmy traces `A @ v` into a 500x300 output). It
  found the division by a constant (section 7), which nvcc's build of emmy's CUDA fails too.
* An independent review (27 Oct) built small legal kernels that the corpus never forms:
  a maximum of per-lane sums after a cooperative loop, `x*(+0)` against `x*(-0)`, integers
  past 2^24, a u32 shifted right, an f64 `Init` of 1.0000000000000002, `sin(1e20)`, a read
  of a tree's shared memory, and a reciprocal written in the IR. Seven were wrong (the
  first only when optimized, the next six in every lowering) and the eighth was the
  division policy. Each is now a GPU test in all three lowerings
  (`tests/test_gpu_lowering.py`), next to positive cases (a sum of sums is still
  regrouped). The fixes changed the PTX of 43 of the 493 kernels; raced against the
  previous build (alternating order, best of three), they run at 0.999x its time.
  It also showed that agreement between `--strict` and `--plain` is not an independent
  oracle: both share the builder, the math library and, until then, the tree's lowering.
  The GPU tests compare against numpy models instead.
* `tools/corpus_holdout3.py` is a third held-out set, written on 27 Oct after those
  fixes were frozen and run once: sines and cosines of large arguments, divisions by
  constants and means at `rtol = 0`, minima, maxima and clamps, norms over odd lengths,
  and new shapes in all five categories. 39 of 39 pass on the first run (the generator
  could not trace a fortieth), the largest error 6.2% of its tolerance.
* `compile.py --report` (`report.py`) prints what the lowering decided for a kernel, as it
  decided it (`Lowerer.remark`): each loop's shape (split across lanes with its group and
  dealing, lane-blocked with its block, dealt, unrolled, rolled), fused loops, merged
  accumulators, re-mapped cells, one-barrier trees, FMA contractions, merged vector loads,
  launch hints, then a static count of the PTX's instructions by kind. The PTX is the same
  with or without it.
* `tools/ref_bench.py` races our PTX against nvcc's build of emmy's reference
  CUDA on identical launches. Timings move by about 2 µs from run to run, so
  every scheduling decision above was kept only after an A/B comparison
  (best of 5 runs) on the problems it affects.
* `tests/` holds unit tests of the builder and of the rewrites, the lane-level
  interpreter for lane blocking, end-to-end compile checks, and hand-built IR
  run on the GPU against numpy models (`EMMY_RUNNER` enables them). The
  compile checks and the GPU tests run every kernel through both the
  optimizing and the plain lowering. Two tests make the optimizing lowering
  fail on purpose, once by raising and once by producing too long a module,
  and check that the fallback takes over.

## 10. Not supported

These raise a clear error (and so cannot reach the GPU half-lowered):
* Hopper's `wgmma`, which the RTX 5090 lacks.
* emmy's TMA pipeline statements (`TmaLoad` and the `Mbarrier*` statements
  around it): a TMA descriptor is built on the host, which the runner's API
  cannot pass. The copy engine of section 3 issues its own `cp.async.bulk`
  and mbarriers, which need no descriptor.
* `SetMaxNReg`, `Carry` and `Pre`.
* The Volta m8n8k4 fragment layout, and the fp8 (m16n8k32) and fp4
  block-scaled tensor-core fragments (fp8 elementwise casts are supported).

Emmy produces these only for other GPUs and for warp-specialized pipelines. Double-precision `tanh`, `erf`, `sin`,
`cos` and `pow` are computed in single precision (section 7).

## 11. Support, limits and next steps

**What each data type supports**

| | f32 | f16 | f64 | integers (i16..i64, u8..u64), bool | fp8 (e4m3, e5m2) |
|---|---|---|---|---|---|
| `+ - * /`, `sqrt`, comparisons, `min`/`max`, `where` | IEEE, rounded once | computed in f32, rounded once to f16 | IEEE in double | numpy's fixed-width semantics (7) | — |
| `exp`, `log`, `sigmoid` | ≤ 1–2 ulp | via f32 | ≤ 1–2 ulp in double | via f32 | — |
| `tanh`, `erf`, `sin`, `cos`, `pow` | ≤ 1–3 ulp (sin, cos for every float) | via f32 | computed in f32 | via f32 | — |
| reductions, shuffles, trees | yes | yes | yes (64-bit shuffles) | yes | — |
| tensor cores (`mma.sync`) | — | m16n8k16, f32 or f16 accumulators | — | — | — |
| loads, stores, casts | yes | yes | yes | yes | emmy's encode and decode casts only |

The default target is sm_89 with PTX ISA 8.0; modules that use clusters or the
copy engine say sm_90, those with fp8 casts ISA 8.1, and every module runs on
the sm_120 runner. The 170-SM count of the RTX 5090 only steers a scheduling
hint (8); no result depends on it.

**What an independent review (27 Oct) raised, and where it stands**

| Point | Status |
|---|---|
| Regrouping must check the combine program; a maximum of partial sums is not a sum | fixed, with refusal reasons in `--report` (4.6) |
| +0 and -0 merged by value numbering | fixed: constants keyed by their bits (7) |
| A tree's shared memory may be read afterwards | fixed: emmy's tree as written unless provably unobservable (4.7) |
| Integer comparisons through f32; unsignedness lost; f64 `Init` through f32 | fixed (7) |
| `sin`/`cos` wrong past about 2^40 | fixed: Payne-Hanek, ≤ 1 ulp for every float (7) |
| Recovering `x / c` from `x * reciprocal(c)` changes the IR's meaning | now only where emmy's graph records a source division (7) |
| Same-value stores by several lanes | one writer per spread cell output and per warp total (4.4, 4.7) |
| The fallback catches failures, not wrong code; strict-versus-plain is not an independent oracle | stated as such (3, 9); the GPU tests compare with numpy models |
| Claims too broad: bit-exact, race-free, accuracy, the 17x broadcast | qualified, with datasets and revisions (README, 4.3, 9) |
| Fresh held-out data after the fixes | `holdout3`: 39 of 39 on first run (9) |
| An independent dtype-faithful IR interpreter, with randomly generated kernels checked against it in every lowering | done: `tests/oracle.py`, 72 random kernels in every lowering (12.1) |
| One planner for repeated work at every scope (lane, warp, block, cluster), choosing by work removed against communication, synchronization and register cost; strided lane classes the `lowbit` domain cannot describe | the analysis and cost model: `planner.py` (12.5); it agrees with the recognisers on 768 of 777 loops. Letting it drive them is next |
| A coalescing model in 32-byte sectors (alignment, dtype width, reuse) instead of 128-byte lines | `EMMY_PTX_COST=sectors`; the same decisions as the line model on all 533 kernels (12.4) |
| Raw paired benchmark data in alternating order, a family-weighted view, and an ablation table per rewrite | `tools/ablate.py`: per rewrite, alternating order, family-weighted (12.2) |
| Compile cost: the largest module (`h2a_1x512x64`, 1.1 MB of PTX) takes about 1.8 s to generate and 6 s in ptxas | `tools/compile_cost.py`, and `compile.py --explore`, a compile budget for exploratory trials (12.3) |
| Analyses that produce facts, plans that record their preconditions, a checker, then emission | the regroup and tree checks follow that shape; the other rewrites still decide while they emit |
| GPU tests and the corpus-dependent compile tests skip when the runner or the problem folders are absent | `EMMY_REQUIRE_ALL=1` turns those skips into failures |

## 12. Checking and measuring the compiler itself

Five longer-term suggestions of the review (28 Oct), as far as they were taken
before the deadline.

### 12.1 An independent interpreter (`tests/oracle.py`)

`tests/oracle.py` runs emmy's Kernel IR directly with numpy: every thread of a
block in lockstep (a value is an array over the block's threads), every value
in the type the IR gives it, result types from emmy's own `dtype_promote` and
reduction identities from emmy's `ElementwiseImpl`. It shares no code with
`emmy_ptx`. Arithmetic is numpy's, the reference the problems are checked
against: float32 in float32 without fused multiply-adds, integers in fixed
width. Cross-thread statements follow emmy's definitions: a butterfly over
aligned lane groups, a tree that folds its shared slab level by level and
leaves it so.

`tests/test_oracle.py` generates random kernels in four families: typed
pointwise expressions over f32, i32 and u32 (−0.0 and 2^24 + 1 among the
constants); reductions that several lanes share, folded with add, maximum or
minimum; cooperative reductions whose cross-lane combine may differ from the
per-lane fold, some reading the tree's shared memory afterwards; and attention
as emmy fuses it, an online softmax over 16 to 100 keys with masks, which the
optimizing lowering lane-blocks (groups of 4 to 32 lanes, partial blocks
included). Each is compiled optimized, strictly and plainly, run on the GPU,
and compared with the interpreter: bit for bit wherever the IR's order of
operations is kept (`--strict`, and every kernel without a sum or an `exp`),
within rounding otherwise. All 72 kernels pass in every lowering. Run against
the build from before the fixes of section 9, 25 of them fail: every pointwise
kernel (integers, unsigned values, signed zeros), the cooperative kernels that
read a tree's slab (in all three lowerings) and one maximum of per-lane sums
(optimized only); the attention kernels pass on both builds. The generator
finds the review's classes of bugs by itself.

### 12.2 What each rewrite contributes (`tools/ablate.py`)

`EMMY_PTX_DISABLE=split,blocking,...` switches rewrites off one at a time
(`lower.REWRITES`); with none set, the PTX is byte-for-byte unchanged. Over the
326 problems of the open set, the corpus and the schedule sweep, each rewrite is
switched off in turn. A problem whose PTX does not change then runs the same
code (its ratio is exactly 1); the others are raced on the runner, full build
against reduced build, in the order full, reduced, reduced, full.

| switched off | problems it changes | time without it (those) | (all 326) | (per family) | most affected |
|---|---|---|---|---|---|
| every rewrite (`--plain`) | 307 | 1.97x | 1.895x | 1.874x | `k_mm_softmax` 294.91x |
| `blocking`: lane blocking (4.3) | 57 | 2.60x | 1.182x | 1.246x | `a_1024x64` 8.04x |
| `hint`: the one-wave launch hint (8) | 289 | 1.10x | 1.087x | 1.108x | `a_256x32` 4.53x |
| `split`: reductions split across the lanes that share them (4.2) | 29 | 5.18x | 1.157x | 1.062x | `k_mm_softmax` 195.78x |
| `placement`: cells placed anew (4.5) | 12 | 2.72x | 1.038x | 1.054x | `k_mlp_gelu_8x512x2048x64` 53.33x |
| `unroll`: short loops unrolled though the IR does not ask | 219 | 1.08x | 1.050x | 1.045x | `a_b8_32x32` 2.70x |
| `fma`: a product folded into a sum as one FMA | 178 | 1.06x | 1.035x | 1.040x | `a_d80` 2.11x |
| `vectorize`: adjacent loads merged into vector loads | 152 | 1.10x | 1.043x | 1.035x | `x_attn__raster-gm8` 3.64x |
| `congruence`: provably equal accumulators merged (5) | 33 | 1.26x | 1.024x | 1.034x | `k_softmax_then_mm` 2.67x |
| `fusion`: sibling loops over the same iterations fused | 8 | 1.74x | 1.014x | 1.020x | `k_mm_softmax` 23.79x |
| `spread`: idle threads joining cells (4.4) | 23 | 5.55x | 1.128x | 1.013x | `x_sum__work-t128_reduce-coop-t` 48.60x |
| `early_loads`: loads issued ahead of the statements that need them | 198 | 1.01x | 1.007x | 1.009x | `p_mix3` 1.52x |
| `regroup`: cooperative reductions in contiguous chunks (4.6) | 41 | 1.09x | 1.011x | 1.005x | `x_sum__work-t32_reduce-coop-r2` 2.84x |
| `dealing`: independent loops dealt to the lanes of a cell (4.4) | 9 | 3.10x | 1.032x | 1.004x | `x_rms__reduce-r4` 4.42x |
| `clusters`: work repeated across blocks shared in a cluster (4.8) | 1 | 2.24x | 1.002x | 1.004x | `k_gelu_mlp` 2.24x |
| `tree`: block reductions with one barrier (4.7) | 100 | 1.01x | 1.002x | 1.002x | `x_rms__work-t512_reduce-coop` 1.36x |
| `coarsen`: two matmul cells a thread (4.9) | 3 | 1.24x | 1.002x | 1.001x | `x_mm_rect__reduce-g4a` 1.39x |

Times are ratios of the reduced build to the full one, geometric means; "per
family" counts the schedule variants of one expression (`x_sum__...`) once, so
the sweep's 106 variants of eight expressions do not outweigh the rest. With
every rewrite off the problems take 1.9x as long; the rewrites that change the
most kernels are the one-wave hint, unrolling, FMAs and vector loads (a few
percent each), while those aimed at repeated work matter a great deal where they
apply: lane blocking 2.6x on 57 problems, the lane split 5.2x on 29, spread cells
5.5x on 23, placement 2.7x on 12 (53x on one fused MLP), clusters 2.2x on one.
The rewrites overlap (a kernel without the split is often lane-blocked instead),
so the rows do not multiply. No reduced build gave a wrong answer. Single
extremes near the ~3 µs launch floor (`p_mix3`'s 1.52x) are runner noise; the
raw pairs are in `docs/results/ablation_2026-10-28.json`.

### 12.3 Compile cost, and a compile budget for exploration (`--explore`)

`tools/compile_cost.py` times a cold `compile.py` (interpreter start, emmy
import, lowering) and a local ptxas for sm_120, one kernel at a time. Over the
533 kernels: `compile.py` takes a median 0.43 s (95th percentile 0.81 s, the
longest 2.6 s), ptxas a median 0.03 s (95th percentile 0.32 s). The longest
used to be 6.1 s (`h3f_mm_softmax`): two thirds of it went to loop fusion's
legality check, which looked again at every statement between two loops for
each later loop it tried. It now looks at each once (`mapping._Between`), with
the same decisions and the same PTX. The modules
are a median 6 KB of PTX; the largest, `h2a_1x512x64`, is 1.1 MB: a 4-thread
kernel whose 32 iterations, each of four 64-trip dot products, are all
unrolled into 28,624 instructions, and ptxas takes 9.7 s over it. An
autotuner pays T = T_compile + T_ptxas + n · T_kernel for every candidate it
tries, so for exploration a smaller module can be the better trade.

`compile.py --explore` bounds the module: past 250,000 characters the kernel is
lowered again with its complete unrolling scaled by 1/4, then 1/16, then none,
until it fits, stopping as soon as less unrolling no longer shrinks it. On the
14 kernels past the bound, ptxas time falls from 21.7 s to 6.9 s in all (9.7 s
→ 0.14 s for the largest); ten modules shrink 7 to 48 times, while four keep
their size, which comes from fragment loops and lane blocks that must stay
unrolled. Their kernels take 1.11x as long (geometric mean of 14). The range
is wide: from 2.7x slower (`a_b8_32x32`) to 2.2x faster (`k_mlp`, 76 → 35 µs,
which the default unrolls past its best). The default, for final kernels,
stays as it was; choosing per kernel would take measurements a compile cannot
make, which is what a calibrated schedule selector would supply.

### 12.4 Coalescing counted in 32-byte sectors

The cost model that picks which loop of a nested reduction to split counts
the 128-byte lines a warp load touches (4.2). `EMMY_PTX_COST=sectors` counts
instead the 32-byte sectors it moves, element width included (a warp load of
32 f32 at stride 64 bytes: 16 lines, but 32 sectors, a quarter of the lines'
bytes). Over the 533 kernels it reaches the same decision everywhere: no
module changes. The line model, which the decisions were tuned with on the
RTX 5090, stays the default; the sector model is there for workloads that do
tell them apart.

### 12.5 One planner for repeated work at every scope (`planner.py`)

The recognisers of section 4 each remove one form of repeated work: lanes
split or block a loop, cells move or spread, clusters share across blocks.
`planner.py` looks at the repetition itself, the same way at every scope. For
each loop of a Tile, at any depth, it finds the Tile axes the loop's work
depends on (through every value defined before it), and so how many cells
compute each instance and how they lie: per warp (an aligned power-of-two run
of lanes, which the uniformity analysis of 4.1 sees; a run of another length,
which only cell placement can regroup; or a strided set), per block, and per
cluster of 8 blocks. It weighs the instructions a warp saves, C (1 − 1/R) for
a loop of cost C repeated R times (lanes run in lockstep, so repetition within
a warp costs issue slots only once), against what sharing costs at that scope:
shuffles in a warp; shared memory and a barrier in a block; cluster barriers
and distributed shared memory in a cluster. Each barrier is spread over the
instances one exchange carries, as `cluster.py` does. `--report` prints the
plan beside what the recognisers did; only the report runs the planner, so it
cannot change the PTX. For example, for `k_gelu_mlp`:
`plan: loop a3 depends on a0, a2_co; each instance is computed by 128 cells
(1 a warp, 1 a block, 8 a cluster); ... best shared at: cluster`, the choice
`cluster.py` makes.

`tools/plan_survey.py` compares the plan with the recognisers' decisions over
the 777 loops of the 533 kernels. They agree on whether to share 768 loops;
84 of these are loops the lowering blocks in order to share the inner work the
planner points at. Where sharing pays, a warp's repeating lanes form an aligned run in
83 loops and a run of another length in 13: the cases cell placement exists
for. Four loops with an estimated saving are left unshared: `m_deep_k`
(cluster), `h2a_1x512x64` (cluster), `h3a_33x77x40` (block) and one schedule
variant of a softmax (warp). Next: let the planner choose among the
recognisers, which today apply in a fixed order (clusters, then spread cells,
then placement), and widen the uniformity analysis to the strided lane sets
it describes.
