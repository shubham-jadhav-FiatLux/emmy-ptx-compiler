# Examples

Three problems, with the PTX the compiler makes for each (`*.ptx`, next to the
IR). Every folder is a complete problem (IR, `meta.json`, reference CUDA,
expected outputs), so it can be compiled and checked on the runner:

```bash
python compile.py examples/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
python tools/run_suite.py examples/r01_array_sum examples/a01_attention examples/m_f16_128 --bench
```

`r01_array_sum` and `a01_attention` are from the open problem set;
`m_f16_128` is from the corpus (`tools/corpus_specs.py`). The excerpts below
are cut from the `.ptx` files, with comments added.

## 1. A reduction: `r01_array_sum`

The IR: each of 128 threads strides over 1024 floats, then a warp butterfly,
the warps' partials in shared memory, a barrier and a halving tree.

```
Tile[a0_co] (N=128)
    for a0 in a0_co..1024:128
        in0 = load x[a0]
        f32 acc0 <- add(acc0, in0)
    WarpShuffle(acc0, length=32)
    Smem float acc0_smem[4]
    if ((lane == 0)):
        acc0_smem[warp] = acc0
    Sync
    TreeHalve(acc0, length=4, tid=warp)
    if ((a0_co == 0)):
        sum_1[] = acc0
```

The strided loop is regrouped (DESIGN 4.6): each thread's partial is never
seen alone, so the threads take contiguous 16-byte runs instead, and four
scalar loads become one vector load:

```ptx
	ld.global.nc.v4.f32 	{%f2, %f3, %f4, %f5}, [%rd6];         // x[4t .. 4t+3]
	add.rn.f32 	%f6, %f2, %f1;
	...
	ld.global.nc.v4.f32 	{%f10, %f11, %f12, %f13}, [%rd6+2048]; // x[512+4t ..]
```

The halving tree takes one barrier instead of one per step (DESIGN 4.7):
every thread reads the four partials and adds them exactly as the tree would
pair them, `(s0 + s2) + (s1 + s3)`:

```ptx
	bar.sync 	0;
	ld.shared.f32 	%f28, [smem_acc0_smem];
	ld.shared.f32 	%f29, [smem_acc0_smem+4];
	ld.shared.f32 	%f30, [smem_acc0_smem+8];
	ld.shared.f32 	%f31, [smem_acc0_smem+12];
	add.rn.f32 	%f33, %f30, %f32;     // s0 + s2
	add.rn.f32 	%f35, %f31, %f34;     // s1 + s3
	add.rn.f32 	%f38, %f37, %f36;     // the root
	bar.sync 	0;
```

## 2. Attention: `a01_attention`

`softmax(q @ kt / 8) @ v` with 64 queries and 64 keys of dimension 64, fused
by emmy into one loop over the keys. Each lane holds four query rows and two
output columns, so the 16 lanes that share those rows would all compute the
same 4 x 64 scores. Lane blocking (DESIGN 4.3) has them take the keys of each
block of 16 in turn. Each lane computes its key's dot products with the four
rows:

```ptx
	fma.rn.f32 	%f883, %f299, %f881, %f700;   // q[r, :] . kt[:, key], 4 elements a load
	...
	mul.rn.f32 	%f893, %f57, 0f3E000000;      // the scale, 1/8, for each of the four rows
	mul.rn.f32 	%f896, %f700, 0f3E000000;
```

Each row's block maximum then needs one butterfly over the 16 lanes, and the
online softmax is rescaled once per block instead of once per key. The two
rescaling factors `exp(m - max)` and `exp(s - max)` come from a single `exp`
(DESIGN 5):

```ptx
	shfl.sync.bfly.b32 	%f898, %f897, 8, 31, %r40;
	max.f32 	%f899, %f898, %f897;
	...                                         // distances 4, 2, 1
	ex2.approx.ftz.f32 	%f917, %f916;          // inside exp, on |f| <= 1/2
	selp.f32 	%f920, 0f3F800000, %f919, %p1;  // the factor that is exp(0) = 1
```

Every lane needs every key's weight for its own output columns. The weights
go through a small shared buffer between two warp barriers, which costs far
less than 16 shuffles; ptxas merges the reads into 128-bit loads:

```ptx
	bar.warp.sync 	%r68;
	st.shared.f32 	[%r76], %f931;      // this lane's key weight, for each query row
	...
	bar.warp.sync 	%r68;
	ld.shared.f32 	%f1053, [%r74];     // key 0's weight
	fma.rn.f32 	%f1058, %f25, %f1053, %f2;   // o[r, c] += v[key, c] * weight
```

The kernel runs in 8.7 µs; nvcc's build of emmy's reference CUDA takes 74 µs.

## 3. Tensor cores: `m_f16_128`

A 128 x 128 x 128 float16 matmul, split along K into two kernels (the
partial products, then their sum). Emmy schedules it as CUTLASS would: tiles
of A and B are copied into swizzled shared memory with `cp.async`, double
buffered; `ldmatrix` fills each warp's register fragments; `mma.sync` does a
16 x 8 x 16 product per instruction (DESIGN 6):

```ptx
	cp.async.cg.shared.global 	[%r36], [%rd8], 16;         // the next tile, 16 bytes a lane
	cp.async.commit_group;
	cp.async.wait_group 	1;                                // the current tile has landed
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 	{%r15, %r16, %r17, %r18}, [%r85];        // A, 16 x 16
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 	{%r19, %r20, %r21, %r22}, [%r94];  // two B fragments
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 	{%f1, %f2, %f3, %f4}, {%r15, %r16, %r17, %r18}, {%r19, %r20}, {%f1, %f2, %f3, %f4};
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 	{%f5, %f6, %f7, %f8}, {%r15, %r16, %r17, %r18}, {%r21, %r22}, {%f5, %f6, %f7, %f8};
```

Each lane then stores its four accumulator elements, a row's two neighbouring
columns as one 8-byte store.
