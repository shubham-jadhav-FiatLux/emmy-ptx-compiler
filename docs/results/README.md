# Raw results

The measurements behind the numbers in the README and DESIGN, as the tools wrote
them. All runs used the evaluation runner (RTX 5090, sm_120); its clocks are not
locked, so a single kernel's time can move by up to a third between runs, while
the geometric means over many problems hold to a few percent. A time is the
runner's median of 20 launches after 3 warm-ups; where a tool repeats a run
(`ref_bench.py`, `ablate.py`) it keeps the best of those medians. No tool keeps
every launch. The problem sets
are the open set, `corpus` (`tools/corpus_specs.py`), `corpus_knobs`
(`tools/corpus_knobs.py`) and the held-out sets (`tools/corpus_holdout*.py`),
generated with the organisers' tool.

| File | What | Code | Command |
|---|---|---|---|
| `nvcc_race_2026-10-26.json` | our time and nvcc's build of emmy's reference CUDA per problem (326 problems; 321 timed with both correct), registers, correctness | commit 9dfc6f3 | `tools/ref_bench.py <open> <corpus> <corpus_knobs>` |
| `suite_default_2026-10-27.json`, `suite_plain_…`, `suite_strict_…` | every problem of all five sets (473) in each lowering: pass, time, tolerance used, worst ulp | the code of commit 8b76863 | `tools/run_suite.py <sets> [--plain \| --strict]` |
| `holdout3_2026-10-28.json` | the third held-out set (39), run once after the fixes were frozen | 8b76863 | `tools/run_suite.py holdout3` |
| `fixes_race_2026-10-28.json` | the 43 problems whose PTX the handling of the edge cases (DESIGN 9) changed, old build against new, in alternating order: 42 timed, and `h2a_f16_64x64x32`, a generator fault, an error on both | 13ee1e4 against 8b76863 | (DESIGN 9) |
| `ablation_2026-10-28.json` | each optional rewrite switched off: every (problem, rewrite) pair whose PTX changes, full build against reduced build, in the order full, reduced, reduced, full (326 problems, 1659 pairs) | default PTX of 8b76863, switches of 454bd21 | `tools/ablate.py <open> <corpus> <corpus_knobs> --reps 1` |
| `compile_cost_2026-10-28.json` | per kernel (533): cold `compile.py` time, PTX size, local ptxas time for sm_120, registers; one kernel at a time | e81a819 (default output of 8b76863) | `tools/compile_cost.py <sets> --jobs 1` |
| `compile_cost_fusion_2026-10-28.json` | the same after loop fusion's legality check was made to look at each statement once: the `compile.py` times DESIGN 12.3 quotes (the PTX, so ptxas's input, is unchanged) | d7e59fc | `tools/compile_cost.py <sets> --jobs 1` |
| `compile_cost_explore_2026-10-28.json`, `explore_kernel_time_2026-10-28.json` | the 14 modules past 250,000 characters under `--explore`: compile cost, and kernel time against the default | e81a819 (before cecbf4c's early stop, which only changes which module is kept for the four that do not shrink) | `tools/compile_cost.py <dirs> --flags=--explore` |
| `plan_survey_2026-10-28.json` | every loop of the 533 kernels: the planner's best scope and estimated saving, and what the lowering did | 5d28a92 | `tools/plan_survey.py <sets>` |
| `plan_survey_2026-10-29.json` | the same with the flow-following dependences (the same plans) and each loop's kernel and enclosing loops, from which the tool counts the 85 loops shared for a loop inside them | b5e2d49 | `tools/plan_survey.py <sets>` |
| `icache_2026-10-29.json` | the instruction cache: one block of 256 threads runs a loop of N FMAs (4 to 256 KB of machine code), about two million instructions a warp; time per instruction against the loop's size | no compiler code | `tools/icache.py --json …` |
| `copies_race_2026-10-29.json` | the 7 kernels whose complete unrollings copy a rolled loop, with those unrollings refused (the rule without its size condition) against the build of af034de, in the order old, new, new, old: `k_mlp` 0.46x the time, `a_d80` 1.47x, the others 0.997 to 1.06x. Only `k_mlp` meets the size condition; its module there is byte-identical to the one the default now gives | af034de against that experiment | (DESIGN 3) |
| `copies_sweep_2026-10-29.json` | `tests/kernels.short_loop_around_a_long_one` (k = 128, one block of 256 threads) at 8 to 64 copies of its rolled loop against none: PTX and SASS size, loads, registers and time of each. The same time either way at every size (0.96 to 1.00) | the build of this commit | (DESIGN 3) |
