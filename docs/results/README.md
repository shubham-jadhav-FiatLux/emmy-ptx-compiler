# Raw results

The measurements behind the numbers in the README and DESIGN, as the tools wrote
them. All runs used the evaluation runner (RTX 5090, sm_120); its clocks are not
locked, so a single kernel's time can move by up to a third between runs, while
the geometric means over many problems hold to a few percent. The problem sets
are the open set, `corpus` (`tools/corpus_specs.py`), `corpus_knobs`
(`tools/corpus_knobs.py`) and the held-out sets (`tools/corpus_holdout*.py`),
generated with the organisers' tool.

| File | What | Code | Command |
|---|---|---|---|
| `nvcc_race_2026-10-26.json` | our time and nvcc's build of emmy's reference CUDA per problem (326 problems; 321 timed with both correct), registers, correctness | commit 9dfc6f3 | `tools/ref_bench.py <open> <corpus> <corpus_knobs>` |
| `suite_default_2026-10-27.json`, `suite_plain_…`, `suite_strict_…` | every problem of all five sets (473) in each lowering: pass, time, tolerance used, worst ulp | the code of commit 8b76863 | `tools/run_suite.py <sets> [--plain \| --strict]` |
| `holdout3_2026-10-28.json` | the third held-out set (39), run once after the fixes were frozen | 8b76863 | `tools/run_suite.py holdout3` |
| `fixes_race_2026-10-28.json` | the 43 problems whose PTX the fixes of 27 Oct changed, old build against new, in alternating order | 13ee1e4 against 8b76863 | (DESIGN 9) |
| `ablation_2026-10-28.json` | each optional rewrite switched off: every (problem, rewrite) pair whose PTX changes, full build against reduced build, in the order full, reduced, reduced, full (326 problems, 1659 pairs) | default PTX of 8b76863, switches of 454bd21 | `tools/ablate.py <open> <corpus> <corpus_knobs> --reps 1` |
| `compile_cost_2026-10-28.json` | per kernel (533): cold `compile.py` time, PTX size, local ptxas time for sm_120, registers; one kernel at a time | e81a819 (default output of 8b76863) | `tools/compile_cost.py <sets> --jobs 1` |
| `compile_cost_fusion_2026-10-28.json` | the same after loop fusion's legality check was made to look at each statement once: the `compile.py` times DESIGN 12.3 quotes (the PTX, so ptxas's input, is unchanged) | d7e59fc | `tools/compile_cost.py <sets> --jobs 1` |
| `compile_cost_explore_2026-10-28.json`, `explore_kernel_time_2026-10-28.json` | the 14 modules past 250,000 characters under `--explore`: compile cost, and kernel time against the default | e81a819 (before cecbf4c's early stop, which only changes which module is kept for the four that do not shrink) | `tools/compile_cost.py <dirs> --flags=--explore` |
| `plan_survey_2026-10-28.json` | every loop of the 533 kernels: the planner's best scope and estimated saving, and what the lowering did | 5d28a92 | `tools/plan_survey.py <sets>` |
| `plan_survey_2026-10-29.json` | the same with the flow-following dependences (the same plans) and each loop's kernel and enclosing loops, from which the tool counts the 85 loops shared for a loop inside them | this commit | `tools/plan_survey.py <sets>` |
