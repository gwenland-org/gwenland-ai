# gl-agents-skills audit after T4 Ceiling Waves 4-10

**Date:** 2026-08-26  
**Scope:** all 37 files under `.agents/skills/*/SKILL.md`  
**Method:** repository audit + original Wave 4-10 reports + primary-source research  
**Deliverable type:** recommendations only; no skill or runtime code was changed

## Executive result

The collection has good first principles, but it does not yet encode the failure
modes that dominated Waves 4-10. The highest-risk gap is not another PTX trick;
it is the definition of proof. Wave 7 passed a new kernel's internal reference
test and gained 23%, yet matched the production `glproc` oracle for only 9/50
greedy tokens. Wave 5 reduced barriers but raised resources and gained only
0.49%. Wave 6's wider tile was numerically correct but lost 13.77%. Wave 8 and
Wave 9 each looked useful in isolation but remained below the 5% production
gate; their combination in Wave 10 still reached only +3.64%.

Four cross-cutting repairs should be made before editing individual prose:

1. **Repair navigation.** A local link check found **223 broken Markdown
   targets across all 37 skills**. The skill tree is flat now, but links still
   assume old category directories such as `../cuda-skills/`; repository links
   are also generally one directory too shallow. This makes the prescribed
   reading order impossible to follow.
2. **Define three separate correctness gates.** Require (a) candidate versus
   retained-kernel bit/reference checks where arithmetic is claimed unchanged,
   (b) hardware parity that proves CUDA actually ran, and (c) exact deterministic
   production-oracle parity for every measured arm. None substitutes for the
   others.
3. **Make compiled resources authoritative.** Parse per-entry `ptxas` output for
   registers, static shared memory, spill stores, and spill loads; combine it
   with launch threads and actual dynamic shared bytes for the tested shape.
   Source declarations and guessed occupancy tiers are hypotheses, not gates.
4. **Standardize paired production experiments.** Pin the artifact stack,
   model bytes, workload, seed, engine, and iteration counts; sanitize arm
   environment variables; interleave arms on one allocation; require every pair
   to clear the chosen throughput threshold without tail/decode regression.
   Microbenchmarks and singleton telemetry remain diagnostic.

These changes are supported both by the project incidents and by NVIDIA's
documentation. Turing occupancy depends on finite registers, warps, blocks, and
shared memory, while NVIDIA explicitly warns that occupancy is not itself a
performance result ([Turing Tuning Guide](https://docs.nvidia.com/cuda/archive/12.0.1/turing-tuning-guide/index.html)). NVIDIA documents compiler resource
reports and spill warnings as the relevant compiled evidence
([CUDA Programming Guide: compiler resource usage](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)). The PTX ISA fixes the exact
per-lane fragment layout for `mma.m8n8k16`, so fragment/layout preservation must
be tested rather than inferred
([PTX ISA: `mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)).

## Evidence ledger

| ID | Observed result | Lesson that is safe to encode |
|---|---|---|
| W4 | Replacing a 16,420 B static attention score array with shape-sized dynamic shared memory (1,012 B at 244 tokens) improved production P50 by 14.34%. | Resource accounting must include static versus launch-time dynamic shared memory at the real workload shape. |
| W5 | A manual A/xscale double buffer cut the barrier schedule from `2*nb` to `nb+1`, but moved 56 regs/3,328 B to 63 regs/6,656 B and gained only 0.49%. | Fewer barriers or more overlap is not a result; resource cost and production timing decide. |
| W6 | A correct 128-row tile lost 13.77% production P50 and regressed P95 by about 15.8%. | Larger tiles/reuse can lose through traffic, grid coverage, latency, or resource pressure; do not promote static occupancy math to a verdict. |
| W7 | W8PC gained 23.00% but achieved only 9/50 exact greedy-token agreement, despite hardware kernel-reference correctness passing. | A changed numerical contract needs full legacy/oracle proof; an internally consistent new reference can validate the wrong semantics. |
| W8 | Row-CTA quantization was bit-identical and 51.1% faster in isolation, but production improved only 3.31%. | Launch-count and microbenchmark wins are diagnostic until production clears the gate. |
| W9 | L2 rastering preserved resources and arithmetic; one GEMM shape improved 6.36%, but production improved only 3.40%. | CTA order is a hint, not a scheduling guarantee; isolated shape results cannot retain it. |
| W10 | Same-allocation 2x2 testing measured +2.33%, +1.55%, and +3.64%; all remained HOLD. Absolute rates differed across Kaggle allocations. | Compare candidates inside one allocation. A positional Latin square is not automatically carryover-balanced, and factorial effects must use all four cells correctly. |

The Wave 10 ordering issue is grounded in standard design-of-experiments
terminology: a Latin square balances position, while first-order carryover
requires every treatment to precede every other equally often
([Penn State STAT 509](https://online.stat.psu.edu/stat509/Lesson12),
[Williams 1949, DOI 10.1071/CH9490149](https://doi.org/10.1071/CH9490149)).

## Per-skill audit

“No wave-specific addition” is deliberate. It means Waves 4-10 did not provide
evidence for changing that domain; inventing a rule would violate the requested
evidence boundary. Every row still inherits the global broken-link repair.

| Skill | Missing | Vague or stale | What to add or change | Evidence |
|---|---|---|---|---|
| `avx2-simd` | Nothing demonstrated by these CUDA waves. | Its cross-links are unusable after the directory flattening. | Repair links only; do not import T4 conclusions into CPU SIMD policy. | Link audit; scope boundary |
| `backend-independence` | A rule that experimental GPU formats/contracts cannot redefine the cross-engine oracle. | “Parity oracle” does not say whether a candidate-created reference is acceptable. | State that `glproc` semantics remain fixed during an A/B; a candidate may have an internal reference only as an additional test. | W7 |
| `branch-strategy` | Clean reconstruction of retained and candidate stacks for performance waves. | “Branch per idea” does not specify how to exclude rejected/HOLD patches or protect a dirty user tree. | Require detached clean worktrees or equivalent immutable trees, record base revision and patch hashes, and assert rejected-wave markers are absent. | W5-W10 candidate composition |
| `check-existing-tests` | Baseline output/artifact capture and a proof that GPU tests did not self-skip. | “Run tests before and after” can be satisfied by a locally green suite with no CUDA execution. | Add baseline archive, hardware identity, executed-versus-skipped counts, candidate/legacy bit tests, and production oracle checks. | W7-W10 |
| `cuda-graphs` | No graph-specific rule was falsified in Waves 4-10. | Cross-links are broken. | Repair links only. Keep graph replay in the common hardware-correctness gate. | W4-W10 correctness suites |
| `dequant-path` | A boundary between format-local correctness and end-to-end semantic correctness. | “Parity” can be read as parity against a newly written dequant reference. | Require retained-format byte/scale-bit parity when a scheduling-only change claims unchanged Q8_0 arithmetic; changed scale contracts require full `glproc` production parity. | W7, W8 |
| `descriptor-sets` | No Wave 4-10-grounded Vulkan addition. | Cross-links are broken. | Repair links only. | Scope boundary |
| `dynamic-loading` | No loader-specific failure occurred in these waves. | It should reference the shared “hardware actually executed” assertion, but this is not unique to loading. | Repair links; point hardware tests to the common executed-versus-skipped contract once added. | W4-W10 |
| `error-handling` | Benchmark/notebook failure preservation. | It does not say how gates should fail while retaining logs and partial results. | Add: fail with the original cause, always write a partial manifest/report/archive in `finally`, and never let reporting cleanup replace the root error. | Repeated structural/resource gate failures during Wave 4-10 notebooks |
| `fallback-chain` | No fallback behavior was changed by these waves. | “Soft failure” is unrelated to candidate validation. | Repair links only; do not let fallback turn a requested CUDA benchmark into a silent CPU success. | W4-W10 engine locks |
| `format-parsing` | No parser defect was observed in Waves 4-10. | Cross-links are broken. | Repair links only. | Scope boundary |
| `gate-integration` | A machine-readable dispatch contract and arm environment isolation. | Source substring markers are treated as structural proof even though equivalent code formatting can break them. | Require runtime banners/telemetry naming the selected kernel and flags, clear all optimization env vars before each arm, then set an explicit allowlist; use AST/compiled-symbol checks where practical. | W4 structural-marker mismatch; W8-W10 dispatch arms |
| `glbench-usage` | A canonical GPU optimization A/B protocol. | Current iteration guidance is generic and does not define paired retention. | Add same-allocation interleaving, pinned GGUF SHA/revision/prompt/seed, exact engine/workload/iteration assertions, two or more paired sessions, P50+mean threshold, P95/decode non-regression, and exact oracle prefixes per session. | W4-W10 |
| `glcore-rules` | Nothing Wave-specific. | The architecture it points to contains obsolete GPU claims, including planned rather than shipped Tensor Core policy. | Repair links and mark historical architecture sections as superseded; do not change glcore’s zero-compute rule. | Current repo versus Wave 4-10 |
| `inference-first` | An explicit statement that fast but semantically divergent tokens are zero performance value. | “Correctness first” lacks a concrete production gate. | Add the Wave 7 counterexample and require performance tables to be invalidated when oracle parity fails. | W7 |
| `kernel-design` | Shape-dependent dynamic-shared accounting, grid-coverage checks, and distinction between internal reference and oracle. | Static stage shares and “prefill is compute-bound” are too absolute; Wave 4-10 stage shares moved materially. | Replace fixed percentages with dated workload measurements; add resource tuple `(regs, static smem, dynamic smem(shape), threads, spills)`, CTA/grid coverage, traffic-per-output, and three-level correctness gates. | W4-W10 |
| `measurement-discipline` | Paired multi-arm order design and correct factorial arithmetic. | “Interleave/reverse” is insufficient for four arms and does not address allocation drift or carryover. | Require one-allocation comparisons; for four sequential arms use a first-order carryover-balanced Williams design or randomization justified by the threat model. Compute 2x2 marginal effects from all four cells and report interaction. | W10; Williams design sources |
| `memory-bandwidth` | No CPU-bandwidth rule follows from these GPU waves. | Cross-links are broken. | Repair links only; do not generalize the T4 prefill result to CPU decode. | Scope boundary |
| `memory-management` | Dynamic shared memory as a launch-time resource contract. | “Predictable fixed allocation” can be misread as requiring maximum static scratch per CTA. | Distinguish persistent VRAM allocation from per-launch shared scratch. Pass exact required dynamic bytes, validate capacity/fallback, and include it in occupancy calculations. | W4 |
| `memory-safety` | Bounds/capacity validation for dynamically partitioned shared scratch. | It covers host RAM but not the host-to-kernel dynamic-smem size contract. | Add checked size arithmetic and tests at short prompt, maximum context, and over-capacity fallback; do not infer safety from PTX static declarations. | W4 |
| `moe-loading` | No MoE lesson occurred in these waves. | Cross-links are broken. | Repair links only. | Scope boundary |
| `pipeline-barriers` | No Vulkan-specific lesson. | It says fusion is the endgame too categorically; T4 showed occupancy and launch geometry can dominate before fusion. | Repair links; soften cross-backend performance analogy unless measured on Vulkan. | W4, W8 |
| `portability` | No Vulkan portability evidence from T4-only experiments. | Cross-links are broken. | Repair links only; explicitly label T4 conclusions NVIDIA-only when referenced. | Scope boundary |
| `ptx-writing` | Standalone assembly/resource verification and scheduling-only invariants. | It incorrectly implies driver JIT is the only useful PTX validation point and says “per-tensor parity” without exact unchanged-path checks. | Require offline `ptxas -arch=sm_75` per module, parsed per-entry resources/spills, ASCII/LF validation, instruction/barrier/layout invariants for scheduling-only edits, hardware execution, and production oracle parity. | W4-W10; NVIDIA compiler/PTX docs |
| `quantization` | GPU activation/weight scale-contract changes need an oracle warning even though this skill is CPU-scoped. | The CPU/GPU separation note does not point to the Wave 7 semantic failure. | Keep CPU rules unchanged; add one cross-reference warning that a new GPU quant contract cannot validate itself. | W7 |
| `quantization-types` | Distinguish storage-layout parity from inference-semantic parity. | “Fast path parity-tests against scalar reference” is insufficient when both implement a newly changed contract. | Add retained byte/scale-bit tests plus end-to-end oracle parity whenever scale granularity, rounding, or fold order changes. | W7, W8 |
| `rca-interpretation` | Amdahl ceilings must be labeled optimistic and invalidated by changed stage mix. | Share-based “infinite-X” projections can look like feasibility proofs. | Add that stage shares are arm/workload/session-specific; use projections to prioritize experiments only. Recompute after retained waves and never claim a hardware ceiling from them. | W4-W10 changing shares and unmet 15k |
| `read-architecture-first` | A precedence rule for stale architecture versus newer retained measurements. | It treats architecture documents as ground truth even where `ArchGLML_X2.md` still describes Tensor Cores as future work and wrong architecture floors. | Add: read dated wave reports/changelog after architecture, flag contradictions, and prefer retained measured state while opening a docs update. | Current X2 versus W4-W10 |
| `rejected-optimizations` | GPU/T4 rejection ledger. | It says patterns generalize but contains only CPU entries, so Wave 5/6/8/9/10 can be accidentally restacked. | Add tier-scoped entries: Wave 5 double buffer rejected (+0.49%, higher resources); Wave 6 r128 rejected (-13.77%, tail regression); Waves 8/9/10 HOLD below 5%; Wave 7 invalid due to parity, not a valid performance rejection. | W5-W10 |
| `spirv-writing` | No SPIR-V evidence occurred. | CUDA analogy is stated as fact and links are broken. | Repair links; retain analogies only as hypotheses to test on Vulkan. | Scope boundary |
| `tensor-cores` | Compiled resource gates, traffic/grid analysis, and exact legacy parity for “arithmetic unchanged.” | “A/B harness is the arbiter” is too narrow; Wave 5/6 microbench results disagreed with production, and “per-tensor tolerance” missed Wave 7’s semantic drift. | Make production glbench the retention arbiter; require PTXAS tuple/spills, actual dynamic smem, bit-identical candidate-vs-retained output for scheduling changes, full oracle for arithmetic changes, and a T4 rejected/HOLD table. | W5-W10 |
| `testing-standards` | Exact deterministic sequence parity and negative proof of hardware skip. | Per-operation tolerances alone do not protect greedy argmax boundaries or changed reference semantics. | Add two test classes: numerical tensor tolerance and exact next-token/prefix parity under pinned greedy settings. Archive device identity and fail if a required CUDA suite skips. | W7-W10 |
| `threading-model` | No CPU-threading lesson occurred. | Cross-links are broken. | Repair links only. | Scope boundary |
| `trait-design` | No trait change was exercised. | Cross-links are broken. | Repair links only. | Scope boundary |
| `unsafe-rules` | No new unsafe-code failure occurred. | Cross-links are broken. | Repair links only. | Scope boundary |
| `wave-confirmation-gates` | A formal retained-baseline ledger and hold/reject semantics. | “One lever per wave” does not say that HOLD patches must not be stacked or that a positive-but-subthreshold result remains unretained. | Record `RETAIN`, `HOLD`, `REJECT`, and `INVALID` separately; every new wave reconstructs only retained patches unless explicitly testing interaction; archive patch hashes and assert excluded markers absent. | W5-W10 |
| `windows-defender-gotcha` | Nothing from Kaggle/T4 Waves 4-10. | Cross-links are broken. | Repair links only. | Scope boundary |

## Exact additions recommended for the high-impact skills

The table above is the complete audit. The following text-level contracts are
the minimum coherent patch set; they prevent different skills from inventing
slightly different meanings for the same gate.

### Shared correctness contract

Add to `testing-standards`, then reference it from `ptx-writing`,
`tensor-cores`, `kernel-design`, `dequant-path`, and `quantization-types`:

> A GPU optimization has three independent correctness gates. First, when it
> claims unchanged arithmetic, candidate output must be bit-identical to the
> retained kernel for targeted edge and production shapes. Second, the full
> hardware suite must prove the requested GPU executed; a skip is a failure for
> a hardware gate. Third, every deterministic production session must match the
> pinned `glproc` greedy token prefix exactly. A candidate-authored scalar or
> kernel reference is useful, but cannot replace the retained implementation or
> cross-engine oracle.

### Shared PTX resource contract

Add to `ptx-writing`, then reference it from `kernel-design` and
`tensor-cores`:

> Assemble every shipped PTX module for the exact target with offline `ptxas`.
> Parse resources per entry point: registers/thread, static shared bytes, spill
> stores, and spill loads. Combine those values with block threads and the
> actual dynamic shared bytes passed for each gated workload shape. Treat source
> declarations and occupancy spreadsheets as predictions; only compiled
> resources plus production measurement are retention evidence.

This follows NVIDIA's documented resource-report/spill facilities and avoids
the erroneous source-level resource gates encountered during the sprint. If
Nsight Compute reports `ERR_NVGPUCTRPERM`, record counters as unavailable—not
zero—and retain the compiled/static evidence; NVIDIA defines that error as a
permissions failure
([Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)).

### Shared performance-retention contract

Add to `measurement-discipline`, then reference it from `glbench-usage`,
`wave-confirmation-gates`, `kernel-design`, and `tensor-cores`:

> A candidate is retained only from paired production sessions on one device
> allocation with immutable model/workload/seed/build inputs. Clear all
> optimization flags before setting the arm allowlist. Each required pair must
> clear the declared P50 and mean threshold; P95 latency and decode throughput
> must remain within their declared non-regression bounds. Exact oracle parity
> and compiled-resource gates must stay green. Microbenchmarks, launch counts,
> occupancy projections, Amdahl ceilings, and singleton telemetry explain a
> result but cannot retain a patch.

### Wave-state contract

Add to `wave-confirmation-gates` and `rejected-optimizations`:

> `RETAIN` may become the next baseline. `HOLD` is correct and promising but
> below the production threshold; it is not silently stackable. `REJECT` is a
> valid but negative experiment. `INVALID` has failed correctness or experiment
> integrity, so its speed is not evidence of a shippable optimization. Every
> candidate manifest lists the exact retained patch stack and asserts that
> excluded wave markers are absent.

## Repository hygiene verification

- 37/37 `SKILL.md` files were decoded successfully as UTF-8.
- The apparent smart-character corruption in PowerShell output is terminal
  rendering, not mojibake stored in the files.
- 223/223 discovered local Markdown targets failed resolution from their
  containing skill file. This is a systematic migration defect, not isolated
  typos. Fix paths and add a link-check test before relying on `Related Skills`
  or `BEFORE YOU START` navigation.
- The audit intentionally proposes no Wave 4-10-derived changes to CPU SIMD,
  MoE, Rust unsafe, traits, Windows Defender, or Vulkan implementation policy;
  those waves did not test those mechanisms.

## Recommended implementation order

1. Fix all local links and add a link checker.
2. Patch the shared correctness contract into `testing-standards` and its five
   consumers.
3. Patch PTX resource and production-retention contracts.
4. Add the four-state wave ledger and T4 Wave 5-10 entries.
5. Update stale `ArchGLML_X2.md` statements or label them historical so
   `read-architecture-first` no longer routes agents to superseded facts.
6. Only then tune the remaining domain-specific prose.

That sequence addresses the failures with the greatest ability to invalidate
an entire sprint before adding lower-risk detail.
