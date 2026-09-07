# Wave 74 - compensated-MMA AV production integration gate

## Outcome

Wave 74 audits the successful Wave 71 T4 feasibility result against the actual
production dataflow. It changes no kernel or dispatcher. The direct result is a
valid license to integrate, but it is not yet a production speed claim and it
cannot be pasted into `gl_attn_mma4_regq_fused_f32` unchanged.

The production candidate will keep K row-major and change only the physical V
cache layout from `[head, seq, dim]` to `[head, dim, seq]`. The allocation has
the same number of f32 elements, so this neither duplicates the KV cache nor
adds a hot-path allocation. All writers and readers selected for that layout
must change together behind one auditable opt-in. The existing row-major path
remains the fallback and A/B baseline.

## Reported T4 evidence

The user ran the corrected Wave 71 notebook on a Tesla T4. The reported archive
is `/kaggle/working/glcuda-t4-wave71-mma-av-results.zip`, with reported SHA-256
`f3058dde7202e9828138ef72215ad47c2f7e25d157b5d190534ca07900d1f5a3`.
That archive has not been ingested into this checkout, so this section records
reported output rather than claiming locally verified archive provenance.

| capacity | scalar AV | MMA AV | speedup | max absolute error |
|---:|---:|---:|---:|---:|
| 1 | 3.236 us | 2.942 us | 1.0999x | 0 |
| 17 | 11.767 us | 4.507 us | 2.6110x | 1.341104507e-7 |
| 241 | 231.168 us | 73.926 us | 3.1270x | 3.390014172e-7 |
| 244 | 237.584 us | 66.880 us | 3.5524x | 3.799796104e-7 |

`ptxas -v` reported 44 registers for the MMA entry and 25 for the scalar
entry, with zero stack, spills, and barriers for both. The aligned and ragged
numeric gates pass by more than an order of magnitude. Capacity 1 correctly
shows that launch overhead dominates tiny work and is not a dispatch regime
for the candidate.

## Why the probe is not drop-in production code

The retained KV cache and every current attention path use
`[layer, K/V, head, seq, dim]`. The Wave 71 candidate instead consumes V as
`[head, dim, capacity]`; this makes the two adjacent K values required by an
MMA B fragment contiguous. Pointing it at the retained row-major V cache would
produce wrong values.

A transient transpose in an existing prefill workspace is not sufficient.
`pf_gate` and `pf_up` are large enough to hold one transposed layer, but they
are overwritten by that layer's FFN. On a later prefill chunk the attention
needs every earlier V row for the same layer, after intervening layers have
reused the scratch. Rebuilding the entire prefix for every layer and chunk
would add a transpose launch plus a complete V read and write immediately
before the timed attention kernel. Keeping a second permanent V image avoids
that work but adds another full V cache and violates the fixed-memory design.

Changing only V's physical layout avoids both costs. K remains row-major for
the retained compensated-MMA QK path. V uses the already allocated element
count, addressed as `[layer, head, dim, max_context]`; the stride is the cache
capacity, not the current score capacity. `gl_kv_write_rows` needs a matching
V-write variant, and prefill plus decode AV readers must use the same layout.
The semantic cache shape and cursor do not change.

## Frozen Wave 75 implementation boundary

Wave 75 is an opt-in production candidate, not a default flip.

1. Add one process-wide V-layout decision owned by the CUDA kernel/attention
   policy. The writer and every reader derive from that one decision; no
   independent flags may create a mixed layout.
2. Keep the allocation size unchanged. Do not add a second V cache, a global
   score matrix, per-request allocation, or a prefix-transpose launch.
3. Add a dimension-major V writer for batched prefill and the single-token
   decode write. Its address contract is
   `head * head_dim * max_context + dim * max_context + position`.
4. Replace only the AV section of `gl_attn_mma4_regq_fused_f32`. Preserve its
   compensated QK instruction order, causal softmax, register-Q schedule,
   16-row grid, score-capacity fallback, and packed f32 output.
5. Adapt decode AV to the selected V layout. This is mandatory even though the
   target is prefill: generation immediately consumes the cache written by
   prefill, and a layout mismatch is silent numerical corruption.
6. Fully predicate both values of every final K8 V fragment. The Wave 71 probe
   predicates probability tails but its V loads at `ptr` and `ptr+4` assume a
   padded dimension-major fixture. Production must not read beyond
   `max_context`, including capacity 1, 17, 241, 244 and a final cache position.
7. Preserve the row-major implementation and default as the controlled
   baseline until production retention passes.

## Required gates

- Host tests and the complete CUDA parity executable must pass. New device
  parity covers prefill and decode at capacities 1, 17, 241, 244, a nonzero
  `pos_base`, a chunk boundary, and the last valid cache position.
- Maximum absolute attention error remains at or below `1e-5`; finite output
  is mandatory and the tolerance is not relaxed.
- `ptxas -v` must report zero stack and zero spills. Registers, shared memory,
  and active blocks per SM are recorded, but occupancy is not inferred from
  register count alone.
- A direct fused-attention A/B includes the V-write cost. The candidate must
  retain at least 1.10x median attention speedup at 244 tokens before spending
  a production run on it.
- Production truth is Q8_0 `glbench` on the pinned 244-token prompt and model,
  with position-balanced baseline/candidate runs, teacher-forced logits
  characterization, and decode/tail gates. The later GwenLand/llama.cpp final
  comparison remains Q8 versus Q8 with ten runs per arm.
- No default flip unless every paired production prefill result is positive
  and the median improvement clears 5%. The 15,000 tok/s objective is complete
  only when the measured production median reaches it.

## Honest ceiling from the direct result

The valid rows decomposition attributed 22.2% of its attention kernel to AV.
Applying the 3.5524x direct AV speedup to that fraction gives an optimistic
attention-kernel reduction of 15.95%. On Wave 58's 4.064 ms traced attention
core, that is at most about 0.648 ms before integration overhead.

| reference | measured latency | AV-only optimistic latency | throughput |
|---|---:|---:|---:|
| Wave 58 traced stack | 20.260 ms | 19.612 ms | about 12,441 tok/s |
| Wave 57 Q8 production | 21.218 ms | 20.570 ms | about 11,862 tok/s |
| 15k target | 16.267 ms | - | 15,000 tok/s |

This is an inference across diagnostics, not a measured production result. It
also uses the rows decomposition because the qk4 stop-0 probe failed its own
reproduction gate. The direct scalar fixture's 237.584 us must not be
subtracted from a different production kernel.

Even the optimistic AV-only ceiling leaves roughly 3.345 ms versus the traced
stack, or 4.303 ms versus the Wave 57 production median, still to remove. AV is
worth integrating if its fused net gain survives, but it cannot honestly be
the sole route to 15k. A later wave must return to the remaining QK and FFN
compute after this candidate has a production number.

## Decision

**IMPLEMENTATION LICENSED, PRODUCTION CLAIM WITHHELD.** Wave 71 establishes a
strong, numerically clean SM75 AV mechanism. Wave 74 establishes the only
fixed-memory production layout that preserves its contiguous MMA operand
without rebuilding or duplicating V. Wave 75 may implement that opt-in path;
only production measurement can decide retention.
