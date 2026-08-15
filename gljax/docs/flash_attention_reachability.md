# FlashAttention reachability — what gljax can and cannot reach, and why

Status: design note, ARTX09 Wave A9.2. No runtime probe. This environment has
neither a GPU nor a CUDA PJRT plugin (`gljax/README.md`: no PJRT plugin at all
on Windows), so nothing below has been measured — every claim is either cited
directly from `gljax/architecture/ARTX09-attention-and-memory-architecture.md`
or is a structural fact checked against gljax's actual `ops::attention` code.
Read this alongside that document, not instead of it.

## The finding that sets the ceiling (ARTX09 §1.2)

> XLA can fuse operations and schedule instructions, but it cannot rewrite
> your algorithm. It cannot infer the streaming/online-softmax reformulation.

FlashAttention is a different algorithm, not a different kernel for the same
one — unlike ARTX08's GEMM story, where `dot_general` is the same computation
regardless of which kernel XLA picks. gljax emitting the natural
`softmax(QKᵀ/√d)·V` graph materializes the `[B, H, S_q, S_kv]` score tensor;
no amount of "better" StableHLO changes that on its own.

## Per-backend reachability (ARTX09 §3)

| Backend | Flash path | gljax reachable? | Mechanism |
|---|---|---|---|
| **CUDA** | cuDNN fused MHA | ✅ | `CudnnFusedMHARewriter` (`--xla_gpu_enable_cudnn_fmha=true`) pattern-matches the `bmm1 → softmax → bmm2` HLO shape and rewrites it to the `__cudnn$fmha` custom-call target. gljax never emits `custom_call` itself — it emits the ordinary attention graph and the rewriter fires (or doesn't) inside XLA. |
| **TPU** | Splash Attention | ❌ | Written in Pallas, lowered through Mosaic. Reaching it means authoring a Pallas kernel, which is exactly what ARTX08 declined to do. This is a real capability gap on gljax's flagship target (TPU v5e), not a footnote — recorded as an explicit open decision for a future wave, the same way ARTX07 deferred PagedAttention. |
| **CPU** | — | ❌ | No flash kernel exists on CPU in this stack. CPU is gljax's oracle backend (F64 correctness checks), not a throughput target — so "CPU doesn't reach flash attention" is expected, not a bug, and there is nothing to probe for here. |

The GPU path is the only one gljax can reach "for free" (no kernel authored,
no `custom_call`, no per-backend branch in gljax's own code) — and it depends
entirely on the rewriter's pattern match firing, which is a property of the
*emitted graph's shape*, not of gljax's intent.

## What the rewriter needs, and a discrepancy this note exists to flag

ARTX09 §3.4 gives the flash-friendly emission order:

```text
scores = dot_general(Q, Kᵀ)      // bmm1
scores = scores * query_scale    // scale AFTER bmm1
scores = apply_mask(scores)      // mask BEFORE softmax
probs  = softmax(scores)
out    = dot_general(probs, V)   // bmm2
```

and says explicitly: *"The op order... is load-bearing. Folding the scale
into Q before the first dot... may defeat the rewriter's pattern match."*

**gljax's actual `ops::attention::gqa_attention` scales Q *before* bmm1**,
not the scores after:

```rust
let q_scaled = q.mul(&scale_t);   // scale BEFORE the dot
let scores = q_scaled.dot_general(&k_exp, ...);
```

This is not an oversight — the code comment gives the reason (scaling Q is
one fewer `S×S`-sized elementwise pass than scaling the full score matrix)
and it's the exact path Gate A5 has verified against real PJRT hardware
(CI runs `30447306245`, `30453269580`). The scale-after-bmm1 form ARTX09
prefers for GPU reachability has never been traced, let alone run.

**This note does not resolve the tension, and gljax's code has not been
changed to "fix" it.** ARTX09 §3.4 itself requires a measurement before
treating either form as correct — *"any deviation must be justified against a
measurement"* — and no GPU exists in this environment to take one. Until a
CUDA plugin is available:

- The current scale-before-matmul form stays, because it is the one path with
  real hardware evidence behind it.
- Whether it defeats `CudnnFusedMHARewriter` on a real GPU is an open
  question, not a known-bad choice — ARTX09's own language is "may defeat,"
  not "does defeat."
- If GPU throughput work ever starts, this is the first thing to measure:
  trace both forms, dump the post-optimization HLO with
  `--xla_dump_to`, and grep for `__cudnn$fmha` in each.

## ARTX11's Architecture descriptor adds more risk here, also unmeasured

ARTX09 §3.4 flags that Gemma-shaped attention (query pre-attention scalar,
QK-normalization — both landed structurally in `gljax::arch`/`ops::attention`
this sprint) inserts additional ops between the Q/K projections and bmm1.
Whether the rewriter still fires for those shapes is explicitly called out as
unknown. Nothing in this sprint's `Architecture::gemma3_shaped` work changes
that — it's still true, still unmeasured, and now has one more architecture
that could be affected.

## What would need to exist before this can move from "design note" to
"measured"

1. A CUDA PJRT plugin, and a machine with an NVIDIA GPU to run it on.
2. `XLA_FLAGS=--xla_dump_to=<dir>` wired into a probe that compiles a real
   attention trace and inspects the post-optimization HLO for
   `__cudnn$fmha` vs. a materialized `dot_general`/`reduce`/`dot_general`
   chain — structurally identical to ARTX10 §5's quantization capability
   probe (ARTX09 §3.2 draws this parallel directly).
3. The same probe run against both the scale-before-bmm1 (current) and
   scale-after-bmm1 (ARTX09-preferred) forms, and against at least one
   `Architecture::gemma3_shaped`-configured trace, to close the two open
   questions above with numbers instead of citations.

None of this exists yet. This document is the citation trail for when it
does.
