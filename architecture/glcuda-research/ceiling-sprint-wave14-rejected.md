# glcuda rejected and deprioritised optimisations (Wave 14 audit)

Things that were measured and will not be built, so nobody spends a wave
re-discovering them. Each entry says what was measured, on what, and what would
have to change for the verdict to be revisited.

---

## 1. Native Q4 weights for the FFN GEMM — **REJECTED**

Rejected 2026-08-28 on counterbalanced evidence, ordered archived by JinXSuper.

**The idea.** FFN weights are the biggest thing the engine reads. Storing them as
Q4 instead of Q8 halves the bytes, so if the GEMM is bandwidth-bound it goes
roughly twice as fast on the weight stream.

**What was measured.** `glcuda/examples/wave14_audit.rs`, section B. Hold
`out_dim` and `ntok` fixed so the grid is pinned at 56 CTAs, then walk `in_dim`
so the weight image sweeps **0.85 MB -> 4.63 MB**, straight across the T4's
**4 MB L2**. If bytes bind, throughput falls when the working set stops fitting.

| in_dim | weight MB | direct GMAC/s | B-stage GMAC/s |
|---:|---:|---:|---:|
| 896 | 0.85 | 2100 | 2298 |
| 1792 | 1.71 | 2081 | 2272 |
| 2688 | 2.56 | 2109 | 2306 |
| 3584 | 3.41 | 2040 | 2271 |
| 4864 | 4.63 | 2080 | 2300 |

**Change across the whole crossing: −0.9% (direct), +0.1% (B-stage).** Flat.
Bytes are not what binds this GEMM, so halving them buys nothing.

⚠️ **Read the history before quoting this.** Two earlier runs of the same sweep
gave **+3.5%** and **−13.9%** for this number, and one of them was reported as
proof. Both were artefacts: the sweep ran monotonically upward, so the largest
image always met the hottest GPU and thermal position was confounded with the
swept axis. Only after each point was measured **forward and backward within one
invocation and averaged** did the spread collapse from 17.4 points to 1.0.

**Prior, independently:** on CPU, native Q4_K was actually built and **lost 33%**
because the nibble unpack was compute-bound. Different machine, different
mechanism, same conclusion.

**What would reopen it.** A shape where the FFN GEMM is shown to be
bandwidth-bound — a much larger model, a longer prompt, or a device with a
smaller L2 relative to the weight tile. Re-run section B there first; do not
re-derive the idea from the byte column alone.

---

## 2. split-K for `ffn_down` — **DEPRIORITISED**

Not rejected: measured, priced, and parked behind a bigger lever.

**The idea.** `ffn_down` is 896x4864, so at the pinned prompt it launches
`ceil(896/64) x ceil(244/64)` = **56 CTAs** on a 40-SM T4 that holds 160
resident. Its K is the longest in the model (4864), so splitting K would
multiply the grid.

**What was measured.** Section A of the same audit, sweeping ntok to sweep the
CTA count. `down` and `gate_up` have **identical MAC counts** at every row, so
their ratio is a pure efficiency comparison:

| ntok | down CTAs | down/gate_up |
|---:|---:|---:|
| 64 | 14 | 0.307 |
| 128 | 28 | 0.555 |
| **244** | **56** | **0.826** |
| 488 | 112 | 0.902 |
| 976 | 224 | 0.968 |

Some of the deficit is genuinely the grid — the ratio climbs and then plateaus.
But it plateaus at **~0.90-0.97, not 1.0**, so closing it is worth
`0.194 x (1 − 0.826/0.968)` of prefill = **+2.5 to +2.9% end-to-end**, not the
+4.1% a formula that assumes a 1.0 plateau printed.

**Why parked.** It needs a new kernel with partial sums and a reduction pass,
and the reduction changes summation order, so it **gives up bit-exactness** —
the property that has made every wave since 12 cheap to verify. +2.9% is not
worth that while a lever worth +24% is unbuilt.

⚠️ The binary "STARVED / NOT STARVED" label from this section flip-flopped across
three runs (STARVED, NOT STARVED, STARVED) because it keys on an **absolute**
gain and `gate_up` itself moved 15-39% across the same sweeps. The ratio column
is the evidence; the label is not.

**What would reopen it.** Finishing the attention wave, then re-pricing against
the new budget: with attention 2.27x faster, `ffn_down` becomes a materially
larger share of what is left.
