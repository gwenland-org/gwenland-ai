# Wave 13B — QKV stacking: design, and what it actually costs

Status: **RETAINED BY EXPLICIT WAIVER, 2026-08-27 (JinXSuper). QKV stacking is frozen.** Verdict below: a real +4.26%
that missed the 5% retention bar, reported as REJECT by a tail gate that
could not measure what it judged. Read the RESULT section before reusing
any threshold from this wave.
Date: 2026-08-27. Follows Wave 13A (attention boundary + observability).

## The claim being tested

`q`, `k` and `v` are three separate GEMMs against the same activation. At the
pinned 244-token prompt on a 40-SM T4:

| projection | out rows | CTAs per 64-token slab | CTAs total |
|---|---:|---:|---:|
| q | 896 | 14 | 56 |
| k | 128 | **2** | **8** |
| v | 128 | **2** | **8** |
| stacked qkv | 1152 | 18 | **72** |

Eight CTAs leave 32 of 40 SMs idle for the whole launch, and it happens twice
per layer, 48 times per prefill. Stacking makes it one launch that covers the
machine.

Per JinXSuper's Wave 13 principles this is a **weight-layout decision**, not a
kernel special case: `W_qkv = [W_q; W_k; W_v]`, and the GEMM keeps seeing only
`M x K` times `N x K`. No `gemm_qkv_special()`.

## Measured starting point (Wave 13A telemetry, T4, 2026-08-27)

| stage | share | GMAC/s | GB/s |
|---|---:|---:|---:|
| qkv | 8.6% | **1249** | 21.7 |
| attn_out (896x896, same shape class as q, one launch) | 3.1% | **2683** | 46.7 |

⚠️ **The `qkv` stage is not only the three GEMMs.** It also contains the input
RMSNorm and the activation quantize (`rms_quantize_q8_rows`, and `quantize_q8`
on the unfused arm). So 1249 GMAC/s understates the GEMMs, and any estimate of
"what stacking buys" built on that number alone is an estimate on a mixed stage.
That is precisely the class of reasoning this repo has been burned by before
(rejected-optimizations: probe said 0.07x-2.40x for one change; an estimate of
15-60 GFLOP/s measured 163).

Hence: screen it in isolation first. `glcuda/examples/wave13b.rs`.

## The screening diagnostic

`cargo run --release -p glcuda --example wave13b` on a T4. One stacked weight
image, three launch geometries over the same rows, plus a bit-exactness check
that one 1152-row launch reproduces three launches value for value:

- `separate_us` — today's three launches
- `kv_stack_us` — `q` alone plus one 256-row `K|V` launch
- `qkv_stack_us` — one 1152-row launch
- `q_only_us`, `k_only_us` — the per-projection split, so the k/v starvation
  is visible as a number rather than inferred from a stage total

Go/no-go before touching any layout: `qkv_stack_speedup` on the isolated GEMM.

## What implementing it actually requires

The weight side is trivial: SoA is row-major, so concatenating three matrices
along rows is a memcpy, and every N128 boundary lands right (`896 = 7*128`,
`1024 = 8*128`, `1152 = 9*128`), which means even the Wave 12 B-stage tiled
image slices correctly.

The **output side is where the cost is**. A single launch writes one
`[n_tokens, 1152]` block, so Q/K/V become column slices with a row stride of
1152. Every consumer today derives its row stride from its row *width*:

| consumer | how it strides today | needs |
|---|---|---|
| `gl_add_bias_rows_f32` (bq/bk/bv) | row width param | row-stride param |
| `gl_rms_norm_rows_f32` (q_norm/k_norm, Qwen3) | `head_dim` rows, contiguous | row-stride param |
| `gl_rope_rows_f32` | `mul.lo.s32 %r21, %r1, %r2` = heads*head_dim | row-stride param |
| `gl_kv_write_rows` | `%r8 = n_kv*head_dim`, used for the bound AND the stride | src-stride param |
| attention (`q` side) | derives `n_heads*head_dim` | q-stride param |

So the full change touches **five kernels**. Each edit is small and surgical
(the row offset is one `mul` in each), but it is PTX, and PTX is where this
project has historically paid.

**Safe migration order**, each step provable by tests that already exist:

1. Add the stride parameter to each kernel and pass the currently-derived value
   at every call site. Bit-exact no-op; `rope_matches_reference_both_styles` and
   `kv_write_places_rows_at_device_pos` prove it.
2. Stack the weights at upload (`w_qkv`, like `w_gate_up` already is).
3. Switch to one launch and pass 1152 as the stride.
4. Production A/B against the Wave 13A baseline.

**Cheaper variant** if the screening says the full change is not worth five
kernels: stack only `K|V` (256 rows, 16 CTAs, one launch instead of two). It
leaves the attention kernel untouched — worth something because that kernel is
about to be replaced by the tiled MMA path in steps 5-7, and adding a parameter
to a kernel that is being rewritten is churn.

## Open question the screening answers

Whether k/v are slow because of CTA starvation at all. Eight CTAs of 64x64x896
should be ~2.3 us of work each; if the isolated `k_only_us` comes back near
that, the starvation story is right and stacking will pay. If it comes back far
higher, something else is wrong with the small-`out_dim` GEMM path (the B-stage
image pads to 128 rows, so a 128-row projection may be staging more B than it
uses) and stacking would be treating a symptom.

---

## RESULT (Kaggle T4, 2026-08-27) and a gate that has to be fixed

Screening: **2.13x** isolated. Production: `qkv` went 1445 -> 3103 GMAC/s
(**2.15x**) and 10.1% -> 5.3% of prefill. The isolated prediction transferred to
within 1%, which is the main methodological result of this wave: screen a layout
change in isolation *before* paying for its engine surgery.

Session throughput: **+4.26%** median, paired +2.65/+4.39/+4.13/+3.45%, 4/4
positive, oracle 50/50 in every session, `[glcuda-qkv]` confirming 24/24 layers
stacked in the candidate and absent in the baseline.

The notebook printed **REJECT**, and the reason was a gate written here, not the
change. The rule demanded every paired repeat hold P95/P99 latency within ±5%.
Measured from the Wave 12 archive, the **within-arm** spread of those statistics
— same binary, same session, interleaved repeats — is:

| statistic | within-arm spread across 4 repeats |
|---|---|
| P95 latency | 5.8% - 8.0% |
| P99 latency | 5.7% - 12.1% |
| P50 throughput | 6.3% - 8.1% (across repeats; paired deltas are far tighter) |

A ±5% band on a statistic that moves 5.8-12.1% on its own cannot discriminate.
The gate was guaranteed to fire on noise. Meanwhile the median P95 moved
31.22 -> 31.33 ms, **+0.35%** — no tail cost at all.

**Corrected rule for later waves** (use this, do not reuse the ±5% paired band):

1. Judge tails on the **median across sessions**, not per-repeat pairs; the
   per-repeat tail is the 2nd-worst of 10 iterations and is the noisiest number
   in the run.
2. If a per-repeat tail band is wanted anyway, derive it from within-arm spread
   measured in the same session, and it will be **>=12%**, not 5%.
3. Never let a tail gate veto the throughput verdict outright. Report them
   separately, so "the change did nothing" and "one session had a jittery tail"
   never print the same word.

**This correction does not retain Wave 13B.** The primary bar (5% median and
every repeat >=5%) was missed too: +4.26% median, best repeat +4.39%. The honest
verdict is *measurable win below the bar*, which is a decision for JinXSuper
rather than something a rule settles.

---

## WAIVER — retained below the bar, 2026-08-27

JinXSuper retains Wave 13B by explicit waiver rather than by the automatic rule.
Recorded here so nobody later reads the +4.26% as if it had cleared 5%.

**What the waiver covers:** the 5% retention bar, missed at +4.26% median
(best repeat +4.39%). Nothing else was waived. Every other gate passed on its
own: oracle 50/50 in all eight sessions, PTX changes confined to the six
entries declared in advance, the tensor-core module byte-identical, parity
running on hardware with the strided-vs-packed test green, and the engine
reporting 24/24 layers stacked in the candidate and none in the baseline.

**Why:** the effect is 4.3x the measured throughput noise floor (0.61%), it is
positive in 4 of 4 interleaved repeats, and an independent isolated screening
predicted it to within 1% (2.13x bench, 2.15x in the stage). It is a structural
change with no VRAM cost and no arithmetic change, so it cannot silently decay
the way a tuned constant can.

**What is NOT claimed:** that the rule was satisfied. It was not. The tail gate
that printed REJECT was separately shown to be unmeasurable (see above) but
fixing it would only have changed the wording to "measurable win below the bar".

**Frozen:** QKV stacking is done. The remaining `qkv` glue (5.3% of prefill,
mostly the input RMSNorm and quantize) is opportunistic cleanup, not a wave.
