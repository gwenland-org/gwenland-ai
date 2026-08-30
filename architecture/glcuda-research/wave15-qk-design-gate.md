# Wave 15 design gate — turning "+24-28%" into something that can fail

Status: **research gate closed, 2026-08-28. Wave 15A is specified and not yet built.**
Ordered by JinXSuper: no kernel work until the claim is a falsifiable
isolated-kernel target.

## 1. The claim, restated as arithmetic

Measured inputs, all from the Wave 14 audit on a T4:

- attention is **34.6%** of prefill
- inside attention, QK is **71-72%**, softmax **2%**, PV **~27%**
  (reproduced across three sessions: 71 / 72 / 72)

If QK gets `N` times faster and nothing else changes, attention's time becomes
`0.71/N + 0.29` of what it was, and the session gain follows from Amdahl on a
34.6% share. That gives the whole table:

| end-to-end target | attention must be | **QK must be** |
|---|---:|---:|
| +10% | 1.36x | 1.59x |
| +15% | 1.61x | 2.13x |
| +20% | 1.93x | 3.11x |
| **+24%** | 2.27x | **4.71x** |
| +28% | 2.72x | **9.13x** |
| +30% | 3.00x | 16.50x |
| QK -> 0 (ceiling) | 3.45x | infinite |

⭐ **Two things this kills immediately.**

First, "+24-28%" is not a range, it is a cliff: the last four points cost
**twice** the QK speedup of the first twenty-four. Quoting the pair as if they
were comparable asks (which I did) hides that.

Second, there is a **hard ceiling of +32.6%** from QK alone. Even an
instantaneous QK leaves softmax, PV and the rest. Anything promising more than
that from this wave is promising something else.

## 2. The pre-registered target

Fixed here, before the kernel exists.

> **Wave 15A passes when the new prefill attention kernel, measured in the same
> isolated run as the retained one, at the production shape (244 tokens, 14
> heads, 2 KV heads, head_dim 64, causal), is at least 2.27x faster overall —
> which requires QK itself to be at least 4.7x.**

Notes that make it falsifiable rather than rubbery:

- **Same-run ratio, never a standalone number.** Absolute GMAC/s on this machine
  moved 5-19% between sessions during Wave 14; only ratios measured inside one
  invocation survived.
- **Isolated first, production second.** The isolated number bounds what the
  wave can buy. The interleaved same-session A/B remains the only retention
  authority, and its bar is unchanged at 5% median.
- **Minimum worth landing:** 1.5x on attention (~+13% end-to-end). Below that,
  the kernel is not worth its own maintenance.
- If the kernel lands between 1.5x and 2.27x, that is a *measurable win below
  the target*, reported as such — the Wave 13B precedent — not a silent pass.

## 3. Which QK, and the precision question, answered locally

Two candidate designs, priced by warp-instructions per score produced at
head_dim 64:

| design | per score | note |
|---|---:|---|
| today: warp-per-key, 5-step shuffle reduction | ~12 | 2 useful FMAs per lane, then ten instructions of reduction |
| **lane-per-key**, K tile staged in shared memory | ~2 | no reduction at all; **~6x fewer instructions** |
| `mma.sync.m16n8k8` | 1024 MAC/instruction | ~32x denser again, but f16 operands |

`mma.sync` on sm_75 takes **f16** inputs (f32 accumulate). That is a numerics
change, and it was settled on this machine, against the oracle, before any PTX:

    glcuda/src/attention/reference.rs
      half_precision_qk_costs_the_answer_almost_nothing

Rounding Q and K through binary16 at the production shape costs:

| | |
|---|---:|
| max absolute deviation | **1.227e-2** |
| rms relative | 1.729e-3 |

⛔ **Judged against this engine's own tolerances, not an invented one.**
`parity.rs` grades the attention path at `EPS_MATMUL = 1e-5`, and `assert_close`
makes that an *absolute* bound for outputs under 1.0. So f16 operands miss the
tolerance attention is currently held to by **three orders of magnitude**, and
land in the same class the engine accepts for a **Q4_K GEMV** (1e-2).

That is the finding: **f16 QK is a deliberate precision downgrade of
quantisation size, not a free change of representation.**

⭐ A side effect worth keeping: the first version of the f16 model asserted that
attention activations never reach the f16 subnormal range, and the first run
falsified it immediately (a Q element at 2^-16). Uniform activations in [-1, 1]
cross 2^-14 about once in sixteen thousand, and there are 218k of them. The
model now handles subnormals on their 2^-24 grid.

## 4. Decision

**Wave 15A = lane-per-key QK in f32.** One lane owns one key and walks all 64
dimensions, so the five-step shuffle reduction disappears entirely. K tiles are
staged into shared memory so the global reads stay coalesced, with the row pitch
padded to 65 floats to keep 32 lanes off one bank.

Why this and not MMA first:

- The instruction count says ~6x, and the target needs **4.7x**. It is the
  cheapest design that can clear the bar.
- It changes **no numerics class**: still f32 operands, f32 accumulate. It is
  gradeable against the existing oracle at the existing tolerance, with no new
  tolerance to argue about.
- The reassociation is real (a serial 64-element accumulate instead of a 2-per-
  lane tree), so it is not bit-exact — but it stays inside `EPS_MATMUL`, which
  f16 does not.

**MMA-f16 is deferred, not rejected.** It is the only way past ~6x, and the
target's own table says the last four points need 9x. When it is taken up it
needs a pre-registered tolerance class of its own and token-level oracle
evidence, never a silent swap.

## 5. Housekeeping ordered with this gate

- **Native Q4 for the FFN: REJECTED.** The counterbalanced sweep walked the
  weight image from 0.85 MB to 4.63 MB across the T4's 4 MB L2 and throughput
  moved **−0.9% (direct) / +0.1% (B-stage)**. Bytes are not what binds, so
  halving them buys nothing. Recorded in
  `ceiling-sprint-wave14-rejected.md`.
- **split-K for `ffn_down`: DEPRIORITISED.** Worth **+2.5 to +2.9%** measured
  from the ratio plateau, needs a new kernel with partial sums and a reduction,
  and gives up bit-exactness. Not worth it while a 24% lever is unbuilt.
