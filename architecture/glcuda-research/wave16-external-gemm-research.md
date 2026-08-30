# External research: where the prefill gap actually lives

Status: **research note, nothing built, nothing measured on hardware yet.**
Written 2026-08-29 after Wave 15 closed with attention exhausted.

## Why attention is finished as a lever

Five kernels were built and measured across Waves 15A-15E, all bit-exact:

| lever | measured | outcome |
|---|---|---|
| four QK chains on the row kernel | **+7.32%** over the old default | shipped as the default |
| four QK chains inside GQA7 | 1.117x isolated | +1.78% over gqa7 |
| two QK chains inside GQA7 | 1.034x isolated, **0.93x** at one group | dropped |
| four-row K/V tile (buy occupancy) | 0.905x | stopped |
| GQA7 KV sharing (pre-existing) | +5.44% over rows | opt-in, shape-limited |

The audits behind them close the question rather than leave it open. Attention
is **34.6%** of prefill; QK is **63%** of the attention kernel (measured on GQA7
itself in Wave 15C, not inherited from the row kernel's 71%). So **a QK that
cost nothing would still only be worth +32.6% end to end**, and four-way ILP
collected about 5% of that headroom because ILP and TLP turned out to be
substitutes — the win grows as occupancy falls and vanishes at full occupancy.

Reaching the 15k tok/s target from the shipped path needs **1.60x**. It is not
in attention. It never was.

## The number that says where it is

`glbench` against `llama.cpp` on one T4, 2026-08-21: **prefill is 5.85x
behind**. And Wave 13's audit measured the healthiest FFN GEMM stage at **8.2%
of the T4's int8 peak**.

Those two agree. 8.2% x 5.85 = 48%, which is what a well-tuned int8 GEMM
actually reaches. The gap is one bucket: **the FFN GEMMs, 45.6% of prefill.**

Arithmetic on the pinned shape, for `ffn_gate_up` (9728 x 896, 244 tokens):

    total work                     2.127e9 MACs
    m8n8k16 does 1024 MACs/instr   2.077e6 MMA instructions
    T4 tensor-core issue capacity  ~63.6e9 MMA/s
    => pure MMA issue time         ~32.7 us
    measured (5338 GMAC/s)         ~398 us

**The tensor cores are idle about 92% of the time.** That is the whole gap, and
it is not bandwidth: Wave 14 walked the weight image 0.85 -> 4.63 MB straight
across the T4's 4 MB L2 and throughput moved -0.9%.

## What the reference implementations do that we do not

Two independent sources, one of them the code we are 5.85x behind.

**CUTLASS `08_turing_tensorop_gemm`**, the vendor's own sm_75 int8 example:

| | CUTLASS | ours (`gl_gemm_mma_q8`) |
|---|---|---|
| MMA instruction | `m8n8k16` | `m8n8k16` — **identical** |
| threadblock tile | **128 x 256 x 64** | 64 (128 with ntile128) x 64 x **32** |
| warp tile | 64 x 64 x 64 | 8 rows per warp |
| **pipeline stages** | **2** | **1** |

CUTLASS's own comment on why two stages: they "hide latency from global and
shared memory loads" by phase-shifting the pipeline "so computation on already
loaded data can proceed while new data is being fetched".

**llama.cpp MMQ**, per the Percival audit (`architecture/percival/CUDA/ARTX12`):

| | MMQ (Turing uses the Ampere config) | ours |
|---|---|---|
| threads | 256 (8 warps) | 256 (8 warps) — **identical** |
| I (output rows/tile) | 128 | 64-128 |
| **K per iteration** | **256** | **32** |
| **shared memory/block** | **38,912 B** | **~3,300 B** |
| resident blocks/SM | **1, deliberately** | many |

⭐ Note the direction: MMQ spends nearly 39 KB of shared memory to run **one
block per SM**, betting everything on data reuse and tensor-core occupancy
rather than latency hiding. Wave 15D bet the other way on attention and lost
10%. These are different kernels with different bottlenecks, and the FFN GEMM
is the one where the reuse bet is standard practice.

## The mechanism this points at, and the prediction that tests it

Our mainloop stages an activation slice, `bar.sync`s, does two MMAs per 32-K
block, and repeats. **Load and math strictly serialize** — there is no second
stage in flight, which is exactly the thing CUTLASS added `NumStages = 2` for.
And with K = 32 per barrier, the barrier is amortised over the least possible
work.

That yields a prediction from numbers we already have, with no new run:

    ffn_gate_up  in_dim  896 -> 896/32  =  28 barriers   5338 GMAC/s
    ffn_down     in_dim 4864 -> 4864/32 = 152 barriers   3602 GMAC/s

**5.4x the barriers per unit of work, 1.48x slower.** Wave 14 attributed the
`down` deficit to grid starvation and priced closing it at +2.5-2.9%; the
occupancy sweep then showed the grid explains only part of it (the ratio
plateaus at 0.90-0.97, not 1.0). Barrier density is a second explanation for
the same residual, it was never tested, and it predicts the sign and rough
size of the gap between two stages doing identical MAC counts.

## Ranked leads

1. **Software-pipeline the mainloop (`NumStages = 2`).** The vendor example
   does it, MMQ's big tiles achieve the same end differently, and our kernel
   does neither. This is the one that plausibly addresses a 12x gap rather
   than a few percent.
2. **Raise K per iteration, 32 -> 128 or 256.** Amortises the barrier over 4-8x
   the work. Cheaper to try than pipelining and tests the same mechanism from
   the other side; it costs shared memory, which is exactly the budget MMQ
   spends freely here.
3. **Stage the weight tile in shared memory.** We stage only activations, so
   weights stream from global every k-step with no CTA-level reuse. MMQ's 38 KB
   is mostly this.
4. **Stream-K** for wave quantisation. Percival already marks it ADOPT and
   llama.cpp switches to it below 90% tile efficiency — `ffn_down` runs 56 CTAs
   on 40 SMs, which is 70%. Worth **+2.5-2.9%** by Wave 14's own measurement:
   real, small, and last in line.

## How to open this without repeating Wave 15's mistakes

- **Probe before building.** Wave 15's productive move every single time was an
  arithmetic or pass-split probe that killed a candidate for the price of one
  session. The equivalent here is a mainloop probe: time the k-loop with the
  MMAs removed, and with the barriers removed, to price staging and
  synchronisation separately before writing a pipelined kernel.
- **Ask the driver, do not compute.** Occupancy for any new tile geometry comes
  from `Cuda::max_active_blocks_per_sm`, added in Wave 15D after arithmetic
  mislabelled a whole sweep axis.
- **Bit-exactness is not free here.** Every attention kernel stayed
  bit-identical, which made correctness a non-issue. A pipelined GEMM keeps the
  same accumulation order per output element, so it can hold that line — but a
  larger K tile or stream-K changes summation order and cannot. Decide which
  before building, not after.
- **Four repeats minimum.** Two repeats overstated the Wave 15A effect by ~2x
  (+3.65% became +1.78%).

## Sources

- NVIDIA CUTLASS, `examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu`
  (threadblock 128x256x64, warp 64x64x64, `m8n8k16`, `NumStages = 2`).
- NVIDIA CUTLASS documentation, "Efficient GEMM in CUDA" (multistage pipelines).
- `architecture/percival/CUDA/ARTX10-GEMM.md` and `ARTX12-MMQ.md` — the
  in-repo llama.cpp audit, including the per-arch tile config table and the
  stream-K decomposition (arXiv:2301.03598).
- In-repo measurements: Wave 13 stage audit (8.2% of int8 peak), Wave 14 L2
  sweep and occupancy sweep, `glbench` vs `llama.cpp` on T4 (5.85x prefill).
