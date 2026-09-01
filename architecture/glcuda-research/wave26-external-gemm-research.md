# glcuda Wave 26 - external Turing GEMM research

**Date:** 2026-09-01
**Target:** Tesla T4 (`sm_75`), Qwen2.5-0.5B-Instruct Q4_K_M
**Status:** research complete; Wave 27 experiment frozen before implementation
**Parent:** Wave 25 rejected 16-byte cooperative staging at +1.116%

## Decision

Wave 26 found one untested mechanism that is still broad enough to cross the
production retention gate: make one warp own **two adjacent N8 output
fragments** while retaining the M64/K32 CTA tile. The Wave 27 candidate is a
warp-N16 register tile implemented with explicit `mma.sync.aligned.m8n8k16`
instructions. It reuses each activation fragment across two independent output
fragments and halves the activation-fragment shared-load issue count per CTA.

This is not a claim that a larger WMMA spelling performs more Tensor Core work
per instruction. PTX is a virtual ISA, and the NVIDIA Turing performance
example itself uses software tiling around the `m8n8k16` TensorOp instruction.
Wave 27 therefore keeps the retained explicit MMA primitive and changes only
the warp-level N ownership.

No kernel, dispatch path, notebook, or production default changes in Wave 26.

## Production budget after Wave 20

Wave 16 measured the FFN GEMMs at 45.6% of the then-current 27.083 ms prefill.
Wave 20 changed attention, not the FFN, and reduced the midpoint median to
23.826 ms. Holding the FFN's absolute time constant gives:

```
FFN time                 = 27.083 ms * 0.456 = 12.350 ms
post-Wave-20 FFN share   = 12.350 / 23.826   = 51.8%
```

The resulting Amdahl screen is:

| FFN kernel speedup | projected prefill speedup |
|---:|---:|
| 1.05x | +2.5% |
| 1.10x | +4.9% |
| **1.12x** | **+5.9%** |
| 1.15x | +7.3% |

Consequently a broad FFN mechanism must reach approximately 1.12x before it
has a credible path through the repository's +5% production gate. Improving
only `ffn_down` cannot do that through CTA count alone. Its old 19.4% share is
about 22.1% after the attention improvement; closing the measured 0.826 to
0.968 occupancy-sweep ratio has an Amdahl ceiling of only about +3.3% prefill.

These are projections, not benchmark results. Wave 27 must measure the two
production GEMM shapes directly and then run the immutable production protocol.

## What Waves 1-25 already closed

| Mechanism | Evidence | State for Wave 26 |
|---|---|---|
| 2D token grid and dynamic-shared attention | Waves 3-4, retained | production baseline |
| QKV stacking | Wave 13B, retained by explicit waiver | frozen; do not modify |
| row-QK and compensated-MMA attention | Waves 15E and 20, retained | attention scope closed |
| A/xscale double buffering | Wave 5, +0.49% | rejected |
| combined M128/N32 geometry | Wave 6, -13.77% | rejected; does not test M64 warp-N16 |
| native-Q4 / L2-residency theory | Wave 14 sweep was flat | rejected |
| register prefetch | Wave 17, direct +3.3% to +3.8% | below Amdahl gate |
| fused SwiGLU | Wave 22, +0.76% | rejected |
| K128 superstage | Wave 23, -14.03%, one block/SM | rejected |
| independent K-half accumulator chains | Wave 24, -0.65% | rejected |
| 16-byte cooperative copy | Wave 25, +1.116%, 2/4 positive | rejected |

The proposed N16 tile is distinct from Wave 24. Wave 24 split the two K16
halves for the same output. N16 keeps each output's K32 accumulation order but
creates a second independent N8 output fragment that consumes the same A
fragment. It is also distinct from Wave 6: the CTA remains M64 and the change
is inside a warp rather than a combined M128/N32 launch geometry.

## External findings

### 1. Turing rewards reuse and independent work, but resources are tight

The [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
documents 32 resident warps, 64K 32-bit registers, 64 KB shared memory, and 16
resident blocks per SM. It also says instruction-level parallelism can reduce
the number of warps required to hide arithmetic latency. This makes a wider
per-warp output tile plausible, but only while register and spill gates hold.

### 2. NVIDIA's fast GEMM structure uses hierarchical software tiles

The [CUTLASS efficient GEMM guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
describes CTA, warp, and instruction tiles and explicitly uses a 2D per-thread
accumulator tile so loaded operands feed multiple independent math operations.
The [Turing INT8 TensorOp example](https://github.com/NVIDIA/cutlass/blob/main/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu)
uses an `m8n8k16` instruction inside larger warp and threadblock tiles. That is
the external precedent for changing operand reuse around the retained MMA,
not for replacing it with an assumed wider hardware opcode.

### 3. Larger integer WMMA shapes exist, but are not yet a speed claim

The [PTX ISA 10.1 archive](https://docs.nvidia.com/cuda/archive/10.1/parallel-thread-execution/index.html)
defines signed-int8 WMMA forms including `m8n32k16` and `m32n8k16` for capable
targets. Their existence proves a warp-level N32 mapping is expressible. It
does not prove that `ptxas` emits fewer Tensor Core instructions than the
explicit `m8n8k16` sequence. A WMMA-only rewrite is therefore a SASS research
probe, not the Wave 27 production candidate.

### 4. `ldmatrix` becomes relevant only with a wider warp tile

CUTLASS's [SM75 memory primitives](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/memory_sm75.h)
provide `ldmatrix.sync.aligned.x1/x2/x4.m8n8.shared.b16`, and its
[TensorOp tile iterator](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h)
uses matrix loads with layouts designed for warp fragments. The retained
conflict-free 48-byte pitch already issues one ordinary shared load for one
fragment, so `ldmatrix.x1` alone has no credible instruction-count advantage.
It becomes interesting when one warp consumes two or four adjacent fragments.
Changing the shared layout at the same time would obscure the N16 result, so
Wave 27 holds the current shared image and ordinary `ld.shared` path fixed.

### 5. Stream-K is valid but cannot clear this workload's gate

The [Stream-K paper](https://arxiv.org/abs/2301.03598) partitions the MAC
iteration space to reduce tile-quantization inefficiency, including hybrid
schedules that apply Stream-K only to the tail. CUTLASS's
[split-K example](https://github.com/NVIDIA/cutlass/blob/main/examples/06_splitK_gemm/splitk_gemm.cu)
also makes the required partial-output workspace and reduction explicit. On
this workload the measured `ffn_down` sweep prices perfect tile balance at
about +3.3% post-Wave-20 prefill before scratch, reduction, or Q8 K32-scale
cost. Stream-K remains technically sound, but it is below the production gate.

## Candidate ranking

| Rank | Candidate | Reach | Main risk | Decision |
|---:|---|---:|---|---|
| 1 | M64/K32 warp-N16 with explicit `m8n8k16` | all FFN GEMMs | register pressure and fewer resident warps | **freeze for Wave 27** |
| 2 | warp-N32 plus `ldmatrix.x4` layout | all FFN GEMMs | roughly 4x accumulator footprint and a new shared layout | research only after N16 |
| 3 | `m8n32k16` WMMA spelling | all FFN GEMMs | unknown SASS expansion; no proven work reduction | SASS probe only |
| 4 | CTA N32 / split-K / Stream-K tail repair | principally `ffn_down` | approximately +3.3% Amdahl ceiling plus reduction cost | below retention gate |
| 5 | `ldmatrix.x1` on retained N8 tile | all FFN GEMMs | no shared-instruction reduction | reject without implementation |

## Frozen Wave 27 experiment

### Single causal lever

Change per-warp ownership from N8 to N16. One warp owns two adjacent N8 output
fragments and reuses each loaded A fragment for two explicit MMA operations.
Everything outside this register tile remains byte-identical to retained K32.

For an N64 CTA the warp count becomes four instead of eight; for N128 it
becomes eight instead of sixteen. The CTA output tile, grid, and global work do
not change. The static instruction body contains 32 explicit `m8n8k16` MMA
sites instead of 16 because each warp computes twice the N extent; total MMA
work per CTA is unchanged.

The expected register model is approximately 49 retained registers plus 16
additional f32 accumulator registers and a small number of B-fragment
temporaries. Wave 27 therefore sets a hard 72-register ceiling. This is an
estimate to be checked by `ptxas`, not a resource claim.

### Invariants

- retained K32-major prepacked B image and Q8_0 contract;
- M64 and K32 CTA dimensions, N64/N128 launch selection, and output grid;
- 9,728 bytes of static shared memory and the existing 48-byte pitch;
- two barriers per K32 iteration;
- ordinary shared loads; no `ldmatrix` or shared-layout rewrite;
- exact two-K16 integer accumulation for every output followed by the same
  conversion, scale multiplication, and f32 FMA order per K32;
- unchanged guards, vector stores, ragged-tail behavior, dispatch, and Wave 20
  attention path; and
- a separate opt-in candidate entry. The retained entry stays unchanged.

### Stop gates before production

1. `ptxas -v -arch=sm_75` must report at most 72 registers, exactly 9,728 B
   static shared memory, and zero stack, spill loads, or spill stores.
2. Structural inspection must find the intended N16 ownership, 32 explicit
   MMA sites, two barriers, and no shared-layout, K-loop, dispatch, or retained
   entry drift.
3. Driver occupancy must be recorded, not inferred. A drop below three active
   blocks/SM at 128 threads or two at 256 threads is a hard stop.
4. Existing library tests, full serial CUDA parity, exact production shapes,
   ragged N64 K160, token oracle, decode, and tail gates must all pass.
5. Direct retained/candidate timings must be bit exact and reach at least
   **1.12x on both** production `ffn_gate_up` 9728x896x244 and `ffn_down`
   896x4864x244. Either miss stops the wave before production timing.

The 1.12x screen is deliberately stronger than earlier 0.95 safety screens:
the external research is useful only if the candidate has an Amdahl path to
the +5% production requirement. Passing the direct screen allows, but does not
replace, the normal four-pair production gate: session-P50 ratio at least
1.05x and all paired deltas positive.

## Boundary

Wave 26 ends at this research and design freeze. Wave 27 may implement exactly
the N16 experiment above after explicit confirmation. A wider N32 tile,
`ldmatrix`, WMMA spelling, split-K, or Stream-K requires a separately frozen
wave and may not be used as fix-forward if N16 fails.
