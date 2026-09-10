# Wave 79 - fused compensated-MMA AV T4 device result

## Intended deliverable

Run the Wave 78 production-shaped attention candidate on two independently
allocated Tesla T4 workers. The gate requires full-module assembly, unchanged
compiler residency, serial CUDA parity, and an interleaved direct A/B against
retained `mma4-regq` before any production `glbench` spend.

## Reproduction contract

- Kaggle kernel: `jinxsuperdev/glcuda-wave79-hard-t4-mma-av-device-gate`;
- versions: 1 and 2;
- hard accelerator selector: `NvidiaTeslaT4`;
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04;
- source revision: `48ae7c7ea53e299efca02afb8c996f938dec16bf`;
- reachable base: `bd5c956bafb3bb6738c3f1de348a4ebb55d9c29f`;
- embedded binary patch SHA-256:
  `da156a5ebc4b9e6b4c89181120f27511092ac1650d4eadffe022a316b34e367c`;
- submitted notebook SHA-256 (CRLF):
  `c36ce71b931b08fd554ffc3f506f92b01a2315cd1de52e72cf8b32e0e677dd03`;
- repository notebook SHA-256 (LF):
  `d0bcfce2655f8477be37885ca79c7e3a539ec5717eec0f98ad9b0420239d2f19`;
- direct timing: 20 warmups, 200 launches, seven alternating rounds;
- direct shape: 244 tokens, 14 query heads, two KV heads, head dimension 64;
- numeric gate: max absolute difference at or below `1e-5`;
- direct speed gate: candidate at least 1.10x retained.

The notebook checks out the remote base and applies the byte-locked patch
because the measured source commit is intentionally not pushed. The applied
file manifest and patch hash are included in both archives. The committed
notebook is JSON-identical to the submitted notebook after universal-newline
normalization; its generator now forces LF so repository whitespace checks are
portable.

## Compiler and occupancy

| entry | registers | stack | spill stores | spill loads | blocks/SM |
|---|---:|---:|---:|---:|---:|
| retained `mma4-regq` | 64 | 0 B | 0 B | 0 B | 4 |
| candidate `mma4-regq-avmma` | 64 | 0 B | 0 B | 0 B | 4 |

Both full-module assemblies passed. The added normalized-probability pass and
MMA AV schedule therefore did not cross a T4 register or residency tier.

## Device correctness

Both versions passed the complete serial CUDA parity suite: 33/33 tests. The
Wave 78 case executed the registered production shape plus non-zero-base,
strided-Q, ragged and short-tail shapes. Neither run contained a device skip.

At the production direct shape, both runs reported the same maximum absolute
difference, `6.556510925e-7`, and RMS `1.168130365e-7`. The maximum is 15.25x
below the registered `1e-5` limit.

## Direct fused-attention result

| version | retained | candidate | speedup | latency reduction |
|---:|---:|---:|---:|---:|
| 1 | 149.321 us | 110.767 us | **1.3481x** | 25.82% |
| 2 | 146.328 us | 107.401 us | **1.3625x** | 26.60% |
| midpoint | 147.8245 us | 109.084 us | **1.3551x** | **26.21%** |

Every result is an end-to-end launch of the fused attention entry, so the new
normalization pass and CTA barrier are included. The A/B still remains a direct
kernel gate rather than production inference.

## Evidence

- version 1 archive:
  `benchmarks/glcuda-t4-wave79-mma-av-device-v1.zip`, SHA-256
  `9971293eb3dfe9916fb910c26793ac29d82c3c6b3136599514acc7098c95d495`;
- version 2 archive:
  `benchmarks/glcuda-t4-wave79-mma-av-device-v2.zip`, SHA-256
  `151565087b6d507abde99391be8bbc237bbe6e7d42c9bf0b4544a1bd63c6583b`;
- parsed summary: `benchmarks/glcuda-t4-wave79-mma-av-device.json`;
- notebook: `notebooks/glcuda_t4_wave79_mma_av_device.ipynb`;
- deterministic builder: `scripts/build_wave79.py`.

Each archive contains the source manifest, full `ptxas -v` output, resources,
serial parity log, direct output, result JSON and PASS marker. Temporary Kaggle
staging was moved to the Windows Recycle Bin after both archives were verified.

## Decision

**REPRODUCED T4 DEVICE PASS.** The Wave 78 candidate is correct on hardware,
keeps the retained compiler resource tier, and clears the fused-attention
direct speed gate twice. It is licensed for a controlled production `glbench`
A/B but remains opt-in and is not yet retained.

No prefill tok/s claim follows from this gate. Probe-to-production disagreement
in this repository has ranged from 0.07x to 2.40x. The next wave must run
position-balanced baseline/candidate production sessions with identical Q8
model, 244-token prompt, seed, warmup and correctness oracle. The 15,000 tok/s
goal remains open.
