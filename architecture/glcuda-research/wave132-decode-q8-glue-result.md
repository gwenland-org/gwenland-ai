# Wave 132 - exact Q8 decode glue result

## Outcome

Wave 132 retains the exact Q8 decode-glue path as an opt-in production
candidate behind `GLCUDA_DECODE_Q8_GLUE=1`. The selector is accepted only when
the existing exact-fusion contract (`GLCUDA_FUSE_Q8_GLUE=1`) is also enabled.

On a Tesla T4, the candidate improved production decode P50 from 208.50 to
214.66 tok/s (+2.95%) without a material prefill change. The candidate passed
the serial CUDA parity, CUDA Graph replay, dispatch-provenance, oracle, and
counterbalanced wall-time gates.

This is a retained improvement, not a decode-ceiling claim. `glbench` still
classifies decode as compute/kernel-bound and estimates a 473.6 tok/s ceiling
from published bandwidth rather than measured device bandwidth.

## Production wall time

Four independent `glbench` processes ran in A-B-B-A order. Each process used
two warmups and ten measured iterations, with a 244-token prompt and 64 decode
tokens. Both arms kept the Wave 131 production prefill path enabled; only
`GLCUDA_DECODE_Q8_GLUE` changed.

Combined prefill distributions (20 samples per arm):

| Arm | P50 | P90 | P99 | Range |
|---|---:|---:|---:|---:|
| retained | 15,768.15 | 15,816.27 | 15,827.79 | 15,557.55-15,830.17 tok/s |
| candidate | 15,759.66 | 15,818.02 | 15,859.38 | 15,474.27-15,865.30 tok/s |

The prefill P50 difference is -0.05%, so the retained 15,000 tok/s production
target remains intact.

Combined decode distributions (20 samples per arm):

| Arm | P50 | P90 | P99 | Range |
|---|---:|---:|---:|---:|
| retained | 208.50 | 209.25 | 209.38 | 206.83-209.41 tok/s |
| candidate | **214.66** | **215.25** | **215.86** | 213.61-215.98 tok/s |

The candidate improves decode P50 by 2.95%. Its two independent session
medians were 215.00 and 214.62 tok/s, while the retained bookends measured
208.74 and 207.48 tok/s. That directional agreement across A-B-B-A rules out a
single favorable run as the explanation.

Every production session matched 50/50 verification tokens against `glproc`.

## What changed

The decode graph reuses existing exact-fusion kernels and fixed workspace
buffers; it adds no allocation and no new PTX kernel. The candidate:

- combines RMS normalization with Q8 quantization for attention input;
- combines the previous FFN residual, the next RMS normalization, and Q8
  quantization across layer boundaries;
- combines SwiGLU with Q8 quantization before the down projection; and
- consumes the final deferred residual in the output normalization path.

Fallback and non-logit paths materialize any deferred residual before their
existing operations, preserving the original execution contract.

## Validation

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Model: 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Host library tests: 74 passed.
- Serial CUDA parity: 36 passed.
- CUDA Graph replay: 4 passed on the T4; the log confirms captured replay.
- Candidate sessions logged `decode_q8_glue:true` and
  `wave132-exact-q8-glue`; retained sessions logged
  `decode_q8_glue:false` and no Wave 132 dispatch.
- Both arms logged the retained `wave129-fused-swiglu-production` prefill path.
- Wave 129 compiler-resource gate remained zero-stack and zero-spill.

## Ceiling status

The candidate reaches about 45.3% of `glbench`'s 473.6 tok/s estimated-table
ceiling, up from 44.0% for the retained path. Because the estimate uses the
T4's published 320 GB/s and the model file size, it is a planning reference,
not a measured hardware roofline. The remaining decode work is led by GEMM and
attention inside the captured graph; a later wave must measure and optimize
those kernels before declaring the decode ceiling reached.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave132-decode-q8-glue.json`.
- Four raw production sessions:
  `benchmarks/glcuda-t4-wave132-glbench-{1..4}-*.json`.
- Full logs and raw records:
  `benchmarks/glcuda-t4-wave132-decode-q8-glue-results.zip`.
- Archive SHA-256:
  `4a4eccb35c48abb6323a10788b5f365bb548547e76dde524e838db2eb73010f0`.
