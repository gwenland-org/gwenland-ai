# Wave 136 - fused QKV decode grid result

## Verdict

Retained behind `GLCUDA_DECODE_QKV_FUSED=1`. One Q8 SoA grid spanning the
independent Q, K, and V output rows improved production decode P50 from
216.61 to 223.63 tok/s (+3.24%) on a Tesla T4. Both candidate sessions beat
the median of the two control sessions, and the combined bootstrap intervals
do not overlap.

The candidate is gated by the retained Wave 132 exact decode-Q8 contract.
Unsupported formats or mismatched projection shapes keep the original three
GEMV launches.

## Mechanism

The retained path submits Q, K, and V as three consecutive kernels even though
they read the same quantized activation and are mutually independent. Wave 136
maps one global row range across Q, K, and V while keeping their weight and
output buffers separate. Each warp still owns one output row and executes the
same loads, DP4A operations, scale folds, shuffle reduction, and final store as
`gl_gemv_q8_0_soa`.

This is graph-structure fusion, not a changed dot product. It removes two
inter-kernel dependency edges and lets the device schedule all 1,152 Q/K/V
rows as one grid without concatenating or duplicating weights.

## Counterbalanced production A/B

Four independent `glbench` processes ran in control-candidate-candidate-control
order. Each used the same 244-token prompt, 64 decode tokens, two warmups, ten
measured iterations, deterministic sampling, and a 50-token `glproc` oracle.
Only `GLCUDA_DECODE_QKV_FUSED` differed.

| Order | Arm | Prefill P50 | Decode P50 | Oracle |
|---:|---|---:|---:|---:|
| 1 | control | 15,855.00 tok/s | 217.14 tok/s | 50/50 |
| 2 | candidate | 15,679.22 tok/s | 223.79 tok/s | 50/50 |
| 3 | candidate | 15,786.56 tok/s | 223.50 tok/s | 50/50 |
| 4 | control | 15,858.15 tok/s | 216.44 tok/s | 50/50 |

Combined distributions (20 samples per arm):

| Phase | Arm | P50 | P90 | P99 | 95% bootstrap CI for P50 |
|---|---|---:|---:|---:|---:|
| prefill | control | 15,858.15 | 15,919.91 | 15,951.27 | 15,830.84-15,890.05 tok/s |
| prefill | candidate | 15,744.51 | 15,891.84 | 15,933.58 | 15,666.63-15,793.59 tok/s |
| decode | control | 216.61 | 217.65 | 217.93 | 216.34-217.14 tok/s |
| decode | candidate | **223.63** | **224.48** | **224.64** | **223.42-223.99 tok/s** |

The candidate is decode-only, so the -0.72% prefill difference is not an
attributed effect. Its prefill P50 remains above the 15,000 tok/s production
floor required by the retention gate.

## Production ceiling observed

The retained Wave 136 path reaches 223.63 decode tok/s at P50, 224.48 at P90,
224.64 at P99, and 224.65 at the best observed sample. Prefill remains
15,744.51 tok/s at P50. These are end-to-end production measurements for the
pinned Qwen2.5-0.5B Q8_0 workload on this T4, not standalone kernel rates.

## Validation

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Model: 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Host library tests: 76 passed.
- Serial CUDA parity: 37 passed, including a five-block-tail case and the
  real 896/128 Qwen projection widths.
- CUDA Graph replay: 4 passed with the fused path selected.
- Candidate resources: 34 registers/thread, zero stack, zero spill, zero
  barriers, 256 threads/block.
- Every production session: 50/50 oracle agreement.
- Source patch SHA-256:
  `90a57a94ce1c1b2ed39e90e6830b5ed6325c4c84caa7fd67bdbeec1da7e2672d`.
- Result archive SHA-256:
  `e82a90be16278b8f02cd79c0d4349e8d3ad06656dee1a9387d1b71be578d02fd`.

## Open ceiling

Wave 136 materially raises decode, but it is not proof of the physical decode
ceiling. The captured profile still ranks gate/up, attention, and down ahead
of QKV. Future work should test another bounded dependency-edge or byte-traffic
reduction and keep this exact production methodology.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave136-qkv-fused.json`.
- Four raw sessions:
  `benchmarks/glcuda-t4-wave136-glbench-{1..4}-*.json`.
- Full logs and raw records:
  `benchmarks/glcuda-t4-wave136-qkv-fused-results.zip`.
- Reproduction notebook:
  `notebooks/glcuda_t4_wave136_qkv_fused.ipynb`.
