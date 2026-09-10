# Wave 111 - cross-layer residual fusion result

## Result

Wave 111 removed the standalone FFN residual add from every non-final prefill
layer. The FFN-down result remains live in `pf_proj` and the next layer's
existing `rms_quantize_q8_rows` residual arm performs the same f32 add while
it reads `pf_x` for attention RMSNorm. The final layer still materializes the
add before output normalization. A 24-layer chunk therefore removes 23 kernel
launches and one full `pf_x` read per boundary without allocating memory or
adding a PTX entry.

Two counterbalanced direct runs on a verified Tesla T4 passed the fixed 1.05x
gate:

| run | retained add + RMS/Q8 | fused RMS/Q8 | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 24.776 us | 18.338 us | **1.3511x** |
| 2 | 20.204 us | 14.760 us | **1.3688x** |

The absolute worker timing shifted between runs, but the within-run speedup
reproduced at 35-37%. All four observable products were bit-exact in both
runs: the updated residual stream, normalized f32 output, Q8 payload bytes,
and Q8 scales.

The rejected Wave 105/109 runtime modules, dispatch arms, PTX images, direct
harnesses, and PTX generators were removed. Their notebooks, raw archives,
and result documents remain as historical evidence.

## Decision for Wave 112

Accept Wave 111 through the direct gate but keep it opt-in. Wave 112 must run
the real Q8 production `glbench` A/B with output parity and counterbalanced
repeats. Only that measurement can decide retention or update the production
throughput record. The fastest valid production measurement remains the
experimental 12,232.7 tok/s configuration; no new production number was
measured in Wave 111.

Raw archive: `benchmarks/glcuda-t4-wave111-residual-fusion-v1.zip`

Archive SHA-256:
`a248549b59ebce3300efdad6ef81b0760c90f2f0f6d4d57ad951794a0ef72a99`

