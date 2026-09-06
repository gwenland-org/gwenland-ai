# Wave 58 — teacher-forced logits diagnostic (T4)

Wave58 isolates numerical drift from autoregressive amplification by comparing CPU `glproc` and GPU `glcuda` logits on the same token history for 50 teacher-forced steps. The run used the pinned Qwen2.5-0.5B-Instruct Q8_0 GGUF on a Tesla T4 (SM75, driver 580.159.04).

The diagnostic completed and all four GPU paths produced the expected JSON output. Top-1 agreement was 49/50 for retained, 48/50 for sequential, 49/50 for pre-N16, and 49/50 for rows. Maximum absolute logit deltas were 0.45–0.53 with mean RMS error about 0.073–0.075. This confirms the earlier 29/50 token oracle result is not a launch-order-only failure, but it does not establish exact numerical parity; the acceptance gate remains failed and no correctness claim is made.

The production telemetry is actionable: prefill median was 11,541.9 tok/s (10 measured iterations), with 20.26 ms median profiled time. FFN gate/up (31.6%), FFN down (21.9%), and FFN elementwise (12.4%) account for 66.0% of traced time; attention is 20.1%. The roofline classifier marks FFN compute/efficiency as the next lever. The retained path used `bstage-n16-m32` for QKV, `mma4-regq` attention, and `bstage-n16` for FFN.

Evidence: `benchmarks/glcuda-t4-wave58-evidence.zip`. The ZIP includes the pinned build, parity logs, PTXAS register report, teacher-forced logits, and profiled/unprofiled benchmark JSON. Results are diagnostic observations only; the `glbench` oracle correctly reports 29/50 agreement for the production profile.
