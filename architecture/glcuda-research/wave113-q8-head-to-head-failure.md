# Wave 113 - Q8 head-to-head probe failure

Wave 113 repaired the model-fetch failure and reached the expensive external
comparison setup: the pinned Q8 model verified, llama.cpp built at the exact
commit, CUDA smoke passed on one Tesla T4, and both GwenLand dispatch probes
selected the intended N16-prefetch/MMA4-AV contract. It stopped before the
first measured session because the production cell still called the old
`save_log` helper while the reconstructed bootstrap exports `save`.

No production throughput number is valid. The run completed zero of the
planned 30 sessions. Wave 114's bounded repair is to normalize that helper
name; all correctness, tokenizer, and 30-session gates remain unchanged.

Evidence archive: `benchmarks/glcuda-t4-wave113-h2h-q8-failure.zip`

Archive SHA-256:
`115897f07c2ae620a1f4dcaa992e757581113f674850e330dfd9125b34f1af1f`
