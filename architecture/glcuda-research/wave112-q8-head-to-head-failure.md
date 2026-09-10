# Wave 112 - Q8 head-to-head pre-measurement failure

## Result

Wave 112 produced no production metric. The Tesla T4 and source gates passed,
including 66 host tests, all 33 serialized CUDA parity tests, and the release
`glbench` build. The notebook then stopped in the pinned-model fetch phase,
before cloning or building llama.cpp and before any of the planned 30
production sessions.

The primary failure was notebook composition: the model-fetch cell inherited
the Wave 57 `sha256_file` call, but the Wave 112 bootstrap did not define that
helper. The first download reached the expected byte count and then raised a
`NameError`; subsequent range retries correctly received HTTP 416 because the
partial file was already complete. The failure handler name was also inherited
from the older template, so the normal partial-result archive was not emitted.

No throughput value is valid from this run. In particular, Wave 111 still has
only its direct microbenchmark evidence; the fastest valid production record
remains 12,232.7 tok/s from Wave 103.

## Evidence

- Kaggle kernel: `jinxsuperdev/glcuda-t4-wave112-q8-head-to-head`, version 1
- Source revision: `6ba97d98549330e76e3961227d53759389e4cfd3`
- Notebook commit: `8f7a32ff212952d336f3180d5d29090508eaac40`
- Raw log archive: `benchmarks/glcuda-t4-wave112-h2h-q8-failure.zip`
- Archive SHA-256: `ae3c555e4dc3bb15a1db6dc293bd59b44139ead3ec0fdf98407542c28786c013`

## Next gate

Wave 113 should add the missing hash helper, use the current `fail_with_archive`
handler consistently, and explicitly promote an already-complete `.part` file
before issuing another range request. It must then rerun the entire pipeline;
the successful tests from this failed notebook do not waive any Wave 113 gate.
