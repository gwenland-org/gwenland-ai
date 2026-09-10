# Wave 115 - Q8 head-to-head tokenizer environment failure

Wave 115 passed the T4, host, CUDA parity, Q8 model, pinned llama.cpp build,
CUDA smoke, and both GwenLand dispatch probes. It stopped before measurement
when the exact tokenizer helper was launched with only `CARGO_TARGET_DIR` in
its environment. On Kaggle's bootstrapped Rust image that omitted
`CARGO_HOME`/`RUSTUP_HOME`, so rustup reported that no default toolchain was
configured.

Zero of 30 production sessions completed and no throughput number is valid.
Wave 116 will pass the complete bootstrap cargo environment to the helper;
all tokenizer, parity, and production gates remain unchanged.

Partial archive: `benchmarks/glcuda-t4-wave115-h2h-q8-failure.zip`

SHA-256: `c089491c295a7e4fcfe01ae17544d6a1fa20f76af5018094aa0ed85d2ac848ee`
