# Gwen Changes — 2026-08-25 — T4 Ceiling Wave 8

## Added

- Added the `gl_quantize_q8_rowcta` PTX kernel for batched Q8_0 activation quantization.
- Added `GLCUDA_Q8_ROWCTA=1` as an opt-in prefill-only dispatch switch.
- Added a bit-exact hardware parity test comparing the retained K32 quantizer with the row-CTA kernel.
- Added a 244x896 old-vs-row-CTA diagnostic to the CUDA benchmark example.
- Added `notebooks/glcuda_t4_ceiling_wave8.ipynb`, which fetches the pinned model and runs clean Wave 4 versus Wave 8 T4 gates.

## Preserved

- Q8_0 remains one f32 activation scale per K32 block.
- Quantization reduction order, rounding, clamping, bytes, and scales are unchanged.
- Single-row decode remains on `gl_quantize_q8`.
- Retained Wave 4 attention and sm_75 MMA PTX are unchanged.
- Rejected Wave 5, Wave 6, and Wave 7 candidates are excluded from the notebook patch stack.

## Validation

- Clean candidate `cargo check`: pass.
- Clean `glcuda` library tests: 43 passed.
- Clean `glcuda` parity test binary: 22 passed locally; real CUDA execution is deferred to Kaggle T4 and is a mandatory gate.
- Embedded Wave 8 patch SHA-256: `03746e114f8228f5aebfe43399b29e179c17c805f084fafd754f9d0cc878fb0e`.

