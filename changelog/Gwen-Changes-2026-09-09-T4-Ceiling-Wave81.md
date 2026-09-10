# T4 Ceiling Wave 81 - M32 wide-grid rejection

- Added a self-contained hard-T4 notebook for the N16/M32 wide-grid test.
- Verified 33/33 CUDA parity, bit-exact direct output, zero spills, and the
  expected 72-register versus 64-register resource difference.
- Measured 161.380 us for retained wide N16 and 173.699 us for M32: the
  candidate is 7.09% slower and fails the frozen 1.10x direct gate.
- Correctly skipped production `glbench`; no throughput claim was made.
- Removed the rejected product dispatch and local harness after archiving the
  notebook, structured result and raw T4 evidence.
- Compressed the embedded reconstruction patch so Kaggle's submitted source is
  211,186 bytes while preserving the decoded patch SHA-256.
