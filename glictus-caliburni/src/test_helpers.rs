//! Shared helpers for in-module unit tests (compiled only under `cfg(test)`).

use std::path::Path;

use crate::execution_unit::ExecutionUnitHeader;

/// Write a minimal valid GLLM execution unit file: a v1 header followed by
/// a few dummy payload bytes. Returns the full file contents.
pub(crate) fn make_test_gllm_file(path: &Path) -> Vec<u8> {
    let mut contents = ExecutionUnitHeader::new_v1().to_bytes().to_vec();
    contents.extend_from_slice(b"dummy-payload");
    std::fs::write(path, &contents).expect("test file writes");
    contents
}
