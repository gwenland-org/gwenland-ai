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

/// Reusable `gllm.json` fixtures (ARTX03).
pub(crate) mod fixtures {
    use std::path::Path;

    /// A well-formed but meaningless digest for manifests that will only
    /// be parsed/validated, never checksum-verified against real files.
    pub(crate) const DUMMY_SHA256: &str =
        "0000000000000000000000000000000000000000000000000000000000000000";

    /// Minimal valid manifest with `num_layers` sequential layer entries,
    /// per-file checksums supplied by the caller (bare hex, no prefix).
    pub(crate) fn manifest_json_with_checksums(
        num_layers: u32,
        shared_hex: &str,
        layer_hex: &[String],
    ) -> String {
        let layers: Vec<serde_json::Value> = (0..num_layers)
            .map(|i| {
                let hex = layer_hex
                    .get(i as usize)
                    .map(String::as_str)
                    .unwrap_or(DUMMY_SHA256);
                serde_json::json!({
                    "index": i,
                    "file": format!("layer_{i:03}.gllm"),
                    "checksum": format!("sha256:{hex}"),
                    "type": "gllm:transformer/standard@v1",
                    "tensors": []
                })
            })
            .collect();
        serde_json::json!({
            "gllm_version": "1.0.0",
            "model_id": "org.gwenland.test-model",
            "architecture": "transformer",
            "metadata": {
                "vocab_size": 1000,
                "context_length": 2048,
                "embedding_length": 64,
                "num_layers": num_layers,
                "num_heads": 8,
                "head_count_kv": 8
            },
            "shared": {
                "file": "shared.gllm",
                "checksum": format!("sha256:{shared_hex}"),
                "tensors": [
                    { "name": "token_embeddings", "shape": [1000, 64],
                      "dtype": "F32", "offset": 0, "size": 256000 }
                ]
            },
            "layers": layers,
            "extensions": ["gllm:transformer/standard@v1"]
        })
        .to_string()
    }

    /// Minimal valid manifest with dummy checksums.
    pub(crate) fn minimal_manifest_json(num_layers: u32) -> String {
        manifest_json_with_checksums(num_layers, DUMMY_SHA256, &[])
    }

    /// Full-featured manifest: GQA metadata, quantization, projector,
    /// device hints, custom metadata — 2 layers.
    pub(crate) fn full_manifest_json() -> String {
        serde_json::json!({
            "gllm_version": "1.0.0",
            "model_id": "org.gwenland.qwen-test-q4km",
            "architecture": "transformer",
            "parameters": 70_000_000_000u64,
            "quantization": "Q4_K_M",
            "metadata": {
                "vocab_size": 128256,
                "context_length": 8192,
                "embedding_length": 8192,
                "num_layers": 2,
                "num_heads": 64,
                "head_count_kv": 8,
                "rope_dims": 128,
                "rope_freq_base": 500000.0,
                "rope_scaling": { "type": "linear", "factor": 2.0 },
                "sliding_window": 4096,
                "attention_bias": false
            },
            "shared": {
                "file": "shared.gllm",
                "checksum": format!("sha256:{DUMMY_SHA256}"),
                "tensors": [
                    { "name": "token_embeddings", "shape": [128256, 8192],
                      "dtype": "Q4_K_M", "offset": 0, "size": 590558208u64 },
                    { "name": "output_norm.weight", "shape": [8192],
                      "dtype": "F32", "offset": 590558208u64, "size": 32768 }
                ]
            },
            "layers": [
                { "index": 0, "file": "layer_000.gllm",
                  "checksum": format!("sha256:{DUMMY_SHA256}"),
                  "type": "gllm:transformer/standard@v1",
                  "tensors": [
                      { "name": "attn_q.weight", "shape": [8192, 8192],
                        "dtype": "Q4_K_M", "offset": 0, "size": 37748736 }
                  ] },
                { "index": 1, "file": "layer_001.gllm",
                  "checksum": format!("sha256:{DUMMY_SHA256}"),
                  "type": "gllm:transformer/standard@v1",
                  "tensors": [],
                  "device": "cuda:0" }
            ],
            "projector": {
                "file": "projector.gllm",
                "checksum": format!("sha256:{DUMMY_SHA256}"),
                "type": "gllm:projector/linear@v1",
                "tensors": []
            },
            "extensions": [
                "gllm:transformer/standard@v1",
                "gllm:projector/linear@v1"
            ],
            "custom_metadata": {
                "converted_by": "glictus-caliburni-test",
                "source": "test-fixture"
            }
        })
        .to_string()
    }

    /// Write a complete on-disk package: valid unit files plus a manifest
    /// whose checksums match the real file contents.
    pub(crate) fn write_manifest_package(dir: &Path, num_layers: u32) {
        use crate::checksum::sha256_file;
        use crate::test_helpers::make_test_gllm_file;

        make_test_gllm_file(&dir.join("shared.gllm"));
        for i in 0..num_layers {
            make_test_gllm_file(&dir.join(format!("layer_{i:03}.gllm")));
        }
        let shared_hex = sha256_file(&dir.join("shared.gllm")).unwrap();
        let layer_hex: Vec<String> = (0..num_layers)
            .map(|i| sha256_file(&dir.join(format!("layer_{i:03}.gllm"))).unwrap())
            .collect();
        let manifest = manifest_json_with_checksums(num_layers, &shared_hex, &layer_hex);
        std::fs::write(dir.join("gllm.json"), manifest).unwrap();
    }
}
