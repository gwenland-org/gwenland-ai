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

/// Write a real unit file containing the named tensors, each filled with a
/// distinct repeating byte so reads can be told apart.
///
/// Goes through [`write_unit_file`](crate::layer_io::write_unit_file) — the
/// same writer the converter uses — so fixtures cannot drift from the real
/// format. Each tensor is declared 1-D `[size]` of `Q8_0` (one byte per
/// element), so `size` is both the element count and the exact byte count.
/// These fixtures exercise layout only — the payload is filler, never decoded.
pub(crate) fn write_test_layer(path: &Path, tensors: &[(&str, u64)]) {
    let payloads: Vec<Vec<u8>> = tensors
        .iter()
        .enumerate()
        .map(|(i, (_, size))| vec![(i as u8).wrapping_add(1); *size as usize])
        .collect();
    let shapes: Vec<[u64; 1]> = tensors.iter().map(|(_, size)| [*size]).collect();

    let specs: Vec<(&str, &[u64], crate::manifest::DType, &[u8])> = tensors
        .iter()
        .zip(&shapes)
        .zip(&payloads)
        .map(|(((name, _), shape), data)| {
            (
                *name,
                shape.as_slice(),
                crate::manifest::DType::Q8_0,
                data.as_slice(),
            )
        })
        .collect();

    crate::layer_io::write_unit_file(path, &specs).expect("test layer writes");
}

/// Metadata for Qwen2.5-0.5B-Instruct — the reference model these crates are
/// verified against. Values are the real ones read from its GGUF (dim 896,
/// 14 query heads / 2 KV heads, head_dim 64, ctx 32768, rope base 1e6, rms
/// eps 1e-6 — confirmed via `gwen info` on the actual file, NOT the 1e-5
/// default; this is the exact value whose mismatch against a hardcoded 1e-5
/// produced garbage E2E output before rms_eps was threaded through
/// ModelMetadata), so tests asserting on derived sizes are checking a shape
/// that actually ships.
pub(crate) fn qwen05b_metadata() -> crate::manifest::ModelMetadata {
    crate::manifest::ModelMetadata {
        vocab_size: 151_936,
        context_length: 32_768,
        embedding_length: 896,
        num_layers: 24,
        num_heads: 14,
        head_count_kv: 2,
        rope_dims: Some(64),
        rope_freq_base: Some(1_000_000.0),
        rope_scaling: None,
        expert_count: None,
        expert_used_count: None,
        sliding_window: None,
        attention_bias: None,
        rms_eps: Some(1e-6),
        // Qwen2.5's real dual EOS: <|im_end|> and <|endoftext|>.
        eos_token_ids: vec![151_645, 151_643],
    }
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
                    "file": crate::manifest::format_layer_filename(i),
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
                "file": crate::constants::SHARED_FILENAME,
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
                "file": crate::constants::SHARED_FILENAME,
                "checksum": format!("sha256:{DUMMY_SHA256}"),
                "tensors": [
                    { "name": "token_embeddings", "shape": [128256, 8192],
                      "dtype": "Q4_K_M", "offset": 0, "size": 590558208u64 },
                    { "name": "output_norm.weight", "shape": [8192],
                      "dtype": "F32", "offset": 590558208u64, "size": 32768 }
                ]
            },
            "layers": [
                { "index": 0, "file": "GLLMTensorLayer-0000.gllm",
                  "checksum": format!("sha256:{DUMMY_SHA256}"),
                  "type": "gllm:transformer/standard@v1",
                  "tensors": [
                      { "name": "attn_q.weight", "shape": [8192, 8192],
                        "dtype": "Q4_K_M", "offset": 0, "size": 37748736 }
                  ] },
                { "index": 1, "file": "GLLMTensorLayer-0001.gllm",
                  "checksum": format!("sha256:{DUMMY_SHA256}"),
                  "type": "gllm:transformer/standard@v1",
                  "tensors": [],
                  "device": "cuda:0" }
            ],
            "projector": {
                "file": crate::constants::PROJECTOR_FILENAME,
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

        make_test_gllm_file(&dir.join(crate::constants::SHARED_FILENAME));
        for i in 0..num_layers {
            make_test_gllm_file(&dir.join(crate::manifest::format_layer_filename(i)));
        }
        let shared_hex = sha256_file(&dir.join(crate::constants::SHARED_FILENAME)).unwrap();
        let layer_hex: Vec<String> = (0..num_layers)
            .map(|i| sha256_file(&dir.join(crate::manifest::format_layer_filename(i))).unwrap())
            .collect();
        let manifest = manifest_json_with_checksums(num_layers, &shared_hex, &layer_hex);
        std::fs::write(dir.join("gllm.json"), manifest).unwrap();
    }
}

/// Write a package the ARTX05 runtime can actually execute, in a fresh
/// temporary directory.
///
/// Differs from [`fixtures::write_manifest_package`] in that every unit file
/// carries a real tensor index, so `LayerFile::parse` succeeds on the mapped
/// bytes — `make_test_gllm_file` writes only a header plus filler, which the
/// runtime rejects. Checksums are computed from the bytes actually written.
pub(crate) fn write_runtime_package(num_layers: u32) -> tempfile::TempDir {
    use crate::checksum::sha256_file;

    let dir = tempfile::TempDir::new().expect("tempdir");
    let root = dir.path();

    write_test_layer(&root.join(crate::constants::SHARED_FILENAME), &[("token_embeddings", 128)]);
    for i in 0..num_layers {
        write_test_layer(
            &root.join(crate::manifest::format_layer_filename(i)),
            &[("attn_q.weight", 64), ("attn_k.weight", 32)],
        );
    }

    let shared_hex = sha256_file(&root.join(crate::constants::SHARED_FILENAME)).unwrap();
    let layer_hex: Vec<String> = (0..num_layers)
        .map(|i| sha256_file(&root.join(crate::manifest::format_layer_filename(i))).unwrap())
        .collect();

    let layers: Vec<serde_json::Value> = (0..num_layers)
        .map(|i| {
            serde_json::json!({
                "index": i,
                "file": crate::manifest::format_layer_filename(i),
                "checksum": format!("sha256:{}", layer_hex[i as usize]),
                "type": "gllm:transformer/standard@v1",
                "tensors": [
                    { "name": "attn_q.weight", "shape": [64], "dtype": "Q8_0",
                      "offset": 0, "size": 64 },
                    { "name": "attn_k.weight", "shape": [32], "dtype": "Q8_0",
                      "offset": 64, "size": 32 }
                ]
            })
        })
        .collect();

    // Metadata is Qwen2.5-0.5B's shape scaled to this fixture's layer count,
    // so KV cache sizing exercises a real GQA config (head_dim 896/14 = 64).
    let manifest = serde_json::json!({
        "gllm_version": "1.0.0",
        "model_id": "org.gwenland.runtime-fixture",
        "architecture": "transformer",
        "metadata": {
            "vocab_size": 151936,
            "context_length": 32768,
            "embedding_length": 896,
            "num_layers": num_layers,
            "num_heads": 14,
            "head_count_kv": 2
        },
        "shared": {
            "file": crate::constants::SHARED_FILENAME,
            "checksum": format!("sha256:{shared_hex}"),
            "tensors": [
                { "name": "token_embeddings", "shape": [128], "dtype": "Q8_0",
                  "offset": 0, "size": 128 }
            ]
        },
        "layers": layers,
        "extensions": ["gllm:transformer/standard@v1"]
    });

    std::fs::write(root.join("gllm.json"), manifest.to_string()).unwrap();
    dir
}
