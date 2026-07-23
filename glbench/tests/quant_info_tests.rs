//! Integration tests for `glbench quant-info`.
//!
//! glbench does not import glictus-caliburni, so these fixtures hand-write
//! `gllm.json` in the real on-disk shape (a package *directory* containing
//! `gllm.json` + `GLLMShared.gllm` + `GLLMTensorLayer-NNNN.gllm`) rather than
//! constructing an actual ZIP — real `.gllm` packages are directories today;
//! ZIP-archive reading is not yet implemented anywhere in this workspace
//! (glictus-caliburni ARTX06).

use std::fs;
use std::path::Path;

use glbench::quant_info::{run_quant_info, QuantInfoArgs, QuantInfoSession};

/// Writes a minimal but structurally real `gllm.json` package directory.
/// `shared_dtypes` and `layer_dtypes` each become one tensor per entry.
fn write_package(dir: &Path, model_id: &str, architecture: &str, num_layers: u32, shared_dtypes: &[&str], layer_dtypes: &[&str]) {
    fs::create_dir_all(dir).unwrap();

    let shared_tensors: Vec<String> = shared_dtypes
        .iter()
        .enumerate()
        .map(|(i, dtype)| {
            format!(
                r#"{{ "name": "shared_{i}", "shape": [4], "dtype": "{dtype}", "offset": 0, "size": 16 }}"#
            )
        })
        .collect();

    let layer_tensors: Vec<String> = layer_dtypes
        .iter()
        .enumerate()
        .map(|(i, dtype)| {
            format!(
                r#"{{ "name": "layer_{i}", "shape": [4], "dtype": "{dtype}", "offset": 0, "size": 16 }}"#
            )
        })
        .collect();

    let manifest = format!(
        r#"{{
  "gllm_version": "1.0.0",
  "model_id": "{model_id}",
  "architecture": "{architecture}",
  "metadata": {{
    "vocab_size": 1000,
    "context_length": 2048,
    "embedding_length": 64,
    "num_layers": {num_layers},
    "num_heads": 8,
    "head_count_kv": 8
  }},
  "shared": {{
    "file": "GLLMShared.gllm",
    "checksum": "sha256:{hex}",
    "tensors": [{shared_list}]
  }},
  "layers": [
    {{ "index": 0, "file": "GLLMTensorLayer-0000.gllm",
       "checksum": "sha256:{hex}",
       "type": "gllm:transformer/standard@v1",
       "tensors": [{layer_list}] }}
  ]
}}"#,
        hex = "0".repeat(64),
        shared_list = shared_tensors.join(", "),
        layer_list = layer_tensors.join(", "),
    );

    fs::write(dir.join("gllm.json"), manifest).unwrap();
    fs::write(dir.join("GLLMShared.gllm"), b"dummy-shared-bytes").unwrap();
    fs::write(dir.join("GLLMTensorLayer-0000.gllm"), b"dummy-layer-bytes").unwrap();
}

#[test]
fn test_quant_info_parses_model_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("model.gllm");
    write_package(&pkg, "org.gwenland.test-model", "qwen2", 24, &["F32"], &["GQ4A"]);

    let out = tmp.path().join("out.json");
    run_quant_info(QuantInfoArgs { model: pkg.clone(), out: Some(out.clone()) }).unwrap();

    let session = QuantInfoSession::from_json(&glbench::export::json::parse(&fs::read_to_string(&out).unwrap()).unwrap()).unwrap();
    assert_eq!(session.model_name, "org.gwenland.test-model");
    assert_eq!(session.architecture, "qwen2");
    assert_eq!(session.num_layers, 24);
}

#[test]
fn test_quant_info_dtype_tally_gq4a_only() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("model.gllm");
    write_package(&pkg, "m", "arch", 1, &["GQ4A", "GQ4A"], &["GQ4A", "GQ4A", "GQ4A"]);

    let out = tmp.path().join("out.json");
    run_quant_info(QuantInfoArgs { model: pkg, out: Some(out.clone()) }).unwrap();
    let session = QuantInfoSession::from_json(&glbench::export::json::parse(&fs::read_to_string(&out).unwrap()).unwrap()).unwrap();

    assert_eq!(session.dtype_tally.get("GQ4A"), Some(&5));
    assert_eq!(session.total_tensors, 5);
    assert_eq!(session.quantized_tensors, 5);
}

#[test]
fn test_quant_info_dtype_tally_mixed_gq4a_gq2a() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("model.gllm");
    write_package(&pkg, "m", "arch", 1, &["F32", "F16"], &["GQ4A", "GQ2A", "GQ2A"]);

    let out = tmp.path().join("out.json");
    run_quant_info(QuantInfoArgs { model: pkg, out: Some(out.clone()) }).unwrap();
    let session = QuantInfoSession::from_json(&glbench::export::json::parse(&fs::read_to_string(&out).unwrap()).unwrap()).unwrap();

    assert_eq!(session.dtype_tally.get("GQ4A"), Some(&1));
    assert_eq!(session.dtype_tally.get("GQ2A"), Some(&2));
    assert_eq!(session.dtype_tally.get("F32"), Some(&1));
    assert_eq!(session.dtype_tally.get("F16"), Some(&1));
    assert_eq!(session.total_tensors, 5);
}

#[test]
fn test_quant_info_coverage_pct() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("model.gllm");
    // 2 kept (F32/F16) + 8 quantized (GQ4A) = 10 total -> 80% coverage.
    let layer_dtypes = vec!["GQ4A"; 8];
    let layer_refs: Vec<&str> = layer_dtypes.iter().map(|s| *s).collect();
    write_package(&pkg, "m", "arch", 1, &["F32", "F16"], &layer_refs);

    let out = tmp.path().join("out.json");
    run_quant_info(QuantInfoArgs { model: pkg, out: Some(out.clone()) }).unwrap();
    let session = QuantInfoSession::from_json(&glbench::export::json::parse(&fs::read_to_string(&out).unwrap()).unwrap()).unwrap();

    assert_eq!(session.total_tensors, 10);
    assert_eq!(session.quantized_tensors, 8);
    assert!((session.coverage_pct - 80.0).abs() < 1e-9);
}

#[test]
fn test_quant_info_file_size_mb() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("model.gllm");
    write_package(&pkg, "m", "arch", 1, &["F32"], &["GQ4A"]);

    let expected_bytes: u64 = fs::read_dir(&pkg)
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    let expected_mb = expected_bytes as f64 / 1024.0 / 1024.0;

    let out = tmp.path().join("out.json");
    run_quant_info(QuantInfoArgs { model: pkg, out: Some(out.clone()) }).unwrap();
    let session = QuantInfoSession::from_json(&glbench::export::json::parse(&fs::read_to_string(&out).unwrap()).unwrap()).unwrap();

    assert!(
        (session.file_size_mb - expected_mb).abs() < 0.01,
        "expected {expected_mb}, got {}",
        session.file_size_mb
    );
    assert_eq!(session.file_size_bytes, expected_bytes);
}

#[test]
fn test_quant_info_serializes_to_json() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("model.gllm");
    write_package(&pkg, "roundtrip-model", "moe", 3, &["F32"], &["GQ2A", "GQ2A"]);

    let out = tmp.path().join("out.json");
    run_quant_info(QuantInfoArgs { model: pkg.clone(), out: Some(out.clone()) }).unwrap();

    let text = fs::read_to_string(&out).unwrap();
    let json = glbench::export::json::parse(&text).unwrap();
    let session = QuantInfoSession::from_json(&json).unwrap();

    // Round-trip through to_json/from_json again to confirm it's stable.
    let back = QuantInfoSession::from_json(&session.to_json()).unwrap();
    assert_eq!(session, back);
    assert_eq!(session.model_name, "roundtrip-model");
    assert_eq!(session.num_layers, 3);
}

#[test]
fn test_quant_info_missing_package_errors() {
    let err = run_quant_info(QuantInfoArgs {
        model: Path::new("/definitely/not/a/real/path.gllm").to_path_buf(),
        out: None,
    })
    .unwrap_err();
    assert!(err.contains("does not exist"));
}

#[test]
fn test_quant_info_zip_archive_gives_actionable_error() {
    let tmp = tempfile::tempdir().unwrap();
    let zip_path = tmp.path().join("model.gllm");
    fs::write(&zip_path, b"PK\x03\x04rest-of-a-zip-file").unwrap();

    let err = run_quant_info(QuantInfoArgs { model: zip_path, out: None }).unwrap_err();
    assert!(err.contains("ARTX06"), "{err}");
}
