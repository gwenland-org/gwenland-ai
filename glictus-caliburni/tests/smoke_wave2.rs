use glictus_caliburni::types::execution::Device;

#[test]
fn fromstr_trait_does_not_recurse() {
    assert_eq!(Device::from_str("cuda:1"), Some(Device::Cuda(1)));
    let parsed: Device = "cuda:1".parse().expect("cuda:1 should parse");
    assert_eq!(parsed, Device::Cuda(1));
    assert!("nonsense".parse::<Device>().is_err());
}

#[test]
fn manifest_json_roundtrip() {
    let json = r#"{
        "gllm_version": "1.0.0",
        "model_id": "qwen2.5-0.5b",
        "architecture": "transformer",
        "parameters": 494032768,
        "quantization": "Q4_K_M",
        "metadata": {
            "vocab_size": 151936, "context_length": 32768,
            "embedding_length": 896, "num_layers": 24, "num_heads": 14,
            "head_count_kv": 2, "rope_dims": 64, "rope_freq_base": 1000000.0
        },
        "shared": { "file": "GLLMShared.gllm", "checksum": "abc", "tensors": [
            { "name": "token_embd", "shape": [151936, 896], "dtype": "Q4_K", "offset": 0, "size": 68067328 }
        ]},
        "layers": [
            { "index": 0, "file": "GLLMTensorLayer-0000.gllm", "checksum": "def",
              "type": "gllm:transformer/standard@v1", "tensors": [], "device": "cpu" }
        ],
        "projector": null,
        "extensions": []
    }"#;
    let m: glictus_caliburni::GllmManifest = serde_json::from_str(json).expect("manifest parses");
    assert_eq!(m.num_layers(), 1);
    assert_eq!(m.estimated_memory_bytes(), 68067328);
    let back = serde_json::to_string(&m).expect("serializes");
    assert!(back.contains("qwen2.5-0.5b"));
}
