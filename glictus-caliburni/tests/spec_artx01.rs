//! Wave 4: the ARTX01 spec's unit tests.
//!
//! These are the acceptance tests named in the boilerplate spec, kept in
//! `tests/` alongside the other suites rather than as a `mod tests` inside
//! `lib.rs`.

use glictus_caliburni::constants::GLLM_MAGIC;
use glictus_caliburni::types::execution::Device;
use glictus_caliburni::types::extension::ExtensionUri;
use glictus_caliburni::ExecutionUnitHeader;
use glictus_caliburni::types::package::GllmPackageMeta;
use glictus_caliburni::types::tensor::DType;

#[test]
fn test_magic_constant() {
    assert_eq!(GLLM_MAGIC, 0x474C4C4D);
    // "GLLM" in ASCII
    assert_eq!(&GLLM_MAGIC.to_be_bytes(), b"GLLM");
}

#[test]
fn test_dtype_roundtrip() {
    let dtypes = [
        DType::F32,
        DType::F16,
        DType::Bf16,
        DType::Q4_0,
        DType::Q8_0,
        DType::Q8K,
        DType::Q4K,
        DType::Q4Km,
    ];
    for dtype in dtypes {
        let code = dtype.to_code();
        let roundtrip = DType::from_code(code).expect("known code round-trips");
        assert_eq!(dtype, roundtrip);
    }
}

#[test]
fn test_dtype_bytes_per_element() {
    // ARTX03 split the API: exact Option<usize> for non-block dtypes,
    // approx f64 (this ARTX01 contract) for memory estimation.
    assert_eq!(DType::F32.approx_bytes_per_element(), 4.0);
    assert_eq!(DType::F16.approx_bytes_per_element(), 2.0);
    assert_eq!(DType::Q8_0.approx_bytes_per_element(), 1.0);
    assert_eq!(DType::Q4_0.approx_bytes_per_element(), 0.5);
}

#[test]
fn test_extension_uri_parse() {
    let uri = "gllm:transformer/standard@v1";
    let parsed = ExtensionUri::parse(uri).expect("valid uri parses");
    // ARTX03: ExtensionUri is a string newtype; parts are methods now.
    assert_eq!(parsed.namespace(), "gllm");
    assert_eq!(parsed.category(), "transformer");
    assert_eq!(parsed.name(), "standard");
    assert_eq!(parsed.version(), "v1");
    assert!(parsed.is_official());
    assert_eq!(parsed.to_string(), uri);
}

#[test]
fn test_extension_uri_moe() {
    let uri = "gllm:transformer/moe@v1";
    let parsed = ExtensionUri::parse(uri).expect("valid uri parses");
    assert_eq!(parsed.name(), "moe");
}

#[test]
fn test_extension_uri_custom() {
    let uri = "myorg:custom/fused@v2";
    let parsed = ExtensionUri::parse(uri).expect("valid uri parses");
    assert!(!parsed.is_official());
    assert_eq!(parsed.namespace(), "myorg");
}

#[test]
fn test_device_from_str() {
    assert_eq!(Device::from_str("cpu"), Some(Device::Cpu));
    assert_eq!(Device::from_str("cuda:0"), Some(Device::Cuda(0)));
    assert_eq!(Device::from_str("cuda:1"), Some(Device::Cuda(1)));
    assert_eq!(Device::from_str("vulkan:0"), Some(Device::Vulkan(0)));
    assert_eq!(Device::from_str("metal"), Some(Device::Metal));
    assert_eq!(Device::from_str("unknown"), None);
}

#[test]
fn test_layer_header_validate() {
    // ARTX04 hybrid: the ARTX01 12-byte LayerHeader was retired; every
    // unit file shares the 16-byte ExecutionUnitHeader, whose bytes 8..12
    // now carry ARTX04's tensor_count. This pins the same contract the
    // old test did: good header validates, bad magic is rejected.
    let header = ExecutionUnitHeader::new_v1_with_tensors(10);
    let parsed = ExecutionUnitHeader::parse(&header.to_bytes()).expect("valid header parses");
    assert_eq!(parsed.tensor_count, 10);

    let mut bad = header.to_bytes();
    bad[0..4].copy_from_slice(&0xDEADBEEFu32.to_be_bytes());
    assert!(ExecutionUnitHeader::parse(&bad).is_err());
}

#[test]
fn test_layer_expected_filename() {
    assert_eq!(GllmPackageMeta::expected_layer_filename(0), "GLLMTensorLayer-0000.gllm");
    assert_eq!(GllmPackageMeta::expected_layer_filename(42), "GLLMTensorLayer-0042.gllm");
    assert_eq!(GllmPackageMeta::expected_layer_filename(79), "GLLMTensorLayer-0079.gllm");
}
