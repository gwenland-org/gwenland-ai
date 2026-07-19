/// GLLM magic bytes: "GLLM" = 0x474C4C4D
pub const GLLM_MAGIC: u32 = 0x474C4C4D;

/// Current format version
pub const GLLM_VERSION_MAJOR: u8 = 1;
pub const GLLM_VERSION_MINOR: u8 = 0;

/// Default manifest filename
pub const MANIFEST_FILENAME: &str = "gllm.json";

/// Shared component filename
pub const SHARED_FILENAME: &str = "shared.gllm";

/// Checksums filename
pub const CHECKSUMS_FILENAME: &str = "checksums.sha256";

/// Tensor data alignment (64 bytes for DMA efficiency)
pub const TENSOR_ALIGNMENT: usize = 64;

/// Layer file naming pattern: layer_NNN.gllm
pub const LAYER_FILE_PREFIX: &str = "layer_";
pub const LAYER_FILE_EXTENSION: &str = ".gllm";

/// Data type codes (dari ARTX spec)
pub mod dtype_codes {
    pub const FP32: u16 = 0x0001;
    pub const FP16: u16 = 0x0002;
    pub const BF16: u16 = 0x0003;
    pub const FP8_E4M3: u16 = 0x0004;
    pub const FP8_E5M2: u16 = 0x0005;
    pub const Q4_0: u16 = 0x0010;
    pub const Q4_1: u16 = 0x0011;
    pub const Q4_K: u16 = 0x0012;
    pub const Q4_K_M: u16 = 0x0013;
    pub const Q4_K_S: u16 = 0x0014;
    // 0x0015 / 0x0018: added for the GGUF converter (ARTX07) — real
    // Q4_K_M models carry Q5_0 fallback rows and Q6_K output heads.
    // Format not yet frozen, so this is an addition, not a minor bump.
    pub const Q5_0: u16 = 0x0015;
    pub const Q6_K: u16 = 0x0018;
    pub const Q8_0: u16 = 0x0020;
    pub const Q8_K: u16 = 0x0021;
    pub const I32: u16 = 0x0030;
}
