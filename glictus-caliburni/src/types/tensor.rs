use serde::{Deserialize, Serialize};
use crate::constants::dtype_codes;
use crate::error::{GllmError, GllmResult};

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DType {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
    Fp8E5m2,
    Q4_0,
    Q4_1,
    Q4K,
    Q4Km,
    Q4Ks,
    Q8_0,
    Q8K,
    I32,
}

impl DType {
    pub fn from_code(code: u16) -> GllmResult<Self> {
        match code {
            dtype_codes::FP32 => Ok(Self::F32),
            dtype_codes::FP16 => Ok(Self::F16),
            dtype_codes::BF16 => Ok(Self::Bf16),
            dtype_codes::FP8_E4M3 => Ok(Self::Fp8E4m3),
            dtype_codes::FP8_E5M2 => Ok(Self::Fp8E5m2),
            dtype_codes::Q4_0 => Ok(Self::Q4_0),
            dtype_codes::Q4_1 => Ok(Self::Q4_1),
            dtype_codes::Q4_K => Ok(Self::Q4K),
            dtype_codes::Q4_K_M => Ok(Self::Q4Km),
            dtype_codes::Q4_K_S => Ok(Self::Q4Ks),
            dtype_codes::Q8_0 => Ok(Self::Q8_0),
            dtype_codes::Q8_K => Ok(Self::Q8K),
            dtype_codes::I32 => Ok(Self::I32),
            code => Err(GllmError::UnsupportedDtype(code)),
        }
    }

    pub fn to_code(self) -> u16 {
        match self {
            Self::F32 => dtype_codes::FP32,
            Self::F16 => dtype_codes::FP16,
            Self::Bf16 => dtype_codes::BF16,
            Self::Fp8E4m3 => dtype_codes::FP8_E4M3,
            Self::Fp8E5m2 => dtype_codes::FP8_E5M2,
            Self::Q4_0 => dtype_codes::Q4_0,
            Self::Q4_1 => dtype_codes::Q4_1,
            Self::Q4K => dtype_codes::Q4_K,
            Self::Q4Km => dtype_codes::Q4_K_M,
            Self::Q4Ks => dtype_codes::Q4_K_S,
            Self::Q8_0 => dtype_codes::Q8_0,
            Self::Q8K => dtype_codes::Q8_K,
            Self::I32 => dtype_codes::I32,
        }
    }

    /// Bytes per element (approximate for quantized types)
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            Self::F32 | Self::I32 => 4.0,
            Self::F16 | Self::Bf16 => 2.0,
            Self::Fp8E4m3 | Self::Fp8E5m2 => 1.0,
            Self::Q8_0 | Self::Q8K => 1.0,
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::Q4Km | Self::Q4Ks => 0.5,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self,
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::Q4Km | Self::Q4Ks |
            Self::Q8_0 | Self::Q8K
        )
    }
}

/// Tensor shape (list of dimensions)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<u32>);

impl Shape {
    pub fn new(dims: Vec<u32>) -> Self { Self(dims) }
    pub fn rank(&self) -> usize { self.0.len() }
    pub fn numel(&self) -> u64 { self.0.iter().map(|&d| d as u64).product() }
}

/// Entry in tensor index (per-tensor metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub offset: u64,   // offset dalam layer file
    pub size: u64,     // size dalam bytes
}

impl TensorEntry {
    /// Estimated memory footprint
    pub fn memory_bytes(&self) -> u64 {
        (self.shape.numel() as f64 * self.dtype.bytes_per_element()) as u64
    }
}
