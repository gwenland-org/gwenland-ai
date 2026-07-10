//! GGUF file parser, written from scratch against the GGUF spec
//! (<https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>).
//!
//! Supports versions 1, 2 and 3 (little-endian). The file is memory-mapped
//! so tensor data access is zero-copy; only the header, metadata and tensor
//! index are eagerly parsed.

use std::collections::HashMap;
use std::fs::File;

use byteorder::{ByteOrder, LittleEndian};

use crate::error::GlError;

/// `"GGUF"` in little-endian byte order.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default tensor-data alignment when `general.alignment` is absent.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Fixed-size GGUF file header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Magic number, always [`GGUF_MAGIC`].
    pub magic: u32,
    /// Format version: 1, 2 or 3.
    pub version: u32,
    /// Number of tensors in the file.
    pub tensor_count: u64,
    /// Number of metadata key/value pairs.
    pub metadata_kv_count: u64,
}

/// GGML tensor element types that can appear in a GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // Q4_0/Q4_K/Q8_0 are the canonical GGML names
pub enum GgufDType {
    /// 32-bit float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// 4-bit quantization, block of 32: f16 scale + 16 packed bytes.
    Q4_0,
    /// 5-bit quantization, block of 32: f16 scale + 4 bytes of high bits +
    /// 16 packed nibble bytes. llama.cpp falls back to this for tensors
    /// whose row size is not a multiple of 256 (dequant lives in glproc).
    Q5_0,
    /// 8-bit quantization, block of 32: f16 scale + 32 signed bytes.
    Q8_0,
    /// 4-bit K-quantization, super-block of 256 (dequant not yet implemented).
    Q4_K,
    /// 6-bit K-quantization, super-block of 256: ql (low 4 bits) + qh (high 2
    /// bits) + 16 per-sub-block i8 scales + one f16 superblock scale.
    Q6_K,
    /// bfloat16.
    BF16,
    /// Any type this parser does not know how to dequantize.
    Unknown(u32),
}

impl GgufDType {
    /// Decode the on-disk ggml type id.
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgufDType::F32,
            1 => GgufDType::F16,
            2 => GgufDType::Q4_0,
            6 => GgufDType::Q5_0,
            8 => GgufDType::Q8_0,
            12 => GgufDType::Q4_K,
            14 => GgufDType::Q6_K,
            30 => GgufDType::BF16,
            other => GgufDType::Unknown(other),
        }
    }

    /// Number of elements per quantization block.
    pub fn block_numel(&self) -> Option<usize> {
        match self {
            GgufDType::F32 | GgufDType::F16 | GgufDType::BF16 => Some(1),
            GgufDType::Q4_0 | GgufDType::Q5_0 | GgufDType::Q8_0 => Some(32),
            GgufDType::Q4_K | GgufDType::Q6_K => Some(256),
            GgufDType::Unknown(_) => None,
        }
    }

    /// Bytes occupied by one block.
    pub fn block_bytes(&self) -> Option<usize> {
        match self {
            GgufDType::F32 => Some(4),
            GgufDType::F16 | GgufDType::BF16 => Some(2),
            GgufDType::Q4_0 => Some(18), // f16 scale + 16 bytes of nibbles
            GgufDType::Q5_0 => Some(22), // f16 scale + 4 high-bit bytes + 16 nibble bytes
            GgufDType::Q8_0 => Some(34), // f16 scale + 32 i8 quants
            GgufDType::Q4_K => Some(144),
            GgufDType::Q6_K => Some(210), // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
            GgufDType::Unknown(_) => None,
        }
    }

    /// Total byte size of a tensor with `numel` elements, if known.
    pub fn tensor_bytes(&self, numel: usize) -> Option<usize> {
        let bn = self.block_numel()?;
        let bb = self.block_bytes()?;
        if numel % bn != 0 {
            return None;
        }
        Some(numel / bn * bb)
    }
}

/// Index entry for one tensor in the file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name, e.g. `"blk.0.attn_q.weight"`.
    pub name: String,
    /// Number of dimensions (1..=4 in practice).
    pub n_dimensions: u32,
    /// Dimension sizes. **GGUF order**: `dimensions[0]` is the fastest-moving
    /// (contiguous) axis — the reverse of row-major shape notation.
    pub dimensions: Vec<u64>,
    /// Element type of the stored data.
    pub dtype: GgufDType,
    /// Byte offset of this tensor's data, relative to the data section start.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }
}

/// A typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array of values.
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Coerce any integer variant to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::I8(v) => u64::try_from(*v).ok(),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::I16(v) => u64::try_from(*v).ok(),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) => u64::try_from(*v).ok(),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Coerce any numeric variant to `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            other => other.as_u64().map(|v| v as f32),
        }
    }

    /// Borrow the string payload, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Borrow the array payload, if this is an array.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// A parsed, memory-mapped GGUF file.
pub struct GgufFile {
    /// Parsed file header.
    pub header: GgufHeader,
    /// All metadata key/value pairs.
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor index, in file order.
    pub tensors: Vec<GgufTensorInfo>,
    /// Absolute byte offset where the tensor data section begins.
    pub data_offset: u64,
    mmap: memmap2::Mmap,
}

/// Cursor over the mmapped bytes with bounds-checked little-endian reads.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GlError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| GlError::Parse("GGUF: unexpected end of file".into()))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, GlError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GlError> {
        Ok(LittleEndian::read_u16(self.take(2)?))
    }

    fn u32(&mut self) -> Result<u32, GlError> {
        Ok(LittleEndian::read_u32(self.take(4)?))
    }

    fn u64(&mut self) -> Result<u64, GlError> {
        Ok(LittleEndian::read_u64(self.take(8)?))
    }

    fn f32(&mut self) -> Result<f32, GlError> {
        Ok(LittleEndian::read_f32(self.take(4)?))
    }

    fn f64(&mut self) -> Result<f64, GlError> {
        Ok(LittleEndian::read_f64(self.take(8)?))
    }

    /// Length-prefixed UTF-8 string. v1 uses u32 lengths, v2+ uses u64.
    fn string(&mut self, version: u32) -> Result<String, GlError> {
        let len = self.len_prefix(version)?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| GlError::Parse(format!("GGUF: invalid UTF-8 string: {e}")))
    }

    /// Count/length field: u32 in v1, u64 in v2+.
    fn len_prefix(&mut self, version: u32) -> Result<usize, GlError> {
        let v = if version == 1 {
            self.u32()? as u64
        } else {
            self.u64()?
        };
        usize::try_from(v).map_err(|_| GlError::Parse("GGUF: length overflows usize".into()))
    }

    fn value(&mut self, type_id: u32, version: u32) -> Result<GgufValue, GlError> {
        Ok(match type_id {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.u8()? as i8),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.u16()? as i16),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(self.f32()?),
            7 => GgufValue::Bool(self.u8()? != 0),
            8 => GgufValue::String(self.string(version)?),
            9 => {
                let elem_type = self.u32()?;
                let count = self.len_prefix(version)?;
                // Cap pre-allocation: a corrupt count must not OOM us.
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(self.value(elem_type, version)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(self.f64()?),
            other => {
                return Err(GlError::Parse(format!(
                    "GGUF: unknown metadata value type {other}"
                )))
            }
        })
    }
}

impl GgufFile {
    /// Open and mmap a GGUF file, parsing header, metadata and tensor index.
    pub fn open(path: &str) -> Result<Self, GlError> {
        let file = File::open(path)?;
        // SAFETY: we map the file read-only and never mutate the mapping.
        // The mapping lives as long as `GgufFile`, and all slices handed out
        // borrow from `self`, so they cannot outlive it. If another process
        // truncates the file concurrently, reads could fault — an accepted
        // (and conventional) risk for mmap-based model loading.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // Staging reads essentially the whole file once, region by region. Hint
        // the kernel to read ahead aggressively instead of faulting in 4 KB
        // pages on demand — on slow/virtualized disks this turns many small
        // random reads into large sequential ones. Best-effort; unix-only
        // (madvise has no Windows equivalent here).
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);
        let mut r = Reader::new(&mmap);

        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GlError::Parse(format!(
                "GGUF: bad magic 0x{magic:08x} (expected 0x{GGUF_MAGIC:08x})"
            )));
        }
        let version = r.u32()?;
        if !(1..=3).contains(&version) {
            return Err(GlError::Parse(format!(
                "GGUF: unsupported version {version} (supported: 1-3)"
            )));
        }

        let tensor_count = r.len_prefix(version)? as u64;
        let metadata_kv_count = r.len_prefix(version)? as u64;
        let header = GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        };

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = r.string(version)?;
            let type_id = r.u32()?;
            let value = r.value(type_id, version)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
        for _ in 0..tensor_count {
            let name = r.string(version)?;
            let n_dimensions = r.u32()?;
            let mut dimensions = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                let d = if version == 1 {
                    r.u32()? as u64
                } else {
                    r.u64()?
                };
                dimensions.push(d);
            }
            let dtype = GgufDType::from_u32(r.u32()?);
            let offset = r.u64()?;
            tensors.push(GgufTensorInfo {
                name,
                n_dimensions,
                dimensions,
                dtype,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .filter(|&a| a > 0)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        let data_offset = (r.pos as u64).div_ceil(alignment) * alignment;

        Ok(GgufFile {
            header,
            metadata,
            tensors,
            data_offset,
            mmap,
        })
    }

    /// Get a metadata value by key.
    pub fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Find a tensor's index entry by name.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get zero-copy tensor data as raw bytes.
    ///
    /// Returns `Err` (instead of panicking) if the tensor extends past the
    /// end of the file or its dtype has an unknown size.
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> Result<&[u8], GlError> {
        let size = info.dtype.tensor_bytes(info.numel()).ok_or_else(|| {
            GlError::UnsupportedDtype(format!("{:?} (tensor {})", info.dtype, info.name))
        })?;
        let start = usize::try_from(self.data_offset + info.offset)
            .map_err(|_| GlError::Parse("GGUF: tensor offset overflows usize".into()))?;
        let end = start
            .checked_add(size)
            .filter(|&e| e <= self.mmap.len())
            .ok_or_else(|| {
                GlError::Parse(format!(
                    "GGUF: tensor '{}' data out of bounds ({} bytes at offset {})",
                    info.name, size, start
                ))
            })?;
        Ok(&self.mmap[start..end])
    }

    /// Dequantize a tensor to an `f32` vector.
    ///
    /// Phase 1 supports `F32` (passthrough), `F16`, `BF16`, `Q4_0`, `Q8_0`.
    /// `Q4_K` and unknown types return [`GlError::UnsupportedDtype`].
    pub fn dequantize(&self, info: &GgufTensorInfo) -> Result<Vec<f32>, GlError> {
        let raw = self.tensor_data(info)?;
        let numel = info.numel();
        match info.dtype {
            GgufDType::F32 => Ok(raw
                .chunks_exact(4)
                .map(LittleEndian::read_f32)
                .collect()),
            GgufDType::F16 => Ok(raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(LittleEndian::read_u16(b)))
                .collect()),
            GgufDType::BF16 => Ok(raw
                .chunks_exact(2)
                .map(|b| f32::from_bits((LittleEndian::read_u16(b) as u32) << 16))
                .collect()),
            GgufDType::Q4_0 => Ok(dequant_q4_0(raw, numel)),
            GgufDType::Q8_0 => Ok(dequant_q8_0(raw, numel)),
            GgufDType::Q6_K => Ok(dequant_q6_k(raw, numel)),
            GgufDType::Q4_K => Err(GlError::UnsupportedDtype(format!(
                "Q4_K (tensor {}) — dequant lives in glproc",
                info.name
            ))),
            GgufDType::Q5_0 => Err(GlError::UnsupportedDtype(format!(
                "Q5_0 (tensor {}) — dequant lives in glproc",
                info.name
            ))),
            GgufDType::Unknown(id) => Err(GlError::UnsupportedDtype(format!(
                "ggml type id {id} (tensor {})",
                info.name
            ))),
        }
    }
}

/// Convert an IEEE half-precision bit pattern to `f32` (handles subnormals,
/// infinities and NaN). Written out by hand — no `half` crate dependency.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31 // signed zero
        } else {
            // Subnormal: normalize into f32 range.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (frac << 13) // inf / NaN
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// Dequantize GGML `Q4_0` blocks: 32 elements per block, stored as an f16
/// scale followed by 16 bytes of packed 4-bit values; `x = (q - 8) * scale`.
/// Low nibbles hold elements 0..16, high nibbles hold elements 16..32.
fn dequant_q4_0(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    for (bi, block) in raw.chunks_exact(18).enumerate() {
        let d = f16_to_f32(LittleEndian::read_u16(&block[0..2]));
        let qs = &block[2..18];
        let base = bi * 32;
        for (j, &b) in qs.iter().enumerate() {
            let lo = (b & 0x0f) as i32 - 8;
            let hi = (b >> 4) as i32 - 8;
            if let Some(slot) = out.get_mut(base + j) {
                *slot = lo as f32 * d;
            }
            if let Some(slot) = out.get_mut(base + j + 16) {
                *slot = hi as f32 * d;
            }
        }
    }
    out
}

/// Dequantize GGML `Q8_0` blocks: 32 elements per block, stored as an f16
/// scale followed by 32 signed bytes; `x = q * scale`.
fn dequant_q8_0(raw: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    for (bi, block) in raw.chunks_exact(34).enumerate() {
        let d = f16_to_f32(LittleEndian::read_u16(&block[0..2]));
        let base = bi * 32;
        for (j, &b) in block[2..34].iter().enumerate() {
            if let Some(slot) = out.get_mut(base + j) {
                *slot = (b as i8) as f32 * d;
            }
        }
    }
    out
}

/// Dequantize GGML `Q6_K` super-blocks: 256 elements per block, 16 sub-blocks
/// of 16 elements each. Each element's signed 6-bit value is reconstructed
/// from `ql` (low 4 bits, 2 values/byte) and `qh` (high 2 bits, 4 values/byte);
/// `x = d * scale[sub_block] * (q6_raw - 32)`.
fn dequant_q6_k(raw: &[u8], numel: usize) -> Vec<f32> {
    const SUPERBLOCK_ELEMENTS: usize = 256;
    const SUBBLOCK_ELEMENTS: usize = 16;
    const BLOCK_BYTES: usize = 210;

    let mut out = Vec::with_capacity(numel);
    for block in raw.chunks_exact(BLOCK_BYTES) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(LittleEndian::read_u16(&block[208..210]));

        let remaining = numel - out.len();
        let count = SUPERBLOCK_ELEMENTS.min(remaining);
        for i in 0..count {
            let low4 = (ql[i / 2] >> ((i & 1) * 4)) & 0x0f;
            let high2 = (qh[i / 4] >> ((i & 3) * 2)) & 0x03;
            let q6_raw = low4 | (high2 << 4); // unsigned [0, 63]
            let q = q6_raw as i32 - 32; // signed [-32, 31]
            let scale = scales[i / SUBBLOCK_ELEMENTS] as i8 as f32;
            out.push(d * scale * q as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal v3 GGUF file in memory: 2 metadata keys, 1 F32 tensor.
    fn build_test_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count

        // metadata: general.architecture = "llama"
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // type: string
        write_string(&mut buf, "llama");
        // metadata: llama.block_count = 2 (u32)
        write_string(&mut buf, "llama.block_count");
        buf.extend_from_slice(&4u32.to_le_bytes()); // type: u32
        buf.extend_from_slice(&2u32.to_le_bytes());

        // tensor info: "t" F32 [2, 3], offset 0
        write_string(&mut buf, "t");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // pad to 32-byte alignment
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parse_minimal_gguf() {
        let bytes = build_test_gguf();
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();
        tmp.flush().unwrap();

        let g = GgufFile::open(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(g.header.version, 3);
        assert_eq!(g.header.tensor_count, 1);
        assert_eq!(
            g.get_meta("general.architecture").and_then(|v| v.as_str()),
            Some("llama")
        );
        assert_eq!(
            g.get_meta("llama.block_count").and_then(|v| v.as_u64()),
            Some(2)
        );

        let info = g.find_tensor("t").unwrap().clone();
        assert_eq!(info.dimensions, vec![2, 3]);
        let data = g.dequantize(&info).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"NOPE0000").unwrap();
        tmp.flush().unwrap();
        assert!(GgufFile::open(tmp.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333252).abs() < 1e-5);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        // smallest subnormal: 2^-24
        assert!((f16_to_f32(0x0001) - 5.9604645e-8).abs() < 1e-12);
    }

    #[test]
    fn q8_0_roundtrip() {
        // one block: scale 0.5, quants 0..32
        let mut raw = Vec::new();
        raw.extend_from_slice(&0x3800u16.to_le_bytes()); // f16 0.5
        for i in 0..32u8 {
            raw.push(i);
        }
        let out = dequant_q8_0(&raw, 32);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - i as f32 * 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn q4_0_layout() {
        // one block: scale 1.0, all nibbles = 0x9 -> (9-8)*1.0 = 1.0
        let mut raw = Vec::new();
        raw.extend_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        raw.extend_from_slice(&[0x99u8; 16]);
        let out = dequant_q4_0(&raw, 32);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn q6_k_reference_block() {
        // One superblock: d=1.0, all sub-block scales=2, all q6_raw=40
        // (low4=0x8, high2=0x2 -> 0x8 | (0x2<<4) = 0x28 = 40) -> q = 40-32 = 8
        // expected value per element: d * scale * q = 1.0 * 2 * 8 = 16.0
        let mut raw = vec![0u8; 210];
        for b in raw[0..128].iter_mut() {
            *b = 0x88; // both nibbles = 0x8
        }
        for b in raw[128..192].iter_mut() {
            *b = 0xaa; // 0b10_10_10_10 -> each 2-bit field = 0x2
        }
        for b in raw[192..208].iter_mut() {
            *b = 2; // sub-block scale
        }
        raw[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0

        let out = dequant_q6_k(&raw, 256);
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| (v - 16.0).abs() < 1e-4));
    }

    #[test]
    fn q6_k_dtype_maps_from_ggml_id_14() {
        assert_eq!(GgufDType::from_u32(14), GgufDType::Q6_K);
        assert_eq!(GgufDType::Q6_K.block_numel(), Some(256));
        assert_eq!(GgufDType::Q6_K.block_bytes(), Some(210));
    }
}
