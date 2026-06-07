/// GGUF binary format parser — minimal hand-written implementation.
///
/// We implement the parser manually rather than pulling in a `gguf` crate
/// because: (1) the crate ecosystem for GGUF is still unstable and we need
/// only a subset of the spec; (2) keeping the dependency surface small is a
/// hard constraint for gwen-core's sub-50 MB target binary size.
///
/// Spec reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
/// (GGUF v3, which is backward-compatible with v1 and v2 for the subset we use)
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// ── Constants ─────────────────────────────────────────────────────────────────

/// GGUF magic number: the four literal bytes 'G', 'G', 'U', 'F'.
/// Present at byte offset 0 in every valid GGUF file.
const GGUF_MAGIC: u32 = 0x4647_4755; // little-endian "GGUF"

/// GGUF supports versions 1, 2, and 3. All three share the same header layout
/// for the fields we read; version affects only KV value alignment in v3.
const GGUF_VERSION_MIN: u32 = 1;
const GGUF_VERSION_MAX: u32 = 3;

/// Block size for Q8_0 and Q4_0 quantization schemes.
/// Each block covers exactly 32 elements and carries one f16 scale value.
/// This is a GGML architectural constant — not configurable per-file.
const GGML_QK: usize = 32;

// ── Public types ──────────────────────────────────────────────────────────────

/// Supported GGUF element types relevant to dequantisation.
///
/// Only the dtypes we know how to dequantise are listed. Any unknown dtype
/// encountered in a real file causes a descriptive error so the user can
/// open a bug report rather than silently producing corrupt output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    /// 32-bit float — no dequantisation needed, direct copy.
    F32,
    /// 16-bit float — upcast to f32 during output.
    F16,
    /// 8-bit quantisation: blocks of 32 elements, each with 1× f16 scale.
    /// Layout per block: [scale: f16][values: i8 × 32]
    Q8_0,
    /// 4-bit quantisation: blocks of 32 elements, each with 1× f16 scale.
    /// Layout per block: [scale: f16][nibbles: u8 × 16]
    /// Values are signed 4-bit integers packed as two per byte (low nibble first).
    Q4_0,
    /// 4-bit K-quant: superblocks of 256 elements, 8 sub-blocks of 32.
    /// Layout: [d: f16][dmin: f16][scales: u8×12][nibbles: u8×128] = 144 bytes.
    /// Formula: W = d * scale[j] * q - dmin * min[j], q ∈ [0, 15].
    Q4_K,
    /// 5-bit K-quant: superblocks of 256 elements, 8 sub-blocks of 32.
    /// Layout: [d: f16][dmin: f16][scales: u8×12][qh: u8×32][ql: u8×128] = 176 bytes.
    /// Formula: W = d * scale[j] * q - dmin * min[j], q ∈ [0, 31].
    Q5_K,
    /// 6-bit K-quant: superblocks of 256 elements, 16 sub-blocks of 16.
    /// Layout: [ql: u8×128][qh: u8×64][scales: i8×16][d: f16] = 210 bytes.
    /// Formula: W = d * scales[j] * q, q ∈ [-32, 31].
    Q6_K,
}

/// Metadata for a single tensor extracted from the GGUF header.
///
/// The actual weight bytes are not loaded here — we store only the byte offset
/// so `dequant.rs` can seek to the right position on demand, avoiding the need
/// to hold the entire multi-GB data block in RAM at once.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name as stored in the GGUF file (e.g. "token_embd.weight").
    pub name: String,
    /// Shape in row-major order (e.g. [4096, 32000] for an embedding matrix).
    pub shape: Vec<u64>,
    /// Quantisation dtype of the stored values.
    pub dtype: GgufDtype,
    /// Byte offset of this tensor's data within the GGUF file's data segment.
    /// Add `GgufFile::data_offset` to get the absolute file offset.
    pub data_offset: u64,
    /// Total number of quantised bytes for this tensor (including block headers).
    pub data_size: usize,
    /// The raw bytes of this tensor's quantised data, loaded eagerly during parse.
    ///
    /// We load eagerly because GGUF files are often gigabytes; loading once in
    /// parse order (sequential I/O) is faster than random-seeking later.
    pub raw_data: Vec<u8>,
}

/// Parsed GGUF file: header metadata plus per-tensor descriptors with data.
pub struct GgufFile {
    /// GGUF format version (1, 2, or 3).
    pub version: u32,
    /// All tensor metadata entries, in the order they appear in the file.
    pub tensors: Vec<TensorInfo>,
}

// ── Parser entry point ────────────────────────────────────────────────────────

/// Parse a GGUF file from `path` and return a `GgufFile` ready for dequant.
///
/// Errors are returned as human-readable strings so the TUI layer can print
/// them directly without unwinding through a custom error type hierarchy.
pub fn parse(path: &Path) -> Result<GgufFile, String> {
    let file = File::open(path)
        .map_err(|e| format!("cannot open '{}': {}", path.display(), e))?;
    let mut reader = BufReader::new(file);
    parse_inner(&mut reader, path)
}

// ── Internal parser ───────────────────────────────────────────────────────────

fn parse_inner<R: Read + Seek>(r: &mut R, path: &Path) -> Result<GgufFile, String> {
    // ── Magic + version ───────────────────────────────────────────────────────
    let magic = read_u32_le(r)?;
    if magic != GGUF_MAGIC {
        return Err(format!(
            "'{}' is not a GGUF file (got magic 0x{:08X}, expected 0x{:08X})",
            path.display(), magic, GGUF_MAGIC
        ));
    }
    let version = read_u32_le(r)?;
    if version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX {
        return Err(format!(
            "unsupported GGUF version {} in '{}' (supported: {}-{})",
            version, path.display(), GGUF_VERSION_MIN, GGUF_VERSION_MAX
        ));
    }

    // ── Header counts ─────────────────────────────────────────────────────────
    let tensor_count = read_u64_le(r)? as usize;
    let kv_count = read_u64_le(r)? as usize;

    // ── Skip KV metadata ──────────────────────────────────────────────────────
    // We don't use any KV metadata for dequantisation. We still must skip over
    // it correctly to reach the tensor info section.
    for _ in 0..kv_count {
        skip_kv_entry(r)?;
    }

    // ── Tensor info section ───────────────────────────────────────────────────
    let mut tensor_infos: Vec<TensorInfo> = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let info = read_tensor_info(r)?;
        tensor_infos.push(info);
    }

    // ── Alignment padding before data block ───────────────────────────────────
    // GGUF v3 aligns the data block to 32 bytes; v1/v2 use no explicit padding.
    // We seek to the next 32-byte boundary unconditionally — for v1/v2 files
    // this is a no-op if the position is already aligned.
    let pos = r.stream_position()
        .map_err(|e| format!("seek error: {}", e))?;
    let alignment: u64 = 32;
    let remainder = pos % alignment;
    if remainder != 0 {
        let padding = alignment - remainder;
        r.seek(SeekFrom::Current(padding as i64))
            .map_err(|e| format!("seek error after alignment padding: {}", e))?;
    }
    let data_base = r.stream_position()
        .map_err(|e| format!("seek error reading data base offset: {}", e))?;

    // ── Load raw tensor data ───────────────────────────────────────────────────
    // Seek to each tensor in order and read its raw bytes. Sequential reads
    // are much faster than random seeks on spinning disks and NVMe alike.
    let mut tensors_out: Vec<TensorInfo> = Vec::with_capacity(tensor_infos.len());
    for mut info in tensor_infos {
        let abs_offset = data_base + info.data_offset;
        r.seek(SeekFrom::Start(abs_offset))
            .map_err(|e| format!("seek error for tensor '{}': {}", info.name, e))?;

        let mut raw_data = vec![0u8; info.data_size];
        r.read_exact(&mut raw_data)
            .map_err(|e| format!("read error for tensor '{}': {}", info.name, e))?;

        info.raw_data = raw_data;
        tensors_out.push(info);
    }

    Ok(GgufFile { version, tensors: tensors_out })
}

// ── Tensor info reader ────────────────────────────────────────────────────────

fn read_tensor_info<R: Read>(r: &mut R) -> Result<TensorInfo, String> {
    let name = read_gguf_string(r)?;
    let n_dims = read_u32_le(r)? as usize;

    // Shape stored as u64 per dimension, n_dims values.
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(read_u64_le(r)?);
    }

    // ggml_type is a u32 enum identifying the element type.
    let dtype_raw = read_u32_le(r)?;
    let dtype = ggml_type_to_dtype(dtype_raw, &name)?;

    // data_offset is relative to the start of the GGUF data segment (after alignment).
    let data_offset = read_u64_le(r)?;

    // Compute raw byte count from shape and dtype block layout.
    let n_elements: u64 = shape.iter().product();
    let data_size = raw_size_bytes(dtype, n_elements as usize);

    Ok(TensorInfo {
        name,
        shape,
        dtype,
        data_offset,
        data_size,
        raw_data: Vec::new(), // filled in after seeking to data segment
    })
}

// ── KV entry skipping ─────────────────────────────────────────────────────────

/// Skip one KV metadata entry without interpreting its value.
///
/// KV entries have a variable-length value depending on their type tag.
/// We must parse enough to know how many bytes to skip, but we discard the
/// content entirely — only tensor data matters for dequantisation.
fn skip_kv_entry<R: Read>(r: &mut R) -> Result<(), String> {
    // Key is a GGUF string (u64 length prefix + bytes, no null terminator).
    let _key = read_gguf_string(r)?;
    let value_type = read_u32_le(r)?;
    skip_kv_value(r, value_type)?;
    Ok(())
}

fn skip_kv_value<R: Read>(r: &mut R, vtype: u32) -> Result<(), String> {
    match vtype {
        0 => { read_u8(r)?; }          // UINT8
        1 => { read_i8(r)?; }          // INT8
        2 => { read_u16_le(r)?; }      // UINT16
        3 => { read_i16_le(r)?; }      // INT16
        4 => { read_u32_le(r)?; }      // UINT32
        5 => { read_i32_le(r)?; }      // INT32
        6 => { read_f32_le(r)?; }      // FLOAT32
        7 => { read_u8(r)?; }          // BOOL (1 byte)
        8 => { read_gguf_string(r)?; } // STRING
        9 => {                          // ARRAY
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)? as usize;
            for _ in 0..count {
                skip_kv_value(r, elem_type)?;
            }
        }
        10 => { read_u64_le(r)?; }     // UINT64
        11 => { read_i64_le(r)?; }     // INT64
        12 => { read_f64_le(r)?; }     // FLOAT64
        other => {
            return Err(format!("unknown KV value type {} in GGUF metadata", other));
        }
    }
    Ok(())
}

// ── dtype helpers ─────────────────────────────────────────────────────────────

fn ggml_type_to_dtype(raw: u32, tensor_name: &str) -> Result<GgufDtype, String> {
    match raw {
        0  => Ok(GgufDtype::F32),
        1  => Ok(GgufDtype::F16),
        6  => Ok(GgufDtype::Q4_0),
        8  => Ok(GgufDtype::Q8_0),
        12 => Ok(GgufDtype::Q4_K),
        13 => Ok(GgufDtype::Q5_K),
        14 => Ok(GgufDtype::Q6_K),
        other => Err(format!(
            "tensor '{}' has unsupported ggml_type {} \
             (supported: F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K)",
            tensor_name, other
        )),
    }
}

/// Compute the number of raw bytes a tensor occupies on disk given its dtype
/// and total element count.
///
/// Block header bytes are included so the caller can allocate the exact read
/// buffer without guessing.
fn raw_size_bytes(dtype: GgufDtype, n_elements: usize) -> usize {
    match dtype {
        GgufDtype::F32 => n_elements * 4,
        GgufDtype::F16 => n_elements * 2,
        GgufDtype::Q8_0 => {
            // Each block: 2 bytes (f16 scale) + 32 bytes (i8 values) = 34 bytes per 32 elements.
            let n_blocks = (n_elements + GGML_QK - 1) / GGML_QK;
            n_blocks * (2 + GGML_QK)
        }
        GgufDtype::Q4_0 => {
            // Each block: 2 bytes (f16 scale) + 16 bytes (nibble-packed i4 values) = 18 bytes per 32 elements.
            let n_blocks = (n_elements + GGML_QK - 1) / GGML_QK;
            n_blocks * (2 + GGML_QK / 2)
        }
        GgufDtype::Q4_K => {
            // Superblock of 256 elements: 2 (d) + 2 (dmin) + 12 (scales) + 128 (nibbles) = 144 bytes.
            const SB: usize = 256;
            let n_blocks = (n_elements + SB - 1) / SB;
            n_blocks * 144
        }
        GgufDtype::Q5_K => {
            // Superblock of 256 elements: 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (ql) = 176 bytes.
            const SB: usize = 256;
            let n_blocks = (n_elements + SB - 1) / SB;
            n_blocks * 176
        }
        GgufDtype::Q6_K => {
            // Superblock of 256 elements: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) = 210 bytes.
            const SB: usize = 256;
            let n_blocks = (n_elements + SB - 1) / SB;
            n_blocks * 210
        }
    }
}

// ── Primitive readers ─────────────────────────────────────────────────────────

/// Read a GGUF-format string: u64 byte length (little-endian) followed by
/// that many UTF-8 bytes with no null terminator.
pub fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read error reading string bytes: {}", e))?;
    String::from_utf8(buf)
        .map_err(|e| format!("GGUF string is not valid UTF-8: {}", e))
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, String> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| io_err("u8", e))?;
    Ok(b[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8, String> {
    read_u8(r).map(|v| v as i8)
}

fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| io_err("u16", e))?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16_le<R: Read>(r: &mut R) -> Result<i16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| io_err("i16", e))?;
    Ok(i16::from_le_bytes(b))
}

pub fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| io_err("u32", e))?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32_le<R: Read>(r: &mut R) -> Result<i32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| io_err("i32", e))?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| io_err("f32", e))?;
    Ok(f32::from_le_bytes(b))
}

pub fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| io_err("u64", e))?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64_le<R: Read>(r: &mut R) -> Result<i64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| io_err("i64", e))?;
    Ok(i64::from_le_bytes(b))
}

fn read_f64_le<R: Read>(r: &mut R) -> Result<f64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| io_err("f64", e))?;
    Ok(f64::from_le_bytes(b))
}

/// Read a 16-bit IEEE 754 half-precision float from a 2-byte little-endian buffer.
///
/// Rust's standard library does not expose f16 natively (stabilised in 1.87 but
/// not yet usable as a primitive in edition 2024 without nightly). We implement
/// the conversion manually using the standard bit-manipulation recipe:
///   - exponent bias: f16 uses 15, f32 uses 127, delta = 112
///   - mantissa shift: f16 has 10 mantissa bits, f32 has 23, shift left by 13
///   - sign bit: same position shift 15→31
///
/// Denormals and infinities are handled correctly by the bit pattern arithmetic.
pub fn read_f16_as_f32(bytes: &[u8]) -> f32 {
    debug_assert!(bytes.len() >= 2, "f16 needs 2 bytes");
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;

    let sign     = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa =  bits        & 0x3FF;

    let f32_bits: u32 = if exponent == 0 {
        // Denormal f16 → normalise into f32 denormal range.
        // The value is (-1)^sign × 2^-14 × (0.mantissa).
        // Re-express as f32 by finding the leading 1 bit of the mantissa.
        if mantissa == 0 {
            // Positive or negative zero.
            sign << 31
        } else {
            // Shift mantissa until the implicit leading 1 is found,
            // adjusting the exponent accordingly.
            let mut m = mantissa;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= !0x400; // strip the implicit leading 1
            // f32 exponent for denormal f16: 127 - 15 - e + 1 = 113 - e
            let exp32 = 127u32.wrapping_sub(14).wrapping_sub(e);
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exponent == 31 {
        // Infinity or NaN: preserve as f32 infinity/NaN.
        (sign << 31) | (0xFF << 23) | (mantissa << 13)
    } else {
        // Normal number: rebias exponent (f16 bias=15, f32 bias=127, delta=112).
        (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

fn io_err(ty: &str, e: io::Error) -> String {
    format!("I/O error reading {}: {}", ty, e)
}
