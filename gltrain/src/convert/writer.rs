/// SafeTensors output writer — manual implementation.
///
/// We write SafeTensors manually rather than using the `safetensors` crate
/// (which is an optional feature in gwen-core's Cargo.toml, gated behind the
/// `candle` feature). The convert command must work without enabling that
/// feature flag so users don't pull in candle-core as a transitive dependency
/// just to run a file conversion.
///
/// SafeTensors format specification:
/// https://huggingface.co/docs/safetensors/index
///
/// Binary layout:
///   [header_size: u64 LE]  — byte length of the JSON header string
///   [header: UTF-8 JSON]   — tensor metadata map
///   [data: raw f32 bytes]  — concatenated tensor data, no padding between tensors
///
/// Header JSON schema (minimal for F32 output):
///   {
///     "__metadata__": {},
///     "tensor_name": {
///       "dtype": "F32",
///       "shape": [d0, d1, ...],
///       "data_offsets": [start, end]   // byte offsets into the data block
///     },
///     ...
///   }
///
/// We write only F32 dtype regardless of input dtype because we've already
/// dequantised everything to Vec<f32> by the time the writer is called.
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write dequantised tensors to a SafeTensors file at `output_path`.
///
/// `tensors` is a slice of (name, shape, f32_weights) tuples in the order they
/// should appear in the output. Order is preserved so tools that rely on
/// tensor order (e.g. HuggingFace `from_pretrained`) behave as expected.
///
/// Returns `Err(String)` on any I/O failure.
pub fn write_safetensors(
    output_path: &Path,
    tensors: &[(String, Vec<u64>, Vec<f32>)],
) -> Result<(), String> {
    // ── Build the JSON header ─────────────────────────────────────────────────

    // Compute the data offset for each tensor in the flat data block.
    // data_offsets[i] = (start_byte, end_byte) where end_byte is exclusive.
    // All tensors are F32 so each element occupies exactly 4 bytes.
    let mut offset: u64 = 0;
    let mut tensor_offsets: Vec<(u64, u64)> = Vec::with_capacity(tensors.len());
    for (_, _, weights) in tensors {
        let n_bytes = weights.len() as u64 * 4;
        tensor_offsets.push((offset, offset + n_bytes));
        offset += n_bytes;
    }

    let header_json = build_header_json(tensors, &tensor_offsets)?;
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    // ── Open output file and write ────────────────────────────────────────────

    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("cannot create '{}': {}", output_path.display(), e))?;
    let mut w = BufWriter::new(file);

    // 8-byte little-endian header size prefix.
    w.write_all(&header_size.to_le_bytes())
        .map_err(|e| format!("write error (header size): {}", e))?;

    // JSON header bytes.
    w.write_all(header_bytes)
        .map_err(|e| format!("write error (header): {}", e))?;

    // Raw f32 data for each tensor — little-endian byte order per the spec.
    // We write tensor-by-tensor in order, matching the data_offsets in the header.
    for (_, _, weights) in tensors {
        for &w_val in weights {
            w.write_all(&w_val.to_le_bytes())
                .map_err(|e| format!("write error (tensor data): {}", e))?;
        }
    }

    w.flush()
        .map_err(|e| format!("flush error writing SafeTensors file: {}", e))?;

    Ok(())
}

/// Build the SafeTensors JSON header string.
///
/// We construct JSON manually rather than via serde to avoid adding a serde
/// dependency path through the convert module. The header is small
/// (proportional to tensor count, not data size) so the overhead is negligible.
///
/// The `__metadata__` key is required by the SafeTensors spec even if empty.
/// Without it, some HuggingFace tools emit a warning; others refuse to load.
fn build_header_json(
    tensors: &[(String, Vec<u64>, Vec<f32>)],
    offsets: &[(u64, u64)],
) -> Result<String, String> {
    let mut json = String::from("{\"__metadata__\":{}");

    for (i, (name, shape, _)) in tensors.iter().enumerate() {
        let (start, end) = offsets[i];

        // Validate tensor name doesn't contain characters that would break JSON.
        // GGUF names are ASCII identifiers so this is a sanity check, not a
        // common code path.
        if name.contains('"') || name.contains('\\') {
            return Err(format!(
                "tensor name '{}' contains characters unsafe for JSON output",
                name
            ));
        }

        // Shape serialised as a JSON array of u64 integers.
        let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let shape_json = format!("[{}]", shape_str.join(","));

        json.push_str(&format!(
            ",\"{}\":{{\"dtype\":\"F32\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            name, shape_json, start, end
        ));
    }

    json.push('}');
    Ok(json)
}
