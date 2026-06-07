/// Convert module — GGUF ↔ SafeTensors format conversion.
///
/// Exists because GGUF is the dominant on-disk format for quantised LLMs
/// (llama.cpp, Ollama) while SafeTensors is the canonical format for
/// HuggingFace tooling and GwenTensor inference. This module bridges the gap
/// so users never have to leave the CLI to convert files.
///
/// Architecture: all conversion logic lives here in gwen-core so gwen-tui
/// can call a single typed function and own only the progress/UI layer.
pub mod gguf_parser;
pub mod dequant;
pub mod writer;

use std::path::Path;
use std::time::Instant;

pub use gguf_parser::{GgufFile, TensorInfo, GgufDtype};
pub use dequant::DequantMode;
pub use writer::write_safetensors;

/// Outcome of a successful GGUF → SafeTensors conversion.
///
/// Returned to the TUI layer so it can render the final summary without
/// needing to know anything about the file format internals.
pub struct ConvertResult {
    /// Number of tensors written to the output file.
    pub tensors_converted: usize,
    /// Dequantization mode used (Standard or Euler).
    pub mode: DequantMode,
    /// Absolute path of the output file.
    pub output_path: std::path::PathBuf,
    /// Wall-clock time for the full conversion.
    pub elapsed_secs: f64,
    /// Optional Euler sweet-spot warning (Some(pct) when pct > 20%).
    pub euler_warning: Option<f64>,
}

/// Progress event emitted once per tensor so the TUI layer can print a line.
///
/// Keeping this in core rather than tui means the same callback type can be
/// used by a hypothetical batch-convert API in the future.
pub struct TensorProgress {
    /// 1-based index of the tensor currently being converted.
    pub index: usize,
    /// Total number of tensors in this file.
    pub total: usize,
    /// Tensor name as stored in the GGUF metadata.
    pub name: String,
    /// Quantization type of the tensor (e.g. "Q4_0", "F32").
    pub dtype: String,
    /// Shape dimensions, e.g. [4096, 32000].
    pub shape: Vec<u64>,
    /// Wall-clock milliseconds spent dequantising this tensor.
    pub elapsed_ms: u128,
}

/// Run a GGUF → SafeTensors conversion.
///
/// `path`     — path to the `.gguf` source file.
/// `mode`     — Standard (linear) or Euler (cosine projection) dequant.
/// `progress` — callback invoked after each tensor is converted; the TUI
///              layer uses this to print one progress line per tensor.
///
/// Returns `Err(String)` on any I/O or parse failure so the caller can print
/// a clean error message without a backtrace.
pub fn convert_gguf(
    path: &Path,
    mode: DequantMode,
    mut progress: impl FnMut(TensorProgress),
) -> Result<ConvertResult, String> {
    let wall_start = Instant::now();

    // Parse the GGUF header and tensor metadata. Data is memory-mapped so we
    // avoid loading the full multi-GB file into RAM before we need it.
    let gguf = gguf_parser::parse(path)?;
    let total = gguf.tensors.len();

    // Output path: same directory, same stem, .safetensors extension.
    // Changing just the extension preserves any version suffix in the filename
    // (e.g. qwen3-8b-Q4_K_M.gguf → qwen3-8b-Q4_K_M.safetensors).
    let output_path = path.with_extension("safetensors");

    let mut all_tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::with_capacity(total);
    let mut euler_outside_sweet = 0usize;
    let mut euler_total_weights = 0usize;

    for (idx, tensor_info) in gguf.tensors.iter().enumerate() {
        let t0 = Instant::now();

        // Dequantise each tensor to f32 using the selected mode. Euler mode
        // also accumulates out-of-sweet-spot counts for the warning check.
        let weights = dequant::dequantize(tensor_info, mode)?;

        if matches!(mode, DequantMode::Euler) {
            // Euler sweet spot: [-0.309, 0.309]. Count weights outside it so we
            // can warn the user if more than 20% fall outside (indicating the
            // model may lose precision under GwenTensor inference).
            for &w in &weights {
                euler_total_weights += 1;
                if w < -0.309 || w > 0.309 {
                    euler_outside_sweet += 1;
                }
            }
        }

        let elapsed_ms = t0.elapsed().as_millis();

        progress(TensorProgress {
            index: idx + 1,
            total,
            name: tensor_info.name.clone(),
            dtype: format!("{:?}", tensor_info.dtype),
            shape: tensor_info.shape.clone(),
            elapsed_ms,
        });

        all_tensors.push((tensor_info.name.clone(), tensor_info.shape.clone(), weights));
    }

    // Write all dequantised tensors to a single SafeTensors file. We collect
    // all tensors first because the SafeTensors header must precede the data
    // and the header size depends on knowing all tensor names and shapes.
    writer::write_safetensors(&output_path, &all_tensors)?;

    // Build the Euler warning only after all tensors are processed so the
    // percentage reflects the whole file, not just the last tensor.
    let euler_warning = if matches!(mode, DequantMode::Euler) && euler_total_weights > 0 {
        let pct = (euler_outside_sweet as f64 / euler_total_weights as f64) * 100.0;
        if pct > 20.0 {
            Some(pct)
        } else {
            None
        }
    } else {
        None
    };

    Ok(ConvertResult {
        tensors_converted: total,
        mode,
        output_path,
        elapsed_secs: wall_start.elapsed().as_secs_f64(),
        euler_warning,
    })
}
