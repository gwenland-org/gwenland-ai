/// Cross-crate entry points for LoRA adapter export and GGUF merge operations.
///
/// @INFO This module is the boundary between gwen-tui (CLI parsing) and the
/// ML-heavy code in lora_bridge / lora_merger. gwen-tui has no candle deps;
/// these functions accept only std types so the TUI crate never needs to know
/// about VarMap, Tensor, or LoraAdapter internals.
use std::collections::HashMap;
use std::path::Path;

use candle_nn::VarMap;

use crate::convert::gguf_parser;
use crate::error::GwenError;
use crate::train::layered_training_loop::shape_to_2d;
use crate::train::lora_bridge::{LoraAdapter, LoraConfig as BridgeLoraConfig, LoraExporter};
use crate::train::lora_merger::{parse_gguf_key, LoraMerger};

// ── export-adapter ────────────────────────────────────────────────────────────

/// Export LoRA adapter weights from a SafeTensors checkpoint to an adapter file.
///
/// Loads `checkpoint_path` into a `VarMap`, extracts all lora_a/lora_b pairs,
/// and writes them to `output_path` in the GwenLand SafeTensors adapter format.
///
/// When `dry_run` is true the checkpoint is loaded and adapters are extracted
/// but nothing is written to disk; the adapter count is returned as `Ok(n)`.
///
/// @INFO Uses `VarMap::load()` which reads the SafeTensors file created by the
/// native training loop. The VarMap must contain keys matching the pattern
/// `lora_{a|b}_layer_{N}_{proj}_proj` or extraction returns an error.
/// @EDITABLE rank and alpha are derived from the checkpoint's actual tensor
/// shapes; no separate config file is needed.
/// @TODO Wave 6: accept an explicit LoraConfig override for rank/alpha so
/// adapters trained with non-default hyperparams can be re-exported cleanly.
///
/// @INFO GWEN-222: when `base_gguf_path` is `Some`, every extracted adapter's
/// `(d_in, d_out)` dimensions are checked against the matching projection tensor
/// in the base GGUF *before* any output is written. A mismatch returns
/// `GwenError::ShapeMismatch` and no file is created. When `None`, validation is
/// skipped (backward-compatible) and the caller is responsible for warning.
pub fn export_adapter(
    checkpoint_path: &Path,
    output_path: &Path,
    dry_run: bool,
    base_gguf_path: Option<&Path>,
) -> std::result::Result<usize, GwenError> {
    if !checkpoint_path.exists() {
        return Err(GwenError::CandleError(format!(
            "checkpoint path does not exist: {}",
            checkpoint_path.display()
        )));
    }

    // Read every tensor from the checkpoint into a VarMap.
    //
    // @DANGER We do NOT use `VarMap::load()` here: candle's `VarMap::load` only
    // refreshes Vars that are ALREADY present in the map (see candle-nn
    // var_map.rs — "values for variables that are currently not in the map are
    // not kept"). Loading into a fresh, empty VarMap is therefore a no-op and
    // `extract_adapters` would return zero adapters. Instead we read all tensors
    // explicitly and insert each as a Var so the adapter pairs are visible.
    let tensors = candle_core::safetensors::load(checkpoint_path, &candle_core::Device::Cpu)
        .map_err(|e| {
            GwenError::CandleError(format!(
                "failed to load checkpoint '{}': {e}",
                checkpoint_path.display()
            ))
        })?;
    let varmap = VarMap::new();
    {
        let mut data = varmap.data().lock().unwrap();
        for (name, tensor) in tensors {
            let var = candle_core::Var::from_tensor(&tensor).map_err(|e| {
                GwenError::CandleError(format!("failed to wrap tensor '{name}' as Var: {e}"))
            })?;
            data.insert(name, var);
        }
    }

    // Build a default LoraConfig — rank/alpha are inferred from tensor shapes
    // inside extract_adapters(), so the values here only govern target_modules.
    let lora_config = BridgeLoraConfig::default();
    let exporter = LoraExporter::new(lora_config);

    // Extract adapters to validate the checkpoint structure.
    let adapters = exporter
        .extract_adapters(&varmap)
        .map_err(|e| GwenError::CandleError(e.to_string()))?;

    let count = adapters.len();

    // GWEN-222: optional pre-write shape validation against a base GGUF. Runs
    // before BOTH the dry-run return and any file write, so a mismatch never
    // leaves a partial output file on disk.
    if let Some(base_path) = base_gguf_path {
        AdapterShapeValidator::validate(&adapters, base_path)?;
    }

    if dry_run {
        // Validation only — do not write output.
        return Ok(count);
    }

    // Validate output parent exists before the (potentially slow) export.
    let output_parent = output_path.parent().ok_or_else(|| {
        GwenError::CandleError(format!(
            "output path has no parent directory: {}",
            output_path.display()
        ))
    })?;
    if !output_parent.exists() {
        return Err(GwenError::CandleError(format!(
            "output parent directory does not exist: {}",
            output_parent.display()
        )));
    }

    // Write the adapter SafeTensors file.
    exporter
        .export_safetensors(&varmap, output_path)
        .map_err(|e| GwenError::CandleError(e.to_string()))?;

    Ok(count)
}

// ── adapter shape validation (GWEN-222) ───────────────────────────────────────

/// Validates that exported LoRA adapters match the dimensions of the base GGUF
/// model's projection tensors, catching shape mismatches *before* an adapter is
/// written or merged.
struct AdapterShapeValidator;

impl AdapterShapeValidator {
    /// Check every adapter's `(d_in, d_out)` against the base GGUF.
    ///
    /// Reads only the GGUF tensor descriptors (`parse_header`, no payloads) and
    /// builds a map `candle_layer_name → (d_out, d_in)` from each projection
    /// tensor. Adapters whose layer name has no matching base tensor are skipped
    /// (the merge step skips them too). A dimension mismatch on a matched adapter
    /// returns `GwenError::ShapeMismatch`.
    fn validate(
        adapters: &[LoraAdapter],
        base_gguf_path: &Path,
    ) -> std::result::Result<(), GwenError> {
        let header = gguf_parser::parse_header(base_gguf_path).map_err(|e| {
            GwenError::ModelLoad(format!(
                "failed to read base GGUF '{}': {e}",
                base_gguf_path.display()
            ))
        })?;

        // candle layer name → (d_out, d_in) for each base projection tensor.
        let mut base_dims: HashMap<String, (usize, usize)> = HashMap::new();
        for tensor in &header.tensors {
            if let Some((layer_idx, proj)) = parse_gguf_key(&tensor.name) {
                if let Ok((d_out, d_in)) = shape_to_2d(&tensor.shape) {
                    base_dims
                        .insert(format!("lora_layer_{layer_idx}_{proj}_proj"), (d_out, d_in));
                }
            }
        }

        for adapter in adapters {
            let Some(&(base_d_out, base_d_in)) = base_dims.get(&adapter.layer_name) else {
                // No matching base tensor; the merger would skip this adapter too.
                continue;
            };

            // lora_a is [rank, d_in]; lora_b is [d_out, rank].
            let adapter_d_in = adapter.lora_a.dims().get(1).copied().unwrap_or(0);
            let adapter_d_out = adapter.lora_b.dims().first().copied().unwrap_or(0);

            if adapter_d_in != base_d_in || adapter_d_out != base_d_out {
                return Err(GwenError::ShapeMismatch {
                    adapter_key: adapter.layer_name.clone(),
                    adapter: vec![adapter_d_out, adapter_d_in],
                    base: vec![base_d_out, base_d_in],
                });
            }
        }

        Ok(())
    }
}

// ── merge-adapter ─────────────────────────────────────────────────────────────

/// Merge a LoRA adapter SafeTensors file into a GGUF base model.
///
/// Delegates to `LoraMerger::merge_into_gguf()` after building the merger with
/// the appropriate memory budget. When `dry_run` is true only path validation
/// is performed (the merge loop never runs).
///
/// @INFO memory_budget controls peak RAM during the merge loop. The default
/// (2 GB) is sized for 8 GB machines. Pass `Some(budget)` to override.
/// @DANGER output_path is created or overwritten without prompting. The caller
/// (gwen-tui) is responsible for showing a warning if the file already exists.
/// @EDITABLE Set memory_budget to a small value (e.g. 64 MB) in tests to
/// verify budget-exceeded behaviour without real model files.
pub fn merge_adapter(
    base_path: &Path,
    adapter_path: &Path,
    output_path: &Path,
    memory_budget: Option<usize>,
    dry_run: bool,
) -> std::result::Result<(), GwenError> {
    let merger = match memory_budget {
        Some(budget) => LoraMerger::with_memory_budget(budget),
        None => LoraMerger::new(),
    };

    if dry_run {
        // Validate paths only — mirror the first three checks in merge_into_gguf().
        if !base_path.exists() {
            return Err(GwenError::CandleError(format!(
                "base model path does not exist: {}",
                base_path.display()
            )));
        }
        if !adapter_path.exists() {
            return Err(GwenError::CandleError(format!(
                "adapter path does not exist: {}",
                adapter_path.display()
            )));
        }
        let output_parent = output_path.parent().ok_or_else(|| {
            GwenError::CandleError(format!(
                "output path has no parent directory: {}",
                output_path.display()
            ))
        })?;
        if !output_parent.exists() {
            return Err(GwenError::CandleError(format!(
                "output parent directory does not exist: {}",
                output_parent.display()
            )));
        }
        return Ok(());
    }

    merger.merge_into_gguf(base_path, adapter_path, output_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    use crate::train::layer_loader::write_transformer_gguf_pub;

    /// Build a LoRA adapter with `lora_a = [rank, d_in]`, `lora_b = [d_out, rank]`.
    fn adapter(layer_name: &str, rank: usize, d_in: usize, d_out: usize) -> LoraAdapter {
        let dev = &Device::Cpu;
        LoraAdapter {
            layer_name: layer_name.to_string(),
            lora_a: Tensor::zeros((rank, d_in), DType::F32, dev).unwrap(),
            lora_b: Tensor::zeros((d_out, rank), DType::F32, dev).unwrap(),
            rank,
            alpha: rank as f32,
        }
    }

    /// Property 9 (pass): adapters whose dims match the base GGUF validate OK.
    /// `write_transformer_gguf_pub` gives `blk.0.attn_q.weight` of shape [4,4],
    /// i.e. q_proj d_in=4, d_out=4.
    #[test]
    fn test_shape_validation_pass() {
        let gguf = write_transformer_gguf_pub(1);
        let adapters = vec![adapter("lora_layer_0_q_proj", 2, 4, 4)];
        AdapterShapeValidator::validate(&adapters, gguf.path())
            .expect("matching adapter must validate");
    }

    /// Property 9 (fail): wrong d_in (lora_a second dim) → ShapeMismatch.
    #[test]
    fn test_shape_validation_mismatch_d_in() {
        let gguf = write_transformer_gguf_pub(1);
        let adapters = vec![adapter("lora_layer_0_q_proj", 2, 8, 4)]; // d_in 8 != 4
        let err = AdapterShapeValidator::validate(&adapters, gguf.path()).unwrap_err();
        assert!(matches!(err, GwenError::ShapeMismatch { .. }), "got {err:?}");
    }

    /// Property 9 (fail): wrong d_out (lora_b first dim) → ShapeMismatch.
    #[test]
    fn test_shape_validation_mismatch_d_out() {
        let gguf = write_transformer_gguf_pub(1);
        let adapters = vec![adapter("lora_layer_0_q_proj", 2, 4, 8)]; // d_out 8 != 4
        let err = AdapterShapeValidator::validate(&adapters, gguf.path()).unwrap_err();
        assert!(matches!(err, GwenError::ShapeMismatch { .. }), "got {err:?}");
    }

    /// An adapter for a layer absent from the base GGUF is skipped, not an error
    /// (mirrors merge behaviour, which only merges matched projections).
    #[test]
    fn test_unmatched_adapter_is_skipped() {
        let gguf = write_transformer_gguf_pub(1);
        let adapters = vec![adapter("lora_layer_9_q_proj", 2, 999, 999)];
        AdapterShapeValidator::validate(&adapters, gguf.path())
            .expect("unmatched adapter must be skipped, not rejected");
    }

    /// Property 11: ShapeMismatch Display contains the adapter key + both shapes.
    #[test]
    fn test_shape_mismatch_error_format() {
        let e = GwenError::ShapeMismatch {
            adapter_key: "lora_layer_0_q_proj".into(),
            adapter: vec![4096, 8],
            base: vec![4096, 4096],
        };
        let s = format!("{e}");
        assert!(s.contains("lora_layer_0_q_proj"), "missing key: {s}");
        assert!(s.contains("[4096, 8]"), "missing adapter dims: {s}");
        assert!(s.contains("[4096, 4096]"), "missing base dims: {s}");
    }
}
