/// Cross-crate entry points for LoRA adapter export and GGUF merge operations.
///
/// @INFO This module is the boundary between gwen-tui (CLI parsing) and the
/// ML-heavy code in lora_bridge / lora_merger. gwen-tui has no candle deps;
/// these functions accept only std types so the TUI crate never needs to know
/// about VarMap, Tensor, or LoraAdapter internals.
use std::path::Path;

use candle_nn::VarMap;

use crate::error::GwenError;
use crate::train::lora_bridge::{LoraConfig as BridgeLoraConfig, LoraExporter};
use crate::train::lora_merger::LoraMerger;

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
pub fn export_adapter(
    checkpoint_path: &Path,
    output_path: &Path,
    dry_run: bool,
) -> std::result::Result<usize, GwenError> {
    if !checkpoint_path.exists() {
        return Err(GwenError::CandleError(format!(
            "checkpoint path does not exist: {}",
            checkpoint_path.display()
        )));
    }

    // Load the checkpoint into a VarMap.
    // @DANGER VarMap::load() reads the full file into RAM. For large checkpoints
    // on 8 GB machines this may be tight; the 2 GB merge budget in LoraMerger
    // is separate from this allocation.
    let mut varmap = VarMap::new();
    varmap
        .load(checkpoint_path)
        .map_err(|e| GwenError::CandleError(format!("VarMap::load failed: {e}")))?;

    // Build a default LoraConfig — rank/alpha are inferred from tensor shapes
    // inside extract_adapters(), so the values here only govern target_modules.
    let lora_config = BridgeLoraConfig::default();
    let exporter = LoraExporter::new(lora_config);

    // Extract adapters to validate the checkpoint structure.
    let adapters = exporter
        .extract_adapters(&varmap)
        .map_err(|e| GwenError::CandleError(e.to_string()))?;

    let count = adapters.len();

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
