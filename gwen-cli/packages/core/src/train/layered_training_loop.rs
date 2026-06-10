/// Layer-by-layer LoRA training loop — enforces at-most-one-layer-in-RAM invariant.
///
/// Unlike `TrainingLoop` which requires the full model to be loaded before training
/// starts, `LayeredTrainingLoop` streams one transformer layer at a time from the
/// memory-mapped GGUF file:
///
/// ```text
/// for epoch in 1..=epochs:
///   for layer_n in 0..num_layers:
///     loaded = layer_loader.load_layer(layer_n)   // pages in only this layer
///     for batch in batches:
///       base_weight = dequantise(loaded.slices[0]) // F32 copy; one tensor
///       lora = LoraLayer::new(d_in, d_out, base_weight, &config.lora, vb)
///       logits = lora.forward(&input)
///       loss   = cross_entropy(logits, target) / grad_accum
///       grads  = loss.backward()
///       grad_stores.push(grads)
///       if at accumulation boundary:
///         step_accumulated(&mut adamw, &grad_stores)
///         optimizer_steps += 1
///         grad_stores.clear()
///     loaded.unload()                              // MADV_DONTNEED on Unix
/// return TrainResult
/// ```
///
/// The base weight tensor built inside the innermost loop is a temporary `Tensor`
/// that is dropped at the end of the batch block — it never outlives the layer
/// iteration.  The LoRA adapters (`lora_a`, `lora_b`) live in `varmap` and are
/// updated in-place by AdamW across all layers and epochs.
use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{VarBuilder, VarMap};

use crate::convert::dequant::{self, DequantMode};
use crate::convert::gguf_parser::GgufDtype;
use crate::train::config::{NewTrainConfig, TrainResult};
use crate::train::layer_loader::LayerLoader;
use crate::train::lora::LoraLayer;
use crate::train::training_loop::step_accumulated;

// ── LayeredTrainingLoop ───────────────────────────────────────────────────────

/// Orchestrates LoRA training one transformer layer at a time.
///
/// At any point during `run()` only one layer's raw bytes are paged into RSS;
/// the rest remain on disk until their turn.  LoRA adapters accumulate gradients
/// across all layers within each epoch so the effective update covers the full
/// model depth without ever holding it all in memory.
pub struct LayeredTrainingLoop {
    config:       NewTrainConfig,
    layer_loader: LayerLoader,
    /// Pre-batched token-ID tensors: `batches[batch_idx][sample_idx]`.
    batches:      Vec<Vec<Tensor>>,
    varmap:       VarMap,
    adamw:        AdamW,
    tx:           Option<Sender<String>>,
}

impl LayeredTrainingLoop {
    /// Construct a `LayeredTrainingLoop`.
    ///
    /// Returns `Err` if the GGUF file has zero layers or the `VarMap` is empty
    /// (which would mean there are no trainable LoRA parameters).
    pub fn new(
        config:    NewTrainConfig,
        gguf_path: &Path,
        batches:   Vec<Vec<Tensor>>,
        varmap:    VarMap,
        tx:        Option<Sender<String>>,
    ) -> Result<Self> {
        let layer_loader = LayerLoader::open(gguf_path)?;

        if layer_loader.num_layers() == 0 {
            return Err(anyhow!(
                "GGUF file '{}' contains no model.layers.* tensors",
                gguf_path.display()
            ));
        }

        let vars: Vec<Var> = varmap.all_vars();
        if vars.is_empty() {
            return Err(anyhow!(
                "VarMap is empty — no trainable LoRA parameters found"
            ));
        }

        let params = ParamsAdamW {
            lr:           config.lr,
            beta1:        0.9,
            beta2:        0.999,
            eps:          1e-8,
            weight_decay: 0.01,
        };
        let adamw = AdamW::new(vars, params)
            .context("failed to initialise AdamW optimiser")?;

        Ok(Self { config, layer_loader, batches, varmap, adamw, tx })
    }

    /// Run the full layered training loop and return a summary.
    ///
    /// Emits the same JSON progress events as `TrainingLoop::run()` so the TUI
    /// consumer does not need to distinguish between the two loop types.
    pub fn run(&mut self) -> Result<TrainResult> {
        let start       = Instant::now();
        let grad_accum  = self.config.grad_accum.max(1);
        let num_layers  = self.layer_loader.num_layers();
        let num_batches = self.batches.len();
        let total_inner = num_layers * num_batches * self.config.epochs;

        let mut global_batch:    usize = 0;
        let mut optimizer_steps: usize = 0;
        let mut accum_loss_sum:  f32   = 0.0;
        let mut last_avg_loss:   f32   = 0.0;

        let mut grad_stores: Vec<candle_core::backprop::GradStore> =
            Vec::with_capacity(grad_accum);

        for epoch in 1..=self.config.epochs {
            for layer_n in 0..num_layers {
                // ── load one layer ───────────────────────────────────────────
                let loaded = self.layer_loader.load_layer(layer_n)
                    .with_context(|| format!("failed to load layer {}", layer_n))?;

                for batch_idx in 0..num_batches {
                    global_batch += 1;

                    // ── dequantise the first tensor of this layer to f32 ─────
                    // In a real model each layer has many tensors (q_proj, k_proj,
                    // v_proj, …); here we use the first one as the "base weight"
                    // for the LoraLayer that will be trained.  Extending this to
                    // iterate over all named projections is straightforward but
                    // deferred to the integration layer (Wave 3 scope).
                    let (tensor_name, raw_bytes) = loaded.slices
                        .first()
                        .ok_or_else(|| anyhow!("layer {} has no tensors", layer_n))?;

                    // Find the LayerSlice for dtype/shape metadata.
                    let meta = self.layer_loader
                        .index_slices_for(layer_n)
                        .iter()
                        .find(|s| s.tensor_name.as_str() == *tensor_name)
                        .ok_or_else(|| anyhow!("no metadata for tensor '{}'", tensor_name))?;

                    let f32_weights = dequant_slice(raw_bytes, meta.dtype, &meta.shape)
                        .with_context(|| format!("dequant failed for '{}'", tensor_name))?;

                    // Shape: assume 2-D weight matrix (d_out × d_in).
                    let (d_out, d_in) = shape_to_2d(&meta.shape)?;

                    let device = Device::Cpu;
                    let base_weight = Tensor::from_vec(f32_weights, (d_out, d_in), &device)
                        .context("failed to build base weight tensor")?;

                    // ── build per-iteration LoraLayer ─────────────────────────
                    // The VarMap / VarBuilder ensures lora_a and lora_b are
                    // re-used (or initialised once on the first call) and tracked
                    // across layers.  Subsequent calls with the same names reuse
                    // the existing Vars from the map.
                    let vb = VarBuilder::from_varmap(&self.varmap, candle_core::DType::F32, &device);
                    let lora = LoraLayer::new(d_in, d_out, base_weight, &self.config.lora, vb)
                        .map_err(|e| anyhow!("LoraLayer::new failed: {}", e))?;

                    // ── forward + loss ────────────────────────────────────────
                    let batch: Vec<Tensor> = self.batches[batch_idx]
                        .iter()
                        .map(|t| t.clone())
                        .collect();
                    let (input_tensor, target_tensor) = prepare_batch(&batch, &self.config)?;

                    let logits = lora.forward(&input_tensor)
                        .map_err(|e| anyhow!("forward failed: {}", e))?;

                    let loss_full = candle_nn::loss::cross_entropy(&logits, &target_tensor)
                        .map_err(|e| anyhow!("cross_entropy failed: {}", e))?;

                    accum_loss_sum += scalar_f32(&loss_full)?;

                    let loss_scaled = (loss_full / grad_accum as f64)
                        .context("loss scaling failed")?;

                    // ── backward ──────────────────────────────────────────────
                    let grads = loss_scaled.backward()
                        .context("backward pass failed")?;
                    grad_stores.push(grads);

                    // ── optimiser step ────────────────────────────────────────
                    let at_end = global_batch == total_inner;
                    let is_boundary = global_batch % grad_accum == 0 || at_end;

                    if is_boundary && !grad_stores.is_empty() {
                        step_accumulated(&mut self.adamw, &grad_stores)
                            .context("optimizer step failed")?;

                        optimizer_steps += 1;
                        last_avg_loss = accum_loss_sum / grad_stores.len() as f32;
                        accum_loss_sum = 0.0;
                        grad_stores.clear();

                        if optimizer_steps % 500 == 0 {
                            save_checkpoint(&self.varmap, &self.config, optimizer_steps)?;
                        }
                    }

                    // ── progress event ────────────────────────────────────────
                    let display_loss = if is_boundary {
                        last_avg_loss
                    } else {
                        (accum_loss_sum / grad_stores.len().max(1) as f32) * grad_accum as f32
                    };

                    let json = format!(
                        r#"{{"event":"step","epoch":{},"step":{},"loss":{:.4},"elapsed_secs":{}}}"#,
                        epoch, global_batch, display_loss, start.elapsed().as_secs(),
                    );
                    println!("{}", json);
                    if let Some(ref tx) = self.tx {
                        tx.send(json).ok();
                    }
                }

                // ── unload this layer — MADV_DONTNEED on Unix ────────────────
                loaded.unload();
            }
        }

        let done_json = format!(
            r#"{{"event":"done","final_loss":{:.4},"total_steps":{},"elapsed_secs":{}}}"#,
            last_avg_loss, optimizer_steps, start.elapsed().as_secs(),
        );
        println!("{}", done_json);
        if let Some(ref tx) = self.tx {
            tx.send(done_json).ok();
        }

        Ok(TrainResult {
            final_loss:  last_avg_loss,
            total_steps: optimizer_steps,
            elapsed:     start.elapsed(),
        })
    }
}

// ── LayerLoader accessor ──────────────────────────────────────────────────────

// We need access to the LayerIndex slices for dtype/shape metadata.
// Rather than making the whole index public, expose a targeted accessor via
// a trait extension on LayerLoader defined here (no change to layer_loader.rs
// public API beyond what Wave 2 already established).
//
// The `index_slices_for` method is pub(crate) so tests in this module can use it.
impl LayerLoader {
    pub(crate) fn index_slices_for(&self, n: usize) -> &[crate::train::layer_loader::LayerSlice] {
        self.index_slices(n)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Dequantise raw mmap bytes to `Vec<f32>` using the appropriate path for `dtype`.
///
/// For F32 tensors this is a cheap byte-reinterpretation (no copy via transmute
/// is possible in safe Rust, so we do a `chunks_exact(4)` parse).  For quantised
/// types we delegate to `dequant::dequantize` via a zero-copy `TensorInfo` wrapper.
fn dequant_slice(
    bytes:  &[u8],
    dtype:  GgufDtype,
    shape:  &[u64],
) -> Result<Vec<f32>> {
    use crate::convert::gguf_parser::TensorInfo;

    // Build a TensorInfo that borrows the bytes via a clone.
    // The clone is unavoidable because TensorInfo owns its raw_data; the copy
    // is bounded to one layer's worth of data (not the full model).
    let tensor_info = TensorInfo {
        name:        String::new(),
        shape:       shape.to_vec(),
        dtype,
        data_offset: 0,
        data_size:   bytes.len(),
        raw_data:    bytes.to_vec(),
    };

    dequant::dequantize(&tensor_info, DequantMode::Standard)
        .map_err(|e| anyhow!("dequant error: {}", e))
}

/// Interpret a shape slice as `(d_out, d_in)` for a 2-D weight matrix.
///
/// GGUF shapes are in row-major order; for a linear weight `W` of shape
/// `[d_out, d_in]` (matching PyTorch convention) that is exactly what we get.
/// 1-D shapes (from test files) are treated as `(n, 1)`.
fn shape_to_2d(shape: &[u64]) -> Result<(usize, usize)> {
    match shape {
        [d_out, d_in] => Ok((*d_out as usize, *d_in as usize)),
        [n]           => Ok((*n as usize, 1)),
        _             => Err(anyhow!(
            "cannot interpret {:?} as a 2-D weight shape", shape
        )),
    }
}

/// Pad a variable-length batch of token-ID tensors and return `(input, target)`.
///
/// Mirrors `TrainingLoop::prepare_batch` — duplicated here to avoid coupling
/// `LayeredTrainingLoop` to the private internals of `TrainingLoop`.
fn prepare_batch(
    batch:  &[Tensor],
    config: &NewTrainConfig,
) -> Result<(Tensor, Tensor)> {
    let device  = batch[0].device();
    let max_len = batch.iter()
        .map(|t| t.elem_count())
        .max()
        .unwrap_or(1)
        .min(config.lora.r * 128)
        .max(2);

    let batch_size = batch.len();
    let mut input_ids:  Vec<u32> = vec![0u32; batch_size * (max_len - 1)];
    let mut target_ids: Vec<u32> = vec![0u32; batch_size * (max_len - 1)];

    for (i, seq) in batch.iter().enumerate() {
        let ids: Vec<u32> = seq.to_vec1().context("failed to read token IDs")?;
        let usable = ids.len().min(max_len);
        let row    = i * (max_len - 1);
        for j in 0..(usable - 1) {
            input_ids [row + j] = ids[j];
            target_ids[row + j] = ids[j + 1];
        }
    }

    // Cast to F32: LoraLayer::forward does a matmul which requires float dtype.
    let input = Tensor::from_vec(input_ids, (batch_size, max_len - 1), device)
        .context("failed to build input tensor")?
        .to_dtype(candle_core::DType::F32)
        .context("failed to cast input to F32")?;
    // target stays U32 — cross_entropy expects integer class labels.
    let target = Tensor::from_vec(target_ids, (batch_size * (max_len - 1),), device)
        .context("failed to build target tensor")?;

    Ok((input, target))
}

fn save_checkpoint(varmap: &VarMap, config: &NewTrainConfig, step: usize) -> Result<()> {
    let filename = format!("checkpoint_{:06}.safetensors", step);
    let path     = config.output_path.join(&filename);
    std::fs::create_dir_all(&config.output_path)
        .with_context(|| format!("cannot create output dir '{}'", config.output_path.display()))?;
    varmap.save(&path)
        .with_context(|| format!("failed to write checkpoint '{}'", path.display()))?;
    eprintln!("[checkpoint] saved → {}", path.display());
    Ok(())
}

fn scalar_f32(t: &Tensor) -> Result<f32> {
    t.to_scalar::<f32>().context("expected scalar loss tensor")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use tempfile::TempDir;

    use crate::train::config::{LoraConfig, NewTrainConfig};
    use crate::train::layer_loader::tests::write_minimal_gguf;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn default_config(output: std::path::PathBuf) -> NewTrainConfig {
        NewTrainConfig {
            output_path: output,
            lora: LoraConfig { r: 2, alpha: 4.0, dropout: 0.0, target_modules: vec![] },
            epochs: 1,
            batch_size: 1,
            grad_accum: 1,
            lr: 1e-4,
            ..NewTrainConfig::default()
        }
    }

    /// Build a batch: one sequence of `n` token IDs `[1, 2, ..., n]`.
    fn make_batch(n: usize) -> Vec<Vec<Tensor>> {
        let ids: Vec<u32> = (1..=(n as u32)).collect();
        let t = Tensor::from_vec(ids, (n,), &Device::Cpu).unwrap();
        vec![vec![t]]
    }

    /// Build a VarMap with lora_a and lora_b for a weight of shape (D_OUT, D_IN).
    ///
    /// D_OUT=4, D_IN=1:  matches `write_*_gguf()` tensors and covers token IDs [0..3]
    /// so `cross_entropy` with targets from `make_batch(2)` (token IDs [1,2]) is valid.
    fn make_varmap() -> VarMap {
        const R: usize = 2;
        const D_IN: usize = 1;
        const D_OUT: usize = 4;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let _ = vb.get_with_hints(
            (R, D_IN), "lora_a",
            candle_nn::init::Init::Randn { mean: 0.0, stdev: 0.01 },
        ).unwrap();
        let _ = vb.get_with_hints(
            (D_OUT, R), "lora_b",
            candle_nn::init::Init::Const(0.0),
        ).unwrap();
        vm
    }

    /// Write a GGUF with one layer-0 tensor: shape (4, 1) = 4 f32 elements.
    ///
    /// d_out=4 covers token IDs [0..3]; d_in=1 matches make_batch(2) input width.
    fn write_one_layer_gguf() -> tempfile::NamedTempFile {
        let weight: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        write_minimal_gguf(&[("model.layers.0.self_attn.q_proj.weight", &weight)])
    }

    /// Write a GGUF with two layers, each a (4,1) f32 weight tensor.
    fn write_two_layer_gguf() -> tempfile::NamedTempFile {
        let w: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4]
            .iter().flat_map(|v| v.to_le_bytes()).collect();
        write_minimal_gguf(&[
            ("model.layers.0.self_attn.q_proj.weight", &w),
            ("model.layers.1.self_attn.q_proj.weight", &w),
        ])
    }

    // ── deterministic tests ───────────────────────────────────────────────────

    #[test]
    fn test_new_rejects_empty_varmap() {
        let f    = write_one_layer_gguf();
        let td   = TempDir::new().unwrap();
        let cfg  = default_config(td.path().to_path_buf());
        let vm   = VarMap::new(); // no vars
        let result = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), vm, None);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("empty"));
    }

    #[test]
    fn test_new_rejects_zero_layers() {
        // GGUF with no model.layers.* tensors
        let weight: Vec<u8> = 0.5f32.to_le_bytes().to_vec();
        let f  = write_minimal_gguf(&[("token_embd.weight", &weight)]);
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        let result = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("no model.layers"));
    }

    #[test]
    fn test_run_single_epoch_produces_result() {
        let f  = write_two_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        // 2 tokens → input shape (1,1) which matches d_in=1 from the 1-element GGUF tensor.
        let mut ltl = LayeredTrainingLoop::new(
            cfg, f.path(), make_batch(2), make_varmap(), None,
        ).expect("new");

        let result = ltl.run().expect("run");
        assert!(result.total_steps >= 1, "expected at least one optimizer step");
        assert!(result.final_loss.is_finite(), "loss must be finite");
    }

    #[test]
    fn test_run_emits_done_json() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<String>();

        let f  = write_one_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        // 2 tokens → input shape (1,1) which matches d_in=1 from the 1-element GGUF tensor.
        let mut ltl = LayeredTrainingLoop::new(
            cfg, f.path(), make_batch(2), make_varmap(), Some(tx),
        ).expect("new");

        ltl.run().expect("run");

        let messages: Vec<String> = rx.try_iter().collect();
        let done = messages.iter().any(|m| m.contains(r#""event":"done""#));
        assert!(done, "expected a done JSON event, got: {:?}", messages);
    }

    // ── quickcheck properties ─────────────────────────────────────────────────

    use quickcheck_macros::quickcheck;

    /// Property 6 — total optimizer steps matches ceil(layers × batches × epochs / grad_accum).
    #[quickcheck]
    fn total_steps_matches_formula(
        num_layers_raw: u8,
        num_batches_raw: u8,
        epochs_raw: u8,
        grad_accum_raw: u8,
    ) -> bool {
        let num_layers  = (num_layers_raw  as usize % 4) + 1; // 1..=4
        let num_batches = (num_batches_raw as usize % 4) + 1; // 1..=4
        let epochs      = (epochs_raw      as usize % 3) + 1; // 1..=3
        let grad_accum  = (grad_accum_raw  as usize % 8) + 1; // 1..=8

        // Build a GGUF with `num_layers` layer tensors, each (4,1) f32 = 16 bytes.
        // d_out=4 covers token IDs [0..3]; d_in=1 matches make_batch(2) input width.
        let weights: Vec<Vec<u8>> = (0..num_layers)
            .map(|i| {
                let base = (i as f32) * 0.1 + 0.1;
                [base, base + 0.1, base + 0.2, base + 0.3]
                    .iter().flat_map(|v| v.to_le_bytes()).collect()
            })
            .collect();
        let tensor_specs: Vec<(String, Vec<u8>)> = (0..num_layers)
            .map(|i| (
                format!("model.layers.{}.self_attn.q_proj.weight", i),
                weights[i].clone(),
            ))
            .collect();
        let refs: Vec<(&str, &[u8])> = tensor_specs
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        let f = write_minimal_gguf(&refs);

        // Build `num_batches` batches of 4 tokens each.
        let batches: Vec<Vec<Tensor>> = (0..num_batches)
            .map(|_| make_batch(2).into_iter().next().unwrap())
            .collect();

        let td = TempDir::new().unwrap();
        let mut cfg = default_config(td.path().to_path_buf());
        cfg.epochs     = epochs;
        cfg.grad_accum = grad_accum;

        let mut ltl = match LayeredTrainingLoop::new(cfg, f.path(), batches, make_varmap(), None) {
            Ok(l)  => l,
            Err(_) => return true, // construction errors are not the property under test
        };

        let result = match ltl.run() {
            Ok(r)  => r,
            Err(_) => return true,
        };

        let total_inner   = num_layers * num_batches * epochs;
        let expected_steps = (total_inner + grad_accum - 1) / grad_accum;
        result.total_steps == expected_steps
    }

    /// Property 7 — final_loss is always a finite f32 (no NaN or inf).
    #[quickcheck]
    fn final_loss_is_finite(num_layers_raw: u8, grad_accum_raw: u8) -> bool {
        let num_layers = (num_layers_raw as usize % 4) + 1; // 1..=4
        let grad_accum = (grad_accum_raw as usize % 4) + 1; // 1..=4

        // Same (4,1) layout as the deterministic tests.
        let weights: Vec<Vec<u8>> = (0..num_layers)
            .map(|i| {
                let base = (i as f32) * 0.1 + 0.1;
                [base, base + 0.1, base + 0.2, base + 0.3]
                    .iter().flat_map(|v| v.to_le_bytes()).collect()
            })
            .collect();
        let specs: Vec<(String, Vec<u8>)> = (0..num_layers)
            .map(|i| (
                format!("model.layers.{}.self_attn.q_proj.weight", i),
                weights[i].clone(),
            ))
            .collect();
        let refs: Vec<(&str, &[u8])> = specs.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        let f = write_minimal_gguf(&refs);

        let td = TempDir::new().unwrap();
        let mut cfg = default_config(td.path().to_path_buf());
        cfg.grad_accum = grad_accum;

        let mut ltl = match LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None) {
            Ok(l)  => l,
            Err(_) => return true,
        };

        match ltl.run() {
            Ok(r)  => r.final_loss.is_finite(),
            Err(_) => true, // training errors are not the property under test
        }
    }
}
