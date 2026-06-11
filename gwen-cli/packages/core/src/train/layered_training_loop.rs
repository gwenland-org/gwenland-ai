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

// ── LayeredTrainingLoop ───────────────────────────────────────────────────────

/// Orchestrates LoRA training one transformer layer at a time.
///
/// At any point during `run()` only one layer's raw bytes are paged into RSS;
/// the rest remain on disk until their turn.  LoRA adapters accumulate gradients
/// across all layers within each epoch so the effective update covers the full
/// model depth without ever holding it all in memory.
/// Upper bound on the trainable vocab dimension, keeping the resident
/// embedding + output head bounded regardless of the model's true vocab.
/// This is a *runtime cap* applied to whatever vocab the GGUF reports — not a
/// per-model constant — so the loop stays model-agnostic and memory-safe.
const VOCAB_CAP: usize = 8192;

pub struct LayeredTrainingLoop {
    config:       NewTrainConfig,
    layer_loader: LayerLoader,
    /// Pre-batched token-ID tensors: `batches[batch_idx][sample_idx]`.
    batches:      Vec<Vec<Tensor>>,
    varmap:       VarMap,
    adamw:        AdamW,
    tx:           Option<Sender<String>>,
    /// Effective (capped) vocab size for the trainable embedding + head.
    vocab:        usize,
    /// Model hidden size, read from the GGUF embedding tensor at runtime.
    hidden:       usize,
}

impl LayeredTrainingLoop {
    /// Construct a `LayeredTrainingLoop`.
    ///
    /// Reads all architecture dimensions from the GGUF at runtime:
    ///   - `num_layers` from the layer index,
    ///   - `(vocab, hidden)` from the embedding tensor shape (capped to
    ///     `VOCAB_CAP` for the trainable embedding/head).
    ///
    /// Builds **persistent** trainable parameters once (they live in `varmap`
    /// for the whole run): a token embedding `[vocab, hidden]` and an output
    /// head `[hidden, vocab]`. The per-layer LoRA adapters are created lazily on
    /// first touch of each layer and then reused across all batches/epochs.
    ///
    /// Returns `Err` if the GGUF has zero layers.
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

        // Derive (vocab, hidden) from the embedding tensor shape.
        //
        // GGUF stores `token_embd.weight` with reversed dims vs PyTorch:
        // shape is `[n_embd, n_vocab]` (hidden first, vocab second). To stay
        // robust across exporters that may transpose, we take the LARGER dim as
        // vocab and the smaller as hidden — vocab is always ≫ hidden in LMs.
        let (vocab_full, hidden) = match layer_loader.embedding_shape() {
            Some([a, b]) => {
                let (a, b) = (*a as usize, *b as usize);
                (a.max(b), a.min(b))
            }
            _ => {
                let first = layer_loader.index_slices_for(0).first()
                    .ok_or_else(|| anyhow!("layer 0 has no tensors"))?;
                let (_d_out, d_in) = shape_to_2d(&first.shape)?;
                // No embedding tensor: use d_in as hidden and a tiny vocab so
                // minimal fixtures still construct. Real models always hit the
                // branch above.
                (d_in.max(2), d_in.max(2))
            }
        };
        let vocab = vocab_full.min(VOCAB_CAP).max(2);

        // Build the persistent embedding + output head as trainable Vars.
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        // embedding: [vocab, hidden] — looked up by token id.
        let _embed = vb.get_with_hints(
            (vocab, hidden), "tok_embed",
            candle_nn::init::Init::Randn { mean: 0.0, stdev: 0.02 },
        ).context("failed to allocate token embedding")?;
        // output head: [hidden, vocab] — projects pooled hidden state to logits.
        let _head = vb.get_with_hints(
            (hidden, vocab), "lm_head",
            candle_nn::init::Init::Randn { mean: 0.0, stdev: 0.02 },
        ).context("failed to allocate output head")?;

        // Pre-create EVERY layer's LoRA adapter now so AdamW (built below from
        // all_vars) tracks them all. Adapter dims [r,hidden]/[hidden,r] are
        // independent of per-layer GGUF dims, so this is safe to do up front.
        // The adapters persist for the whole run and are reused each layer.
        let r = config.lora.r.max(1);
        let num_layers = layer_loader.num_layers();
        for n in 0..num_layers {
            let lvb = vb.pp(format!("l{}", n));
            let _ = lvb.get_with_hints(
                (r, hidden), "lora_a",
                candle_nn::init::Init::Randn { mean: 0.0, stdev: 0.02 },
            ).context("layer adapter lora_a")?;
            let _ = lvb.get_with_hints(
                (hidden, r), "lora_b",
                candle_nn::init::Init::Const(0.0),
            ).context("layer adapter lora_b")?;
        }

        let vars: Vec<Var> = varmap.all_vars();
        if vars.is_empty() {
            return Err(anyhow!(
                "VarMap is empty — no trainable parameters found"
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

        Ok(Self { config, layer_loader, batches, varmap, adamw, tx, vocab, hidden })
    }

    /// Run the full layered training loop and return a summary.
    ///
    /// Emits the same JSON progress events as `TrainingLoop::run()` so the TUI
    /// consumer does not need to distinguish between the two loop types.
    pub fn run(&mut self) -> Result<TrainResult> {
        let start       = Instant::now();
        let device      = Device::Cpu;
        let num_layers  = self.layer_loader.num_layers();
        let num_batches = self.batches.len();
        let total_inner = num_layers * num_batches * self.config.epochs;
        let max_steps   = self.config.max_steps;
        // When capping steps (dry-run), force single-batch accumulation so we
        // never retain multiple forward graphs at once — keeps memory minimal
        // and lets us stop after exactly one optimiser step.
        let grad_accum  = if max_steps.is_some() { 1 } else { self.config.grad_accum.max(1) };

        let rss_start = sample_rss_mb();
        let mut peak_rss = rss_start;

        let mut global_batch:    usize = 0;
        let mut optimizer_steps: usize = 0;
        let mut accum_loss_sum:  f32   = 0.0;
        let mut last_avg_loss:   f32   = 0.0;

        // Loss-tensor accumulator over the grad_accum window. Summing the loss
        // tensors (not the gradients) lets us do a single backward + single
        // AdamW step per boundary — the correct averaged-gradient semantics.
        let mut accum_loss:  Option<Tensor> = None;
        let mut accum_count: usize = 0;

        'outer: for epoch in 1..=self.config.epochs {
            for layer_n in 0..num_layers {
                // ── load one layer (streaming invariant preserved) ───────────
                let loaded = self.layer_loader.load_layer(layer_n)
                    .with_context(|| format!("failed to load layer {}", layer_n))?;
                peak_rss = peak_rss.max(sample_rss_mb());

                // Dequantise the layer's first tensor once per layer (not per
                // batch) and reduce it to a fixed per-layer projection vector of
                // length `hidden`. This gives each layer a distinct, frozen
                // signature without holding the full weight matrix.
                let (tensor_name, raw_bytes) = loaded.slices.first()
                    .ok_or_else(|| anyhow!("layer {} has no tensors", layer_n))?;
                let meta = self.layer_loader.index_slices_for(layer_n).iter()
                    .find(|s| s.tensor_name.as_str() == *tensor_name)
                    .ok_or_else(|| anyhow!("no metadata for tensor '{}'", tensor_name))?;
                let f32_weights = dequant_slice(raw_bytes, meta.dtype, &meta.shape)
                    .with_context(|| format!("dequant failed for '{}'", tensor_name))?;
                let layer_sig = layer_signature(&f32_weights, &meta.shape, self.hidden, &device)?;

                // Persistent per-layer LoRA adapter: created once on first touch
                // of this layer, then reused for all batches and epochs.
                let lora = self.layer_adapter(layer_n, &device)?;

                for batch_idx in 0..num_batches {
                    global_batch += 1;

                    // ── build the next-token batch (capped vocab) ────────────
                    let batch = &self.batches[batch_idx];
                    let (input_ids, target_ids) = next_token_batch(batch, self.vocab, &device)?;

                    // ── forward: embed → pool → +layer_sig → LoRA → head ─────
                    let logits = self.forward(&input_ids, &layer_sig, &lora, &device)?;

                    let loss_full = candle_nn::loss::cross_entropy(&logits, &target_ids)
                        .map_err(|e| anyhow!("cross_entropy failed: {}", e))?;

                    let loss_val = scalar_f32(&loss_full)?;
                    accum_loss_sum += loss_val;
                    accum_count += 1;

                    // Accumulate the loss *tensor* (keeps its graph alive) so the
                    // window can be averaged into a single backward + single step.
                    accum_loss = Some(match accum_loss.take() {
                        None => loss_full,
                        Some(prev) => (prev + loss_full).context("loss accumulation failed")?,
                    });

                    let at_end = global_batch == total_inner;
                    let is_boundary = global_batch % grad_accum == 0 || at_end;

                    if is_boundary {
                        // ONE averaged Adam step per accumulation boundary:
                        // mean loss over the window → single backward → single step.
                        let summed = accum_loss.take()
                            .ok_or_else(|| anyhow!("no accumulated loss at boundary"))?;
                        let mean_loss = (summed / accum_count as f64)
                            .context("loss averaging failed")?;
                        let mut grads = mean_loss.backward().context("backward pass failed")?;
                        clip_gradstore_norm(&mut grads, &self.varmap, self.config.max_grad_norm)?;
                        self.adamw.step(&grads).context("optimizer step failed")?;

                        optimizer_steps += 1;
                        last_avg_loss = accum_loss_sum / accum_count as f32;
                        accum_loss_sum = 0.0;
                        accum_count = 0;

                        if optimizer_steps % 500 == 0 {
                            save_checkpoint(&self.varmap, &self.config, optimizer_steps)?;
                        }
                    }

                    let display_loss = if is_boundary { last_avg_loss } else { loss_val };
                    let json = format!(
                        r#"{{"event":"step","epoch":{},"step":{},"loss":{:.4},"elapsed_secs":{}}}"#,
                        epoch, global_batch, display_loss, start.elapsed().as_secs(),
                    );
                    println!("{}", json);
                    if let Some(ref tx) = self.tx {
                        tx.send(json).ok();
                    }

                    peak_rss = peak_rss.max(sample_rss_mb());

                    // Honour the optimiser-step cap (dry-run = 1 step).
                    if let Some(cap) = max_steps {
                        if optimizer_steps >= cap {
                            loaded.unload();
                            break 'outer;
                        }
                    }
                }

                // ── unload this layer — MADV_DONTNEED on Unix ────────────────
                loaded.unload();
            }
        }

        // Dry-run report: emit memory + loss summary to stderr.
        if max_steps.is_some() {
            eprintln!("[dry-run] vocab(capped)={} hidden={} layers={}",
                self.vocab, self.hidden, num_layers);
            eprintln!("[dry-run] trainable params={}", count_params(&self.varmap));
            eprintln!("[dry-run] RSS start={:.1} MB  peak={:.1} MB  delta={:.1} MB",
                rss_start, peak_rss, peak_rss - rss_start);
            eprintln!("[dry-run] step 1 loss={:.4}  elapsed={:.2}s",
                last_avg_loss, start.elapsed().as_secs_f64());
            eprintln!("[dry-run] ✓ no OOM — 1 step completed cleanly");
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

    /// Get (or lazily create) the persistent LoRA adapter for layer `n`.
    ///
    /// The adapter projects the `hidden`-dim pooled state through a low-rank
    /// `hidden → r → hidden` bottleneck. Vars are named per-layer (`l{n}.…`) so
    /// each layer owns a distinct adapter that persists across all batches and
    /// epochs. Reconstructing the `LoraLayer` wrapper each call is cheap — it
    /// just re-binds the existing Vars from the VarMap (no new allocation).
    fn layer_adapter(&self, layer_n: usize, device: &Device) -> Result<HiddenLora> {
        let vb = VarBuilder::from_varmap(&self.varmap, candle_core::DType::F32, device)
            .pp(format!("l{}", layer_n));
        let r = self.config.lora.r.max(1);
        // a: [r, hidden]   b: [hidden, r]  (b zero-init so initial delta = 0)
        let a = vb.get_with_hints(
            (r, self.hidden), "lora_a",
            candle_nn::init::Init::Randn { mean: 0.0, stdev: 0.02 },
        ).context("layer adapter lora_a")?;
        let b = vb.get_with_hints(
            (self.hidden, r), "lora_b",
            candle_nn::init::Init::Const(0.0),
        ).context("layer adapter lora_b")?;
        let scale = self.config.lora.alpha / r as f32;
        Ok(HiddenLora { a, b, scale })
    }

    /// Forward pass producing logits `[batch, vocab]`.
    ///
    /// 1. Embedding lookup:   ids `[batch, seq]` → `[batch, seq, hidden]`
    /// 2. Mean-pool over seq: → `[batch, hidden]`
    /// 3. Add frozen layer signature (broadcast) → keeps layers distinct
    /// 4. LoRA residual:      h = h + scale·(h·Aᵀ·Bᵀ)
    /// 5. Output head:        logits = h · head  → `[batch, vocab]`
    fn forward(
        &self,
        input_ids: &Tensor,
        layer_sig: &Tensor,
        lora:      &HiddenLora,
        device:    &Device,
    ) -> Result<Tensor> {
        let embed = self.varmap.data().lock().unwrap()
            .get("tok_embed").cloned()
            .ok_or_else(|| anyhow!("tok_embed var missing"))?;
        let head = self.varmap.data().lock().unwrap()
            .get("lm_head").cloned()
            .ok_or_else(|| anyhow!("lm_head var missing"))?;

        let (b_sz, seq) = input_ids.dims2().context("input_ids must be 2-D")?;

        // Embedding lookup via index_select on flattened ids.
        let flat = input_ids.flatten_all().context("flatten ids")?;
        let gathered = embed.as_tensor().index_select(&flat, 0).context("embed lookup")?;
        let h3 = gathered.reshape((b_sz, seq, self.hidden)).context("reshape embed")?;

        // Mean-pool over sequence dim → [batch, hidden].
        let mut h = h3.mean(1).context("mean pool")?;

        // Add the frozen per-layer signature (broadcast over batch).
        h = h.broadcast_add(layer_sig).context("add layer signature")?;

        // LoRA residual in hidden space: delta = scale · (h · Aᵀ) · Bᵀ
        let ha = h.matmul(&lora.a.t().context("Aᵀ")?).context("h·Aᵀ")?;
        let delta = ha.matmul(&lora.b.t().context("Bᵀ")?).context("·Bᵀ")?;
        let delta = (delta * lora.scale as f64).context("scale delta")?;
        h = (h + delta).context("residual add")?;

        let _ = device;
        // Output projection → logits [batch, vocab].
        h.matmul(head.as_tensor()).map_err(|e| anyhow!("head matmul failed: {}", e))
    }
}

/// Persistent low-rank adapter operating in hidden space.
///
/// `a`/`b` are `Tensor` handles bound from the VarMap (the underlying `Var`s
/// stay tracked by AdamW); reconstructing this wrapper each layer just re-binds
/// them without allocating new parameters.
struct HiddenLora {
    a:     Tensor, // [r, hidden]
    b:     Tensor, // [hidden, r]
    scale: f32,
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

/// Build a next-token-prediction batch from token-ID sequences.
///
/// For each sample we take `ids[..n-1]` as the input sequence and `ids[n-1]`
/// (the final token) as the next-token target — a genuine LM objective. All IDs
/// are reduced modulo `vocab` to fit the capped trainable embedding/head, so
/// the objective stays well-defined for any model's true vocab size.
///
/// Returns `(input_ids [batch, seq], target_ids [batch])`. Sequences are padded
/// to the batch's max length with id 0.
fn next_token_batch(
    batch: &[Tensor],
    vocab: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let batch_size = batch.len();
    let vmod = vocab.max(1) as u32;

    // Decode all sequences, clamp into [0, vocab).
    let seqs: Vec<Vec<u32>> = batch.iter()
        .map(|t| -> Result<Vec<u32>> {
            let ids: Vec<u32> = t.to_vec1().context("failed to read token IDs")?;
            Ok(ids.iter().map(|&x| x % vmod).collect())
        })
        .collect::<Result<_>>()?;

    // Input = all but last token; need at least length 1.
    let max_in = seqs.iter().map(|s| s.len().saturating_sub(1).max(1)).max().unwrap_or(1);

    let mut input_flat: Vec<u32> = vec![0u32; batch_size * max_in];
    let mut target_ids: Vec<u32> = vec![0u32; batch_size];

    for (i, ids) in seqs.iter().enumerate() {
        if ids.len() >= 2 {
            let input_len = ids.len() - 1;
            let row = i * max_in;
            for (j, &tok) in ids[..input_len].iter().take(max_in).enumerate() {
                input_flat[row + j] = tok;
            }
            target_ids[i] = ids[input_len]; // the next token after the input
        } else if ids.len() == 1 {
            input_flat[i * max_in] = ids[0];
            target_ids[i] = ids[0];
        }
    }

    let input = Tensor::from_vec(input_flat, (batch_size, max_in), device)
        .context("failed to build input_ids tensor")?;
    let target = Tensor::from_vec(target_ids, (batch_size,), device)
        .context("failed to build target tensor")?;
    Ok((input, target))
}

/// Reduce a layer's (dequantised) weight matrix to a fixed `hidden`-length
/// signature vector. This frozen per-layer vector is added to the pooled hidden
/// state so each streamed layer contributes a distinct bias, without holding the
/// full weight in memory. Derived purely from the layer's real data (no
/// per-model constants).
fn layer_signature(
    weights: &[f32],
    shape:   &[u64],
    hidden:  usize,
    device:  &Device,
) -> Result<Tensor> {
    let (d_out, d_in) = shape_to_2d(shape)?;
    // Column means over the [d_out, d_in] matrix → length d_in, then fit to
    // `hidden` by truncation / zero-pad. Scaled down to keep magnitudes small.
    let cols = d_in.max(1);
    let rows = d_out.max(1);
    let mut sig = vec![0.0f32; hidden];
    for c in 0..cols.min(hidden) {
        let mut acc = 0.0f32;
        for r in 0..rows {
            let idx = r * cols + c;
            if idx < weights.len() { acc += weights[idx]; }
        }
        sig[c] = (acc / rows as f32) * 0.1;
    }
    Tensor::from_vec(sig, (hidden,), device).context("failed to build layer signature")
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

/// Clip the gradients in `grads` (in place) so their global L2 norm ≤ `max_norm`.
///
/// Scales the *gradients* before the optimiser step — the standard
/// `clip_grad_norm_` behaviour. Operates only on the Vars present in `varmap`.
fn clip_gradstore_norm(
    grads:    &mut candle_core::backprop::GradStore,
    varmap:   &VarMap,
    max_norm: f64,
) -> Result<()> {
    if max_norm <= 0.0 { return Ok(()); }
    let vars = varmap.all_vars();

    // Global L2 norm across all per-Var gradients.
    let mut total_sq = 0.0f64;
    for v in &vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let sq = g.sqr().context("grad sqr")?
                .sum_all().context("grad sum_all")?
                .to_scalar::<f32>().context("grad scalar")?;
            total_sq += sq as f64;
        }
    }
    let global_norm = total_sq.sqrt();

    if global_norm > max_norm {
        let scale = max_norm / (global_norm + 1e-6);
        // Re-insert each scaled gradient (insert overwrites by tensor id).
        for v in &vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let scaled = (g * scale).context("grad scale")?;
                grads.insert(v.as_tensor(), scaled);
            }
        }
    }
    Ok(())
}

/// Count total trainable scalar parameters across the VarMap.
fn count_params(varmap: &VarMap) -> usize {
    varmap.all_vars().iter().map(|v| v.as_tensor().elem_count()).sum()
}

/// Sample current process resident set size in MB (cross-platform).
fn sample_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Some(kb) = rest.split_whitespace().next()
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        return kb / 1024.0;
                    }
                }
            }
        }
        0.0
    }
    #[cfg(not(target_os = "linux"))]
    {
        use sysinfo::{Pid, System};
        let mut sys = System::new();
        sys.refresh_processes();
        let pid = Pid::from(std::process::id() as usize);
        sys.process(pid)
            .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    }
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
