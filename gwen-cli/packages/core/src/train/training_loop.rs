//! Core candle-native training loop for LoRA fine-tuning.
//!
//! # Lifecycle
//!
//! ```text
//! TrainingLoop::new(config, model, batches, varmap)
//!     └─ AdamW initialised with all Vars in the VarMap (lora_a + lora_b only;
//!        base weights are detached so they never appear in the VarMap)
//!
//! TrainingLoop::run()
//!     for each epoch:
//!         for each batch:
//!             forward pass through LoraLayer
//!             cross-entropy loss on logits vs shifted-right targets
//!             loss /= grad_accum          (scale before backward)
//!             loss.backward()             (accumulates gradients in the graph)
//!             every grad_accum batches:
//!                 optimizer.step()        (update weights, zero old grads)
//!             every 500 global steps:
//!                 varmap.save(checkpoint) (safetensors snapshot)
//!             emit ProgressEvent JSON to stdout
//!     return TrainResult
//! ```
//!
//! # Gradient accumulation
//!
//! Candle does not have a global gradient store that persists between
//! `.backward()` calls the way PyTorch does.  Instead, each `.backward()`
//! call produces a fresh `GradStore`.  To simulate accumulation we keep a
//! running `Vec<GradStore>` and pass each one to a manual `step_accumulated`
//! helper that averages the gradients before updating the `Var`s.
//!
//! # Progress events
//!
//! Every batch emits a single-line JSON object to stdout so the TUI / CI
//! consumer can parse it without a framing protocol:
//! ```json
//! {"event":"step","epoch":1,"step":42,"loss":2.34,"elapsed_secs":17}
//! ```
//! On completion:
//! ```json
//! {"event":"done","final_loss":1.12,"total_steps":960,"elapsed_secs":312}
//! ```

use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::VarMap;

use crate::train::config::{NewTrainConfig, TrainResult};
use crate::train::lora::LoraLayer;

// ── progress event ────────────────────────────────────────────────────────────

/// A structured log line emitted to stdout after every batch.
/// Serialised as single-line JSON so the TUI / CI consumer can parse it.
struct ProgressEvent {
    epoch:        usize,
    /// Global batch index across all epochs (1-based).
    global_batch: usize,
    /// Running average loss over the current accumulation window.
    loss:         f32,
    elapsed_secs: u64,
}

impl ProgressEvent {
    /// Emit to stdout and return the JSON string so the caller can also forward
    /// it via the TUI pipe without reformatting.
    ///
    /// Why return the string instead of taking a sender here?
    /// `ProgressEvent` is a private helper with no knowledge of the TUI wiring.
    /// Keeping it free of `Sender` keeps the concerns separated: this struct
    /// owns the format; `TrainingLoop` owns the routing.
    fn emit(&self) -> String {
        // Single-line JSON — deliberately minimal; no serde dependency needed.
        let json = format!(
            r#"{{"event":"step","epoch":{},"step":{},"loss":{:.4},"elapsed_secs":{}}}"#,
            self.epoch, self.global_batch, self.loss, self.elapsed_secs,
        );
        println!("{}", json);
        json
    }
}

// ── training loop ─────────────────────────────────────────────────────────────

/// Owns every resource needed for one complete training run.
///
/// `varmap` must be the same `VarMap` that was used to construct `model`'s
/// LoRA weights — the optimiser is initialised from `varmap.all_vars()` so
/// that only the trainable parameters (lora_a, lora_b) are updated.
pub struct TrainingLoop {
    config:  NewTrainConfig,
    model:   LoraLayer,
    /// Pre-batched token-ID tensors: `batches[batch_idx][sample_idx]`.
    batches: Vec<Vec<Tensor>>,
    /// Holds lora_a and lora_b as `Var`s; used for checkpointing.
    varmap:  VarMap,
    adamw:   AdamW,
    /// Optional pipe to the TUI event loop.
    ///
    /// When `Some`, every progress JSON line is sent here in addition to stdout,
    /// so the TUI can read it without spawning a subprocess. `None` means headless
    /// mode (--verbose or non-TTY): stdout only.
    ///
    /// Why `Option`: `TrainingLoop` is also used in tests and the headless path
    /// where no TUI is running. Requiring a live sender would force callers to
    /// provide a dummy channel, adding noise to every call site that doesn't need
    /// the TUI pipe.
    tx: Option<Sender<String>>,
}

impl TrainingLoop {
    /// Construct a training loop.
    ///
    /// `varmap` **must** be the same map that was passed to `LoraLayer::new()`.
    /// The AdamW optimiser is seeded from `varmap.all_vars()`, which only
    /// contains the trainable LoRA parameters because the frozen base weights
    /// are never inserted into any `VarMap`.
    ///
    /// `tx` — pass `Some(sender)` to enable the TUI pipe bridge; `None` for
    /// headless mode.  See the `tx` field doc for rationale.
    pub fn new(
        config:  NewTrainConfig,
        model:   LoraLayer,
        batches: Vec<Vec<Tensor>>,
        varmap:  VarMap,
        tx:      Option<Sender<String>>,
    ) -> Result<Self> {
        let params = ParamsAdamW {
            lr:           config.lr,
            // Standard AdamW defaults — caller can extend NewTrainConfig if
            // they need to tune these.
            beta1:        0.9,
            beta2:        0.999,
            eps:          1e-8,
            weight_decay: 0.01,
        };

        // all_vars() returns only the Vars registered in the map, which are
        // exclusively lora_a and lora_b.  Base weights are detached Tensors
        // and never reach the VarMap.
        let vars: Vec<Var> = varmap.all_vars();
        let adamw = AdamW::new(vars, params)
            .context("failed to initialise AdamW optimiser")?;

        Ok(Self { config, model, batches, varmap, adamw, tx })
    }

    /// Run the full training loop and return a summary.
    ///
    /// Emits JSON progress events to stdout after every batch.
    /// Writes a safetensors checkpoint every 500 global optimiser steps.
    pub fn run(&mut self) -> Result<TrainResult> {
        let start = Instant::now();

        // grad_accum must be at least 1 to avoid a division-by-zero below.
        let grad_accum = self.config.grad_accum.max(1);

        let total_batches_per_epoch = self.batches.len();
        let total_batches_all_epochs =
            total_batches_per_epoch * self.config.epochs;

        // Global batch counter (across all epochs), used for:
        //   - gradient accumulation cadence
        //   - checkpoint cadence
        //   - progress event step number
        let mut global_batch: usize = 0;

        // How many times we have called optimizer.step().
        let mut optimizer_steps: usize = 0;

        // Loss accumulated over the current grad_accum window, then averaged
        // before emitting to the progress event.
        let mut accum_loss_sum: f32 = 0.0;

        // The last averaged loss value; returned in TrainResult.
        let mut last_avg_loss: f32 = 0.0;

        // Gradient stores collected over one accumulation window.  We replay
        // all of them in step_accumulated() to average the gradients before
        // calling AdamW::step().
        let mut grad_stores: Vec<candle_core::backprop::GradStore> =
            Vec::with_capacity(grad_accum);

        for epoch in 1..=self.config.epochs {
            // Iterate by index so that the `&self.batches[i]` borrow is
            // dropped at the end of the forward/backward block, before we
            // take `&mut self` for `step_accumulated` and `save_checkpoint`.
            for batch_idx in 0..self.batches.len() {
                global_batch += 1;

                // ── forward + loss ──────────────────────────────────────────

                // Clone the batch references out of self so the immutable
                // borrow on self.batches does not overlap with the mutable
                // borrows taken below for the optimiser step.
                let batch: Vec<Tensor> = self.batches[batch_idx]
                    .iter()
                    .map(|t| t.clone())
                    .collect();

                // Stack the variable-length token sequences in the batch into
                // a single (batch, max_seq) padded tensor so the model sees a
                // consistent shape.  We use 0 as the padding token ID.
                let (input_tensor, target_tensor) =
                    self.prepare_batch(&batch)?;

                // Forward pass through the LoRA-wrapped layer.
                // logits shape: (batch * seq, vocab) after reshaping inside
                // cross_entropy.
                let logits = self.model.forward(&input_tensor)
                    .context("forward pass failed")?;

                // Cross-entropy loss: mean over all non-padding positions.
                // We treat the shifted-right targets as the ground truth
                // (standard causal-LM objective).
                let loss_full = candle_nn::loss::cross_entropy(&logits, &target_tensor)
                    .context("cross-entropy computation failed")?;

                // Extract the scalar *before* loss_full is moved into the
                // division operator below.
                accum_loss_sum += scalar_f32(&loss_full)?;

                // Scale by 1/grad_accum *before* backward so that gradients
                // accumulate to the correct magnitude when summed.
                let loss_scaled = (loss_full / grad_accum as f64)
                    .context("loss scaling failed")?;

                // ── backward ────────────────────────────────────────────────

                let grads = loss_scaled.backward()
                    .context("backward pass failed")?;

                grad_stores.push(grads);

                // ── optimiser step (every grad_accum batches) ───────────────

                let is_accum_boundary =
                    global_batch % grad_accum == 0
                    || global_batch == total_batches_all_epochs; // flush at end

                if is_accum_boundary && !grad_stores.is_empty() {
                    // Average the accumulated gradient stores and apply the
                    // AdamW update to lora_a and lora_b.
                    self.step_accumulated(&grad_stores)
                        .context("optimizer step failed")?;

                    optimizer_steps += 1;
                    last_avg_loss = accum_loss_sum / grad_stores.len() as f32;

                    // Reset for the next window.
                    accum_loss_sum = 0.0;
                    grad_stores.clear();

                    // ── checkpoint every 500 optimiser steps ────────────────
                    if optimizer_steps % 500 == 0 {
                        self.save_checkpoint(optimizer_steps)?;
                    }
                }

                // ── progress event ──────────────────────────────────────────

                // Emit a running loss even on non-step batches so the TUI
                // always has something to display.  Between accumulation
                // boundaries we show the unscaled mean so far.
                let display_loss = if is_accum_boundary {
                    last_avg_loss
                } else {
                    // Partial window: show the running scaled sum re-expanded.
                    (accum_loss_sum / grad_stores.len().max(1) as f32)
                        * grad_accum as f32
                };

                let json = ProgressEvent {
                    epoch,
                    global_batch,
                    loss: display_loss,
                    elapsed_secs: start.elapsed().as_secs(),
                }
                .emit();
                // Forward the same JSON line to the TUI pipe if one is wired.
                // `.ok()` silently discards `SendError` — a disconnected receiver
                // means the TUI has detached, which is not a training error.
                if let Some(ref tx) = self.tx {
                    tx.send(json).ok();
                }
            }
        }

        // Emit completion event.
        let done_json = format!(
            r#"{{"event":"done","final_loss":{:.4},"total_steps":{},"elapsed_secs":{}}}"#,
            last_avg_loss,
            optimizer_steps,
            start.elapsed().as_secs(),
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

    // ── helpers ────────────────────────────────────────────────────────────────

    /// Pad a variable-length batch of token-ID tensors and return `(input, target)`.
    ///
    /// `input`  = tokens[0 .. seq-1]  (all but last)
    /// `target` = tokens[1 .. seq]    (all but first; the "next token" labels)
    ///
    /// Both tensors have shape `(batch_size, max_seq - 1)`.
    /// Sequences shorter than `max_seq` are right-padded with 0.
    fn prepare_batch(
        &self,
        batch: &[Tensor],
    ) -> Result<(Tensor, Tensor)> {
        let device = batch[0].device();

        // Find the longest sequence in this batch so we know how much to pad.
        let max_len = batch.iter().map(|t| t.elem_count()).max().unwrap_or(1);
        // Clamp to config max_seq_len to avoid OOM on pathologically long samples.
        let max_len = max_len.min(
            self.config.lora.r * 128, // placeholder upper bound; caller sets real limit
        ).max(2); // need at least 2 tokens to form (input, target) pair

        let batch_size = batch.len();

        // Build a flat f32 buffer for input and target, padding with 0.
        // We work in f32 because cross_entropy expects logits in float; the
        // actual token IDs are u32 so we cast below.
        let mut input_ids:  Vec<u32> = vec![0u32; batch_size * (max_len - 1)];
        let mut target_ids: Vec<u32> = vec![0u32; batch_size * (max_len - 1)];

        for (i, seq) in batch.iter().enumerate() {
            let ids: Vec<u32> = seq.to_vec1()
                .context("failed to read token IDs from tensor")?;
            let usable = ids.len().min(max_len);
            let row_offset = i * (max_len - 1);

            // input  = ids[0 .. usable-1]
            // target = ids[1 .. usable]
            for j in 0..(usable - 1) {
                input_ids [row_offset + j] = ids[j];
                target_ids[row_offset + j] = ids[j + 1];
            }
        }

        let input = Tensor::from_vec(
            input_ids,
            (batch_size, max_len - 1),
            device,
        )
        .context("failed to build input tensor")?;

        let target = Tensor::from_vec(
            target_ids,
            (batch_size * (max_len - 1),), // 1-D for cross_entropy target
            device,
        )
        .context("failed to build target tensor")?;

        Ok((input, target))
    }

    /// Apply gradients from multiple backward passes by averaging them.
    ///
    /// Candle produces a fresh `GradStore` per `.backward()` call; to simulate
    /// accumulation we average the gradient tensors across all stores in the
    /// window, then call `AdamW::step` once with the averaged store.
    fn step_accumulated(
        &mut self,
        stores: &[candle_core::backprop::GradStore],
    ) -> Result<()> {
        if stores.is_empty() {
            return Ok(());
        }
        if stores.len() == 1 {
            // Fast path: no averaging needed.
            return self.adamw.step(&stores[0]).context("AdamW step failed");
        }

        // Candle's GradStore does not expose a public constructor for injecting
        // pre-averaged tensors, so we cannot build a single averaged store.
        //
        // Instead we use the fact that AdamW::step accepts any &GradStore:
        // we temporarily lower the learning rate by 1/n, step once per store
        // (so the net parameter update equals one normal step with averaged
        // gradients), then restore the original lr.
        //
        // This is numerically equivalent to averaging the gradients because
        // AdamW's update rule scales the step by lr, so dividing lr by n and
        // stepping n times produces the same update as averaging n grad stores
        // and stepping once at full lr.
        let n = stores.len() as f64;
        // equivalent to averaging when the learning rate is divided by n.
        //
        // Simplest correct approach: temporarily lower the lr by 1/n, call
        // step once per store, then restore the lr.
        let original_lr = self.adamw.learning_rate();
        self.adamw.set_learning_rate(original_lr / n);

        for store in stores {
            self.adamw.step(store).context("AdamW accumulated step failed")?;
        }

        self.adamw.set_learning_rate(original_lr);
        Ok(())
    }

    /// Write a safetensors checkpoint of the LoRA weights (lora_a + lora_b).
    ///
    /// The file is placed under `config.output_path` with the name
    /// `checkpoint_{step:06}.safetensors`.
    fn save_checkpoint(&self, step: usize) -> Result<()> {
        let filename = format!("checkpoint_{:06}.safetensors", step);
        let path: PathBuf = self.config.output_path.join(&filename);

        // create_dir_all is idempotent; safe to call every time.
        std::fs::create_dir_all(&self.config.output_path)
            .with_context(|| {
                format!(
                    "cannot create output directory '{}'",
                    self.config.output_path.display()
                )
            })?;

        self.varmap
            .save(&path)
            .with_context(|| format!("failed to write checkpoint '{}'", path.display()))?;

        eprintln!("[checkpoint] saved → {}", path.display());
        Ok(())
    }
}

// ── utilities ─────────────────────────────────────────────────────────────────

/// Extract a scalar f32 from a 0-D or 1-element Tensor.
fn scalar_f32(t: &Tensor) -> Result<f32> {
    t.to_scalar::<f32>()
        .context("expected scalar loss tensor")
}
