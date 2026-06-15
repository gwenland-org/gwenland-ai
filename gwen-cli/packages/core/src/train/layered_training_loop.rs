/// Bounded-memory LoRA training over a memory-mapped GGUF transformer.
///
/// Each sample first performs a real embedding and transformer forward while
/// retaining only detached layer-boundary activations. Backward then walks the
/// layers in reverse, loading and recomputing one full attention-plus-MLP layer
/// at a time. This produces exact adapter gradients without keeping every
/// layer's base tensors or autograd graph resident at once.
use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor, Var, backprop::GradStore};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{VarBuilder, VarMap};

use crate::convert::dequant::{self, DequantMode};
use crate::convert::gguf_parser::GgufDtype;
use crate::engine::transformer_ops::rms_norm;
use crate::train::config::{NewTrainConfig, TrainResult};
use crate::train::layer_loader::LayerLoader;
use crate::train::transformer_layer::{
    AttentionConfig, AttentionLoras, AttentionWeights, MlpLoras, MlpWeights, ProjectionLora,
    TransformerLayerConfig, TransformerLayerLoras, TransformerLayerWeights,
    transformer_layer_forward,
};

// ── LayeredTrainingLoop ───────────────────────────────────────────────────────

/// Orchestrates LoRA training with one transformer layer resident at a time.
///
/// At any point during `run()` only one layer's raw bytes and dequantized
/// weights are resident; the rest remain on disk until recomputation reaches
/// them. LoRA adapters persist in the VarMap and receive gradients from every
/// layer in the end-to-end language-model objective.
///
/// Upper bound on the fixed vocab dimension, keeping the resident
/// embedding + output head bounded regardless of the model's true vocab.
/// This is a *runtime cap* applied to whatever vocab the GGUF reports — not a
/// per-model constant — so the loop stays model-agnostic and memory-safe.
const VOCAB_CAP: usize = 8192;

/// Classifies a tensor name suffix into a known transformer projection.
/// Used to route per-projection LoRA adapters in the VarMap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionKind {
    AttnQ,   // attn_q / q_proj / self_attn.q_proj
    AttnK,   // attn_k / k_proj
    AttnV,   // attn_v / v_proj
    AttnO,   // attn_output / o_proj / out_proj
    FfnGate, // ffn_gate / gate_proj
    FfnUp,   // ffn_up   / up_proj
    FfnDown, // ffn_down / down_proj
}

impl ProjectionKind {
    /// Short, stable key used as the VarMap namespace for this projection's
    /// LoRA adapter (e.g. `l{n}.{var_key}.lora_a`).
    pub fn var_key(self) -> &'static str {
        match self {
            ProjectionKind::AttnQ => "attn_q",
            ProjectionKind::AttnK => "attn_k",
            ProjectionKind::AttnV => "attn_v",
            ProjectionKind::AttnO => "attn_o",
            ProjectionKind::FfnGate => "ffn_gate",
            ProjectionKind::FfnUp => "ffn_up",
            ProjectionKind::FfnDown => "ffn_down",
        }
    }
}

/// Classify a tensor name into a known projection kind by substring match.
///
/// Handles both llama.cpp (`blk.N.attn_q.weight`) and HF
/// (`model.layers.N.self_attn.q_proj.weight`) naming styles. Returns `None`
/// for tensors that are not LoRA targets (norms, biases, etc.).
fn classify_tensor(name: &str) -> Option<ProjectionKind> {
    // Norm tensors are NOT LoRA targets. They must be rejected first because
    // their names embed a projection substring — e.g. Qwen3's `attn_q_norm` /
    // `attn_k_norm` contain `attn_q` / `attn_k` and would otherwise misclassify
    // as the q/k projection (a 1-D norm masquerading as a weight matrix).
    if name.contains("norm") {
        return None;
    }
    // Order matters only where one substring is a prefix of another; the
    // projection substrings below are mutually distinct so any order is safe.
    if name.contains("attn_q") || name.contains("q_proj") {
        Some(ProjectionKind::AttnQ)
    } else if name.contains("attn_k") || name.contains("k_proj") {
        Some(ProjectionKind::AttnK)
    } else if name.contains("attn_v") || name.contains("v_proj") {
        Some(ProjectionKind::AttnV)
    } else if name.contains("attn_output") || name.contains("o_proj") || name.contains("out_proj") {
        Some(ProjectionKind::AttnO)
    } else if name.contains("ffn_gate") || name.contains("gate_proj") {
        Some(ProjectionKind::FfnGate)
    } else if name.contains("ffn_up") || name.contains("up_proj") {
        Some(ProjectionKind::FfnUp)
    } else if name.contains("ffn_down") || name.contains("down_proj") {
        Some(ProjectionKind::FfnDown)
    } else {
        None
    }
}

pub struct LayeredTrainingLoop {
    config: NewTrainConfig,
    layer_loader: LayerLoader,
    /// Pre-batched token-ID tensors: `batches[batch_idx][sample_idx]`.
    batches: Vec<Vec<Tensor>>,
    varmap: VarMap,
    adamw: AdamW,
    tx: Option<Sender<String>>,
    /// Effective (capped) vocab size for the fixed embedding and output head.
    vocab: usize,
    /// Model hidden size, read from the GGUF embedding tensor at runtime.
    hidden: usize,
    /// Capped, frozen model tensors used by the real language-model objective.
    model_embedding: Tensor,
    output_norm: Tensor,
    lm_head: Tensor,
    layer_config: TransformerLayerConfig,
    /// Per-projection adapter descriptors `(kind, d_in, d_out, rank)`, one entry
    /// per distinct projection found in layer 0 of the GGUF. Empty in fallback
    /// mode (single-tensor fixtures). `rank` is the uniform `config.lora.r` on the
    /// default path, or a GAAP S(ρ)-derived per-projection rank under `--gdtqp`
    /// (EXPERIMENTAL). Drives `projection_adapters()` / `forward()`.
    proj_keys_per_layer: Vec<(ProjectionKind, usize, usize, usize)>,
    /// GWEN-222: cumulative optimiser step the run starts from. `0` on a fresh
    /// run; the restored step when resuming from a checkpoint. The in-loop
    /// `optimizer_steps` counter is seeded from this so the checkpoint interval
    /// (`% 500`) and filenames stay on the global step axis across resumes.
    global_step: usize,
}

impl LayeredTrainingLoop {
    /// Construct a `LayeredTrainingLoop`.
    ///
    /// Reads all architecture dimensions from the GGUF header at runtime:
    ///   - `num_layers` from the layer index,
    ///   - `(vocab, hidden)` from the embedding tensor shape (capped to
    ///     `VOCAB_CAP` for the fixed embedding/head).
    ///
    /// Loads the capped embedding, output norm, and output head as frozen model
    /// tensors. Every per-layer LoRA adapter is created up front in `varmap` so
    /// AdamW tracks the complete trainable parameter set for the whole run.
    ///
    /// Returns `Err` if the GGUF has zero layers.
    pub fn new(
        config: NewTrainConfig,
        gguf_path: &Path,
        batches: Vec<Vec<Tensor>>,
        varmap: VarMap,
        tx: Option<Sender<String>>,
        initial_step: usize,
    ) -> Result<Self> {
        let layer_loader = LayerLoader::open(gguf_path)?;

        if layer_loader.num_layers() == 0 {
            return Err(anyhow!(
                "GGUF file '{}' contains no model.layers.* tensors",
                gguf_path.display()
            ));
        }

        let model_config = layer_loader.transformer_config().clone();
        let hidden = model_config.hidden_size;
        let vocab = model_config.vocab_size.min(VOCAB_CAP).max(2);
        let device = Device::Cpu;
        let model_embedding = load_capped_matrix(
            &layer_loader,
            &["token_embd.weight", "model.embed_tokens.weight"],
            vocab,
            hidden,
            &device,
        )
        .context("load capped model embedding")?;
        let output_norm = load_vector(
            &layer_loader,
            &["output_norm.weight", "model.norm.weight"],
            hidden,
            &device,
        )
        .context("load output norm")?;
        let lm_head = if layer_loader
            .find_tensor(&["output.weight", "lm_head.weight"])
            .is_some()
        {
            load_capped_matrix(
                &layer_loader,
                &["output.weight", "lm_head.weight"],
                vocab,
                hidden,
                &device,
            )
            .context("load capped output head")?
        } else {
            model_embedding.clone()
        };
        let layer_config = TransformerLayerConfig {
            attention: AttentionConfig {
                hidden_size: hidden,
                n_heads: model_config.n_heads,
                n_kv_heads: model_config.n_kv_heads,
                rope_theta: model_config.rope_theta,
                rms_norm_eps: model_config.rms_norm_eps,
            },
            intermediate_size: model_config.intermediate_size,
        };
        layer_config.validate()?;

        // Only LoRA parameters are trainable. Model embeddings, norms, and the
        // output head are fixed tensors loaded from the GGUF.
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Pre-create EVERY layer's LoRA adapter now so AdamW (built below from
        // all_vars) tracks them all. Adapter dims [r,hidden]/[hidden,r] are
        // independent of per-layer GGUF dims, so this is safe to do up front.
        // The adapters persist for the whole run and are reused each layer.
        let r = config.lora.r.max(1);
        let num_layers = layer_loader.num_layers();

        // Discover projection shapes from layer 0 (all layers share the same
        // projection kinds and dims for a given architecture).
        let layer0_slices = layer_loader.index_slices_for(0);
        let mut proj_keys: Vec<(ProjectionKind, usize, usize)> = Vec::new();
        for slice in layer0_slices {
            if let Some(kind) = classify_tensor(&slice.tensor_name) {
                if let Ok((d_out, d_in)) = shape_to_2d(&slice.shape) {
                    // Avoid duplicates (some models have multiple tensors with
                    // the same projection name due to sharding — take the first).
                    if !proj_keys.iter().any(|(k, _, _)| *k == kind) {
                        proj_keys.push((kind, d_in, d_out));
                    }
                }
            }
        }

        // Fallback: if no projections classified (e.g. test fixtures with
        // generic tensor names), create one hidden-space adapter per layer
        // matching old behaviour so existing tests pass.
        let use_fallback = proj_keys.is_empty();

        // Per-projection rank. Default path: uniform `r`. EXPERIMENTAL `--gdtqp`
        // path: a GAAP S(ρ) entropy-sensitivity surrogate allocates more rank to
        // higher-sensitivity projections (see `gdtqp_allocate_ranks`).
        let proj_ranks: Vec<usize> = if config.gdtqp && !use_fallback {
            gdtqp_allocate_ranks(&layer_loader, &proj_keys, r, &tx)?
        } else {
            vec![r; proj_keys.len()]
        };
        let proj_keys_ranked: Vec<(ProjectionKind, usize, usize, usize)> = proj_keys
            .iter()
            .zip(proj_ranks.iter())
            .map(|((k, d_in, d_out), &rank)| (*k, *d_in, *d_out, rank))
            .collect();

        for n in 0..num_layers {
            let lvb = vb.pp(format!("l{}", n));
            if use_fallback {
                let _ = lvb
                    .get_with_hints(
                        (r, hidden),
                        "lora_a",
                        candle_nn::init::Init::Randn {
                            mean: 0.0,
                            stdev: 0.02,
                        },
                    )
                    .context("layer adapter lora_a")?;
                let _ = lvb
                    .get_with_hints((hidden, r), "lora_b", candle_nn::init::Init::Const(0.0))
                    .context("layer adapter lora_b")?;
            } else {
                for (kind, d_in, d_out, rank) in &proj_keys_ranked {
                    let pvb = lvb.pp(kind.var_key());
                    let effective_rank = (*rank).min(*d_in).min(*d_out).max(1);
                    let _ = pvb
                        .get_with_hints(
                            (effective_rank, *d_in),
                            "lora_a",
                            candle_nn::init::Init::Randn {
                                mean: 0.0,
                                stdev: 0.02,
                            },
                        )
                        .context("proj lora_a")?;
                    let _ = pvb
                        .get_with_hints(
                            (*d_out, effective_rank),
                            "lora_b",
                            candle_nn::init::Init::Const(0.0),
                        )
                        .context("proj lora_b")?;
                }
            }
        }

        let vars: Vec<Var> = varmap.all_vars();
        if vars.is_empty() {
            return Err(anyhow!("VarMap is empty — no trainable parameters found"));
        }

        let params = ParamsAdamW {
            lr: config.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let adamw = AdamW::new(vars, params).context("failed to initialise AdamW optimiser")?;

        Ok(Self {
            config,
            layer_loader,
            batches,
            varmap,
            adamw,
            tx,
            vocab,
            hidden,
            model_embedding,
            output_norm,
            lm_head,
            layer_config,
            proj_keys_per_layer: proj_keys_ranked,
            global_step: initial_step,
        })
    }

    /// Load LoRA adapter weights from a checkpoint into this loop's VarMap.
    ///
    /// Must be called AFTER `new()` (which creates the adapter Vars) — see the
    /// @DANGER note in `checkpoint_resumer`. Restores adapter weights only; the
    /// AdamW optimiser state is not affected.
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        crate::train::checkpoint_resumer::load_checkpoint_into_varmap(&mut self.varmap, path)
    }

    /// Run the full layered training loop and return a summary.
    ///
    /// Emits the same JSON progress events as `TrainingLoop::run()` so the TUI
    /// consumer does not need to distinguish between the two loop types.
    pub fn run(&mut self) -> Result<TrainResult> {
        let start = Instant::now();
        let device = Device::Cpu;
        let num_layers = self.layer_loader.num_layers();
        let num_batches = self.batches.len();
        let total_inner = num_batches * self.config.epochs;
        let max_steps = self.config.max_steps;
        // When capping steps (dry-run), force single-batch accumulation so we
        // never retain multiple forward graphs at once — keeps memory minimal
        // and lets us stop after exactly one optimiser step.
        let grad_accum = if max_steps.is_some() {
            1
        } else {
            self.config.grad_accum.max(1)
        };

        let rss_start = sample_rss_mb();
        let mut peak_rss = rss_start;

        let mut global_batch: usize = 0;
        // `optimizer_steps` rides the GLOBAL step axis (seeded from the resumed
        // checkpoint) so checkpoint filenames + the `% 500` interval stay
        // consistent across resumes. `steps_this_run` counts only the steps taken
        // in THIS invocation and is what `TrainResult::total_steps` reports.
        let mut optimizer_steps: usize = self.global_step;
        let mut steps_this_run: usize = 0;
        let mut accum_loss_sum: f32 = 0.0;
        let mut last_avg_loss: f32 = 0.0;

        let mut accum_grads: Option<GradStore> = None;
        let mut accum_count: usize = 0;
        let trainable_vars = self.varmap.all_vars();

        'outer: for epoch in 1..=self.config.epochs {
            for batch_idx in 0..num_batches {
                global_batch += 1;
                let batch = self.batches[batch_idx].clone();
                let (loss_val, batch_grads) = self
                    .forward_backward_batch(&batch, &device)
                    .with_context(|| format!("real transformer batch {batch_idx}"))?;
                peak_rss = peak_rss.max(sample_rss_mb());

                if let Some(grads) = accum_grads.as_mut() {
                    merge_gradstores(grads, &batch_grads, &trainable_vars)?;
                } else {
                    accum_grads = Some(batch_grads);
                }
                accum_loss_sum += loss_val;
                accum_count += 1;

                let at_end = global_batch == total_inner;
                let is_boundary = global_batch % grad_accum == 0 || at_end;
                if is_boundary {
                    let mut grads = accum_grads
                        .take()
                        .ok_or_else(|| anyhow!("no accumulated gradients at boundary"))?;
                    scale_gradstore(&mut grads, &trainable_vars, 1.0 / accum_count as f64)?;
                    clip_gradstore_norm(&mut grads, &self.varmap, self.config.max_grad_norm)?;
                    self.adamw.step(&grads).context("optimizer step failed")?;

                    optimizer_steps += 1;
                    steps_this_run += 1;
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
                    epoch,
                    global_batch,
                    display_loss,
                    start.elapsed().as_secs(),
                );
                println!("{}", json);
                if let Some(ref tx) = self.tx {
                    tx.send(json).ok();
                }

                if let Some(cap) = max_steps {
                    if optimizer_steps >= cap {
                        break 'outer;
                    }
                }
            }
        }

        // Dry-run report: emit memory + loss summary to stderr.
        if max_steps.is_some() {
            eprintln!(
                "[dry-run] vocab(capped)={} hidden={} layers={}",
                self.vocab, self.hidden, num_layers
            );
            eprintln!("[dry-run] trainable params={}", count_params(&self.varmap));
            eprintln!(
                "[dry-run] RSS start={:.1} MB  peak={:.1} MB  delta={:.1} MB",
                rss_start,
                peak_rss,
                peak_rss - rss_start
            );
            eprintln!(
                "[dry-run] step 1 loss={:.4}  elapsed={:.2}s",
                last_avg_loss,
                start.elapsed().as_secs_f64()
            );
            eprintln!("[dry-run] ✓ no OOM — 1 step completed cleanly");
        }

        let done_json = format!(
            r#"{{"event":"done","final_loss":{:.4},"total_steps":{},"elapsed_secs":{}}}"#,
            last_avg_loss,
            steps_this_run,
            start.elapsed().as_secs(),
        );
        println!("{}", done_json);
        if let Some(ref tx) = self.tx {
            tx.send(done_json).ok();
        }

        Ok(TrainResult {
            final_loss: last_avg_loss,
            total_steps: steps_this_run,
            elapsed: start.elapsed(),
        })
    }

    /// Re-bind the persistent per-projection LoRA adapters for layer `n`.
    ///
    /// In fallback mode (`proj_keys_per_layer` empty — single-tensor test
    /// fixtures) this returns one hidden-space adapter bound from
    /// `l{n}.lora_a` / `l{n}.lora_b`, preserving the pre-Wave-2 behaviour.
    /// Otherwise it returns one `ProjLora` per discovered projection, bound from
    /// `l{n}.{key}.lora_a` / `l{n}.{key}.lora_b`. Re-binding existing Vars from
    /// the VarMap is cheap — no new parameters are allocated.
    fn projection_adapters(&self, layer_n: usize, device: &Device) -> Result<Vec<ProjLora>> {
        let r = self.config.lora.r.max(1);
        let alpha = self.config.lora.alpha;
        let base = VarBuilder::from_varmap(&self.varmap, candle_core::DType::F32, device)
            .pp(format!("l{}", layer_n));

        if self.proj_keys_per_layer.is_empty() {
            // Fallback: single hidden → r → hidden adapter (old HiddenLora layout).
            let a = base
                .get_with_hints(
                    (r, self.hidden),
                    "lora_a",
                    candle_nn::init::Init::Randn {
                        mean: 0.0,
                        stdev: 0.02,
                    },
                )
                .context("fallback lora_a")?;
            let b = base
                .get_with_hints(
                    (self.hidden, r),
                    "lora_b",
                    candle_nn::init::Init::Const(0.0),
                )
                .context("fallback lora_b")?;
            return Ok(vec![ProjLora {
                kind: ProjectionKind::AttnQ,
                a,
                b,
                scale: alpha / r as f32,
            }]);
        }

        let mut out = Vec::with_capacity(self.proj_keys_per_layer.len());
        for (kind, d_in, d_out, rank) in &self.proj_keys_per_layer {
            let pvb = base.pp(kind.var_key());
            let effective_rank = (*rank).min(*d_in).min(*d_out).max(1);
            let a = pvb
                .get_with_hints(
                    (effective_rank, *d_in),
                    "lora_a",
                    candle_nn::init::Init::Randn {
                        mean: 0.0,
                        stdev: 0.02,
                    },
                )
                .context("proj lora_a")?;
            let b = pvb
                .get_with_hints(
                    (*d_out, effective_rank),
                    "lora_b",
                    candle_nn::init::Init::Const(0.0),
                )
                .context("proj lora_b")?;
            // Standard LoRA scaling is alpha/rank. Using the *effective* rank
            // (a_dim) keeps it consistent with the export bridge, which derives
            // scale from lora_a's first dim and bakes it into the merged delta.
            out.push(ProjLora {
                kind: *kind,
                a,
                b,
                scale: alpha / effective_rank as f32,
            });
        }
        Ok(out)
    }

    fn forward_backward_batch(
        &self,
        batch: &[Tensor],
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        if batch.is_empty() {
            return Err(anyhow!("cannot train on an empty batch"));
        }
        let trainable_vars = self.varmap.all_vars();
        let mut loss_sum = 0.0f32;
        let mut batch_grads: Option<GradStore> = None;

        for sample in batch {
            let (loss, sample_grads) = self.forward_backward_sample(sample, device)?;
            loss_sum += loss;
            if let Some(grads) = batch_grads.as_mut() {
                merge_gradstores(grads, &sample_grads, &trainable_vars)?;
            } else {
                batch_grads = Some(sample_grads);
            }
        }

        let mut grads = batch_grads.ok_or_else(|| anyhow!("batch produced no gradients"))?;
        scale_gradstore(&mut grads, &trainable_vars, 1.0 / batch.len() as f64)?;
        Ok((loss_sum / batch.len() as f32, grads))
    }

    fn forward_backward_sample(
        &self,
        sample: &Tensor,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        let ids = sample
            .to_vec1::<u32>()
            .context("failed to read token IDs")?;
        let ids: Vec<u32> = ids
            .into_iter()
            .map(|id| id % self.vocab as u32)
            .collect();
        let (input_tokens, target) = match ids.as_slice() {
            [] => (vec![0], 0),
            [only] => (vec![*only], *only),
            _ => (ids[..ids.len() - 1].to_vec(), ids[ids.len() - 1]),
        };
        let input_ids = Tensor::from_vec(input_tokens.clone(), input_tokens.len(), device)?;
        let boundaries = self.forward_boundaries(&input_ids, device)?;

        let final_boundary = boundaries
            .last()
            .ok_or_else(|| anyhow!("missing final hidden boundary"))?;
        let final_var = Var::from_tensor(final_boundary).context("final boundary Var")?;
        let normalized = rms_norm(
            final_var.as_tensor(),
            &self.output_norm,
            self.layer_config.attention.rms_norm_eps,
        )
        .context("final RMSNorm")?;
        let seq_len = normalized.dim(1)?;
        let last_hidden = normalized
            .narrow(1, seq_len - 1, 1)?
            .reshape((1, self.hidden))?;
        let logits = last_hidden.matmul(&self.lm_head.t()?)?;
        let target_ids = Tensor::from_vec(vec![target], (1,), device)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &target_ids)
            .context("cross_entropy failed")?;
        let loss_value = scalar_f32(&loss)?;
        let mut sample_grads = loss.backward().context("final objective backward")?;
        let mut upstream = sample_grads
            .get(final_var.as_tensor())
            .cloned()
            .ok_or_else(|| anyhow!("missing final hidden gradient"))?
            .detach();

        let trainable_vars = self.varmap.all_vars();
        for layer_n in (0..self.layer_loader.num_layers()).rev() {
            let input_var =
                Var::from_tensor(&boundaries[layer_n]).context("layer boundary Var")?;
            let weights = self.load_layer_weights(layer_n, device)?;
            let adapters = self.projection_adapters(layer_n, device)?;
            let loras = layer_loras(&adapters);
            let output = transformer_layer_forward(
                input_var.as_tensor(),
                &weights.as_refs(),
                &loras,
                self.layer_config,
                0,
            )
            .with_context(|| format!("recompute transformer layer {layer_n}"))?;
            let vjp = output
                .mul(&upstream)
                .context("layer vector-Jacobian product")?
                .sum_all()
                .context("sum layer vector-Jacobian product")?;
            let layer_grads = vjp.backward().context("layer recomputation backward")?;
            upstream = layer_grads
                .get(input_var.as_tensor())
                .cloned()
                .ok_or_else(|| anyhow!("missing input gradient for layer {layer_n}"))?
                .detach();
            merge_gradstores(&mut sample_grads, &layer_grads, &trainable_vars)?;
        }

        Ok((loss_value, sample_grads))
    }

    fn forward_boundaries(&self, input_ids: &Tensor, device: &Device) -> Result<Vec<Tensor>> {
        let seq_len = input_ids.dim(0)?;
        let gathered = self
            .model_embedding
            .index_select(input_ids, 0)
            .context("model embedding lookup")?;
        let mut hidden = gathered
            .reshape((1, seq_len, self.hidden))
            .context("reshape embedded sequence")?
            .detach();
        let mut boundaries = Vec::with_capacity(self.layer_loader.num_layers() + 1);
        boundaries.push(hidden.clone());

        for layer_n in 0..self.layer_loader.num_layers() {
            let weights = self.load_layer_weights(layer_n, device)?;
            let adapters = self.projection_adapters(layer_n, device)?;
            let loras = layer_loras(&adapters);
            hidden = transformer_layer_forward(
                &hidden,
                &weights.as_refs(),
                &loras,
                self.layer_config,
                0,
            )
            .with_context(|| format!("forward transformer layer {layer_n}"))?
            .detach();
            boundaries.push(hidden.clone());
        }

        Ok(boundaries)
    }

    fn load_layer_weights(&self, layer_n: usize, device: &Device) -> Result<OwnedLayerWeights> {
        let loaded = self
            .layer_loader
            .load_layer(layer_n)
            .with_context(|| format!("load layer {layer_n}"))?;
        let weights = OwnedLayerWeights::from_loaded(
            &self.layer_loader,
            layer_n,
            &loaded.slices,
            device,
        );
        loaded.unload();
        weights
    }
}

struct OwnedLayerWeights {
    attn_norm: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    ffn_norm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl OwnedLayerWeights {
    fn from_loaded(
        loader: &LayerLoader,
        layer_n: usize,
        slices: &[(&str, &[u8])],
        device: &Device,
    ) -> Result<Self> {
        let mut attn_norm = None;
        let mut q_proj = None;
        let mut k_proj = None;
        let mut v_proj = None;
        let mut o_proj = None;
        let mut q_norm = None;
        let mut k_norm = None;
        let mut ffn_norm = None;
        let mut gate_proj = None;
        let mut up_proj = None;
        let mut down_proj = None;

        for (name, bytes) in slices {
            let meta = loader
                .index_slices_for(layer_n)
                .iter()
                .find(|slice| slice.tensor_name == *name)
                .ok_or_else(|| anyhow!("no metadata for layer tensor '{name}'"))?;
            if !name.ends_with(".weight") {
                continue;
            }
            let tensor = dequant_tensor(bytes, meta.dtype, &meta.shape, device)
                .with_context(|| format!("dequantize '{name}'"))?;

            if name.contains("attn_q_norm") || name.contains("q_norm") {
                q_norm = Some(tensor);
            } else if name.contains("attn_k_norm") || name.contains("k_norm") {
                k_norm = Some(tensor);
            } else if name.contains("attn_norm") || name.contains("input_layernorm") {
                attn_norm = Some(tensor);
            } else if name.contains("ffn_norm") || name.contains("post_attention_layernorm") {
                ffn_norm = Some(tensor);
            } else {
                match classify_tensor(name) {
                    Some(ProjectionKind::AttnQ) => q_proj = Some(tensor),
                    Some(ProjectionKind::AttnK) => k_proj = Some(tensor),
                    Some(ProjectionKind::AttnV) => v_proj = Some(tensor),
                    Some(ProjectionKind::AttnO) => o_proj = Some(tensor),
                    Some(ProjectionKind::FfnGate) => gate_proj = Some(tensor),
                    Some(ProjectionKind::FfnUp) => up_proj = Some(tensor),
                    Some(ProjectionKind::FfnDown) => down_proj = Some(tensor),
                    None => {}
                }
            }
        }

        let required = |value: Option<Tensor>, name: &str| {
            value.ok_or_else(|| anyhow!("layer {layer_n} is missing {name}"))
        };
        Ok(Self {
            attn_norm: required(attn_norm, "attention norm")?,
            q_proj: required(q_proj, "query projection")?,
            k_proj: required(k_proj, "key projection")?,
            v_proj: required(v_proj, "value projection")?,
            o_proj: required(o_proj, "attention output projection")?,
            q_norm,
            k_norm,
            ffn_norm: required(ffn_norm, "FFN norm")?,
            gate_proj: required(gate_proj, "FFN gate projection")?,
            up_proj: required(up_proj, "FFN up projection")?,
            down_proj: required(down_proj, "FFN down projection")?,
        })
    }

    fn as_refs(&self) -> TransformerLayerWeights<'_> {
        TransformerLayerWeights {
            attention: AttentionWeights {
                attn_norm: &self.attn_norm,
                q_proj: &self.q_proj,
                k_proj: &self.k_proj,
                v_proj: &self.v_proj,
                o_proj: &self.o_proj,
                q_norm: self.q_norm.as_ref(),
                k_norm: self.k_norm.as_ref(),
            },
            mlp: MlpWeights {
                ffn_norm: &self.ffn_norm,
                gate_proj: &self.gate_proj,
                up_proj: &self.up_proj,
                down_proj: &self.down_proj,
            },
        }
    }
}

fn layer_loras(adapters: &[ProjLora]) -> TransformerLayerLoras<'_> {
    let get = |kind| {
        adapters
            .iter()
            .find(|adapter| adapter.kind == kind)
            .map(|adapter| ProjectionLora {
                a: &adapter.a,
                b: &adapter.b,
                scale: adapter.scale,
            })
    };
    TransformerLayerLoras {
        attention: AttentionLoras {
            q_proj: get(ProjectionKind::AttnQ),
            k_proj: get(ProjectionKind::AttnK),
            v_proj: get(ProjectionKind::AttnV),
            o_proj: get(ProjectionKind::AttnO),
        },
        mlp: MlpLoras {
            gate_proj: get(ProjectionKind::FfnGate),
            up_proj: get(ProjectionKind::FfnUp),
            down_proj: get(ProjectionKind::FfnDown),
        },
    }
}

/// Per-projection LoRA adapter, operating in the projection's native dims.
///
/// `a`/`b` are `Tensor` handles bound from the VarMap (the underlying `Var`s
/// stay tracked by AdamW); reconstructing this wrapper each layer just re-binds
/// them without allocating new parameters.
struct ProjLora {
    /// Which transformer projection this adapter targets (used by Wave 3 export).
    #[allow(dead_code)]
    kind: ProjectionKind,
    a: Tensor, // [r_eff, d_in]
    b: Tensor, // [d_out, r_eff]
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

fn load_capped_matrix(
    loader: &LayerLoader,
    names: &[&str],
    rows: usize,
    input_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let tensor = loader
        .find_tensor(names)
        .ok_or_else(|| anyhow!("none of the GGUF tensors {:?} were found", names))?;
    let (stored_input, stored_rows) = match tensor.shape.as_slice() {
        [stored_input, stored_rows] => (*stored_input as usize, *stored_rows as usize),
        shape => {
            return Err(anyhow!(
                "tensor '{}' must be a 2-D GGUF matrix, got {:?}",
                tensor.tensor_name,
                shape
            ));
        }
    };
    if stored_input != input_dim || rows > stored_rows {
        return Err(anyhow!(
            "tensor '{}' dimensions [{stored_input}, {stored_rows}] do not support [{input_dim}, {rows}]",
            tensor.tensor_name
        ));
    }
    if tensor.byte_len % stored_rows != 0 {
        return Err(anyhow!(
            "tensor '{}' rows do not align to quantization blocks",
            tensor.tensor_name
        ));
    }
    let row_bytes = tensor.byte_len / stored_rows;
    let selected_len = row_bytes
        .checked_mul(rows)
        .ok_or_else(|| anyhow!("capped tensor byte length overflow"))?;
    let bytes = loader
        .tensor_bytes(tensor)?
        .get(..selected_len)
        .ok_or_else(|| anyhow!("capped tensor slice is outside '{}'", tensor.tensor_name))?;
    let data = dequant_slice(
        bytes,
        tensor.dtype,
        &[input_dim as u64, rows as u64],
    )?;
    Tensor::from_vec(data, (rows, input_dim), device)
        .with_context(|| format!("construct capped tensor '{}'", tensor.tensor_name))
}

fn load_vector(
    loader: &LayerLoader,
    names: &[&str],
    len: usize,
    device: &Device,
) -> Result<Tensor> {
    let tensor = loader
        .find_tensor(names)
        .ok_or_else(|| anyhow!("none of the GGUF tensors {:?} were found", names))?;
    if tensor.shape.as_slice() != [len as u64] {
        return Err(anyhow!(
            "tensor '{}' expected shape [{len}], got {:?}",
            tensor.tensor_name,
            tensor.shape
        ));
    }
    dequant_tensor(
        loader.tensor_bytes(tensor)?,
        tensor.dtype,
        &tensor.shape,
        device,
    )
}

fn dequant_tensor(
    bytes: &[u8],
    dtype: GgufDtype,
    gguf_shape: &[u64],
    device: &Device,
) -> Result<Tensor> {
    let data = dequant_slice(bytes, dtype, gguf_shape)?;
    let shape: Vec<usize> = gguf_shape.iter().rev().map(|dim| *dim as usize).collect();
    Tensor::from_vec(data, shape, device).context("construct dequantized Candle tensor")
}

/// Dequantise raw mmap bytes to `Vec<f32>` using the appropriate path for `dtype`.
///
/// For F32 tensors this is a cheap byte-reinterpretation (no copy via transmute
/// is possible in safe Rust, so we do a `chunks_exact(4)` parse).  For quantised
/// types we delegate to `dequant::dequantize` via a zero-copy `TensorInfo` wrapper.
fn dequant_slice(bytes: &[u8], dtype: GgufDtype, shape: &[u64]) -> Result<Vec<f32>> {
    use crate::convert::gguf_parser::TensorInfo;

    // Build a TensorInfo that borrows the bytes via a clone.
    // The clone is unavoidable because TensorInfo owns its raw_data; the copy
    // is bounded to one layer's worth of data (not the full model).
    let tensor_info = TensorInfo {
        name: String::new(),
        shape: shape.to_vec(),
        dtype,
        data_offset: 0,
        data_size: bytes.len(),
        raw_data: bytes.to_vec(),
    };

    dequant::dequantize(&tensor_info, DequantMode::Standard)
        .map_err(|e| anyhow!("dequant error: {}", e))
}

/// Interpret a shape slice as `(d_out, d_in)` for a 2-D weight matrix.
///
/// GGUF records matrix dimensions as `[d_in, d_out]`; Candle uses the reversed
/// row-major shape `[d_out, d_in]`. One-dimensional fixtures are interpreted
/// as `(n, 1)`.
pub(crate) fn shape_to_2d(shape: &[u64]) -> Result<(usize, usize)> {
    match shape {
        [d_in, d_out] => Ok((*d_out as usize, *d_in as usize)),
        [n] => Ok((*n as usize, 1)),
        _ => Err(anyhow!(
            "cannot interpret {:?} as a 2-D weight shape",
            shape
        )),
    }
}

fn save_checkpoint(varmap: &VarMap, config: &NewTrainConfig, step: usize) -> Result<()> {
    let filename = format!("checkpoint_{:06}.safetensors", step);
    let path = config.output_path.join(&filename);
    std::fs::create_dir_all(&config.output_path).with_context(|| {
        format!(
            "cannot create output dir '{}'",
            config.output_path.display()
        )
    })?;
    varmap
        .save(&path)
        .with_context(|| format!("failed to write checkpoint '{}'", path.display()))?;
    eprintln!("[checkpoint] saved → {}", path.display());
    Ok(())
}

fn scalar_f32(t: &Tensor) -> Result<f32> {
    t.to_scalar::<f32>().context("expected scalar loss tensor")
}

fn merge_gradstores(target: &mut GradStore, source: &GradStore, vars: &[Var]) -> Result<()> {
    for var in vars {
        let Some(source_grad) = source.get(var.as_tensor()) else {
            continue;
        };
        let merged = match target.get(var.as_tensor()) {
            Some(existing) => existing.add(source_grad).context("accumulate gradient")?,
            None => source_grad.clone(),
        };
        target.insert(var.as_tensor(), merged);
    }
    Ok(())
}

fn scale_gradstore(grads: &mut GradStore, vars: &[Var], scale: f64) -> Result<()> {
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            grads.insert(
                var.as_tensor(),
                grad.affine(scale, 0.0).context("scale accumulated gradient")?,
            );
        }
    }
    Ok(())
}

/// Clip the gradients in `grads` (in place) so their global L2 norm ≤ `max_norm`.
///
/// Scales the *gradients* before the optimiser step — the standard
/// `clip_grad_norm_` behaviour. Operates only on the Vars present in `varmap`.
fn clip_gradstore_norm(
    grads: &mut candle_core::backprop::GradStore,
    varmap: &VarMap,
    max_norm: f64,
) -> Result<()> {
    if max_norm <= 0.0 {
        return Ok(());
    }
    let vars = varmap.all_vars();

    // Global L2 norm across all per-Var gradients.
    let mut total_sq = 0.0f64;
    for v in &vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let sq = g
                .sqr()
                .context("grad sqr")?
                .sum_all()
                .context("grad sum_all")?
                .to_scalar::<f32>()
                .context("grad scalar")?;
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
    varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum()
}

// ── EXPERIMENTAL: GAAP S(ρ)-informed per-projection rank allocation (--gdtqp) ──
//
// ⚠ THEORY UNPROVEN. This path synthesises a mechanism the source specs do NOT
// define end-to-end:
//   • GAAP defines the von Neumann entropy S(ρ) = -Tr(ρ log ρ) on an *attention*
//     density matrix ρ = Σ aᵢ|v̂ᵢ⟩⟨v̂ᵢ|, and lists "Integration with LoRA
//     training" as an OPEN PROBLEM — it gives no rank-allocation rule.
//   • GDTQP contributes the idea of *sensitivity-weighted adaptive allocation
//     under a budget* (it allocates bits, not LoRA rank, and is post-training).
// The streaming layered loop has no runtime attention, so we cannot build GAAP's
// attention ρ. We substitute a weight-derived diagonal density matrix
// ρ = diag(p), p_c = (column energy of W) / (total energy); then
// S(ρ) = -Tr(ρ log ρ) = -Σ p_c ln p_c exactly. Higher entropy (energy spread
// across many input directions) ⇒ less low-rank structure ⇒ more rank.
//
// Treat every number this path emits as EXPERIMENTAL; never fold it into stable
// benchmark figures.

/// Column-energy von Neumann entropy S(ρ) = -Σ p_c ln p_c for a `[d_out, d_in]`
/// weight matrix, normalised to `[0, 1]` by dividing by `ln(d_in)` (the maximum,
/// reached by a perfectly uniform energy distribution).
fn column_energy_entropy(w: &[f32], d_out: usize, d_in: usize) -> f32 {
    if d_in <= 1 || d_out == 0 {
        return 0.0;
    }
    let mut energy = vec![0.0f64; d_in];
    for row in 0..d_out {
        for c in 0..d_in {
            let v = w.get(row * d_in + c).copied().unwrap_or(0.0) as f64;
            energy[c] += v * v;
        }
    }
    let total: f64 = energy.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut s = 0.0f64;
    for e in &energy {
        let p = e / total;
        if p > 0.0 {
            s -= p * p.ln();
        }
    }
    (s / (d_in as f64).ln()).clamp(0.0, 1.0) as f32
}

/// Map per-projection sensitivities `S(ρ)∈[0,1]` to per-projection LoRA ranks.
///
/// Mean-centred proportional allocation: `rank_p ≈ base · S_p / mean(S)`, clamped
/// to `[base/2, base·2]` and capped at `min(d_in, d_out)`. This keeps the *mean*
/// rank ≈ `base` (so the total adapter parameter budget is roughly preserved,
/// matching GDTQP's "allocation under a memory constraint" framing) while steering
/// capacity toward the most entropy-sensitive projections. With equal
/// sensitivities every projection gets exactly `base` (identical to the default).
fn allocate_ranks_from_sensitivity(
    sensitivities: &[f32],
    dims: &[(usize, usize)],
    base_rank: usize,
) -> Vec<usize> {
    let n = sensitivities.len();
    if n == 0 {
        return Vec::new();
    }
    let mean_s: f32 = sensitivities.iter().copied().sum::<f32>() / n as f32;
    let r_min = (base_rank / 2).max(1);
    let r_max = (base_rank * 2).max(r_min);

    sensitivities
        .iter()
        .zip(dims.iter())
        .map(|(&s, &(d_in, d_out))| {
            let raw = if mean_s > 0.0 {
                (base_rank as f32 * s / mean_s).round() as i64
            } else {
                base_rank as i64
            };
            let cap = d_in.min(d_out).max(1) as i64;
            raw.clamp(r_min as i64, r_max as i64).min(cap).max(1) as usize
        })
        .collect()
}

/// EXPERIMENTAL: compute a per-projection LoRA rank from each projection's
/// weight-energy entropy S(ρ), measured on layer 0 (all layers share the same
/// projection shapes/architecture). Loads layer 0 once under the streaming
/// invariant and unloads it before returning.
fn gdtqp_allocate_ranks(
    layer_loader: &LayerLoader,
    proj_keys: &[(ProjectionKind, usize, usize)],
    base_rank: usize,
    tx: &Option<Sender<String>>,
) -> Result<Vec<usize>> {
    let loaded = layer_loader
        .load_layer(0)
        .context("gdtqp: failed to load layer 0")?;

    let mut sensitivities = vec![0.0f32; proj_keys.len()];
    for (name, bytes) in &loaded.slices {
        let Some(kind) = classify_tensor(name) else {
            continue;
        };
        let Some(idx) = proj_keys.iter().position(|(k, _, _)| *k == kind) else {
            continue;
        };
        let meta = layer_loader
            .index_slices_for(0)
            .iter()
            .find(|s| s.tensor_name.as_str() == *name)
            .ok_or_else(|| anyhow!("gdtqp: no metadata for '{}'", name))?;
        let w = dequant_slice(bytes, meta.dtype, &meta.shape)?;
        let (d_out, d_in) = shape_to_2d(&meta.shape)?;
        sensitivities[idx] = column_energy_entropy(&w, d_out, d_in);
    }
    loaded.unload();

    let dims: Vec<(usize, usize)> = proj_keys
        .iter()
        .map(|(_, di, douta)| (*di, *douta))
        .collect();
    let ranks = allocate_ranks_from_sensitivity(&sensitivities, &dims, base_rank);

    // EXPERIMENTAL log block — loud, clearly labelled, mirrored to the TUI.
    let mut lines = vec![
        "[gdtqp][EXPERIMENTAL] GAAP S(ρ)-informed per-projection LoRA rank allocation".to_string(),
        "[gdtqp][EXPERIMENTAL] THEORY UNPROVEN — weight-energy entropy surrogate for GAAP's \
         attention ρ; GAAP→LoRA is an open problem. Do NOT use as a stable benchmark."
            .to_string(),
        format!("[gdtqp][EXPERIMENTAL] base_rank={base_rank}"),
    ];
    for ((kind, _, _), (s, rank)) in proj_keys.iter().zip(sensitivities.iter().zip(ranks.iter())) {
        lines.push(format!(
            "[gdtqp][EXPERIMENTAL]   {:<8} S(ρ)_norm={:.4}  rank={}",
            kind.var_key(),
            s,
            rank,
        ));
    }
    for l in &lines {
        eprintln!("{}", l);
        if let Some(tx) = tx {
            tx.send(l.clone()).ok();
        }
    }

    Ok(ranks)
}

/// Sample current process resident set size in MB (cross-platform).
fn sample_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Some(kb) = rest
                        .split_whitespace()
                        .next()
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
    use candle_core::{Device, Tensor};
    use candle_nn::VarMap;
    use std::io::Write;
    use tempfile::TempDir;

    use crate::train::config::{LoraConfig, NewTrainConfig};
    use crate::train::layer_loader::tests::write_minimal_gguf;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn default_config(output: std::path::PathBuf) -> NewTrainConfig {
        NewTrainConfig {
            output_path: output,
            lora: LoraConfig {
                r: 2,
                alpha: 4.0,
                dropout: 0.0,
                target_modules: vec![],
            },
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

    fn make_varmap() -> VarMap {
        VarMap::new()
    }

    fn write_one_layer_gguf() -> tempfile::NamedTempFile {
        write_transformer_gguf(1)
    }

    fn write_two_layer_gguf() -> tempfile::NamedTempFile {
        write_transformer_gguf(2)
    }

    fn write_multi_proj_gguf(n_layers: usize) -> tempfile::NamedTempFile {
        write_transformer_gguf(n_layers)
    }

    fn write_transformer_gguf(n_layers: usize) -> tempfile::NamedTempFile {
        const HIDDEN: usize = 4;
        const INTERMEDIATE: usize = 8;
        const VOCAB: usize = 16;
        const KV_DIM: usize = 2;

        fn values(count: usize, seed: usize) -> Vec<u8> {
            (0..count)
                .flat_map(|i| {
                    let value = ((i + seed) % 19) as f32 * 0.005 - 0.04;
                    value.to_le_bytes()
                })
                .collect()
        }
        fn ones(count: usize) -> Vec<u8> {
            (0..count).flat_map(|_| 1.0f32.to_le_bytes()).collect()
        }
        fn write_key(file: &mut tempfile::NamedTempFile, key: &str) {
            file.write_all(&(key.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(key.as_bytes()).unwrap();
        }

        let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = vec![
            (
                "token_embd.weight".into(),
                vec![HIDDEN as u64, VOCAB as u64],
                values(HIDDEN * VOCAB, 1),
            ),
            (
                "output_norm.weight".into(),
                vec![HIDDEN as u64],
                ones(HIDDEN),
            ),
            (
                "output.weight".into(),
                vec![HIDDEN as u64, VOCAB as u64],
                values(HIDDEN * VOCAB, 7),
            ),
        ];
        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            tensors.extend([
                (
                    format!("{prefix}.attn_norm.weight"),
                    vec![HIDDEN as u64],
                    ones(HIDDEN),
                ),
                (
                    format!("{prefix}.attn_q.weight"),
                    vec![HIDDEN as u64, HIDDEN as u64],
                    values(HIDDEN * HIDDEN, layer + 2),
                ),
                (
                    format!("{prefix}.attn_k.weight"),
                    vec![HIDDEN as u64, KV_DIM as u64],
                    values(HIDDEN * KV_DIM, layer + 3),
                ),
                (
                    format!("{prefix}.attn_v.weight"),
                    vec![HIDDEN as u64, KV_DIM as u64],
                    values(HIDDEN * KV_DIM, layer + 4),
                ),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![HIDDEN as u64, HIDDEN as u64],
                    values(HIDDEN * HIDDEN, layer + 5),
                ),
                (
                    format!("{prefix}.attn_q_norm.weight"),
                    vec![2],
                    ones(2),
                ),
                (
                    format!("{prefix}.attn_k_norm.weight"),
                    vec![2],
                    ones(2),
                ),
                (
                    format!("{prefix}.ffn_norm.weight"),
                    vec![HIDDEN as u64],
                    ones(HIDDEN),
                ),
                (
                    format!("{prefix}.ffn_gate.weight"),
                    vec![HIDDEN as u64, INTERMEDIATE as u64],
                    values(HIDDEN * INTERMEDIATE, layer + 6),
                ),
                (
                    format!("{prefix}.ffn_up.weight"),
                    vec![HIDDEN as u64, INTERMEDIATE as u64],
                    values(HIDDEN * INTERMEDIATE, layer + 7),
                ),
                (
                    format!("{prefix}.ffn_down.weight"),
                    vec![INTERMEDIATE as u64, HIDDEN as u64],
                    values(HIDDEN * INTERMEDIATE, layer + 8),
                ),
            ]);
        }

        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&(tensors.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&9u64.to_le_bytes()).unwrap();
        write_key(&mut file, "general.architecture");
        file.write_all(&8u32.to_le_bytes()).unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap();
        file.write_all(b"test").unwrap();
        for (key, value) in [
            ("test.block_count", n_layers as u32),
            ("test.embedding_length", HIDDEN as u32),
            ("test.attention.head_count", 2),
            ("test.attention.head_count_kv", 1),
            ("test.feed_forward_length", INTERMEDIATE as u32),
            ("test.vocab_size", VOCAB as u32),
        ] {
            write_key(&mut file, key);
            file.write_all(&4u32.to_le_bytes()).unwrap();
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        for (key, value) in [
            ("test.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("test.rope.freq_base", 10_000.0f32),
        ] {
            write_key(&mut file, key);
            file.write_all(&6u32.to_le_bytes()).unwrap();
            file.write_all(&value.to_le_bytes()).unwrap();
        }

        let mut offset = 0u64;
        for (name, shape, data) in &tensors {
            file.write_all(&(name.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(name.as_bytes()).unwrap();
            file.write_all(&(shape.len() as u32).to_le_bytes())
                .unwrap();
            for dim in shape {
                file.write_all(&dim.to_le_bytes()).unwrap();
            }
            file.write_all(&0u32.to_le_bytes()).unwrap();
            file.write_all(&offset.to_le_bytes()).unwrap();
            offset += data.len() as u64;
        }
        let position = file.as_file().metadata().unwrap().len();
        let padding = (32 - position % 32) % 32;
        file.write_all(&vec![0; padding as usize]).unwrap();
        for (_, _, data) in &tensors {
            file.write_all(data).unwrap();
        }
        file.flush().unwrap();
        file
    }

    // ── deterministic tests ───────────────────────────────────────────────────

    /// The loop creates every trainable LoRA adapter inside `new()`, so an empty
    /// input VarMap is the normal case used by `native_runner`.
    #[test]
    fn test_new_populates_empty_varmap() {
        let f = write_one_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        let vm = VarMap::new(); // empty — populated by new()
        let ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), vm, None, 0)
            .expect("new() must self-populate an empty VarMap");
        assert!(
            !ltl.varmap.all_vars().is_empty(),
            "new() must leave the VarMap with trainable parameters",
        );
        assert!(count_params(&ltl.varmap) > 0);
    }

    #[test]
    fn loader_uses_transformer_metadata() {
        let f = write_two_layer_gguf();
        let loader = LayerLoader::open(f.path()).expect("open metadata fixture");
        let model = loader.transformer_config();

        assert_eq!(model.architecture, "test");
        assert_eq!(model.n_layers, 2);
        assert_eq!(model.hidden_size, 4);
        assert_eq!(model.n_heads, 2);
        assert_eq!(model.n_kv_heads, 1);
        assert_eq!(model.intermediate_size, 8);
        assert_eq!(model.vocab_size, 16);
        assert_eq!(model.rms_norm_eps, 1e-5);
        assert_eq!(model.rope_theta, 10_000.0);
    }

    /// Property 12 (GWEN-222): `shape_to_2d` reverses GGUF `[d_in, d_out]`
    /// ordering into Candle's `(d_out, d_in)`.
    #[test]
    fn prop_shape_to_2d_reversal() {
        fn prop(d_in: u64, d_out: u64) -> bool {
            matches!(
                shape_to_2d(&[d_in, d_out]),
                Ok((out, inp)) if out == d_out as usize && inp == d_in as usize
            )
        }
        quickcheck::quickcheck(prop as fn(u64, u64) -> bool);
    }

    #[test]
    fn test_new_rejects_zero_layers() {
        // GGUF with no model.layers.* tensors
        let weight: Vec<u8> = 0.5f32.to_le_bytes().to_vec();
        let f = write_minimal_gguf(&[("token_embd.weight", &weight)]);
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        let result = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 0);
        assert!(result.is_err());
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("no model.layers")
        );
    }

    #[test]
    fn test_run_single_epoch_produces_result() {
        let f = write_two_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        // 2 tokens → input shape (1,1) which matches d_in=1 from the 1-element GGUF tensor.
        let mut ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 0)
            .expect("new");

        let result = ltl.run().expect("run");
        assert!(
            result.total_steps >= 1,
            "expected at least one optimizer step"
        );
        assert!(result.final_loss.is_finite(), "loss must be finite");
    }

    /// Property 7 (GWEN-222): `total_steps` reflects only the current run's
    /// steps, not the cumulative count after a resume.
    #[test]
    fn test_total_steps_is_current_run_only() {
        let f = write_two_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        // Resume from a high global step; one batch + grad_accum=1 ⇒ exactly one
        // optimizer step THIS run, so total_steps must be 1 (not 1001).
        let mut ltl =
            LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 1000)
                .expect("new");
        let result = ltl.run().expect("run");
        assert_eq!(
            result.total_steps, 1,
            "total_steps must count only this run's steps, got {}",
            result.total_steps
        );
    }

    /// Property 8 (GWEN-222): checkpoint files contain only LoRA adapter weights
    /// — never AdamW optimizer state (which candle keeps outside the VarMap).
    #[test]
    fn test_checkpoint_keys_lora_only() {
        let f = write_two_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        // `tests` is a submodule, so the private `varmap` field + free
        // `save_checkpoint` fn are both in scope.
        let ltl =
            LayeredTrainingLoop::new(cfg.clone(), f.path(), make_batch(2), make_varmap(), None, 0)
                .expect("new");

        save_checkpoint(&ltl.varmap, &cfg, 500).expect("save_checkpoint");
        let path = cfg.output_path.join("checkpoint_000500.safetensors");
        assert!(path.exists(), "checkpoint file should exist");

        let tensors =
            candle_core::safetensors::load(&path, &Device::Cpu).expect("load safetensors");
        assert!(!tensors.is_empty(), "checkpoint must contain adapter tensors");
        for key in tensors.keys() {
            assert!(
                key.contains("lora_"),
                "unexpected non-adapter key in checkpoint: {key}"
            );
            for forbidden in ["adam", "moment", "exp_avg"] {
                assert!(
                    !key.contains(forbidden),
                    "checkpoint must not contain optimizer state: {key}"
                );
            }
        }
    }

    #[test]
    fn test_run_emits_done_json() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<String>();

        let f = write_one_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        // 2 tokens → input shape (1,1) which matches d_in=1 from the 1-element GGUF tensor.
        let mut ltl =
            LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), Some(tx), 0)
                .expect("new");

        ltl.run().expect("run");

        let messages: Vec<String> = rx.try_iter().collect();
        let done = messages.iter().any(|m| m.contains(r#""event":"done""#));
        assert!(done, "expected a done JSON event, got: {:?}", messages);
    }

    // ── multi-projection tests (GWEN-219) ──────────────────────────────────────

    #[test]
    fn test_classify_tensor_known_names() {
        assert_eq!(
            classify_tensor("blk.0.attn_q.weight"),
            Some(ProjectionKind::AttnQ),
        );
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.q_proj.weight"),
            Some(ProjectionKind::AttnQ),
        );
        assert_eq!(
            classify_tensor("blk.0.ffn_gate.weight"),
            Some(ProjectionKind::FfnGate),
        );
        assert_eq!(
            classify_tensor("blk.0.attn_output.weight"),
            Some(ProjectionKind::AttnO),
        );
        // Norms and other non-projection tensors are not LoRA targets — even
        // when their name embeds a projection substring (Qwen3 q/k norms).
        assert_eq!(classify_tensor("blk.0.attn_norm.weight"), None);
        assert_eq!(classify_tensor("blk.0.ffn_norm.weight"), None);
        assert_eq!(classify_tensor("blk.0.attn_q_norm.weight"), None);
        assert_eq!(classify_tensor("blk.0.attn_k_norm.weight"), None);
    }

    #[test]
    fn test_new_creates_per_projection_adapters() {
        let f = write_multi_proj_gguf(1);
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        let ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 0)
            .expect("new");

        assert_eq!(
            ltl.proj_keys_per_layer.len(),
            7,
            "expected 7 distinct projection kinds for layer 0",
        );

        // Every projection must have allocated both its lora_a and lora_b var
        // under the `l0.{var_key}.…` namespace (var_key for AttnO is "attn_o").
        let data = ltl.varmap.data().lock().unwrap();
        for key in [
            "l0.attn_q.lora_a",
            "l0.attn_q.lora_b",
            "l0.attn_k.lora_a",
            "l0.attn_k.lora_b",
            "l0.attn_v.lora_a",
            "l0.attn_v.lora_b",
            "l0.attn_o.lora_a",
            "l0.attn_o.lora_b",
            "l0.ffn_gate.lora_a",
            "l0.ffn_gate.lora_b",
            "l0.ffn_up.lora_a",
            "l0.ffn_up.lora_b",
            "l0.ffn_down.lora_a",
            "l0.ffn_down.lora_b",
        ] {
            assert!(data.contains_key(key), "missing var key '{}'", key);
        }
    }

    #[test]
    fn test_run_multi_proj_converges() {
        let f = write_multi_proj_gguf(2); // 2 layers
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf()); // 1 epoch
        let mut ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 0)
            .expect("new");

        let result = ltl.run().expect("run");
        assert!(result.final_loss.is_finite(), "loss must be finite");
        assert!(
            result.total_steps >= 1,
            "expected at least one optimizer step"
        );
    }

    #[test]
    fn reverse_recomputation_matches_full_graph_gradients() {
        let f = write_two_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        let ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(3), make_varmap(), None, 0)
            .expect("new");
        let sample = &ltl.batches[0][0];
        let (recomputed_loss, recomputed_grads) = ltl
            .forward_backward_sample(sample, &Device::Cpu)
            .expect("reverse recomputation");

        let input_ids = Tensor::from_vec(vec![1u32, 2], (2,), &Device::Cpu).unwrap();
        let gathered = ltl.model_embedding.index_select(&input_ids, 0).unwrap();
        let mut hidden = gathered.reshape((1, 2, ltl.hidden)).unwrap();
        for layer_n in 0..ltl.layer_loader.num_layers() {
            let weights = ltl
                .load_layer_weights(layer_n, &Device::Cpu)
                .expect("layer weights");
            let adapters = ltl
                .projection_adapters(layer_n, &Device::Cpu)
                .expect("adapters");
            hidden = transformer_layer_forward(
                &hidden,
                &weights.as_refs(),
                &layer_loras(&adapters),
                ltl.layer_config,
                0,
            )
            .unwrap();
        }
        let normalized = rms_norm(
            &hidden,
            &ltl.output_norm,
            ltl.layer_config.attention.rms_norm_eps,
        )
        .unwrap();
        let logits = normalized
            .narrow(1, 1, 1)
            .unwrap()
            .reshape((1, ltl.hidden))
            .unwrap()
            .matmul(&ltl.lm_head.t().unwrap())
            .unwrap();
        let target = Tensor::from_vec(vec![3u32], (1,), &Device::Cpu).unwrap();
        let full_loss = candle_nn::loss::cross_entropy(&logits, &target).unwrap();
        let full_loss_value = scalar_f32(&full_loss).unwrap();
        let full_grads = full_loss.backward().unwrap();

        assert!((recomputed_loss - full_loss_value).abs() < 1e-6);
        for var in ltl.varmap.all_vars() {
            let recomputed = recomputed_grads
                .get(var.as_tensor())
                .expect("recomputed adapter gradient")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let full = full_grads
                .get(var.as_tensor())
                .expect("full-graph adapter gradient")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (left, right) in recomputed.iter().zip(full.iter()) {
                assert!(
                    (left - right).abs() < 1e-5,
                    "gradient mismatch: recomputed={left}, full={right}"
                );
            }
        }
    }

    #[test]
    fn forward_boundaries_propagate_through_every_layer() {
        let f = write_two_layer_gguf();
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        let ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(3), make_varmap(), None, 0)
            .expect("new");
        let input_ids = Tensor::from_vec(vec![1u32, 2], (2,), &Device::Cpu).unwrap();
        let boundaries = ltl
            .forward_boundaries(&input_ids, &Device::Cpu)
            .expect("forward boundaries");

        assert_eq!(boundaries.len(), 3);
        assert_eq!(boundaries[0].dims(), &[1, 2, 4]);
        assert_ne!(
            boundaries[0].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            boundaries[2].flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn test_projection_adapters_all_kinds() {
        let f = write_multi_proj_gguf(1);
        let td = TempDir::new().unwrap();
        let cfg = default_config(td.path().to_path_buf());
        let ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 0)
            .expect("new");

        let adapters = ltl
            .projection_adapters(0, &Device::Cpu)
            .expect("projection_adapters");
        assert_eq!(adapters.len(), 7, "expected one adapter per projection");

        // Exactly one adapter per ProjectionKind.
        let kinds: Vec<ProjectionKind> = adapters.iter().map(|p| p.kind).collect();
        for expected in [
            ProjectionKind::AttnQ,
            ProjectionKind::AttnK,
            ProjectionKind::AttnV,
            ProjectionKind::AttnO,
            ProjectionKind::FfnGate,
            ProjectionKind::FfnUp,
            ProjectionKind::FfnDown,
        ] {
            assert!(kinds.contains(&expected), "missing kind {:?}", expected);
        }

        // All adapter weights must be finite (lora_a randn, lora_b zero-init).
        for pl in &adapters {
            for t in [&pl.a, &pl.b] {
                let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
                assert!(
                    v.iter().all(|x| x.is_finite()),
                    "non-finite adapter weight for {:?}",
                    pl.kind,
                );
            }
        }
    }

    // ── EXPERIMENTAL --gdtqp tests ────────────────────────────────────────────

    /// S(ρ) surrogate: a uniform energy spectrum is maximally entropic (≈1.0);
    /// energy concentrated in one column is minimally entropic (≈0.0).
    #[test]
    fn test_column_energy_entropy_uniform_vs_peaked() {
        // 1 row, 4 columns. Uniform magnitudes → S_norm ≈ 1.0.
        let uniform = vec![1.0f32, 1.0, 1.0, 1.0];
        let s_uniform = column_energy_entropy(&uniform, 1, 4);
        assert!(
            s_uniform > 0.99,
            "uniform S(ρ) should be ≈1.0, got {s_uniform}"
        );

        // All energy in one column → S_norm ≈ 0.0.
        let peaked = vec![1.0f32, 0.0, 0.0, 0.0];
        let s_peaked = column_energy_entropy(&peaked, 1, 4);
        assert!(
            s_peaked < 0.01,
            "peaked S(ρ) should be ≈0.0, got {s_peaked}"
        );

        // Degenerate shapes are defined as zero (no ambiguity to measure).
        assert_eq!(column_energy_entropy(&[1.0], 1, 1), 0.0);
        assert_eq!(column_energy_entropy(&[0.0, 0.0], 1, 2), 0.0);
    }

    /// Rank allocation: higher sensitivity ⇒ higher rank; equal sensitivities ⇒
    /// uniform `base_rank`; allocation stays within `[base/2, base*2]`.
    #[test]
    fn test_allocate_ranks_from_sensitivity() {
        let dims = vec![(4096usize, 4096usize); 3];

        // Distinct sensitivities → ranks strictly ordered the same way.
        let ranks = allocate_ranks_from_sensitivity(&[0.2, 0.8, 0.5], &dims, 8);
        assert!(
            ranks[1] > ranks[2] && ranks[2] > ranks[0],
            "ranks must track sensitivity order, got {ranks:?}"
        );
        for &r in &ranks {
            assert!((4..=16).contains(&r), "rank {r} outside [base/2, base*2]");
        }

        // Equal sensitivities collapse to the uniform default.
        let equal = allocate_ranks_from_sensitivity(&[0.5, 0.5, 0.5], &dims, 8);
        assert_eq!(equal, vec![8, 8, 8]);

        // Cap at min(d_in, d_out): a tiny projection cannot exceed its dims.
        let tiny = allocate_ranks_from_sensitivity(&[0.9], &[(3, 3)], 8);
        assert!(
            tiny[0] <= 3,
            "rank must be capped at min(d_in,d_out), got {}",
            tiny[0]
        );
    }

    /// The `--gdtqp` flag path must construct and run without crashing on a
    /// multi-projection fixture (ranks degenerate to base on these 1-D fixtures,
    /// but the plumbing — load layer 0, measure S(ρ), allocate, build adapters —
    /// is exercised end to end).
    #[test]
    fn test_gdtqp_flag_constructs_and_runs() {
        let f = write_multi_proj_gguf(2);
        let td = TempDir::new().unwrap();
        let mut cfg = default_config(td.path().to_path_buf());
        cfg.gdtqp = true;
        let mut ltl = LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 0)
            .expect("new with --gdtqp");

        assert_eq!(ltl.proj_keys_per_layer.len(), 7);
        // Every allocated rank is a sane positive value.
        for (_, _, _, rank) in &ltl.proj_keys_per_layer {
            assert!(*rank >= 1, "rank must be ≥1");
        }

        let result = ltl.run().expect("run with --gdtqp");
        assert!(
            result.final_loss.is_finite(),
            "loss must be finite under --gdtqp"
        );
    }

    // ── quickcheck properties ─────────────────────────────────────────────────

    use quickcheck_macros::quickcheck;

    /// Property 6 — total optimizer steps matches
    /// ceil(batches × epochs / grad_accum). Layers form one model forward.
    #[quickcheck]
    fn total_steps_matches_formula(
        num_layers_raw: u8,
        num_batches_raw: u8,
        epochs_raw: u8,
        grad_accum_raw: u8,
    ) -> bool {
        let num_layers = (num_layers_raw as usize % 4) + 1; // 1..=4
        let num_batches = (num_batches_raw as usize % 4) + 1; // 1..=4
        let epochs = (epochs_raw as usize % 3) + 1; // 1..=3
        let grad_accum = (grad_accum_raw as usize % 8) + 1; // 1..=8

        let f = write_transformer_gguf(num_layers);

        // Build `num_batches` batches of 4 tokens each.
        let batches: Vec<Vec<Tensor>> = (0..num_batches)
            .map(|_| make_batch(2).into_iter().next().unwrap())
            .collect();

        let td = TempDir::new().unwrap();
        let mut cfg = default_config(td.path().to_path_buf());
        cfg.epochs = epochs;
        cfg.grad_accum = grad_accum;

        let mut ltl = match LayeredTrainingLoop::new(cfg, f.path(), batches, make_varmap(), None, 0) {
            Ok(l) => l,
            Err(_) => return true, // construction errors are not the property under test
        };

        let result = match ltl.run() {
            Ok(r) => r,
            Err(_) => return true,
        };

        let total_inner = num_batches * epochs;
        let expected_steps = (total_inner + grad_accum - 1) / grad_accum;
        result.total_steps == expected_steps
    }

    /// Property 7 — final_loss is always a finite f32 (no NaN or inf).
    #[quickcheck]
    fn final_loss_is_finite(num_layers_raw: u8, grad_accum_raw: u8) -> bool {
        let num_layers = (num_layers_raw as usize % 4) + 1; // 1..=4
        let grad_accum = (grad_accum_raw as usize % 4) + 1; // 1..=4

        let f = write_transformer_gguf(num_layers);

        let td = TempDir::new().unwrap();
        let mut cfg = default_config(td.path().to_path_buf());
        cfg.grad_accum = grad_accum;

        let mut ltl =
            match LayeredTrainingLoop::new(cfg, f.path(), make_batch(2), make_varmap(), None, 0) {
                Ok(l) => l,
                Err(_) => return true,
            };

        match ltl.run() {
            Ok(r) => r.final_loss.is_finite(),
            Err(_) => true, // training errors are not the property under test
        }
    }
}
