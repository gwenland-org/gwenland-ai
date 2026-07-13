//! `GlprocEngine` — the CPU implementation of the `GlEngine` trait.

use std::time::Instant;

use glcore::engine_trait::{EngineSpec, GlEngine, InferInput, InferOutput};
use glcore::format::gguf::GgufFile;
use glcore::tokenizer::Tokenizer;
use glcore::GlError;

use crate::loader::load_gguf;
use crate::model::GlprocModel;
use crate::runner::Runner;
use crate::sampler::{Sampler, SamplerConfig};

/// Engine-level knobs (kept small for M1).
#[derive(Debug, Clone, Default)]
pub struct GlprocConfig {
    /// Fixed RNG seed for reproducible sampling; `None` = time-seeded.
    pub seed: Option<u64>,
}

/// Pure-Rust CPU inference engine. The source of truth all GPU backends
/// are validated against.
#[derive(Default)]
pub struct GlprocEngine {
    model: Option<GlprocModel>,
    tokenizer: Option<Tokenizer>,
    config: GlprocConfig,
    /// Telemetry from the most recent `infer`/`stream`.
    ///
    /// A `Mutex` because `GlEngine::infer` takes `&self` (the trait is
    /// `Send + Sync` so `Runtime` can hold engines behind a `Box<dyn>`), yet
    /// the run produces facts we must keep. Locked exactly twice per inference
    /// — once to store, once when a consumer pulls — never in the hot path.
    last_telemetry: std::sync::Mutex<Option<glcore::telemetry::EngineTelemetry>>,
}

impl GlprocEngine {
    /// Create an engine with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an engine with an explicit configuration.
    pub fn with_config(config: GlprocConfig) -> Self {
        GlprocEngine {
            model: None,
            tokenizer: None,
            config,
            last_telemetry: std::sync::Mutex::new(None),
        }
    }

    fn model(&self) -> Result<&GlprocModel, GlError> {
        self.model
            .as_ref()
            .ok_or_else(|| GlError::Engine("no model loaded".into()))
    }

    fn sampler_for(&self, input: &InferInput) -> Sampler {
        Sampler::new(SamplerConfig {
            temperature: input.temperature,
            top_k: input.top_k,
            top_p: input.top_p,
            repeat_penalty: input.repeat_penalty,
            seed: self.config.seed,
        })
    }

    /// Run generation, invoking `on_token` per token when provided.
    fn run(
        &self,
        input: &InferInput,
        mut on_token: Option<&mut dyn FnMut(u32, &str)>,
    ) -> Result<InferOutput, GlError> {
        let model = self.model()?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| GlError::Engine("no tokenizer loaded".into()))?;

        let mut runner = Runner::new(model);
        let mut sampler = self.sampler_for(input);
        let started = Instant::now();
        let mut text = String::new();

        let (token_ids, timing) = runner.generate(
            &input.token_ids,
            input.max_new_tokens,
            &mut sampler,
            |id| tokenizer.is_stop_token(id),
            |id| {
                let piece = tokenizer.decode_token_text(id);
                if let Some(cb) = on_token.as_deref_mut() {
                    cb(id, &piece);
                }
                text.push_str(&piece);
            },
        )?;

        // Capture the run's telemetry before the Runner (and its counters) drop.
        // The Runner knows stage timings and expert load; only the engine knows
        // the SIMD strategy, the model's byte size, and how many layers are
        // routed — so enrich here rather than plumbing the model into `Prof`.
        if let Some(mut t) = runner.telemetry() {
            t.backend = Some(self.backend_telemetry(model));
            t.memory = Some(self.memory_telemetry(model));
            if let Some(moe) = t.moe.as_mut() {
                moe.moe_layers = model
                    .layers
                    .iter()
                    .filter(|l| matches!(l.ffn, crate::model::FfnLayer::MoE(_)))
                    .count();
            }
            // A poisoned lock here must not fail the inference — the tokens are
            // already generated and correct. Drop the telemetry instead.
            if let Ok(mut slot) = self.last_telemetry.lock() {
                *slot = Some(t);
            }
        }

        let tokens_generated = token_ids.len();
        Ok(InferOutput {
            token_ids,
            text,
            tokens_generated,
            elapsed_ms: started.elapsed().as_millis() as u64,
            prompt_tokens: timing.prompt_tokens,
            prefill_ms: timing.prefill.as_secs_f64() * 1e3,
            generation_ms: timing.decode.as_secs_f64() * 1e3,
        })
    }

    /// What the engine actually *chose* — not what the CPU supports.
    ///
    /// These differ, and the gap is load-bearing: glproc deliberately rejects
    /// AVX-512 on low-core parts because it downclocks, so a machine whose
    /// CPUID says `avx512f` may still be running AVX2 kernels. Reporting the
    /// CPU's capability alone would hide that.
    fn backend_telemetry(&self, model: &GlprocModel) -> glcore::telemetry::BackendTelemetry {
        use crate::model::{FfnLayer, GateUp, WeightMatrix};
        let strategy = crate::simd_strategy::SimdStrategy::detect();

        let kernel_of = |w: &WeightMatrix| match w {
            WeightMatrix::F32(_) => "f32 dense".to_string(),
            WeightMatrix::Quant(fmt, _) if crate::kernels::qdot::supports(*fmt) => {
                format!("{fmt:?} integer-dot")
            }
            WeightMatrix::Quant(fmt, _) => format!("{fmt:?} f32-bridge"),
        };

        let mut kernels = Vec::new();
        if let Some(l) = model.layers.first() {
            match &l.ffn {
                FfnLayer::Dense { gate_up, w_down } => {
                    kernels.push((
                        "ffn_gate_up".to_string(),
                        match gate_up {
                            GateUp::FusedQuant(fmt, _) => {
                                format!("{fmt:?} fused-swiglu integer-dot")
                            }
                            GateUp::Split(g, _) => kernel_of(g),
                        },
                    ));
                    kernels.push(("ffn_down".to_string(), kernel_of(w_down)));
                }
                FfnLayer::MoE(moe) => {
                    if let Some(e) = moe.experts.first() {
                        kernels.push((
                            "moe_expert_gate_up".to_string(),
                            match &e.gate_up {
                                GateUp::FusedQuant(fmt, _) => {
                                    format!("{fmt:?} fused-swiglu integer-dot")
                                }
                                GateUp::Split(g, _) => kernel_of(g),
                            },
                        ));
                        kernels.push(("moe_expert_down".to_string(), kernel_of(&e.w_down)));
                    }
                }
            }
        }
        kernels.push(("lm_head".to_string(), kernel_of(&model.output)));

        glcore::telemetry::BackendTelemetry {
            simd_path: format!("{:?}", strategy).to_lowercase(),
            threads: crate::runner::thread_count(),
            kernels,
        }
    }

    /// Weight bytes vs KV cache vs scratch. One "peak RSS" number cannot tell
    /// these apart, yet only the KV cache grows with context length — so a
    /// combined figure hides the thing you would actually tune.
    fn memory_telemetry(&self, model: &GlprocModel) -> glcore::telemetry::MemoryTelemetry {
        use crate::model::{FfnLayer, GateUp, WeightMatrix};
        let wbytes = |w: &WeightMatrix| match w {
            WeightMatrix::F32(v) => v.len() * 4,
            WeightMatrix::Quant(_, b) => b.len(),
        };
        let gu_bytes = |gu: &GateUp| match gu {
            GateUp::FusedQuant(_, p) => p.len(),
            GateUp::Split(g, u) => wbytes(g) + wbytes(u),
        };

        let mut model_bytes = wbytes(&model.token_embd) + wbytes(&model.output);
        for l in &model.layers {
            model_bytes += wbytes(&l.wo);
            model_bytes += match &l.qkv {
                crate::model::QkvWeights::FusedQuant(_, p) => p.len(),
                crate::model::QkvWeights::Split(q, k, v) => wbytes(q) + wbytes(k) + wbytes(v),
            };
            model_bytes += match &l.ffn {
                FfnLayer::Dense { gate_up, w_down } => gu_bytes(gate_up) + wbytes(w_down),
                // Every expert counts, not just the active ones: they are all
                // resident, which is exactly why MoE trades RAM for compute.
                FfnLayer::MoE(moe) => {
                    moe.router.len() * 4
                        + moe
                            .experts
                            .iter()
                            .map(|e| gu_bytes(&e.gate_up) + wbytes(&e.w_down))
                            .sum::<usize>()
                }
            };
        }

        glcore::telemetry::MemoryTelemetry {
            model_bytes: model_bytes as u64,
            kv_cache_bytes: crate::runner::kv_cache_bytes(&model.config) as u64,
            scratch_bytes: 0, // workspace is per-Runner and already dropped
        }
    }
}

/// One startup line naming the SIMD strategy and the kernel path each hot
/// weight class will take — a scalar fallback in the FFN would silently eat
/// the whole token budget, so make the dispatch visible.
fn log_simd_paths(model: &GlprocModel) {
    use crate::model::{FfnLayer, GateUp, WeightMatrix};
    let strategy = crate::simd_strategy::SimdStrategy::detect();
    let vnni = crate::kernels::qdot::has_vnni_256();
    let path = |w: &WeightMatrix| match w {
        WeightMatrix::F32(_) => "f32 dense".to_string(),
        WeightMatrix::Quant(fmt, _) if crate::kernels::qdot::supports(*fmt) => {
            format!("{fmt:?} integer-dot")
        }
        WeightMatrix::Quant(fmt, _) => format!("{fmt:?} f32-bridge"),
    };
    let gu_path = |gu: &GateUp| match gu {
        GateUp::FusedQuant(fmt, _) => format!("{fmt:?} fused-swiglu integer-dot"),
        GateUp::Split(g, _) => path(g),
    };
    // Report the MoE shape up front: on a routed model the expert count and
    // top-k are the first thing worth confirming, since the `_exps` tensor
    // layout is the one part of the load path not verified against a real file.
    let n_moe = model
        .layers
        .iter()
        .filter(|l| matches!(l.ffn, FfnLayer::MoE(_)))
        .count();
    if n_moe > 0 {
        if let Some(FfnLayer::MoE(moe)) = model.layers.iter().find_map(|l| match &l.ffn {
            m @ FfnLayer::MoE(_) => Some(m),
            _ => None,
        }) {
            let c = &moe.config;
            eprintln!(
                "[moe] {n_moe}/{} layers routed | {} experts, top-{} | expert_ffn {} | \
                 norm_topk {}",
                model.layers.len(),
                c.num_experts,
                c.num_experts_per_tok,
                c.expert_ffn_size,
                c.norm_topk_prob,
            );
        }
    }
    let (ffn_gateup, ffn_down) = model
        .layers
        .first()
        .map(|l| match &l.ffn {
            FfnLayer::Dense { gate_up, w_down } => (gu_path(gate_up), path(w_down)),
            FfnLayer::MoE(moe) => match moe.experts.first() {
                Some(e) => (
                    format!("moe expert[0] {}", gu_path(&e.gate_up)),
                    path(&e.w_down),
                ),
                None => ("moe (no experts)".into(), "?".into()),
            },
        })
        .unwrap_or_else(|| ("?".into(), "?".into()));
    eprintln!(
        "[simd] strategy: {strategy:?}{} | ffn gate/up: {ffn_gateup} | \
         ffn down: {ffn_down} | lm_head: {}",
        if vnni { "+vnni256" } else { "" },
        path(&model.output),
    );
    if strategy == crate::simd_strategy::SimdStrategy::Scalar {
        eprintln!("[simd] WARNING: scalar fallback — AVX2 not active, expect ~10x slowdown");
    }
}

impl GlEngine for GlprocEngine {
    fn init(&mut self) -> Result<(), GlError> {
        // CPU backend: nothing to detect or allocate up front.
        Ok(())
    }

    /// Telemetry from the most recent run, or `None` if profiling was off
    /// (`GLPROC_PROFILE` unset) or nothing has run yet.
    fn telemetry(&self) -> Option<glcore::telemetry::EngineTelemetry> {
        self.last_telemetry.lock().ok()?.clone()
    }

    fn load_model(&mut self, path: &str) -> Result<(), GlError> {
        if path.to_ascii_lowercase().ends_with(".safetensors") {
            return Err(GlError::Engine(
                "glproc M1 loads GGUF only — safetensors inference needs a \
                 config.json sidecar and lands in M2"
                    .into(),
            ));
        }
        let gguf = GgufFile::open(path)?;

        // Warm the page cache behind the mmap. mmap gives address space, not
        // physical pages — without this, the first decode pass takes the page
        // faults and stalls. A sequential background read pulls the whole
        // file in while metadata parsing and dequantization proceed.
        {
            let path = path.to_string();
            std::thread::spawn(move || {
                use std::io::Read;
                if let Ok(mut f) = std::fs::File::open(&path) {
                    let mut buf = vec![0u8; 1 << 20];
                    while matches!(f.read(&mut buf), Ok(n) if n > 0) {}
                }
            });
        }

        let t_parse = Instant::now();
        self.tokenizer = Some(Tokenizer::from_gguf(&gguf)?);
        let parse_s = t_parse.elapsed().as_secs_f64();
        let t_weights = Instant::now();
        let model = load_gguf(&gguf)?;
        let weights_s = t_weights.elapsed().as_secs_f64();
        // X5 step 1: fault every weight page in and pin it before the first
        // token, so no decode ever stalls on a page fault or swap-in.
        let t_pin = Instant::now();
        crate::loader::warm_and_lock_model(&model);
        eprintln!(
            "[load] tokenizer {parse_s:.2}s | weights {weights_s:.2}s | pin {:.2}s",
            t_pin.elapsed().as_secs_f64(),
        );
        log_simd_paths(&model);
        self.model = Some(model);
        Ok(())
    }

    fn infer(&self, input: InferInput) -> Result<InferOutput, GlError> {
        self.run(&input, None)
    }

    fn stream(
        &self,
        input: InferInput,
        on_token: &(dyn Fn(u32, &str) + Send),
    ) -> Result<InferOutput, GlError> {
        let mut forward = |id: u32, piece: &str| on_token(id, piece);
        self.run(&input, Some(&mut forward))
    }

    fn shutdown(&mut self) {
        self.model = None;
        self.tokenizer = None;
    }

    fn capabilities(&self) -> EngineSpec {
        EngineSpec {
            name: "glproc",
            backend: "cpu",
            available: true,
        }
    }
}
