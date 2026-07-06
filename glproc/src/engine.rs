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

        let token_ids = runner.generate(
            &input.token_ids,
            input.max_new_tokens,
            &mut sampler,
            tokenizer.eos_id(),
            |id| {
                let piece = tokenizer.decode_token_text(id);
                if let Some(cb) = on_token.as_deref_mut() {
                    cb(id, &piece);
                }
                text.push_str(&piece);
            },
        )?;

        let tokens_generated = token_ids.len();
        Ok(InferOutput {
            token_ids,
            text,
            tokens_generated,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}

impl GlEngine for GlprocEngine {
    fn init(&mut self) -> Result<(), GlError> {
        // CPU backend: nothing to detect or allocate up front.
        Ok(())
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

        self.tokenizer = Some(Tokenizer::from_gguf(&gguf)?);
        self.model = Some(load_gguf(&gguf)?);
        Ok(())
    }

    fn infer(&self, input: InferInput) -> Result<InferOutput, GlError> {
        self.run(&input, None)
    }

    fn stream(
        &self,
        input: InferInput,
        on_token: &(dyn Fn(u32, &str) + Send),
    ) -> Result<(), GlError> {
        let mut forward = |id: u32, piece: &str| on_token(id, piece);
        self.run(&input, Some(&mut forward))?;
        Ok(())
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
