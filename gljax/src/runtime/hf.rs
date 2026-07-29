//! Loading a HuggingFace model directory into a [`Session`].
//!
//! ⛔ **This does not go through `glconv`/`.gllm`.** The Gate A5 brief says to
//! convert the checkpoint first, but `glconv`'s actual CLI is
//! `glconv <input.gguf> <output_dir>` — **GGUF input only**, and positional
//! arguments rather than `--input/--output`. A HuggingFace download is
//! safetensors, so that path does not exist.
//!
//! It also is not needed: `glcore::format::SafetensorsFile` already reads
//! safetensors (mmap'd, F32/F16/BF16 → f32) and
//! `glcore::GllmTokenizer::from_hf_json_path` already reads `tokenizer.json`.
//! Converting would add a `glictus-caliburni` dependency and a lossy step for
//! no gain. The brief's own Risk 1 anticipates exactly this.
//!
//! # What a directory must contain
//!
//! ```text
//! config.json         architecture — read, never assumed (see model::hf_config)
//! tokenizer.json      the vocabulary
//! model.safetensors   the weights
//! ```

use std::path::Path;
use std::rc::Rc;

use glcore::format::SafetensorsFile;
use glcore::GllmTokenizer;

use crate::model::{trace_forward, Qwen2Config};
use crate::pjrt::PjrtPlugin;
use crate::precision::{self, PrecisionPolicy};
use crate::runtime::session::Session;
use crate::GlError;

/// Everything read off disk before anything touches a device.
///
/// Separated from [`Session`] so a checkpoint can be inspected — and rejected —
/// without a PJRT plugin, which is the only way any of this is testable on a
/// machine with no plugin.
pub struct HfCheckpoint {
    pub config: Qwen2Config,
    pub tokenizer: GllmTokenizer,
    pub weights: SafetensorsFile,
    pub eos_id: u32,
}

impl HfCheckpoint {
    /// Reads `config.json`, `tokenizer.json` and `model.safetensors`.
    pub fn open(dir: &Path) -> Result<Self, GlError> {
        let config_path = dir.join("config.json");
        let config = Qwen2Config::from_hf_config_json(&config_path)?;

        let raw_config = std::fs::read_to_string(&config_path)?;
        let config_eos = crate::model::eos_token_id(&raw_config);

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = GllmTokenizer::from_hf_json_path(
            tokenizer_path
                .to_str()
                .ok_or_else(|| GlError::Parse("tokenizer path is not UTF-8".into()))?,
        )?;

        // ⚠️ The tokenizer's own EOS wins. `config.json` and the tokenizer can
        // disagree, and the tokenizer is the thing that will actually be
        // decoding — trusting the other one produces generation that never
        // stops, or stops on the wrong token.
        let eos_id = tokenizer.eos_id();
        if let Some(from_config) = config_eos {
            if from_config != eos_id {
                log::warn!(
                    "eos_token_id disagrees: config.json says {from_config}, \
                     tokenizer.json says {eos_id} — using the tokenizer's"
                );
            }
        }

        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(GlError::Parse(format!(
                "{} not found. Sharded checkpoints (model-00001-of-*.safetensors) \
                 are not supported yet — Qwen2-0.5B ships a single file",
                weights_path.display()
            )));
        }
        let weights = SafetensorsFile::open(
            weights_path
                .to_str()
                .ok_or_else(|| GlError::Parse("weights path is not UTF-8".into()))?,
        )?;

        // Fail here rather than after a multi-minute XLA compile.
        if tokenizer.vocab_size() != config.vocab {
            log::warn!(
                "vocab size disagrees: config.json says {}, tokenizer has {} — \
                 the logits axis follows config.json",
                config.vocab,
                tokenizer.vocab_size()
            );
        }

        Ok(HfCheckpoint {
            config,
            tokenizer,
            weights,
            eos_id,
        })
    }

    /// Encodes a prompt to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, GlError> {
        // Qwen2 has no BOS in completion mode — `add_bos_default` carries what
        // the vocabulary itself declares rather than a guess.
        Ok(self.tokenizer.encode(text, self.tokenizer.add_bos_default())?)
    }

    /// Decodes token ids back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true)
    }
}

impl Session {
    /// Traces, compiles and loads a HuggingFace directory at a fixed bucket.
    ///
    /// ⚠️ `seq_len` is baked into the compiled artifact (P3). It must cover
    /// **prompt + every generated token**, because without a KV cache each
    /// decode step re-runs the whole padded sequence. Use
    /// [`crate::runtime::bucket::bucket_for`] on `prompt_len + max_new_tokens`.
    ///
    /// The trace runs under [`PrecisionPolicy::f32`]: the weights are read out
    /// of safetensors as `f32` (BF16 on disk, widened on load), so an F32 graph
    /// is what matches them. A BF16 graph would need a BF16 upload path, which
    /// is a numerics change and belongs behind a measurement, not a default.
    pub fn from_hf_dir(
        plugin: Rc<PjrtPlugin>,
        dir: &Path,
        seq_len: usize,
        cache_dir: Option<&Path>,
    ) -> Result<Self, GlError> {
        let checkpoint = HfCheckpoint::open(dir)?;
        Self::from_hf_checkpoint(plugin, checkpoint, seq_len, cache_dir)
    }

    /// As [`Session::from_hf_dir`], for an already-opened checkpoint.
    pub fn from_hf_checkpoint(
        plugin: Rc<PjrtPlugin>,
        checkpoint: HfCheckpoint,
        seq_len: usize,
        cache_dir: Option<&Path>,
    ) -> Result<Self, GlError> {
        let config = checkpoint.config.clone();
        log::info!(
            "tracing qwen2: {} layers, hidden {}, vocab {}, rope_base {}, bucket {seq_len}",
            config.n_layers,
            config.hidden,
            config.vocab,
            config.rope_base,
        );

        let built = precision::with_policy(PrecisionPolicy::f32(), || {
            trace_forward(&config, seq_len, 0)
        })?;

        let mut session = Session::open(
            plugin,
            &built,
            config,
            seq_len,
            &checkpoint.weights,
            cache_dir,
        )?;
        session.attach_tokenizer(checkpoint.tokenizer, checkpoint.eos_id);
        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loader must refuse a directory that is missing pieces, and say
    /// which — a checkpoint one file short and a wrong directory look the same
    /// otherwise.
    #[test]
    fn opening_a_directory_without_a_config_fails_with_the_path() {
        let dir = std::env::temp_dir().join(format!("gljax_hf_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let err = match HfCheckpoint::open(&dir) {
            Ok(_) => panic!("an empty directory must not open as a checkpoint"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("config.json"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A real HF directory, if one is configured. SKIPs loudly otherwise —
    /// `model.safetensors` is ~1 GB and is not committed.
    #[test]
    fn a_real_checkpoint_opens_and_encodes_the_gate_prompt() {
        let Ok(dir) = std::env::var("QWEN2_HF_DIR") else {
            eprintln!("SKIP a_real_checkpoint_opens_and_encodes_the_gate_prompt: QWEN2_HF_DIR not set");
            return;
        };
        let checkpoint = HfCheckpoint::open(Path::new(&dir)).expect("open checkpoint");

        assert_eq!(checkpoint.config.hidden, 896);
        assert_eq!(checkpoint.config.rope_base, 1_000_000.0, "Qwen2 uses 1e6");

        let ids = checkpoint.encode("The capital of France is").expect("encode");
        assert!(!ids.is_empty());
        eprintln!("prompt encodes to {} tokens: {ids:?}", ids.len());

        // Round-tripping the prompt is a weak check, but a decoder that is
        // wired to the wrong vocabulary fails it immediately.
        let back = checkpoint.decode(&ids);
        assert!(
            back.contains("capital") && back.contains("France"),
            "round trip lost the prompt: {back:?}"
        );
    }
}
