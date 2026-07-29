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

/// `config.json` + `tokenizer.json`, without the weights.
///
/// Split out because the weights are ~1 GB and the metadata is where the
/// interesting failures live — a wrong `rope_theta`, an EOS that stops nothing,
/// a vocabulary that lost its added tokens. A test that needs a 1 GB download
/// is a test that does not run.
pub struct HfMetadata {
    pub config: Qwen2Config,
    pub tokenizer: GllmTokenizer,
    pub eos_id: u32,
}

impl HfMetadata {
    /// Reads `config.json` and `tokenizer.json`.
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

        // ⛔ **`config.json` wins on EOS**, and this is not a preference.
        //
        // `tokenizer.json` has no unambiguous EOS field, so
        // `Vocab::from_hf_json` falls back to "the last added token". For
        // `Qwen/Qwen2-0.5B` that is `<|im_end|>` (151645) — the *chat*
        // terminator — while the model's own `config.json` declares
        // `eos_token_id: 151643` (`<|endoftext|>`).
        //
        // A base model generating until it emits a chat terminator it was
        // never trained to emit does not stop: every run would burn the full
        // `max_new_tokens` and the transcript would run past the answer. The
        // model's own config is the authority on which token ends its output.
        let tokenizer_eos = tokenizer.eos_id();
        let eos_id = match config_eos {
            Some(from_config) => {
                if from_config != tokenizer_eos {
                    log::info!(
                        "eos_token_id: config.json says {from_config}, tokenizer.json's \
                         heuristic says {tokenizer_eos} — using config.json's"
                    );
                }
                from_config
            }
            None => {
                log::warn!(
                    "config.json has no eos_token_id; falling back to the tokenizer's \
                     heuristic ({tokenizer_eos}), which is positional and may be wrong"
                );
                tokenizer_eos
            }
        };

        // ⚠️ `config.vocab` is padded for tensor alignment (Qwen2: 151936)
        // while the tokenizer holds the real tokens (151646). The logits axis
        // follows config.json — the embedding matrix really is that wide — so
        // the tokenizer being *smaller* is expected. It being larger is not.
        if tokenizer.vocab_size() > config.vocab {
            return Err(GlError::Engine(format!(
                "tokenizer has {} tokens but config.json's vocab_size is {} — \
                 the tokenizer cannot be wider than the embedding matrix",
                tokenizer.vocab_size(),
                config.vocab
            )));
        }

        Ok(HfMetadata {
            config,
            tokenizer,
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

impl HfCheckpoint {
    /// Reads the metadata, then mmaps `model.safetensors`.
    pub fn open(dir: &Path) -> Result<Self, GlError> {
        let meta = HfMetadata::open(dir)?;

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

        Ok(HfCheckpoint {
            config: meta.config,
            tokenizer: meta.tokenizer,
            weights,
            eos_id: meta.eos_id,
        })
    }

    /// Encodes a prompt to token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, GlError> {
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

    /// ⭐ The metadata half against the **real** `Qwen/Qwen2-0.5B` files.
    ///
    /// Needs only `config.json` + `tokenizer.json` (~7 MB), not the 1 GB of
    /// weights, so it is cheap enough to actually run. Point
    /// `QWEN2_META_DIR` at a directory holding those two.
    #[test]
    fn real_qwen2_metadata_loads_with_the_right_rope_base_and_eos() {
        let Ok(dir) = std::env::var("QWEN2_META_DIR") else {
            eprintln!("SKIP real_qwen2_metadata_loads...: QWEN2_META_DIR not set");
            return;
        };
        let meta = HfMetadata::open(Path::new(&dir)).expect("metadata must load");

        // ⛔ The bug this whole path exists to prevent.
        assert_eq!(meta.config.rope_base, 1_000_000.0, "Qwen2 uses 1e6, not 1e4");
        assert_eq!(meta.config.hidden, 896);
        assert_eq!(meta.config.n_heads, 14);
        assert_eq!(meta.config.n_kv_heads, 2);
        assert_eq!(meta.config.vocab, 151_936);

        // ⛔ config.json's eos wins over the tokenizer's positional guess:
        // 151643 <|endoftext|>, not 151645 <|im_end|>.
        assert_eq!(meta.eos_id, 151_643, "base model stops on <|endoftext|>");

        // ⛔ added_tokens must be in the vocabulary. Before the glcore fix this
        // loader failed outright: "eos id 151645 is outside a vocabulary of
        // 151643".
        assert_eq!(
            meta.tokenizer.vocab_size(),
            151_646,
            "151643 vocab entries + 3 added tokens"
        );

        let ids = meta.encode("The capital of France is").expect("encode");
        assert!(!ids.is_empty());
        eprintln!("prompt -> {} tokens {ids:?}", ids.len());
        let back = meta.decode(&ids);
        assert!(
            back.contains("capital") && back.contains("France"),
            "round trip lost the prompt: {back:?}"
        );
    }
}
