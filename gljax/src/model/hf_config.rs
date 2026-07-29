//! Reading a HuggingFace `config.json`.
//!
//! ⛔ **Hardcoding a config is how you get silently wrong output.** The
//! motivating case is real: gljax's `Qwen2Config::qwen2_0_5b()` shipped with
//! `rope_base: 10_000.0`, the value every Llama-family model uses.
//! `Qwen/Qwen2-0.5B`'s `config.json` says **`"rope_theta": 1000000.0`** — a
//! hundred times larger. Every position would have been rotated by the wrong
//! angle, with no shape error and no crash, degrading further into the
//! sequence. P4 exactly.
//!
//! # Why a hand-rolled scanner
//!
//! ARTX01 §5.4 caps gljax at three dependencies (`glcore`, `libloading`,
//! `log`). `serde_json` is already in the workspace lockfile via `glcore`, so
//! taking it would cost no build time — but that is a dependency decision, and
//! the same reasoning that produced a hand-written SHA-256 applies here.
//!
//! A HuggingFace `config.json` is flat: string keys mapping to numbers, bools,
//! strings and one array of architecture names. Nothing nested that this needs.
//! The scanner reads exactly that shape and **refuses anything else** rather
//! than guessing — an unparseable config must not silently fall back to
//! defaults, because defaults are what this module exists to prevent.
//!
//! ⭐ If a fourth dependency ever becomes acceptable, delete this file and use
//! `serde_json`. The tests here pin the real published config, so a swap is
//! verifiable.

use crate::GlError;

/// A `"key": value` pair from a flat JSON object, values kept as text.
struct Fields<'a> {
    src: &'a str,
}

impl<'a> Fields<'a> {
    fn new(src: &'a str) -> Self {
        Fields { src }
    }

    /// The raw text of `key`'s value, with surrounding whitespace trimmed and
    /// string quotes stripped.
    ///
    /// Scans for `"key"` followed by `:`, then takes everything up to the next
    /// top-level `,` or `}`. Adequate for a flat object; deliberately not a
    /// JSON parser.
    fn raw(&self, key: &str) -> Option<&'a str> {
        let needle = format!("\"{key}\"");
        let mut from = 0usize;
        while let Some(rel) = self.src[from..].find(&needle) {
            let at = from + rel;
            let after = &self.src[at + needle.len()..];
            let after_trimmed = after.trim_start();
            // Only a `:` makes this a key rather than a string value that
            // happens to equal the key name.
            if let Some(rest) = after_trimmed.strip_prefix(':') {
                let rest = rest.trim_start();
                let end = rest.find([',', '}', '\n']).unwrap_or(rest.len());
                let value = rest[..end].trim().trim_end_matches(',');
                return Some(value.trim().trim_matches('"'));
            }
            from = at + needle.len();
        }
        None
    }

    fn usize_of(&self, key: &str) -> Result<usize, GlError> {
        let raw = self.require(key)?;
        raw.parse::<usize>()
            .map_err(|_| self.bad(key, raw, "a non-negative integer"))
    }

    fn f64_of(&self, key: &str) -> Result<f64, GlError> {
        let raw = self.require(key)?;
        raw.parse::<f64>()
            .map_err(|_| self.bad(key, raw, "a number"))
    }

    fn bool_of(&self, key: &str, default: bool) -> Result<bool, GlError> {
        match self.raw(key) {
            None => Ok(default),
            Some("true") => Ok(true),
            Some("false") => Ok(false),
            Some(other) => Err(self.bad(key, other, "true or false")),
        }
    }

    fn require(&self, key: &str) -> Result<&'a str, GlError> {
        self.raw(key).ok_or_else(|| {
            GlError::Parse(format!(
                "config.json: missing required field {key:?}. gljax refuses to \
                 substitute a default — a wrong hyperparameter produces fluent, \
                 wrong output with no error anywhere"
            ))
        })
    }

    fn bad(&self, key: &str, got: &str, want: &str) -> GlError {
        GlError::Parse(format!(
            "config.json: field {key:?} is {got:?}, expected {want}"
        ))
    }
}

impl super::Qwen2Config {
    /// Reads a HuggingFace `config.json`.
    ///
    /// Every architectural field comes from the file. Nothing is defaulted
    /// except the two booleans HF omits when false.
    ///
    /// # Errors
    /// If the file is missing, a required field is absent, a value does not
    /// parse, or `model_type` is not `qwen2` (P5 — refusing beats running a
    /// Llama checkpoint through a Qwen2 graph, which differs in exactly the
    /// places that produce fluent wrong output).
    pub fn from_hf_config_json(path: &std::path::Path) -> Result<Self, GlError> {
        let src = std::fs::read_to_string(path).map_err(|e| {
            GlError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", path.display()),
            ))
        })?;
        Self::parse_hf_config(&src)
    }

    /// The parsing half, separated so it is testable without a file.
    pub fn parse_hf_config(src: &str) -> Result<Self, GlError> {
        let f = Fields::new(src);

        let model_type = f.require("model_type")?;
        if model_type != "qwen2" {
            return Err(GlError::Engine(format!(
                "config.json: model_type is {model_type:?}, gljax's only \
                 architecture is \"qwen2\". Refusing rather than approximating — \
                 a near-miss architecture differs in exactly the places that \
                 produce fluent wrong output (biases, norm placement, RoPE style)"
            )));
        }

        let hidden = f.usize_of("hidden_size")?;
        let n_heads = f.usize_of("num_attention_heads")?;
        if n_heads == 0 || hidden % n_heads != 0 {
            return Err(GlError::Parse(format!(
                "config.json: hidden_size {hidden} is not divisible by \
                 num_attention_heads {n_heads}"
            )));
        }

        let cfg = super::Qwen2Config {
            hidden,
            n_layers: f.usize_of("num_hidden_layers")?,
            n_heads,
            n_kv_heads: f.usize_of("num_key_value_heads")?,
            // HF does not store head_dim for Qwen2; it is hidden / heads.
            head_dim: hidden / n_heads,
            ffn: f.usize_of("intermediate_size")?,
            vocab: f.usize_of("vocab_size")?,
            rms_eps: f.f64_of("rms_norm_eps")?,
            // ⛔ The field that motivated this whole module.
            rope_base: f.f64_of("rope_theta")? as f32,
            max_seq_len: f.usize_of("max_position_embeddings")?,
            tie_word_embeddings: f.bool_of("tie_word_embeddings", false)?,
            // Qwen2 always biases q/k/v. HF encodes this in the architecture
            // name rather than a flag, so it follows from model_type == qwen2.
            attn_bias: true,
        };
        Ok(cfg)
    }
}

/// The end-of-sequence token id from a HuggingFace `config.json`.
///
/// Returned separately from [`super::Qwen2Config`] because it is a tokenizer
/// fact, not an architecture one — and the tokenizer carries its own, which
/// should win when they disagree.
pub fn eos_token_id(src: &str) -> Option<u32> {
    Fields::new(src).raw("eos_token_id")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Qwen2Config;

    /// The published `Qwen/Qwen2-0.5B/config.json`, verbatim, at revision
    /// `91d2aff3f957f99e4c74c962f2f408dcc88a18d8`.
    const QWEN2_0_5B: &str = r#"{
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 131072,
  "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}"#;

    #[test]
    fn parses_the_published_qwen2_0_5b_config() {
        let cfg = Qwen2Config::parse_hf_config(QWEN2_0_5B).expect("must parse");
        assert_eq!(cfg.hidden, 896);
        assert_eq!(cfg.n_layers, 24);
        assert_eq!(cfg.n_heads, 14);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.ffn, 4864);
        assert_eq!(cfg.vocab, 151_936);
        assert_eq!(cfg.rms_eps, 1e-6);
        assert_eq!(cfg.max_seq_len, 131_072);
        assert!(cfg.tie_word_embeddings);
        assert!(cfg.attn_bias);
        assert_eq!(cfg.gqa_repeat(), 7);
    }

    /// ⭐ The field this module exists for. 1e6, not 1e4.
    #[test]
    fn rope_theta_is_one_million_not_ten_thousand() {
        let cfg = Qwen2Config::parse_hf_config(QWEN2_0_5B).expect("must parse");
        assert_eq!(cfg.rope_base, 1_000_000.0);
        assert_ne!(
            cfg.rope_base,
            crate::ops::DEFAULT_ROPE_BASE,
            "Qwen2 does not use the Llama base"
        );
    }

    /// The hardcoded constructor must agree with the published file, or the
    /// two drift and only one of them is right.
    #[test]
    fn the_hardcoded_config_matches_the_published_one() {
        let parsed = Qwen2Config::parse_hf_config(QWEN2_0_5B).expect("must parse");
        let hardcoded = Qwen2Config::qwen2_0_5b();
        assert_eq!(parsed, hardcoded);
    }

    #[test]
    fn eos_token_id_is_read_out_of_the_same_file() {
        assert_eq!(eos_token_id(QWEN2_0_5B), Some(151_643));
    }

    /// ⛔ A missing field must fail loudly. Defaulting is the failure mode this
    /// module was written to remove.
    #[test]
    fn a_missing_required_field_is_refused_not_defaulted() {
        let without = QWEN2_0_5B.replace("\"rope_theta\": 1000000.0,\n", "");
        let err = Qwen2Config::parse_hf_config(&without).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("rope_theta"), "{msg}");
        assert!(msg.contains("refuses to substitute a default"), "{msg}");
    }

    #[test]
    fn a_non_qwen2_architecture_is_refused() {
        let llama = QWEN2_0_5B.replace("\"model_type\": \"qwen2\"", "\"model_type\": \"llama\"");
        let err = Qwen2Config::parse_hf_config(&llama).expect_err("must refuse");
        assert!(err.to_string().contains("llama"), "{err}");
    }

    #[test]
    fn a_malformed_value_is_refused() {
        let bad = QWEN2_0_5B.replace("\"hidden_size\": 896", "\"hidden_size\": \"big\"");
        let err = Qwen2Config::parse_hf_config(&bad).expect_err("must refuse");
        assert!(err.to_string().contains("hidden_size"), "{err}");
    }

    #[test]
    fn inconsistent_head_geometry_is_refused() {
        let bad = QWEN2_0_5B.replace("\"num_attention_heads\": 14", "\"num_attention_heads\": 15");
        let err = Qwen2Config::parse_hf_config(&bad).expect_err("must refuse");
        assert!(err.to_string().contains("divisible"), "{err}");
    }

    /// The scanner must not match a key name that appears as a *value*.
    #[test]
    fn a_key_name_appearing_as_a_string_value_is_not_matched() {
        let tricky = r#"{
  "model_type": "qwen2",
  "some_note": "rope_theta",
  "rope_theta": 1000000.0,
  "hidden_size": 896,
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "intermediate_size": 4864,
  "vocab_size": 151936,
  "rms_norm_eps": 1e-06,
  "max_position_embeddings": 131072,
  "tie_word_embeddings": true
}"#;
        let cfg = Qwen2Config::parse_hf_config(tricky).expect("must parse");
        assert_eq!(cfg.rope_base, 1_000_000.0);
    }

    #[test]
    fn tie_word_embeddings_defaults_to_false_when_absent() {
        let untied = QWEN2_0_5B.replace("\"tie_word_embeddings\": true,\n", "");
        let cfg = Qwen2Config::parse_hf_config(&untied).expect("must parse");
        assert!(!cfg.tie_word_embeddings);
    }
}
