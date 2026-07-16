//! Model-file probe — what the GGUF header declares about the model.
//!
//! Fills the [`crate::engine::metadata::EngineMetadata`] fields that until now
//! stayed `None` (architecture, quantization), plus the one derived flag the
//! quality analysis needs: whether the model is *thinking-capable* (CoT). A
//! thinking model that emits a `<think>` segment is *expected* to run at very
//! low entropy for long stretches — flagging that as an anomaly would be a
//! false alarm, and not flagging a non-thinking model's entropy collapse would
//! miss a real one. See [`crate::behavior::cot`].
//!
//! Detection is dual (per the v2 decision record): automatic from GGUF metadata
//! (`tokenizer.chat_template` containing a think tag, or a known thinking-model
//! architecture), overridable by the workload's `cot_mode` flag. The probe only
//! reads the header through [`glcore::format::gguf::GgufFile`] — glbench never
//! parses model bytes itself.

use glcore::format::gguf::GgufFile;

/// Facts the model file declares about itself. Every field is `Option`: a file
/// that could not be opened or a key that is absent yields `None` (not probed),
/// never a guessed value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelProbe {
    /// `general.architecture`, e.g. `"qwen3"`.
    pub arch: Option<String>,
    /// Quantization label decoded from `general.file_type`, e.g. `"Q4_K_M"`.
    pub quantization: Option<String>,
    /// Whether the model is thinking-capable (CoT). `Some(true)` when the chat
    /// template contains a think tag or the architecture is a known thinking
    /// family; `Some(false)` when the header was readable but shows neither;
    /// `None` when the file could not be probed at all.
    pub thinking_capable: Option<bool>,
}

impl ModelProbe {
    /// Probe the GGUF header at `path`. Never fails: an unreadable or
    /// non-GGUF file yields an all-`None` probe, because the benchmark must
    /// still run — metadata is context, not a prerequisite.
    pub fn probe(path: &str) -> ModelProbe {
        match GgufFile::open(path) {
            Ok(gguf) => {
                let arch = gguf
                    .get_meta("general.architecture")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let quantization = gguf
                    .get_meta("general.file_type")
                    .and_then(|v| v.as_u64())
                    .and_then(file_type_label)
                    .map(String::from);
                let chat_template = gguf
                    .get_meta("tokenizer.chat_template")
                    .and_then(|v| v.as_str());
                let thinking_capable =
                    Some(detect_thinking(arch.as_deref(), chat_template));
                ModelProbe { arch, quantization, thinking_capable }
            }
            Err(_) => ModelProbe::default(),
        }
    }
}

/// Decide thinking capability from header facts. Pure and separately testable.
///
/// The chat template is the strongest signal: a template that emits `<think>`
/// tokens *is* the thinking mode. The architecture list is the fallback for
/// files whose template was stripped; it names families that ship thinking
/// mode by default, and is deliberately short — an unknown architecture is
/// `false` (no evidence), which the workload's `cot_mode` flag can override.
fn detect_thinking(arch: Option<&str>, chat_template: Option<&str>) -> bool {
    if let Some(t) = chat_template {
        if t.contains("<think>") || t.contains("enable_thinking") {
            return true;
        }
    }
    matches!(arch, Some("qwen3" | "qwen3moe" | "deepseek2" | "deepseek3"))
}

/// Map `general.file_type` (the llama.cpp `LLAMA_FTYPE` enum, stable across
/// the ecosystem) to its conventional label. Unknown values yield `None` —
/// reported as "not probed", never as a made-up label.
fn file_type_label(ft: u64) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        30 => "BF16",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn think_tag_in_template_wins_regardless_of_arch() {
        assert!(detect_thinking(Some("llama"), Some("...{% if %}<think>...")));
    }

    #[test]
    fn known_thinking_arch_without_template() {
        assert!(detect_thinking(Some("qwen3"), None));
        assert!(detect_thinking(Some("qwen3moe"), None));
    }

    #[test]
    fn plain_model_is_not_thinking() {
        assert!(!detect_thinking(Some("qwen2"), Some("{{ messages }}")));
        assert!(!detect_thinking(None, None));
    }

    #[test]
    fn file_type_labels_common_quants() {
        assert_eq!(file_type_label(7), Some("Q8_0"));
        assert_eq!(file_type_label(15), Some("Q4_K_M"));
        assert_eq!(file_type_label(999), None, "unknown must be None, not a guess");
    }

    #[test]
    fn unreadable_file_probes_to_all_none() {
        let p = ModelProbe::probe("definitely/not/a/file.gguf");
        assert_eq!(p, ModelProbe::default());
    }
}
