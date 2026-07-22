//! CPP (Combined Precision Policy) Stage 1 — hardcoded sensitivity table
//! (Pridwen v5 §5, §7 Stage 1). No MCKP solver, no calibration data: a tensor
//! name maps straight to a sensitivity bucket via a lookup table.
//!
//! Phase 1 scope is GQ4A only, so [`assign_gq4a_cpp`] is the only assignment
//! function — per v5 §5's own note, GQ4A_CPP is a **degenerate, homogeneous**
//! case: every quantized tensor gets `GQ4A` regardless of bucket, and only
//! the two "always" rows (`output_norm`→F32, `attn_norm`/`ffn_norm`→F16) ever
//! produce a different dtype. This does not exercise heterogeneous
//! per-tensor assignment (that starts at GQ2A_CPP, Phase 2) — Stage 1's
//! sensitivity table exists here in Phase 1 purely so the assignment-engine
//! *pipeline shape* (name → bucket → dtype) is proven before a second format
//! gives it something to actually choose between.

use crate::manifest::DType;

/// Sensitivity bucket from the CPP Stage 1 reference table (Pridwen v5 §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensitivity {
    Extreme,
    High,
    MediumHigh,
    Medium,
    MediumLow,
}

/// Look up which sensitivity bucket a GGUF-style layer/tensor name belongs
/// to. Matches on the tensor's role suffix (e.g. `"blk.3.attn_q.weight"` and
/// `"attn_q"` both match `attn_q`), not the full dotted name, since layer
/// index and `.weight` suffix vary per tensor but the role name doesn't.
///
/// Returns `None` for any name the Stage 1 table doesn't recognize — Phase 1
/// callers should treat that as "not a standard transformer tensor" rather
/// than guessing a bucket.
pub fn sensitivity_bucket_for(name: &str) -> Option<Sensitivity> {
    // Strip a `blk.N.` prefix and a `.weight`/`.bias` suffix if present, so
    // both raw GGUF names and already-stripped GLLM layer tensor names work.
    let stripped = name.strip_prefix("blk.").and_then(|s| s.split_once('.')).map(|(_, rest)| rest).unwrap_or(name);
    let role = stripped
        .strip_suffix(".weight")
        .or_else(|| stripped.strip_suffix(".bias"))
        .unwrap_or(stripped);

    match role {
        "token_embd" | "token_embeddings" | "output" | "output_head" => Some(Sensitivity::Extreme),
        "output_norm" => Some(Sensitivity::Extreme),
        "attn_norm" | "ffn_norm" => Some(Sensitivity::High),
        "attn_q" | "attn_k" => Some(Sensitivity::High),
        "attn_v" | "attn_output" => Some(Sensitivity::MediumHigh),
        "ffn_gate" | "ffn_up" => Some(Sensitivity::Medium),
        "ffn_down" => Some(Sensitivity::MediumLow),
        _ => None,
    }
}

/// GQ4A_CPP assignment (Pridwen v5 §5 table, "GQ4A_CPP assign" column).
///
/// Degenerate by design: every bucket except the two "always" norm rows
/// resolves to `GQ4A`. Returns `None` for a name outside the Stage 1 table —
/// callers decide the fallback (Phase 1's glconv wiring uses the tensor's
/// original GGUF dtype unchanged for anything unrecognized, same as today's
/// unmapped-tensor handling).
pub fn assign_gq4a_cpp(name: &str) -> Option<DType> {
    let bucket = sensitivity_bucket_for(name)?;
    // Only output_norm (Extreme, "F32 always") and attn_norm/ffn_norm (High,
    // "F16 always") diverge from GQ4A in this column; every other bucket
    // assigns GQ4A uniformly (v5 §5 table + note).
    let stripped = name.strip_prefix("blk.").and_then(|s| s.split_once('.')).map(|(_, rest)| rest).unwrap_or(name);
    let role = stripped
        .strip_suffix(".weight")
        .or_else(|| stripped.strip_suffix(".bias"))
        .unwrap_or(stripped);
    match role {
        "output_norm" => Some(DType::F32),
        "attn_norm" | "ffn_norm" => Some(DType::F16),
        _ => {
            let _ = bucket;
            Some(DType::GQ4A)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_every_real_qwen_layer_name() {
        for name in [
            "token_embd", "output", "output_norm", "attn_norm", "ffn_norm",
            "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
        ] {
            assert!(sensitivity_bucket_for(name).is_some(), "missing bucket for {name}");
        }
    }

    #[test]
    fn matches_prefixed_and_suffixed_names() {
        assert_eq!(
            sensitivity_bucket_for("blk.12.attn_q.weight"),
            sensitivity_bucket_for("attn_q")
        );
    }

    #[test]
    fn gq4a_cpp_is_degenerate_except_norms() {
        assert_eq!(assign_gq4a_cpp("blk.0.attn_q.weight"), Some(DType::GQ4A));
        assert_eq!(assign_gq4a_cpp("blk.0.ffn_down.weight"), Some(DType::GQ4A));
        assert!(assign_gq4a_cpp("token_embd.weight").is_some());
        assert_eq!(assign_gq4a_cpp("output_norm.weight"), Some(DType::F32));
        assert_eq!(assign_gq4a_cpp("blk.0.attn_norm.weight"), Some(DType::F16));
        assert_eq!(assign_gq4a_cpp("blk.0.ffn_norm.weight"), Some(DType::F16));
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(sensitivity_bucket_for("rope_freqs").is_none());
        assert!(assign_gq4a_cpp("rope_freqs.weight").is_none());
    }
}
