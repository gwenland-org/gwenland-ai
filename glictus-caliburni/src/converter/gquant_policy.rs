//! CPP (Combined Precision Policy) Stage 1 — hardcoded sensitivity table
//! (Pridwen v5 §5, §7 Stage 1). No MCKP solver, no calibration data: a tensor
//! name maps straight to a sensitivity bucket via a lookup table.
//!
//! [`assign_gq4a_cpp`] (Phase 1) is a **degenerate, homogeneous** case: every
//! quantized tensor gets `GQ4A` regardless of bucket, and only the two
//! "always" rows (`output_norm`→F32, `attn_norm`/`ffn_norm`→F16) ever produce
//! a different dtype — it does not exercise heterogeneous per-tensor
//! assignment. [`assign_gq2a_cpp`] (Phase 2) is where that starts: EXTREME/
//! HIGH-sensitivity tensors escape to GQ4A, everything else (MEDIUM-HIGH and
//! below) gets GQ2A — a real per-tensor format *decision*, not CPP applied
//! to a one-format universe.

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
    match role_of(name) {
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

/// Strip a `blk.N.` prefix and a `.weight`/`.bias` suffix if present, so both
/// raw GGUF names and already-stripped GLLM layer tensor names resolve to
/// the same bare role (e.g. `"blk.3.attn_q.weight"` and `"attn_q"` both
/// yield `"attn_q"`). Shared by every assignment function in this module so
/// the stripping rule only needs to agree with itself once.
fn role_of(name: &str) -> &str {
    let stripped = name.strip_prefix("blk.").and_then(|s| s.split_once('.')).map(|(_, rest)| rest).unwrap_or(name);
    stripped.strip_suffix(".weight").or_else(|| stripped.strip_suffix(".bias")).unwrap_or(stripped)
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
    match role_of(name) {
        "output_norm" => Some(DType::F32),
        "attn_norm" | "ffn_norm" => Some(DType::F16),
        _ => {
            let _ = bucket;
            Some(DType::GQ4A)
        }
    }
}

/// GQ2A_CPP assignment (Pridwen v5 §5 table, "GQ2A_CPP assign" column).
///
/// The first Stage 1 policy that actually chooses between two quantized
/// formats: EXTREME/HIGH-sensitivity tensors escape to `GQ4A` (token_embd,
/// output, attn_q, attn_k — too sensitive for 2-bit per the sensitivity
/// research this Stage 1 table is based on), the two norm rows keep their
/// "always" F32/F16 assignment (identical to `assign_gq4a_cpp`), and
/// everything else (MEDIUM-HIGH and below: attn_v, attn_output, ffn_gate,
/// ffn_up, ffn_down) gets `GQ2A`. Returns `None` for a name outside the
/// Stage 1 table, same fallback contract as `assign_gq4a_cpp`.
pub fn assign_gq2a_cpp(name: &str) -> Option<DType> {
    let _bucket = sensitivity_bucket_for(name)?;
    match role_of(name) {
        "output_norm" => Some(DType::F32),
        "attn_norm" | "ffn_norm" => Some(DType::F16),
        "token_embd" | "token_embeddings" | "output" | "output_head" => Some(DType::GQ4A),
        "attn_q" | "attn_k" => Some(DType::GQ4A),
        "attn_v" | "attn_output" | "ffn_gate" | "ffn_up" | "ffn_down" => Some(DType::GQ2A),
        _ => unreachable!("role_of returned a role sensitivity_bucket_for didn't recognize"),
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

    #[test]
    fn gq2a_cpp_escapes_extreme_and_high_sensitivity_to_gq4a() {
        assert_eq!(assign_gq2a_cpp("token_embd.weight"), Some(DType::GQ4A));
        assert_eq!(assign_gq2a_cpp("output.weight"), Some(DType::GQ4A));
        assert_eq!(assign_gq2a_cpp("blk.0.attn_q.weight"), Some(DType::GQ4A));
        assert_eq!(assign_gq2a_cpp("blk.0.attn_k.weight"), Some(DType::GQ4A));
    }

    #[test]
    fn gq2a_cpp_assigns_gq2a_to_medium_and_below() {
        assert_eq!(assign_gq2a_cpp("blk.0.attn_v.weight"), Some(DType::GQ2A));
        assert_eq!(assign_gq2a_cpp("blk.0.attn_output.weight"), Some(DType::GQ2A));
        assert_eq!(assign_gq2a_cpp("blk.0.ffn_gate.weight"), Some(DType::GQ2A));
        assert_eq!(assign_gq2a_cpp("blk.0.ffn_up.weight"), Some(DType::GQ2A));
        assert_eq!(assign_gq2a_cpp("blk.0.ffn_down.weight"), Some(DType::GQ2A));
    }

    #[test]
    fn gq2a_cpp_norms_match_gq4a_cpp_always_rows() {
        // Both policies' "always" rows are identical (Pridwen v5 §5 table) —
        // only the escape-hatch/mid-sensitivity rows differ between columns.
        assert_eq!(assign_gq2a_cpp("output_norm.weight"), Some(DType::F32));
        assert_eq!(assign_gq2a_cpp("blk.0.attn_norm.weight"), Some(DType::F16));
        assert_eq!(assign_gq2a_cpp("blk.0.ffn_norm.weight"), Some(DType::F16));
        assert_eq!(
            assign_gq2a_cpp("output_norm.weight"),
            assign_gq4a_cpp("output_norm.weight")
        );
    }

    #[test]
    fn gq2a_cpp_is_heterogeneous_unlike_gq4a_cpp() {
        // The whole point of Phase 2's policy: two different quantized
        // dtypes appear across a realistic tensor set, unlike GQ4A_CPP's
        // uniform column (Pridwen v5 §5's note).
        let names = [
            "token_embd.weight", "output.weight", "output_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
            "blk.0.attn_q.weight", "blk.0.attn_k.weight",
            "blk.0.attn_v.weight", "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        ];
        let dtypes: std::collections::HashSet<DType> =
            names.iter().filter_map(|n| assign_gq2a_cpp(n)).collect();
        assert!(dtypes.contains(&DType::GQ4A));
        assert!(dtypes.contains(&DType::GQ2A));
        assert!(dtypes.contains(&DType::F32));
        assert!(dtypes.contains(&DType::F16));
    }

    #[test]
    fn gq2a_cpp_unknown_name_returns_none() {
        assert!(assign_gq2a_cpp("rope_freqs.weight").is_none());
    }
}
