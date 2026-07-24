//! `glbench tensor-stats` — per-tensor value statistics over every real
//! tensor in a `.gllm` package, decoded to f32 exactly like the runtime
//! does, flagging only unambiguous structural anomalies. Gated behind
//! `gllm-bench`, same as [`crate::ppl`] and [`crate::kl_divergence`].
//!
//! # Vetted against `RESEARCH_REQUIREMENTS.md`'s 8 mandatory questions
//!
//! 1. **What engineering problem does this solve?** `quant-info` (Wave 1)
//!    reads `gllm.json` only — dtype labels and counts, never the actual
//!    tensor bytes. There is currently no way to ask "does any tensor in
//!    this package actually contain NaN, Inf, or a degenerate (zero-variance)
//!    block" without writing a one-off script. That is a real gap: a
//!    corrupted conversion, a bad dequant kernel, or a truncated write can
//!    all produce exactly this signature.
//! 2. **Who benefits?** Anyone converting a new model or changing the
//!    converter/dequant path, who wants a fast, purely structural sanity
//!    check *before* running inference or `kl-div` at all.
//! 3. **Used in production/research?** Yes — this is standard practice for
//!    checkpoint validation (e.g. `torch.isnan`/`isinf` sweeps before
//!    deploying a fine-tuned checkpoint); nothing novel about the technique
//!    itself, only its absence from this codebase.
//! 4. **How is it calculated?** Decode every tensor to f32 (the same
//!    dispatch `GllmEngine::load_shared`/`GlprocBackend::required_tensor`
//!    use: `GQ4A`/`GQ2A` via `glproc::kernels::gquant`, everything else via
//!    `glcore::format::decode_tensor`), then a single pass computing count,
//!    mean, population std-dev, min, max, NaN count, Inf count.
//! 5. **Reproducible?** Yes — pure function of the package's own bytes, no
//!    randomness, no sampling.
//! 6. **Actionable insight?** Yes, directly: it names the exact tensor with
//!    the problem, not just "something in this package is wrong."
//! 7. **Lightweight?** Yes — one linear pass per tensor, no repeated
//!    inference, no engine load. The whole scan is faster than a single
//!    `kl-div` forward pass.
//! 8. **Aligns with philosophy?** Yes — read-only, reports facts (counts),
//!    not verdicts.
//!
//! # Deliberately NOT included: magnitude-based outlier detection
//!
//! An earlier draft of this module considered flagging tensors by "unusually
//! large" max-magnitude. Dropped: there is no principled threshold for
//! "unusual" without a calibration baseline (the same weight role varies
//! genuinely by layer depth and architecture), and inventing one would be
//! exactly the kind of "state a theory as a fact" `DESIGN.md` §6 and this
//! crate's whole behavior-signal design already argue against. NaN, Inf, and
//! exact-zero-variance are the only three conditions flagged here because
//! all three are unambiguous regardless of what a "normal" distribution for
//! that specific tensor looks like — every other one still needs a human
//! (or a future, evidence-backed baseline) to judge.
//!
//! # Why this reads glictus-caliburni's real manifest, not a second parser
//!
//! Unlike `quant_info.rs` (deliberately independent, JSON-only, no
//! `glictus-caliburni` dependency — see that module's own docs), this
//! module needs the actual tensor *bytes*, which only the real
//! `glictus-caliburni::package`/`manifest` reader and `glproc`'s dequant
//! kernels can decode correctly for `GQ4A`/`GQ2A`. Hand-rolling a second
//! decoder here would be exactly the "two independent implementations of
//! the same format" risk `architecture/mensura-veritatis-v3/ARTX2-Quant.md`
//! documents — so this is gated behind `gllm-bench` and reuses the real
//! reader, the same trade `ppl`/`kl-div` already made.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use glictus_caliburni::manifest::{DType, TensorEntry};
use glictus_caliburni::package::GllmPackage;
use glictus_caliburni::runtime::LayerMapping;

use crate::export::json::Json;

/// CLI arguments for `tensor-stats`.
pub struct TensorStatsArgs {
    pub model: PathBuf,
    pub out: Option<PathBuf>,
    /// Include a full per-tensor distribution (count/mean/std/min/max) in the
    /// output, not just the NaN/Inf/zero-variance issue list. Off by default:
    /// a real model has hundreds of tensors, and the existing fast scan (the
    /// module's original purpose) should stay cheap to read by default.
    ///
    /// [Vetted per `RESEARCH_REQUIREMENTS.md`'s 8 questions — "Weight
    /// distribution" candidate] 1. Problem: the mean/std/min/max were already
    /// computed internally by `compute_stats` to *detect* the three flagged
    /// conditions, then thrown away — a reader who wants to see the actual
    /// shape of a tensor's values (not just "is it broken") had no way to.
    /// 2. Who benefits: anyone comparing a converted tensor's distribution
    /// against the original checkpoint's, or eyeballing a specific layer's
    /// weight scale. 3/5/7. Same as the base scan: standard, reproducible,
    /// zero extra decode cost — reuses the pass already running. 6.
    /// Actionable: a mean/std wildly different from neighboring layers is a
    /// lead, even without a principled "abnormal" threshold (see the module's
    /// own "deliberately not included" note on why *flagging* by magnitude
    /// was rejected — *reporting* the number is not the same claim as
    /// *flagging* it). 8. Read-only, reports facts, no verdict.
    pub full: bool,
    /// Restrict the scan to normalization-layer weight tensors only (any
    /// tensor name containing `norm.weight` — `attn_norm.weight`,
    /// `ffn_norm.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`,
    /// `output_norm.weight`, verified against `glproc`'s and
    /// `glictus-caliburni`'s actual GGUF-derived naming). This is the
    /// **static weight** reading of the "RMSNorm analysis" candidate — see
    /// `RESEARCH_REQUIREMENTS.md` for why the **runtime/activation** reading
    /// (post-normalization statistics during a forward pass) was deferred
    /// instead: it needs new engine instrumentation this module deliberately
    /// does not have, while the gamma weights themselves are already exactly
    /// what this module decodes for every other tensor.
    pub norm_only: bool,
}

/// One unambiguous structural anomaly found in a tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorIssue {
    pub tensor_name: String,
    pub dtype: String,
    pub reason: String,
    pub detail: String,
}

/// One tensor's full value distribution — only populated when `--full` is
/// requested (see [`TensorStatsArgs::full`]'s doc comment for why this is
/// opt-in rather than always collected).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDistribution {
    pub tensor_name: String,
    pub dtype: String,
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

/// The JSON-serializable summary of one `tensor-stats` scan.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorStatsSession {
    pub model_path: String,
    pub tensors_scanned: usize,
    pub tensors_skipped: usize,
    pub skipped_dtypes: BTreeMap<String, usize>,
    pub issues: Vec<TensorIssue>,
    /// Empty unless `--full` was passed — see [`TensorStatsArgs::full`].
    pub distributions: Vec<TensorDistribution>,
    pub timestamp: String,
}

impl TensorStatsSession {
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    pub fn to_json(&self) -> Json {
        let skipped = self
            .skipped_dtypes
            .iter()
            .map(|(k, v)| (k.clone(), Json::n(*v as f64)))
            .collect::<BTreeMap<_, _>>();
        let issues = self
            .issues
            .iter()
            .map(|i| {
                Json::obj([
                    ("tensor_name", Json::s(i.tensor_name.clone())),
                    ("dtype", Json::s(i.dtype.clone())),
                    ("reason", Json::s(i.reason.clone())),
                    ("detail", Json::s(i.detail.clone())),
                ])
            })
            .collect();
        let distributions = self
            .distributions
            .iter()
            .map(|d| {
                Json::obj([
                    ("tensor_name", Json::s(d.tensor_name.clone())),
                    ("dtype", Json::s(d.dtype.clone())),
                    ("count", Json::n(d.count as f64)),
                    ("mean", Json::n(d.mean)),
                    ("std_dev", Json::n(d.std_dev)),
                    ("min", Json::n(d.min)),
                    ("max", Json::n(d.max)),
                ])
            })
            .collect();
        Json::obj([
            ("model_path", Json::s(self.model_path.clone())),
            ("tensors_scanned", Json::n(self.tensors_scanned as f64)),
            ("tensors_skipped", Json::n(self.tensors_skipped as f64)),
            ("skipped_dtypes", Json::Obj(skipped)),
            ("issues", Json::Arr(issues)),
            ("distributions", Json::Arr(distributions)),
            ("timestamp", Json::s(self.timestamp.clone())),
        ])
    }

    pub fn from_json(v: &Json) -> Result<TensorStatsSession, String> {
        let skipped_dtypes = v
            .get("skipped_dtypes")
            .and_then(|t| t.as_obj())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0) as usize)).collect())
            .unwrap_or_default();
        let issues = v
            .get("issues")
            .and_then(|a| a.as_arr())
            .map(|a| {
                a.iter()
                    .map(|i| TensorIssue {
                        tensor_name: i.get("tensor_name").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                        dtype: i.get("dtype").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                        reason: i.get("reason").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                        detail: i.get("detail").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let distributions = v
            .get("distributions")
            .and_then(|a| a.as_arr())
            .map(|a| {
                a.iter()
                    .map(|d| TensorDistribution {
                        tensor_name: d.get("tensor_name").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                        dtype: d.get("dtype").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                        count: d.get("count").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
                        mean: d.get("mean").and_then(|n| n.as_f64()).unwrap_or(0.0),
                        std_dev: d.get("std_dev").and_then(|n| n.as_f64()).unwrap_or(0.0),
                        min: d.get("min").and_then(|n| n.as_f64()).unwrap_or(0.0),
                        max: d.get("max").and_then(|n| n.as_f64()).unwrap_or(0.0),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(TensorStatsSession {
            model_path: v.get("model_path").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            tensors_scanned: v.get("tensors_scanned").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            tensors_skipped: v.get("tensors_skipped").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            skipped_dtypes,
            issues,
            distributions,
            timestamp: v.get("timestamp").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        })
    }
}

/// Per-tensor value facts, computed in one pass. Not part of the public
/// session (would be hundreds of entries for a real model) — only feeds
/// [`TensorIssue`] detection.
struct ValueStats {
    count: usize,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    nan_count: usize,
    inf_count: usize,
}

fn compute_stats(values: &[f32]) -> ValueStats {
    let mut sum = 0f64;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut finite_count = 0usize;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for &v in values {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        sum += v as f64;
        finite_count += 1;
        min = min.min(v as f64);
        max = max.max(v as f64);
    }

    let mean = if finite_count > 0 { sum / finite_count as f64 } else { 0.0 };
    let mut var_sum = 0f64;
    for &v in values {
        if v.is_finite() {
            let d = v as f64 - mean;
            var_sum += d * d;
        }
    }
    let std_dev = if finite_count > 0 { (var_sum / finite_count as f64).sqrt() } else { 0.0 };

    ValueStats {
        count: values.len(),
        mean,
        std_dev,
        // No finite values at all (e.g. an all-NaN tensor): report 0 rather
        // than the +-infinity sentinels, matching `mean`'s same fallback.
        min: if finite_count > 0 { min } else { 0.0 },
        max: if finite_count > 0 { max } else { 0.0 },
        nan_count,
        inf_count,
    }
}

/// Decode one tensor to f32, the same dispatch
/// `GllmEngine::load_shared`/`GlprocBackend::required_tensor` use. `None`
/// for a dtype this Wave-1-scope decoder cannot read (native GGUF quant
/// dtypes stored as-is — see `architecture/mensura-veritatis-v3/ARTX3-Format.md`).
fn decode(dtype: DType, shape: &[u64], bytes: &[u8]) -> Option<Vec<f32>> {
    match dtype {
        DType::GQ4A => Some(glproc::kernels::gquant::dequant_gq4a_stream(bytes)),
        DType::GQ2A => Some(glproc::kernels::gquant::dequant_gq2a_stream(bytes)),
        DType::F32 | DType::F16 | DType::Bf16 => {
            let label = dtype_label(dtype);
            glcore::format::decode_tensor("tensor", shape, label, bytes).ok().map(|t| t.data)
        }
        _ => None,
    }
}

/// The subset of dtype labels [`decode`] actually passes to
/// `glcore::format::decode_tensor` — only F32/F16/BF16 need a string label
/// here (GQ4A/GQ2A go through glproc directly, everything else is skipped).
fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::Bf16 => "BF16",
        _ => "UNKNOWN",
    }
}

fn dtype_display(dtype: DType) -> String {
    format!("{dtype:?}")
}

/// Run `glbench tensor-stats --model <package-dir> [--out <file.json>]`.
pub fn run_tensor_stats(args: TensorStatsArgs) -> Result<(), String> {
    eprintln!("opening .gllm package from {} ...", args.model.display());
    let package = GllmPackage::open(&args.model).map_err(|e| format!("opening {}: {e}", args.model.display()))?;
    let manifest = package.manifest();

    let mut all: Vec<(TensorEntry, PathBuf, Option<u32>)> = Vec::new();
    for t in &manifest.shared.tensors {
        all.push((t.clone(), package.layout.shared_path.clone(), None));
    }
    for (idx, layer) in manifest.layers.iter().enumerate() {
        let Some(layer_file) = package.layout.layer_path(idx as u32) else {
            continue;
        };
        for t in &layer.tensors {
            all.push((t.clone(), layer_file.path.clone(), Some(idx as u32)));
        }
    }

    if args.norm_only {
        all.retain(|(entry, _, _)| entry.name.contains("norm.weight"));
    }

    let mut tensors_scanned = 0usize;
    let mut tensors_skipped = 0usize;
    let mut skipped_dtypes: BTreeMap<String, usize> = BTreeMap::new();
    let mut issues = Vec::new();
    let mut distributions = Vec::new();
    let mut mapping_cache: BTreeMap<PathBuf, LayerMapping> = BTreeMap::new();

    for (entry, path, layer_idx) in &all {
        if !mapping_cache.contains_key(path) {
            let m = LayerMapping::open(path, *layer_idx, false)
                .map_err(|e| format!("opening {}: {e}", path.display()))?;
            mapping_cache.insert(path.clone(), m);
        }
        let mapping = mapping_cache.get(path).expect("just inserted");
        let Some(bytes) = mapping.tensor_bytes(&entry.name) else {
            return Err(format!("{}: tensor {:?} has no data in {}", args.model.display(), entry.name, path.display()));
        };

        let Some(values) = decode(entry.dtype, &entry.shape, bytes) else {
            tensors_skipped += 1;
            *skipped_dtypes.entry(dtype_display(entry.dtype)).or_insert(0) += 1;
            continue;
        };
        tensors_scanned += 1;

        let stats = compute_stats(&values);
        let dtype_str = dtype_display(entry.dtype);

        if args.full {
            distributions.push(TensorDistribution {
                tensor_name: entry.name.clone(),
                dtype: dtype_str.clone(),
                count: stats.count,
                mean: stats.mean,
                std_dev: stats.std_dev,
                min: stats.min,
                max: stats.max,
            });
        }

        if stats.nan_count > 0 {
            issues.push(TensorIssue {
                tensor_name: entry.name.clone(),
                dtype: dtype_str.clone(),
                reason: "NaN values present".to_string(),
                detail: format!("{} NaN out of {} elements", stats.nan_count, stats.count),
            });
        }
        if stats.inf_count > 0 {
            issues.push(TensorIssue {
                tensor_name: entry.name.clone(),
                dtype: dtype_str.clone(),
                reason: "Inf values present".to_string(),
                detail: format!("{} Inf out of {} elements", stats.inf_count, stats.count),
            });
        }
        if stats.nan_count == 0 && stats.inf_count == 0 && stats.count > 1 && stats.std_dev == 0.0 {
            issues.push(TensorIssue {
                tensor_name: entry.name.clone(),
                dtype: dtype_str,
                reason: "zero variance (degenerate)".to_string(),
                detail: format!("all {} elements equal {:.6}", stats.count, stats.mean),
            });
        }
    }

    let session = TensorStatsSession {
        model_path: args.model.display().to_string(),
        tensors_scanned,
        tensors_skipped,
        skipped_dtypes,
        issues,
        distributions,
        timestamp: iso8601_now(),
    };

    print!("{}", render_table(&session));

    if let Some(out) = &args.out {
        std::fs::write(out, session.to_json().to_pretty()).map_err(|e| format!("writing {}: {e}", out.display()))?;
        eprintln!("wrote {}", out.display());
    }

    Ok(())
}

fn render_table(s: &TensorStatsSession) -> String {
    let mut out = String::new();
    let rule = "\u{2500}".repeat(52);

    out.push_str(&format!("glbench tensor-stats: {}\n", s.model_path));
    out.push_str(&format!("{rule}\n"));
    out.push_str(&format!("Tensors scanned   {}\n", s.tensors_scanned));
    out.push_str(&format!("Tensors skipped   {}\n", s.tensors_skipped));
    for (dtype, count) in &s.skipped_dtypes {
        out.push_str(&format!("  {dtype:<14} {count} (not decodable in Wave 1 scope)\n"));
    }
    out.push_str(&format!("{rule}\n"));
    if s.is_clean() {
        out.push_str("Result            CLEAN — no NaN/Inf/degenerate tensors found\n");
    } else {
        out.push_str(&format!("Result            {} issue(s) found\n", s.issues.len()));
        for issue in &s.issues {
            out.push_str(&format!("  [{}] {} ({}) — {}\n", issue.reason, issue.tensor_name, issue.dtype, issue.detail));
        }
    }
    out.push_str(&format!("{rule}\n"));

    if !s.distributions.is_empty() {
        out.push_str(&format!("Distribution ({} tensors)\n", s.distributions.len()));
        for d in &s.distributions {
            out.push_str(&format!(
                "  {:<40} n={:<9} mean={:>10.5} std={:>10.5} min={:>10.5} max={:>10.5}\n",
                d.tensor_name, d.count, d.mean, d.std_dev, d.min, d.max
            ));
        }
        out.push_str(&format!("{rule}\n"));
    }
    out
}

/// Manual ISO 8601 UTC timestamp (no `chrono` — glbench's zero-dep rule,
/// same implementation as `ppl.rs`'s and `quant_info.rs`'s).
fn iso8601_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    civil_from_unix(secs)
}

fn civil_from_unix(unix_secs: u64) -> String {
    let days = (unix_secs / 86400) as i64;
    let secs_of_day = unix_secs % 86400;
    let (hour, minute, second) = (secs_of_day / 3600, (secs_of_day % 3600) / 60, secs_of_day % 60);

    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_stats_clean_data_has_no_nan_or_inf() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let s = compute_stats(&values);
        assert_eq!(s.nan_count, 0);
        assert_eq!(s.inf_count, 0);
        assert!((s.mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn compute_stats_detects_nan_and_inf_separately() {
        let values = vec![1.0f32, f32::NAN, 2.0, f32::INFINITY, 3.0];
        let s = compute_stats(&values);
        assert_eq!(s.nan_count, 1);
        assert_eq!(s.inf_count, 1);
        assert_eq!(s.count, 5);
        // mean/min/max computed only from the 3 finite values (1, 2, 3).
        assert!((s.mean - 2.0).abs() < 1e-6);
    }

    #[test]
    fn compute_stats_zero_variance_for_constant_tensor() {
        let values = vec![0.5f32; 100];
        let s = compute_stats(&values);
        assert_eq!(s.std_dev, 0.0);
        assert!((s.mean - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compute_stats_min_max_ignore_nan_and_inf() {
        let values = vec![5.0f32, -3.0, f32::NAN, f32::INFINITY, 1.0];
        let s = compute_stats(&values);
        assert_eq!(s.min, -3.0);
        assert_eq!(s.max, 5.0);
    }

    #[test]
    fn compute_stats_all_nan_reports_zero_not_infinity_sentinels() {
        let values = vec![f32::NAN, f32::NAN];
        let s = compute_stats(&values);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
    }

    #[test]
    fn tensor_stats_session_serializes() {
        let mut skipped_dtypes = BTreeMap::new();
        skipped_dtypes.insert("Q5_0".to_string(), 133);
        let session = TensorStatsSession {
            model_path: "model.gllm".into(),
            tensors_scanned: 158,
            tensors_skipped: 133,
            skipped_dtypes,
            issues: vec![TensorIssue {
                tensor_name: "blk.5.ffn_down.weight".into(),
                dtype: "F32".into(),
                reason: "NaN values present".into(),
                detail: "3 NaN out of 4358144 elements".into(),
            }],
            distributions: vec![TensorDistribution {
                tensor_name: "blk.0.attn_norm.weight".into(),
                dtype: "F32".into(),
                count: 896,
                mean: 1.02,
                std_dev: 0.15,
                min: 0.4,
                max: 1.8,
            }],
            timestamp: "2026-07-24T00:00:00Z".into(),
        };
        let back = TensorStatsSession::from_json(&session.to_json()).unwrap();
        assert_eq!(session, back);
    }

    #[test]
    fn is_clean_reflects_empty_issues() {
        let session = TensorStatsSession {
            model_path: "m".into(),
            tensors_scanned: 10,
            tensors_skipped: 0,
            skipped_dtypes: BTreeMap::new(),
            issues: vec![],
            distributions: vec![],
            timestamp: "2026-07-24T00:00:00Z".into(),
        };
        assert!(session.is_clean());
    }

    #[test]
    fn norm_only_filter_matches_all_known_norm_tensor_names() {
        // Verified against glproc's actual GGUF-derived loader names
        // (loader.rs: attn_norm, attn_q_norm, attn_k_norm, ffn_norm,
        // output_norm) — every one must contain "norm.weight".
        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
        ] {
            assert!(name.contains("norm.weight"), "{name} should match the norm filter");
        }
        for name in ["blk.0.attn_q.weight", "blk.0.ffn_down.weight", "token_embd.weight"] {
            assert!(!name.contains("norm.weight"), "{name} should NOT match the norm filter");
        }
    }

    #[test]
    fn iso8601_epoch_is_1970() {
        assert_eq!(civil_from_unix(0), "1970-01-01T00:00:00Z");
    }
}
