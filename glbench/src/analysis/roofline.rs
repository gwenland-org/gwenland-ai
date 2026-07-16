//! Roofline analysis: which side of the roofline is each part of the model on?
//!
//! Two layers of answer live here:
//!
//! - [`Roofline`] — the classic single-number form: arithmetic intensity vs
//!   the machine's ridge point. For a token-decode workload streaming `W`
//!   bytes of weights and doing ~`2·W/bpw` FLOPs against them, intensity is
//!   well under 1 FLOP/byte, firmly on the bandwidth-bound side.
//! - [`RooflineReport`] — the v2 per-bucket form, computed from the engine's
//!   own stage telemetry. Stages are grouped into **Attention / FFN / lm_head**
//!   buckets and each bucket gets its achieved fraction of the *measured*
//!   bandwidth ceiling plus its arithmetic intensity. This is what turns
//!   "decode is memory-bound" into "FFN is at 85% of ceiling but attention is
//!   at 29% and therefore NOT bandwidth-bound" — the distinction whose
//!   confusion cost the native-Q4_K experiment 33%.
//!
//! Verdicts here are classifications of measured numbers, not actions. The
//! thresholds mirror the ones the telemetry renderer already prints (`!` under
//! 25%): a bucket near the ceiling cannot be sped up without reading fewer
//! bytes; a bucket far below it will not benefit from reading fewer bytes.

use glcore::telemetry::{PhaseProfile, StageTiming};

/// The ridge point of a roofline and where a workload sits relative to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roofline {
    /// Arithmetic intensity of the workload, FLOP per byte.
    pub intensity_flop_per_byte: f64,
    /// The machine balance (ridge point): peak_compute / peak_bandwidth,
    /// FLOP per byte. Below this the workload is bandwidth-bound.
    pub ridge_flop_per_byte: f64,
}

impl Roofline {
    /// True if the workload is on the memory-bound side of the ridge.
    pub fn is_memory_bound(&self) -> bool {
        self.intensity_flop_per_byte < self.ridge_flop_per_byte
    }

    /// Build from peak compute (FLOP/s) and peak bandwidth (bytes/s) plus the
    /// workload's FLOP and byte counts.
    pub fn new(
        workload_flops: f64,
        workload_bytes: f64,
        peak_flops: f64,
        peak_bytes_per_s: f64,
    ) -> Option<Roofline> {
        if workload_bytes <= 0.0 || peak_bytes_per_s <= 0.0 {
            return None;
        }
        Some(Roofline {
            intensity_flop_per_byte: workload_flops / workload_bytes,
            ridge_flop_per_byte: peak_flops / peak_bytes_per_s,
        })
    }
}

/// A bucket is bandwidth-bound above this fraction of the measured ceiling.
pub const BANDWIDTH_BOUND_FRAC: f64 = 0.70;

/// Below this fraction a bucket is definitively NOT bandwidth-bound — the same
/// 25% line the telemetry table flags with `!`.
pub const NOT_BANDWIDTH_BOUND_FRAC: f64 = 0.25;

/// Which part of the model a stage belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    /// QKV projection, attention core, output projection.
    Attention,
    /// Gate/up/down feed-forward matmuls (dense or MoE).
    Ffn,
    /// The vocabulary projection.
    LmHead,
    /// Everything the mapping does not recognize (sampler, serial sections).
    Other,
}

impl Bucket {
    /// Stable lowercase identifier for archives and rendering.
    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::Attention => "attention",
            Bucket::Ffn => "ffn",
            Bucket::LmHead => "lm_head",
            Bucket::Other => "other",
        }
    }
}

/// Where a bucket sits relative to the bandwidth ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketVerdict {
    /// ≥ [`BANDWIDTH_BOUND_FRAC`] of ceiling: streaming as fast as DRAM
    /// allows; only reading fewer bytes can speed it up.
    BandwidthBound,
    /// < [`NOT_BANDWIDTH_BOUND_FRAC`] of ceiling: stalled on something other
    /// than bandwidth (compute, latency, serial section).
    NotBandwidthBound,
    /// Between the two lines — the data does not justify either claim.
    Indeterminate,
    /// No bytes attribution or no ceiling: cannot classify, and saying so
    /// beats inventing a verdict.
    Unknown,
}

impl BucketVerdict {
    /// Stable identifier for archives and rendering.
    pub fn as_str(self) -> &'static str {
        match self {
            BucketVerdict::BandwidthBound => "bandwidth-bound",
            BucketVerdict::NotBandwidthBound => "not-bandwidth-bound",
            BucketVerdict::Indeterminate => "indeterminate",
            BucketVerdict::Unknown => "unknown",
        }
    }
}

/// One bucket's roofline facts and its classification.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketRoofline {
    /// Which part of the model.
    pub bucket: Bucket,
    /// Wall time summed over the bucket's stages, ms.
    pub total_ms: f64,
    /// Share of the phase total, if the phase recorded one.
    pub share: Option<f64>,
    /// Achieved read bandwidth, GB/s (needs bytes attribution).
    pub gb_per_s: Option<f64>,
    /// Fraction of the bandwidth ceiling reached (needs a ceiling too).
    pub ceiling_frac: Option<f64>,
    /// Arithmetic intensity, FLOP/byte (2 FLOPs per MAC; needs both counters).
    pub intensity_flop_per_byte: Option<f64>,
    /// The classification derived from `ceiling_frac`.
    pub verdict: BucketVerdict,
}

/// Per-bucket roofline for one phase, plus the ceiling it was judged against.
#[derive(Debug, Clone, PartialEq)]
pub struct RooflineReport {
    /// The bandwidth ceiling used, GB/s. On CPU this is the *measured* read
    /// ceiling; on GPU a published spec. `None` when neither exists, in which
    /// case every verdict is `Unknown`.
    pub ceiling_gbs: Option<f64>,
    /// Decode-phase buckets (present when the engine reported decode stages).
    pub decode: Vec<BucketRoofline>,
    /// Prefill-phase buckets.
    pub prefill: Vec<BucketRoofline>,
}

impl RooflineReport {
    /// Build from engine telemetry and the machine's bandwidth ceiling.
    /// Returns `None` when the engine reported no stage telemetry at all —
    /// no data is not the same as an empty report.
    pub fn compute(
        t: &glcore::telemetry::EngineTelemetry,
        ceiling_gbs: Option<f64>,
    ) -> Option<RooflineReport> {
        let decode = t.decode.as_ref().map(|p| bucketize(p, ceiling_gbs)).unwrap_or_default();
        let prefill = t.prefill.as_ref().map(|p| bucketize(p, ceiling_gbs)).unwrap_or_default();
        if decode.is_empty() && prefill.is_empty() {
            return None;
        }
        Some(RooflineReport { ceiling_gbs, decode, prefill })
    }
}

/// Map a stage name onto its bucket.
///
/// Names are engine-chosen and glbench must not force a vocabulary
/// (`glcore::telemetry` docs), so this is a heuristic over the conventions the
/// existing engines use (`qkv`/`attention`/`attn_out`/`fixup`,
/// `ffn_gate_up`/`ffn_down[q]`, `lm_head`). Unrecognized names land in
/// `Other` rather than being guessed into a model bucket.
pub fn bucket_of(stage_name: &str) -> Bucket {
    let n = stage_name.to_ascii_lowercase();
    if n.contains("lm_head") || n.contains("logits") {
        Bucket::LmHead
    } else if n.contains("attn") || n.contains("attention") || n == "qkv" || n == "fixup" {
        Bucket::Attention
    } else if n.contains("ffn") || n.contains("mlp") || n.contains("gate") || n.contains("moe") {
        Bucket::Ffn
    } else {
        Bucket::Other
    }
}

/// Group a phase's stages into buckets and classify each against the ceiling.
fn bucketize(phase: &PhaseProfile, ceiling_gbs: Option<f64>) -> Vec<BucketRoofline> {
    // Accumulate raw counters per bucket. Bytes/macs stay None-aware: a bucket
    // where any stage lacks attribution keeps what the others reported (the
    // sum is still a fact about the attributed stages), but a bucket with no
    // attribution at all reports None.
    struct Acc {
        total_ms: f64,
        bytes: Option<u64>,
        macs: Option<u64>,
        seen: bool,
    }
    let mut accs: [(Bucket, Acc); 4] = [
        (Bucket::Attention, Acc { total_ms: 0.0, bytes: None, macs: None, seen: false }),
        (Bucket::Ffn, Acc { total_ms: 0.0, bytes: None, macs: None, seen: false }),
        (Bucket::LmHead, Acc { total_ms: 0.0, bytes: None, macs: None, seen: false }),
        (Bucket::Other, Acc { total_ms: 0.0, bytes: None, macs: None, seen: false }),
    ];

    for st in &phase.stages {
        let b = bucket_of(&st.name);
        let acc = &mut accs.iter_mut().find(|(bb, _)| *bb == b).unwrap().1;
        acc.seen = true;
        acc.total_ms += st.total_ms;
        if let Some(by) = st.bytes_read {
            *acc.bytes.get_or_insert(0) += by;
        }
        if let Some(mc) = st.macs {
            *acc.macs.get_or_insert(0) += mc;
        }
    }

    accs.into_iter()
        .filter(|(_, a)| a.seen)
        .map(|(bucket, a)| {
            // Reuse StageTiming's arithmetic for the derived rates so bucket
            // math cannot drift from stage math.
            let as_stage = StageTiming {
                name: bucket.as_str().to_string(),
                total_ms: a.total_ms,
                calls: 1,
                bytes_read: a.bytes,
                macs: a.macs,
            };
            let gb_per_s = as_stage.gb_per_s();
            let ceiling_frac = ceiling_gbs.and_then(|c| as_stage.ceiling_frac(c));
            let intensity = match (a.macs, a.bytes) {
                (Some(m), Some(b)) if b > 0 => Some(2.0 * m as f64 / b as f64),
                _ => None,
            };
            let verdict = match ceiling_frac {
                Some(f) if f >= BANDWIDTH_BOUND_FRAC => BucketVerdict::BandwidthBound,
                Some(f) if f < NOT_BANDWIDTH_BOUND_FRAC => BucketVerdict::NotBandwidthBound,
                Some(_) => BucketVerdict::Indeterminate,
                None => BucketVerdict::Unknown,
            };
            BucketRoofline {
                bucket,
                total_ms: a.total_ms,
                share: as_stage.share_of(phase.total_ms),
                gb_per_s,
                ceiling_frac,
                intensity_flop_per_byte: intensity,
                verdict,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_intensity_is_memory_bound() {
        // decode: ~0.25 FLOP/byte, ridge at ~140 (T4 65 TFLOP / 320 GB/s).
        let r = Roofline::new(1.0, 4.0, 65e12, 320e9).unwrap();
        assert!(r.is_memory_bound());
    }

    fn stage(name: &str, ms: f64, bytes: Option<u64>, macs: Option<u64>) -> StageTiming {
        StageTiming { name: name.into(), total_ms: ms, calls: 10, bytes_read: bytes, macs }
    }

    fn phase(stages: Vec<StageTiming>) -> PhaseProfile {
        let total = stages.iter().map(|s| s.total_ms).sum::<f64>() + 5.0;
        PhaseProfile { stages, total_ms: total }
    }

    #[test]
    fn stage_names_map_to_the_glproc_buckets() {
        assert_eq!(bucket_of("qkv"), Bucket::Attention);
        assert_eq!(bucket_of("attention"), Bucket::Attention);
        assert_eq!(bucket_of("attn_out"), Bucket::Attention);
        assert_eq!(bucket_of("fixup"), Bucket::Attention);
        assert_eq!(bucket_of("ffn_gate_up"), Bucket::Ffn);
        assert_eq!(bucket_of("ffn_down"), Bucket::Ffn);
        assert_eq!(bucket_of("ffn_downq"), Bucket::Ffn);
        assert_eq!(bucket_of("lm_head"), Bucket::LmHead);
        assert_eq!(bucket_of("sampler"), Bucket::Other);
        assert_eq!(bucket_of("serial"), Bucket::Other);
    }

    #[test]
    fn ffn_near_ceiling_and_attention_far_below_get_opposite_verdicts() {
        // Ceiling 30 GB/s. FFN: 27 GB/s (90%) -> bandwidth-bound.
        // Attention: 6 GB/s (20%) -> NOT bandwidth-bound. The v2 headline case.
        let p = phase(vec![
            // 100 ms at 27 GB/s = 2.7 GB read.
            stage("ffn_gate_up", 100.0, Some(2_700_000_000), Some(1_000_000_000)),
            // 100 ms at 6 GB/s = 0.6 GB read.
            stage("attention", 100.0, Some(600_000_000), Some(500_000_000)),
        ]);
        let buckets = bucketize(&p, Some(30.0));
        let ffn = buckets.iter().find(|b| b.bucket == Bucket::Ffn).unwrap();
        let attn = buckets.iter().find(|b| b.bucket == Bucket::Attention).unwrap();
        assert_eq!(ffn.verdict, BucketVerdict::BandwidthBound);
        assert_eq!(attn.verdict, BucketVerdict::NotBandwidthBound);
        assert!((ffn.ceiling_frac.unwrap() - 0.9).abs() < 1e-6);
        assert!((attn.ceiling_frac.unwrap() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn attention_stages_are_summed_into_one_bucket() {
        let p = phase(vec![
            stage("qkv", 10.0, Some(1_000_000), Some(500_000)),
            stage("attention", 20.0, Some(2_000_000), Some(1_000_000)),
            stage("attn_out", 5.0, Some(500_000), None),
        ]);
        let buckets = bucketize(&p, None);
        let attn = buckets.iter().find(|b| b.bucket == Bucket::Attention).unwrap();
        assert!((attn.total_ms - 35.0).abs() < 1e-9);
        // Bytes sum across all three; macs across the two that reported them.
        assert!((attn.gb_per_s.unwrap() - 3_500_000.0 / 0.035 / 1e9).abs() < 1e-9);
        // No ceiling -> Unknown, never a guessed verdict.
        assert_eq!(attn.verdict, BucketVerdict::Unknown);
    }

    #[test]
    fn intensity_is_two_flops_per_mac_over_bytes() {
        let p = phase(vec![stage("ffn_down", 10.0, Some(1000), Some(2000))]);
        let b = &bucketize(&p, None)[0];
        assert!((b.intensity_flop_per_byte.unwrap() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn no_telemetry_yields_no_report() {
        let t = glcore::telemetry::EngineTelemetry::default();
        assert!(RooflineReport::compute(&t, Some(30.0)).is_none());
    }
}
