//! Shared telemetry vocabulary: what an engine reports about *itself*.
//!
//! This module exists so glbench can read rich per-stage facts out of any
//! backend without naming a concrete engine type, and without any backend
//! knowing glbench exists.
//!
//! # Why data, not a callback trait
//!
//! The obvious design is a `BenchmarkReporter` trait with `begin_stage` /
//! `end_stage` that the engine calls. It is rejected on purpose: it inverts
//! the dependency (the engine would import harness types) and it hands the
//! harness a hook to inject behavior into the hot path. glbench's own DESIGN.md
//! draws that line — *"glbench observes performance; it does not optimize it"*
//! — and a callback seam is exactly how that line erodes.
//!
//! So telemetry flows **outward as plain data**:
//!
//! ```text
//!   engine (glproc, glcuda, ...)  --fills-->  EngineTelemetry
//!                                                   |
//!                                                   v
//!                                   glbench reads it, renders it, archives it
//! ```
//!
//! The engine owns its counters, decides when they are cheap enough to keep,
//! and hands over a snapshot on request. It never learns who is asking.
//!
//! # Cost when disabled
//!
//! Every field is optional and every accessor returns `Option`. An engine with
//! profiling off returns `None` and pays nothing — no branch in the hot loop,
//! no allocation. glproc gates its collection behind `GLPROC_PROFILE` and
//! stores the counters in a `Box`, so the disabled path is a null-pointer
//! check per layer, not per operation.

/// One named stage of the forward pass, and how long it took.
///
/// Stages are whatever the engine finds meaningful to separate — glproc
/// reports `qkv`, `attn`, `wo`, `gateup`, `down`; a GPU backend might report
/// kernel launches instead. glbench does not interpret the names, it only
/// ranks them by share of total, so backends stay free to disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct StageTiming {
    /// Stage name, e.g. `"attention"`. Stable across runs of the same engine.
    pub name: String,
    /// Accumulated wall-clock time in this stage, milliseconds.
    pub total_ms: f64,
    /// How many times the stage ran (layers x tokens, typically).
    pub calls: u64,

    /// Bytes this stage read, in total, across all `calls`. `None` when the
    /// engine cannot attribute traffic to the stage.
    ///
    /// Together with `total_ms` this yields the stage's achieved GB/s — and
    /// therefore what fraction of the machine's bandwidth ceiling it reached.
    /// A bandwidth-bound stage should sit near the ceiling; one far below it is
    /// stalled on something else, and that gap is the signal.
    pub bytes_read: Option<u64>,

    /// Multiply-accumulate operations this stage performed, in total.
    ///
    /// **This is the metric that actually diagnoses a kernel.** GB/s alone
    /// cannot compare formats — Q4_K and Q8_0 move different bytes per MAC, so
    /// a slower kernel can look "efficient" simply by reading less. GMAC/s is
    /// format-independent, and it is what exposed both real kernel bugs found
    /// so far:
    ///
    /// - attention ran at **0.83 GMAC/s** while `qkv` on the same machine hit
    ///   **18.1** — a 22x gap that `share%` showed as an unremarkable 14.9%.
    /// - a native Q4_K kernel hit **1.5–2.0 GMAC/s** against Q8_0's **3.3**,
    ///   which is why it lost 33% end-to-end despite reading 1.89x fewer bytes.
    ///
    /// Neither was visible in wall-time share. Both were obvious in GMAC/s.
    pub macs: Option<u64>,
}

impl StageTiming {
    /// Share of `total_ms` across a whole phase, as a fraction in `[0, 1]`.
    /// Returns `None` when the denominator is zero — a phase that never ran
    /// has no meaningful breakdown, and reporting `0%` would imply it did.
    pub fn share_of(&self, phase_total_ms: f64) -> Option<f64> {
        (phase_total_ms > 0.0).then(|| self.total_ms / phase_total_ms)
    }

    /// Achieved read bandwidth, GB/s. `None` when the engine did not attribute
    /// bytes to this stage, or the stage took no measurable time.
    pub fn gb_per_s(&self) -> Option<f64> {
        let bytes = self.bytes_read? as f64;
        (self.total_ms > 0.0).then(|| bytes / (self.total_ms / 1e3) / 1e9)
    }

    /// Achieved compute throughput, GMAC/s. See [`StageTiming::macs`] for why
    /// this is the number to look at rather than GB/s.
    pub fn gmac_per_s(&self) -> Option<f64> {
        let macs = self.macs? as f64;
        (self.total_ms > 0.0).then(|| macs / (self.total_ms / 1e3) / 1e9)
    }

    /// Fraction of the machine's bandwidth ceiling this stage reached, `[0, 1]`.
    ///
    /// A stage near 1.0 is bandwidth-bound and cannot be made faster without
    /// reading fewer bytes. A stage far below it is bound by something else —
    /// compute, latency, or a serial section — and reading fewer bytes will not
    /// help it. **Confusing these two is how the Q4_K experiment lost 33%.**
    pub fn ceiling_frac(&self, ceiling_gbs: f64) -> Option<f64> {
        (ceiling_gbs > 0.0).then(|| self.gb_per_s())?.map(|gbs| gbs / ceiling_gbs)
    }
}

/// Per-stage timing for one phase of inference.
///
/// Prefill and decode are kept apart because they are different workloads with
/// different bottlenecks — prefill is compute-bound and batched, decode is
/// bandwidth-bound and serial. Summing them produces a number that describes
/// neither.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PhaseProfile {
    /// Stages, in the order the engine reports them (execution order).
    pub stages: Vec<StageTiming>,
    /// Total wall time attributed to this phase, milliseconds. May exceed the
    /// sum of `stages` — the remainder is unattributed overhead, and glbench
    /// shows it rather than hiding it by normalizing the stages to 100%.
    pub total_ms: f64,
}

impl PhaseProfile {
    /// Time not accounted for by any named stage, in milliseconds.
    ///
    /// A large residual means the engine's instrumentation has a blind spot.
    /// That is a fact worth surfacing, not an error to paper over, so it is
    /// computed rather than assumed to be zero.
    pub fn unattributed_ms(&self) -> f64 {
        let named: f64 = self.stages.iter().map(|s| s.total_ms).sum();
        (self.total_ms - named).max(0.0)
    }

    /// Stages sorted by time descending — the hotspot ranking.
    pub fn hotspots(&self) -> Vec<&StageTiming> {
        let mut v: Vec<&StageTiming> = self.stages.iter().collect();
        v.sort_by(|a, b| b.total_ms.partial_cmp(&a.total_ms).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

/// Mixture-of-Experts routing behavior, accumulated over a run.
///
/// Load balance is the number that matters: MoE's entire speed argument is
/// that only `top_k` of `num_experts` run per token. If routing collapses onto
/// a few experts, the model still produces correct output while quietly losing
/// the benefit — a failure that is invisible without this data.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeTelemetry {
    /// Experts held per MoE layer.
    pub num_experts: usize,
    /// Experts each token is routed through (top-k).
    pub num_experts_per_tok: usize,
    /// Tokens routed to each expert, summed over every MoE layer and token.
    /// Length is `num_experts`.
    pub expert_load: Vec<u64>,
    /// MoE layers in the model (vs dense layers).
    pub moe_layers: usize,
}

impl MoeTelemetry {
    /// Experts that received at least one token.
    pub fn experts_touched(&self) -> usize {
        self.expert_load.iter().filter(|&&c| c > 0).count()
    }

    /// Min / max / mean tokens per expert, over experts that were touched.
    /// Returns `None` if nothing was routed (no MoE layer ran).
    pub fn load_balance(&self) -> Option<(u64, u64, f64)> {
        let live: Vec<u64> = self.expert_load.iter().copied().filter(|&c| c > 0).collect();
        if live.is_empty() {
            return None;
        }
        let min = *live.iter().min().unwrap();
        let max = *live.iter().max().unwrap();
        let mean = live.iter().sum::<u64>() as f64 / live.len() as f64;
        Some((min, max, mean))
    }

    /// Normalized routing entropy in `[0, 1]`: 1.0 means tokens spread evenly
    /// over all experts, 0.0 means every token hit one expert.
    ///
    /// This is the single number that says whether routing is healthy. A
    /// collapsing router shows up here long before it shows up in output
    /// quality, and it is the metric a load-balancing loss is trying to move.
    pub fn routing_entropy(&self) -> Option<f64> {
        let total: u64 = self.expert_load.iter().sum();
        if total == 0 || self.num_experts <= 1 {
            return None;
        }
        let h: f64 = self
            .expert_load
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.ln()
            })
            .sum();
        // Divide by ln(num_experts) — the entropy of a perfectly uniform
        // router — so the result is comparable across models with different
        // expert counts.
        Some(h / (self.num_experts as f64).ln())
    }
}

/// Where an engine's memory went.
///
/// Split by role rather than reported as one total: a 4 GB peak means
/// something very different if it is 3.8 GB of weights vs 3.8 GB of KV cache,
/// and only the second one grows with context length.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryTelemetry {
    /// Model weights resident in RAM, bytes.
    pub model_bytes: u64,
    /// KV cache allocation, bytes.
    pub kv_cache_bytes: u64,
    /// Per-run scratch/workspace buffers, bytes.
    pub scratch_bytes: u64,
}

/// What the engine actually chose to run on this machine.
///
/// Not what the CPU *supports* — what the engine *picked*. Those differ, and
/// the difference is often the whole story: glproc deliberately rejects AVX-512
/// on low-core parts because it downclocks, so a machine reporting `avx512f:
/// true` may still be running AVX2 kernels. Reporting only the CPU's
/// capabilities would make that invisible.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendTelemetry {
    /// SIMD backend actually selected, e.g. `"avx2"`, `"avx512"`, `"scalar"`.
    pub simd_path: String,
    /// Worker threads in the compute pool.
    pub threads: usize,
    /// Kernel path per weight class, e.g. `("ffn_gate_up", "Q8_0 integer-dot")`.
    /// Free-form: backends name their own kernels, glbench just displays them.
    pub kernels: Vec<(String, String)>,
}

/// Everything an engine chooses to report about a completed run.
///
/// All fields optional: an engine that collects nothing returns
/// `EngineTelemetry::default()` and glbench renders the sections it has.
/// Adding a field here never breaks a backend.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineTelemetry {
    /// Per-stage timing during prompt processing.
    pub prefill: Option<PhaseProfile>,
    /// Per-stage timing during token generation.
    pub decode: Option<PhaseProfile>,
    /// Which kernels and SIMD path the engine selected.
    pub backend: Option<BackendTelemetry>,
    /// Memory breakdown.
    pub memory: Option<MemoryTelemetry>,
    /// MoE routing stats. `None` on a dense model — which is itself the signal
    /// that no expert routing happened.
    pub moe: Option<MoeTelemetry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(name: &str, ms: f64, calls: u64) -> StageTiming {
        StageTiming {
            name: name.into(),
            total_ms: ms,
            calls,
            bytes_read: None,
            macs: None,
        }
    }

    #[test]
    fn hotspots_rank_by_time_descending() {
        let p = PhaseProfile {
            stages: vec![stage("attn", 40.0, 24), stage("ffn", 55.0, 24), stage("qkv", 5.0, 24)],
            total_ms: 100.0,
        };
        let names: Vec<&str> = p.hotspots().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["ffn", "attn", "qkv"]);
    }

    #[test]
    fn unattributed_time_is_surfaced_not_hidden() {
        // Stages sum to 90 of 100 ms. The missing 10 ms is instrumentation
        // blind spot and must be reported, not normalized away.
        let p = PhaseProfile {
            stages: vec![stage("attn", 40.0, 1), stage("ffn", 50.0, 1)],
            total_ms: 100.0,
        };
        assert!((p.unattributed_ms() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn unattributed_never_negative() {
        // Stage timers can overshoot the phase total (nested timing, clock
        // granularity). Clamp rather than report a negative blind spot.
        let p = PhaseProfile {
            stages: vec![stage("a", 60.0, 1), stage("b", 60.0, 1)],
            total_ms: 100.0,
        };
        assert_eq!(p.unattributed_ms(), 0.0);
    }

    #[test]
    fn share_of_zero_phase_is_none_not_zero() {
        // A phase that never ran has no breakdown. Reporting 0% would imply
        // the stage ran and took no time, which is a different claim.
        let s = stage("attn", 0.0, 0);
        assert_eq!(s.share_of(0.0), None);
        assert_eq!(stage("attn", 25.0, 1).share_of(100.0), Some(0.25));
    }

    #[test]
    fn moe_load_balance_ignores_untouched_experts() {
        // 4 of 8 experts got tokens. min/max/mean describe the LIVE ones —
        // folding in the zeros would report min=0 and drag the mean down,
        // making a healthy top-k router look collapsed.
        let m = MoeTelemetry {
            num_experts: 8,
            num_experts_per_tok: 2,
            expert_load: vec![10, 0, 20, 0, 30, 0, 40, 0],
            moe_layers: 1,
        };
        assert_eq!(m.experts_touched(), 4);
        let (min, max, mean) = m.load_balance().unwrap();
        assert_eq!((min, max), (10, 40));
        assert!((mean - 25.0).abs() < 1e-9);
    }

    #[test]
    fn routing_entropy_uniform_is_one_collapsed_is_zero() {
        let uniform = MoeTelemetry {
            num_experts: 4,
            num_experts_per_tok: 4,
            expert_load: vec![25, 25, 25, 25],
            moe_layers: 1,
        };
        let e = uniform.routing_entropy().unwrap();
        assert!((e - 1.0).abs() < 1e-9, "uniform routing should be 1.0, got {e}");

        // Every token to one expert: the pathological case MoE must never hit.
        let collapsed = MoeTelemetry {
            num_experts: 4,
            num_experts_per_tok: 1,
            expert_load: vec![100, 0, 0, 0],
            moe_layers: 1,
        };
        assert_eq!(collapsed.routing_entropy(), Some(0.0));
    }

    #[test]
    fn routing_entropy_none_when_nothing_routed() {
        let dense = MoeTelemetry {
            num_experts: 8,
            num_experts_per_tok: 2,
            expert_load: vec![0; 8],
            moe_layers: 0,
        };
        assert_eq!(dense.routing_entropy(), None);
        assert_eq!(dense.load_balance(), None);
    }

    #[test]
    fn empty_telemetry_is_all_none() {
        // An engine that collects nothing must cost nothing and report nothing.
        let t = EngineTelemetry::default();
        assert!(t.prefill.is_none() && t.decode.is_none() && t.moe.is_none());
    }
}
