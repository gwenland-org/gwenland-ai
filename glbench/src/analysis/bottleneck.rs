//! Bottleneck classification.
//!
//! This is where measurements become an interpretation — and it is the sharp
//! line the spec draws: glbench may *say* "memory bound" but must never *act*
//! on it. The classification is a hint for the engine developer, produced from
//! the facts (ceiling efficiency, prefill vs decode balance), and it always
//! comes with a recommendation phrased as an observation.

use crate::analysis::ceiling::Ceiling;
use crate::core::session::BenchmarkSession;

/// The dominant limiting factor for a workload, as inferred from the facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// Bound by memory bandwidth — running near the streaming ceiling.
    MemoryBound,
    /// Bound by compute throughput.
    ComputeBound,
    /// Bound by host↔device transfer / launch overhead.
    TransferBound,
    /// Bound by per-launch/host overhead (GPU idle much of the time).
    LaunchOverhead,
    /// Not enough information to classify.
    Undetermined,
}

impl Bottleneck {
    /// Stable identifier for archives and rendering.
    pub fn as_str(self) -> &'static str {
        match self {
            Bottleneck::MemoryBound => "memory_bound",
            Bottleneck::ComputeBound => "compute_bound",
            Bottleneck::TransferBound => "transfer_bound",
            Bottleneck::LaunchOverhead => "launch_overhead",
            Bottleneck::Undetermined => "undetermined",
        }
    }

    /// A recommendation phrased as an observation — never an instruction to
    /// glbench itself, always guidance for the engine developer.
    pub fn recommendation(self) -> &'static str {
        match self {
            Bottleneck::MemoryBound => {
                "Decode is memory-bandwidth bound and near the ceiling; further speedup requires \
                 fewer bytes streamed per token (higher quantization) rather than more compute."
            }
            Bottleneck::ComputeBound => {
                "Throughput is well under the bandwidth ceiling; the limiter is compute or \
                 kernel efficiency, so tensor-core / vectorization work is the lever."
            }
            Bottleneck::TransferBound => {
                "Host-device transfer dominates; batching or keeping data resident on the device \
                 is the lever."
            }
            Bottleneck::LaunchOverhead => {
                "The device appears idle much of the time; kernel-launch overhead or \
                 serialization is the suspect — fusing or batching launches is the lever."
            }
            Bottleneck::Undetermined => {
                "Not enough signal to classify the bottleneck; capture a hardware ceiling \
                 (peak bandwidth) and more iterations."
            }
        }
    }
}

/// Classify the bottleneck from ceiling efficiency and phase balance.
///
/// Heuristic, and honestly labelled as such: at ≥85% of the bandwidth ceiling
/// decode is memory-bound; well below it with a known ceiling, compute-bound;
/// with no ceiling we fall back to the prefill/decode balance and otherwise
/// decline to guess.
pub fn classify(session: &BenchmarkSession, ceiling: &Ceiling) -> Bottleneck {
    if let Some(eff) = ceiling.efficiency {
        return if eff >= 0.85 {
            Bottleneck::MemoryBound
        } else if eff >= 0.40 {
            // Meaningfully below peak but still using the machine: most likely
            // compute/kernel-efficiency bound.
            Bottleneck::ComputeBound
        } else {
            // Far below peak: the machine is mostly idle -> overhead/serialization.
            Bottleneck::LaunchOverhead
        };
    }

    // No ceiling: fall back to observable phase balance. A very low prefill:decode
    // token-rate ratio points at prefill inefficiency, but without a ceiling we
    // cannot separate compute from bandwidth, so we decline to over-claim.
    let m = &session.measurements;
    if m.is_empty() {
        return Bottleneck::Undetermined;
    }
    Bottleneck::Undetermined
}
