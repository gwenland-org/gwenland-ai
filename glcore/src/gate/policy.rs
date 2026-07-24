//! `ExecutionPolicy` — the five preset policies and their weight vectors
//! (paper §4.4 policy definition, §7/Table 2).

use crate::gate::metrics::WeightVector;

/// A named deployment-context policy, resolving to a [`WeightVector`] —
/// paper's `π: 𝒦 → Δ⁴` (§4.4). See `architecture/GATE/GATE-policy.md` for
/// when to use which policy in GwenLand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionPolicy {
    /// Real-time, interactive services. `w = (0.6, 0.1, 0.1, 0.1, 0.1)`.
    Latency,
    /// Edge devices, multi-model batching. `w = (0.1, 0.6, 0.1, 0.1, 0.1)`.
    Memory,
    /// Mobile, battery, green computing. `w = (0.1, 0.1, 0.1, 0.6, 0.1)`.
    Energy,
    /// General-purpose, benchmarking. `w = (0.2, 0.2, 0.2, 0.2, 0.2)`.
    Balanced,
    /// Domain-specific, user-supplied weight vector.
    Custom(WeightVector),
}

impl ExecutionPolicy {
    /// Resolve this policy to its weight vector (Table 2, paper §7).
    ///
    /// By Theorem 4.2 (Policy Independence of Validity), changing which
    /// policy is active can never change which plans are *valid* — only
    /// which valid plan is selected. See `architecture/GATE/GATE-policy.md`.
    pub fn weight_vector(&self) -> WeightVector {
        let weights = match self {
            ExecutionPolicy::Latency => [0.6, 0.1, 0.1, 0.1, 0.1],
            ExecutionPolicy::Memory => [0.1, 0.6, 0.1, 0.1, 0.1],
            ExecutionPolicy::Energy => [0.1, 0.1, 0.1, 0.6, 0.1],
            ExecutionPolicy::Balanced => [0.2, 0.2, 0.2, 0.2, 0.2],
            ExecutionPolicy::Custom(w) => return *w,
        };
        WeightVector { weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_weight_vectors_sum_to_one() {
        for policy in [
            ExecutionPolicy::Latency,
            ExecutionPolicy::Memory,
            ExecutionPolicy::Energy,
            ExecutionPolicy::Balanced,
        ] {
            let sum: f64 = policy.weight_vector().weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "{policy:?} weights sum to {sum}");
        }
    }
}
