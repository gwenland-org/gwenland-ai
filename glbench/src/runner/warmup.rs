//! Warmup phase.
//!
//! The first inference after load pays one-time costs — kernel JIT (glcuda
//! compiles PTX at load), page-ins, cold caches, CUDA-graph capture. Those are
//! not what a throughput benchmark wants to measure, so warmup runs the same
//! request untimed to move them out of the measured window. The planner drives
//! the loop; this module documents the contract and offers the count policy.

use crate::core::workload::WorkloadSpec;

/// Recommend a warmup count for a spec. A spec that asked for zero is honored
/// (the user may want cold-start numbers), otherwise at least one warmup for
/// any GPU backend, which has the largest cold-start cost (PTX JIT + capture).
pub fn recommended_warmup(spec: &WorkloadSpec, backend: &str) -> usize {
    if spec.warmup_iters > 0 {
        return spec.warmup_iters;
    }
    if backend != "cpu" {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::workload::WorkloadSpec;

    #[test]
    fn gpu_gets_a_warmup_when_unset() {
        let spec = WorkloadSpec { warmup_iters: 0, ..Default::default() };
        assert_eq!(recommended_warmup(&spec, "cuda"), 1);
        assert_eq!(recommended_warmup(&spec, "cpu"), 0);
    }

    #[test]
    fn explicit_warmup_is_honored() {
        let spec = WorkloadSpec { warmup_iters: 3, ..Default::default() };
        assert_eq!(recommended_warmup(&spec, "cpu"), 3);
    }
}
