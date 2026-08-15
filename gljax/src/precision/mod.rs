//! Precision policy (ARTX02 §8).
//!
//! StableHLO has no implicit casting: every dtype boundary is an explicit
//! `stablehlo.convert` (ARTX01 §3.3). Rather than thread a dtype argument
//! through every op signature, gljax keeps the policy in a thread-local and
//! ops read it where they need it — so model code is written once and the same
//! trace produces a BF16 program, an F32 program, or an F64 oracle program.
//!
//! That last one is the point: ARTX01 §3.4's oracle pattern re-traces the
//! *same* model at F64 on the CPU plugin and compares. If precision were baked
//! into the model code, the oracle would be a second implementation — and a
//! second implementation is a second thing that can be wrong.

use std::cell::Cell;

use crate::stablehlo::types::DType;

/// Which dtype each class of operation runs in.
///
/// The four reduction/accumulation fields exist separately from `activation`
/// because ARTX01 §3.2 records that TPU v5e's MXU accumulates in **BF16**,
/// unlike A100's FP32 accumulate. Where that matters — softmax over a long
/// context, the sum of squares in RMSNorm — the reduce is upcast explicitly.
///
/// ⚠️ Overall-Architecture §6 open question 5 asks whether that TPU claim is
/// even true (published TPU docs say FP32). The policy is shaped so the answer
/// changes one constructor, not every op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecisionPolicy {
    /// Default dtype for activations flowing between ops.
    pub activation: DType,
    /// Default dtype weights are declared in.
    pub weight: DType,
    /// Dtype for RMSNorm's sum-of-squares reduce.
    pub norm_reduce: DType,
    /// Dtype for softmax's max and sum reduces.
    pub softmax_reduce: DType,
    /// Dtype the RoPE cos/sin rotation is applied in.
    pub rope: DType,
}

impl PrecisionPolicy {
    /// BF16 activations and weights, F32 reductions. The production setting for
    /// A100 and TPU v5e.
    pub const fn bf16() -> Self {
        PrecisionPolicy {
            activation: DType::BF16,
            weight: DType::BF16,
            norm_reduce: DType::F32,
            softmax_reduce: DType::F32,
            rope: DType::F32,
        }
    }

    /// Everything in F32.
    pub const fn f32() -> Self {
        PrecisionPolicy {
            activation: DType::F32,
            weight: DType::F32,
            norm_reduce: DType::F32,
            softmax_reduce: DType::F32,
            rope: DType::F32,
        }
    }

    /// Everything in F64 — the correctness oracle.
    ///
    /// ⛔ CPU and CUDA plugins only. ARTX01 §3.1: TPU v5e has no FP64 hardware
    /// and XLA will either error or software-emulate unreliably. Never send an
    /// F64 program to a TPU plugin.
    pub const fn f64_oracle() -> Self {
        PrecisionPolicy {
            activation: DType::F64,
            weight: DType::F64,
            norm_reduce: DType::F64,
            softmax_reduce: DType::F64,
            rope: DType::F64,
        }
    }
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::bf16()
    }
}

thread_local! {
    static CURRENT: Cell<PrecisionPolicy> = const { Cell::new(PrecisionPolicy::bf16()) };
}

/// The policy in effect on this thread.
pub fn current() -> PrecisionPolicy {
    CURRENT.with(|p| p.get())
}

/// Runs `f` under `policy`, restoring the previous one afterwards.
///
/// ⚠️ Restores on the normal path only. A panic inside `f` unwinds past the
/// restore and leaves the thread's policy changed — acceptable because a
/// panicking trace is a programming error that aborts the trace anyway
/// (ARTX02 §5's shape-error decision), and because a fresh `TraceCx` is
/// created per compilation.
pub fn with_policy<T>(policy: PrecisionPolicy, f: impl FnOnce() -> T) -> T {
    CURRENT.with(|p| {
        let prev = p.get();
        p.set(policy);
        let result = f();
        p.set(prev);
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_bf16_activations_with_f32_reductions() {
        let p = PrecisionPolicy::default();
        assert_eq!(p.activation, DType::BF16);
        assert_eq!(p.weight, DType::BF16);
        assert_eq!(p.norm_reduce, DType::F32);
        assert_eq!(p.softmax_reduce, DType::F32);
    }

    #[test]
    fn with_policy_restores_the_previous_policy() {
        let before = current();
        let inside = with_policy(PrecisionPolicy::f64_oracle(), current);
        assert_eq!(inside, PrecisionPolicy::f64_oracle());
        assert_eq!(current(), before, "policy leaked out of with_policy");
    }

    #[test]
    fn with_policy_nests() {
        with_policy(PrecisionPolicy::f32(), || {
            assert_eq!(current().activation, DType::F32);
            with_policy(PrecisionPolicy::bf16(), || {
                assert_eq!(current().activation, DType::BF16);
            });
            assert_eq!(
                current().activation,
                DType::F32,
                "inner scope clobbered the outer policy"
            );
        });
    }

    #[test]
    fn oracle_policy_is_f64_everywhere() {
        // A single field left at F32 would make the oracle agree with the
        // thing it is supposed to be checking.
        let p = PrecisionPolicy::f64_oracle();
        for dt in [p.activation, p.weight, p.norm_reduce, p.softmax_reduce, p.rope] {
            assert_eq!(dt, DType::F64);
        }
    }
}
