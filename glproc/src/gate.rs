//! glproc's `CandidateSource`: real per-op weight-format candidates for
//! `glcore::gate::Planner` to drive.
//!
//! This is the concrete instance of the pattern `architecture/GATE/
//! GATE-constraints.md`'s Composability section describes for
//! `Constraint`: `glcore::gate` stays protocol-only, and the backend crate
//! that can actually evaluate a candidate supplies it. Here, the candidate
//! set is the FFN weight-format decision `crate::loader::weight` already
//! makes at load time (repack Q4_K to Q8_0, or keep it native) — this
//! module does not introduce a new decision, it exposes the existing one
//! to GATE's `Planner` so `generate_candidates`/`select_best` have a real
//! multi-candidate case to run against instead of a synthetic one.
//!
//! # Where `MetricVector.latency_ms` comes from (Wave C)
//!
//! Each candidate's `latency_ms` is populated by [`calibrate`], which
//! loads the target model twice — once per weight format — and runs a
//! small number of real decode steps through an actual [`crate::runner::
//! Runner`] on each. This is **not** a claim of measurement-discipline.md
//! decision-grade rigor (that standard asks for `--warmup ≥ 1, --iters ≥
//! 5` through the full `glbench` harness, with cold/warm separation and
//! percentile reporting — `glbench` depends on `glcore`/`glproc`, so
//! glproc cannot call back into it without inverting that dependency).
//! This is a lighter calibration tier: real tokens, real hardware, this
//! machine, this model, right now — strictly better grounding than a
//! hardcoded constant from a different model's benchmark session, but
//! explicitly not glbench's own bar. Anyone quoting this module's numbers
//! as a decision-grade benchmark result is misusing them.
//!
//! Calibration cost is paid at most once per process (see [`CALIBRATION`],
//! a `OnceLock`): the common case (Q8_0 wins, which it has every time this
//! has been measured on the reference tier) costs one extra full model
//! load the first time `resolve_prefer_q4k_native` runs in a process,
//! never again after. If a caller cannot afford that cost even once, the
//! only escape hatch today is bypassing `GlprocEngine::load_model`
//! entirely and calling `loader::load_gguf` directly (see `kernels::qdot::
//! q4k_native`'s doc) — GATE has no cheaper path yet.

use std::sync::OnceLock;

use glcore::format::gguf::GgufFile;
use glcore::gate::{
    BackendKind, CandidateSource, CostEvaluator, ExecutionPlan, ExecutionPolicy, GateError,
    MetricVector, Planner, TensorGraph, TensorOp, Validator,
};

use crate::loader::load_gguf_with_gate;
use crate::runner::Runner;
use crate::sampler::{Sampler, SamplerConfig};

/// Decode steps timed per candidate, after one untimed warmup step.
/// Smaller than `measurement-discipline.md`'s `--iters ≥ 5` floor on
/// purpose (see this module's doc) — this is a session-init cost paid
/// synchronously inside `load_model`, not an offline benchmark run.
const CALIBRATION_STEPS: usize = 3;

/// A tiny fixed prompt, just enough to prime one real forward pass before
/// timing decode. Token id `1` is not guaranteed meaningful for every
/// vocabulary, but calibration only measures *throughput*, never reads the
/// generated text — an arbitrary in-range id is fine, and `1` is in range
/// for every GGUF vocabulary this project has loaded (BOS/near-BOS region
/// is always populated).
const CALIBRATION_PROMPT: [u32; 1] = [1];

/// This process's calibrated `(q8_0_tok_s, native_q4k_tok_s)`, computed at
/// most once. `OnceLock`, not per-call: calibration's cost (a second full
/// model load) is only worth paying once per process lifetime, per this
/// module's doc — every subsequent `resolve_prefer_q4k_native` call in the
/// same process reuses this run's numbers rather than repeating the load.
static CALIBRATION: OnceLock<(f64, f64)> = OnceLock::new();

/// Load `gguf` fully in `format`, run [`CALIBRATION_STEPS`] real decode
/// steps through an actual `Runner`, and return the measured decode
/// tok/s. `Err` on any failure to load or generate — the caller (
/// [`calibrate`]) turns a failure into an `Undetermined State` for that
/// one candidate rather than failing the whole calibration.
fn measure_decode_tok_s(gguf: &GgufFile, prefer_native: bool) -> Result<f64, glcore::GlError> {
    let model = load_gguf_with_gate(gguf, Some(prefer_native))?;
    let mut runner = Runner::new(&model);
    let mut sampler = Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() });

    // One untimed warmup step (page faults, branch predictor, cache
    // warming) — the same warmup-then-measure separation `glbench` itself
    // uses, at a smaller scale.
    runner.generate(&CALIBRATION_PROMPT, 1, &mut sampler, |_| false, |_| {})?;

    let (_, timing) = runner.generate(
        &CALIBRATION_PROMPT,
        CALIBRATION_STEPS,
        &mut sampler,
        |_| false,
        |_| {},
    )?;
    let decode_s = timing.decode.as_secs_f64();
    if decode_s <= 0.0 {
        return Err(glcore::GlError::Engine(
            "calibration decode phase reported zero elapsed time".into(),
        ));
    }
    Ok(CALIBRATION_STEPS as f64 / decode_s)
}

/// Run calibration for both weight-format candidates against `gguf`,
/// caching the result for the rest of the process. A format whose
/// measurement fails becomes `f64::INFINITY` tok/s (⇒ `latency_ms = 0.0`
/// is wrong and `f64::INFINITY` is right — see the inversion note below)
/// — logged at warn level, never silently defaulted, per `MetricVector`'s
/// `Undetermined State` contract.
fn calibrate(gguf: &GgufFile) -> (f64, f64) {
    *CALIBRATION.get_or_init(|| {
        let measure = |prefer_native: bool, label: &str| match measure_decode_tok_s(gguf, prefer_native) {
            Ok(tok_s) => tok_s,
            Err(e) => {
                eprintln!(
                    "[gate] latency undetermined for candidate {label} — {e} — \
                     falling back to constraint-only selection for this candidate"
                );
                // tok/s -> ms/token inverts: an UNDETERMINED tok/s must
                // become an INFINITE latency_ms (worst possible cost), so
                // this returns 0.0 tok/s here and `to_latency_ms` below
                // maps 0.0 tok/s to `f64::INFINITY` ms — never the other
                // way around, which would make a failed measurement look
                // like a zero-latency (best possible) candidate.
                0.0
            }
        };
        let q8_0 = measure(false, "Q8_0 repack");
        let native = measure(true, "native Q4_K");
        (q8_0, native)
    })
}

/// `tok/s` → `MetricVector.latency_ms`, mapping `0.0` tok/s (this module's
/// `Undetermined State` sentinel — see [`calibrate`]) to `f64::INFINITY`
/// ms rather than dividing into an infinite or `NaN` result implicitly.
/// Any real measurement is strictly positive, so `0.0` cannot occur except
/// via the sentinel path.
fn to_latency_ms(tok_s: f64) -> f64 {
    if tok_s <= 0.0 {
        f64::INFINITY
    } else {
        1000.0 / tok_s
    }
}

/// glproc's `CandidateSource`: for any op, offers the two weight-format
/// strategies `crate::loader::weight` already implements — Q8_0 repack
/// (the shipped default) and native Q4_K (opt-in via `GLPROC_Q4K_NATIVE`,
/// kept for a future wider-vector kernel per `qdot::supports`'s doc).
///
/// Not gated on the op's tensor actually being Q4_K-quantized: this is a
/// GATE-plumbing demonstration over the real, already-measured decision,
/// not a new dispatch path — `crate::loader::weight` remains the only
/// place that decision is actually made for real inference.
///
/// Holds the `GgufFile` calibration is run against — a candidate source
/// now has real work to do per instance (Wave C), unlike Wave A's
/// zero-sized marker type.
pub struct FfnWeightFormatSource<'g> {
    gguf: &'g GgufFile,
}

impl<'g> FfnWeightFormatSource<'g> {
    /// A candidate source that calibrates against `gguf` on first use.
    pub fn new(gguf: &'g GgufFile) -> Self {
        FfnWeightFormatSource { gguf }
    }
}

/// Suffix `candidates_for` tags the Q8_0-repack candidate's op with, so its
/// winning plan is identified by name rather than by reverse-engineering a
/// `latency_ms` float value (fragile if two candidates ever tie exactly).
const Q8_0_TAG: &str = "::q8_0_repack";

/// Suffix for the native-Q4_K candidate (see [`Q8_0_TAG`]).
const NATIVE_Q4K_TAG: &str = "::native_q4k";

impl CandidateSource for FfnWeightFormatSource<'_> {
    fn candidates_for(&self, op: &TensorOp) -> Vec<ExecutionPlan> {
        let (q8_0_tok_s, native_tok_s) = calibrate(self.gguf);
        let plan_with = |tag: &str, latency_ms: f64| ExecutionPlan {
            ordering: vec![TensorOp::new(format!("{}{tag}", op.tensor_name))],
            backend: BackendKind::Glproc,
            layouts: Default::default(),
            metrics: MetricVector { latency_ms, ..Default::default() },
        };
        vec![
            plan_with(Q8_0_TAG, to_latency_ms(q8_0_tok_s)),
            plan_with(NATIVE_Q4K_TAG, to_latency_ms(native_tok_s)),
        ]
    }
}

/// Run GATE once, at session init, to decide whether this session's FFN
/// weight tensors should load as native Q4_K or repacked Q8_0 — the answer
/// `GlprocEngine::load_model` feeds into `loader::load_gguf_with_gate`.
///
/// **This supersedes `GLPROC_Q4K_NATIVE` for any caller going through
/// `GlprocEngine::load_model`** (see `kernels::qdot::q4k_native`'s doc):
/// GATE is now the decision-maker for this choice, not the env var. The
/// env var remains fully functional for direct `loader::load_gguf`
/// callers that bypass `GlprocEngine` entirely.
///
/// One `Planner` run for the whole session (not per tensor): the two
/// candidates represent the session-wide policy choice
/// `qdot::q4k_native()` already applies uniformly to every Q4_K tensor it
/// sees, so one `TensorGraph` node stands in for that one decision rather
/// than enumerating every tensor the model happens to have.
///
/// `validator` is caller-supplied (not constructed here) so a caller that
/// wants real constraints — or a test proving the `ConstraintViolation`
/// path — can register them; glproc itself registers none today (see this
/// module's doc: no concrete `Constraint` exists in this crate yet, and
/// inventing one to look complete would be exactly the "speculative
/// design" `architecture/GATE/GATE-mapping.md`'s Gap 1 note already
/// warned against once this session).
///
/// Returns `Err(GateError::NoValidPlan)` if every candidate was rejected —
/// propagated, never silently defaulted, per GATE's Theorem 10.2 ("never
/// dispatch an invalid plan"). With today's empty default validator this
/// cannot happen (`Pvalid = Pcand` vacuously), but the error path is real,
/// not dead code kept for show: the moment a real `Constraint` is
/// registered, this is the path that keeps the guarantee honest.
pub fn resolve_prefer_q4k_native(
    gguf: &GgufFile,
    validator: Validator,
    policy: ExecutionPolicy,
) -> Result<bool, GateError> {
    let graph = TensorGraph { ops: vec![TensorOp::new("ffn_weight_format")] };
    let evaluator = CostEvaluator::new(policy.weight_vector());
    let planner = Planner::new(validator, evaluator, 8);
    let source = FfnWeightFormatSource::new(gguf);
    let mut candidates = planner.generate_candidates(&graph, policy, &source);
    let best = planner.select_best(&mut candidates)?;
    // Recover which candidate won by its tagged op name — the translation
    // from GATE's generic `ExecutionPlan` back to glproc's own
    // `prefer_q4k_native: bool` happens here, entirely inside glproc, so
    // `glcore::gate` never needs a glproc-specific field (see this
    // module's doc for why `ExecutionPlan` itself can't carry that).
    let winning_name = best.ordering.first().map(|op| op.tensor_name.as_str()).unwrap_or("");
    Ok(winning_name.ends_with(NATIVE_Q4K_TAG))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glcore::gate::{Constraint, ValidationResult};

    /// Every calibration-dependent test needs a real GGUF (calibration
    /// loads the model twice through the real loader) — same fixture
    /// convention as `tests/q4k_e2e.rs`: read `GLPROC_E2E_MODEL`, skip
    /// loudly when unset, so CI without a multi-hundred-MB model stays
    /// green. Returns `None` (skip) rather than panicking, so every test
    /// using this can `let Some(gguf) = ... else { return };` uniformly.
    fn e2e_gguf() -> Option<(String, GgufFile)> {
        let path = match std::env::var("GLPROC_E2E_MODEL") {
            Ok(p) => p,
            Err(_) => {
                eprintln!(
                    "SKIP gate::tests: set GLPROC_E2E_MODEL to a .gguf to run \
                     calibration-dependent gate tests"
                );
                return None;
            }
        };
        let gguf = GgufFile::open(&path).expect("open GLPROC_E2E_MODEL");
        Some((path, gguf))
    }

    /// End-to-end: a real `TensorGraph` over glproc's real, calibrated
    /// Q4_K/Q8_0 candidates, run through the full `Planner` (Steps 1, 5,
    /// 6), must select the Q8_0 repack — the actually-faster path on this
    /// hardware — under a latency-weighted policy. This is the worked
    /// instance ARTX03/ARTX05 (GateCostModel) describe: a cost model
    /// scoring only bytes-moved would have picked the wrong one; this one,
    /// scored on real measured latency, picks correctly.
    #[test]
    fn planner_selects_the_faster_measured_format() {
        let Some((_, gguf)) = e2e_gguf() else { return };
        let graph = TensorGraph { ops: vec![TensorOp::new("blk.0.ffn_gate.weight")] };
        let evaluator =
            CostEvaluator::new(glcore::gate::WeightVector { weights: [1.0, 0.0, 0.0, 0.0, 0.0] });
        let planner = Planner::new(Validator::new(), evaluator, 8);
        let source = FfnWeightFormatSource::new(&gguf);

        let mut candidates =
            planner.generate_candidates(&graph, ExecutionPolicy::Latency, &source);
        assert_eq!(candidates.len(), 2, "one candidate per weight format");

        let best = planner.select_best(&mut candidates).expect("a valid plan exists");
        let winner = best.ordering.first().map(|op| op.tensor_name.as_str()).unwrap_or("");
        assert!(
            winner.ends_with(Q8_0_TAG),
            "must select the Q8_0-repack candidate, got {winner} (latency_ms={})",
            best.metrics.latency_ms
        );
    }

    /// Proves calibration replaced the old hardcoded constants: the
    /// winning candidate's `latency_ms` must NOT equal either of the two
    /// literal values Wave A shipped (`1000.0 / 14.13`, `1000.0 / 9.5`) —
    /// a real measurement on real hardware will not coincidentally land on
    /// those exact floats.
    #[test]
    fn ffn_source_populates_latency_from_measurement_not_constant() {
        let Some((_, gguf)) = e2e_gguf() else { return };
        let source = FfnWeightFormatSource::new(&gguf);
        let candidates = source.candidates_for(&TensorOp::new("blk.0.ffn_gate.weight"));
        assert_eq!(candidates.len(), 2);

        const OLD_Q8_0_MS: f64 = 1000.0 / 14.13;
        const OLD_NATIVE_MS: f64 = 1000.0 / 9.5;
        for c in &candidates {
            assert!(
                (c.metrics.latency_ms - OLD_Q8_0_MS).abs() > 1e-9,
                "latency_ms {} must not be the old Wave-A Q8_0 constant",
                c.metrics.latency_ms
            );
            assert!(
                (c.metrics.latency_ms - OLD_NATIVE_MS).abs() > 1e-9,
                "latency_ms {} must not be the old Wave-A native constant",
                c.metrics.latency_ms
            );
            assert!(c.metrics.latency_ms.is_finite(), "a real measurement must be finite");
            assert!(c.metrics.latency_ms > 0.0, "a real measurement must be positive");
        }
    }

    /// The session-init entry point `GlprocEngine::load_model` calls: with
    /// no constraints registered (today's real default), it must resolve
    /// to `false` (prefer Q8_0 repack) using real calibrated numbers —
    /// matching the shipped default `qdot::q4k_native()` already returns
    /// when `GLPROC_Q4K_NATIVE` is unset.
    #[test]
    fn gate_selects_correct_path_with_real_metrics() {
        let Some((_, gguf)) = e2e_gguf() else { return };
        let prefer_native =
            resolve_prefer_q4k_native(&gguf, Validator::new(), ExecutionPolicy::Latency)
                .expect("empty validator accepts every candidate");
        assert!(!prefer_native, "must prefer Q8_0 repack, the measurably faster path");
    }

    /// If every candidate is rejected (`Pvalid = ∅`), the error must
    /// surface cleanly — no panic, no silent fallback to a default. Proves
    /// the propagation path this module's doc promises is real, not dead
    /// code: with a real `Constraint` registered, `NoValidPlan` can and
    /// does fire. Uses calibration (needs a real GGUF) so this exercises
    /// the exact path production hits, not a synthetic stand-in.
    #[test]
    fn gate_constraint_violation_propagates_correctly() {
        let Some((_, gguf)) = e2e_gguf() else { return };
        struct AlwaysReject;
        impl Constraint for AlwaysReject {
            fn validate(&self, _plan: &ExecutionPlan) -> ValidationResult {
                ValidationResult::Reject { reason: "always rejects".into() }
            }
            fn name(&self) -> &'static str {
                "AlwaysReject"
            }
        }
        let mut validator = Validator::new();
        validator.register(Box::new(AlwaysReject));

        match resolve_prefer_q4k_native(&gguf, validator, ExecutionPolicy::Latency) {
            Err(GateError::NoValidPlan { candidates_tried }) => {
                assert_eq!(candidates_tried, 2, "both format candidates were tried and rejected")
            }
            other => panic!("expected NoValidPlan, got {other:?}"),
        }
    }

    /// `weighted_term`'s zero-weight/`Undetermined` interaction (glcore's
    /// concern, tested there) has a glproc-facing counterpart worth
    /// pinning here: `to_latency_ms(0.0)` — the sentinel `calibrate` uses
    /// for a failed measurement — must map to `f64::INFINITY`, never to
    /// `0.0` or `NaN`. This does not need a real model.
    #[test]
    fn to_latency_ms_maps_failed_measurement_sentinel_to_infinity() {
        assert_eq!(to_latency_ms(0.0), f64::INFINITY);
        assert!(to_latency_ms(14.13).is_finite());
    }
}
