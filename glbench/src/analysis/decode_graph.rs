//! What the production decode graph spends its time on, estimated.
//!
//! A CUDA Graph replays as one opaque launch, so the only way to see inside it
//! is to capture timing events between its stages, and those events are not
//! free: on a T4 they made decode 1.15x slower, and 2.05x in Wave 133. The raw
//! stage times therefore describe a program that never runs in production.
//!
//! This takes the slowdown as a measurement rather than a caveat. The added time
//! per token is known, from a production run and a profiled run of the same
//! workload, and so is the number of timed boundaries per token. Assuming every
//! boundary costs the same, removing that cost from each stage estimates the
//! stage in the production graph. What is left of production time after the
//! stages is work outside the graph: host sampling and token handling.
//!
//! An estimate, and labelled one: the equal-cost assumption is unverified, and
//! nothing here can measure the production graph directly.

use crate::core::session::BenchmarkSession;

/// One decode-graph stage.
#[derive(Debug, Clone, PartialEq)]
pub struct VLGraphStage {
    /// Stage name, as the engine reports it.
    pub name: String,
    /// Device time per token as measured, event cost included.
    pub measured_ms_per_token: f64,
    /// Device time per token with the event cost removed.
    pub estimated_ms_per_token: f64,
}

/// A decode graph's estimated breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct VLGraphAttribution {
    /// Stages, costliest first.
    pub stages: Vec<VLGraphStage>,
    /// Decode time per token in the production run.
    pub production_ms_per_token: f64,
    /// Decode time per token with stage timing on.
    pub instrumented_ms_per_token: f64,
    /// Production time the stages do not account for: host-side work per token.
    /// Negative means the equal-cost assumption failed on this run.
    pub outside_graph_ms_per_token: f64,
}

/// Decode milliseconds per generated token.
fn ms_per_token(session: &BenchmarkSession) -> Option<f64> {
    let iterations = &session.measurements.iterations;
    let tokens: u64 = iterations.iter().map(|i| i.generated_tokens).sum();
    let ms: f64 = iterations.iter().map(|i| i.decode_ms).sum();
    (tokens > 0 && ms > 0.0).then(|| ms / tokens as f64)
}

/// Estimate the production graph from a production and a profiled session.
/// `None` unless the profiled session carries decode stages.
pub fn attribute(
    production: &BenchmarkSession,
    instrumented: &BenchmarkSession,
) -> Option<VLGraphAttribution> {
    let phase = instrumented.telemetry.as_ref()?.decode.as_ref()?;
    let production_ms = ms_per_token(production)?;
    let instrumented_ms = ms_per_token(instrumented)?;
    let tokens: u64 = instrumented.measurements.iterations.iter().map(|i| i.generated_tokens).sum();
    if phase.stages.is_empty() || tokens == 0 {
        return None;
    }

    let boundaries: u64 = phase.stages.iter().map(|s| s.calls).sum();
    if boundaries == 0 {
        return None;
    }
    let added_ms = (instrumented_ms - production_ms).max(0.0) * tokens as f64;
    let per_boundary_ms = added_ms / boundaries as f64;

    let mut stages: Vec<VLGraphStage> = phase
        .stages
        .iter()
        .map(|s| VLGraphStage {
            name: s.name.clone(),
            measured_ms_per_token: s.total_ms / tokens as f64,
            estimated_ms_per_token: ((s.total_ms - per_boundary_ms * s.calls as f64)
                / tokens as f64)
                .max(0.0),
        })
        .collect();
    stages.sort_by(|a, b| b.estimated_ms_per_token.total_cmp(&a.estimated_ms_per_token));

    let in_graph: f64 = stages.iter().map(|s| s.estimated_ms_per_token).sum();
    Some(VLGraphAttribution {
        stages,
        production_ms_per_token: production_ms,
        instrumented_ms_per_token: instrumented_ms,
        outside_graph_ms_per_token: production_ms - in_graph,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;
    use glcore::telemetry::{EngineTelemetry, PhaseProfile, StageTiming};

    fn session(decode_ms: f64) -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 10,
            prefill_ms: 10.0,
            decode_ms,
            total_ms: 0.0,
            ttft_ms: None,
        });
        BenchmarkSession::new(
            SessionMetadata::new("t"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glcuda".into(),
                backend: "cuda".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            WorkloadSpec::default(),
            m,
        )
    }

    fn stage(name: &str, total_ms: f64, calls: u64) -> StageTiming {
        StageTiming { name: name.into(), total_ms, calls, bytes_read: None, macs: None }
    }

    fn profiled(decode_ms: f64, stages: Vec<StageTiming>) -> BenchmarkSession {
        let mut s = session(decode_ms);
        let total_ms = stages.iter().map(|st| st.total_ms).sum();
        s.telemetry = Some(EngineTelemetry {
            decode: Some(PhaseProfile { stages, total_ms }),
            ..Default::default()
        });
        s
    }

    #[test]
    fn event_cost_comes_off_each_stage_by_its_boundary_count() {
        // Production 4 ms/token, profiled 8: 40 ms added over 10 tokens, across
        // 200 boundaries, so 0.2 ms per boundary. "ffn" has 150 of them.
        let instrumented =
            profiled(80.0, vec![stage("ffn", 50.0, 150), stage("lm_head", 20.0, 50)]);
        let a = attribute(&session(40.0), &instrumented).unwrap();

        assert_eq!(a.stages[0].name, "ffn");
        assert!((a.stages[0].measured_ms_per_token - 5.0).abs() < 1e-9);
        assert!((a.stages[0].estimated_ms_per_token - 2.0).abs() < 1e-9);
        assert!((a.stages[1].estimated_ms_per_token - 1.0).abs() < 1e-9);
        assert!((a.outside_graph_ms_per_token - 1.0).abs() < 1e-9, "{a:?}");
    }

    #[test]
    fn no_slowdown_leaves_the_stages_as_measured() {
        let instrumented = profiled(40.0, vec![stage("ffn", 30.0, 100)]);
        let a = attribute(&session(40.0), &instrumented).unwrap();
        assert!((a.stages[0].estimated_ms_per_token - 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_session_without_decode_stages_has_no_attribution() {
        assert!(attribute(&session(40.0), &session(80.0)).is_none());
    }
}
