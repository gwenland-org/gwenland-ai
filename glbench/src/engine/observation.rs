//! Engine observation mode and dispatch-configuration provenance.
//!
//! Profiling is configured before engine construction because both glproc and
//! glcuda read their switches while building runtime state. The configuration
//! is then captured from the engine's own environment namespace (`GLPROC_*`,
//! `GLCUDA_*`) so an archive can reproduce the dispatch inputs without copying
//! unrelated process environment.
//!
//! # Why a namespace and not a list of names
//!
//! This used to be a hand-maintained allowlist. It silently fell behind the
//! engine: the 2026-11-05 T4 audit found `GLCUDA_DECODE_Q8_GLUE` and
//! `GLCUDA_DECODE_QKV_FUSED` missing, so the two arms of the Wave 132 and
//! Wave 136 A/Bs carried the *same* dispatch fingerprint. A list keyed by name
//! cannot notice a name it was never told about; the prefix is the property
//! every switch actually shares.

use glcore::GlError;

use crate::core::result::{ENMeasurementMode, VLDispatchProvenance};
use crate::core::schema::GLBENCH_VERSION;
use crate::core::workload::WorkloadSpec;

/// Every glproc switch lives under this prefix.
const GLPROC_PREFIX: &str = "GLPROC_";

/// Every glcuda switch lives under this prefix.
const GLCUDA_PREFIX: &str = "GLCUDA_";

/// glcuda switches that add timing instrumentation. Presence semantics: any
/// value, including "0", enables them. `GLCUDA_PROFILE_DECODE_STAGES` turns
/// captured decode-graph nodes into timed events, which slowed decode 1.15x to
/// 2.05x on the T4, so a run with it set is never production.
const GLCUDA_INSTRUMENTATION: &[&str] = &[
    "GLCUDA_PROFILE_DECODE",
    "GLCUDA_PROFILE_DECODE_STAGES",
    "GLCUDA_PROFILE_PREFILL",
    "GLCUDA_TELEMETRY",
];

/// Observation facts fixed before an engine starts.
#[derive(Debug, Clone)]
pub struct VLObservationPlan {
    /// Whether the run's throughput is production evidence.
    pub measurement_mode: ENMeasurementMode,
    /// Whitelisted engine configuration and its stable fingerprint.
    pub dispatch: VLDispatchProvenance,
}

impl VLObservationPlan {
    /// Validate an explicit profiling request, then capture the effective
    /// engine switches. The CLI starts a child with profiling configured before
    /// engine startup; library callers must establish the same contract.
    pub fn prepare(spec: &WorkloadSpec) -> Result<VLObservationPlan, GlError> {
        if spec.instrument_engine && !instrumentation_active(&spec.engine) {
            let hint = match spec.engine.as_str() {
                "glproc" => "set GLPROC_PROFILE=1",
                "glcuda" => "set GLCUDA_TELEMETRY=1",
                "gllm" => {
                    return Err(GlError::Engine(
                        "--profile stages is unsupported for gllm: the engine exposes no stage telemetry"
                            .into(),
                    ))
                }
                other => {
                    return Err(GlError::Engine(format!(
                        "--profile stages is unsupported for unknown engine '{other}'"
                    )))
                }
            };
            return Err(GlError::Engine(format!(
                "stage profiling was requested but not configured before engine startup; {hint}"
            )));
        }

        let overrides = capture_overrides(&spec.engine);
        let measurement_mode = if instrumentation_active(&spec.engine) {
            ENMeasurementMode::Instrumented
        } else {
            ENMeasurementMode::Production
        };
        let mut canonical = format!("engine={}\nglbench={}\n", spec.engine, GLBENCH_VERSION);
        for (name, value) in &overrides {
            canonical.push_str(name);
            canonical.push('=');
            canonical.push_str(value);
            canonical.push('\n');
        }
        let fingerprint = glcore::hash::sha256_128_hex(canonical.as_bytes());

        Ok(VLObservationPlan {
            measurement_mode,
            dispatch: VLDispatchProvenance {
                config_fingerprint: format!("sha256-128:{fingerprint}"),
                engine_overrides: overrides,
            },
        })
    }
}

fn instrumentation_active(engine: &str) -> bool {
    match engine {
        // glproc treats empty and "0" as disabled.
        "glproc" => std::env::var("GLPROC_PROFILE")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0"),
        // glcuda's switches use presence semantics, including a value of "0".
        "glcuda" => GLCUDA_INSTRUMENTATION
            .iter()
            .any(|key| std::env::var_os(key).is_some()),
        _ => false,
    }
}

fn capture_overrides(engine: &str) -> Vec<(String, String)> {
    let prefix = match engine {
        "glproc" => GLPROC_PREFIX,
        "glcuda" => GLCUDA_PREFIX,
        _ => return Vec::new(),
    };
    overrides_in(std::env::vars_os(), prefix)
}

/// Keep the variables under `prefix`, sorted by name so the fingerprint does
/// not depend on the order the OS hands the environment over.
fn overrides_in(
    vars: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    prefix: &str,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = vars
        .into_iter()
        .filter_map(|(name, value)| {
            let name = name.into_string().ok()?;
            name.starts_with(prefix)
                .then(|| (name, value.to_string_lossy().into_owned()))
        })
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
        pairs.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect()
    }

    /// The switches the old allowlist missed are captured without anyone
    /// having to list them.
    #[test]
    fn every_switch_in_the_engine_namespace_is_captured() {
        let vars = env(&[
            ("GLCUDA_DECODE_QKV_FUSED", "1"),
            ("GLCUDA_DECODE_Q8_GLUE", "1"),
            ("GLCUDA_SOME_FUTURE_SWITCH", "0"),
        ]);
        let names: Vec<String> =
            overrides_in(vars, GLCUDA_PREFIX).into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            ["GLCUDA_DECODE_Q8_GLUE", "GLCUDA_DECODE_QKV_FUSED", "GLCUDA_SOME_FUTURE_SWITCH"]
        );
    }

    #[test]
    fn unrelated_environment_and_the_other_engine_are_not_captured() {
        let vars = env(&[
            ("PATH", "/usr/bin"),
            ("GLPROC_THREADS", "4"),
            ("XGLCUDA_FAKE", "1"),
            ("GLCUDA_FORCE_Q8", "1"),
        ]);
        assert_eq!(
            overrides_in(vars, GLCUDA_PREFIX),
            vec![("GLCUDA_FORCE_Q8".to_string(), "1".to_string())]
        );
    }

    #[test]
    fn capture_order_does_not_depend_on_environment_order() {
        let a = env(&[("GLCUDA_B", "1"), ("GLCUDA_A", "1")]);
        let b = env(&[("GLCUDA_A", "1"), ("GLCUDA_B", "1")]);
        assert_eq!(overrides_in(a, GLCUDA_PREFIX), overrides_in(b, GLCUDA_PREFIX));
    }

    /// Every timing switch glcuda reads must mark the run instrumented. Read
    /// from the engine's own source so a new profiling switch cannot be added
    /// there and forgotten here — the same drift that broke the allowlist.
    #[test]
    fn every_glcuda_profiling_switch_marks_the_run_instrumented() {
        let sources = [
            include_str!("../../../glcuda/src/runner.rs"),
            include_str!("../../../glcuda/src/model.rs"),
            include_str!("../../../glcuda/src/driver.rs"),
        ];
        let mut found = Vec::new();
        for source in sources {
            for (index, _) in source.match_indices("\"GLCUDA_PROFILE") {
                let name: String = source[index + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                    .collect();
                found.push(name);
            }
        }
        found.sort();
        found.dedup();
        assert!(!found.is_empty(), "scan found no GLCUDA_PROFILE_* switch at all");
        for name in found {
            assert!(
                GLCUDA_INSTRUMENTATION.contains(&name.as_str()),
                "glcuda reads {name} but glbench would label that run production"
            );
        }
    }
}
