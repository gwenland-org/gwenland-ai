//! Engine observation mode and dispatch-configuration provenance.
//!
//! Profiling is configured before engine construction because both glproc and
//! glcuda read their switches while building runtime state. The configuration
//! is then captured from a strict allowlist so an archive can reproduce the
//! dispatch inputs without copying unrelated process environment.

use glcore::GlError;

use crate::core::result::{ENMeasurementMode, VLDispatchProvenance};
use crate::core::schema::GLBENCH_VERSION;
use crate::core::workload::WorkloadSpec;

const GLPROC_OVERRIDES: &[&str] = &[
    "GLPROC_ATTN_SEQ",
    "GLPROC_FORCE_SCALAR",
    "GLPROC_NO_LOCK",
    "GLPROC_PROFILE",
    "GLPROC_Q4K_NATIVE",
    "GLPROC_ROW_TILE",
    "GLPROC_THREADS",
    "GLPROC_VNNI512",
];

const GLCUDA_OVERRIDES: &[&str] = &[
    "GLCUDA_ATTN_MMA4",
    "GLCUDA_ATTN_MMA4_AV",
    "GLCUDA_ATTN_MMA4_REGQ",
    "GLCUDA_ATTN_ROWS",
    "GLCUDA_BSTAGE",
    "GLCUDA_CACHE",
    "GLCUDA_DEFER_FFN_RESIDUAL",
    "GLCUDA_FFN_GATE_UP_STACKED",
    "GLCUDA_FORCE_Q8",
    "GLCUDA_FUSE_Q8_GLUE",
    "GLCUDA_GEMM_N16",
    "GLCUDA_GEMM_N16_PREFETCH",
    "GLCUDA_GEMM_N32",
    "GLCUDA_GQA7_CHAINS",
    "GLCUDA_GQA_GROUP",
    "GLCUDA_GRID2D",
    "GLCUDA_MULTI_STREAM_PREFILL",
    "GLCUDA_N16_FUSED_SWIGLU",
    "GLCUDA_NO_MMA",
    "GLCUDA_NTILE128",
    "GLCUDA_PROFILE_DECODE",
    "GLCUDA_PROFILE_PREFILL",
    "GLCUDA_Q8_NOSTORE",
    "GLCUDA_R256",
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
        "glcuda" => [
            "GLCUDA_TELEMETRY",
            "GLCUDA_PROFILE_PREFILL",
            "GLCUDA_PROFILE_DECODE",
        ]
        .iter()
        .any(|key| std::env::var_os(key).is_some()),
        _ => false,
    }
}

fn capture_overrides(engine: &str) -> Vec<(String, String)> {
    let allowlist = match engine {
        "glproc" => GLPROC_OVERRIDES,
        "glcuda" => GLCUDA_OVERRIDES,
        _ => &[],
    };
    allowlist
        .iter()
        .filter_map(|name| {
            std::env::var_os(name)
                .map(|value| ((*name).to_string(), value.to_string_lossy().into_owned()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_allowlists_are_sorted_for_stable_fingerprints() {
        assert!(GLPROC_OVERRIDES.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(GLCUDA_OVERRIDES.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn allowlists_do_not_capture_unrelated_environment() {
        assert!(!GLPROC_OVERRIDES.contains(&"PATH"));
        assert!(!GLCUDA_OVERRIDES.contains(&"PATH"));
    }
}
