// engine/inference/params.rs — Inference generation parameter configuration.
//
// Provides `InferParams`, a validated configuration struct for controlling
// language-model generation behaviour (temperature, top-p, token budget, etc.).

use serde::{Deserialize, Serialize};

/// Configuration for a single inference request.
///
/// All numeric fields are validated by [`InferParams::validate`] before use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferParams {
    /// Maximum number of tokens to generate. Must be ≥ 1.
    pub max_tokens: usize,
    /// Sampling temperature. Must be in the range (0.0, 2.0].
    pub temperature: f32,
    /// Nucleus-sampling probability mass. Must be in the range (0.0, 1.0].
    pub top_p: f32,
    /// Optional top-k candidate limit.
    pub top_k: Option<usize>,
    /// Optional repetition-penalty multiplier.
    pub repetition_penalty: Option<f32>,
    /// Sequences that, when generated, halt further token production.
    pub stop_sequences: Vec<String>,
    /// Optional RNG seed for reproducible sampling. `None` means non-deterministic.
    pub seed: Option<u64>,
}

impl Default for InferParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            repetition_penalty: None,
            stop_sequences: vec![],
            seed: None,
        }
    }
}

impl InferParams {
    /// Validate the parameter values against the allowed ranges.
    ///
    /// # Errors
    ///
    /// Returns `Err` when:
    /// - `temperature` is ≤ 0.0 or > 2.0  (Requirement 2.2)
    /// - `top_p` is ≤ 0.0 or > 1.0         (Requirement 2.3)
    /// - `max_tokens` is 0                  (Requirement 2.4)
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.temperature <= 0.0 || self.temperature > 2.0 {
            anyhow::bail!(
                "temperature must be in range (0.0, 2.0], got {}",
                self.temperature
            );
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            anyhow::bail!(
                "top_p must be in range (0.0, 1.0], got {}",
                self.top_p
            );
        }
        if self.max_tokens == 0 {
            anyhow::bail!("max_tokens must be at least 1, got 0");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default values ────────────────────────────────────────────────────────

    #[test]
    fn default_max_tokens_is_512() {
        assert_eq!(InferParams::default().max_tokens, 512);
    }

    #[test]
    fn default_temperature_is_0_7() {
        assert!((InferParams::default().temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn default_top_p_is_0_9() {
        assert!((InferParams::default().top_p - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn default_top_k_is_none() {
        assert!(InferParams::default().top_k.is_none());
    }

    #[test]
    fn default_repetition_penalty_is_none() {
        assert!(InferParams::default().repetition_penalty.is_none());
    }

    #[test]
    fn default_stop_sequences_is_empty() {
        assert!(InferParams::default().stop_sequences.is_empty());
    }

    // ── Valid params pass ─────────────────────────────────────────────────────

    #[test]
    fn valid_default_params_pass_validation() {
        assert!(InferParams::default().validate().is_ok());
    }

    #[test]
    fn valid_custom_params_pass_validation() {
        let p = InferParams {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 0.95,
            top_k: Some(50),
            repetition_penalty: Some(1.1),
            stop_sequences: vec!["</s>".to_string()],
            seed: None,
        };
        assert!(p.validate().is_ok());
    }

    // ── Temperature boundary tests ────────────────────────────────────────────

    #[test]
    fn temperature_zero_fails() {
        let p = InferParams { temperature: 0.0, ..InferParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn temperature_two_passes() {
        let p = InferParams { temperature: 2.0, ..InferParams::default() };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn temperature_above_two_fails() {
        let p = InferParams { temperature: 2.001, ..InferParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn temperature_negative_fails() {
        let p = InferParams { temperature: -0.1, ..InferParams::default() };
        assert!(p.validate().is_err());
    }

    // ── top_p boundary tests ──────────────────────────────────────────────────

    #[test]
    fn top_p_zero_fails() {
        let p = InferParams { top_p: 0.0, ..InferParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn top_p_one_passes() {
        let p = InferParams { top_p: 1.0, ..InferParams::default() };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn top_p_above_one_fails() {
        let p = InferParams { top_p: 1.001, ..InferParams::default() };
        assert!(p.validate().is_err());
    }

    // ── max_tokens boundary tests ─────────────────────────────────────────────

    #[test]
    fn max_tokens_zero_fails() {
        let p = InferParams { max_tokens: 0, ..InferParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn max_tokens_one_passes() {
        let p = InferParams { max_tokens: 1, ..InferParams::default() };
        assert!(p.validate().is_ok());
    }
}
