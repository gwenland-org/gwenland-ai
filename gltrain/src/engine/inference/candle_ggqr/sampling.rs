// engine/inference/candle_ggqr/sampling.rs — Token sampling strategies.
//
// Provides three sampling primitives and a single dispatcher:
//
//   greedy_sample   — argmax (temperature = 0)
//   top_p_sample    — nucleus sampling over a pre-scaled probability vector
//   sample_token    — dispatcher: scales logits by temperature, optionally
//                     applies top-k masking, then calls greedy or top-p
//
// All functions operate on `Vec<f32>` logits extracted from the forward-pass
// output tensor so they have no candle dependency and are easy to unit-test.
//
// Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::error::GwenError;
use crate::engine::inference::params::InferParams;

// ── 14.1 Greedy sampling ──────────────────────────────────────────────────────

/// Return the token ID with the highest logit value (argmax).
///
/// Requirement: 6.1
pub fn greedy_sample(logits: &[f32]) -> Result<u32, GwenError> {
    if logits.is_empty() {
        return Err(GwenError::InferenceError {
            layer: "sampling".to_string(),
            operation: "greedy_sample".to_string(),
            error: "logits vector is empty".to_string(),
        });
    }
    // `max_by` on a non-empty iterator always returns `Some`; `unwrap_or(0)`
    // is the panic-free spelling of what would otherwise be a safe `.unwrap()`.
    let idx = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(idx as u32)
}

// ── 14.2 Nucleus (top-p) sampling ────────────────────────────────────────────

/// Sample from the smallest set of tokens whose cumulative probability ≥ `p`.
///
/// `probs` must be a probability distribution (non-negative, sums to ≈1).
/// Caller is responsible for applying softmax / temperature scaling first.
///
/// Requirements: 6.2, 6.6
pub fn top_p_sample(probs: &[f32], top_p: f32, rng: &mut impl Rng) -> Result<u32, GwenError> {
    if probs.is_empty() {
        return Err(GwenError::InferenceError {
            layer: "sampling".to_string(),
            operation: "top_p_sample".to_string(),
            error: "probability vector is empty".to_string(),
        });
    }

    // Sort indices by probability descending.
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate until cumulative mass reaches top_p.
    let mut cumulative = 0.0f32;
    let nucleus: Vec<(usize, f32)> = indexed
        .into_iter()
        .take_while(|(_, p)| {
            let prev = cumulative;
            cumulative += p;
            // Include the first token that pushes past top_p so we always have
            // at least one candidate.
            prev < top_p
        })
        .collect();

    // Re-normalise within the nucleus.
    let total: f32 = nucleus.iter().map(|(_, p)| p).sum();
    if total <= 0.0 {
        // Fallback: return the highest-probability token.
        return Ok(nucleus.first().map(|(i, _)| *i as u32).unwrap_or(0));
    }

    let threshold: f32 = rng.r#gen::<f32>() * total;
    let mut acc = 0.0f32;
    for (idx, p) in &nucleus {
        acc += p;
        if acc >= threshold {
            return Ok(*idx as u32);
        }
    }

    // Rounding edge: return last nucleus token.
    Ok(nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0))
}

// ── 14.3 Sample-token dispatcher ─────────────────────────────────────────────

/// Dispatcher: apply temperature / top-k / top-p then sample one token.
///
/// Decision table:
///   temperature == 0.0           → greedy_sample (argmax)
///   temperature  > 0.0           → scale logits, convert to probs via softmax
///     top_k is Some(k)           → mask all but the top-k entries (set rest to -inf)
///     top_p < 1.0                → top_p_sample on the probability vector
///     top_p == 1.0               → sample from the full scaled distribution
///
/// Requirements: 6.3, 6.4, 6.5, 6.7
pub fn sample_token(logits: &[f32], params: &InferParams) -> Result<u32, GwenError> {
    if logits.is_empty() {
        return Err(GwenError::InferenceError {
            layer: "sampling".to_string(),
            operation: "sample_token".to_string(),
            error: "logits vector is empty".to_string(),
        });
    }

    // temperature = 0 → greedy
    if params.temperature == 0.0 {
        return greedy_sample(logits);
    }

    // Scale logits by 1/temperature.
    let inv_temp = 1.0 / params.temperature;
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l * inv_temp).collect();

    // Top-k masking: set all but the k highest logits to -inf before softmax.
    if let Some(k) = params.top_k {
        let k = k.min(scaled.len());
        if k > 0 {
            // Find the k-th largest value as a threshold.
            let mut sorted = scaled.clone();
            sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k - 1];
            for v in &mut scaled {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Softmax over scaled (numerically stable).
    let max_v = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        // All -inf: uniform fallback
        vec![1.0 / logits.len() as f32; logits.len()]
    };

    // Build RNG (seeded for reproducibility when params.seed is set).
    let mut rng = match params.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None    => StdRng::from_entropy(),
    };

    // top_p < 1.0 → nucleus sampling; otherwise sample from full distribution.
    if params.top_p < 1.0 {
        top_p_sample(&probs, params.top_p, &mut rng)
    } else {
        // Full-distribution categorical sample.
        let threshold: f32 = rng.r#gen::<f32>();
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if acc >= threshold {
                return Ok(i as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::inference::params::InferParams;

    // ── 14.4 greedy_sample ────────────────────────────────────────────────────

    #[test]
    fn greedy_returns_argmax_token() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.7];
        assert_eq!(greedy_sample(&logits).unwrap(), 1);
    }

    #[test]
    fn greedy_returns_argmax_at_end() {
        let logits = vec![0.1f32, 0.2, 0.3, 5.0];
        assert_eq!(greedy_sample(&logits).unwrap(), 3);
    }

    #[test]
    fn greedy_single_element_returns_zero() {
        assert_eq!(greedy_sample(&[42.0f32]).unwrap(), 0);
    }

    #[test]
    fn greedy_empty_returns_inference_error() {
        let err = greedy_sample(&[]).unwrap_err();
        assert!(matches!(err, GwenError::InferenceError { .. }));
    }

    // ── 14.4 top_p_sample ────────────────────────────────────────────────────

    #[test]
    fn top_p_sample_always_returns_valid_index() {
        // Uniform distribution, p=0.5 — any token in the top half is valid.
        let probs = vec![0.25f32, 0.25, 0.25, 0.25];
        let mut rng = StdRng::seed_from_u64(42);
        let tok = top_p_sample(&probs, 0.5, &mut rng).unwrap();
        assert!((tok as usize) < probs.len());
    }

    #[test]
    fn top_p_sample_with_p_one_covers_full_distribution() {
        // With p=1.0 any token can be selected; just confirm no panic / OOB.
        let probs = vec![0.1f32, 0.4, 0.3, 0.2];
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20 {
            let tok = top_p_sample(&probs, 1.0, &mut rng).unwrap();
            assert!((tok as usize) < probs.len());
        }
    }

    #[test]
    fn top_p_sample_concentrated_mass_returns_high_prob_token() {
        // Token 2 has 99% of the mass → virtually always selected.
        let probs = vec![0.003f32, 0.003, 0.99, 0.004];
        let mut rng = StdRng::seed_from_u64(0);
        let mut counts = [0usize; 4];
        for _ in 0..100 {
            let tok = top_p_sample(&probs, 0.95, &mut rng).unwrap() as usize;
            counts[tok] += 1;
        }
        // Token 2 should be selected in the vast majority of runs.
        assert!(counts[2] > 90, "expected token 2 to dominate, got counts {counts:?}");
    }

    #[test]
    fn top_p_sample_empty_returns_inference_error() {
        let mut rng = StdRng::seed_from_u64(0);
        let err = top_p_sample(&[], 0.9, &mut rng).unwrap_err();
        assert!(matches!(err, GwenError::InferenceError { .. }));
    }

    // ── 14.4 sample_token dispatcher ─────────────────────────────────────────

    #[test]
    fn sample_token_temperature_zero_is_greedy() {
        let logits = vec![0.1f32, 5.0, 0.3, 0.2];
        let params = InferParams {
            temperature: 0.0,
            ..InferParams::default()
        };
        // Greedy should always return token 1 (highest logit).
        assert_eq!(sample_token(&logits, &params).unwrap(), 1);
    }

    #[test]
    fn sample_token_seeded_is_reproducible() {
        let logits: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let params = InferParams {
            temperature: 1.0,
            top_p: 0.9,
            seed: Some(1234),
            ..InferParams::default()
        };
        let a = sample_token(&logits, &params).unwrap();
        let b = sample_token(&logits, &params).unwrap();
        assert_eq!(a, b, "same seed should produce same token");
    }

    #[test]
    fn sample_token_top_k_limits_candidates() {
        // Logits heavily favour token 31, but top_k=3 caps at the 3 highest.
        // Token 31 is within top-3, so it should still be selectable.
        let mut logits = vec![0.0f32; 32];
        logits[29] = 1.0;
        logits[30] = 2.0;
        logits[31] = 10.0; // clear winner
        let params = InferParams {
            temperature: 0.01, // near-greedy but stochastic path
            top_p: 1.0,
            top_k: Some(3),
            seed: Some(99),
            ..InferParams::default()
        };
        // With top_k=3 and near-zero temp, token 31 should dominate.
        let mut counts = [0usize; 32];
        for _ in 0..50 {
            let tok = sample_token(&logits, &params).unwrap() as usize;
            counts[tok] += 1;
        }
        // Token 31 should win most of the time.
        assert!(counts[31] > 40, "expected token 31 to dominate, got counts at 31={}", counts[31]);
    }

    #[test]
    fn sample_token_temperature_scaling_affects_distribution() {
        // High temperature → more uniform; low temperature → more peaked.
        // Run many times with both and compare entropy proxy (unique tokens).
        let logits: Vec<f32> = vec![0.1f32, 0.5, 2.0, 0.3, 0.8, 1.2, 0.4, 0.7];
        let high_temp = InferParams {
            temperature: 1.5,
            top_p: 1.0,
            seed: Some(42),
            ..InferParams::default()
        };
        let low_temp = InferParams {
            temperature: 0.1,
            top_p: 1.0,
            seed: Some(42),
            ..InferParams::default()
        };

        let mut high_unique = std::collections::HashSet::new();
        let mut low_unique = std::collections::HashSet::new();
        for i in 0..200 {
            let params_h = InferParams { seed: Some(42 + i), ..high_temp.clone() };
            let params_l = InferParams { seed: Some(42 + i), ..low_temp.clone() };
            high_unique.insert(sample_token(&logits, &params_h).unwrap());
            low_unique.insert(sample_token(&logits, &params_l).unwrap());
        }
        // High temperature should produce more diversity.
        assert!(
            high_unique.len() >= low_unique.len(),
            "high temp should produce at least as many unique tokens: high={} low={}",
            high_unique.len(), low_unique.len()
        );
    }

    #[test]
    fn sample_token_empty_logits_returns_inference_error() {
        let params = InferParams::default();
        let err = sample_token(&[], &params).unwrap_err();
        assert!(matches!(err, GwenError::InferenceError { .. }));
    }
}
