// engine/inference/sampler.rs — Token sampling strategies.
//
// Implements three complementary sampling methods that are applied in sequence
// after each forward pass:
//
//   1. Repetition penalty — down-weights logits for tokens already generated.
//      Prevents the model from looping ("the the the the…").
//
//   2. Temperature scaling — divides all logits by T before softmax.
//      T = 0.0 → greedy (argmax); T = 1.0 → unscaled; T > 1.0 → more random.
//
//   3. Top-p nucleus sampling — keeps only the smallest set of tokens whose
//      cumulative probability ≥ p, zeroing the rest before sampling.
//      Prevents extremely low-probability tokens from being picked.
//
// Why not top-k?
// Top-p is strictly more principled: it adapts to the shape of each
// distribution rather than hard-coding a fixed count. Top-k is provided
// as a comment-out option if needed later.

use anyhow::{bail, Result};
use candle_core::Tensor;

/// Parameters controlling how the next token is selected.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            max_tokens: 512,
        }
    }
}

impl SamplerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 {
            bail!("--temperature must be >= 0.0");
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            bail!("--top-p must be in [0.0, 1.0]");
        }
        if self.repeat_penalty < 1.0 {
            bail!("--repeat-penalty must be >= 1.0 (1.0 = disabled)");
        }
        Ok(())
    }

    /// True when temperature is effectively zero — use greedy decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-6
    }
}

/// Sample the next token id from a logit tensor.
///
/// `logits`         — shape [vocab_size], raw model output for the last position
/// `previous_tokens`— all token ids generated so far (for repetition penalty)
/// `cfg`            — sampling hyper-parameters
///
/// Returns the sampled token id as `u32`.
pub fn sample(logits: &Tensor, previous_tokens: &[u32], cfg: &SamplerConfig) -> Result<u32> {
    // Work on CPU f32 for the sampling arithmetic.
    let mut logits: Vec<f32> = logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;

    // 1. Repetition penalty — reduce logits for already-seen tokens.
    if cfg.repeat_penalty != 1.0 {
        for &tok in previous_tokens {
            let tok = tok as usize;
            if tok < logits.len() {
                if logits[tok] >= 0.0 {
                    logits[tok] /= cfg.repeat_penalty;
                } else {
                    logits[tok] *= cfg.repeat_penalty;
                }
            }
        }
    }

    // 2. Greedy — argmax, skip temperature / top-p.
    if cfg.is_greedy() {
        return Ok(argmax(&logits));
    }

    // 3. Temperature scaling.
    for l in &mut logits {
        *l /= cfg.temperature;
    }

    // 4. Softmax to get probabilities.
    let probs = softmax(&logits);

    // 5. Top-p nucleus sampling.
    let token_id = top_p_sample(&probs, cfg.top_p)?;

    Ok(token_id)
}

// ── helpers ────────────────────────────────────────────────────────────────────

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn top_p_sample(probs: &[f32], p: f32) -> Result<u32> {
    // Sort indices by descending probability.
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Keep tokens until cumulative mass ≥ p.
    let mut cumulative = 0.0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for (idx, prob) in &indexed {
        cumulative += prob;
        nucleus.push((*idx, *prob));
        if cumulative >= p {
            break;
        }
    }

    // Renormalize the nucleus.
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    if nucleus_sum == 0.0 {
        // Safety net: fall back to greedy.
        return Ok(indexed[0].0 as u32);
    }

    // Sample from the nucleus using a uniform draw.
    let mut rng_val: f32 = rand::random::<f32>() * nucleus_sum;
    for (idx, prob) in &nucleus {
        rng_val -= prob;
        if rng_val <= 0.0 {
            return Ok(*idx as u32);
        }
    }

    Ok(nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_temperature_zero_is_greedy() {
        let cfg = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(cfg.is_greedy());
    }

    #[test]
    fn test_sampler_default_not_greedy() {
        let cfg = SamplerConfig::default();
        assert!(!cfg.is_greedy());
    }

    #[test]
    fn test_argmax_returns_highest() {
        let logits = vec![0.1f32, 0.9, 0.5];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
