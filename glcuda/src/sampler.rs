//! Token sampling: greedy, temperature, top-k and top-p (nucleus).
//!
//! Engine-owned copy of glproc's sampler (ADR-001: sampler logic is
//! duplicated per engine rather than shared). Same pipeline, same
//! xorshift64* RNG, same seeding — a fixed seed draws the same tokens from
//! the same probabilities on both engines. One deliberate difference: the
//! candidate softmax uses exact `exp` (glproc routes through its SIMD
//! `fast_exp`, whose ~1e-4 approximation is a CPU-side throughput trade
//! this engine does not need to copy).

/// Sampling hyperparameters.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// `1.0` = no change; `0.0` (or below) falls back to greedy.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens. `0` = disabled.
    pub top_k: usize,
    /// Nucleus sampling probability mass. `1.0` = disabled.
    pub top_p: f32,
    /// Repetition penalty on recently generated tokens. `1.0` = disabled.
    pub repeat_penalty: f32,
    /// RNG seed; `None` seeds from the system clock.
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: None,
        }
    }
}

/// Apply a repetition penalty to `logits` in place (per occurrence, like
/// glproc): `penalty > 1.0` shrinks positive logits, amplifies negative.
pub fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &token_id in recent_tokens {
        let idx = token_id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Numerically stable in-place softmax (exact `exp`).
fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let n = x.len() as f32;
        x.fill(1.0 / n);
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// xorshift64* — tiny deterministic PRNG, identical to glproc's.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64 { state: seed.max(1) }
    }

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((bits >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Stateful token sampler.
pub struct Sampler {
    config: SamplerConfig,
    rng: XorShift64,
    /// `(id, scaled logit)` working set, reused so steady-state sampling
    /// allocates nothing.
    candidates: Vec<(usize, f32)>,
    /// Softmax scratch matching `candidates`.
    probs: Vec<f32>,
}

impl Sampler {
    /// Create a sampler from a config, seeding the RNG.
    pub fn new(config: SamplerConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5EED)
        });
        Sampler {
            config,
            rng: XorShift64::new(seed),
            candidates: Vec::new(),
            probs: Vec::new(),
        }
    }

    /// The configured repetition penalty (`1.0` = disabled).
    pub fn repeat_penalty(&self) -> f32 {
        self.config.repeat_penalty
    }

    /// Greedy sampling: always pick the argmax logit.
    pub fn greedy(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best as u32
    }

    /// Sample the next token id from a full-vocabulary logits slice.
    /// Pipeline: temperature → top-k → softmax → top-p → weighted draw.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if self.config.temperature <= 0.0 {
            return Self::greedy(logits);
        }

        let inv_temp = 1.0 / self.config.temperature;
        let candidates = &mut self.candidates;
        candidates.clear();
        candidates.extend(logits.iter().enumerate().map(|(i, &v)| (i, v * inv_temp)));

        // top-k via O(n) partition selection, then sort only the survivors.
        let k = self.config.top_k;
        if k > 0 && k < candidates.len() {
            candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let probs = &mut self.probs;
        probs.clear();
        probs.extend(candidates.iter().map(|&(_, v)| v));
        softmax(probs);

        // top-p: cut the sorted tail once cumulative mass exceeds p.
        let p = self.config.top_p;
        if p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut cutoff = probs.len();
            for (i, &pr) in probs.iter().enumerate() {
                cumulative += pr;
                if cumulative >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            candidates.truncate(cutoff);
            let total: f32 = probs.iter().sum();
            if total > 0.0 {
                for pr in probs.iter_mut() {
                    *pr /= total;
                }
            }
        }

        let roll = self.rng.next_f32();
        let mut cumulative = 0.0f32;
        for (i, &pr) in probs.iter().enumerate() {
            cumulative += pr;
            if roll < cumulative {
                return candidates[i].0 as u32;
            }
        }
        candidates.last().map(|&(i, _)| i as u32).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        assert_eq!(Sampler::greedy(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(Sampler::greedy(&[5.0, -1.0, 4.9]), 0);
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: Some(42),
        });
        for _ in 0..10 {
            assert_eq!(s.sample(&[0.1, 0.9, 0.3]), 1);
        }
    }

    #[test]
    fn seeded_sampling_matches_glproc() {
        // Same seed, same logits → same draws as the glproc sampler. The
        // exact-exp softmax difference cannot flip a draw here because the
        // probabilities are far from any roll boundary.
        let logits = [1.0, 2.0, 3.0, 0.5];
        let mut ours = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: Some(123),
        });
        let mut theirs = glproc::sampler::Sampler::new(glproc::sampler::SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: Some(123),
        });
        for _ in 0..20 {
            assert_eq!(ours.sample(&logits), theirs.sample(&logits));
        }
    }

    #[test]
    fn repetition_penalty_matches_glproc_semantics() {
        let mut a = vec![2.0f32, -2.0, 1.0];
        let mut b = a.clone();
        apply_repetition_penalty(&mut a, &[0, 1, 57], 2.0);
        glproc::sampler::apply_repetition_penalty(&mut b, &[0, 1, 57], 2.0);
        assert_eq!(a, b);
    }
}
