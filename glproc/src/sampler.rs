//! Token sampling: greedy, temperature, top-k and top-p (nucleus).

use crate::attention::softmax;

/// Sampling hyperparameters.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// `1.0` = no change; `0.0` (or below) falls back to greedy.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens. `0` = disabled.
    pub top_k: usize,
    /// Nucleus sampling probability mass. `1.0` = disabled.
    pub top_p: f32,
    /// RNG seed; `None` seeds from the system clock.
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            seed: None,
        }
    }
}

/// xorshift64* — a tiny, deterministic PRNG; no `rand` dependency.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64 {
            state: seed.max(1), // xorshift must not start at 0
        }
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
        }
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
    ///
    /// Pipeline: temperature scaling → top-k filter → softmax → top-p
    /// (nucleus) truncation → weighted draw.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        if self.config.temperature <= 0.0 {
            return Self::greedy(logits);
        }

        // (id, logit) working set, temperature applied.
        let inv_temp = 1.0 / self.config.temperature;
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v * inv_temp))
            .collect();

        // top-k: keep only the k highest logits.
        let k = self.config.top_k;
        if k > 0 && k < candidates.len() {
            candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            candidates.truncate(k);
        } else {
            candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        }

        // softmax over the surviving logits.
        let mut probs: Vec<f32> = candidates.iter().map(|&(_, v)| v).collect();
        softmax(&mut probs);

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

        // weighted draw.
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
            seed: Some(42),
        });
        for _ in 0..10 {
            assert_eq!(s.sample(&[0.1, 0.9, 0.3]), 1);
        }
    }

    #[test]
    fn top_k_one_is_deterministic() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: Some(7),
        });
        for _ in 0..10 {
            assert_eq!(s.sample(&[1.0, 3.0, 2.0]), 1);
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(123),
        };
        let logits = [1.0, 2.0, 3.0, 0.5];
        let a: Vec<u32> = {
            let mut s = Sampler::new(cfg.clone());
            (0..20).map(|_| s.sample(&logits)).collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(cfg);
            (0..20).map(|_| s.sample(&logits)).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn dominant_logit_wins_with_nucleus() {
        // One token holds ~all probability mass; top-p must keep it.
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            seed: Some(99),
        });
        let mut logits = vec![0.0f32; 100];
        logits[37] = 50.0;
        for _ in 0..20 {
            assert_eq!(s.sample(&logits), 37);
        }
    }
}
