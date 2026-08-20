//! Deterministic PRNG for parameter initialization.
//!
//! LoRA needs a Gaussian for its A matrix, and there is no `rand` dependency in
//! this repo: `glproc/src/sampler.rs` writes its own xorshift64* for exactly
//! this reason. Same choice here, same algorithm, plus Box-Muller on top for the
//! normal distribution.
//!
//! Sits outside the four Breton sub-systems (like `error.rs`), so it carries no
//! codename.
//!
//! # Why determinism is a requirement, not a nicety
//!
//! Two separate things depend on it:
//!
//! 1. `testing-standards.md` rule: anything sampling-shaped is only testable as
//!    a deterministic function of (seed, params). An init that cannot be
//!    reproduced cannot be asserted on.
//! 2. VeRA (M3) stores an RNG **seed** in its checkpoint instead of its frozen
//!    random matrices, and regenerates them on load. That only works if the
//!    generator is bit-stable across runs and platforms. This one is: it is
//!    integer arithmetic with a fixed multiplier and no floating-point state.

/// xorshift64* PRNG. Deterministic, seedable, no dependency.
///
/// Not cryptographic and not trying to be. It is a weight initializer.
#[derive(Debug, Clone)]
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// Seed the generator. A zero seed is bumped to 1, since xorshift is stuck
    /// at zero forever if it ever reaches it.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f32` in `[0, 1)`.
    ///
    /// Takes the top 24 bits, which is exactly the f32 mantissa width, so every
    /// value is representable and the spacing is uniform. Using the low bits
    /// instead would expose xorshift's weaker low-order behaviour.
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64();
        ((bits >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Uniform `f32` in `(0, 1)`, both ends excluded.
    ///
    /// Box-Muller takes `ln(u)`, so a `u` of exactly 0 would give `-inf`. This
    /// nudges the closed end open rather than rejection-sampling, which would
    /// make the stream length depend on the values drawn and break
    /// reproducibility across a code change that reorders draws.
    fn next_f32_open(&mut self) -> f32 {
        // 2^-24 is the smallest positive value next_f32 can return.
        const TINY: f32 = 1.0 / ((1u64 << 24) as f32);
        let u = self.next_f32();
        if u <= 0.0 {
            TINY
        } else {
            u
        }
    }

    /// One sample from `N(0, 1)` via the Box-Muller transform.
    ///
    /// Box-Muller produces samples in pairs; this discards the second. That
    /// wastes half the draws and is entirely fine for initializing a matrix
    /// once. Caching the spare would make the output depend on how many times
    /// the function had been called before, which is a worse property than being
    /// slow at startup.
    pub fn next_standard_normal(&mut self) -> f32 {
        let u1 = self.next_f32_open();
        let u2 = self.next_f32();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    /// `n` samples from `N(0, std^2)`.
    pub fn normal_vec(&mut self, n: usize, std: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_standard_normal() * std).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mean of N samples of N(0,1) has standard error 1/sqrt(N). At
    /// N = 100_000 that is ~0.0032, so 4 standard errors is ~0.013. A tolerance
    /// of 0.02 passes a correct generator essentially always while still
    /// catching a systematic offset.
    const TOL_MEAN: f32 = 0.02;
    /// Same reasoning for the variance estimate, which converges more slowly.
    const TOL_VAR: f32 = 0.05;
    /// Uniform draws are exact rationals over 2^24; comparing two runs is a
    /// bit-for-bit check, so no slack is needed.
    const TOL_EXACT: f32 = 0.0;

    #[test]
    fn same_seed_reproduces_the_same_stream() {
        let mut a = Xorshift64Star::new(42);
        let mut b = Xorshift64Star::new(42);
        for _ in 0..1000 {
            assert!((a.next_f32() - b.next_f32()).abs() <= TOL_EXACT);
        }
    }

    #[test]
    fn different_seeds_produce_different_streams() {
        let mut a = Xorshift64Star::new(1);
        let mut b = Xorshift64Star::new(2);
        let differing = (0..100)
            .filter(|_| (a.next_f32() - b.next_f32()).abs() > 1e-9)
            .count();
        assert!(differing > 90, "only {differing}/100 draws differed");
    }

    #[test]
    fn zero_seed_does_not_lock_the_generator_at_zero() {
        let mut rng = Xorshift64Star::new(0);
        let first = rng.next_u64();
        let second = rng.next_u64();
        assert_ne!(first, 0, "a zero seed must be bumped, not accepted");
        assert_ne!(first, second);
    }

    #[test]
    fn uniform_draws_stay_in_unit_interval() {
        let mut rng = Xorshift64Star::new(7);
        for _ in 0..10_000 {
            let u = rng.next_f32();
            assert!((0.0..1.0).contains(&u), "u = {u} escaped [0,1)");
        }
    }

    #[test]
    fn standard_normal_has_unit_mean_and_variance() {
        let mut rng = Xorshift64Star::new(2026);
        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| rng.next_standard_normal()).collect();
        let mean = samples.iter().sum::<f32>() / n as f32;
        // f64 accumulation: 100k f32 squares summed in f32 loses enough
        // precision to move the third decimal, which is inside the tolerance
        // being asserted.
        let var = samples
            .iter()
            .map(|x| {
                let d = (*x - mean) as f64;
                d * d
            })
            .sum::<f64>() as f32
            / n as f32;
        assert!(mean.abs() < TOL_MEAN, "mean = {mean}");
        assert!((var - 1.0).abs() < TOL_VAR, "var = {var}");
    }

    #[test]
    fn normal_vec_scales_by_std() {
        let mut rng = Xorshift64Star::new(99);
        let n = 100_000;
        let std = 0.02;
        let v = rng.normal_vec(n, std);
        assert_eq!(v.len(), n);
        let mean = v.iter().sum::<f32>() / n as f32;
        let var = v
            .iter()
            .map(|x| {
                let d = (*x - mean) as f64;
                d * d
            })
            .sum::<f64>() as f32
            / n as f32;
        // Variance scales by std^2, so compare the ratio rather than the value.
        let ratio = var / (std * std);
        assert!((ratio - 1.0).abs() < TOL_VAR, "var/std^2 = {ratio}");
    }

    #[test]
    fn normal_draws_are_finite() {
        // Guards the ln(0) -> -inf path in Box-Muller. next_f32_open exists
        // solely to make this test unable to fail.
        let mut rng = Xorshift64Star::new(1);
        for _ in 0..50_000 {
            let x = rng.next_standard_normal();
            assert!(x.is_finite(), "non-finite sample {x}");
        }
    }
}
