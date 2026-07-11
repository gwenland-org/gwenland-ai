//! Numerical validation against the glproc oracle.
//!
//! DESIGN.md names glproc the ground-truth oracle: the pure-Rust scalar engine
//! whose output every accelerated engine must match. This module compares a
//! candidate engine's generated token ids against glproc's for the same prompt
//! and sampling config, reporting the longest matching prefix. It does not run
//! inference itself — the caller supplies both token streams (glbench runs
//! engines through the adapter, not by duplicating inference logic).

/// The result of comparing a candidate's tokens to the oracle's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericalCheck {
    /// Number of leading tokens that matched exactly.
    pub matching_prefix: usize,
    /// Total tokens compared (min of the two lengths).
    pub compared: usize,
}

impl NumericalCheck {
    /// True if every compared token matched.
    pub fn exact(&self) -> bool {
        self.compared > 0 && self.matching_prefix == self.compared
    }

    /// Fraction of compared tokens that matched, 0.0..=1.0.
    pub fn agreement(&self) -> f64 {
        if self.compared == 0 {
            0.0
        } else {
            self.matching_prefix as f64 / self.compared as f64
        }
    }
}

/// Compare a candidate token stream against the oracle's, greedily from the
/// front. Greedy decoding (temperature 0, same seed) should agree token-for-
/// token; divergence marks the first numerically significant difference.
pub fn compare_tokens(oracle: &[u32], candidate: &[u32]) -> NumericalCheck {
    let compared = oracle.len().min(candidate.len());
    let mut matching = 0;
    for i in 0..compared {
        if oracle[i] == candidate[i] {
            matching += 1;
        } else {
            break;
        }
    }
    NumericalCheck { matching_prefix: matching, compared }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let c = compare_tokens(&[1, 2, 3], &[1, 2, 3]);
        assert!(c.exact());
        assert_eq!(c.agreement(), 1.0);
    }

    #[test]
    fn diverges_midway() {
        let c = compare_tokens(&[1, 2, 3, 4], &[1, 2, 9, 4]);
        assert_eq!(c.matching_prefix, 2);
        assert_eq!(c.compared, 4);
        assert!(!c.exact());
    }
}
