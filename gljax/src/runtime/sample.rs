//! Sampling — argmax only (ARTX05's "minimum to get a token out").
//!
//! ⚠️ Full sampling is ARTX14: temperature, top-k, top-p, penalties, and the
//! device/host split that keeps a 38.9 MB logits transfer from happening every
//! step. None of that is here. Argmax is what the correctness gate needs, and
//! shipping a half-sampler now would mean two implementations to reconcile
//! later.

/// Index of the largest logit.
///
/// ⭐ **Ties break toward the lower index**, matching `argmax` in NumPy, PyTorch
/// and llama.cpp. This matters for the ARTX12 oracle comparison: two engines
/// that break ties differently diverge on the first tie and never re-converge,
/// and the divergence looks like a numerics bug rather than a convention
/// mismatch.
///
/// Returns `None` for an empty slice.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        // NaN never compares greater, so a NaN logit is skipped rather than
        // winning by accident.
        match best {
            None if !v.is_nan() => best = Some((i, v)),
            Some((_, bv)) if v > bv => best = Some((i, v)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
}

/// Argmax over one row of a `[1, seq, vocab]` logits buffer.
///
/// The logits come back flattened. Reading the wrong row is easy and silent:
/// with right padding, the row that matters is the **last real token**, not the
/// last row of the bucket.
///
/// # Errors
/// If the buffer is not exactly `seq_len × vocab`, or `position` is out of
/// range — both of which mean the caller and the compiled program disagree
/// about the shape.
pub fn argmax_at(
    logits: &[f32],
    seq_len: usize,
    vocab: usize,
    position: usize,
) -> Result<usize, crate::GlError> {
    if logits.len() != seq_len * vocab {
        return Err(crate::GlError::ShapeMismatch {
            expected: vec![seq_len, vocab],
            got: vec![logits.len()],
        });
    }
    if position >= seq_len {
        return Err(crate::GlError::Engine(format!(
            "argmax_at: position {position} out of range for seq_len {seq_len}"
        )));
    }
    let row = &logits[position * vocab..(position + 1) * vocab];
    argmax(row).ok_or_else(|| {
        crate::GlError::Engine("argmax_at: empty vocabulary".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_finds_the_largest_logit() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), Some(1));
        assert_eq!(argmax(&[-5.0, -1.0, -3.0]), Some(1));
        assert_eq!(argmax(&[2.0]), Some(0));
        assert_eq!(argmax(&[]), None);
    }

    /// ⭐ Tie-breaking is a convention, and disagreeing with the oracle on it
    /// produces a divergence that looks like a numerics bug.
    #[test]
    fn ties_break_toward_the_lower_index() {
        assert_eq!(argmax(&[1.0, 1.0, 1.0]), Some(0));
        assert_eq!(argmax(&[0.5, 2.0, 2.0]), Some(1));
    }

    /// A NaN must not win. `NaN > x` is false, so a naive fold that seeds with
    /// the first element would return index 0 forever once a NaN appears there.
    #[test]
    fn a_nan_logit_never_wins() {
        assert_eq!(argmax(&[f32::NAN, 1.0, 2.0]), Some(2));
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), Some(2));
        assert_eq!(argmax(&[2.0, f32::NAN, 1.0]), Some(0));
        // All-NaN has no defensible answer; None is the honest one.
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), None);
    }

    #[test]
    fn argmax_at_reads_the_requested_row() {
        // seq 3, vocab 2: rows [0,1], [5,4], [2,9]
        let logits = [0.0, 1.0, 5.0, 4.0, 2.0, 9.0];
        assert_eq!(argmax_at(&logits, 3, 2, 0).unwrap(), 1);
        assert_eq!(argmax_at(&logits, 3, 2, 1).unwrap(), 0);
        assert_eq!(argmax_at(&logits, 3, 2, 2).unwrap(), 1);
    }

    #[test]
    fn argmax_at_refuses_a_buffer_that_is_not_the_declared_shape() {
        let logits = [0.0; 5];
        assert!(argmax_at(&logits, 3, 2, 0).is_err(), "5 != 3*2");
        let ok = [0.0; 6];
        assert!(argmax_at(&ok, 3, 2, 3).is_err(), "position out of range");
    }
}
