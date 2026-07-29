//! `SsaValue` — an SSA name paired with the shape it was produced at
//! (ARTX02 §4).

use crate::stablehlo::emitter::SsaName;
use crate::stablehlo::types::{DType, Shape};

/// A typed SSA value: `%vN` plus the output type of the op that produced it.
///
/// ⚠️ `Clone`, not `Copy` — `Shape::dims` is a `Vec<usize>`. ARTX02 §4 notes
/// the clone is a small heap allocation at rank ≤ 5 and that switching `dims`
/// to a `SmallVec` is the fix *if profiling ever shows it*. There is no
/// measurement yet, so there is no change yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsaValue {
    pub name: SsaName,
    pub shape: Shape,
}

impl SsaValue {
    pub fn new(name: SsaName, shape: Shape) -> Self {
        SsaValue { name, shape }
    }

    pub fn ssa(&self) -> SsaName {
        self.name
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.shape.dtype
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Size of dimension `i`.
    ///
    /// # Panics
    /// If `i` is out of range. Shape indexing errors are trace-time
    /// programming errors (ARTX02 §5), not runtime conditions.
    pub fn dim(&self, i: usize) -> usize {
        assert!(
            i < self.rank(),
            "dim({i}) out of range for {} (rank {})",
            self.shape.mlir_type(),
            self.rank()
        );
        self.shape.dims[i]
    }
}

impl std::fmt::Display for SsaValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} : {}", self.name, self.shape.mlir_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_shows_name_and_mlir_type() {
        let v = SsaValue::new(SsaName(7), Shape::new([4, 8], DType::BF16));
        assert_eq!(v.to_string(), "%v7 : tensor<4x8xbf16>");
    }

    #[test]
    #[should_panic(expected = "dim(2) out of range")]
    fn out_of_range_dim_names_the_shape_it_failed_on() {
        let v = SsaValue::new(SsaName(0), Shape::new([4, 8], DType::F32));
        let _ = v.dim(2);
    }
}
