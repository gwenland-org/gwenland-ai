//! `Tensor` — the public tracing handle (ARTX02 §7).
//!
//! Model code only ever touches `Tensor`. It pairs an
//! [`SsaValue`](crate::graph::SsaValue) — its identity in the graph — with a
//! shared handle to the [`FuncBuilder`] that ops get pushed into.
//!
//! ⚠️ A `Tensor` is a **handle, not a buffer**. Nothing here holds data;
//! `x.add(&y)` emits a line of MLIR and returns a name for its result. The
//! numbers only exist once PJRT runs the compiled program.

use std::cell::{RefCell, RefMut};
use std::rc::Rc;

use crate::graph::builder::FuncBuilder;
use crate::graph::value::SsaValue;
use crate::stablehlo::ops::DotDimensionNumbers;
use crate::stablehlo::types::{DType, Shape};

/// A traced tensor.
pub struct Tensor {
    value: SsaValue,
    builder: Rc<RefCell<FuncBuilder>>,
}

impl Tensor {
    pub fn new(value: SsaValue, builder: Rc<RefCell<FuncBuilder>>) -> Self {
        Tensor { value, builder }
    }

    pub fn value(&self) -> &SsaValue {
        &self.value
    }

    pub fn shape(&self) -> &Shape {
        self.value.shape()
    }

    pub fn dtype(&self) -> DType {
        self.value.dtype()
    }

    pub fn rank(&self) -> usize {
        self.value.rank()
    }

    pub fn dim(&self, i: usize) -> usize {
        self.value.dim(i)
    }

    /// The builder, for ops that need to emit directly.
    pub fn builder(&self) -> &Rc<RefCell<FuncBuilder>> {
        &self.builder
    }

    /// A second handle to the same graph node. Emits nothing — this is how a
    /// residual connection keeps a reference to its input.
    pub fn clone_ref(&self) -> Tensor {
        Tensor::new(self.value.clone(), Rc::clone(&self.builder))
    }

    fn b(&self) -> RefMut<'_, FuncBuilder> {
        self.builder.borrow_mut()
    }

    fn wrap(&self, value: SsaValue) -> Tensor {
        Tensor::new(value, Rc::clone(&self.builder))
    }

    // ── Elementwise ────────────────────────────────────────────────────────

    pub fn add(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().add(&self.value, &rhs.value);
        self.wrap(v)
    }

    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().subtract(&self.value, &rhs.value);
        self.wrap(v)
    }

    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().multiply(&self.value, &rhs.value);
        self.wrap(v)
    }

    pub fn div(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().divide(&self.value, &rhs.value);
        self.wrap(v)
    }

    pub fn max(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().maximum(&self.value, &rhs.value);
        self.wrap(v)
    }

    pub fn min(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().minimum(&self.value, &rhs.value);
        self.wrap(v)
    }

    pub fn neg(&self) -> Tensor {
        let v = self.b().negate(&self.value);
        self.wrap(v)
    }

    pub fn rsqrt(&self) -> Tensor {
        let v = self.b().rsqrt(&self.value);
        self.wrap(v)
    }

    pub fn sqrt(&self) -> Tensor {
        let v = self.b().sqrt(&self.value);
        self.wrap(v)
    }

    /// Sigmoid — `stablehlo.logistic`.
    pub fn logistic(&self) -> Tensor {
        let v = self.b().logistic(&self.value);
        self.wrap(v)
    }

    pub fn exp(&self) -> Tensor {
        let v = self.b().exponential(&self.value);
        self.wrap(v)
    }

    pub fn log(&self) -> Tensor {
        let v = self.b().log(&self.value);
        self.wrap(v)
    }

    pub fn tanh(&self) -> Tensor {
        let v = self.b().tanh(&self.value);
        self.wrap(v)
    }

    pub fn abs(&self) -> Tensor {
        let v = self.b().abs(&self.value);
        self.wrap(v)
    }

    /// SiLU / swish — `x * sigmoid(x)`.
    ///
    /// Built from `logistic` and `multiply` rather than approximated, because
    /// ARTX01 §7.5 records that this exact pair is what XLA's epilogue fusion
    /// recognises. An approximation would be both less accurate and slower.
    pub fn silu(&self) -> Tensor {
        let sigmoid = self.logistic();
        self.mul(&sigmoid)
    }

    // ── Shape ──────────────────────────────────────────────────────────────

    pub fn reshape(&self, new_dims: Vec<usize>) -> Tensor {
        let v = self.b().reshape(&self.value, new_dims);
        self.wrap(v)
    }

    pub fn transpose(&self, permutation: Vec<usize>) -> Tensor {
        let v = self.b().transpose(&self.value, permutation);
        self.wrap(v)
    }

    pub fn slice(&self, starts: Vec<usize>, limits: Vec<usize>, strides: Vec<usize>) -> Tensor {
        let v = self.b().slice(&self.value, starts, limits, strides);
        self.wrap(v)
    }

    /// `broadcast_dims[i]` is the **output** axis that input axis `i` maps to.
    pub fn broadcast_to(&self, broadcast_dims: Vec<usize>, out_dims: Vec<usize>) -> Tensor {
        let v = self
            .b()
            .broadcast_in_dim(&self.value, broadcast_dims, out_dims);
        self.wrap(v)
    }

    /// Concatenates along `dimension`. All tensors must come from the same
    /// trace.
    ///
    /// # Panics
    /// If `tensors` is empty.
    pub fn concat(tensors: &[&Tensor], dimension: usize) -> Tensor {
        assert!(!tensors.is_empty(), "concat: no tensors");
        let values: Vec<&SsaValue> = tensors.iter().map(|t| &t.value).collect();
        let v = tensors[0].b().concatenate(&values, dimension);
        tensors[0].wrap(v)
    }

    // ── Precision ──────────────────────────────────────────────────────────

    /// Inserts a `stablehlo.convert`, or returns a fresh handle to the same
    /// value if it is already at `dtype`.
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if self.dtype() == dtype {
            return self.clone_ref();
        }
        let v = self.b().convert(&self.value, dtype);
        self.wrap(v)
    }

    // ── Matmul ─────────────────────────────────────────────────────────────

    pub fn dot_general(&self, rhs: &Tensor, dnums: &DotDimensionNumbers) -> Tensor {
        let v = self.b().dot_general(&self.value, &rhs.value, dnums);
        self.wrap(v)
    }

    /// Batched matmul over the trailing two dimensions. See
    /// [`FuncBuilder::matmul`] for how the batch dimensions are chosen.
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().matmul(&self.value, &rhs.value);
        self.wrap(v)
    }

    // ── Reduce ─────────────────────────────────────────────────────────────

    /// Sums over `dims`, dropping them from the result.
    pub fn reduce_sum(&self, dims: &[usize]) -> Tensor {
        let init = {
            let mut b = self.b();
            b.constant_scalar(0.0, self.dtype())
        };
        let v = self.b().reduce_add(&self.value, &init, dims);
        self.wrap(v)
    }

    /// Maximum over `dims`, dropping them from the result.
    ///
    /// ⚠️ `init` is the caller's problem, and it must be a value no element can
    /// exceed. The mathematically right choice is −∞, which
    /// [`crate::stablehlo::ops::emit_constant_splat`] currently refuses to
    /// spell (Wave A3 adds the hex literal). Until then, pass the init
    /// explicitly via [`FuncBuilder::reduce_max`].
    pub fn reduce_max_with_init(&self, init: &Tensor, dims: &[usize]) -> Tensor {
        let v = self.b().reduce_max(&self.value, &init.value, dims);
        self.wrap(v)
    }
}

// ── Operator overloads ──────────────────────────────────────────────────────
//
// Defined on `&Tensor` rather than `Tensor` so `&a + &b` reads naturally in
// model code without moving handles that are usually needed again (a residual
// connection uses its input twice by construction).

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        Tensor::add(self, rhs)
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        Tensor::sub(self, rhs)
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        Tensor::mul(self, rhs)
    }
}

impl std::ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        Tensor::div(self, rhs)
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({})", self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;

    fn cx() -> TraceCx {
        TraceCx::new("main", "t")
    }

    #[test]
    fn operator_overloads_emit_the_same_ops_as_the_methods() {
        let mut c = cx();
        let a = c.input("a", Shape::new([4], DType::F32));
        let b = c.input("b", Shape::new([4], DType::F32));
        let sum = &a + &b;
        let built = c.finish(&[&sum]);
        assert_eq!(built.mlir.matches("stablehlo.add").count(), 1, "{}", built.mlir);
    }

    #[test]
    fn silu_is_logistic_times_the_input() {
        let mut c = cx();
        let x = c.input("x", Shape::new([2, 4], DType::F32));
        let y = x.silu();
        let built = c.finish(&[&y]);
        assert!(built.mlir.contains(r#""stablehlo.logistic"(%v0)"#), "{}", built.mlir);
        assert!(
            built.mlir.contains(r#""stablehlo.multiply"(%v0, %v1)"#),
            "silu must multiply the input by its own sigmoid:\n{}",
            built.mlir
        );
    }

    #[test]
    fn clone_ref_emits_nothing_and_keeps_the_same_ssa_name() {
        let mut c = cx();
        let x = c.input("x", Shape::new([4], DType::F32));
        let again = x.clone_ref();
        assert_eq!(again.value(), x.value());
        let built = c.finish(&[&x]);
        // Only the func.func header and the return — no ops at all.
        assert!(
            !built.mlir.contains("stablehlo."),
            "clone_ref emitted an op:\n{}",
            built.mlir
        );
    }

    #[test]
    fn to_dtype_is_a_no_op_when_already_at_the_target() {
        let mut c = cx();
        let x = c.input("x", Shape::new([4], DType::BF16));
        let same = x.to_dtype(DType::BF16);
        let widened = same.to_dtype(DType::F32);
        let built = c.finish(&[&widened]);
        assert_eq!(
            built.mlir.matches("stablehlo.convert").count(),
            1,
            "only the bf16->f32 step should emit a convert:\n{}",
            built.mlir
        );
    }

    #[test]
    fn reduce_sum_drops_the_reduced_axis() {
        let mut c = cx();
        let x = c.input("x", Shape::new([2, 6], DType::F32));
        let s = x.reduce_sum(&[1]);
        assert_eq!(s.shape().dims, vec![2]);
        let built = c.finish(&[&s]);
        assert!(built.mlir.contains(r#""stablehlo.reduce""#), "{}", built.mlir);
    }

    #[test]
    fn concat_sums_the_concatenated_axis() {
        let mut c = cx();
        let a = c.input("a", Shape::new([2, 3], DType::F32));
        let b = c.input("b", Shape::new([2, 5], DType::F32));
        let j = Tensor::concat(&[&a, &b], 1);
        assert_eq!(j.shape().dims, vec![2, 8]);
    }
}
