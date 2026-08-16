//! Stummañ Kevrin — Tensor abstraction.
//!
//! `Tensor<B>` is the core data structure for Stummañ.
//! It wraps backend storage with shape metadata.
//! Wave 1: pure data.
//! Wave 2: adds an identity, a `requires_grad` flag, and an optional shared
//! tape, so ops can *record* the forward pass. Still no gradients — no `grad`
//! field, and nothing replays the tape. That is Wave 3.

use crate::autograd::node::{BackwardFn, ComputationNode, TensorId};
use crate::autograd::ops::{matmul_f32, transpose_2d};
use crate::autograd::tape::{Tape, TensorMeta};
use crate::error::{GlTrainError, Result};
use crate::tensor::backend::Backend;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Global tensor ID counter. Each Tensor gets a unique, monotonically
/// increasing ID at creation. Thread-safe via AtomicUsize.
///
/// # ID lifecycle
///
/// **IDs are process-global and never reused.** Every `Tensor` — tracked or
/// not, on any tape, on any backend — draws from this one counter, so an ID
/// identifies a tensor uniquely for the lifetime of the process.
///
/// **[`Tape::clear()`] resets node IDs but tensor IDs keep climbing.** The two
/// counters are unrelated: node IDs are per-tape and restart at 0 on every
/// clear, while this one only ever increases. After ten training steps the
/// tape's nodes are numbered from 0 again but its tensors are in the thousands.
///
/// **Do not persist or compare IDs across process restarts.** They are
/// allocation order, not identity — the same logical weight gets a different ID
/// on the next run. Checkpoints must key on parameter names, never on these.
static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(0);

fn next_tensor_id() -> TensorId {
    NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed)
}

/// Lock the shared tape, recovering the guard if the mutex was poisoned.
///
/// Poisoning means some other thread panicked while holding the lock. The tape
/// is a plain `Vec` plus a `HashMap` with no cross-field invariant that a
/// partial write could corrupt, so reclaiming the guard is sound — and it is
/// strictly better than the alternatives available here: `with_grad` returns
/// `Self`, not `Result`, so it cannot propagate a lock error, which would
/// leave `unwrap()` as the only other option. The project forbids `unwrap()`
/// outside tests.
fn lock_tape(tape: &Arc<Mutex<Tape>>) -> MutexGuard<'_, Tape> {
    tape.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A multi-dimensional tensor backed by compute backend B.
///
/// Shape semantics:
/// - 1D: `[N]`          — vector
/// - 2D: `[M, N]`       — matrix (row-major)
/// - 3D: `[B, M, N]`    — batched matrix
///
/// Storage is reference-counted (Arc) so tensors can be cloned cheaply.
/// Cloning a Tensor shares the underlying storage (copy-on-write NOT implemented
/// in Wave 1 — mutating ops always allocate new storage).
#[derive(Clone)]
pub struct Tensor<B: Backend> {
    /// Shape of the tensor (e.g., [4, 8] for a 4×8 matrix)
    pub(crate) shape: Vec<usize>,

    /// Backend-specific storage (reference-counted for cheap clone)
    pub(crate) storage: Arc<B::Storage>,

    // ── Autograd fields (added Wave 2) ───────────────────────────────
    /// Unique tensor ID, assigned at creation. Stable across moves.
    pub(crate) id: TensorId,

    /// Whether this tensor participates in gradient tracking.
    /// If false, ops on this tensor do not record to the tape.
    pub(crate) requires_grad: bool,

    /// Shared tape reference. `Some(..)` means this tensor is being
    /// tracked; `None` means no-grad mode (inference / frozen weights).
    pub(crate) tape: Option<Arc<Mutex<Tape>>>,
    // ─────────────────────────────────────────────────────────────────
    /// Phantom marker for backend type
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Tensor<B> {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Create a tensor filled with zeros of the given shape.
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let n = shape_to_n_elems(shape)?;
        let storage = B::zeros(n)?;
        Ok(Self::from_storage(storage, shape.to_vec()))
    }

    /// Create a tensor filled with ones of the given shape.
    pub fn ones(shape: &[usize]) -> Result<Self> {
        let n = shape_to_n_elems(shape)?;
        let storage = B::ones(n)?;
        Ok(Self::from_storage(storage, shape.to_vec()))
    }

    /// Create a tensor from a host `Vec<f32>` with the given shape.
    ///
    /// Returns [`GlTrainError::ShapeMismatch`] if `data.len()` disagrees with
    /// the product of `shape`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let n = shape_to_n_elems(shape)?;
        if data.len() != n {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![n],
                got: vec![data.len()],
            });
        }
        let storage = B::from_vec(data)?;
        Ok(Self::from_storage(storage, shape.to_vec()))
    }

    /// Internal constructor from raw storage + shape.
    ///
    /// Every tensor gets a fresh ID here. Grad tracking is opt-in: a tensor is
    /// born detached and only joins a tape via [`Tensor::with_grad`] or by
    /// being produced by a tracked op.
    pub(crate) fn from_storage(storage: B::Storage, shape: Vec<usize>) -> Self {
        Self {
            shape,
            storage: Arc::new(storage),
            id: next_tensor_id(),
            requires_grad: false,
            tape: None,
            _backend: std::marker::PhantomData,
        }
    }

    // ── Shape accessors ──────────────────────────────────────────────────

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn n_elems(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    // ── Data access ──────────────────────────────────────────────────────

    /// Copy tensor data to a host `Vec<f32>`.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        B::to_vec(&self.storage)
    }

    /// Return a single scalar value (only valid for 0-d or 1-elem tensors).
    pub fn item(&self) -> Result<f32> {
        let v = self.to_vec()?;
        if v.len() != 1 {
            return Err(GlTrainError::InvalidOp(format!(
                "item() called on tensor with {} elements (expected 1)",
                v.len()
            )));
        }
        Ok(v[0])
    }

    // ── Autograd API (Wave 2) ─────────────────────────────────────────────

    /// Mark this tensor as requiring gradient tracking, attaching it to `tape`.
    ///
    /// Returns self (builder pattern) for ergonomic use:
    /// `let x = Tensor::from_vec(data, shape)?.with_grad(tape.clone());`
    pub fn with_grad(mut self, tape: Arc<Mutex<Tape>>) -> Self {
        self.requires_grad = true;
        // Register this tensor's metadata in the tape.
        {
            let mut guard = lock_tape(&tape);
            guard.register_tensor(TensorMeta {
                id: self.id,
                shape: self.shape.clone(),
                requires_grad: true,
            });
        }
        self.tape = Some(tape);
        self
    }

    /// Return this tensor's unique ID.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Whether this tensor is being tracked for gradients.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Return a reference to this tensor's tape, if any.
    pub fn tape(&self) -> Option<&Arc<Mutex<Tape>>> {
        self.tape.as_ref()
    }

    /// Detach from the tape — a new Tensor with the same data but
    /// `requires_grad = false` and no tape. Used for frozen base weights
    /// in LoRA (M2+).
    ///
    /// The storage is shared, not copied; only the autograd identity is new.
    pub fn detach(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            storage: self.storage.clone(),
            id: next_tensor_id(), // new ID — a detached tensor is a new leaf
            requires_grad: false,
            tape: None,
            _backend: std::marker::PhantomData,
        }
    }

    /// Record one op on the shared tape and attach that tape to `output`.
    ///
    /// Every tracked op funnels through here so the bookkeeping (resolve the
    /// tape, allocate a node id, register the output, push the node) cannot
    /// drift between call sites. A no-op when no operand is tracked.
    ///
    /// `make_backward` is only invoked when the op is actually being recorded.
    /// Building a backward closure usually means copying the forward operands
    /// out of backend storage, and a no-grad forward pass should not pay for
    /// data nothing will read.
    ///
    /// # All operands must share one tape (KL-002)
    ///
    /// If two operands carry *different* tapes this returns
    /// [`GlTrainError::InvalidOp`] and neither tape is touched. The first
    /// operand's tape used to win silently: the node landed on one tape
    /// referencing an input the other owned, and the second tape never learned
    /// the op happened, a split graph nothing downstream could detect. There is
    /// no sensible merge, since node IDs from two tapes collide, so it is
    /// rejected.
    ///
    /// # Untracked operands are expected, not an error (KL-003)
    ///
    /// The node lists *every* operand, including untracked ones, and an
    /// untracked operand is deliberately left unregistered, so looking it up in
    /// the tape yields `None`.
    ///
    /// **A `None` input ID means a frozen/untracked operand: no gradient is
    /// computed for it, and this is not an error.** That is the LoRA case, a
    /// frozen base weight consumed by a trainable activation.
    /// [`Tape::backward`] skips such inputs rather than failing on them.
    fn record_op<F>(
        output: &mut Tensor<B>,
        inputs: &[&Tensor<B>],
        op_name: &'static str,
        make_backward: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<BackwardFn>,
    {
        // Resolve which tape this op belongs to, rejecting a mixed set.
        let mut tape: Option<Arc<Mutex<Tape>>> = None;
        for input in inputs {
            let Some(candidate) = &input.tape else { continue };
            match &tape {
                None => tape = Some(candidate.clone()),
                Some(existing) => {
                    if !Arc::ptr_eq(existing, candidate) {
                        return Err(GlTrainError::InvalidOp(
                            "operands must share the same tape".into(),
                        ));
                    }
                }
            }
        }

        if !inputs.iter().any(|i| i.requires_grad) {
            return Ok(());
        }

        if let Some(ref t) = tape {
            let backward_fn = make_backward()?;
            let out_id = output.id;
            let out_shape = output.shape.clone();
            let mut guard = lock_tape(t);
            let node_id = guard.next_node_id();
            guard.register_tensor(TensorMeta {
                id: out_id,
                shape: out_shape,
                requires_grad: true,
            });
            guard.push(ComputationNode {
                id: node_id,
                op_name,
                // Every operand is listed, tracked or not. See KL-003 above.
                inputs: inputs.iter().map(|i| i.id).collect(),
                output: out_id,
                backward_fn,
            });
        }
        output.requires_grad = true;
        output.tape = tape;
        Ok(())
    }

    // ── Ops (dispatch to backend) ─────────────────────────────────────────

    /// Matrix multiplication: self @ other
    /// self: [M, K], other: [K, N] → result: [M, N]
    ///
    /// Records a "Matmul" node when either operand is tracked.
    pub fn matmul(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_matmul_shapes(&self.shape, &other.shape)?;
        let m = self.shape[0];
        let n = other.shape[other.shape.len() - 1];
        let storage = B::matmul(&self.storage, &other.storage, &self.shape, &other.shape)?;

        let mut output = Tensor::from_storage(storage, vec![m, n]);
        Self::record_op(&mut output, &[self, other], "Matmul", move || {
            // dA = dC @ B^T, dB = A^T @ dC.
            let a_tracked = self.requires_grad;
            let b_tracked = other.requires_grad;
            // dA reads B and dB reads A, so each operand's data is only worth
            // copying when the *other* one wants a gradient.
            let b_data = if a_tracked {
                B::to_vec(&other.storage)?
            } else {
                Vec::new()
            };
            let a_data = if b_tracked {
                B::to_vec(&self.storage)?
            } else {
                Vec::new()
            };
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();
            Ok(Arc::new(
                move |grad_output: &[f32], out_shape: &[usize]| {
                    let grad_a = if a_tracked {
                        let bt = transpose_2d(&b_data, &b_shape)?;
                        let bt_shape = vec![b_shape[1], b_shape[0]];
                        Some((
                            matmul_f32(grad_output, out_shape, &bt, &bt_shape)?,
                            a_shape.clone(),
                        ))
                    } else {
                        None
                    };
                    let grad_b = if b_tracked {
                        let at = transpose_2d(&a_data, &a_shape)?;
                        let at_shape = vec![a_shape[1], a_shape[0]];
                        Some((
                            matmul_f32(&at, &at_shape, grad_output, out_shape)?,
                            b_shape.clone(),
                        ))
                    } else {
                        None
                    };
                    Ok(vec![grad_a, grad_b])
                },
            ) as BackwardFn)
        })?;
        Ok(output)
    }

    /// Transpose a 2D tensor: [M, N] to [N, M]
    pub fn transpose(&self) -> Result<Tensor<B>> {
        if self.shape.len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "transpose() requires a 2D tensor, got shape {:?}",
                self.shape
            )));
        }
        let storage = B::transpose(&self.storage, &self.shape)?;
        let mut output = Tensor::from_storage(storage, vec![self.shape[1], self.shape[0]]);
        Self::record_op(&mut output, &[self], "Transpose", move || {
            // dA = dC^T, transposing [N,M] back to [M,N].
            let in_shape = self.shape.clone();
            let out_shape = vec![self.shape[1], self.shape[0]];
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                Ok(vec![Some((
                    transpose_2d(grad_output, &out_shape)?,
                    in_shape.clone(),
                ))])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    /// Element-wise addition: self + other (shapes must match exactly)
    ///
    /// Records an "Add" node when either operand is tracked.
    pub fn add(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::add(&self.storage, &other.storage, n)?;

        let mut output = Tensor::from_storage(storage, self.shape.clone());
        Self::record_op(&mut output, &[self, other], "Add", move || {
            // dA = dC, dB = dC.
            let a_tracked = self.requires_grad;
            let b_tracked = other.requires_grad;
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                Ok(vec![
                    a_tracked.then(|| (grad_output.to_vec(), a_shape.clone())),
                    b_tracked.then(|| (grad_output.to_vec(), b_shape.clone())),
                ])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::sub(&self.storage, &other.storage, n)?;

        let mut output = Tensor::from_storage(storage, self.shape.clone());
        Self::record_op(&mut output, &[self, other], "Sub", move || {
            // dA = dC, dB = -dC.
            let a_tracked = self.requires_grad;
            let b_tracked = other.requires_grad;
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                Ok(vec![
                    a_tracked.then(|| (grad_output.to_vec(), a_shape.clone())),
                    b_tracked.then(|| {
                        (
                            grad_output.iter().map(|g| -g).collect::<Vec<f32>>(),
                            b_shape.clone(),
                        )
                    }),
                ])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::mul(&self.storage, &other.storage, n)?;

        let mut output = Tensor::from_storage(storage, self.shape.clone());
        Self::record_op(&mut output, &[self, other], "Mul", move || {
            // dA = dC * B, dB = dC * A. Same asymmetry as matmul: each
            // operand's data is only needed if the other wants a gradient.
            let a_tracked = self.requires_grad;
            let b_tracked = other.requires_grad;
            let b_data = if a_tracked {
                B::to_vec(&other.storage)?
            } else {
                Vec::new()
            };
            let a_data = if b_tracked {
                B::to_vec(&self.storage)?
            } else {
                Vec::new()
            };
            let a_shape = self.shape.clone();
            let b_shape = other.shape.clone();
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                Ok(vec![
                    a_tracked.then(|| {
                        (
                            grad_output
                                .iter()
                                .zip(&b_data)
                                .map(|(g, b)| g * b)
                                .collect::<Vec<f32>>(),
                            a_shape.clone(),
                        )
                    }),
                    b_tracked.then(|| {
                        (
                            grad_output
                                .iter()
                                .zip(&a_data)
                                .map(|(g, a)| g * a)
                                .collect::<Vec<f32>>(),
                            b_shape.clone(),
                        )
                    }),
                ])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    /// Scale all elements by scalar: self * scalar
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor<B>> {
        let n = self.n_elems();
        let storage = B::mul_scalar(&self.storage, scalar, n)?;

        let mut output = Tensor::from_storage(storage, self.shape.clone());
        Self::record_op(&mut output, &[self], "MulScalar", move || {
            // dA = dC * scalar.
            let in_shape = self.shape.clone();
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                let grad_a: Vec<f32> = grad_output.iter().map(|g| g * scalar).collect();
                Ok(vec![Some((grad_a, in_shape.clone()))])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Result<Tensor<B>> {
        let n = self.n_elems();
        let storage = B::relu(&self.storage, n)?;

        let mut output = Tensor::from_storage(storage, self.shape.clone());
        Self::record_op(&mut output, &[self], "Relu", move || {
            // dA = dC where the forward input was positive, else 0.
            let a_data = B::to_vec(&self.storage)?;
            let in_shape = self.shape.clone();
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                let grad_a: Vec<f32> = grad_output
                    .iter()
                    .zip(&a_data)
                    .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
                    .collect();
                Ok(vec![Some((grad_a, in_shape.clone()))])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    // ── Reductions ────────────────────────────────────────────────────────
    //
    // These return a rank-1 tensor of shape [1], not a bare f32, so a loss can
    // sit on the tape and be differentiated (KL-004). Use `sum_scalar` /
    // `mean_scalar` when you just want the number and no tape node.

    /// Sum every element into a tensor of shape `[1]`.
    pub fn sum(&self) -> Result<Tensor<B>> {
        let total = B::sum(&self.storage)?;
        let storage = B::from_vec(vec![total])?;

        let mut output = Tensor::from_storage(storage, vec![1]);
        Self::record_op(&mut output, &[self], "Sum", move || {
            // Every element contributed once, so each gets the whole upstream
            // gradient.
            let in_shape = self.shape.clone();
            let n = self.n_elems();
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                let g = grad_output.first().copied().unwrap_or(1.0);
                Ok(vec![Some((vec![g; n], in_shape.clone()))])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    /// Mean of every element, as a tensor of shape `[1]`.
    pub fn mean(&self) -> Result<Tensor<B>> {
        let n = self.n_elems();
        let avg = B::mean(&self.storage, n)?;
        let storage = B::from_vec(vec![avg])?;

        let mut output = Tensor::from_storage(storage, vec![1]);
        Self::record_op(&mut output, &[self], "Mean", move || {
            // Same as sum, divided by the element count.
            let in_shape = self.shape.clone();
            Ok(Arc::new(move |grad_output: &[f32], _: &[usize]| {
                let g = grad_output.first().copied().unwrap_or(1.0) / n as f32;
                Ok(vec![Some((vec![g; n], in_shape.clone()))])
            }) as BackwardFn)
        })?;
        Ok(output)
    }

    /// Sum every element and return the raw number. Records nothing.
    pub fn sum_scalar(&self) -> Result<f32> {
        B::sum(&self.storage)
    }

    /// Mean of every element as a raw number. Records nothing.
    pub fn mean_scalar(&self) -> Result<f32> {
        B::mean(&self.storage, self.n_elems())
    }
}

// ── Shape validation helpers ─────────────────────────────────────────────────

fn shape_to_n_elems(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Err(GlTrainError::InvalidOp("shape cannot be empty".into()));
    }
    Ok(shape.iter().product())
}

fn check_same_shape(a: &[usize], b: &[usize]) -> Result<()> {
    if a != b {
        return Err(GlTrainError::ShapeMismatch {
            expected: a.to_vec(),
            got: b.to_vec(),
        });
    }
    Ok(())
}

fn check_matmul_shapes(a: &[usize], b: &[usize]) -> Result<()> {
    // Support 2D only in Wave 1: [M, K] @ [K, N]
    if a.len() != 2 || b.len() != 2 {
        return Err(GlTrainError::InvalidOp(format!(
            "matmul requires 2D tensors in Wave 1, got shapes {a:?} and {b:?}"
        )));
    }
    if a[1] != b[0] {
        return Err(GlTrainError::ShapeMismatch {
            expected: vec![a[0], a[1], b[1]],
            got: vec![a[0], b[0], b[1]],
        });
    }
    Ok(())
}

// ── Debug ────────────────────────────────────────────────────────────────────

impl<B: Backend> std::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?})", self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// Tolerance for f32 elementwise ops (rounding only, no accumulation).
    const TOL_ELEM: f32 = 1e-6;

    /// Tolerance for matmul (accumulation over K dimension can drift).
    const TOL_MATMUL: f32 = 1e-4;

    /// Relative tolerance for matmul at realistic K, as a fraction of the
    /// largest output magnitude. At K=64 an absolute bound no longer means
    /// anything — the error grows with the accumulation, so the bound must too.
    const TOL_MATMUL_REL: f32 = 1e-5;

    // ── Allocation ───────────────────────────────────────────────────────

    #[test]
    fn zeros_produces_correct_shape_and_values() {
        let t = Tensor::<GlProc>::zeros(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.n_elems(), 6);
        let v = t.to_vec().unwrap();
        assert!(
            v.iter().all(|&x| x == 0.0),
            "expected all zeros, got {v:?}"
        );
    }

    #[test]
    fn from_vec_roundtrip_preserves_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::<GlProc>::from_vec(data.clone(), &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        let out = t.to_vec().unwrap();
        for (a, b) in out.iter().zip(&data) {
            assert!((a - b).abs() < TOL_ELEM, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn from_vec_rejects_size_mismatch() {
        let result = Tensor::<GlProc>::from_vec(vec![1.0, 2.0], &[2, 2]);
        assert!(result.is_err(), "expected Err for size mismatch");
    }

    // ── Matmul ───────────────────────────────────────────────────────────

    #[test]
    fn matmul_2x2_by_2x2_produces_correct_result() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<GlProc>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let v = c.to_vec().unwrap();
        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < TOL_MATMUL,
                "matmul mismatch: got {got}, expected {exp}"
            );
        }
    }

    /// A LoRA-shaped matmul: [16, 64] @ [64, 32] is the shape of a rank-ish
    /// projection on a batch of 16, and unlike the 2×2 case above it actually
    /// reaches glproc's AVX2 path — M, K and N are all whole multiples of the
    /// 8-float lane, so the SIMD body runs with no scalar tail to hide behind.
    ///
    /// Checked against an f64 accumulation of the same product computed here,
    /// so the reference does not share any code with the kernel under test.
    #[test]
    fn matmul_16x64_by_64x32_matches_f64_reference() {
        let (m, k, n) = (16usize, 64usize, 32usize);
        // Deterministic, mixed-sign, O(1) magnitude — no RNG, so a failure is
        // always reproducible.
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0)
            .collect();

        let got = Tensor::<GlProc>::from_vec(a.clone(), &[m, k])
            .unwrap()
            .matmul(&Tensor::<GlProc>::from_vec(b.clone(), &[k, n]).unwrap())
            .unwrap();
        assert_eq!(got.shape(), &[m, n]);
        let got = got.to_vec().unwrap();
        assert_eq!(got.len(), m * n);

        let mut expected = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
                }
                expected[i * n + j] = acc;
            }
        }

        // K=64 accumulation makes an absolute bound meaningless, so the
        // tolerance scales with the largest output magnitude.
        let scale = expected.iter().fold(1.0f64, |acc, v| acc.max(v.abs()));
        for (idx, (got, exp)) in got.iter().zip(&expected).enumerate() {
            let err = (f64::from(*got) - exp).abs();
            assert!(
                err <= f64::from(TOL_MATMUL_REL) * scale,
                "matmul mismatch at {idx}: got {got}, expected {exp}, err {err:.3e}"
            );
        }
    }

    #[test]
    fn matmul_shape_mismatch_returns_error() {
        // [2, 3] @ [2, 2] is invalid (K mismatch: 3 ≠ 2)
        let a = Tensor::<GlProc>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<GlProc>::zeros(&[2, 2]).unwrap();
        assert!(a.matmul(&b).is_err(), "expected Err for K mismatch");
    }

    // ── Transpose ────────────────────────────────────────────────────────

    #[test]
    fn transpose_2x3_produces_3x2() {
        // [[1, 2, 3], [4, 5, 6]] → [[1, 4], [2, 5], [3, 6]]
        let t =
            Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let tr = t.transpose().unwrap();
        assert_eq!(tr.shape(), &[3, 2]);
        let v = tr.to_vec().unwrap();
        let expected = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < TOL_ELEM,
                "transpose mismatch: got {got}, expected {exp}"
            );
        }
    }

    // ── Elementwise ──────────────────────────────────────────────────────

    #[test]
    fn add_two_tensors_element_wise() {
        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<GlProc>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = a.add(&b).unwrap();
        let v = c.to_vec().unwrap();
        let expected = [5.0f32, 7.0, 9.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!((got - exp).abs() < TOL_ELEM, "{got} vs {exp}");
        }
    }

    #[test]
    fn relu_zeroes_negative_values() {
        let t = Tensor::<GlProc>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let r = t.relu().unwrap();
        let v = r.to_vec().unwrap();
        let expected = [0.0f32, 0.0, 0.0, 1.0, 2.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!((got - exp).abs() < TOL_ELEM, "{got} vs {exp}");
        }
    }

    #[test]
    fn mul_scalar_scales_all_elements() {
        let t = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let scaled = t.mul_scalar(2.5).unwrap();
        let v = scaled.to_vec().unwrap();
        let expected = [2.5f32, 5.0, 7.5, 10.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!((got - exp).abs() < TOL_ELEM, "{got} vs {exp}");
        }
    }

    // ── Reduction ────────────────────────────────────────────────────────

    #[test]
    fn sum_of_ones_tensor_equals_n_elems() {
        let t = Tensor::<GlProc>::ones(&[3, 4]).unwrap();
        // Wave 3 turned sum() into a tape-recording op returning a [1] tensor;
        // sum_scalar() is the raw-number path this test always wanted.
        let s = t.sum_scalar().unwrap();
        assert!((s - 12.0).abs() < TOL_ELEM, "sum={s}, expected 12.0");
    }

    #[test]
    fn sum_returns_rank_one_tensor_holding_the_total() {
        let t = Tensor::<GlProc>::ones(&[3, 4]).unwrap();
        let s = t.sum().unwrap();
        assert_eq!(s.shape(), &[1], "reductions produce shape [1], see KL-004");
        assert!((s.item().unwrap() - 12.0).abs() < TOL_ELEM);
    }

    #[test]
    fn mean_returns_rank_one_tensor_holding_the_average() {
        let t = Tensor::<GlProc>::ones(&[3, 4]).unwrap();
        let m = t.mean().unwrap();
        assert_eq!(m.shape(), &[1]);
        assert!((m.item().unwrap() - 1.0).abs() < TOL_ELEM);
    }

    // ── Wave 2: Autograd tape integration ────────────────────────────────

    #[test]
    fn tensor_without_grad_has_no_tape() {
        let t = Tensor::<GlProc>::zeros(&[2, 2]).unwrap();
        assert!(!t.requires_grad());
        assert!(t.tape().is_none());
    }

    #[test]
    fn with_grad_marks_tensor_as_tracked() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let t = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape.clone());

        assert!(t.requires_grad());
        assert!(t.tape().is_some());
    }

    #[test]
    fn matmul_on_tracked_tensors_records_one_node() {
        let tape = Arc::new(Mutex::new(Tape::new()));

        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let b = Tensor::<GlProc>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());

        let _c = a.matmul(&b).unwrap();

        let tape_guard = lock_tape(&tape);
        assert_eq!(tape_guard.len(), 1, "expected 1 node recorded for matmul");
        assert_eq!(tape_guard.op_names(), vec!["Matmul"]);
    }

    #[test]
    fn chained_ops_record_nodes_in_forward_order() {
        let tape = Arc::new(Mutex::new(Tape::new()));

        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let b = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let c = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape.clone());

        // y = (a @ b) + c  → should record [Matmul, Add]
        let ab = a.matmul(&b).unwrap();
        let _y = ab.add(&c).unwrap();

        let tape_guard = lock_tape(&tape);
        assert_eq!(tape_guard.len(), 2);
        assert_eq!(tape_guard.op_names(), vec!["Matmul", "Add"]);
    }

    #[test]
    fn no_grad_tensors_do_not_record_to_tape() {
        let tape = Arc::new(Mutex::new(Tape::new()));

        // Neither tensor has with_grad — tape should stay empty
        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<GlProc>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let _c = a.matmul(&b).unwrap();

        let tape_guard = lock_tape(&tape);
        assert_eq!(tape_guard.len(), 0, "no-grad ops must not record to tape");
    }

    #[test]
    fn detach_produces_no_grad_tensor_with_same_data() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let t = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());

        let detached = t.detach();
        assert!(!detached.requires_grad());
        assert!(detached.tape().is_none());

        // Data must be identical
        let orig = t.to_vec().unwrap();
        let det = detached.to_vec().unwrap();
        for (a, b) in orig.iter().zip(&det) {
            assert!((a - b).abs() < TOL_ELEM, "detach must preserve data");
        }
    }

    /// KL-003, made executable: mixing a tracked operand with a frozen one is
    /// the LoRA case and must work. The node records BOTH input IDs, but only
    /// the tracked operand resolves in the tape — a `None` lookup on the other
    /// means "no gradient wanted here", not a failure.
    #[test]
    fn untracked_operand_records_node_with_partial_inputs() {
        let tape = Arc::new(Mutex::new(Tape::new()));

        let tracked = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        // A frozen base weight: never attached to the tape.
        let frozen = Tensor::<GlProc>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let out = tracked
            .matmul(&frozen)
            .expect("tracked @ untracked must succeed, not error");

        // The op is still recorded, and the result stays tracked.
        assert!(out.requires_grad(), "output must stay tracked");

        let guard = lock_tape(&tape);
        assert_eq!(guard.len(), 1, "the op must still be recorded");

        let node = &guard.nodes()[0];
        assert_eq!(
            node.inputs,
            vec![tracked.id(), frozen.id()],
            "both operands must be listed, tracked or not"
        );

        // The intentional asymmetry: tracked resolves, frozen does not.
        assert!(
            guard.get_tensor_meta(tracked.id()).is_some(),
            "tracked operand must be registered"
        );
        assert!(
            guard.get_tensor_meta(frozen.id()).is_none(),
            "untracked operand is deliberately unregistered — a None lookup \
             here means 'frozen, no gradient', which is not an error"
        );
    }

    /// KL-002: two operands on different tapes used to silently split the
    /// graph. It is now a hard error.
    #[test]
    fn ops_across_two_different_tapes_are_rejected() {
        let tape1 = Arc::new(Mutex::new(Tape::new()));
        let tape2 = Arc::new(Mutex::new(Tape::new()));

        let a = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape1.clone());
        let b = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape2.clone());

        let err = a.matmul(&b).expect_err("mixed tapes must be rejected");
        assert!(
            matches!(err, GlTrainError::InvalidOp(ref m) if m.contains("same tape")),
            "expected InvalidOp about tape sharing, got: {err}"
        );

        // add() must reject it too, not just matmul.
        assert!(a.add(&b).is_err(), "add must reject mixed tapes as well");

        // Neither tape may have been mutated by the rejected ops.
        assert_eq!(lock_tape(&tape1).len(), 0, "tape1 must be untouched");
        assert_eq!(lock_tape(&tape2).len(), 0, "tape2 must be untouched");
    }

    /// Two tensors on the *same* tape must still work — the KL-002 check
    /// compares Arc identity, not tape emptiness or content.
    #[test]
    fn ops_on_the_same_tape_are_accepted() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let a = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        // A second handle to the same tape — a clone of the Arc, not a new tape.
        let b = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(Arc::clone(&tape));

        assert!(
            a.matmul(&b).is_ok(),
            "operands sharing one tape must be accepted"
        );
        assert_eq!(lock_tape(&tape).len(), 1);
    }

    #[test]
    fn output_of_tracked_matmul_inherits_tape() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let a = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let b = Tensor::<GlProc>::zeros(&[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let c = a.matmul(&b).unwrap();

        // Output tensor must carry the tape forward (for chaining)
        assert!(c.requires_grad(), "output of tracked matmul must require grad");
        assert!(c.tape().is_some(), "output of tracked matmul must carry tape");
    }
}

/// Finite-difference gradient checks for every op with a backward function.
///
/// Runs on `SisdBackend`, the plain scalar backend, so a failure points at the
/// backward math rather than at a SIMD kernel.
#[cfg(test)]
mod grad_check {
    use super::*;
    use crate::backend::SisdBackend;

    /// Finite-difference step. Large enough that the f32 subtraction in
    /// `(f(x+h) - f(x-h))` keeps significant digits, small enough that the
    /// central difference stays close to the true derivative.
    const GRAD_CHECK_EPS: f32 = 1e-2;

    /// Tolerance on the *relative* gradient error. Plan section 1.7 asks for
    /// rtol=1e-3, which is what this is.
    ///
    /// An absolute bound cannot work across these tests. The finite difference
    /// loses about `|L| * f32_eps / h` to cancellation, and `|L|` grows with
    /// the gradient magnitude, so a fixed atol that suits a gradient of 1.0
    /// fails one of 21.0 for reasons that have nothing to do with correctness.
    /// Normalising by the magnitude tracks the error the method actually has.
    const GRAD_CHECK_TOL: f32 = 1e-3;

    /// Tolerance for hand-computed exact comparisons, where the only error is
    /// f32 rounding on small integers.
    const TOL_EXACT: f32 = 1e-5;

    /// Compare the analytic gradient from `backward()` against a central
    /// finite difference, and return the largest *relative* disagreement,
    /// `|a - n| / max(1, |a|, |n|)`.
    ///
    /// `f` receives the tape so anything it builds attaches to the same one.
    /// Building an operand on a different tape would trip the KL-002 guard and
    /// fail the forward pass instead of checking a gradient.
    ///
    /// `backward()` seeds ones, so the quantity being differentiated is
    /// `sum(f(x))`. The numeric side calls `sum_scalar()` to match.
    fn finite_diff_check<F>(input: &[f32], shape: &[usize], f: F) -> f32
    where
        F: Fn(&Tensor<SisdBackend>, &Arc<Mutex<Tape>>) -> Result<Tensor<SisdBackend>>,
    {
        // Analytic side.
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<SisdBackend>::from_vec(input.to_vec(), shape)
            .unwrap()
            .with_grad(tape.clone());
        let x_id = x.id();
        f(&x, &tape).expect("forward pass must succeed");

        let analytic = {
            let mut guard = lock_tape(&tape);
            guard.backward().expect("backward must succeed");
            // Never let a missing gradient pass as "zero error".
            guard
                .grad(x_id)
                .unwrap_or_else(|| panic!("backward produced no gradient for the input"))
                .0
                .clone()
        };
        assert_eq!(
            analytic.len(),
            input.len(),
            "gradient length must match the input"
        );

        // Numeric side.
        let eval = |data: Vec<f32>| -> f32 {
            let tp = Arc::new(Mutex::new(Tape::new()));
            let xt = Tensor::<SisdBackend>::from_vec(data, shape)
                .unwrap()
                .with_grad(tp.clone());
            f(&xt, &tp).unwrap().sum_scalar().unwrap()
        };

        let mut max_err = 0.0f32;
        for i in 0..input.len() {
            let mut plus = input.to_vec();
            let mut minus = input.to_vec();
            plus[i] += GRAD_CHECK_EPS;
            minus[i] -= GRAD_CHECK_EPS;
            let numeric = (eval(plus) - eval(minus)) / (2.0 * GRAD_CHECK_EPS);
            let scale = 1.0f32.max(analytic[i].abs()).max(numeric.abs());
            let err = (analytic[i] - numeric).abs() / scale;
            if err > max_err {
                max_err = err;
            }
        }
        max_err
    }

    /// Exact check, no finite differences involved.
    ///
    /// With a ones seed, `dA[i,j]` is the sum of row `j` of B and `dB[j,k]` is
    /// the sum of column `j` of A. Both work out to small integers here, so
    /// this pins the backward math far more tightly than a finite difference
    /// can, and it settles whether any FD disagreement is a bug or just noise.
    #[test]
    fn matmul_backward_matches_hand_computed_gradient() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        // A = [[1,2,3],[4,5,6]]
        let a = Tensor::<SisdBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap()
            .with_grad(tape.clone());
        // B = [[0.5,1,1.5,2],[2.5,3,3.5,4],[4.5,5,5.5,6]]
        let b = Tensor::<SisdBackend>::from_vec(
            vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            &[3, 4],
        )
        .unwrap()
        .with_grad(tape.clone());
        let (a_id, b_id) = (a.id(), b.id());
        let _c = a.matmul(&b).unwrap();

        let mut guard = lock_tape(&tape);
        guard.backward().unwrap();

        // Row sums of B: 5, 13, 21. Every row of dA is that vector.
        let (grad_a, shape_a) = guard.grad(a_id).expect("A must have a gradient");
        assert_eq!(shape_a, &vec![2, 3]);
        let want_a = [5.0f32, 13.0, 21.0, 5.0, 13.0, 21.0];
        for (i, (got, want)) in grad_a.iter().zip(&want_a).enumerate() {
            assert!(
                (got - want).abs() < TOL_EXACT,
                "dA[{i}] = {got}, expected {want}"
            );
        }

        // Column sums of A: 5, 7, 9. Every column of dB is that vector.
        let (grad_b, shape_b) = guard.grad(b_id).expect("B must have a gradient");
        assert_eq!(shape_b, &vec![3, 4]);
        let want_b = [5.0f32, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0];
        for (i, (got, want)) in grad_b.iter().zip(&want_b).enumerate() {
            assert!(
                (got - want).abs() < TOL_EXACT,
                "dB[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn grad_check_matmul() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = finite_diff_check(&a, &[2, 2], |x, tp| {
            let b = Tensor::<SisdBackend>::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2])
                .unwrap()
                .with_grad(tp.clone());
            x.matmul(&b)
        });
        assert!(err < GRAD_CHECK_TOL, "matmul grad error {err} exceeds tolerance");
    }

    /// Non-square catches a transposed-index bug that [2,2] hides.
    #[test]
    fn grad_check_matmul_non_square() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = finite_diff_check(&a, &[2, 3], |x, tp| {
            let b = Tensor::<SisdBackend>::from_vec(
                vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                &[3, 4],
            )
            .unwrap()
            .with_grad(tp.clone());
            x.matmul(&b)
        });
        assert!(
            err < GRAD_CHECK_TOL,
            "non-square matmul grad error {err} exceeds tolerance"
        );
    }

    #[test]
    fn grad_check_add() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = finite_diff_check(&a, &[4], |x, tp| {
            let b = Tensor::<SisdBackend>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])
                .unwrap()
                .with_grad(tp.clone());
            x.add(&b)
        });
        assert!(err < GRAD_CHECK_TOL, "add grad error {err} exceeds tolerance");
    }

    #[test]
    fn grad_check_sub() {
        let a = vec![3.0f32, 1.0, 4.0, 1.0];
        let err = finite_diff_check(&a, &[4], |x, tp| {
            let b = Tensor::<SisdBackend>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4])
                .unwrap()
                .with_grad(tp.clone());
            x.sub(&b)
        });
        assert!(err < GRAD_CHECK_TOL, "sub grad error {err} exceeds tolerance");
    }

    #[test]
    fn grad_check_mul() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = finite_diff_check(&a, &[4], |x, tp| {
            let b = Tensor::<SisdBackend>::from_vec(vec![2.0, -1.0, 0.5, 3.0], &[4])
                .unwrap()
                .with_grad(tp.clone());
            x.mul(&b)
        });
        assert!(err < GRAD_CHECK_TOL, "mul grad error {err} exceeds tolerance");
    }

    #[test]
    fn grad_check_mul_scalar() {
        let a = vec![1.0f32, 2.0, 3.0];
        let err = finite_diff_check(&a, &[3], |x, _| x.mul_scalar(3.0));
        assert!(
            err < GRAD_CHECK_TOL,
            "mul_scalar grad error {err} exceeds tolerance"
        );
    }

    #[test]
    fn grad_check_relu() {
        // Mixed signs exercise both branches. None sit within EPS of zero, so
        // the finite difference never straddles the kink.
        let a = vec![-1.0f32, 0.5, -0.5, 2.0];
        let err = finite_diff_check(&a, &[4], |x, _| x.relu());
        assert!(err < GRAD_CHECK_TOL, "relu grad error {err} exceeds tolerance");
    }

    #[test]
    fn grad_check_transpose() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = finite_diff_check(&a, &[2, 3], |x, _| x.transpose());
        assert!(
            err < GRAD_CHECK_TOL,
            "transpose grad error {err} exceeds tolerance"
        );
    }

    #[test]
    fn grad_check_sum() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = finite_diff_check(&a, &[4], |x, _| x.sum());
        assert!(err < GRAD_CHECK_TOL, "sum grad error {err} exceeds tolerance");
    }

    #[test]
    fn grad_check_mean() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = finite_diff_check(&a, &[4], |x, _| x.mean());
        assert!(err < GRAD_CHECK_TOL, "mean grad error {err} exceeds tolerance");
    }

    /// Two ops chained, so the gradient has to flow through a node it did not
    /// start at.
    #[test]
    fn grad_check_chained_matmul_then_relu() {
        let a = vec![1.0f32, -2.0, 3.0, 0.5];
        let err = finite_diff_check(&a, &[2, 2], |x, tp| {
            let w = Tensor::<SisdBackend>::from_vec(vec![1.0, -0.5, 0.25, 2.0], &[2, 2])
                .unwrap()
                .with_grad(tp.clone());
            x.matmul(&w)?.relu()
        });
        assert!(
            err < GRAD_CHECK_TOL,
            "chained matmul->relu grad error {err} exceeds tolerance"
        );
    }

    #[test]
    fn backward_accumulates_grad_for_shared_tensor() {
        // y = x @ x. x is both operands, so its gradient is the sum of two
        // contributions.
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<SisdBackend>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let x_id = x.id();
        let _y = x.matmul(&x).unwrap();

        let mut guard = lock_tape(&tape);
        guard.backward().unwrap();
        let (grad, shape) = guard.grad(x_id).expect("x must have a gradient");
        assert_eq!(shape, &vec![2, 2]);
        // For x = I with a ones seed, dL/dx = dC @ x^T + x^T @ dC = 2 everywhere.
        for (i, g) in grad.iter().enumerate() {
            assert!(
                (g - 2.0).abs() < GRAD_CHECK_TOL,
                "element {i}: expected both paths to contribute 1.0 each, got {g}"
            );
        }
    }

    #[test]
    fn backward_frozen_operand_has_no_gradient() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<SisdBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        // Frozen base weight: never attached to a tape.
        let w = Tensor::<SisdBackend>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let x_id = x.id();
        let w_id = w.id();
        let _y = x.matmul(&w).unwrap();

        let mut guard = lock_tape(&tape);
        guard.backward().unwrap();
        assert!(guard.grad(x_id).is_some(), "tracked tensor must have a gradient");
        assert!(
            guard.grad(w_id).is_none(),
            "frozen tensor must not have a gradient (KL-003)"
        );
    }

    #[test]
    fn backward_on_empty_tape_is_a_noop() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let mut guard = lock_tape(&tape);
        guard.backward().expect("empty tape must not error");
        assert!(guard.grad_store().is_empty());
    }

    /// A second `backward()` used to triple the gradient rather than double
    /// it: the seed accumulates onto itself and then propagates. Measured
    /// `[1,1,1,1]` then `[3,3,3,3]` before the guard landed.
    #[test]
    fn backward_twice_without_zero_grad_is_rejected() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<SisdBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let id = x.id();
        let w = Tensor::<SisdBackend>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let _y = x.matmul(&w).unwrap();

        let mut guard = lock_tape(&tape);
        guard.backward().expect("first backward must succeed");
        let first = guard.grad(id).expect("x must have a gradient").0.clone();

        let err = guard
            .backward()
            .expect_err("a second backward on a dirty store must be rejected");
        assert!(
            matches!(err, GlTrainError::InvalidOp(ref m) if m.contains("zero_grad")),
            "expected InvalidOp naming zero_grad, got: {err}"
        );

        // The rejected call must not have altered anything.
        let after = guard.grad(id).expect("gradient must survive").0.clone();
        assert_eq!(first, after, "a rejected backward must not touch gradients");

        // After zero_grad the same pass is allowed again and reproduces it.
        guard.zero_grad();
        guard.backward().expect("backward must work after zero_grad");
        let again = guard.grad(id).expect("x must have a gradient").0.clone();
        assert_eq!(first, again, "backward must be reproducible after zero_grad");
    }

    #[test]
    fn zero_grad_clears_gradients_but_keeps_the_graph() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<SisdBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let _y = x.matmul(&x).unwrap();

        let mut guard = lock_tape(&tape);
        guard.backward().unwrap();
        assert!(!guard.grad_store().is_empty());
        let nodes_before = guard.len();
        guard.zero_grad();
        assert!(guard.grad_store().is_empty(), "zero_grad must drop gradients");
        assert_eq!(guard.len(), nodes_before, "zero_grad must keep the graph");
    }
}
