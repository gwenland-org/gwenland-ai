//! Stummañ Kevrin — Tensor abstraction.
//!
//! `Tensor<B>` is the core data structure for Stummañ.
//! It wraps backend storage with shape metadata.
//! Wave 1: pure data.
//! Wave 2: adds an identity, a `requires_grad` flag, and an optional shared
//! tape, so ops can *record* the forward pass. Still no gradients — no `grad`
//! field, and nothing replays the tape. That is Wave 3.

use crate::autograd::node::{ComputationNode, TensorId};
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

    /// Record one binary op on the shared tape and attach that tape to `output`.
    ///
    /// Every tracked op funnels through here so the bookkeeping — resolve the
    /// tape, allocate a node id, register the output's metadata, push the node
    /// — cannot drift between call sites. A no-op when neither operand is
    /// tracked.
    ///
    /// # Both operands must share one tape (KL-002)
    ///
    /// If `lhs` and `rhs` carry *different* tapes this returns
    /// [`GlTrainError::InvalidOp`]. Previously the first operand's tape won
    /// silently: the node landed on one tape referencing an input the other
    /// tape owned, and the second tape never learned the op happened at all —
    /// a split graph that no later stage could detect. There is no sensible
    /// merge, so it is rejected.
    ///
    /// The check runs after the forward compute has already happened. That is
    /// deliberate: mismatched tapes are a programming error that never occurs
    /// on a correct path, so the wasted arithmetic costs nothing in practice
    /// and keeps the tape logic in exactly one place.
    ///
    /// # Untracked operands are expected, not an error (KL-003)
    ///
    /// When only one operand is tracked, the node still lists **both** input
    /// IDs, and the untracked one is deliberately left unregistered — looking
    /// it up in the tape yields `None`.
    ///
    /// **A `None` input ID means a frozen/untracked operand: no gradient is
    /// computed for it, and this is not an error.** This is the LoRA case —
    /// a frozen base weight multiplied by a trainable activation. Wave 3's
    /// `backward()` must skip unresolvable input IDs rather than fail on them.
    fn record_op(
        output: &mut Tensor<B>,
        lhs: &Tensor<B>,
        rhs: &Tensor<B>,
        op_name: &'static str,
    ) -> Result<()> {
        // Resolve which tape this op belongs to, rejecting a mixed pair.
        let tape = match (&lhs.tape, &rhs.tape) {
            (Some(t1), Some(t2)) => {
                if !Arc::ptr_eq(t1, t2) {
                    return Err(GlTrainError::InvalidOp(
                        "operands must share the same tape".into(),
                    ));
                }
                Some(t1.clone())
            }
            (Some(t), None) | (None, Some(t)) => Some(t.clone()),
            (None, None) => None,
        };

        if !(lhs.requires_grad || rhs.requires_grad) {
            return Ok(());
        }

        if let Some(ref t) = tape {
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
                // Both operands are listed even when one is untracked — see
                // the KL-003 note above.
                inputs: vec![lhs.id, rhs.id],
                output: out_id,
                // Wave 2 placeholder — stored, never called. See `BackwardFn`.
                backward_fn: Arc::new(|| ()),
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
        Self::record_op(&mut output, self, other, "Matmul")?;
        Ok(output)
    }

    /// Transpose a 2D tensor: [M, N] → [N, M]
    pub fn transpose(&self) -> Result<Tensor<B>> {
        if self.shape.len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "transpose() requires a 2D tensor, got shape {:?}",
                self.shape
            )));
        }
        let storage = B::transpose(&self.storage, &self.shape)?;
        Ok(Tensor::from_storage(
            storage,
            vec![self.shape[1], self.shape[0]],
        ))
    }

    /// Element-wise addition: self + other (shapes must match exactly)
    ///
    /// Records an "Add" node when either operand is tracked.
    pub fn add(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::add(&self.storage, &other.storage, n)?;

        let mut output = Tensor::from_storage(storage, self.shape.clone());
        Self::record_op(&mut output, self, other, "Add")?;
        Ok(output)
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::sub(&self.storage, &other.storage, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::mul(&self.storage, &other.storage, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// Scale all elements by scalar: self * scalar
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor<B>> {
        let n = self.n_elems();
        let storage = B::mul_scalar(&self.storage, scalar, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Result<Tensor<B>> {
        let n = self.n_elems();
        let storage = B::relu(&self.storage, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// Sum all elements to a scalar.
    pub fn sum(&self) -> Result<f32> {
        B::sum(&self.storage)
    }

    /// Mean of all elements.
    pub fn mean(&self) -> Result<f32> {
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
        let s = t.sum().unwrap();
        assert!((s - 12.0).abs() < TOL_ELEM, "sum={s}, expected 12.0");
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
