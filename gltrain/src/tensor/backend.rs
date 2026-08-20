//! Stummañ Karg — Backend trait.
//!
//! Every compute backend (GlProc, GlCuda, GlJax) implements this trait.
//! `Tensor<B>` is generic over B: Backend; ops dispatch through these methods.

use crate::error::Result;

/// Compute backend trait for Stummañ.
///
/// Implementors: GlProc (CPU/AVX2), GlCuda (GPU/PTX), GlJax (TPU/PJRT).
/// All storage types must be Clone + Send + Sync for multi-threaded training.
///
/// Every method takes the element count or shape explicitly rather than
/// deriving it from the storage: a future device backend's storage need not
/// expose a length, and the caller ([`crate::tensor::Tensor`]) always knows it.
/// Implementors must validate that the storage actually matches — a backend
/// that trusts the count silently produces a wrong-sized result.
///
/// # KNOWN LIMITATION (KL-001) — this trait is not dyn-compatible
///
/// `Backend` cannot be used as a trait object. `Box<dyn Backend>` and
/// `&dyn Backend` fail with E0038. Two blockers, each verified sufficient on
/// its own:
///
/// 1. The `Clone` supertrait implies `Self: Sized`, which excludes the trait
///    from dyn-compatibility outright. This is the one rustc reports first.
/// 2. Every method is an *associated function* with no `self` receiver, so
///    there is no vtable to dispatch through.
///
/// Binding the associated type (`dyn Backend<Storage = Vec<f32>>`) does **not**
/// rescue it — both blockers survive that. And binding it would defeat the
/// purpose anyway: the point of `Storage` being associated is that a GPU
/// backend brings a device buffer, not a `Vec<f32>`, so a single erased type
/// cannot span the backends.
///
/// This is deliberate — dispatch is static, resolved at compile time, which is
/// what the plan calls for (STUMMAN_PLAN.md §3.6, "Static Backend Selection").
/// It costs nothing at runtime and lets each backend pick its own `Storage`.
///
/// The conflict to be aware of: §3.6's *GATE Integration* sketch returns
/// `Box<dyn Backend>` from `auto_backend()`. That sketch will not compile
/// against this trait as written. The `match backend { "cpu" =>
/// train::<GlProc>(), ... }` form in the same section does work and needs no
/// trait objects.
///
/// Deferred to **M4**, when runtime backend selection actually ships
/// (`gwen train --backend cuda`). Do not restructure this trait in Wave 2–3 to
/// pre-empt it: autograd needs only static dispatch, and widening the contract
/// now would be speculative. See `gltrain/KNOWN_ISSUES.md` for the resolution
/// options.
pub trait Backend: Clone + Send + Sync + 'static {
    /// Device-local tensor storage.
    /// GlProc: `Vec<f32>`
    /// GlCuda (future): `CudaBuffer<f32>`
    type Storage: Clone + Send + Sync;

    // ── Allocation ───────────────────────────────────────────────────────

    /// Allocate storage filled with zeros of the given total element count.
    fn zeros(n_elems: usize) -> Result<Self::Storage>;

    /// Allocate storage filled with ones.
    fn ones(n_elems: usize) -> Result<Self::Storage>;

    /// Create storage from a host `Vec<f32>`.
    fn from_vec(data: Vec<f32>) -> Result<Self::Storage>;

    /// Copy storage to a host `Vec<f32>`.
    fn to_vec(storage: &Self::Storage) -> Result<Vec<f32>>;

    // ── Linear algebra ───────────────────────────────────────────────────

    /// Matrix multiplication: C = A @ B
    /// a_shape: [M, K], b_shape: [K, N] → output shape: [M, N]
    fn matmul(
        a: &Self::Storage,
        b: &Self::Storage,
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Result<Self::Storage>;

    /// Transpose a 2D matrix.
    /// shape: [M, N] → output shape: [N, M]
    fn transpose(a: &Self::Storage, shape: &[usize]) -> Result<Self::Storage>;

    // ── Elementwise ──────────────────────────────────────────────────────

    /// Element-wise addition: C = A + B (shapes must match)
    fn add(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    /// Element-wise subtraction: C = A - B
    fn sub(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    /// Element-wise multiplication: C = A * B
    fn mul(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    /// Scale all elements by a scalar: B = A * scalar
    fn mul_scalar(a: &Self::Storage, scalar: f32, n_elems: usize) -> Result<Self::Storage>;

    /// Element-wise division: C = A / B
    ///
    /// Added for M2's optimizer. Implementors must reject a zero divisor rather
    /// than emit an infinity: AdamW's denominator is `sqrt(v) + eps` and can
    /// only be zero if `eps` was passed as zero, which is a configuration bug
    /// worth naming instead of propagating as a NaN weight three steps later.
    fn div(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    /// Element-wise square root.
    ///
    /// Implementors must reject a negative input. AdamW only ever calls this on
    /// a second moment, which is a sum of squares and cannot be negative, so a
    /// negative here means the state is already corrupt.
    fn sqrt(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    /// Add a scalar to every element: B = A + scalar
    fn add_scalar(a: &Self::Storage, scalar: f32, n_elems: usize) -> Result<Self::Storage>;

    /// Element-wise negation: B = -A
    fn neg(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    /// Element-wise sign: -1.0, 0.0 or +1.0.
    ///
    /// Added for [`crate::optim::OPLion`], whose whole update is
    /// `sign(beta1*m + (1-beta1)*g)`. Lion is a stub on M2, so this method has
    /// no caller inside the crate yet. It is on the trait rather than in Lion's
    /// own file because a backend op belongs to the backend: putting it in the
    /// optimizer would hand the eventual GPU backend a scalar loop it cannot
    /// replace.
    ///
    /// Zero maps to zero, matching `f32::signum` everywhere except at zero:
    /// `signum` returns +1.0 for `0.0` and -1.0 for `-0.0`, which would give a
    /// parameter with no momentum a full-size update in an arbitrary direction.
    fn sign(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    /// ReLU activation: y = max(0, x)
    fn relu(x: &Self::Storage, n_elems: usize) -> Result<Self::Storage>;

    // ── Reduction ────────────────────────────────────────────────────────

    /// Sum all elements to a scalar.
    fn sum(a: &Self::Storage) -> Result<f32>;

    /// Mean of all elements.
    fn mean(a: &Self::Storage, n_elems: usize) -> Result<f32>;
}
