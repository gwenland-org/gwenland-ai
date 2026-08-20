//! Stummañ Karg — backend implementations.
//!
//! Wave 1 ships CPU only:
//! - [`GlProc`] — production path, glproc's SIMD-dispatched kernels.
//! - [`SisdBackend`] — pure-scalar reference, the oracle GlProc is checked
//!   against (and the numerical reference for the Wave 4 gradient check).
//!
//! GlCuda (GPU/PTX) and GlJax (TPU/PJRT) land in M4.

pub mod glproc;
pub mod sisd;

pub use glproc::GlProc;
pub use sisd::SisdBackend;
