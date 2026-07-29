//! Error type.
//!
//! gljax uses [`glcore::error::GlError`] like every other GL crate — no
//! crate-local error enum. PJRT failures land in [`GlError::Engine`], which is
//! documented as "init, load, inference, missing hardware"; a plugin that will
//! not load or a program that will not compile is exactly that.
//!
//! ⚠️ The sprint brief suggests adding a `GlError::Pjrt(String)` variant. That
//! is deliberately **not** done here: no caller today distinguishes a PJRT
//! failure from any other engine failure, and a variant nobody matches on is
//! churn across every crate that touches `GlError`. Every message this crate
//! produces is prefixed `PJRT <call>:` so the origin is still greppable. Add
//! the variant when something actually branches on it.

pub use glcore::error::GlError;
