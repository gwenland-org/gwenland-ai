//! Runtime — compile cache, execution plan, session (ARTX04).
//!
//! ⛔ **Split by what has been executed.** The device half of this module has
//! never run: there is no PJRT plugin for Windows (see `gljax/README.md`).
//!
//! | Module | Executed? |
//! |---|---|
//! | [`digest`] | ✅ against the published SHA-256 vectors |
//! | [`cache`] | ✅ real files on disk |
//! | [`plan`] | ✅ pure host-side validation |
//! | [`bucket`] | ✅ pure host-side logic |
//! | [`sample`] | ✅ pure host-side logic |
//! | [`session`] | ⛔ compiles, never constructed |

pub mod bucket;
pub mod cache;
pub mod digest;
pub mod plan;
pub mod sample;
pub mod session;

pub use bucket::{bucket_for, pad_to_bucket, BUCKETS};
pub use cache::{CompileCache, CompileKey};
pub use sample::{argmax, argmax_at};
pub use plan::PlanSignature;
pub use session::Session;
