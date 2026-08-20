//! Validation: is this benchmark trustworthy? Checks the *conditions* of a run
//! (integrity, determinism, reproducibility) and offers numerical comparison
//! against the glproc oracle. [`integrity::validate`] runs the condition checks;
//! [`numerical`] is called by the caller with both token streams.

pub mod availability;
pub mod deterministic;
pub mod integrity;
pub mod memory;
pub mod numerical;
pub mod parity;
pub mod reproducibility;

pub use integrity::{validate, ValidationReport};
pub use parity::{validate_against_oracle, ParityReport};
