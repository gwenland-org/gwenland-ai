//! ARTX01 compatibility shim — the canonical tensor types moved to
//! [`crate::manifest::types`] in ARTX03. (`Shape` was dropped: the new
//! `TensorEntry.shape` is a plain `Vec<u64>` and nothing else used it.)

pub use crate::manifest::types::{DType, TensorEntry};
