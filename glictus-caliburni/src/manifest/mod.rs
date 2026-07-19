//! `gllm.json` manifest — parsing, data types, and semantic validation
//! (ARTX03).
//!
//! The manifest is the single source of truth for a GLLM package: after
//! parsing it, a runtime can construct the execution graph, plan memory,
//! and verify integrity without reading any tensor data.

pub mod types;

pub use types::{DType, ExtensionUri, TensorEntry, known_extensions};
