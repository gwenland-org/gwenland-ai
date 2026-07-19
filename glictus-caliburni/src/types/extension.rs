//! ARTX01 compatibility shim — the canonical `ExtensionUri` moved to
//! [`crate::manifest::types`] in ARTX03 (now a validated string newtype
//! with accessor methods instead of a struct of owned fields).

pub use crate::manifest::types::{ExtensionUri, known_extensions};
