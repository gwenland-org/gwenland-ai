//! Schema and version constants, plus the [`ToJson`] convention every part of
//! the [`crate::core::session::BenchmarkSession`] implements.
//!
//! The data model is serialization-agnostic in principle, but in practice JSON
//! is the archive format, so each data struct knows how to render itself to a
//! [`Json`] value and (where it needs to round-trip) how to read itself back.

use crate::export::json::Json;

/// The glbench crate version, stamped into every archived session so a reader
/// can tell which tool produced a file.
pub const GLBENCH_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The archive schema version. Bump this only on a breaking change to the
/// on-disk [`crate::core::session::BenchmarkSession`] JSON shape.
pub const SCHEMA_VERSION: u32 = 1;

/// Everything in the data model can render itself to a [`Json`] value.
pub trait ToJson {
    /// Produce the JSON representation of this value.
    fn to_json(&self) -> Json;
}

/// Data that also needs to be read back from an archive implements this.
pub trait FromJson: Sized {
    /// Parse this value from a [`Json`], returning a message on shape mismatch.
    fn from_json(v: &Json) -> Result<Self, String>;
}

/// Helper: pull a required object field or return a descriptive error.
pub fn field<'a>(v: &'a Json, key: &str) -> Result<&'a Json, String> {
    v.get(key).ok_or_else(|| format!("missing field '{key}'"))
}

/// Helper: read a required f64 field.
pub fn field_f64(v: &Json, key: &str) -> Result<f64, String> {
    field(v, key)?
        .as_f64()
        .ok_or_else(|| format!("field '{key}' is not a number"))
}

/// Helper: read a required string field.
pub fn field_str(v: &Json, key: &str) -> Result<String, String> {
    field(v, key)?
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("field '{key}' is not a string"))
}
