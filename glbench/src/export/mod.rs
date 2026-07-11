//! Export: turn a session into bytes. JSON is the archive format (and the basis
//! of the hand-rolled [`json`] value model the whole crate serializes through);
//! [`markdown`] and [`csv`] are human/spreadsheet reports. No external
//! serialization crates.

pub mod csv;
pub mod json;
pub mod markdown;
