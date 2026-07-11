//! Storage: user-managed archive files, no database. [`archive`] reads/writes a
//! single-session JSON file; [`manifest`] summarizes a directory of them.

pub mod archive;
pub mod manifest;
