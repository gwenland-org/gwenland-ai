//! The engine boundary. glbench runs inference only through [`adapter`], which
//! drives glcore's `Runtime`; it never duplicates inference logic. [`metadata`]
//! and [`capability`] carry the engine/device facts a session records.

pub mod adapter;
pub mod capability;
pub mod metadata;
