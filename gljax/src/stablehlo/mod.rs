//! StableHLO MLIR **text** emission (ARTX01 §2, ARTX02).
//!
//! gljax builds MLIR as a string and hands it to `PJRT_Client_Compile`. There
//! is no MLIR C API dependency and no protobuf: ARTX01 §9.4 settles this —
//! text is what PJRT parses, it is debuggable by `println!`, and the compile
//! call is the validation step anyway.

pub mod emitter;
pub mod ops;
pub mod smoke;
pub mod types;

pub use emitter::{MlirEmitter, SsaName};
pub use types::{DType, ParamKind, Shape};
