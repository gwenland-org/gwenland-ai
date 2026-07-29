//! # gljax — a pure-Rust XLA/PJRT client
//!
//! gljax emits StableHLO MLIR text and hands it to a dynamically loaded PJRT
//! plugin. It owns no kernels: no CUDA, no PTX, no hand-rolled matmul. What
//! XLA does with the IR is XLA's decision — gljax's job is to state the
//! computation portably and then **measure** what the backend actually did.
//!
//! Design series: `gljax/architecture/ARTX01`…`ARTX16`.
//! Start at `Overall-Architecture.md`.
//!
//! ## The five principles
//!
//! * **P1** — gljax produces StableHLO and owns no kernels.
//! * **P2** — emit standard IR, let the backend decide, measure whether it did.
//! * **P3** — static shapes are the organizing constraint; each feature adds a
//!   compile-cache key dimension.
//! * **P4** — the bug class is *silent wrong output*, and it recurs everywhere.
//! * **P5** — refuse rather than approximate.
//!
//! ## Status
//!
//! Wave A1 of the ARTX01–05 bring-up: the crate exists, the PJRT FFI is
//! written, and the StableHLO emitter produces text.
//!
//! ⛔ **Nothing in `pjrt` has been executed against a real plugin.** See
//! `gljax/README.md` — there is no PJRT plugin binary for Windows, and this is
//! a Windows machine. Everything under [`stablehlo`] is exercised by tests;
//! everything under [`pjrt`] and [`sys`] is written against
//! `xla/pjrt/c/pjrt_c_api.h` at API 0.114 and is unverified at runtime.
//!
//! ## Quick start (on a platform with a plugin)
//!
//! ```no_run
//! use gljax::pjrt::{PjrtClientHandle, PjrtPlugin};
//! use gljax::stablehlo::{smoke, DType, Shape};
//!
//! # fn main() -> Result<(), gljax::GlError> {
//! let plugin = std::rc::Rc::new(PjrtPlugin::load_cpu_from_env()?);
//! let client = PjrtClientHandle::create(plugin)?;
//! let device = client.default_device()?;
//!
//! let program = client.compile(&smoke::add_scalar_module())?;
//! let scalar = Shape::scalar(DType::F32);
//! let a = client.buffer_from_host_f32(&[2.0], &scalar, &device)?;
//! let b = client.buffer_from_host_f32(&[3.0], &scalar, &device)?;
//!
//! let out = program.execute(&[&a, &b])?;
//! assert_eq!(out[0].to_host_f32()?, vec![5.0]);
//! # Ok(())
//! # }
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

pub mod arch;
pub mod checkpoint;
pub mod error;
pub mod graph;
pub mod matrix;
pub mod model;
pub mod oracle;
pub mod ops;
pub mod pjrt;
pub mod precision;
pub mod runtime;
pub mod stablehlo;
pub mod sys;
pub mod tensor;
pub mod tok;

pub use error::GlError;
pub use graph::{BuiltFunc, FuncBuilder, Signature, SsaValue, TraceCx};
pub use matrix::{DotAlgorithm, DotNumerics, MatmulOpts};
pub use precision::{with_policy, PrecisionPolicy};
pub use stablehlo::{DType, Shape};
pub use tensor::Tensor;

/// The PJRT C API version gljax's bindings were written against.
pub const PJRT_API_VERSION_BOUND: (i32, i32) =
    (sys::types::PJRT_API_MAJOR, sys::types::PJRT_API_MINOR);
