//! Raw FFI bindings for the PJRT C API (`xla/pjrt/c/pjrt_c_api.h`).
//!
//! These are **hand-written**, not `bindgen` output. ARTX01 §1.5 assumes
//! `bindgen` generates the `PJRT_*_STRUCT_SIZE` constants; gljax has a
//! zero-build-dependency policy (ARTX01 §5.4: "No `build.rs` monster needed"),
//! so the structs and the size constants are written out by hand instead and
//! [`struct_size!`] reproduces the C `PJRT_STRUCT_SIZE` macro exactly.
//!
//! # Audit contract
//!
//! [`types::PjrtApi`] is a **vtable whose field order is load-bearing**: the
//! plugin hands back a pointer and every call is an offset into it. One field
//! out of order and gljax calls the wrong function with the wrong arguments —
//! P4's silent-wrong-output bug class, in FFI form.
//!
//! Every one of the 138 function-pointer slots is therefore named after its C
//! counterpart and listed in header order, even the ones gljax never calls
//! (those are typed `*mut c_void` so they cannot be invoked by accident). The
//! file is meant to be diffed line-by-line against the header.
//!
//! **Bound against:** `PJRT_API_MAJOR 0`, `PJRT_API_MINOR 114`
//! (openxla/xla `main`, fetched 2026-07-29).

pub mod ffi;
pub mod types;

/// Reproduces the C `PJRT_STRUCT_SIZE(struct_type, last_field)` macro:
/// `offsetof(T, last_field) + sizeof(last_field)`.
///
/// This is deliberately **not** `size_of::<T>()` — the C macro excludes
/// trailing padding, so a caller that sent the padded Rust size would be
/// claiming a larger struct than it actually filled in.
///
/// The field's type is spelled out at the call site because `offset_of!` gives
/// the offset but not the type; keeping both visible makes the constant
/// checkable against the header without following any indirection.
#[macro_export]
macro_rules! struct_size {
    ($t:ty, $field:ident : $field_ty:ty) => {
        ::core::mem::offset_of!($t, $field) + ::core::mem::size_of::<$field_ty>()
    };
}
