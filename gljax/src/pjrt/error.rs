//! `PJRT_Error*` → [`GlError`] conversion.
//!
//! ARTX01 §1.6: `PJRT_Error*` is an **owned** pointer. Non-null means failure,
//! and not destroying it leaks. The message must be read *before* the destroy,
//! because it has the lifetime of the error object.
//!
//! Every call into the plugin goes through [`check`]. A `PJRT_Error*` must
//! never be propagated up the call stack.

use crate::sys::types::*;
use crate::{struct_size, GlError};

/// Converts the result of a PJRT call into a `Result`, consuming and
/// destroying the error object either way.
///
/// # Safety
///
/// * `api` must be the live vtable the call was made through.
/// * `err` must be the `PJRT_Error*` that call returned, and must not be used
///   by the caller afterwards — this function takes ownership of it.
pub(crate) unsafe fn check(api: *const PjrtApi, err: *mut PjrtError, context: &str) -> Result<(), GlError> {
    if err.is_null() {
        return Ok(());
    }
    // SAFETY: `err` is non-null and owned by us per the contract above; `api`
    // is live. The two accessors are mandatory slots present in every PJRT
    // version gljax accepts (checked in `plugin::PjrtPlugin::load`).
    let message = unsafe { error_message(api, err) };
    let code = unsafe { error_code(api, err) };
    // SAFETY: same, and this is the last use of `err`.
    unsafe { error_destroy(api, err) };

    Err(GlError::Engine(match code {
        Some(c) => format!("PJRT {context}: {c:?}: {message}"),
        None => format!("PJRT {context}: {message}"),
    }))
}

/// Reads the human-readable message out of an error object.
///
/// # Safety
/// `api` live, `err` a valid non-null error that has not yet been destroyed.
unsafe fn error_message(api: *const PjrtApi, err: *mut PjrtError) -> String {
    // SAFETY: caller guarantees `api` points at a live vtable.
    let Some(f) = (unsafe { (*api).PJRT_Error_Message }) else {
        return "<plugin has no PJRT_Error_Message>".to_owned();
    };
    let mut args = PjrtErrorMessageArgs {
        struct_size: struct_size!(PjrtErrorMessageArgs, message_size: usize),
        extension_start: core::ptr::null_mut(),
        error: err,
        message: core::ptr::null(),
        message_size: 0,
    };
    // SAFETY: `args` is fully initialized with a correct `struct_size`; the
    // plugin writes `message`/`message_size` and returns nothing.
    unsafe { f(&mut args) };
    if args.message.is_null() || args.message_size == 0 {
        return "<empty PJRT error message>".to_owned();
    }
    // SAFETY: the plugin promises `message` points at `message_size` bytes
    // living as long as `err`, which has not been destroyed yet. The bytes are
    // not guaranteed UTF-8, so this is the lossy conversion, not
    // `from_utf8_unchecked` as ARTX01 §1.6 sketches.
    let bytes = unsafe { core::slice::from_raw_parts(args.message.cast::<u8>(), args.message_size) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Reads the status code, or `None` if the plugin predates `PJRT_Error_GetCode`.
///
/// # Safety
/// `api` live, `err` a valid non-null error that has not yet been destroyed.
unsafe fn error_code(api: *const PjrtApi, err: *mut PjrtError) -> Option<PjrtErrorCode> {
    // SAFETY: caller guarantees `api` points at a live vtable.
    let f = unsafe { (*api).PJRT_Error_GetCode }?;
    let mut args = PjrtErrorGetCodeArgs {
        struct_size: struct_size!(PjrtErrorGetCodeArgs, code: PjrtErrorCode),
        extension_start: core::ptr::null_mut(),
        error: err,
        code: PjrtErrorCode::Ok,
    };
    // SAFETY: `args` fully initialized. `PJRT_Error_GetCode` may itself return
    // an error object; we leak nothing by ignoring the code and destroying it.
    let nested = unsafe { f(&mut args) };
    if !nested.is_null() {
        // SAFETY: `nested` is a fresh owned error, independent of `err`.
        unsafe { error_destroy(api, nested) };
        return None;
    }
    Some(args.code)
}

/// Frees an error object.
///
/// # Safety
/// `api` live, `err` valid and not previously destroyed. `err` is dangling
/// afterwards.
unsafe fn error_destroy(api: *const PjrtApi, err: *mut PjrtError) {
    // SAFETY: caller guarantees `api` points at a live vtable.
    let Some(f) = (unsafe { (*api).PJRT_Error_Destroy }) else {
        // Nothing we can do but leak; say so rather than pretend.
        log::warn!("PJRT plugin exposes no PJRT_Error_Destroy — leaking an error object");
        return;
    };
    let mut args = PjrtErrorDestroyArgs {
        struct_size: struct_size!(PjrtErrorDestroyArgs, error: *mut PjrtError),
        extension_start: core::ptr::null_mut(),
        error: err,
    };
    // SAFETY: `args` fully initialized; this is the last use of `err`.
    unsafe { f(&mut args) };
}

/// Reads a `(ptr, len)` string pair the plugin owns.
///
/// # Safety
/// `ptr`/`len` must describe bytes that outlive this call.
pub(crate) unsafe fn borrowed_str(ptr: *const core::ffi::c_char, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    // SAFETY: caller guarantees the range is valid for `len` bytes.
    let bytes = unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// The vtable slot a call needs was null.
///
/// A missing slot is not a bug in gljax and not a corrupt plugin — it is an
/// older plugin that predates the entry point. P5: refuse, with the name.
pub(crate) fn missing_slot(name: &str) -> GlError {
    GlError::Engine(format!(
        "PJRT plugin does not implement {name} — the plugin is older than the API gljax needs"
    ))
}
