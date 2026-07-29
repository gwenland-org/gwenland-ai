//! `PJRT_Event` disposal.
//!
//! Every async PJRT call hands back an event the caller owns. gljax v1 is
//! synchronous end to end, so the only thing it ever does with one is block on
//! it and free it — there is no completion-callback path yet.

use crate::pjrt::error::{check, missing_slot};
use crate::sys::types::*;
use crate::{struct_size, GlError};

/// Blocks until `event` completes, then destroys it.
///
/// The await's own status is the operation's status: `PJRT_Event_Await`
/// returns the error the asynchronous work failed with, not an error about
/// waiting.
///
/// # Safety
///
/// `api` must be a live vtable and `event` a non-null event owned by the
/// caller and not yet destroyed. `event` is dangling on return.
pub(crate) unsafe fn await_and_destroy(
    api: *const PjrtApi,
    event: *mut PjrtEvent,
    context: &str,
) -> Result<(), GlError> {
    // SAFETY: caller guarantees `api` is live.
    let Some(await_fn) = (unsafe { (*api).PJRT_Event_Await }) else {
        return Err(missing_slot("PJRT_Event_Await"));
    };
    let mut await_args = PjrtEventAwaitArgs {
        struct_size: struct_size!(PjrtEventAwaitArgs, event: *mut PjrtEvent),
        extension_start: core::ptr::null_mut(),
        event,
    };
    // SAFETY: `await_args` fully initialized; `event` is live.
    let err = unsafe { await_fn(&mut await_args) };
    // SAFETY: owned error from the call above.
    let awaited = unsafe { check(api, err, context) };

    // Destroy regardless of the await's outcome: an event that failed is
    // still an event we own.
    // SAFETY: caller guarantees `api` is live.
    if let Some(destroy_fn) = unsafe { (*api).PJRT_Event_Destroy } {
        let mut destroy_args = PjrtEventDestroyArgs {
            struct_size: struct_size!(PjrtEventDestroyArgs, event: *mut PjrtEvent),
            extension_start: core::ptr::null_mut(),
            event,
        };
        // SAFETY: `destroy_args` fully initialized; last use of `event`.
        let err = unsafe { destroy_fn(&mut destroy_args) };
        // SAFETY: owned error from the call above.
        if let Err(e) = unsafe { check(api, err, "Event_Destroy") } {
            log::warn!("PJRT event destroy failed after {context}: {e}");
        }
    } else {
        log::warn!("PJRT plugin exposes no PJRT_Event_Destroy — leaking an event from {context}");
    }

    awaited
}
