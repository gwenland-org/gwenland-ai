//! A safe handle to a `PJRT_Device`.
//!
//! `PJRT_Device*` is owned by the client and has the client's lifetime — the
//! caller never frees one. Wrapping it keeps raw pointers out of the public API
//! (`rust-skills/unsafe-rules.md` rule 4).

use crate::sys::types::PjrtDevice;

/// A device belonging to a client.
///
/// `Copy` because it is a borrowed handle, not an owned resource: there is
/// nothing to destroy and duplicating it costs nothing.
///
/// ⚠️ It carries no back-reference to its client, so using one after its client
/// is dropped is a use-after-free that the type system will not catch. Every
/// gljax path obtains a device from the client it is about to call, which keeps
/// the window closed; that is a convention, not a proof.
#[derive(Clone, Copy)]
pub struct PjrtDeviceRef {
    raw: *mut PjrtDevice,
}

impl PjrtDeviceRef {
    pub(crate) fn from_raw(raw: *mut PjrtDevice) -> Self {
        PjrtDeviceRef { raw }
    }

    pub(crate) fn raw(self) -> *mut PjrtDevice {
        self.raw
    }
}
