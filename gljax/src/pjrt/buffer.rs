//! `PJRT_Buffer` — device-resident tensors and the transfer back to host.
//!
//! ⚠️ ARTX01 §1.7: `PJRT_Buffer` is **not** thread-safe. A buffer belongs to
//! one thread at a time. [`PjrtBufferHandle`] is deliberately neither `Send`
//! nor `Sync` (it holds raw pointers, so it is not by default) — do not add
//! those impls without the synchronization ARTX01 describes.

use crate::pjrt::client::PjrtClientHandle;
use std::rc::Rc;
use crate::pjrt::error::{check, missing_slot};
use crate::pjrt::event::await_and_destroy;
use crate::stablehlo::types::DType;
use crate::sys::types::*;
use crate::{struct_size, GlError};

/// A device buffer, borrowed from the client that owns its memory.
pub struct PjrtBufferHandle {
    client: Rc<PjrtClientHandle>,
    raw: *mut PjrtBuffer,
}

impl PjrtBufferHandle {
    pub(crate) fn from_raw(client: Rc<PjrtClientHandle>, raw: *mut PjrtBuffer) -> Self {
        PjrtBufferHandle { client, raw }
    }

    pub(crate) fn raw(&self) -> *mut PjrtBuffer {
        self.raw
    }

    /// The buffer's element type as PJRT reports it.
    pub fn element_type(&self) -> Result<PjrtBufferType, GlError> {
        let api = self.client.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Buffer_ElementType }) else {
            return Err(missing_slot("PJRT_Buffer_ElementType"));
        };
        let mut args = PjrtBufferElementTypeArgs {
            struct_size: struct_size!(PjrtBufferElementTypeArgs, type_: PjrtBufferType),
            extension_start: core::ptr::null_mut(),
            buffer: self.raw,
            type_: PjrtBufferType::Invalid,
        };
        // SAFETY: `args` fully initialized; `self.raw` is a live buffer.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Buffer_ElementType")? };
        Ok(args.type_)
    }

    /// The buffer's shape.
    pub fn dims(&self) -> Result<Vec<usize>, GlError> {
        let api = self.client.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Buffer_Dimensions }) else {
            return Err(missing_slot("PJRT_Buffer_Dimensions"));
        };
        let mut args = PjrtBufferDimensionsArgs {
            struct_size: struct_size!(PjrtBufferDimensionsArgs, num_dims: usize),
            extension_start: core::ptr::null_mut(),
            buffer: self.raw,
            dims: core::ptr::null(),
            num_dims: 0,
        };
        // SAFETY: `args` fully initialized; `self.raw` is a live buffer.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Buffer_Dimensions")? };

        if args.dims.is_null() || args.num_dims == 0 {
            // A 0-dim tensor is a legitimate result here, not a failure —
            // `tensor<f32>` is how StableHLO spells a scalar.
            return Ok(Vec::new());
        }
        // SAFETY: the plugin wrote `num_dims` values at `dims`, living as long
        // as the buffer.
        let slice = unsafe { core::slice::from_raw_parts(args.dims, args.num_dims) };
        Ok(slice.iter().map(|&d| d as usize).collect())
    }

    /// Copies the buffer back to the host as `f32`.
    ///
    /// Refuses if the device buffer is not F32 rather than reinterpreting the
    /// bytes (P5). Silently reading BF16 as F32 is precisely the failure mode
    /// that produces plausible-looking wrong numbers.
    pub fn to_host_f32(&self) -> Result<Vec<f32>, GlError> {
        let element_type = self.element_type()?;
        if element_type != PjrtBufferType::F32 {
            return Err(GlError::UnsupportedDtype(format!(
                "to_host_f32 called on a {element_type:?} buffer"
            )));
        }
        let numel: usize = self.dims()?.iter().product();
        let mut out = vec![0f32; numel];
        let byte_len = numel * DType::F32.byte_size();

        // SAFETY: `out` is a live, uniquely-owned allocation of exactly
        // `byte_len` bytes with alignment ≥ 4, and f32 has no invalid bit
        // patterns, so any bytes PJRT writes make a valid `[f32]`.
        unsafe {
            self.to_host_raw(out.as_mut_ptr().cast::<core::ffi::c_void>(), byte_len)?;
        }
        Ok(out)
    }

    /// # Safety
    ///
    /// `dst` must be valid for writes of `dst_size` bytes, and the caller must
    /// be able to interpret whatever the device's element type writes there.
    pub(crate) unsafe fn to_host_raw(
        &self,
        dst: *mut core::ffi::c_void,
        dst_size: usize,
    ) -> Result<(), GlError> {
        let api = self.client.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Buffer_ToHostBuffer }) else {
            return Err(missing_slot("PJRT_Buffer_ToHostBuffer"));
        };
        let mut args = PjrtBufferToHostBufferArgs {
            struct_size: struct_size!(PjrtBufferToHostBufferArgs, event: *mut PjrtEvent),
            extension_start: core::ptr::null_mut(),
            src: self.raw,
            host_layout: core::ptr::null_mut(),
            dst,
            dst_size,
            event: core::ptr::null_mut(),
        };
        // SAFETY: `args` fully initialized; `dst` is valid for `dst_size`
        // bytes per this function's contract.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Buffer_ToHostBuffer")? };

        // The copy is asynchronous: without this await, `dst` may still be
        // uninitialized when the caller reads it.
        if args.event.is_null() {
            return Err(GlError::Engine(
                "PJRT_Buffer_ToHostBuffer returned no completion event — cannot know when the \
                 host copy is finished"
                    .to_owned(),
            ));
        }
        // SAFETY: a live event owned by us.
        unsafe { await_and_destroy(api, args.event, "Buffer_ToHostBuffer") }
    }
}

impl Drop for PjrtBufferHandle {
    fn drop(&mut self) {
        let api = self.client.api();
        // SAFETY: vtable is live — the client outlives this buffer.
        let Some(f) = (unsafe { (*api).PJRT_Buffer_Destroy }) else {
            log::warn!("PJRT plugin exposes no PJRT_Buffer_Destroy — leaking device memory");
            return;
        };
        let mut args = PjrtBufferDestroyArgs {
            struct_size: struct_size!(PjrtBufferDestroyArgs, buffer: *mut PjrtBuffer),
            extension_start: core::ptr::null_mut(),
            buffer: self.raw,
        };
        // SAFETY: `args` fully initialized; last use of `self.raw`.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        if let Err(e) = unsafe { check(api, err, "Buffer_Destroy") } {
            log::warn!("PJRT buffer destroy failed: {e}");
        }
    }
}
