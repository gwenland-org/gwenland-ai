//! Compilation options and executable execution.
//!
//! # The protobuf gap
//!
//! ⛔ `PJRT_Client_Compile_Args::compile_options` is a **serialized
//! `CompileOptionsProto`**, not a struct. ARTX01 does not mention this — §6.3
//! discusses caching compiled artifacts but never says the compile call itself
//! needs a protobuf on the way in.
//!
//! Taking `prost`/`protobuf` as a dependency to encode three integers would
//! contradict ARTX01 §5.4's dependency budget, so gljax hand-encodes the
//! handful of bytes instead. Protobuf's wire format makes this small and
//! checkable: a field is `(field_number << 3) | wire_type` followed by a
//! varint or a length-delimited payload.
//!
//! What actually has to be set is narrow:
//!
//! ```text
//! CompileOptionsProto {
//!   executable_build_options = 3 {          // ExecutableBuildOptionsProto
//!     num_replicas   = 4 : 1
//!     num_partitions = 5 : 1
//!   }
//! }
//! ```
//!
//! ⚠️ `num_replicas` and `num_partitions` must be written **explicitly**.
//! proto3 omits default-valued scalars, so leaving them out parses back as
//! `0` replicas and `0` partitions — not as "unset, use 1".
//!
//! `device_ordinal` (field 1) is deliberately omitted: absent means 0, which
//! is the single-device default gljax wants anyway.

use crate::pjrt::client::PjrtClientHandle;
use std::rc::Rc;
use crate::pjrt::error::{check, missing_slot};
use crate::pjrt::event::await_and_destroy;
use crate::sys::types::*;
use crate::{struct_size, GlError};

/// Protobuf wire type 0 — varint.
const WIRE_VARINT: u8 = 0;
/// Protobuf wire type 2 — length-delimited.
const WIRE_LEN: u8 = 2;

/// A protobuf field tag: `(field_number << 3) | wire_type`.
const fn tag(field: u8, wire_type: u8) -> u8 {
    (field << 3) | wire_type
}

/// A serialized `CompileOptionsProto` for a single-device, single-partition
/// compile. See the module docs for what is in it and why.
pub(crate) fn compile_options_single_device() -> Vec<u8> {
    // ExecutableBuildOptionsProto { num_replicas: 1, num_partitions: 1 }
    let build_options = vec![
        tag(4, WIRE_VARINT),
        1, // num_replicas = 1
        tag(5, WIRE_VARINT),
        1, // num_partitions = 1
    ];

    // CompileOptionsProto { executable_build_options: <build_options> }
    let mut out = Vec::with_capacity(build_options.len() + 2);
    out.push(tag(3, WIRE_LEN));
    // Payload is 4 bytes, so the length varint is a single byte. Asserting
    // rather than implementing full varint encoding keeps this honest: if the
    // options ever grow past 127 bytes, this must be revisited, not silently
    // truncated.
    debug_assert!(
        build_options.len() < 128,
        "build_options outgrew single-byte varint length encoding"
    );
    out.push(build_options.len() as u8);
    out.extend_from_slice(&build_options);
    out
}

/// A compiled program loaded on a device.
///
/// Holds an [`Rc`] to the client that compiled it — the executable lives inside
/// the client's allocator, so the client must outlive it.
pub struct LoadedExecutable {
    client: Rc<PjrtClientHandle>,
    raw: *mut PjrtLoadedExecutable,
}

impl LoadedExecutable {
    pub(crate) fn from_raw(client: Rc<PjrtClientHandle>, raw: *mut PjrtLoadedExecutable) -> Self {
        LoadedExecutable { client, raw }
    }

    /// Number of outputs the program produces per device.
    ///
    /// Goes through `PJRT_LoadedExecutable_GetExecutable`, which hands back a
    /// separate `PJRT_Executable` that the caller must destroy.
    pub fn num_outputs(&self) -> Result<usize, GlError> {
        let api = self.client.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(get_exec) = (unsafe { (*api).PJRT_LoadedExecutable_GetExecutable }) else {
            return Err(missing_slot("PJRT_LoadedExecutable_GetExecutable"));
        };
        let mut get_args = PjrtLoadedExecutableGetExecutableArgs {
            struct_size: struct_size!(PjrtLoadedExecutableGetExecutableArgs, executable: *mut PjrtExecutable),
            extension_start: core::ptr::null_mut(),
            loaded_executable: self.raw,
            executable: core::ptr::null_mut(),
        };
        // SAFETY: `get_args` fully initialized; `self.raw` is live.
        let err = unsafe { get_exec(&mut get_args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "LoadedExecutable_GetExecutable")? };

        let executable = get_args.executable;
        if executable.is_null() {
            return Err(GlError::Engine(
                "PJRT_LoadedExecutable_GetExecutable returned a null executable".to_owned(),
            ));
        }

        // SAFETY: vtable is live for the plugin's lifetime.
        let count = match unsafe { (*api).PJRT_Executable_NumOutputs } {
            Some(num_outputs) => {
                let mut args = PjrtExecutableNumOutputsArgs {
                    struct_size: struct_size!(PjrtExecutableNumOutputsArgs, num_outputs: usize),
                    extension_start: core::ptr::null_mut(),
                    executable,
                    num_outputs: 0,
                };
                // SAFETY: `args` fully initialized; `executable` is live.
                let err = unsafe { num_outputs(&mut args) };
                // SAFETY: owned error from the call above.
                unsafe { check(api, err, "Executable_NumOutputs") }.map(|()| args.num_outputs)
            }
            None => Err(missing_slot("PJRT_Executable_NumOutputs")),
        };

        // Destroy the borrowed-out executable whether or not the query worked.
        // SAFETY: vtable is live for the plugin's lifetime.
        if let Some(destroy) = unsafe { (*api).PJRT_Executable_Destroy } {
            let mut args = PjrtExecutableDestroyArgs {
                struct_size: struct_size!(PjrtExecutableDestroyArgs, executable: *mut PjrtExecutable),
                extension_start: core::ptr::null_mut(),
                executable,
            };
            // SAFETY: `args` fully initialized; last use of `executable`.
            let err = unsafe { destroy(&mut args) };
            // SAFETY: owned error from the call above.
            if let Err(e) = unsafe { check(api, err, "Executable_Destroy") } {
                log::warn!("PJRT executable destroy failed: {e}");
            }
        }

        count
    }

    /// Runs the program on one device with the given argument buffers.
    ///
    /// Single-device only: `num_devices` is 1 and `execute_device` is left
    /// null so the executable runs where it was compiled to run. Multi-device
    /// execution is ARTX06's problem.
    pub fn execute(
        &self,
        arguments: &[&crate::pjrt::buffer::PjrtBufferHandle],
    ) -> Result<Vec<crate::pjrt::buffer::PjrtBufferHandle>, GlError> {
        let api = self.client.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_LoadedExecutable_Execute }) else {
            return Err(missing_slot("PJRT_LoadedExecutable_Execute"));
        };

        let num_outputs = self.num_outputs()?;

        // argument_lists is [num_devices][num_args]; both levels are the
        // caller's to allocate.
        let arg_ptrs: Vec<*mut PjrtBuffer> = arguments.iter().map(|b| b.raw()).collect();
        let arg_lists: [*const *mut PjrtBuffer; 1] = [arg_ptrs.as_ptr()];

        // Same for outputs, except PJRT writes into them.
        let mut out_ptrs: Vec<*mut PjrtBuffer> = vec![core::ptr::null_mut(); num_outputs];
        let out_lists: [*mut *mut PjrtBuffer; 1] = [out_ptrs.as_mut_ptr()];

        let mut device_complete_event: *mut PjrtEvent = core::ptr::null_mut();

        let mut options = PjrtExecuteOptions {
            struct_size: struct_size!(PjrtExecuteOptions, num_hlo_output_callbacks: usize),
            extension_start: core::ptr::null_mut(),
            send_callbacks: core::ptr::null_mut(),
            recv_callbacks: core::ptr::null_mut(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id: 0,
            non_donatable_input_indices: core::ptr::null(),
            num_non_donatable_input_indices: 0,
            context: core::ptr::null_mut(),
            call_location: core::ptr::null(),
            num_tasks: 0,
            task_ids: core::ptr::null_mut(),
            incarnation_ids: core::ptr::null_mut(),
            multi_slice_config: core::ptr::null_mut(),
            use_major_to_minor_data_layout_for_callbacks: false,
            hlo_output_callbacks: core::ptr::null_mut(),
            num_hlo_output_callbacks: 0,
        };

        let mut args = PjrtLoadedExecutableExecuteArgs {
            struct_size: struct_size!(PjrtLoadedExecutableExecuteArgs, execute_device: *mut PjrtDevice),
            extension_start: core::ptr::null_mut(),
            executable: self.raw,
            options: &mut options,
            argument_lists: arg_lists.as_ptr(),
            num_devices: 1,
            num_args: arg_ptrs.len(),
            output_lists: out_lists.as_ptr(),
            device_complete_events: &mut device_complete_event,
            execute_device: core::ptr::null_mut(),
        };
        // SAFETY: `args`, `options`, and both pointer arrays live across the
        // call; the output array has room for `num_outputs` entries as PJRT
        // requires.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "LoadedExecutable_Execute")? };

        // On failure PJRT documents that the event is not populated, so this
        // is only reached on success.
        if !device_complete_event.is_null() {
            // SAFETY: a live event owned by us.
            unsafe { await_and_destroy(api, device_complete_event, "Execute")? };
        }

        let mut outputs = Vec::with_capacity(num_outputs);
        for (i, raw) in out_ptrs.into_iter().enumerate() {
            if raw.is_null() {
                return Err(GlError::Engine(format!(
                    "PJRT_LoadedExecutable_Execute left output {i} of {num_outputs} null"
                )));
            }
            outputs.push(crate::pjrt::buffer::PjrtBufferHandle::from_raw(
                Rc::clone(&self.client),
                raw,
            ));
        }
        Ok(outputs)
    }
}

impl Drop for LoadedExecutable {
    fn drop(&mut self) {
        let api = self.client.api();
        // SAFETY: vtable is live — the client outlives this executable.
        let Some(f) = (unsafe { (*api).PJRT_LoadedExecutable_Destroy }) else {
            log::warn!("PJRT plugin exposes no PJRT_LoadedExecutable_Destroy — leaking a program");
            return;
        };
        let mut args = PjrtLoadedExecutableDestroyArgs {
            struct_size: struct_size!(PjrtLoadedExecutableDestroyArgs, executable: *mut PjrtLoadedExecutable),
            extension_start: core::ptr::null_mut(),
            executable: self.raw,
        };
        // SAFETY: `args` fully initialized; last use of `self.raw`.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        if let Err(e) = unsafe { check(api, err, "LoadedExecutable_Destroy") } {
            log::warn!("PJRT loaded-executable destroy failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The encoding is six bytes and every one of them is load-bearing; this
    /// pins them so a "harmless" edit cannot silently produce
    /// `num_replicas: 0`, which XLA rejects far from here.
    #[test]
    fn compile_options_encode_one_replica_one_partition() {
        let bytes = compile_options_single_device();
        assert_eq!(
            bytes,
            vec![
                0x1a, // field 3 (executable_build_options), wire type 2
                0x04, // payload length
                0x20, // field 4 (num_replicas), wire type 0
                0x01, //   = 1
                0x28, // field 5 (num_partitions), wire type 0
                0x01, //   = 1
            ]
        );
    }

    #[test]
    fn protobuf_tag_helper_matches_the_wire_format() {
        // (field_number << 3) | wire_type
        assert_eq!(tag(4, WIRE_VARINT), 0x20);
        assert_eq!(tag(5, WIRE_VARINT), 0x28);
        assert_eq!(tag(3, WIRE_LEN), 0x1a);
    }
}
