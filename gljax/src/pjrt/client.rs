//! `PJRT_Client` lifecycle and device enumeration (ARTX01 §1.2, §4.1).

use crate::pjrt::buffer::PjrtBufferHandle;
use crate::pjrt::compile::{compile_options_single_device, LoadedExecutable};
use crate::pjrt::device::PjrtDeviceRef;
use crate::pjrt::error::{borrowed_str, check, missing_slot};
use crate::pjrt::plugin::PjrtPlugin;
use std::rc::Rc;
use crate::stablehlo::types::{DType, Shape};
use crate::sys::types::*;
use crate::{struct_size, GlError};

/// PJRT's format string for MLIR input. ARTX01 §9.4: gljax emits **text**,
/// which this format accepts alongside bytecode.
const PROGRAM_FORMAT_MLIR: &str = "mlir";

/// A live `PJRT_Client`.
///
/// Holds an [`Rc`] to the plugin rather than a borrow: every pointer in the
/// client is code and data inside the plugin binary, so the plugin must outlive
/// it — and a `Session` (ARTX01 §1.3) needs to own both in one struct, which a
/// lifetime cannot express.
///
/// ⚠️ `Rc`, not `Arc`. ARTX01 §1.7 says `PJRT_Client` *is* thread-safe for
/// concurrent compile and execute, but `PJRT_Buffer` is not, and gljax v1's
/// runtime is single-threaded throughout. Reaching for `Arc` means designing
/// the buffer ownership rules first, not just changing the pointer type.
pub struct PjrtClientHandle {
    plugin: Rc<PjrtPlugin>,
    raw: *mut PjrtClient,
}

impl PjrtClientHandle {
    /// Creates a client with default options.
    ///
    /// gljax passes no `create_options` and no key-value callbacks: those are
    /// the multi-process coordination path, which ARTX16 §2.1 records as
    /// entirely undesigned.
    pub fn create(plugin: Rc<PjrtPlugin>) -> Result<Rc<Self>, GlError> {
        let api = plugin.api();
        // SAFETY: the plugin holds the library open, so the vtable is live.
        let Some(f) = (unsafe { (*api).PJRT_Client_Create }) else {
            return Err(missing_slot("PJRT_Client_Create"));
        };
        let mut args = PjrtClientCreateArgs {
            struct_size: struct_size!(PjrtClientCreateArgs, kv_try_get_user_arg: *mut core::ffi::c_void),
            extension_start: core::ptr::null_mut(),
            create_options: core::ptr::null(),
            num_options: 0,
            kv_get_callback: core::ptr::null_mut(),
            kv_get_user_arg: core::ptr::null_mut(),
            kv_put_callback: core::ptr::null_mut(),
            kv_put_user_arg: core::ptr::null_mut(),
            client: core::ptr::null_mut(),
            kv_try_get_callback: core::ptr::null_mut(),
            kv_try_get_user_arg: core::ptr::null_mut(),
        };
        // SAFETY: `args` is fully initialized with a correct `struct_size`.
        let err = unsafe { f(&mut args) };
        // SAFETY: `err` is the owned error this call returned.
        unsafe { check(api, err, "Client_Create")? };

        if args.client.is_null() {
            return Err(GlError::Engine(
                "PJRT_Client_Create reported success but returned a null client".to_owned(),
            ));
        }
        Ok(Rc::new(PjrtClientHandle {
            plugin,
            raw: args.client,
        }))
    }

    pub(crate) fn api(&self) -> *const PjrtApi {
        self.plugin.api()
    }

    /// The plugin backing this client.
    pub fn plugin(&self) -> &Rc<PjrtPlugin> {
        &self.plugin
    }

    /// Platform identifier: `"cpu"`, `"cuda"`, `"tpu"`, ...
    pub fn platform_name(&self) -> Result<String, GlError> {
        let api = self.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Client_PlatformName }) else {
            return Err(missing_slot("PJRT_Client_PlatformName"));
        };
        let mut args = PjrtClientPlatformNameArgs {
            struct_size: struct_size!(PjrtClientPlatformNameArgs, platform_name_size: usize),
            extension_start: core::ptr::null_mut(),
            client: self.raw,
            platform_name: core::ptr::null(),
            platform_name_size: 0,
        };
        // SAFETY: `args` fully initialized; `self.raw` is a live client.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Client_PlatformName")? };
        // SAFETY: the plugin promises the string outlives the client.
        Ok(unsafe { borrowed_str(args.platform_name, args.platform_name_size) })
    }

    /// Human-readable backend version string.
    pub fn platform_version(&self) -> Result<String, GlError> {
        let api = self.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Client_PlatformVersion }) else {
            return Err(missing_slot("PJRT_Client_PlatformVersion"));
        };
        let mut args = PjrtClientPlatformVersionArgs {
            struct_size: struct_size!(PjrtClientPlatformVersionArgs, platform_version_size: usize),
            extension_start: core::ptr::null_mut(),
            client: self.raw,
            platform_version: core::ptr::null(),
            platform_version_size: 0,
        };
        // SAFETY: `args` fully initialized; `self.raw` is a live client.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Client_PlatformVersion")? };
        // SAFETY: the plugin promises the string outlives the client.
        Ok(unsafe { borrowed_str(args.platform_version, args.platform_version_size) })
    }

    /// Devices this client can issue commands to.
    ///
    /// The handles are owned by the client and must not outlive it, which the
    /// borrow inside [`PjrtDeviceRef`] enforces.
    pub fn addressable_devices(&self) -> Result<Vec<PjrtDeviceRef>, GlError> {
        let api = self.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Client_AddressableDevices }) else {
            return Err(missing_slot("PJRT_Client_AddressableDevices"));
        };
        let mut args = PjrtClientAddressableDevicesArgs {
            struct_size: struct_size!(PjrtClientAddressableDevicesArgs, num_addressable_devices: usize),
            extension_start: core::ptr::null_mut(),
            client: self.raw,
            addressable_devices: core::ptr::null(),
            num_addressable_devices: 0,
        };
        // SAFETY: `args` fully initialized; `self.raw` is a live client.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Client_AddressableDevices")? };

        if args.addressable_devices.is_null() || args.num_addressable_devices == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: the plugin wrote `num_addressable_devices` pointers at
        // `addressable_devices`, owned by and outliving the client.
        let slice = unsafe {
            core::slice::from_raw_parts(args.addressable_devices, args.num_addressable_devices)
        };
        Ok(slice
            .iter()
            .map(|&raw| PjrtDeviceRef::from_raw(raw))
            .collect())
    }

    /// The first addressable device, or an error naming the platform if there
    /// are none. Every gljax path today is single-device (ARTX01 §4.2:
    /// "Skip Shardy" for v1).
    pub fn default_device(&self) -> Result<PjrtDeviceRef, GlError> {
        let devices = self.addressable_devices()?;
        devices.first().copied().ok_or_else(|| {
            GlError::Engine(format!(
                "PJRT client on platform {:?} has no addressable devices",
                self.platform_name().unwrap_or_else(|_| "<unknown>".into())
            ))
        })
    }

    /// Compiles StableHLO MLIR **text** into a loaded executable.
    ///
    /// ARTX01 §2.4: there is no separate validate-only entry point, so a
    /// malformed module surfaces here as a compile error carrying the MLIR
    /// verifier's output. Set `GLJAX_DUMP_MLIR=1` to log the module text that
    /// was rejected.
    pub fn compile(self: &Rc<Self>, mlir_text: &str) -> Result<LoadedExecutable, GlError> {
        if std::env::var("GLJAX_DUMP_MLIR").is_ok_and(|v| v != "0") {
            log::info!("gljax compiling module:\n{mlir_text}");
        }

        let api = self.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Client_Compile }) else {
            return Err(missing_slot("PJRT_Client_Compile"));
        };

        // `PJRT_Program::code` is `char*` (non-const) but only read during the
        // call; a local copy keeps the caller's `&str` immutable regardless.
        let mut code = mlir_text.as_bytes().to_vec();
        let options = compile_options_single_device();

        let program = PjrtProgram {
            struct_size: struct_size!(PjrtProgram, format_size: usize),
            extension_start: core::ptr::null_mut(),
            code: code.as_mut_ptr().cast::<core::ffi::c_char>(),
            code_size: code.len(),
            format: PROGRAM_FORMAT_MLIR.as_ptr().cast::<core::ffi::c_char>(),
            format_size: PROGRAM_FORMAT_MLIR.len(),
        };
        let mut args = PjrtClientCompileArgs {
            struct_size: struct_size!(PjrtClientCompileArgs, executable: *mut PjrtLoadedExecutable),
            extension_start: core::ptr::null_mut(),
            client: self.raw,
            program: &program,
            compile_options: options.as_ptr().cast::<core::ffi::c_char>(),
            compile_options_size: options.len(),
            executable: core::ptr::null_mut(),
        };
        // SAFETY: `args`, `program` and both byte buffers are alive for the
        // whole call, which is all PJRT requires of them ("Only needs to stay
        // alive for the duration of the Compile call").
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Client_Compile")? };

        // Keep the buffers alive until after the call, explicitly.
        drop(code);
        drop(options);

        if args.executable.is_null() {
            return Err(GlError::Engine(
                "PJRT_Client_Compile reported success but returned a null executable".to_owned(),
            ));
        }
        Ok(LoadedExecutable::from_raw(Rc::clone(self), args.executable))
    }

    /// Uploads a host `f32` slice as a device buffer of the given shape.
    ///
    /// Refuses on a dtype/length mismatch rather than transferring the wrong
    /// number of bytes (P5) — a length that disagrees with the shape is
    /// exactly the kind of error that otherwise shows up as wrong numbers.
    pub fn buffer_from_host_f32(
        self: &Rc<Self>,
        data: &[f32],
        shape: &Shape,
        device: &PjrtDeviceRef,
    ) -> Result<PjrtBufferHandle, GlError> {
        if shape.dtype != DType::F32 {
            return Err(GlError::UnsupportedDtype(format!(
                "buffer_from_host_f32 called with a {:?} shape",
                shape.dtype
            )));
        }
        if data.len() != shape.numel() {
            return Err(GlError::ShapeMismatch {
                expected: shape.dims.clone(),
                got: vec![data.len()],
            });
        }
        // SAFETY: `data` is a live `&[f32]` for the duration of the call, and
        // `ImmutableOnlyDuringCall` is exactly the promise that covers that.
        unsafe {
            self.buffer_from_host_raw(
                data.as_ptr().cast::<core::ffi::c_void>(),
                PjrtBufferType::F32,
                &shape.dims,
                device,
            )
        }
    }

    /// # Safety
    ///
    /// `data` must point at `shape.numel()` elements of `element_type`, valid
    /// for reads for the duration of this call.
    pub(crate) unsafe fn buffer_from_host_raw(
        self: &Rc<Self>,
        data: *const core::ffi::c_void,
        element_type: PjrtBufferType,
        dims: &[usize],
        device: &PjrtDeviceRef,
    ) -> Result<PjrtBufferHandle, GlError> {
        let api = self.api();
        // SAFETY: vtable is live for the plugin's lifetime.
        let Some(f) = (unsafe { (*api).PJRT_Client_BufferFromHostBuffer }) else {
            return Err(missing_slot("PJRT_Client_BufferFromHostBuffer"));
        };

        let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let mut args = PjrtClientBufferFromHostBufferArgs {
            struct_size: struct_size!(PjrtClientBufferFromHostBufferArgs, buffer: *mut PjrtBuffer),
            extension_start: core::ptr::null_mut(),
            client: self.raw,
            data,
            type_: element_type,
            dims: dims_i64.as_ptr(),
            num_dims: dims_i64.len(),
            // Empty byte_strides = dense, major-to-minor. Anything else would
            // have to agree with what the emitter declared in the MLIR type.
            byte_strides: core::ptr::null(),
            num_byte_strides: 0,
            host_buffer_semantics: PjrtHostBufferSemantics::ImmutableOnlyDuringCall,
            device: device.raw(),
            memory: core::ptr::null_mut(),
            device_layout: core::ptr::null_mut(),
            done_with_host_buffer: core::ptr::null_mut(),
            buffer: core::ptr::null_mut(),
        };
        // SAFETY: `args` fully initialized; `data` is valid per this function's
        // contract; `dims_i64` outlives the call.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above.
        unsafe { check(api, err, "Client_BufferFromHostBuffer")? };

        // Under `ImmutableOnlyDuringCall` the runtime is already done with the
        // host pointer, but it still hands back an event that we own.
        if !args.done_with_host_buffer.is_null() {
            // SAFETY: a live event owned by us; awaiting then destroying it is
            // the documented disposal.
            unsafe { crate::pjrt::event::await_and_destroy(api, args.done_with_host_buffer, "BufferFromHostBuffer")? };
        }

        if args.buffer.is_null() {
            return Err(GlError::Engine(
                "PJRT_Client_BufferFromHostBuffer reported success but returned a null buffer"
                    .to_owned(),
            ));
        }
        Ok(PjrtBufferHandle::from_raw(Rc::clone(self), args.buffer))
    }
}

impl Drop for PjrtClientHandle {
    fn drop(&mut self) {
        let api = self.plugin.api();
        // SAFETY: vtable is live — the plugin outlives this client by the
        // `'p` borrow.
        let Some(f) = (unsafe { (*api).PJRT_Client_Destroy }) else {
            log::warn!("PJRT plugin exposes no PJRT_Client_Destroy — leaking a client");
            return;
        };
        let mut args = PjrtClientDestroyArgs {
            struct_size: struct_size!(PjrtClientDestroyArgs, client: *mut PjrtClient),
            extension_start: core::ptr::null_mut(),
            client: self.raw,
        };
        // SAFETY: `args` fully initialized; this is the last use of `self.raw`.
        let err = unsafe { f(&mut args) };
        // SAFETY: owned error from the call above. Drop cannot return it, so
        // it is logged — a swallowed teardown failure is how leaks stay hidden.
        if let Err(e) = unsafe { check(api, err, "Client_Destroy") } {
            log::warn!("PJRT client destroy failed: {e}");
        }
    }
}
