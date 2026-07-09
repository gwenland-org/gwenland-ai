//! Safe wrapper over the raw CUDA Driver API.
//!
//! Owns exactly the responsibilities ArchGLML_X2 §16 assigns the host side:
//! device detection, context lifetime, PTX module loading, memory transfer
//! and kernel launch. Everything numeric happens in the PTX kernels.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use glcore::GlError;

use crate::ffi::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CUresult, DriverApi,
    ATTR_COMPUTE_CAPABILITY_MAJOR, ATTR_COMPUTE_CAPABILITY_MINOR, ATTR_MULTIPROCESSOR_COUNT,
    CUDA_SUCCESS,
};

/// The process-wide driver API table, loaded on first use. `None` when the
/// machine has no CUDA driver — cached so repeated probes cost one atomic
/// load, not a filesystem search.
fn api() -> Option<&'static Arc<DriverApi>> {
    static API: OnceLock<Option<Arc<DriverApi>>> = OnceLock::new();
    API.get_or_init(|| DriverApi::load().ok().map(Arc::new)).as_ref()
}

/// Map a `CUresult` to a `GlError`, naming the failing call.
fn check(api: &DriverApi, res: CUresult, what: &str) -> Result<(), GlError> {
    if res == CUDA_SUCCESS {
        return Ok(());
    }
    let mut name: *const i8 = std::ptr::null();
    // SAFETY: cu_get_error_name writes a static string pointer or leaves
    // it null for unknown codes.
    let known = unsafe { (api.cu_get_error_name)(res, &mut name) } == CUDA_SUCCESS;
    let name = if known && !name.is_null() {
        // SAFETY: the driver returns a NUL-terminated static string.
        unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned()
    } else {
        format!("CUDA error {res}")
    };
    Err(GlError::Engine(format!("{what} failed: {name}")))
}

/// True when a CUDA driver *and* at least one device are present. Cached;
/// safe to call from `capabilities()` on every render tick.
pub fn cuda_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let Some(api) = api() else { return false };
        // SAFETY: cuInit is idempotent; count pointer is valid.
        unsafe {
            if (api.cu_init)(0) != CUDA_SUCCESS {
                return false;
            }
            let mut n = 0i32;
            (api.cu_device_get_count)(&mut n) == CUDA_SUCCESS && n > 0
        }
    })
}

/// Static facts about the selected device, gathered once at probe time.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Marketing name, e.g. `"NVIDIA GeForce GTX 1660"`.
    pub name: String,
    /// Compute capability major (M2 requires ≥ 7 per ArchGLML_X2 §6).
    pub sm_major: i32,
    /// Compute capability minor.
    pub sm_minor: i32,
    /// Number of streaming multiprocessors.
    pub sm_count: i32,
    /// Total VRAM in bytes.
    pub total_mem: usize,
    /// Driver version as reported by `cuDriverGetVersion` (e.g. 12040).
    pub driver_version: i32,
}

/// A live CUDA device + primary context. One per engine instance.
///
/// The primary context is retained (refcounted by the driver) rather than
/// created, so multiple engines or tests share one context per device.
pub struct Cuda {
    api: Arc<DriverApi>,
    device: CUdevice,
    ctx: CUcontext,
    /// Facts about the device this handle is bound to.
    pub info: DeviceInfo,
}

// SAFETY: the CUDA driver API is thread-safe; the context handle is a
// process-wide primary context, valid from any thread once retained.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

impl Cuda {
    /// Detect device 0 and bind its primary context to the calling thread.
    ///
    /// Errors (never panics) when no driver, no device, or the device is
    /// below sm_70 — M2's floor, because the kernels rely on Volta warp
    /// semantics (`__shfl_*_sync`, independent thread scheduling).
    pub fn probe() -> Result<Cuda, GlError> {
        let api = api()
            .ok_or_else(|| {
                GlError::Engine("CUDA driver library not found (nvcuda.dll / libcuda.so)".into())
            })?
            .clone();

        // SAFETY: every call below follows the driver API contract; all out
        // pointers are valid locals.
        unsafe {
            check(&api, (api.cu_init)(0), "cuInit")?;
            let mut count = 0i32;
            check(&api, (api.cu_device_get_count)(&mut count), "cuDeviceGetCount")?;
            if count == 0 {
                return Err(GlError::Engine("no CUDA device present".into()));
            }
            let mut device: CUdevice = 0;
            check(&api, (api.cu_device_get)(&mut device, 0), "cuDeviceGet")?;

            let mut name_buf = [0u8; 128];
            check(
                &api,
                (api.cu_device_get_name)(name_buf.as_mut_ptr(), name_buf.len() as i32, device),
                "cuDeviceGetName",
            )?;
            let name_len = name_buf.iter().position(|&b| b == 0).unwrap_or(name_buf.len());
            let name = String::from_utf8_lossy(&name_buf[..name_len]).into_owned();

            let attr = |sel: i32, what: &str| -> Result<i32, GlError> {
                let mut v = 0i32;
                check(&api, (api.cu_device_get_attribute)(&mut v, sel, device), what)?;
                Ok(v)
            };
            let sm_major = attr(ATTR_COMPUTE_CAPABILITY_MAJOR, "cuDeviceGetAttribute(cc major)")?;
            let sm_minor = attr(ATTR_COMPUTE_CAPABILITY_MINOR, "cuDeviceGetAttribute(cc minor)")?;
            let sm_count = attr(ATTR_MULTIPROCESSOR_COUNT, "cuDeviceGetAttribute(sm count)")?;

            let mut total_mem = 0usize;
            check(&api, (api.cu_device_total_mem)(&mut total_mem, device), "cuDeviceTotalMem")?;
            let mut driver_version = 0i32;
            check(&api, (api.cu_driver_get_version)(&mut driver_version), "cuDriverGetVersion")?;

            if sm_major < 7 {
                return Err(GlError::Engine(format!(
                    "{name} is sm_{sm_major}{sm_minor}; glcuda M2 requires sm_70+ \
                     (Volta or later)"
                )));
            }

            let mut ctx: CUcontext = std::ptr::null_mut();
            check(
                &api,
                (api.cu_device_primary_ctx_retain)(&mut ctx, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            if let Err(e) = check(&api, (api.cu_ctx_set_current)(ctx), "cuCtxSetCurrent") {
                let _ = (api.cu_device_primary_ctx_release)(device);
                return Err(e);
            }

            Ok(Cuda {
                api,
                device,
                ctx,
                info: DeviceInfo { name, sm_major, sm_minor, sm_count, total_mem, driver_version },
            })
        }
    }

    /// Bind this handle's context to the calling thread. Needed when a
    /// `Cuda` created on one thread is used from another.
    pub fn make_current(&self) -> Result<(), GlError> {
        // SAFETY: ctx is a live retained primary context.
        unsafe { check(&self.api, (self.api.cu_ctx_set_current)(self.ctx), "cuCtxSetCurrent") }
    }

    /// Allocate raw VRAM. Cold path only — the hot path never allocates
    /// (ArchGLML_X2 §8 Principle 3); [`crate::buffer::BackendBuffer`] calls
    /// this exactly once per engine init.
    pub fn mem_alloc(&self, bytes: usize) -> Result<CUdeviceptr, GlError> {
        let mut dptr: CUdeviceptr = 0;
        // SAFETY: out pointer valid; nonzero size enforced by caller logic
        // (cuMemAlloc rejects 0 with an error we surface).
        unsafe { check(&self.api, (self.api.cu_mem_alloc)(&mut dptr, bytes), "cuMemAlloc")? };
        Ok(dptr)
    }

    /// Free VRAM allocated with [`Cuda::mem_alloc`].
    pub fn mem_free(&self, dptr: CUdeviceptr) -> Result<(), GlError> {
        // SAFETY: caller guarantees dptr came from mem_alloc and is unused.
        unsafe { check(&self.api, (self.api.cu_mem_free)(dptr), "cuMemFree") }
    }

    /// (free, total) VRAM in bytes — the leak-check primitive from the M2
    /// definition of done.
    pub fn mem_get_info(&self) -> Result<(usize, usize), GlError> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: out pointers are valid locals.
        unsafe {
            check(&self.api, (self.api.cu_mem_get_info)(&mut free, &mut total), "cuMemGetInfo")?
        };
        Ok((free, total))
    }

    /// Copy host → device.
    pub fn htod(&self, dst: CUdeviceptr, src: &[u8]) -> Result<(), GlError> {
        // SAFETY: src range is valid for src.len() bytes; dst sized by caller.
        unsafe {
            check(
                &self.api,
                (self.api.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len()),
                "cuMemcpyHtoD",
            )
        }
    }

    /// Copy host f32 slice → device.
    pub fn htod_f32(&self, dst: CUdeviceptr, src: &[f32]) -> Result<(), GlError> {
        // SAFETY: f32 slice reinterpreted as bytes — always valid.
        let bytes = unsafe {
            std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src))
        };
        self.htod(dst, bytes)
    }

    /// Copy device → device (stream-0 ordered) — the KV-cache write path.
    pub fn dtod(&self, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Result<(), GlError> {
        // SAFETY: caller guarantees both regions are live and sized.
        unsafe { check(&self.api, (self.api.cu_memcpy_dtod)(dst, src, bytes), "cuMemcpyDtoD") }
    }

    /// Copy device → host f32 slice.
    pub fn dtoh_f32(&self, dst: &mut [f32], src: CUdeviceptr) -> Result<(), GlError> {
        // SAFETY: dst range is valid for the full byte length; src sized by
        // caller.
        unsafe {
            check(
                &self.api,
                (self.api.cu_memcpy_dtoh)(
                    dst.as_mut_ptr().cast(),
                    src,
                    std::mem::size_of_val(dst),
                ),
                "cuMemcpyDtoH",
            )
        }
    }

    /// Block until all queued work on this context has finished.
    pub fn synchronize(&self) -> Result<(), GlError> {
        // SAFETY: no preconditions beyond a current context.
        unsafe { check(&self.api, (self.api.cu_ctx_synchronize)(), "cuCtxSynchronize") }
    }

    /// JIT-load a PTX image. The driver compiles it for the actual device
    /// architecture (ADR-004: ahead-of-time PTX, no runtime codegen of ours).
    ///
    /// Uses `cuModuleLoadDataEx` with a JIT error-log buffer so a rejected
    /// image reports the assembler's own diagnostic (line + reason), not a
    /// bare `CUDA_ERROR_INVALID_PTX`.
    pub fn load_module(&self, ptx: &str) -> Result<Module, GlError> {
        use crate::ffi::{JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES};

        // cuModuleLoadData* requires a NUL-terminated image for PTX text.
        let image = std::ffi::CString::new(ptx)
            .map_err(|_| GlError::Engine("PTX image contains interior NUL".into()))?;

        let mut err_log = vec![0u8; 16 * 1024];
        let mut err_size: usize = err_log.len();
        let mut options = [JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        // The size option's value is passed by value in the pointer slot
        // (CUDA's documented convention for scalar JIT options).
        let mut values: [*mut std::ffi::c_void; 2] =
            [err_log.as_mut_ptr().cast(), err_size as *mut std::ffi::c_void];

        let mut raw: CUmodule = std::ptr::null_mut();
        // SAFETY: image and buffers outlive the call; out pointer is valid;
        // option/value arrays are length 2 as declared to numOptions.
        let res = unsafe {
            (self.api.cu_module_load_data_ex)(
                &mut raw,
                image.as_ptr().cast(),
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if res != crate::ffi::CUDA_SUCCESS {
            // The driver wrote the used length back into the value slot.
            err_size = values[1] as usize;
            let log = String::from_utf8_lossy(&err_log[..err_size.min(err_log.len())]);
            let log = log.trim_end_matches('\0').trim();
            return Err(GlError::Engine(if log.is_empty() {
                "cuModuleLoadDataEx(PTX JIT) failed with no log".into()
            } else {
                format!("cuModuleLoadDataEx(PTX JIT) failed:\n{log}")
            }));
        }
        Ok(Module { api: self.api.clone(), raw })
    }

    /// Launch `f` with the given geometry. `params` holds one pointer per
    /// kernel parameter, in declaration order, each pointing at a live
    /// host value (the driver copies them synchronously at launch).
    pub fn launch(
        &self,
        f: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), GlError> {
        // SAFETY: f belongs to a live module on this context; params
        // pointers are valid for the duration of the (synchronous) call.
        unsafe {
            check(
                &self.api,
                (self.api.cu_launch_kernel)(
                    f.0,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_bytes,
                    std::ptr::null_mut(), // default stream
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        // SAFETY: releasing a context we retained; errors on teardown are
        // unreportable, so they are intentionally ignored.
        unsafe {
            let _ = (self.api.cu_device_primary_ctx_release)(self.device);
        }
    }
}

/// A resolved kernel handle, owned by its module — valid only while that
/// `Module` is alive. Plain data; copying does not duplicate GPU state.
#[derive(Clone, Copy)]
pub struct Kernel(CUfunction);

// SAFETY: function handles are context-level objects; the driver API is
// thread-safe.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

/// A loaded PTX module. Kernel handles resolved from it stay valid for the
/// module's lifetime — holders must keep the `Module` alive.
pub struct Module {
    api: Arc<DriverApi>,
    raw: CUmodule,
}

// SAFETY: module handles are context-level objects; the driver API is
// thread-safe.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    /// Resolve a kernel by its `.entry` name.
    pub fn get_function(&self, name: &str) -> Result<Kernel, GlError> {
        let cname = std::ffi::CString::new(name)
            .map_err(|_| GlError::Engine("kernel name contains NUL".into()))?;
        let mut f: CUfunction = std::ptr::null_mut();
        // SAFETY: raw is a live module; name is NUL-terminated.
        unsafe {
            check(
                &self.api,
                (self.api.cu_module_get_function)(&mut f, self.raw, cname.as_ptr().cast()),
                "cuModuleGetFunction",
            )?
        };
        Ok(Kernel(f))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: raw is live and owned by us; teardown errors ignored.
        unsafe {
            let _ = (self.api.cu_module_unload)(self.raw);
        }
    }
}
