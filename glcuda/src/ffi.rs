//! Raw CUDA Driver API bindings, loaded dynamically at runtime.
//!
//! ArchGLML_X2 §18: the FFI boundary is intentionally narrow — the driver
//! API only. The driver library (`nvcuda.dll` / `libcuda.so.1`) is loaded
//! with `LoadLibrary`/`dlopen` rather than linked, so glcuda compiles and
//! links on every machine; machines without an NVIDIA driver simply report
//! the engine as unavailable and the runtime falls back down the chain.

use std::ffi::c_void;

use glcore::GlError;

/// CUDA driver status code. 0 (`CUDA_SUCCESS`) means success.
pub type CUresult = i32;
/// Device ordinal handle.
pub type CUdevice = i32;
/// Opaque context handle.
pub type CUcontext = *mut c_void;
/// Opaque module handle (one loaded PTX image).
pub type CUmodule = *mut c_void;
/// Opaque kernel handle, owned by its module.
pub type CUfunction = *mut c_void;
/// Opaque stream handle. Null = the default stream.
pub type CUstream = *mut c_void;
/// Device memory address. Always 64-bit, even on 32-bit hosts.
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;

// cuDeviceGetAttribute selectors (CUdevice_attribute).
pub const ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Owned handle to the dynamically loaded driver library. Never unloaded —
/// the driver stays resident for the life of the process, matching the
/// lifetime of the function pointers handed out below. The handle is held
/// only to document that ownership; nothing reads it back.
struct Lib(#[allow(dead_code)] *mut c_void);

// SAFETY: the handle is only used to resolve symbols at load time and is
// otherwise inert; the OS loader allows cross-thread use of module handles.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    pub fn open() -> *mut c_void {
        // SAFETY: NUL-terminated literal; LoadLibraryA has no other
        // preconditions. A null return means "driver not installed".
        unsafe { LoadLibraryA(c"nvcuda.dll".as_ptr().cast()) }
    }

    pub fn sym(lib: *mut c_void, name: &[u8]) -> *mut c_void {
        debug_assert_eq!(name.last(), Some(&0));
        // SAFETY: lib is a live module handle; name is NUL-terminated.
        unsafe { GetProcAddress(lib, name.as_ptr()) }
    }
}

#[cfg(unix)]
mod sys {
    use std::ffi::c_void;

    pub fn open() -> *mut c_void {
        // SAFETY: NUL-terminated literals; dlopen tolerates missing files.
        unsafe {
            let h = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW);
            if !h.is_null() {
                return h;
            }
            libc::dlopen(c"libcuda.so".as_ptr(), libc::RTLD_NOW)
        }
    }

    pub fn sym(lib: *mut c_void, name: &[u8]) -> *mut c_void {
        debug_assert_eq!(name.last(), Some(&0));
        // SAFETY: lib is a live dlopen handle; name is NUL-terminated.
        unsafe { libc::dlsym(lib, name.as_ptr().cast()) }
    }
}

/// Resolved CUDA Driver API entry points. One instance per process,
/// created by [`DriverApi::load`].
///
/// All functions use the `extern "system"` ABI (`__stdcall` on 32-bit
/// Windows, the platform C ABI everywhere else) matching `CUDAAPI`.
#[allow(clippy::type_complexity)]
pub struct DriverApi {
    _lib: Lib,
    pub cu_init: unsafe extern "system" fn(u32) -> CUresult,
    pub cu_driver_get_version: unsafe extern "system" fn(*mut i32) -> CUresult,
    pub cu_device_get_count: unsafe extern "system" fn(*mut i32) -> CUresult,
    pub cu_device_get: unsafe extern "system" fn(*mut CUdevice, i32) -> CUresult,
    pub cu_device_get_name: unsafe extern "system" fn(*mut u8, i32, CUdevice) -> CUresult,
    pub cu_device_get_attribute: unsafe extern "system" fn(*mut i32, i32, CUdevice) -> CUresult,
    pub cu_device_total_mem: unsafe extern "system" fn(*mut usize, CUdevice) -> CUresult,
    pub cu_device_primary_ctx_retain:
        unsafe extern "system" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cu_device_primary_ctx_release: unsafe extern "system" fn(CUdevice) -> CUresult,
    pub cu_ctx_set_current: unsafe extern "system" fn(CUcontext) -> CUresult,
    pub cu_ctx_synchronize: unsafe extern "system" fn() -> CUresult,
    pub cu_module_load_data: unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cu_module_unload: unsafe extern "system" fn(CUmodule) -> CUresult,
    pub cu_module_get_function:
        unsafe extern "system" fn(*mut CUfunction, CUmodule, *const u8) -> CUresult,
    pub cu_mem_alloc: unsafe extern "system" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cu_mem_free: unsafe extern "system" fn(CUdeviceptr) -> CUresult,
    pub cu_memcpy_htod: unsafe extern "system" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    pub cu_memcpy_dtoh: unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cu_memcpy_dtod: unsafe extern "system" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    pub cu_mem_get_info: unsafe extern "system" fn(*mut usize, *mut usize) -> CUresult,
    #[allow(clippy::too_many_arguments)]
    pub cu_launch_kernel: unsafe extern "system" fn(
        CUfunction,
        u32, // gridDimX
        u32,
        u32,
        u32, // blockDimX
        u32,
        u32,
        u32,      // sharedMemBytes
        CUstream, // hStream
        *mut *mut c_void, // kernelParams
        *mut *mut c_void, // extra
    ) -> CUresult,
    pub cu_get_error_name: unsafe extern "system" fn(CUresult, *mut *const i8) -> CUresult,
}

/// Resolve one symbol into an arbitrary fn-pointer type.
fn sym<T>(lib: *mut c_void, name: &[u8]) -> Result<T, GlError> {
    let p = sys::sym(lib, name);
    if p.is_null() {
        return Err(GlError::Engine(format!(
            "CUDA driver is missing symbol {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        )));
    }
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: T is always a fn pointer of the matching driver signature;
    // same size and validity as the non-null raw pointer.
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

/// Resolve a versioned symbol (`name_v2`), falling back to the unversioned
/// name on very old drivers.
fn sym_v2<T>(lib: *mut c_void, v2: &[u8], v1: &[u8]) -> Result<T, GlError> {
    sym(lib, v2).or_else(|_| sym(lib, v1))
}

impl DriverApi {
    /// Load the CUDA driver library and resolve every entry point.
    /// Fails cleanly (no panic, no partial state) when the driver is not
    /// installed — that is the normal case on non-NVIDIA machines.
    pub fn load() -> Result<DriverApi, GlError> {
        let lib = sys::open();
        if lib.is_null() {
            return Err(GlError::Engine(
                "CUDA driver library not found (nvcuda.dll / libcuda.so)".into(),
            ));
        }
        Ok(DriverApi {
            cu_init: sym(lib, b"cuInit\0")?,
            cu_driver_get_version: sym(lib, b"cuDriverGetVersion\0")?,
            cu_device_get_count: sym(lib, b"cuDeviceGetCount\0")?,
            cu_device_get: sym(lib, b"cuDeviceGet\0")?,
            cu_device_get_name: sym(lib, b"cuDeviceGetName\0")?,
            cu_device_get_attribute: sym(lib, b"cuDeviceGetAttribute\0")?,
            cu_device_total_mem: sym_v2(lib, b"cuDeviceTotalMem_v2\0", b"cuDeviceTotalMem\0")?,
            cu_device_primary_ctx_retain: sym(lib, b"cuDevicePrimaryCtxRetain\0")?,
            cu_device_primary_ctx_release: sym_v2(
                lib,
                b"cuDevicePrimaryCtxRelease_v2\0",
                b"cuDevicePrimaryCtxRelease\0",
            )?,
            cu_ctx_set_current: sym(lib, b"cuCtxSetCurrent\0")?,
            cu_ctx_synchronize: sym(lib, b"cuCtxSynchronize\0")?,
            cu_module_load_data: sym(lib, b"cuModuleLoadData\0")?,
            cu_module_unload: sym(lib, b"cuModuleUnload\0")?,
            cu_module_get_function: sym(lib, b"cuModuleGetFunction\0")?,
            cu_mem_alloc: sym_v2(lib, b"cuMemAlloc_v2\0", b"cuMemAlloc\0")?,
            cu_mem_free: sym_v2(lib, b"cuMemFree_v2\0", b"cuMemFree\0")?,
            cu_memcpy_htod: sym_v2(lib, b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0")?,
            cu_memcpy_dtoh: sym_v2(lib, b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0")?,
            cu_memcpy_dtod: sym_v2(lib, b"cuMemcpyDtoD_v2\0", b"cuMemcpyDtoD\0")?,
            cu_mem_get_info: sym_v2(lib, b"cuMemGetInfo_v2\0", b"cuMemGetInfo\0")?,
            cu_launch_kernel: sym(lib, b"cuLaunchKernel\0")?,
            cu_get_error_name: sym(lib, b"cuGetErrorName\0")?,
            _lib: Lib(lib),
        })
    }
}
