//! Safe wrapper over the raw CUDA Driver API.
//!
//! Owns exactly the responsibilities ArchGLML_X2 §16 assigns the host side:
//! device detection, context lifetime, PTX module loading, memory transfer
//! and kernel launch. Everything numeric happens in the PTX kernels.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use glcore::GlError;

use crate::ffi::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUgraph, CUgraphExec, CUmodule, CUresult,
    CUstream, DriverApi, ATTR_COMPUTE_CAPABILITY_MAJOR, ATTR_COMPUTE_CAPABILITY_MINOR,
    ATTR_MULTIPROCESSOR_COUNT, CUDA_SUCCESS,
};

/// The process-wide driver API table, loaded on first use. `None` when the
/// machine has no CUDA driver — cached so repeated probes cost one atomic
/// load, not a filesystem search.
fn api() -> Option<&'static Arc<DriverApi>> {
    static API: OnceLock<Option<Arc<DriverApi>>> = OnceLock::new();
    API.get_or_init(|| DriverApi::load().ok().map(Arc::new)).as_ref()
}

/// A fixed pool of reusable CUDA events used as stage boundaries.
///
/// The point is what it does NOT do: `record` only enqueues a timestamp in
/// stream order, so marking a stage boundary costs no host synchronization.
/// Marks accumulate through a whole chunk and are read once, at the end, with
/// a single sync — where the alternative (wrapping each stage in
/// `cuCtxSynchronize`) drains the pipeline at every boundary and reports
/// timings for an execution schedule that never runs in production.
pub struct EventRing {
    api: Arc<DriverApi>,
    events: Vec<CUevent>,
}

impl EventRing {
    /// How many boundaries this ring can mark.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Never true for a ring built by [`Cuda::event_ring`]; present because
    /// clippy asks for it alongside `len`.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Enqueue mark `i` on the current launch stream. Out-of-range indices are
    /// ignored: a ring too small to hold a run's marks degrades to partial
    /// timing rather than aborting the inference that carries it.
    pub fn record(&self, cuda: &Cuda, i: usize) {
        let (Some(rec), Some(&e)) = (self.api.cu_event_record, self.events.get(i)) else {
            return;
        };
        let stream = cuda.launch_stream.load(std::sync::atomic::Ordering::Relaxed);
        // SAFETY: e is a live event from this ring; stream is NULL or a live
        // stream owned for the length of the call.
        unsafe {
            let _ = rec(e, stream);
        }
    }

    /// Milliseconds between marks `a` and `b`, after waiting for `b`.
    ///
    /// Returns `None` when either index is unmarked or the driver refuses —
    /// an unmeasured stage must read as absent, never as zero.
    pub fn elapsed_ms(&self, a: usize, b: usize) -> Option<f64> {
        let (sync, elapsed) = (self.api.cu_event_synchronize?, self.api.cu_event_elapsed_time?);
        let (&ea, &eb) = (self.events.get(a)?, self.events.get(b)?);
        let mut ms = 0f32;
        // SAFETY: both events belong to this ring and were created above.
        unsafe {
            if sync(eb) != CUDA_SUCCESS || elapsed(&mut ms, ea, eb) != CUDA_SUCCESS {
                return None;
            }
        }
        Some(ms as f64)
    }
}

impl Drop for EventRing {
    fn drop(&mut self) {
        if let Some(destroy) = self.api.cu_event_destroy {
            for e in self.events.drain(..) {
                // SAFETY: each event was created by this ring; teardown
                // errors are unreportable.
                unsafe {
                    let _ = destroy(e);
                }
            }
        }
    }
}

// SAFETY: events are context-level objects and the driver API is thread-safe.
unsafe impl Send for EventRing {}
unsafe impl Sync for EventRing {}

/// Returned when graph capture or replay is asked of a driver that does not
/// export the CUDA Graph API. Not a failure state: the caller is expected to
/// fall back to issuing kernels individually.
const GRAPHS_UNSUPPORTED: &str =
    "CUDA Graph API not available on this driver (needs CUDA 10+); \
     the caller should issue kernels individually instead";

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
    /// The stream every kernel launch targets. NULL = the default stream
    /// (normal execution). During graph capture it is pointed at the
    /// capture stream so the unchanged `KernelSet` wrappers record into the
    /// graph instead of executing. `AtomicPtr` so `Cuda` stays `Sync`;
    /// only ever flipped between launches by the single owning thread.
    launch_stream: std::sync::atomic::AtomicPtr<c_void>,
    /// Streams for issuing independent prefill sub-slabs concurrently.
    ///
    /// `None` unless `GLCUDA_MULTI_STREAM_PREFILL` is set, and never created
    /// otherwise — the default execution model stays one stream, as
    /// documented in the crate root. See [`Cuda::prefill_streams`].
    prefill_streams: std::sync::OnceLock<Option<StreamPool>>,
    /// Facts about the device this handle is bound to.
    pub info: DeviceInfo,
}

/// How many streams the prefill pool holds when enabled without an explicit
/// count. A 220-token prompt splits into four 64-row sub-slabs, so four
/// covers the shape that motivated this.
const DEFAULT_PREFILL_STREAMS: usize = 4;

/// A set of non-blocking streams for issuing independent work concurrently.
///
/// Owns its streams and destroys them on drop. Non-blocking is required: a
/// stream created without that flag implicitly synchronizes with the default
/// stream, which would serialize exactly the work this exists to overlap.
pub struct StreamPool {
    api: Arc<DriverApi>,
    streams: Vec<CUstream>,
}

impl StreamPool {
    /// Number of streams in the pool. Always >= 1 when the pool exists.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Never true for a pool that was successfully created; present because
    /// clippy asks for it alongside `len`.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

impl Drop for StreamPool {
    fn drop(&mut self) {
        for s in self.streams.drain(..) {
            // SAFETY: each stream was created by this pool and is no longer
            // referenced. Teardown errors are unreportable.
            unsafe {
                let _ = (self.api.cu_stream_destroy)(s);
            }
        }
    }
}

// SAFETY: CUDA streams are context-level objects and the driver API is
// thread-safe; the pool hands out no interior mutability of its own.
unsafe impl Send for StreamPool {}
unsafe impl Sync for StreamPool {}

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
                launch_stream: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
                prefill_streams: std::sync::OnceLock::new(),
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
        use crate::ffi::{
            JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES, JIT_INFO_LOG_BUFFER,
            JIT_INFO_LOG_BUFFER_SIZE_BYTES, JIT_LOG_VERBOSE,
        };

        // cuModuleLoadData* requires a NUL-terminated image for PTX text.
        let image = std::ffi::CString::new(ptx)
            .map_err(|_| GlError::Engine("PTX image contains interior NUL".into()))?;

        let mut err_log = vec![0u8; 16 * 1024];
        let err_cap = err_log.len();
        // Info log + CU_JIT_LOG_VERBOSE: with GLCUDA_JIT_VERBOSE=1 the driver
        // writes per-function register/shared-mem usage here (the ptxas -v
        // numbers, unavailable to us otherwise since we JIT at runtime). Off
        // by default — pure diagnostic, no effect on the compiled module.
        let want_info = std::env::var_os("GLCUDA_JIT_VERBOSE").is_some();
        let mut info_log = vec![0u8; 16 * 1024];
        let info_cap = info_log.len();

        // The size option's value is passed by value in the pointer slot
        // (CUDA's documented convention for scalar JIT options).
        let (mut options, mut values): (Vec<i32>, Vec<*mut std::ffi::c_void>) = (
            vec![JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES],
            vec![err_log.as_mut_ptr().cast(), err_cap as *mut std::ffi::c_void],
        );
        if want_info {
            options.extend_from_slice(&[
                JIT_INFO_LOG_BUFFER,
                JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                JIT_LOG_VERBOSE,
            ]);
            values.extend_from_slice(&[
                info_log.as_mut_ptr().cast(),
                info_cap as *mut std::ffi::c_void,
                1_usize as *mut std::ffi::c_void, // verbose = true
            ]);
        }

        let mut raw: CUmodule = std::ptr::null_mut();
        // SAFETY: image and buffers outlive the call; out pointer is valid;
        // options/values are the same length, matching numOptions.
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
            let err_size = values[1] as usize;
            let log = String::from_utf8_lossy(&err_log[..err_size.min(err_cap)]);
            let log = log.trim_end_matches('\0').trim();
            return Err(GlError::Engine(if log.is_empty() {
                "cuModuleLoadDataEx(PTX JIT) failed with no log".into()
            } else {
                format!("cuModuleLoadDataEx(PTX JIT) failed:\n{log}")
            }));
        }
        if want_info {
            // values[3] slot (INFO_LOG_BUFFER_SIZE_BYTES) holds the used length.
            let info_size = values[3] as usize;
            let log = String::from_utf8_lossy(&info_log[..info_size.min(info_cap)]);
            let log = log.trim_end_matches('\0').trim();
            if !log.is_empty() {
                eprintln!("[glcuda jit] register/smem usage (ptxas -v):\n{log}");
            }
        }
        Ok(Module { api: self.api.clone(), raw })
    }

    /// Launch `f` with the given geometry onto the current launch stream
    /// ([`Cuda::launch_stream`] — NULL for normal execution, or the capture
    /// stream while a graph is being recorded). `params` holds one pointer
    /// per kernel parameter, in declaration order, each pointing at a live
    /// host value (the driver reads them at launch/record time).
    pub fn launch(
        &self,
        f: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), GlError> {
        let stream = self.launch_stream.load(std::sync::atomic::Ordering::Relaxed);
        // SAFETY: f belongs to a live module on this context; params
        // pointers are valid for the duration of the call; stream is NULL
        // or a live stream owned for the length of a capture.
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
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    /// True when this driver exports the CUDA event entry points, i.e.
    /// [`EventRing`] can time stages on-stream.
    pub fn events_available(&self) -> bool {
        self.api.events_available()
    }

    /// Allocate `n` reusable events for on-stream stage timing.
    ///
    /// `None` when the driver does not export the event API — the caller
    /// falls back to coarse timing rather than failing.
    pub fn event_ring(&self, n: usize) -> Option<EventRing> {
        if !self.api.events_available() {
            return None;
        }
        let create = self.api.cu_event_create?;
        let mut events = Vec::with_capacity(n);
        for _ in 0..n {
            let mut e: CUevent = std::ptr::null_mut();
            // SAFETY: out pointer valid; flag 0 = CU_EVENT_DEFAULT, which
            // enables timing (CU_EVENT_DISABLE_TIMING would not).
            if unsafe { create(&mut e, 0) } != CUDA_SUCCESS {
                drop(EventRing { api: self.api.clone(), events });
                return None;
            }
            events.push(e);
        }
        Some(EventRing { api: self.api.clone(), events })
    }

    /// True when this driver exports the CUDA Graph entry points, i.e.
    /// [`Cuda::capture`] can succeed.
    ///
    /// Callers on the decode path should branch on this rather than treating
    /// a capture failure as fatal: replaying a captured graph is an
    /// optimization, and issuing the same kernels individually is the
    /// supported degraded mode (see `GpuModel::decode_step`).
    pub fn graphs_available(&self) -> bool {
        self.api.graphs_available()
    }

    /// The optional `cuGraphLaunch` entry point, if the driver exports it.
    fn cu_graph_launch_fn(
        &self,
    ) -> Option<unsafe extern "system" fn(CUgraphExec, CUstream) -> CUresult> {
        self.api.cu_graph_launch
    }

    /// Capture everything launched by `body` into a replayable [`GraphExec`].
    ///
    /// Creates a fresh non-blocking stream, points all launches at it for
    /// the duration of `body` (so the unchanged `KernelSet` wrappers record
    /// instead of execute), ends capture, and instantiates the graph. The
    /// launch stream is restored to NULL afterward even on error.
    ///
    /// `body` must issue only stream-ordered work (kernel launches, async
    /// copies) — a synchronizing call inside capture aborts it.
    pub fn capture<F>(&self, body: F) -> Result<GraphExec, GlError>
    where
        F: FnOnce() -> Result<(), GlError>,
    {
        use crate::ffi::{CU_STREAM_CAPTURE_MODE_GLOBAL, CU_STREAM_NON_BLOCKING};
        use std::sync::atomic::Ordering;

        // Resolve the optional graph entry points BEFORE touching a stream.
        // Bailing here leaves nothing to unwind; discovering a missing symbol
        // after `cuStreamBeginCapture` would strand the stream in capture mode.
        let (begin, end, instantiate, destroy) = match (
            self.api.cu_stream_begin_capture,
            self.api.cu_stream_end_capture,
            self.api.cu_graph_instantiate,
            self.api.cu_graph_destroy,
        ) {
            (Some(b), Some(e), Some(i), Some(d)) if self.api.graphs_available() => (b, e, i, d),
            _ => return Err(GlError::Engine(GRAPHS_UNSUPPORTED.into())),
        };

        // Dedicated capturable stream.
        let mut stream: CUstream = std::ptr::null_mut();
        // SAFETY: out pointer valid.
        unsafe {
            check(
                &self.api,
                (self.api.cu_stream_create)(&mut stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate",
            )?
        };

        // Point launches at the capture stream; guarantee restore + destroy.
        self.launch_stream.store(stream, Ordering::Relaxed);
        let result = (|| -> Result<GraphExec, GlError> {
            // SAFETY: stream is live and idle.
            unsafe {
                check(
                    &self.api,
                    begin(stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
                    "cuStreamBeginCapture",
                )?
            };
            body()?;
            let mut graph: CUgraph = std::ptr::null_mut();
            // SAFETY: ends the capture opened above; out pointer valid.
            unsafe {
                check(
                    &self.api,
                    end(stream, &mut graph),
                    "cuStreamEndCapture",
                )?
            };
            let mut exec: CUgraphExec = std::ptr::null_mut();
            // SAFETY: graph is a valid captured graph; flags 0.
            let inst = unsafe { instantiate(&mut exec, graph, 0) };
            // The graph template is no longer needed once instantiated.
            // SAFETY: graph is live and owned here.
            unsafe {
                let _ = destroy(graph);
            }
            check(&self.api, inst, "cuGraphInstantiate")?;
            Ok(GraphExec { api: self.api.clone(), exec })
        })();

        // Restore normal (default-stream) execution and free the capture
        // stream regardless of outcome.
        self.launch_stream.store(std::ptr::null_mut(), Ordering::Relaxed);
        // SAFETY: stream is live and no longer referenced.
        unsafe {
            let _ = (self.api.cu_stream_destroy)(stream);
        }
        result
    }

    /// The prefill stream pool, or `None` when multi-stream prefill is off.
    ///
    /// # Why this exists
    ///
    /// The MMA GEMM's grid is `ceil_div(out_dim, 64)` — nothing splits K or
    /// tokens — so `out_dim` alone decides how much of the device is used. On
    /// a 40-SM T4 running Qwen2.5-0.5B that is 76 blocks for `gate`/`up` and
    /// **14 for `down`**, which a measured prefill profile put at 66% of
    /// prefill time while `gate`+`up`, moving twice the weight bytes, took
    /// 10%.
    ///
    /// A prompt longer than 64 tokens is already issued as several
    /// independent sub-slabs: same weights, different activation rows,
    /// disjoint output. They are only sequential because they share one
    /// stream. Issuing them on separate streams puts 4x the blocks in flight
    /// without touching the kernel — and, as a side effect, the concurrent
    /// sub-slabs read the same weights, which L2 may serve once instead of
    /// four times.
    ///
    /// # Why it is opt-in
    ///
    /// The crate root documents one stream and one sync per token. This is an
    /// experiment against that invariant, so it is off unless
    /// `GLCUDA_MULTI_STREAM_PREFILL` is set, and a failure to create the pool
    /// degrades to the single-stream path rather than failing the run. Set the
    /// variable to a number to choose the pool size.
    pub fn prefill_streams(&self) -> Option<&StreamPool> {
        self.prefill_streams
            .get_or_init(|| {
                let raw = std::env::var_os("GLCUDA_MULTI_STREAM_PREFILL")?;
                let n = raw
                    .to_str()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .filter(|n| *n >= 2)
                    .unwrap_or(DEFAULT_PREFILL_STREAMS);
                match self.make_stream_pool(n) {
                    Ok(pool) => {
                        eprintln!("[glcuda] multi-stream prefill: {n} streams");
                        Some(pool)
                    }
                    Err(e) => {
                        eprintln!(
                            "[glcuda] multi-stream prefill unavailable ({e}); \
                             staying on the single-stream path"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Create `n` non-blocking streams.
    fn make_stream_pool(&self, n: usize) -> Result<StreamPool, GlError> {
        use crate::ffi::CU_STREAM_NON_BLOCKING;
        let mut streams = Vec::with_capacity(n);
        for _ in 0..n {
            let mut s: CUstream = std::ptr::null_mut();
            // SAFETY: out pointer valid; flag is the documented non-blocking
            // constant.
            unsafe {
                check(
                    &self.api,
                    (self.api.cu_stream_create)(&mut s, CU_STREAM_NON_BLOCKING),
                    "cuStreamCreate",
                )
            }
            // Streams already created are freed by the partial pool's Drop.
            .map_err(|e| {
                drop(StreamPool { api: self.api.clone(), streams: std::mem::take(&mut streams) });
                e
            })?;
            streams.push(s);
        }
        Ok(StreamPool { api: self.api.clone(), streams })
    }

    /// Point launches at `pool`'s stream `i` for the duration of `body`.
    ///
    /// The previous launch target is saved and restored rather than reset to
    /// NULL: forcing the default stream here would silently break an enclosing
    /// graph capture. Prefill is not captured today, and this keeps that from
    /// becoming a trap if it ever is.
    pub fn on_stream<F>(&self, pool: &StreamPool, i: usize, body: F) -> Result<(), GlError>
    where
        F: FnOnce() -> Result<(), GlError>,
    {
        use std::sync::atomic::Ordering;
        let previous = self.launch_stream.load(Ordering::Relaxed);
        self.launch_stream.store(pool.streams[i % pool.streams.len()], Ordering::Relaxed);
        let result = body();
        self.launch_stream.store(previous, Ordering::Relaxed);
        result
    }

    /// Wait for every stream in `pool`.
    ///
    /// Blunt on purpose: without an event API this is what makes work issued
    /// across the pool visible to the next kernel on the default stream. It
    /// costs a host round-trip per call, which is the price of measuring the
    /// idea before building the machinery to do it properly.
    pub fn sync_pool(&self, pool: &StreamPool) -> Result<(), GlError> {
        for s in &pool.streams {
            // SAFETY: streams are live and owned by the pool.
            unsafe {
                check(&self.api, (self.api.cu_stream_synchronize)(*s), "cuStreamSynchronize")?
            };
        }
        Ok(())
    }

    /// Replay a captured graph on the default stream and wait for it.
    pub fn graph_launch(&self, exec: &GraphExec) -> Result<(), GlError> {
        // A live GraphExec can only exist if capture succeeded, so this is
        // always Some in practice; the check keeps the unwrap out of the code.
        let launch = self
            .cu_graph_launch_fn()
            .ok_or_else(|| GlError::Engine(GRAPHS_UNSUPPORTED.into()))?;
        // SAFETY: exec is a live instantiated graph; NULL = default stream.
        unsafe { check(&self.api, launch(exec.exec, std::ptr::null_mut()), "cuGraphLaunch")? };
        self.synchronize()
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        // ⛔ Streams first, context second, and the order is not stylistic.
        //
        // Rust drops a struct's fields AFTER its own `Drop::drop` body runs,
        // so leaving `prefill_streams` to the implicit path would call
        // cuStreamDestroy against a context this function had already
        // released — SIGSEGV at exit, which is exactly what the drop-order
        // note in the crate root warns about for `GlcudaEngine`'s fields.
        // That note protects resources held *beside* `Cuda`; it cannot
        // protect one held *inside* it, which is how this got in.
        //
        // Observed as rc=-11 on every run that created a pool: the archive
        // was already written, so the measurements survived and only the
        // process died. A crash that arrives after the useful work is the
        // easiest kind to dismiss and still means undefined behaviour ran.
        drop(self.prefill_streams.take());
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

/// An instantiated, replayable CUDA graph — the captured per-token kernel
/// sequence collapsed into a single launchable unit (M2.2). Replay it with
/// [`Cuda::graph_launch`]; one launch submits the whole DAG, removing the
/// per-kernel host round-trips that leave the GPU idle between ops.
pub struct GraphExec {
    api: Arc<DriverApi>,
    exec: CUgraphExec,
}

// SAFETY: graph-exec handles are context-level objects; the driver API is
// thread-safe.
unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // SAFETY: exec is live and owned by us; teardown errors ignored.
        // The symbol must exist — this object could not have been built
        // without it — but a missing one leaks rather than panics in drop.
        if let Some(destroy) = self.api.cu_graph_exec_destroy {
            unsafe {
                let _ = destroy(self.exec);
            }
        }
    }
}
