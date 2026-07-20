//! Core runtime types: configuration, execution state, statistics, buffers.
//!
//! ARTX05 Â§"Execution Model". These types carry no behaviour beyond
//! construction and defaults â€” the orchestration lives in
//! [`GllmRuntime`](crate::runtime::GllmRuntime).

use crate::manifest::DevicePlacement;

/// Verbosity levels for [`RuntimeLogger`](crate::runtime::logger::RuntimeLogger).
///
/// ARTX05 Â§"Logging and Diagnostics" defines exactly four levels. Ordering is
/// by severity so a minimum level can be compared directly: `Error` is the
/// most severe and sorts lowest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum RuntimeLogLevel {
    /// Fatal or recoverable errors.
    Error,
    /// Suboptimal configuration (CPU fallback, shrunken prefetch window).
    #[default]
    Warn,
    /// Load progress, per-layer timing, memory usage.
    Info,
    /// Tensor shapes, offsets, mapping addresses.
    Debug,
}

impl RuntimeLogLevel {
    /// Uppercase tag used in rendered log lines.
    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeLogLevel::Error => "ERROR",
            RuntimeLogLevel::Warn => "WARN",
            RuntimeLogLevel::Info => "INFO",
            RuntimeLogLevel::Debug => "DEBUG",
        }
    }
}

/// Runtime tuning knobs.
///
/// [`Default`] targets the reference 8 GB machine: a small prefetch window,
/// unmap after execution, and checksum verification left *off* (it re-reads
/// every byte of every layer â€” opt in when a package is suspect).
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Layers to map ahead of the executing one. `0` disables prefetch.
    pub prefetch_window: usize,
    /// Unmap a layer once it has executed. Keep `true` unless the whole model
    /// comfortably fits in RAM.
    pub unmap_after_exec: bool,
    /// Worker threads for the CPU path. `0` means "detect physical cores".
    ///
    /// Note: layer execution is delegated to a backend, which owns its own
    /// threading. This is a hint the backend may read, not a pool the runtime
    /// spawns.
    pub num_threads: usize,
    /// Minimum severity that will be recorded.
    pub log_level: RuntimeLogLevel,
    /// Re-verify each layer's SHA-256 as it is mapped. Expensive.
    pub verify_on_load: bool,
    /// Maximum sequence length the KV cache is sized for.
    pub max_seq_len: u32,
    /// Device used when the manifest gives no per-layer placement.
    pub device: DevicePlacement,

    /// Bytes per KV cache element, detected from the backend.
    ///
    /// Private on purpose: the backend writes the KV cache, so it alone knows
    /// the element format. A caller-supplied value could disagree with the
    /// backend and silently mis-size the allocation, so this is set only by
    /// [`set_kv_element_size`](Self::set_kv_element_size) during runtime
    /// initialization. Defaults to `f32` until a backend reports otherwise.
    kv_element_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            prefetch_window: 2,
            unmap_after_exec: true,
            num_threads: 0,
            log_level: RuntimeLogLevel::Info,
            verify_on_load: false,
            max_seq_len: 2048,
            device: DevicePlacement::CPU,
            kv_element_size: crate::runtime::backend::KV_ELEMENT_SIZE_F32,
        }
    }
}

impl RuntimeConfig {
    /// Default config with `max_seq_len` overridden.
    ///
    /// A private field makes `RuntimeConfig { max_seq_len, ..Default::default() }`
    /// illegal outside this module, so the common adjustments get builders
    /// instead. They chain: `RuntimeConfig::with_max_seq_len(512).with_prefetch_window(0)`.
    pub fn with_max_seq_len(max_seq_len: u32) -> Self {
        RuntimeConfig {
            max_seq_len,
            ..Default::default()
        }
    }

    /// Set the prefetch window, consuming and returning `self`.
    pub fn with_prefetch_window(mut self, window: usize) -> Self {
        self.prefetch_window = window;
        self
    }

    /// Set the log level, consuming and returning `self`.
    pub fn with_log_level(mut self, level: RuntimeLogLevel) -> Self {
        self.log_level = level;
        self
    }

    /// Set the thread hint passed to the backend, consuming and returning
    /// `self`. `0` means "detect".
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set the default device, consuming and returning `self`.
    pub fn with_device(mut self, device: DevicePlacement) -> Self {
        self.device = device;
        self
    }

    /// Bytes per KV cache element for the active backend.
    ///
    /// Reflects what the backend reported at initialization; `f32` before any
    /// backend has been attached.
    pub fn kv_element_size(&self) -> usize {
        self.kv_element_size
    }

    /// Record the element size reported by the backend.
    ///
    /// Crate-internal: called once by the runtime from
    /// [`ExecutionBackend::kv_element_size`](crate::runtime::backend::ExecutionBackend::kv_element_size).
    /// A zero is ignored â€” it would size the whole cache to nothing.
    // The production caller is `GllmRuntime::open` (ARTX05 Phase 7), which does
    // not exist yet; until then only tests exercise this. Kept crate-private
    // rather than `pub` so the backend stays the single source of truth.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_kv_element_size(&mut self, size: usize) {
        if size > 0 {
            self.kv_element_size = size;
        }
    }
}

/// Where the runtime is in a forward pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ExecutionState {
    /// Constructed, or reset; no pass in flight.
    #[default]
    Idle,
    /// Mapping and validating `shared.gllm`.
    LoadingShared,
    /// Executing the given layer index.
    ExecutingLayer(u32),
    /// Mapping the given layer ahead of its execution.
    Prefetching(u32),
    /// A pass finished successfully.
    Completed,
    /// A pass aborted; the string is the fatal reason.
    Failed(String),
}

impl ExecutionState {
    /// Whether this is a terminal state (`Completed` or `Failed`).
    pub fn is_terminal(&self) -> bool {
        matches!(self, ExecutionState::Completed | ExecutionState::Failed(_))
    }
}

/// Counters accumulated over a forward pass.
///
/// Reset by [`GllmRuntime::reset`](crate::runtime::GllmRuntime::reset).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionStats {
    /// Layers that ran to completion.
    pub layers_executed: u32,
    /// Layers mapped ahead of need.
    pub layers_prefetched: u32,
    /// Layers whose checksum was re-verified (`verify_on_load`).
    pub layers_verified: u32,
    /// Layers explicitly unmapped after execution.
    pub layers_unmapped: u32,
    /// Recoverable errors survived (each logged at WARN).
    pub recoverable_errors: u32,
    /// Wall time from the start of the pass to its terminal state.
    pub total_exec_time_ms: u64,
}

/// A dense f32 activation buffer passed between layers.
///
/// Deliberately minimal: ARTX05 delegates all tensor math to the backend
/// ([`ExecutionBackend`](crate::runtime::backend::ExecutionBackend)), so this
/// type only needs to carry bytes and shape across a layer boundary. It is
/// **not** a tensor library and must not grow into one.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationBuffer {
    /// Row-major elements, `len() == shape.iter().product()`.
    pub data: Vec<f32>,
    /// Logical dimensions, outermost first.
    pub shape: Vec<usize>,
}

impl ActivationBuffer {
    /// Allocate a zeroed buffer of the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        ActivationBuffer {
            data: vec![0.0; len],
            shape,
        }
    }

    /// Element count implied by `shape`.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether `data.len()` agrees with `shape`.
    ///
    /// The two can only diverge if a caller mutates `data` directly, so this
    /// is a debug aid rather than an invariant the type enforces.
    pub fn is_consistent(&self) -> bool {
        self.data.len() == self.element_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_config_defaults_target_the_8gb_machine() {
        let c = RuntimeConfig::default();
        assert_eq!(c.prefetch_window, 2);
        assert!(c.unmap_after_exec, "sequential loading is the whole point");
        assert_eq!(c.num_threads, 0, "0 = detect");
        assert_eq!(c.log_level, RuntimeLogLevel::Info);
        assert!(!c.verify_on_load, "re-reading every layer is opt-in");
        assert_eq!(c.max_seq_len, 2048);
        assert_eq!(c.device, DevicePlacement::CPU);
    }

    #[test]
    fn log_levels_order_by_severity() {
        assert!(RuntimeLogLevel::Error < RuntimeLogLevel::Warn);
        assert!(RuntimeLogLevel::Warn < RuntimeLogLevel::Info);
        assert!(RuntimeLogLevel::Info < RuntimeLogLevel::Debug);
        assert_eq!(RuntimeLogLevel::Error.as_str(), "ERROR");
        assert_eq!(RuntimeLogLevel::Debug.as_str(), "DEBUG");
    }

    #[test]
    fn execution_state_default_is_idle_and_non_terminal() {
        let s = ExecutionState::default();
        assert_eq!(s, ExecutionState::Idle);
        assert!(!s.is_terminal());
        assert!(!ExecutionState::ExecutingLayer(3).is_terminal());
        assert!(ExecutionState::Completed.is_terminal());
        assert!(ExecutionState::Failed("boom".into()).is_terminal());
    }

    #[test]
    fn execution_stats_default_is_all_zero() {
        let s = ExecutionStats::default();
        assert_eq!(s.layers_executed, 0);
        assert_eq!(s.layers_prefetched, 0);
        assert_eq!(s.layers_verified, 0);
        assert_eq!(s.layers_unmapped, 0);
        assert_eq!(s.recoverable_errors, 0);
        assert_eq!(s.total_exec_time_ms, 0);
    }

    #[test]
    fn activation_buffer_zeros_matches_shape() {
        let b = ActivationBuffer::zeros(vec![2, 3, 4]);
        assert_eq!(b.data.len(), 24);
        assert_eq!(b.element_count(), 24);
        assert!(b.is_consistent());
        assert!(b.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn activation_buffer_detects_shape_mismatch() {
        let mut b = ActivationBuffer::zeros(vec![4]);
        b.data.push(1.0);
        assert!(!b.is_consistent(), "5 elements cannot be shape [4]");
    }

    #[test]
    fn activation_buffer_scalar_shape_has_one_element() {
        // Empty shape => product of nothing => 1. Guards against a `0` that
        // would silently make every scalar buffer look empty.
        let b = ActivationBuffer::zeros(vec![]);
        assert_eq!(b.element_count(), 1);
        assert_eq!(b.data.len(), 1);
    }
}
