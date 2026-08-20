//! CPU execution path (ARTX05 §"CPU Runtime").
//!
//! This is a thin, timed wrapper around an [`ExecutionBackend`]. It owns no
//! thread pool: ARTX05 keeps layer execution sequential, and the parallelism
//! that matters lives *inside* a layer's kernels, which belong to the backend
//! (glproc has its own persistent pool). A second pool here would only
//! contend for the same cores.
//!
//! What it does own is the timing the scheduler needs to size its prefetch
//! window, and the WARN that fires when a layer is slower than expected.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{GllmError, GllmResult};
use crate::manifest::LayerManifest;
use crate::runtime::backend::ExecutionBackend;
use crate::runtime::kv_cache::KvCacheSlot;
use crate::runtime::logger::RuntimeLogger;
use crate::runtime::mmap::LayerMapping;
use crate::runtime::types::{ActivationBuffer, RuntimeConfig};

/// Executes layers on the CPU by delegating to a backend.
pub struct CpuRuntime {
    backend: Arc<dyn ExecutionBackend>,
    logger: Arc<RuntimeLogger>,
    num_threads: usize,
}

impl CpuRuntime {
    /// Wire a backend to the CPU path.
    ///
    /// Fails with [`GllmError::NoBackendAvailable`] if the backend reports
    /// itself unusable — better to refuse here than at the first layer.
    pub fn new(
        config: &RuntimeConfig,
        backend: Arc<dyn ExecutionBackend>,
        logger: Arc<RuntimeLogger>,
    ) -> GllmResult<Self> {
        if !backend.is_available() {
            return Err(GllmError::NoBackendAvailable {
                device: format!("cpu (backend {:?})", backend.name()),
            });
        }

        let num_threads = if config.num_threads == 0 {
            detect_threads()
        } else {
            config.num_threads
        };

        logger.info(
            format!("CPU runtime ready on backend {:?}", backend.name()),
            Some(format!(
                "threads={num_threads} kv_element_size={}",
                backend.kv_element_size()
            )),
        );

        Ok(CpuRuntime {
            backend,
            logger,
            num_threads,
        })
    }

    /// Execute one layer, returning how long the backend took.
    ///
    /// The duration feeds
    /// [`Scheduler::record_timings`](crate::runtime::Scheduler::record_timings);
    /// it measures the backend call alone, not mapping, so the prefetcher
    /// compares like with like.
    pub fn execute_layer(
        &self,
        mapping: &LayerMapping,
        manifest: &LayerManifest,
        kv_cache: &mut KvCacheSlot,
        input: &ActivationBuffer,
        output: &mut ActivationBuffer,
    ) -> GllmResult<Duration> {
        let layer = mapping.layer_index.unwrap_or(manifest.index);

        // Catch a mis-wired pass before the backend reads from the wrong file.
        if let Some(mapped) = mapping.layer_index
            && mapped != manifest.index
        {
            return Err(GllmError::ExecutionFailed {
                layer,
                reason: format!(
                    "mapping is layer {mapped} but manifest describes layer {}",
                    manifest.index
                ),
            });
        }

        if !input.is_consistent() {
            return Err(GllmError::ExecutionFailed {
                layer,
                reason: format!(
                    "input buffer has {} elements but shape {:?} implies {}",
                    input.data.len(),
                    input.shape,
                    input.element_count()
                ),
            });
        }

        self.logger.debug(
            format!("executing layer {layer}"),
            Some(format!(
                "tensors={} bytes={} in_shape={:?}",
                mapping.tensor_count(),
                mapping.file_size,
                input.shape
            )),
        );

        let start = Instant::now();
        let result = self
            .backend
            .execute_layer(mapping, manifest, kv_cache, input, output);
        let elapsed = start.elapsed();

        match result {
            Ok(()) => {
                self.logger.debug(
                    format!("layer {layer} done"),
                    Some(format!("{:.3} ms", elapsed.as_secs_f64() * 1000.0)),
                );
                Ok(elapsed)
            }
            Err(e) => {
                // The backend's own error is preserved as the reason; the
                // runtime only adds which layer it happened in.
                self.logger.error(
                    format!("layer {layer} failed"),
                    Some(e.to_string()),
                );
                Err(GllmError::ExecutionFailed {
                    layer,
                    reason: e.to_string(),
                })
            }
        }
    }

    /// The backend this runtime delegates to.
    pub fn backend(&self) -> &Arc<dyn ExecutionBackend> {
        &self.backend
    }

    /// Threads the backend was told it may use.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl std::fmt::Debug for CpuRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuRuntime")
            .field("backend", &self.backend.name())
            .field("num_threads", &self.num_threads)
            .finish()
    }
}

/// Best-effort core count, defaulting to 1.
///
/// `available_parallelism` reports logical CPUs and honours cgroup limits, so
/// it is the right number for a container too. It is a *hint* passed to the
/// backend; nothing here spawns threads.
fn detect_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ExtensionUri;
    use crate::runtime::backend::NullBackend;
    use crate::runtime::kv_cache::KvCacheConfig;
    use crate::runtime::types::RuntimeLogLevel;
    use crate::test_helpers::write_test_layer;
    use tempfile::TempDir;

    struct Fixture {
        _dir: TempDir,
        mapping: LayerMapping,
        manifest: LayerManifest,
        slot: KvCacheSlot,
    }

    fn fixture(layer_index: u32) -> Fixture {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(crate::manifest::format_layer_filename(layer_index));
        write_test_layer(&path, &[("attn_q.weight", 64), ("attn_k.weight", 32)]);

        let mapping = LayerMapping::open(&path, Some(layer_index), false).unwrap();
        let manifest = LayerManifest {
            index: layer_index,
            file: crate::manifest::format_layer_filename(layer_index),
            checksum: format!("sha256:{}", "0".repeat(64)),
            layer_type: ExtensionUri::parse("gllm:transformer/standard@v1").unwrap(),
            tensors: Vec::new(),
            device: None,
        };
        let kv_config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 16,
            element_size: 4,
        };
        let slot = KvCacheSlot::new(layer_index, &kv_config).unwrap();

        Fixture {
            _dir: dir,
            mapping,
            manifest,
            slot,
        }
    }

    fn cpu_runtime(backend: Arc<dyn ExecutionBackend>) -> CpuRuntime {
        CpuRuntime::new(
            &RuntimeConfig::default(),
            backend,
            RuntimeLogger::shared_silent(RuntimeLogLevel::Debug),
        )
        .unwrap()
    }

    #[test]
    fn new_detects_threads_and_reports_the_backend() {
        let logger = RuntimeLogger::shared_silent(RuntimeLogLevel::Info);
        let rt = CpuRuntime::new(
            &RuntimeConfig::default(),
            Arc::new(NullBackend::new()),
            Arc::clone(&logger),
        )
        .unwrap();

        assert!(rt.num_threads() >= 1, "must detect at least one thread");
        assert_eq!(rt.backend().name(), "null");
        assert!(
            logger.entries().iter().any(|e| e.message.contains("CPU runtime ready")),
            "startup must be logged"
        );
    }

    #[test]
    fn new_honours_an_explicit_thread_count() {
        let config = RuntimeConfig::default().with_num_threads(3);
        let rt = CpuRuntime::new(
            &config,
            Arc::new(NullBackend::new()),
            RuntimeLogger::shared_silent(RuntimeLogLevel::Error),
        )
        .unwrap();
        assert_eq!(rt.num_threads(), 3);
    }

    #[test]
    fn new_refuses_an_unavailable_backend() {
        struct Unavailable;
        impl ExecutionBackend for Unavailable {
            fn name(&self) -> &'static str {
                "unavailable"
            }
            fn is_available(&self) -> bool {
                false
            }
            fn execute_layer(
                &self,
                _l: &LayerMapping,
                _m: &LayerManifest,
                _k: &mut KvCacheSlot,
                _i: &ActivationBuffer,
                _o: &mut ActivationBuffer,
            ) -> GllmResult<()> {
                unreachable!("must never be called")
            }
        }

        let err = CpuRuntime::new(
            &RuntimeConfig::default(),
            Arc::new(Unavailable),
            RuntimeLogger::shared_silent(RuntimeLogLevel::Error),
        )
        .unwrap_err();
        assert!(matches!(err, GllmError::NoBackendAvailable { .. }), "got {err:?}");
    }

    #[test]
    fn execute_layer_delegates_and_times_the_backend() {
        let backend = Arc::new(NullBackend::new());
        let rt = cpu_runtime(backend.clone());
        let mut f = fixture(0);

        let input = ActivationBuffer::zeros(vec![4]);
        let mut output = ActivationBuffer::zeros(vec![4]);

        let elapsed = rt
            .execute_layer(&f.mapping, &f.manifest, &mut f.slot, &input, &mut output)
            .unwrap();

        assert_eq!(backend.calls(), vec![0], "backend saw exactly layer 0");
        // A duration is always produced; on a coarse clock it may be zero,
        // which the prefetcher already handles.
        assert!(elapsed.as_nanos() < 5_000_000_000, "sane duration");
        assert_eq!(output.shape, input.shape);
    }

    #[test]
    fn execute_layer_reports_which_layer_failed() {
        struct Failing;
        impl ExecutionBackend for Failing {
            fn name(&self) -> &'static str {
                "failing"
            }
            fn is_available(&self) -> bool {
                true
            }
            fn execute_layer(
                &self,
                _l: &LayerMapping,
                _m: &LayerManifest,
                _k: &mut KvCacheSlot,
                _i: &ActivationBuffer,
                _o: &mut ActivationBuffer,
            ) -> GllmResult<()> {
                Err(GllmError::IntegrityError("kernel exploded".into()))
            }
        }

        let logger = RuntimeLogger::shared_silent(RuntimeLogLevel::Error);
        let rt = CpuRuntime::new(&RuntimeConfig::default(), Arc::new(Failing), Arc::clone(&logger))
            .unwrap();
        let mut f = fixture(9);
        let input = ActivationBuffer::zeros(vec![2]);
        let mut output = ActivationBuffer::zeros(vec![2]);

        let err = rt
            .execute_layer(&f.mapping, &f.manifest, &mut f.slot, &input, &mut output)
            .unwrap_err();

        match err {
            GllmError::ExecutionFailed { layer, reason } => {
                assert_eq!(layer, 9, "the failing layer must be identified");
                assert!(
                    reason.contains("kernel exploded"),
                    "the backend's own error must survive: {reason}"
                );
            }
            other => panic!("expected ExecutionFailed, got {other:?}"),
        }
        assert_eq!(logger.count_at(RuntimeLogLevel::Error), 1);
    }

    #[test]
    fn execute_layer_rejects_a_mapping_manifest_mismatch() {
        let backend = Arc::new(NullBackend::new());
        let rt = cpu_runtime(backend.clone());
        let mut f = fixture(2);
        f.manifest.index = 5; // mapping says 2, manifest says 5

        let input = ActivationBuffer::zeros(vec![1]);
        let mut output = ActivationBuffer::zeros(vec![1]);

        let err = rt
            .execute_layer(&f.mapping, &f.manifest, &mut f.slot, &input, &mut output)
            .unwrap_err();

        assert!(matches!(err, GllmError::ExecutionFailed { layer: 2, .. }), "got {err:?}");
        assert_eq!(backend.call_count(), 0, "backend must not run on a mismatch");
    }

    #[test]
    fn execute_layer_rejects_an_inconsistent_input_buffer() {
        let backend = Arc::new(NullBackend::new());
        let rt = cpu_runtime(backend.clone());
        let mut f = fixture(0);

        let mut input = ActivationBuffer::zeros(vec![4]);
        input.data.push(1.0); // 5 elements, shape says 4
        let mut output = ActivationBuffer::zeros(vec![4]);

        let err = rt
            .execute_layer(&f.mapping, &f.manifest, &mut f.slot, &input, &mut output)
            .unwrap_err();

        assert!(matches!(err, GllmError::ExecutionFailed { .. }), "got {err:?}");
        assert_eq!(backend.call_count(), 0, "a bad buffer must not reach the backend");
    }

    #[test]
    fn repeated_execution_accumulates_backend_calls_in_order() {
        let backend = Arc::new(NullBackend::new());
        let rt = cpu_runtime(backend.clone());
        let input = ActivationBuffer::zeros(vec![2]);
        let mut output = ActivationBuffer::zeros(vec![2]);

        for i in [0u32, 1, 2] {
            let mut f = fixture(i);
            rt.execute_layer(&f.mapping, &f.manifest, &mut f.slot, &input, &mut output)
                .unwrap();
        }
        assert_eq!(backend.calls(), vec![0, 1, 2], "sequential order preserved");
    }
}
