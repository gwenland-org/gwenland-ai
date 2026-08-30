//! The top-level orchestrator (ARTX05 §"Execution Model").
//!
//! [`GllmRuntime`] ties the previous phases together into one forward pass:
//! map a layer, resolve its device, hand it to a backend, record timings,
//! unmap, repeat. It computes nothing itself.
//!
//! ## What a pass is, and is not
//!
//! [`run`](GllmRuntime::run) executes every layer once, in order — one
//! forward pass over an activation the caller supplies. It is not a
//! generation loop: there is no sampling, no token loop, and no tokenizer
//! (ARTX1 OQ3 is unresolved, so a package carries no vocabulary). Turning a
//! pass into text is a job for a future `GllmInference` implementation.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::error::{GllmError, GllmResult};
use crate::manifest::GllmManifest;
use crate::package::GllmPackage;
use crate::runtime::backend::ExecutionBackend;
use crate::runtime::cpu::CpuRuntime;
use crate::runtime::device::{DeviceMapConfig, DeviceMapResolver};
use crate::runtime::kv_cache::{KvCache, KvCacheConfig};
use crate::runtime::logger::{LogEntry, RuntimeLogger};
use crate::runtime::mmap::LayerMapping;
use crate::runtime::scheduler::Scheduler;
use crate::runtime::types::{
    ActivationBuffer, ExecutionState, ExecutionStats, RuntimeConfig, RuntimeLogLevel,
};

/// Orchestrates layer-sequential execution over an opened package.
pub struct GllmRuntime {
    package: GllmPackage,
    config: RuntimeConfig,
    logger: Arc<RuntimeLogger>,
    scheduler: Scheduler,
    cpu: CpuRuntime,
    resolver: DeviceMapResolver,
    device_map: DeviceMapConfig,
    kv_cache: KvCache,
    state: ExecutionState,
    stats: ExecutionStats,
}

impl GllmRuntime {
    /// Open a package and prepare to execute it. Runs no layers.
    ///
    /// Initialization order matters: the backend's KV element size is read
    /// *before* the cache is sized, so the allocation matches what the
    /// backend will actually write.
    pub fn open(
        root: &Path,
        config: RuntimeConfig,
        backend: Arc<dyn ExecutionBackend>,
    ) -> GllmResult<Self> {
        let logger = Arc::new(RuntimeLogger::new(config.log_level));
        Self::open_with_logger(root, config, backend, logger)
    }

    /// As [`open`](Self::open), with a caller-supplied logger.
    ///
    /// Lets an embedder collect entries itself, and lets tests assert on what
    /// was logged.
    pub fn open_with_logger(
        root: &Path,
        mut config: RuntimeConfig,
        backend: Arc<dyn ExecutionBackend>,
        logger: Arc<RuntimeLogger>,
    ) -> GllmResult<Self> {
        let package = GllmPackage::open(root)?;
        logger.info(
            format!("opened package {:?}", package.manifest().model_id),
            Some(format!(
                "layers={} format={:?}",
                package.layer_count(),
                package.layout.format
            )),
        );
        for w in package.warnings() {
            logger.warn("manifest warning", Some(w.clone()));
        }

        // The backend owns the KV format; ask it before sizing anything.
        config.set_kv_element_size(backend.kv_element_size());

        let cpu = CpuRuntime::new(&config, backend, Arc::clone(&logger))?;

        let kv_config = KvCacheConfig::from_metadata(&package.manifest().metadata, &config)?;
        logger.info(
            "sizing KV cache",
            Some(format!(
                "{} MiB ({} layers x {} B, seq {}, elem {} B)",
                kv_config.total_size_bytes() / (1024 * 1024),
                kv_config.num_layers,
                kv_config.slot_size_bytes(),
                kv_config.max_seq_len,
                kv_config.element_size,
            )),
        );
        let kv_cache = KvCache::allocate(kv_config)?;

        let layer_count = u32::try_from(package.layer_count()).map_err(|_| {
            GllmError::RuntimeNotReady(format!("absurd layer count {}", package.layer_count()))
        })?;

        Ok(GllmRuntime {
            scheduler: Scheduler::new(layer_count, &config),
            resolver: DeviceMapResolver::cpu_only(),
            device_map: DeviceMapConfig::new(config.device.clone()),
            package,
            config,
            logger,
            cpu,
            kv_cache,
            state: ExecutionState::Idle,
            stats: ExecutionStats::default(),
        })
    }

    /// Open with [`RuntimeConfig::default`].
    pub fn open_default(root: &Path, backend: Arc<dyn ExecutionBackend>) -> GllmResult<Self> {
        Self::open(root, RuntimeConfig::default(), backend)
    }

    /// Replace the device map (ARTX06 range assignments).
    pub fn set_device_map(&mut self, device_map: DeviceMapConfig) {
        self.device_map = device_map;
    }

    /// Replace the device resolver — used to declare which accelerators are
    /// present. Defaults to CPU-only.
    pub fn set_device_resolver(&mut self, resolver: DeviceMapResolver) {
        self.resolver = resolver;
    }

    /// Run one forward pass over every layer.
    ///
    /// `activation` is the input to layer 0 and is replaced by the output of
    /// the final layer. On a fatal error the state becomes
    /// [`ExecutionState::Failed`] and the error is returned; stats up to that
    /// point are preserved for diagnosis.
    pub fn run(&mut self, activation: &mut ActivationBuffer) -> GllmResult<ExecutionStats> {
        let started = Instant::now();
        self.scheduler.reset();
        // Stats describe *this* pass. Without clearing them, a second run()
        // would report the sum of both — the KV cache and logger are left
        // alone, since carrying a sequence across passes is a valid use.
        self.stats = ExecutionStats::default();
        self.state = ExecutionState::LoadingShared;

        match self.run_inner(activation) {
            Ok(()) => {
                self.state = ExecutionState::Completed;
                self.stats.total_exec_time_ms = started.elapsed().as_millis() as u64;
                self.logger.info(
                    "pass complete",
                    Some(format!(
                        "{} layers in {} ms (prefetched {}, unmapped {})",
                        self.stats.layers_executed,
                        self.stats.total_exec_time_ms,
                        self.stats.layers_prefetched,
                        self.stats.layers_unmapped
                    )),
                );
                Ok(self.stats.clone())
            }
            Err(e) => {
                self.state = ExecutionState::Failed(e.to_string());
                self.stats.total_exec_time_ms = started.elapsed().as_millis() as u64;
                self.logger.error("pass failed", Some(e.to_string()));
                Err(e)
            }
        }
    }

    fn run_inner(&mut self, activation: &mut ActivationBuffer) -> GllmResult<()> {
        // Shared components are already parsed and validated by
        // GllmPackage::open; this only records that the phase happened.
        self.logger.debug("shared components ready", None);

        let mut scratch = ActivationBuffer::zeros(activation.shape.clone());

        while let Some(index) = self.scheduler.current() {
            self.state = ExecutionState::ExecutingLayer(index);

            // --- map ---------------------------------------------------
            let load_start = Instant::now();
            let path = self
                .package
                .layout
                .layer_path(index)
                .ok_or_else(|| GllmError::LayerOutOfBounds {
                    index: index as usize,
                    max: self.package.layer_count(),
                })?
                .path
                .clone();
            let mapping = LayerMapping::open(&path, Some(index), false)?;
            let load_time = load_start.elapsed();
            self.scheduler.mark_ready(index);

            if self.config.verify_on_load {
                self.verify_layer(index)?;
            }

            // --- resolve device ----------------------------------------
            let layer_manifest = self
                .package
                .layer_manifest(index)
                .ok_or_else(|| GllmError::MissingExecutionUnit(format!("layer {index}")))?
                .clone();

            let resolved = self.resolver.resolve(
                index,
                Some(&layer_manifest),
                &self.device_map,
                &self.config.device,
            );
            if resolved.is_fallback() {
                // Recoverable per AD-07: the layer still runs, on CPU.
                self.stats.recoverable_errors += 1;
                if resolved.is_distributed_unavailable() {
                    // ARTX10 Wave 2: the manifest names a rank placement, but
                    // this build has no distributed runtime (no NCCL/RCCL/
                    // Gloo — deferred to Sanctum Visibilia). Worded
                    // differently from a plain missing-device fallback so
                    // this doesn't read as "install a GPU driver" when the
                    // real gap is "stand up multi-node execution".
                    self.logger.warn(
                        format!(
                            "layer {index}: distributed placement requested but no \
                             distributed runtime is available — running on CPU"
                        ),
                        resolved.overridden.as_ref().map(|p| format!("requested {p}")),
                    );
                } else {
                    self.logger.warn(
                        format!("layer {index}: falling back to CPU"),
                        resolved.overridden.as_ref().map(|p| format!("requested {p}")),
                    );
                }
            }

            // --- execute -----------------------------------------------
            let slot = self
                .kv_cache
                .slot_mut(index)
                .ok_or_else(|| GllmError::ExecutionFailed {
                    layer: index,
                    reason: format!("no KV cache slot for layer {index}"),
                })?;

            let exec_time =
                self.cpu
                    .execute_layer(&mapping, &layer_manifest, slot, activation, &mut scratch)?;

            // The output of this layer is the input to the next. Swapping
            // reuses both allocations instead of copying per layer.
            std::mem::swap(activation, &mut scratch);

            self.scheduler.record_timings(load_time, exec_time);
            self.stats.layers_executed += 1;

            // --- unmap -------------------------------------------------
            if self.config.unmap_after_exec {
                drop(mapping);
                self.stats.layers_unmapped += 1;
            }

            self.scheduler.advance();
        }

        Ok(())
    }

    /// Re-verify one layer's checksum against the manifest.
    fn verify_layer(&mut self, index: u32) -> GllmResult<()> {
        let issues = self.package.cross_check_layer(index)?;
        if !issues.is_empty() {
            return Err(GllmError::IntegrityError(format!(
                "layer {index}: {}",
                issues.join("; ")
            )));
        }
        self.stats.layers_verified += 1;
        Ok(())
    }

    /// Clear stats, KV cache, and state, ready for another pass.
    ///
    /// Keeps the KV allocation — that is the point of pre-allocating.
    pub fn reset(&mut self) {
        self.scheduler.reset();
        self.kv_cache.reset_all();
        self.stats = ExecutionStats::default();
        self.state = ExecutionState::Idle;
        self.logger.clear();
    }

    /// Current execution state.
    pub fn state(&self) -> &ExecutionState {
        &self.state
    }

    /// Counters from the last (or in-flight) pass.
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// The package's manifest.
    pub fn manifest(&self) -> &GllmManifest {
        self.package.manifest()
    }

    /// Manifest warnings raised when the package was opened.
    pub fn warnings(&self) -> &[String] {
        self.package.warnings()
    }

    /// Snapshot of everything logged so far.
    pub fn log_entries(&self) -> Vec<LogEntry> {
        self.logger.entries()
    }

    /// How many ERROR entries were logged.
    pub fn error_count(&self) -> usize {
        self.logger.count_at(RuntimeLogLevel::Error)
    }

    /// The KV cache.
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }

    /// The effective runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Layers in this model.
    pub fn layer_count(&self) -> u32 {
        self.scheduler.total_layers()
    }

    /// The opened package.
    pub fn package(&self) -> &GllmPackage {
        &self.package
    }
}

impl std::fmt::Debug for GllmRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GllmRuntime")
            .field("layers", &self.scheduler.total_layers())
            .field("state", &self.state)
            .field("backend", &self.cpu.backend().name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::backend::NullBackend;
    use crate::test_helpers::write_runtime_package;
    use tempfile::TempDir;

    fn open_runtime(
        dir: &TempDir,
        backend: Arc<dyn ExecutionBackend>,
    ) -> GllmResult<(GllmRuntime, Arc<RuntimeLogger>)> {
        let logger = RuntimeLogger::shared_silent(RuntimeLogLevel::Debug);
        let rt = GllmRuntime::open_with_logger(
            dir.path(),
            RuntimeConfig::default(),
            backend,
            Arc::clone(&logger),
        )?;
        Ok((rt, logger))
    }

    #[test]
    fn open_leaves_the_runtime_idle() {
        let dir = write_runtime_package(3);
        let (rt, _) = open_runtime(&dir, Arc::new(NullBackend::new())).unwrap();

        assert_eq!(*rt.state(), ExecutionState::Idle);
        assert_eq!(rt.layer_count(), 3);
        assert_eq!(rt.stats().layers_executed, 0);
        assert_eq!(rt.kv_cache().len(), 3, "one slot per layer");
    }

    #[test]
    fn open_fails_on_a_missing_package() {
        let dir = TempDir::new().unwrap();
        let err = GllmRuntime::open_default(dir.path(), Arc::new(NullBackend::new())).unwrap_err();
        assert!(!matches!(err, GllmError::ExecutionFailed { .. }), "got {err:?}");
    }

    #[test]
    fn run_executes_every_layer_in_order() {
        let dir = write_runtime_package(3);
        let backend = Arc::new(NullBackend::new());
        let (mut rt, _) = open_runtime(&dir, backend.clone()).unwrap();

        let mut act = ActivationBuffer::zeros(vec![4]);
        let stats = rt.run(&mut act).unwrap();

        assert_eq!(backend.calls(), vec![0, 1, 2], "sequential order");
        assert_eq!(stats.layers_executed, 3);
        assert_eq!(*rt.state(), ExecutionState::Completed);
        assert_eq!(rt.error_count(), 0);
    }

    #[test]
    fn a_zero_layer_package_completes_without_executing() {
        let dir = write_runtime_package(0);
        let backend = Arc::new(NullBackend::new());
        let (mut rt, _) = open_runtime(&dir, backend.clone()).unwrap();

        let mut act = ActivationBuffer::zeros(vec![2]);
        let stats = rt.run(&mut act).unwrap();

        assert_eq!(stats.layers_executed, 0);
        assert_eq!(backend.call_count(), 0);
        assert_eq!(*rt.state(), ExecutionState::Completed);
    }

    #[test]
    fn unmap_after_exec_is_counted() {
        let dir = write_runtime_package(2);
        let (mut rt, _) = open_runtime(&dir, Arc::new(NullBackend::new())).unwrap();
        assert!(rt.config().unmap_after_exec);

        let mut act = ActivationBuffer::zeros(vec![2]);
        let stats = rt.run(&mut act).unwrap();
        assert_eq!(stats.layers_unmapped, 2, "each layer unmapped after use");
    }

    #[test]
    fn verify_on_load_cross_checks_each_layer() {
        let dir = write_runtime_package(2);
        let logger = RuntimeLogger::shared_silent(RuntimeLogLevel::Info);
        let config = RuntimeConfig::default();
        let mut rt = GllmRuntime::open_with_logger(
            dir.path(),
            verify_on_load(config),
            Arc::new(NullBackend::new()),
            logger,
        )
        .unwrap();

        let mut act = ActivationBuffer::zeros(vec![2]);
        let stats = rt.run(&mut act).unwrap();
        assert_eq!(stats.layers_verified, 2);
    }

    fn verify_on_load(mut c: RuntimeConfig) -> RuntimeConfig {
        c.verify_on_load = true;
        c
    }

    #[test]
    fn a_failing_backend_marks_the_pass_failed() {
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
                _m: &crate::manifest::LayerManifest,
                _k: &mut crate::runtime::kv_cache::KvCacheSlot,
                _i: &ActivationBuffer,
                _o: &mut ActivationBuffer,
            ) -> GllmResult<()> {
                Err(GllmError::IntegrityError("bad kernel".into()))
            }
        }

        let dir = write_runtime_package(3);
        let (mut rt, _) = open_runtime(&dir, Arc::new(Failing)).unwrap();

        let mut act = ActivationBuffer::zeros(vec![2]);
        let err = rt.run(&mut act).unwrap_err();

        assert!(matches!(err, GllmError::ExecutionFailed { layer: 0, .. }), "got {err:?}");
        assert!(matches!(rt.state(), ExecutionState::Failed(_)));
        assert_eq!(rt.stats().layers_executed, 0, "stopped at the first layer");
        assert!(rt.error_count() > 0, "the failure must be logged");
    }

    #[test]
    fn a_corrupt_layer_aborts_the_pass() {
        let dir = write_runtime_package(2);
        // Corrupt layer 1's magic after the package was written.
        let path = dir.path().join("GLLMTensorLayer-0001.gllm");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let backend = Arc::new(NullBackend::new());
        let (mut rt, _) = open_runtime(&dir, backend.clone()).unwrap();
        let mut act = ActivationBuffer::zeros(vec![2]);
        let err = rt.run(&mut act).unwrap_err();

        assert!(matches!(err, GllmError::InvalidMagic(_)), "got {err:?}");
        assert!(matches!(rt.state(), ExecutionState::Failed(_)));
        assert_eq!(backend.calls(), vec![0], "layer 0 ran, layer 1 never did");
    }

    #[test]
    fn reset_rewinds_state_stats_and_cache() {
        let dir = write_runtime_package(2);
        let (mut rt, _) = open_runtime(&dir, Arc::new(NullBackend::new())).unwrap();

        let mut act = ActivationBuffer::zeros(vec![2]);
        rt.run(&mut act).unwrap();
        rt.kv_cache.slot_mut(0).unwrap().advance(5).unwrap();
        assert_eq!(rt.kv_cache().max_current_seq_len(), 5);

        let allocated = rt.kv_cache().total_allocated_bytes();
        rt.reset();

        assert_eq!(*rt.state(), ExecutionState::Idle);
        assert_eq!(*rt.stats(), ExecutionStats::default());
        assert_eq!(rt.kv_cache().max_current_seq_len(), 0);
        assert_eq!(
            rt.kv_cache().total_allocated_bytes(),
            allocated,
            "reset must not free the KV allocation"
        );
    }

    #[test]
    fn a_runtime_can_run_twice_after_reset() {
        let dir = write_runtime_package(2);
        let backend = Arc::new(NullBackend::new());
        let (mut rt, _) = open_runtime(&dir, backend.clone()).unwrap();

        let mut act = ActivationBuffer::zeros(vec![2]);
        rt.run(&mut act).unwrap();
        rt.reset();
        let stats = rt.run(&mut act).unwrap();

        assert_eq!(stats.layers_executed, 2, "second pass is complete");
        assert_eq!(backend.calls(), vec![0, 1, 0, 1], "both passes ran");
    }

    #[test]
    fn run_is_repeatable_without_reset() {
        // run() resets the scheduler itself, so a second call must not stall
        // on a scheduler still marked complete.
        let dir = write_runtime_package(2);
        let backend = Arc::new(NullBackend::new());
        let (mut rt, _) = open_runtime(&dir, backend.clone()).unwrap();

        let mut act = ActivationBuffer::zeros(vec![2]);
        rt.run(&mut act).unwrap();
        let stats = rt.run(&mut act).unwrap();
        assert_eq!(stats.layers_executed, 2);
        assert_eq!(backend.call_count(), 4);
    }

    #[test]
    fn per_layer_progress_is_logged() {
        let dir = write_runtime_package(2);
        let (mut rt, logger) = open_runtime(&dir, Arc::new(NullBackend::new())).unwrap();

        let mut act = ActivationBuffer::zeros(vec![2]);
        rt.run(&mut act).unwrap();

        let entries = logger.entries();
        assert!(
            entries.iter().any(|e| e.message.contains("pass complete")),
            "completion must be logged"
        );
        assert!(
            entries.iter().any(|e| e.message.contains("executing layer 0")),
            "each layer must be logged at DEBUG"
        );
    }

    #[test]
    fn kv_cache_is_sized_from_the_backend_element_size() {
        use crate::runtime::backend::KV_ELEMENT_SIZE_F16;

        let dir = write_runtime_package(2);
        let (f32_rt, _) = open_runtime(&dir, Arc::new(NullBackend::new())).unwrap();
        let (f16_rt, _) = open_runtime(
            &dir,
            Arc::new(NullBackend::with_kv_element_size(KV_ELEMENT_SIZE_F16)),
        )
        .unwrap();

        assert_eq!(f32_rt.config().kv_element_size(), 4);
        assert_eq!(f16_rt.config().kv_element_size(), 2);
        assert_eq!(
            f32_rt.kv_cache().total_allocated_bytes(),
            f16_rt.kv_cache().total_allocated_bytes() * 2,
            "an f16 backend must halve the cache"
        );
    }

    #[test]
    fn a_gpu_request_without_a_gpu_falls_back_and_is_counted() {
        let dir = write_runtime_package(2);
        let (mut rt, logger) = open_runtime(&dir, Arc::new(NullBackend::new())).unwrap();

        // Ask for cuda:0 on a CPU-only resolver.
        let mut map = DeviceMapConfig::default();
        map.assign_range("0-1", crate::manifest::DevicePlacement::CUDA0)
            .unwrap();
        rt.set_device_map(map);

        let mut act = ActivationBuffer::zeros(vec![2]);
        let stats = rt.run(&mut act).unwrap();

        assert_eq!(stats.layers_executed, 2, "the pass still completes on CPU");
        assert_eq!(stats.recoverable_errors, 2, "one fallback per layer");
        assert!(
            logger
                .entries()
                .iter()
                .any(|e| e.message.contains("falling back to CPU")),
            "the fallback must be visible"
        );
    }

    #[test]
    fn a_rank_placement_warns_distinctly_and_still_completes_on_cpu() {
        // ARTX10 Wave 2: a manifest naming "rank:N/cuda:M" must run (on CPU —
        // no distributed runtime exists) and must say *why* it isn't a plain
        // missing-device fallback.
        let dir = write_runtime_package(2);
        let (mut rt, logger) = open_runtime(&dir, Arc::new(NullBackend::new())).unwrap();

        let mut map = DeviceMapConfig::default();
        map.assign_range("0-1", crate::manifest::DevicePlacement::Other("rank:0/cuda:0".into()))
            .unwrap();
        rt.set_device_map(map);

        let mut act = ActivationBuffer::zeros(vec![2]);
        let stats = rt.run(&mut act).unwrap();

        assert_eq!(stats.layers_executed, 2, "the pass still completes on CPU");
        assert_eq!(stats.recoverable_errors, 2, "one fallback per layer");
        assert!(
            logger
                .entries()
                .iter()
                .any(|e| e.message.contains("distributed runtime is available")),
            "the distributed-specific reason must be visible, not a generic fallback"
        );
        assert!(
            logger.entries().iter().all(|e| !e.message.contains("falling back to CPU")),
            "a rank placement must not also log the generic message"
        );
    }
}
