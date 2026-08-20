//! Stummañ Pik: adapter checkpoint. **FULL implementation.**
//!
//! Layout on disk:
//!
//! ```text
//! checkpoint_000500/
//!   manifest.json          format version, adapter type, r, alpha, rslora,
//!                          d_in/d_out, step, base model id
//!   adapter.safetensors    {"lora_a": ..., "lora_b": ...}
//!   optimizer.safetensors  OPTIONAL: {"lora_a.m": ..., "lora_a.v": ..., ...}
//! ```
//!
//! Tensor names inside `adapter.safetensors` are the parameter's own names
//! (`lora_a`, `lora_b`), not a synthetic index. That is not cosmetic:
//! `TensorId` is process-global and not persistable, so the name is the only
//! key that survives a restart.
//!
//! The optimizer segment is optional and separately loadable. A deploy path
//! reads `adapter.safetensors` and never touches the optimizer bytes, which
//! are 2x the adapter's size and useless outside a resumed run.

use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{lora::VLLoraConfig, LRLora};
use crate::nn::param::TPParameter;
use crate::nn::adapter::Adapter;
use crate::optim::{Optimizer, VLNamedTensor, OPAdamW};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::collections::BTreeMap;
use std::path::Path;

use super::manifest::{VLManifest, VLValidation, MANIFEST_FILE};
use super::{
    is_transposed, safetensors, CheckpointStore, ENSegment, ENSkillStatus, VLCheckpoint,
    VLCheckpointFormat,
};

/// CPLora's capability record.
pub static CAPABILITY: &VLCheckpointFormat = &VLCheckpointFormat {
    id: "lora",
    status: ENSkillStatus::Full,
    round_trips: true,
    segments: &[MANIFEST_FILE, "adapter.safetensors", "optimizer.safetensors"],
    source: "safetensors format spec; segment split from M2_RESEARCH.md 7-E",
};

/// Adapter-only checkpoint: a directory of segments under one manifest.
///
/// A unit struct. There is no per-instance configuration, and the layout is
/// host data throughout, so it carries no backend parameter: `CPLora` writes
/// byte-identical files whichever backend produced the tensors.
pub struct CPLora;

impl CPLora {
    /// Bundle an adapter, and optionally its optimizer state, into a
    /// checkpoint ready to [`CheckpointStore::save`].
    ///
    /// Generic over the backend because this is the boundary where live tensors
    /// become host data. Everything downstream of it is backend-free.
    pub fn checkpoint_from<B: Backend>(
        adapter: &LRLora<B>,
        optimizer: Option<&OPAdamW<B>>,
        step: usize,
        base_model_id: Option<String>,
    ) -> Result<VLCheckpoint> {
        let mut manifest = VLManifest::for_lora(adapter.config(), step);
        manifest.base_model_id = base_model_id;

        let params = adapter.parameters();
        let adapter_tensors = params
            .iter()
            .map(|p| {
                Ok(VLNamedTensor::new(
                    p.name(),
                    p.to_vec()?,
                    p.shape().to_vec(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut ckpt = VLCheckpoint::new(manifest).with_segment(ENSegment::Adapter, adapter_tensors);

        if let Some(opt) = optimizer {
            // `state_tensors` is where TensorId becomes name. That translation
            // happens here and nowhere else.
            let state = opt.state_tensors(&params)?;
            ckpt.manifest.has_optimizer_state = true;
            ckpt = ckpt.with_segment(ENSegment::Optimizer, state);
        }
        Ok(ckpt)
    }

    /// Rebuild the adapter from a loaded checkpoint.
    ///
    /// Shapes are checked against the manifest rather than trusted:
    /// [`LRLora::from_tensors`] rejects a transposed `A`, which has the same
    /// element count as a correct one whenever `d_in == r`.
    pub fn restore_adapter<B: Backend>(ckpt: &VLCheckpoint) -> Result<LRLora<B>> {
        if ckpt.manifest.adapter_type != "lora" {
            return Err(GlTrainError::Checkpoint(format!(
                "CPLora cannot restore an adapter of type '{}'",
                ckpt.manifest.adapter_type
            )));
        }
        let config: VLLoraConfig = ckpt.manifest.lora_config();
        let a = ckpt.require_tensor(ENSegment::Adapter, "lora_a")?;
        let b = ckpt.require_tensor(ENSegment::Adapter, "lora_b")?;
        LRLora::from_tensors(
            config,
            Tensor::<B>::from_vec(a.data.clone(), &a.shape)?,
            Tensor::<B>::from_vec(b.data.clone(), &b.shape)?,
        )
    }

    /// Restore optimizer state into `optimizer`, re-keying by name.
    ///
    /// `params` must be the adapter's parameters *as they exist now*: they
    /// carry this process's `TensorId`s, which is what the saved names get
    /// mapped back onto.
    pub fn restore_optimizer<B: Backend>(
        ckpt: &VLCheckpoint,
        optimizer: &mut OPAdamW<B>,
        params: &[&TPParameter<B>],
    ) -> Result<()> {
        let Some(state) = ckpt.segment(ENSegment::Optimizer) else {
            return Err(GlTrainError::Checkpoint(
                "checkpoint has no optimizer segment; it was saved for deployment, not resume"
                    .into(),
            ));
        };
        optimizer.load_state(params, state)
    }
}

impl CheckpointStore for CPLora {
    fn save(&self, dir: &Path, ckpt: &VLCheckpoint) -> Result<()> {
        std::fs::create_dir_all(dir)?;

        // The manifest is written last, so a directory carrying one is a
        // directory whose segments are already complete. A crash mid-save
        // leaves an obviously-incomplete checkpoint rather than a manifest
        // pointing at files that are not there.
        for (kind, tensors) in &ckpt.segments {
            let mut meta = BTreeMap::new();
            meta.insert("adapter_type".to_string(), ckpt.manifest.adapter_type.clone());
            meta.insert("step".to_string(), ckpt.manifest.step.to_string());
            meta.insert(
                "format_version".to_string(),
                ckpt.manifest.format_version.to_string(),
            );
            safetensors::write(&dir.join(kind.file_name()), tensors, &meta)?;
        }

        let mut manifest = ckpt.manifest.clone();
        manifest.has_optimizer_state = ckpt.segments.contains_key(&ENSegment::Optimizer);
        std::fs::write(dir.join(MANIFEST_FILE), manifest.to_json())?;
        Ok(())
    }

    fn load(&self, dir: &Path) -> Result<VLCheckpoint> {
        let manifest_path = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&manifest_path).map_err(|e| {
            GlTrainError::Checkpoint(format!(
                "cannot read {}: {e}",
                manifest_path.display()
            ))
        })?;
        let manifest = VLManifest::from_json(&text)?;

        // A major-version difference means the layout is not the one this code
        // knows how to parse, so refuse before reading any tensor bytes.
        let compat = manifest
            .format_version
            .is_compatible_with(&super::FORMAT_VERSION);
        if compat == super::ENVersionCompatibility::Incompatible {
            return Err(GlTrainError::Checkpoint(format!(
                "checkpoint format version {} cannot be read by this build ({}): the major \
                 version differs, so the layout is not the same",
                manifest.format_version,
                super::FORMAT_VERSION
            )));
        }

        let mut ckpt = VLCheckpoint::new(manifest);
        let adapter_path = dir.join(ENSegment::Adapter.file_name());
        if !adapter_path.exists() {
            return Err(GlTrainError::Checkpoint(format!(
                "{} is missing its adapter segment ({})",
                dir.display(),
                ENSegment::Adapter.file_name()
            )));
        }
        ckpt = ckpt.with_segment(ENSegment::Adapter, safetensors::read(&adapter_path)?);

        // Loaded independently: a deploy path can stop here and never parse
        // optimizer bytes it will not use.
        let optim_path = dir.join(ENSegment::Optimizer.file_name());
        if optim_path.exists() {
            ckpt = ckpt.with_segment(ENSegment::Optimizer, safetensors::read(&optim_path)?);
        } else if ckpt.manifest.has_optimizer_state {
            return Err(GlTrainError::Checkpoint(format!(
                "{MANIFEST_FILE} says optimizer state was saved, but {} is missing",
                ENSegment::Optimizer.file_name()
            )));
        }
        Ok(ckpt)
    }

    fn validate(&self, dir: &Path, against: &VLManifest) -> Result<VLValidation> {
        let mut v = VLValidation::new();

        let manifest_path = dir.join(MANIFEST_FILE);
        let Ok(text) = std::fs::read_to_string(&manifest_path) else {
            v.error(format!("{} is missing or unreadable", manifest_path.display()));
            return Ok(v);
        };
        let manifest = match VLManifest::from_json(&text) {
            Ok(m) => m,
            Err(e) => {
                v.error(format!("{MANIFEST_FILE} could not be parsed: {e}"));
                return Ok(v);
            }
        };
        // Every manifest-level disagreement, in one pass.
        v = manifest.validate_against(against);

        let adapter_path = dir.join(ENSegment::Adapter.file_name());
        if !adapter_path.exists() {
            v.error(format!(
                "{} is missing",
                ENSegment::Adapter.file_name()
            ));
            return Ok(v);
        }
        let tensors = match safetensors::read(&adapter_path) {
            Ok(t) => t,
            Err(e) => {
                v.error(format!(
                    "{} could not be read: {e}",
                    ENSegment::Adapter.file_name()
                ));
                return Ok(v);
            }
        };

        // Shapes, not element counts. `[d_in, r]` and `[r, d_in]` have the
        // same length whenever d_in == r, and would load silently.
        let expected: [(&str, Vec<usize>); 2] = [
            ("lora_a", vec![against.d_in, against.r]),
            ("lora_b", vec![against.r, against.d_out]),
        ];
        let mut present = Vec::new();
        for (name, want) in &expected {
            match tensors.iter().find(|t| &t.name == name) {
                None => v.error(format!("adapter segment is missing '{name}'")),
                Some(t) => {
                    present.push(name.to_string());
                    if &t.shape != want {
                        v.error(format!(
                            "'{name}' has shape {:?}, expected {want:?}{}",
                            t.shape,
                            if is_transposed(&t.shape, want) {
                                " (transposed: same element count, different layout)"
                            } else {
                                ""
                            }
                        ));
                    }
                    let n: usize = t.shape.iter().product();
                    if t.data.len() != n {
                        v.error(format!(
                            "'{name}' declares {n} elements but holds {}",
                            t.data.len()
                        ));
                    }
                }
            }
        }
        // Not an error: optimizer state and future segments legitimately add
        // keys an adapter loader does not consume.
        for t in &tensors {
            if !present.contains(&t.name) {
                v.warn(format!(
                    "adapter segment has a tensor this loader does not consume: '{}'",
                    t.name
                ));
            }
        }
        Ok(v)
    }

    fn capability(&self) -> &'static VLCheckpointFormat {
        CAPABILITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::grad_store::VLGradStore;
    use crate::backend::GlProc;
    use crate::nn::adapter::VLAdapterSpec;
    use crate::optim::VLAdamWConfig;

    /// Checkpoint values round-trip through f32 bytes with no arithmetic, so
    /// any difference at all is a real bug.
    const TOL_EXACT: f32 = 0.0;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "stumman_cp_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn adapter() -> LRLora<GlProc> {
        LRLora::new(&VLAdapterSpec::new(6, 4, 2, 7)).unwrap()
    }

    /// An adapter whose B is non-zero, so a round trip has something to prove:
    /// B starts at exactly zero, and comparing zeros to zeros would pass even
    /// if the tensor were never written.
    fn trained_adapter() -> (LRLora<GlProc>, OPAdamW<GlProc>) {
        let mut a = adapter();
        let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
            lr: 0.1,
            ..VLAdamWConfig::default()
        });
        let mut grads = VLGradStore::new();
        for p in a.parameters() {
            let g: Vec<f32> = (0..p.n_elems()).map(|i| 0.1 + i as f32 * 0.01).collect();
            grads.accumulate(p.id(), g, p.shape().to_vec()).unwrap();
        }
        let mut params = a.parameters_mut();
        opt.step(&mut params, &grads).unwrap();
        (a, opt)
    }

    #[test]
    fn cplora_save_creates_the_expected_files() {
        let dir = tmp_dir("layout");
        let (a, opt) = trained_adapter();
        let ckpt = CPLora::checkpoint_from(&a, Some(&opt), 500, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        assert!(dir.join("manifest.json").exists());
        assert!(dir.join("adapter.safetensors").exists());
        assert!(dir.join("optimizer.safetensors").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Without optimizer state, the sidecar must not exist at all: a deploy
    /// path should not have to skip a file it never wanted.
    #[test]
    fn cplora_save_omits_the_optimizer_file_when_there_is_no_optimizer_state() {
        let dir = tmp_dir("noopt");
        let a = adapter();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        assert!(dir.join("adapter.safetensors").exists());
        assert!(!dir.join("optimizer.safetensors").exists());
        let back = CPLora.load(&dir).unwrap();
        assert!(back.segment(ENSegment::Optimizer).is_none());
        assert!(!back.manifest.has_optimizer_state);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cplora_round_trip_preserves_the_adapter_weights() {
        let dir = tmp_dir("adapter");
        let (a, _) = trained_adapter();
        let want_a = a.a().to_vec().unwrap();
        let want_b = a.b().to_vec().unwrap();
        // The trained B must be non-zero, or this test proves nothing.
        assert!(want_b.iter().any(|v| *v != 0.0), "B is still zero");

        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 25, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        let loaded = CPLora.load(&dir).unwrap();
        let back = CPLora::restore_adapter::<GlProc>(&loaded).unwrap();
        assert_eq!(back.config(), a.config());
        for (g, w) in back.a().to_vec().unwrap().iter().zip(&want_a) {
            assert!((g - w).abs() <= TOL_EXACT, "A: {g} != {w}");
        }
        for (g, w) in back.b().to_vec().unwrap().iter().zip(&want_b) {
            assert!((g - w).abs() <= TOL_EXACT, "B: {g} != {w}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cplora_round_trip_preserves_the_optimizer_state() {
        let dir = tmp_dir("optstate");
        let (a, opt) = trained_adapter();
        let ckpt = CPLora::checkpoint_from(&a, Some(&opt), 25, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        let loaded = CPLora.load(&dir).unwrap();
        assert!(loaded.manifest.has_optimizer_state);

        // A fresh adapter carries different TensorIds, exactly as it would
        // after a restart. The state has to find its way back by name.
        let fresh = adapter();
        assert_ne!(fresh.a().id(), a.a().id(), "test needs genuinely new IDs");

        let mut restored = OPAdamW::<GlProc>::new(VLAdamWConfig::default());
        let params = fresh.parameters();
        CPLora::restore_optimizer(&loaded, &mut restored, &params).unwrap();

        assert_eq!(restored.step_count(), opt.step_count());
        for (fresh_p, orig_p) in fresh.parameters().iter().zip(a.parameters()) {
            let want = opt.moments(orig_p.id()).expect("original state");
            let got = restored.moments(fresh_p.id()).expect("restored state");
            let (wm, gm) = (GlProc::to_vec(&want.m).unwrap(), GlProc::to_vec(&got.m).unwrap());
            let (wv, gv) = (GlProc::to_vec(&want.v).unwrap(), GlProc::to_vec(&got.v).unwrap());
            assert_eq!(gm, wm, "m for '{}'", fresh_p.name());
            assert_eq!(gv, wv, "v for '{}'", fresh_p.name());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Tensor names on disk must be the parameter names, since `TensorId` is
    /// process-local and could not be matched after a restart.
    #[test]
    fn cplora_keys_tensors_by_parameter_name() {
        let dir = tmp_dir("names");
        let (a, opt) = trained_adapter();
        let ckpt = CPLora::checkpoint_from(&a, Some(&opt), 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        let loaded = CPLora.load(&dir).unwrap();
        let names: Vec<&str> = loaded
            .segment(ENSegment::Adapter)
            .unwrap()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(names, vec!["lora_a", "lora_b"]);

        let opt_names: Vec<&str> = loaded
            .segment(ENSegment::Optimizer)
            .unwrap()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert!(opt_names.contains(&"lora_a.m"), "{opt_names:?}");
        assert!(opt_names.contains(&"lora_b.v"), "{opt_names:?}");
        assert!(
            !opt_names.iter().any(|n| n.parse::<usize>().is_ok()),
            "a bare numeric key would be a TensorId: {opt_names:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A major version bump means the layout changed, so refuse before reading
    /// any tensor bytes.
    #[test]
    fn cplora_load_rejects_a_major_version_mismatch() {
        let dir = tmp_dir("majorver");
        let a = adapter();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        let path = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replace(r#""1.0""#, r#""2.0""#)).unwrap();

        let err = CPLora.load(&dir).unwrap_err().to_string();
        assert!(err.contains("major"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A minor bump is additive, so the file stays loadable.
    #[test]
    fn cplora_load_accepts_a_minor_version_mismatch() {
        let dir = tmp_dir("minorver");
        let a = adapter();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        let path = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, text.replace(r#""1.0""#, r#""1.7""#)).unwrap();

        let loaded = CPLora.load(&dir).unwrap();
        let v = CPLora.validate(&dir, &loaded.manifest.clone()).unwrap();
        assert!(v.ok(), "minor mismatch must stay loadable: {v:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The bug class this repo has actually hit: `[d_in, r]` and `[r, d_in]`
    /// have the same element count, so only a shape check separates them.
    #[test]
    fn validator_catches_a_transposed_adapter_tensor() {
        let dir = tmp_dir("transposed");
        // r == d_in == 4, so A is [4,4] and a transpose has an identical
        // element count AND an identical shape. Use d_in=6, r=2: [6,2] vs
        // [2,6], both 12 elements.
        let a = LRLora::<GlProc>::new(&VLAdapterSpec::new(6, 4, 2, 7)).unwrap();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        // Rewrite the adapter segment with A transposed: same 12 values.
        let mut tensors = safetensors::read(&dir.join("adapter.safetensors")).unwrap();
        for t in &mut tensors {
            if t.name == "lora_a" {
                assert_eq!(t.shape, vec![6, 2]);
                t.shape = vec![2, 6];
            }
        }
        safetensors::write(
            &dir.join("adapter.safetensors"),
            &tensors,
            &BTreeMap::new(),
        )
        .unwrap();

        let v = CPLora.validate(&dir, &ckpt.manifest).unwrap();
        assert!(!v.ok(), "a transposed A must be an error, not a warning");
        let msg = v.errors.join(" | ");
        assert!(msg.contains("lora_a"), "{msg}");
        assert!(
            msg.contains("transposed"),
            "the error must name the transposed case: {msg}"
        );
        // The transposed case must not be downgraded to a warning. Other
        // warnings are fine and expected here: this checkpoint records no base
        // model id, which warns on its own.
        assert!(
            !v.warnings.iter().any(|w| w.contains("lora_a")),
            "a transposed tensor must be fatal, not a warning: {v:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// One pass, every problem. Re-running to discover the second error is
    /// exactly the workflow this avoids.
    #[test]
    fn validator_reports_every_error_in_one_pass() {
        let dir = tmp_dir("multierr");
        let a = adapter();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        // Expect a completely different geometry AND a different adapter type.
        let mut expected = ckpt.manifest.clone();
        expected.r = 8;
        expected.d_in = 64;
        expected.adapter_type = "dora".to_string();

        let v = CPLora.validate(&dir, &expected).unwrap();
        assert!(!v.ok());
        assert!(
            v.errors.len() >= 4,
            "expected manifest errors plus a shape error, got {v:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn validator_reports_a_missing_adapter_tensor() {
        let dir = tmp_dir("missing");
        let a = adapter();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        let tensors: Vec<VLNamedTensor> = safetensors::read(&dir.join("adapter.safetensors"))
            .unwrap()
            .into_iter()
            .filter(|t| t.name != "lora_b")
            .collect();
        safetensors::write(&dir.join("adapter.safetensors"), &tensors, &BTreeMap::new()).unwrap();

        let v = CPLora.validate(&dir, &ckpt.manifest).unwrap();
        assert!(!v.ok());
        assert!(v.errors.iter().any(|e| e.contains("lora_b")), "{v:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An extra tensor is not an error: optimizer state and future segments
    /// legitimately add keys an adapter loader does not consume.
    #[test]
    fn validator_warns_rather_than_errors_on_an_unconsumed_tensor() {
        let dir = tmp_dir("extra");
        let a = adapter();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();

        let mut tensors = safetensors::read(&dir.join("adapter.safetensors")).unwrap();
        tensors.push(VLNamedTensor::new("lora_magnitude", vec![1.0; 4], vec![1, 4]));
        safetensors::write(&dir.join("adapter.safetensors"), &tensors, &BTreeMap::new()).unwrap();

        let v = CPLora.validate(&dir, &ckpt.manifest).unwrap();
        assert!(v.ok(), "an extra tensor is loadable: {v:?}");
        assert!(
            v.warnings.iter().any(|w| w.contains("lora_magnitude")),
            "{v:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The manifest is written last, so it is the marker that the segments are
    /// complete. A directory without one is not a checkpoint.
    #[test]
    fn cplora_load_rejects_a_directory_with_no_manifest() {
        let dir = tmp_dir("nomanifest");
        assert!(CPLora.load(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A manifest promising optimizer state whose file is absent is a
    /// truncated checkpoint, not a deploy-only one.
    #[test]
    fn cplora_load_rejects_a_manifest_promising_absent_optimizer_state() {
        let dir = tmp_dir("truncated");
        let (a, opt) = trained_adapter();
        let ckpt = CPLora::checkpoint_from(&a, Some(&opt), 1, None).unwrap();
        CPLora.save(&dir, &ckpt).unwrap();
        std::fs::remove_file(dir.join("optimizer.safetensors")).unwrap();

        let err = CPLora.load(&dir).unwrap_err().to_string();
        assert!(err.contains("optimizer"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restore_optimizer_refuses_a_deploy_only_checkpoint() {
        let a = adapter();
        let ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig::default());
        let params = a.parameters();
        assert!(CPLora::restore_optimizer(&ckpt, &mut opt, &params).is_err());
    }

    /// Determinism: the same state written twice must produce identical bytes,
    /// or a content-addressed store would see spurious changes.
    #[test]
    fn saving_the_same_checkpoint_twice_produces_identical_bytes() {
        let d1 = tmp_dir("det1");
        let d2 = tmp_dir("det2");
        let (a, opt) = trained_adapter();
        let ckpt = CPLora::checkpoint_from(&a, Some(&opt), 7, Some("qwen".into())).unwrap();
        CPLora.save(&d1, &ckpt).unwrap();
        CPLora.save(&d2, &ckpt).unwrap();

        for f in ["manifest.json", "adapter.safetensors", "optimizer.safetensors"] {
            assert_eq!(
                std::fs::read(d1.join(f)).unwrap(),
                std::fs::read(d2.join(f)).unwrap(),
                "{f} differs between two saves of the same checkpoint"
            );
        }
        std::fs::remove_dir_all(&d1).ok();
        std::fs::remove_dir_all(&d2).ok();
    }

    #[test]
    fn restore_adapter_refuses_a_checkpoint_of_another_adapter_type() {
        let a = adapter();
        let mut ckpt = CPLora::checkpoint_from(&a, None::<&OPAdamW<GlProc>>, 1, None).unwrap();
        ckpt.manifest.adapter_type = "dora".to_string();
        assert!(CPLora::restore_adapter::<GlProc>(&ckpt).is_err());
    }
}
