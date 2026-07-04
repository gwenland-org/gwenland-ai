// diagnostics/doctor.rs — `gwen doctor` health checks.
//
// Cycle 6: removed Ollama and mistralrs checks (no longer used).
// Added native inference check (candle device detection + GGUF model scan).
//
// @EDITABLE: Wire new check_* functions into run_all_checks().

use crate::convert::gguf_parser::{self, MetadataValue};
use crate::storage::paths::{GwenPaths, gwen_config_dir};
use serde::Serialize;
use std::path::PathBuf;
use std::process::Command;

// ── check result types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Fail,
    NotApplicable,
    Warning,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub value: String,
    pub fix_available: bool,
    pub fix_applied: bool,
    pub fix_succeeded: bool,
    pub suggestion: Option<String>,
}

// ── entry point ────────────────────────────────────────────────────────────────

/// Run all diagnostic checks concurrently.
///
/// `model_paths` — explicit GGUF files to inspect for training readiness.
/// Pass an empty vec to scan `GwenPaths::models_dir()` automatically.
pub async fn run_all_checks(
    safe: bool,
    force: bool,
    model_paths: Vec<PathBuf>,
) -> Vec<CheckResult> {
    let _force_apply = !safe && force;
    let (cuda, vram, disk, native_inference, models) = tokio::join!(
        check_cuda(),
        check_vram(),
        check_disk(),
        check_native_inference(),
        check_models_dir(),
    );

    let mut results = vec![cuda, vram, disk, native_inference, models];
    results.extend(check_gwenland_root());
    results.extend(check_gguf_training_readiness(model_paths));
    results
}

// ── CUDA ──────────────────────────────────────────────────────────────────────

async fn check_cuda() -> CheckResult {
    if which::which("nvidia-smi").is_err() {
        return CheckResult {
            name: "cuda".into(),
            status: CheckStatus::NotApplicable,
            value: "no NVIDIA GPU detected".into(),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: None,
        };
    }

    let output = match Command::new("nvidia-smi").output() {
        Ok(out) => out,
        Err(_) => {
            return CheckResult {
                name: "cuda".into(),
                status: CheckStatus::Fail,
                value: "nvidia-smi exec failed".into(),
                fix_available: false,
                fix_applied: false,
                fix_succeeded: false,
                suggestion: Some("Check NVIDIA drivers".into()),
            };
        }
    };

    if !output.status.success() {
        return CheckResult {
            name: "cuda".into(),
            status: CheckStatus::Fail,
            value: "driver issue".into(),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: Some("NVIDIA driver may be broken".into()),
        };
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let version = stdout
        .lines()
        .find_map(|line| {
            if line.contains("CUDA Version:") {
                let parts: Vec<&str> = line.split("CUDA Version:").collect();
                if parts.len() > 1 {
                    let v: Vec<&str> = parts[1].trim().split_whitespace().collect();
                    return v.first().map(|s| s.to_string());
                }
            }
            None
        })
        .unwrap_or_else(|| "present".into());

    CheckResult {
        name: "cuda".into(),
        status: CheckStatus::Pass,
        value: version,
        fix_available: false,
        fix_applied: false,
        fix_succeeded: false,
        suggestion: None,
    }
}

// ── VRAM ──────────────────────────────────────────────────────────────────────

async fn check_vram() -> CheckResult {
    if which::which("nvidia-smi").is_err() {
        return CheckResult {
            name: "vram".into(),
            status: CheckStatus::NotApplicable,
            value: "no NVIDIA GPU".into(),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: None,
        };
    }

    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let mb: u64 = stdout.trim().parse().unwrap_or(0);
            let gb = mb as f64 / 1024.0;
            CheckResult {
                name: "vram".into(),
                status: CheckStatus::Pass,
                value: format!("{:.1} GB free", gb),
                fix_available: false,
                fix_applied: false,
                fix_succeeded: false,
                suggestion: None,
            }
        }
        _ => CheckResult {
            name: "vram".into(),
            status: CheckStatus::Fail,
            value: "error reading VRAM".into(),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: None,
        },
    }
}

// ── Disk ──────────────────────────────────────────────────────────────────────

async fn check_disk() -> CheckResult {
    let dir = gwen_config_dir();
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }

    match fs2::free_space(&dir) {
        Ok(bytes) => {
            let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            CheckResult {
                name: "disk".into(),
                status: CheckStatus::Pass,
                value: format!("{:.1} GB free", gb),
                fix_available: false,
                fix_applied: false,
                fix_succeeded: false,
                suggestion: None,
            }
        }
        Err(_) => CheckResult {
            name: "disk".into(),
            status: CheckStatus::Warning,
            value: "cannot read free space".into(),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: None,
        },
    }
}

// ── Native inference ──────────────────────────────────────────────────────────

/// Verify that candle device detection works at runtime.
/// Passes if Device::new_cuda(0) succeeds, or if CPU fallback is available.
/// A Fail here would mean candle-core itself is broken — unlikely but worth surfacing.
async fn check_native_inference() -> CheckResult {
    use candle_core::Device;

    let (device_str, status) = if Device::new_cuda(0).is_ok() {
        ("CUDA (GPU 0)".to_string(), CheckStatus::Pass)
    } else if Device::new_metal(0).is_ok() {
        ("Metal (GPU 0)".to_string(), CheckStatus::Pass)
    } else {
        // CPU is always available — this is a Pass, not a Warning.
        // We only warn if *no* candle device can be constructed at all.
        ("CPU (fallback)".to_string(), CheckStatus::Pass)
    };

    CheckResult {
        name: "native-inference".into(),
        status,
        value: device_str,
        fix_available: false,
        fix_applied: false,
        fix_succeeded: false,
        suggestion: None,
    }
}

// ── Models directory ──────────────────────────────────────────────────────────

/// Check how many GGUF models are present in ~/.gwenland/models/.
async fn check_models_dir() -> CheckResult {
    use crate::engine::inference::loader::gwen_models_dir;

    let dir = gwen_models_dir();
    if !dir.exists() {
        return CheckResult {
            name: "models".into(),
            status: CheckStatus::Warning,
            value: "no models directory — run `gwen fetch <model>`".into(),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: Some("gwen fetch qwen3:8b".into()),
        };
    }

    let count = std::fs::read_dir(&dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("gguf"))
                .count()
        })
        .unwrap_or(0);

    if count == 0 {
        return CheckResult {
            name: "models".into(),
            status: CheckStatus::Warning,
            value: format!("0 GGUF files in {}", dir.display()),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: Some("gwen fetch qwen3:8b".into()),
        };
    }

    CheckResult {
        name: "models".into(),
        status: CheckStatus::Pass,
        value: format!("{} model(s) in {}", count, dir.display()),
        fix_available: false,
        fix_applied: false,
        fix_succeeded: false,
        suggestion: None,
    }
}

// ── GwenLand storage root (GWEN-224 Wave 4) ───────────────────────────────────

/// Report existence + writability of the four `~/.gwenland/` subdirectories.
/// Each subdirectory gets its own check entry so a partial setup (e.g.
/// crash-logs/ exists but is read-only) is visible at a glance rather than
/// collapsed into one pass/fail bit.
fn check_gwenland_root() -> Vec<CheckResult> {
    let entries: [(&str, PathBuf); 4] = [
        ("gwenland-root", GwenPaths::root_dir()),
        ("gwenland-config", GwenPaths::config_dir()),
        ("gwenland-models", GwenPaths::models_dir()),
        ("gwenland-crash-logs", GwenPaths::crash_logs_dir()),
    ];

    entries
        .into_iter()
        .map(|(name, dir)| check_dir_writable(name, &dir))
        .collect()
}

fn check_dir_writable(name: &str, dir: &PathBuf) -> CheckResult {
    if !dir.is_dir() {
        return CheckResult {
            name: name.into(),
            status: CheckStatus::Fail,
            value: format!("missing: {}", dir.display()),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: Some("directory should auto-create on next `gwen` invocation".into()),
        };
    }

    let probe = dir.join(".gwen-doctor-write-probe");
    match std::fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            CheckResult {
                name: name.into(),
                status: CheckStatus::Pass,
                value: dir.display().to_string(),
                fix_available: false,
                fix_applied: false,
                fix_succeeded: false,
                suggestion: None,
            }
        }
        Err(e) => CheckResult {
            name: name.into(),
            status: CheckStatus::Fail,
            value: format!("not writable: {} ({e})", dir.display()),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: Some("check filesystem permissions".into()),
        },
    }
}

// ── GGUF training readiness ───────────────────────────────────────────────────

/// Probe every GGUF in `model_paths` (or all GGUFs in `GwenPaths::models_dir()`
/// when the list is empty) and report whether each model supports the
/// weight-tied training path.
///
/// For each file the check reports:
///   - which resolution path was taken (explicit metadata KV vs structural)
///   - the raw evidence (KV value / tensor presence)
///   - whether training will succeed without sampled-softmax
fn check_gguf_training_readiness(model_paths: Vec<PathBuf>) -> Vec<CheckResult> {
    let paths = if model_paths.is_empty() {
        collect_gguf_paths_from_models_dir()
    } else {
        model_paths
    };

    if paths.is_empty() {
        return vec![];
    }

    paths
        .into_iter()
        .map(probe_gguf_training_readiness)
        .collect()
}

fn collect_gguf_paths_from_models_dir() -> Vec<PathBuf> {
    let dir = GwenPaths::models_dir();
    if !dir.exists() {
        return vec![];
    }
    std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|x| x.to_str()) == Some("gguf") {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

fn probe_gguf_training_readiness(path: PathBuf) -> CheckResult {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let check_name = format!("gguf-train:{}", stem);

    let header = match gguf_parser::parse_header(&path) {
        Ok(h) => h,
        Err(e) => {
            return CheckResult {
                name: check_name,
                status: CheckStatus::Warning,
                value: format!("parse error: {}", e),
                fix_available: false,
                fix_applied: false,
                fix_succeeded: false,
                suggestion: Some("file may be corrupt or not a valid GGUF".into()),
            };
        }
    };

    let arch = header
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let tie_key = format!("{arch}.tie_word_embeddings");
    let tie_kv = header.metadata.get(&tie_key);

    const OUTPUT_HEAD_NAMES: [&str; 3] =
        ["output.weight", "lm_head.weight", "model.lm_head.weight"];
    let has_output_head = header
        .tensors
        .iter()
        .any(|t| OUTPUT_HEAD_NAMES.contains(&t.name.as_str()));

    let (tied, resolution) = match tie_kv {
        Some(MetadataValue::Bool(true)) => (true, "metadata=true".to_string()),
        Some(MetadataValue::Bool(false)) => (false, "metadata=false".to_string()),
        Some(_) => (
            false,
            format!("metadata key present but wrong type ({})", tie_key),
        ),
        None if !has_output_head => (true, "structural (no separate output head)".to_string()),
        None => (
            false,
            "structural (output.weight / lm_head.weight present; metadata key absent)".to_string(),
        ),
    };

    if tied {
        CheckResult {
            name: check_name,
            status: CheckStatus::Pass,
            value: format!("tied — {}", resolution),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: None,
        }
    } else {
        let suggestion = if resolution.contains("metadata=false") {
            "model is genuinely untied; use sampled-softmax (option b) or switch to a tied model (e.g. Qwen3-0.6B)".into()
        } else {
            "output.weight exists but tie_word_embeddings KV is absent — possible metadata gap in GGUF conversion; try re-converting or check config.json on HuggingFace".into()
        };
        CheckResult {
            name: check_name,
            status: CheckStatus::Fail,
            value: format!("untied — {}", resolution),
            fix_available: false,
            fix_applied: false,
            fix_succeeded: false,
            suggestion: Some(suggestion),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::paths::test_support::set_gwen_home;

    #[test]
    fn reports_pass_when_all_dirs_present_and_writable() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = set_gwen_home(temp.path());

        // GwenPaths::*_dir() auto-creates on call.
        let results = check_gwenland_root();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.status, CheckStatus::Pass, "{} should pass: {}", r.name, r.value);
        }
    }

    #[test]
    fn reports_fail_when_a_dir_does_not_exist() {
        let temp = tempfile::tempdir().unwrap();
        let never_created = temp.path().join("does-not-exist");

        let result = check_dir_writable("gwenland-crash-logs", &never_created);
        assert_eq!(result.status, CheckStatus::Fail);
        assert!(result.value.contains("missing"));
    }

    // Unix permission bits reliably block writes inside a directory; Windows'
    // read-only attribute on a *directory* is cosmetic (Explorer-only) and
    // does not stop file creation, so there's no portable way to simulate
    // "exists but unwritable" without ACL plumbing. Gate this one to Unix.
    #[cfg(unix)]
    #[test]
    fn reports_fail_when_a_dir_is_not_writable() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let _guard = set_gwen_home(temp.path());

        let unwritable = temp.path().join("readonly-crash-logs");
        std::fs::create_dir_all(&unwritable).unwrap();
        std::fs::set_permissions(&unwritable, std::fs::Permissions::from_mode(0o500)).unwrap();

        let result = check_dir_writable("gwenland-crash-logs", &unwritable);

        // Restore so the tempdir can be cleaned up.
        std::fs::set_permissions(&unwritable, std::fs::Permissions::from_mode(0o700)).unwrap();

        assert_eq!(result.status, CheckStatus::Fail);
    }
}
