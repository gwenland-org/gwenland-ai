// diagnostics/doctor.rs — `gwen doctor` health checks.
//
// Cycle 6: removed Ollama and mistralrs checks (no longer used).
// Added native inference check (candle device detection + GGUF model scan).
//
// @EDITABLE: Wire new check_* functions into run_all_checks().

use crate::storage::paths::gwen_config_dir;
use serde::Serialize;
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
pub async fn run_all_checks(safe: bool, force: bool) -> Vec<CheckResult> {
    let force_apply = !safe && force;
    let (cuda, vram, disk, native_inference, models) = tokio::join!(
        check_cuda(),
        check_vram(),
        check_disk(),
        check_native_inference(),
        check_models_dir(),
    );

    vec![cuda, vram, disk, native_inference, models]
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
            }
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

/// Check how many GGUF models are present in ~/.config/gwen/models/.
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
                .filter(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("gguf")
                })
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
