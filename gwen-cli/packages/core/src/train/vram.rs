use crate::train::config::TrainConfig;

// ── lookup table ─────────────────────────────────────────────────────────────

pub struct VramEntry {
    pub model_label: &'static str,
    pub param_billions: f32,
    pub quant: &'static str,
    pub base_vram_gb: f32,
}

pub const VRAM_TABLE: &[VramEntry] = &[
    // ── 1B models ──
    VramEntry { model_label: "1B model at 4-bit  (e.g. SmolLM 1.7B)",       param_billions: 1.0,  quant: "q4",  base_vram_gb: 0.9  },
    VramEntry { model_label: "1B model at 8-bit  (e.g. SmolLM 1.7B)",       param_billions: 1.0,  quant: "q8",  base_vram_gb: 1.7  },
    VramEntry { model_label: "1B model full precision",                       param_billions: 1.0,  quant: "f16", base_vram_gb: 3.4  },
    // ── 3B models ──
    VramEntry { model_label: "3B model at 4-bit  (e.g. Phi-3 Mini, Qwen 3B)",   param_billions: 3.0,  quant: "q4",  base_vram_gb: 2.0  },
    VramEntry { model_label: "3B model at 8-bit  (e.g. Phi-3 Mini, Qwen 3B)",   param_billions: 3.0,  quant: "q8",  base_vram_gb: 3.5  },
    VramEntry { model_label: "3B model full precision",                           param_billions: 3.0,  quant: "f16", base_vram_gb: 6.0  },
    // ── 7B models ──
    VramEntry { model_label: "7B model at 4-bit  (e.g. Mistral 7B, LLaMA 3.1 8B)",  param_billions: 7.0,  quant: "q4",  base_vram_gb: 4.2  },
    VramEntry { model_label: "7B model at 8-bit  (e.g. Mistral 7B, LLaMA 3.1 8B)",  param_billions: 7.0,  quant: "q8",  base_vram_gb: 7.8  },
    VramEntry { model_label: "7B model full precision",                               param_billions: 7.0,  quant: "f16", base_vram_gb: 14.0 },
    // ── 13B models ──
    VramEntry { model_label: "13B model at 4-bit  (e.g. LLaMA 2 13B, Qwen 14B)",  param_billions: 13.0, quant: "q4",  base_vram_gb: 7.8  },
    VramEntry { model_label: "13B model at 8-bit  (e.g. LLaMA 2 13B, Qwen 14B)",  param_billions: 13.0, quant: "q8",  base_vram_gb: 14.0 },
    VramEntry { model_label: "13B model full precision",                             param_billions: 13.0, quant: "f16", base_vram_gb: 26.0 },
    // ── 27B models ──
    VramEntry { model_label: "27B model at 4-bit  (e.g. Gemma 2 27B, Qwen 32B)",  param_billions: 27.0, quant: "q4",  base_vram_gb: 15.0 },
    VramEntry { model_label: "27B model at 8-bit  (e.g. Gemma 2 27B, Qwen 32B)",  param_billions: 27.0, quant: "q8",  base_vram_gb: 27.0 },
    VramEntry { model_label: "27B model full precision",                             param_billions: 27.0, quant: "f16", base_vram_gb: 54.0 },
    // ── 70B models ──
    VramEntry { model_label: "70B model at 4-bit  (e.g. LLaMA 3.1 70B, Qwen 72B)",  param_billions: 70.0, quant: "q4",  base_vram_gb: 38.0  },
    VramEntry { model_label: "70B model at 8-bit  (e.g. LLaMA 3.1 70B, Qwen 72B)",  param_billions: 70.0, quant: "q8",  base_vram_gb: 70.0  },
    VramEntry { model_label: "70B model full precision",                               param_billions: 70.0, quant: "f16", base_vram_gb: 140.0 },
];

// ── estimate types ────────────────────────────────────────────────────────────

pub struct VramEstimate {
    pub model_label: String,
    pub base_gb: f32,
    pub lora_gb: f32,
    pub activation_gb: f32,
    pub optimizer_gb: f32,
    pub safety_gb: f32,
    pub total_gb: f32,
    pub available_gb: Option<f32>,
    pub gpu_name: Option<String>,
    pub fits: bool,
}

// ── public API ────────────────────────────────────────────────────────────────

pub fn estimate_vram(config: &TrainConfig) -> VramEstimate {
    let entry = lookup_entry(config);
    let base_gb = entry.base_vram_gb;

    let lora_gb = (config.lora_r as f32 * config.lora_alpha as f32 * 0.001).max(0.1);
    let activation_gb = (config.batch_size as f32 * config.max_seq_len as f32 * 0.001).max(0.5);
    let optimizer_gb = if config.optimizer.contains("8bit") {
        base_gb * 0.4
    } else {
        base_gb * 0.8
    };

    let subtotal = base_gb + lora_gb + activation_gb + optimizer_gb;
    let safety_gb = subtotal * 0.2;
    let total_gb = subtotal + safety_gb;

    let (available_gb, gpu_name) = query_gpu_vram();

    VramEstimate {
        model_label: entry.model_label.to_string(),
        base_gb,
        lora_gb,
        activation_gb,
        optimizer_gb,
        safety_gb,
        total_gb,
        available_gb,
        gpu_name,
        fits: available_gb.map(|v| v >= total_gb).unwrap_or(false),
    }
}

pub fn estimate_vram_gb(config: &TrainConfig) -> f64 {
    estimate_vram(config).total_gb as f64
}

pub fn estimate_train_time(config: &TrainConfig, total_samples: usize) -> String {
    let params = parse_param_billions(&config.model);
    let steps_per_epoch = (total_samples / config.batch_size.max(1) as usize).max(1);
    let total_steps = steps_per_epoch * config.epochs as usize;

    let steps_per_sec = 0.8 / (params / 7.0).max(0.1);
    let total_secs = total_steps as f32 / steps_per_sec;

    format_duration(total_secs as u64)
}

pub fn vram_suggestions(config: &TrainConfig, estimate: &VramEstimate) -> Vec<String> {
    let mut suggestions = vec![];
    if !config.qlora {
        suggestions.push("→ Enable QLoRA (4-bit): add `qlora: true` to config".to_string());
    }
    if config.batch_size > 1 {
        suggestions.push(format!("→ Reduce batch size: `batch_size: 1`"));
    }
    if estimate.base_gb > 15.0 {
        suggestions.push("→ Try a smaller model: Mistral 7B or LLaMA 3.1 8B fit on most GPUs".to_string());
    }
    suggestions.push("→ Use gradient checkpointing to reduce activation memory".to_string());
    suggestions
}

// ── internal helpers ──────────────────────────────────────────────────────────

fn lookup_entry(config: &TrainConfig) -> &'static VramEntry {
    let params = parse_param_billions(&config.model);
    let quant = parse_quant(config);

    // Snap param count to the nearest tier in our table
    let tier = snap_to_tier(params);

    VRAM_TABLE
        .iter()
        .find(|e| (e.param_billions - tier).abs() < 0.1 && e.quant == quant)
        // Fall back to 7B q4 — the most common training scenario
        .or_else(|| VRAM_TABLE.iter().find(|e| (e.param_billions - 7.0).abs() < 0.1 && e.quant == "q4"))
        .unwrap_or(&VRAM_TABLE[6]) // 7B q4 is index 6; static fallback
}

fn snap_to_tier(params: f32) -> f32 {
    // Map raw param count to the closest lookup table tier
    const TIERS: &[(f32, f32)] = &[
        (0.0,  1.0),
        (2.0,  3.0),
        (5.5,  7.0),
        (10.0, 13.0),
        (20.0, 27.0),
        (50.0, 70.0),
    ];
    for &(threshold, tier) in TIERS.iter().rev() {
        if params >= threshold {
            return tier;
        }
    }
    1.0
}

fn parse_param_billions(model_name: &str) -> f32 {
    // Scan the model name for patterns like 7b, 7B, 8B, 13B, 32B, 72B, etc.
    let lower = model_name.to_lowercase();
    let mut i = 0;
    let chars: Vec<char> = lower.chars().collect();

    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                j += 1;
            }
            if j < chars.len() && chars[j] == 'b' {
                if let Ok(val) = lower[i..j].parse::<f32>() {
                    return val;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    7.0 // fallback: assume 7B
}

fn parse_quant(config: &TrainConfig) -> &'static str {
    if config.qlora { "q4" }
    else if config.fp16 { "f16" }
    else { "q8" }
}

fn query_gpu_vram() -> (Option<f32>, Option<String>) {
    // Try nvidia-smi first (most reliable)
    if let Some((gb, name)) = query_nvidia_smi() {
        return (Some(gb), Some(name));
    }

    // Try /sys/class/drm for AMD/Intel on Linux
    #[cfg(target_os = "linux")]
    if let Some(gb) = query_sysfs_vram() {
        return (Some(gb), None);
    }

    (None, None)
}

fn query_nvidia_smi() -> Option<(f32, String)> {
    which::which("nvidia-smi").ok()?;

    let total_out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !total_out.status.success() {
        return None;
    }
    let mb: f32 = String::from_utf8_lossy(&total_out.stdout)
        .lines()
        .next()
        .and_then(|l| l.trim().parse().ok())?;

    let name_out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    let gpu_name = if name_out.status.success() {
        String::from_utf8_lossy(&name_out.stdout)
            .lines()
            .next()
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| "NVIDIA GPU".to_string())
    } else {
        "NVIDIA GPU".to_string()
    };

    Some((mb / 1024.0, gpu_name))
}

#[cfg(target_os = "linux")]
fn query_sysfs_vram() -> Option<f32> {
    for card in 0..4u8 {
        let path = format!("/sys/class/drm/card{}/device/mem_info_vram_total", card);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(bytes) = content.trim().parse::<u64>() {
                if bytes > 0 {
                    return Some(bytes as f32 / (1024.0 * 1024.0 * 1024.0));
                }
            }
        }
    }
    None
}

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}
