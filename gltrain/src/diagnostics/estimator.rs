use std::time::Duration;

// ── types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct VramEstimate {
    /// Frozen model weights in bf16 (2 bytes/param).
    pub model_mb: f32,
    /// Forward-pass activations across all layers.
    pub activations_mb: f32,
    /// AdamW optimizer states for LoRA params only (8 bytes/param: m + v in fp32).
    pub optimizer_mb: f32,
    /// Sum of the above × 1.2 safety buffer.
    pub total_mb: f32,
}

// ── known device TFLOPS ───────────────────────────────────────────────────────

pub enum Device {
    Cpu,
    T4,
    A100,
    Rtx3080,
    Rtx4090,
    Unknown,
}

impl Device {
    /// Peak bf16/fp16 TFLOPS for the device.
    pub fn tflops(&self) -> f32 {
        match self {
            Device::Cpu     =>   0.1,
            Device::T4      =>  65.0,
            Device::A100    => 312.0,
            Device::Rtx3080 => 119.0,
            Device::Rtx4090 => 330.0,
            Device::Unknown =>   1.0,
        }
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Pure VRAM estimate from first principles.
///
/// - `total_params`  — full model parameter count (all layers, not just LoRA)
/// - `lora_params`   — trainable LoRA parameters only (from `LoraLayer::trainable_params()`)
/// - `batch_size`    — micro-batch size
/// - `seq_len`       — sequence length in tokens
/// - `d_model`       — hidden dimension of the model
/// - `num_layers`    — number of transformer layers
pub fn estimate_vram(
    total_params: usize,
    lora_params: usize,
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    num_layers: usize,
) -> VramEstimate {
    // bf16: 2 bytes per parameter
    let model_mb = (total_params as f32 * 2.0) / 1_048_576.0;

    // Activation tensor per layer: (batch × seq × d_model) in bf16 (2 bytes), times all layers
    let activations_mb =
        (batch_size as f32 * seq_len as f32 * d_model as f32 * num_layers as f32 * 2.0)
            / 1_048_576.0;

    // AdamW first + second moment for LoRA params in fp32: 4 bytes × 2 states = 8 bytes
    let optimizer_mb = (lora_params as f32 * 8.0) / 1_048_576.0;

    let total_mb = (model_mb + activations_mb + optimizer_mb) * 1.2;

    VramEstimate { model_mb, activations_mb, optimizer_mb, total_mb }
}

/// Pure training-time estimate.
///
/// - `total_tokens`   — total tokens across the full dataset (all samples)
/// - `epochs`         — number of training epochs
/// - `device_tflops`  — device peak TFLOPS; use `Device::tflops()` or pass directly
///
/// Uses the standard 6-FLOPs-per-token-per-parameter estimate for transformer
/// training, with 0.6 MFU (model flop utilisation).
pub fn estimate_time(
    total_tokens: usize,
    epochs: usize,
    device_tflops: f32,
) -> Duration {
    let flops = total_tokens as f64 * epochs as f64 * 6.0;
    // effective throughput in FLOPS/s: tflops × 0.6 × 1e12
    let effective = device_tflops as f64 * 0.6 * 1e12;
    let seconds = flops / effective;
    Duration::from_secs_f64(seconds)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_components_sum_with_buffer() {
        let est = estimate_vram(
            7_000_000_000, // 7B model
            2_097_152,     // ~2M LoRA params
            1,             // batch size
            1024,          // seq len
            4096,          // d_model
            32,            // num_layers
        );

        let subtotal = est.model_mb + est.activations_mb + est.optimizer_mb;
        let expected_total = subtotal * 1.2;
        assert!((est.total_mb - expected_total).abs() < 0.01,
            "total_mb={} expected={}", est.total_mb, expected_total);
    }

    #[test]
    fn vram_model_mb_formula() {
        // 1M params × 2 bytes = 2 MB exactly
        let est = estimate_vram(1_048_576, 0, 1, 1, 1, 1);
        assert!((est.model_mb - 2.0).abs() < 1e-4);
    }

    #[test]
    fn vram_zero_lora_zero_optimizer() {
        let est = estimate_vram(0, 0, 1, 1, 1, 1);
        assert_eq!(est.optimizer_mb, 0.0);
    }

    #[test]
    fn estimate_time_a100_reasonable() {
        // 1B tokens × 1 epoch on A100 → should be well under an hour
        let dur = estimate_time(1_000_000_000, 1, Device::A100.tflops());
        assert!(dur.as_secs() < 3600,
            "expected < 1h, got {}s", dur.as_secs());
    }

    #[test]
    fn estimate_time_cpu_slow() {
        // CPU should be much slower than A100 for the same workload
        let cpu  = estimate_time(1_000_000_000, 1, Device::Cpu.tflops());
        let a100 = estimate_time(1_000_000_000, 1, Device::A100.tflops());
        assert!(cpu > a100, "CPU should be slower than A100");
    }

    #[test]
    fn estimate_time_scales_with_epochs() {
        let one   = estimate_time(1_000_000, 1, Device::T4.tflops());
        let three = estimate_time(1_000_000, 3, Device::T4.tflops());
        assert_eq!(three, one * 3);
    }

    #[test]
    fn device_tflops_table() {
        assert_eq!(Device::Cpu.tflops(),       0.1);
        assert_eq!(Device::T4.tflops(),       65.0);
        assert_eq!(Device::A100.tflops(),    312.0);
        assert_eq!(Device::Rtx3080.tflops(), 119.0);
        assert_eq!(Device::Rtx4090.tflops(), 330.0);
        assert_eq!(Device::Unknown.tflops(),   1.0);
    }
}
