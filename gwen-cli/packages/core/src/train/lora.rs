use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::train::config::LoraConfig;

pub struct LoraLayer {
    /// Frozen pre-trained weights — detached from the computation graph.
    base: Linear,
    /// Trainable down-projection (d_in → r).
    lora_a: Linear,
    /// Trainable up-projection (r → d_out), zero-initialised.
    lora_b: Linear,
    /// alpha / r, applied to the low-rank delta before adding to base output.
    scale: f32,
}

impl LoraLayer {
    /// Build a LoRA-wrapped linear layer.
    ///
    /// `base_weight` must already be loaded (e.g. from a safetensors file).
    /// It is detached here so it never accumulates gradients.
    /// `vb` must come from a `VarMap` so that lora_a / lora_b are tracked by
    /// the optimizer.
    pub fn new(
        d_in: usize,
        d_out: usize,
        base_weight: Tensor,
        config: &LoraConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Freeze base: detach removes it from the autograd graph entirely.
        let base = Linear::new(base_weight.detach(), None);

        // lora_a: random normal (mean=0, std=1). Shape (r, d_in) — projects d_in → r.
        // get_with_hints stores the tensor in the VarMap so the optimizer can reach it.
        let a_weight = vb.get_with_hints(
            (config.r, d_in),
            "lora_a",
            candle_nn::init::Init::Randn { mean: 0.0, stdev: 1.0 },
        )?;
        let lora_a = Linear::new(a_weight, None);

        // lora_b: zeros. Shape (d_out, r) so that lora_b(·) projects r → d_out.
        let b_weight = vb.get_with_hints(
            (d_out, config.r),
            "lora_b",
            candle_nn::init::Init::Const(0.0),
        )?;
        let lora_b = Linear::new(b_weight, None);

        let scale = config.alpha / config.r as f32;

        Ok(Self { base, lora_a, lora_b, scale })
    }

    /// output = base(x) + lora_b(lora_a(x)) * scale
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out  = self.base.forward(x)?;
        let lora_out  = self.lora_b.forward(&self.lora_a.forward(x)?)?;
        base_out + (lora_out * self.scale as f64)?
    }

    /// Number of trainable scalar parameters (lora_a + lora_b weights only).
    pub fn trainable_params(&self) -> usize {
        let a = self.lora_a.weight().elem_count();
        let b = self.lora_b.weight().elem_count();
        a + b
    }
}
