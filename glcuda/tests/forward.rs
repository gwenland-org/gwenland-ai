//! End-to-end forward-pass parity: the same tiny f32 model in glcuda and
//! glproc must produce matching logits and identical greedy continuations.
//!
//! This is the M2 definition-of-done test in miniature: full transformer
//! layers (GQA attention, NeoX RoPE, qwen2-style biases, qwen3-style Q/K
//! head norms, SwiGLU) through the real upload + static-graph runner.
//! Skips on machines without a CUDA device.

use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;
use glcuda::model::{
    GpuModel, GpuModelConfig, HostLayer, HostMat, HostModel, HostWeight, RopeStyle,
};
use glcuda::sampler::{Sampler, SamplerConfig};

use glproc::model::{
    GateUp, GlprocModel, LayerWeights, ModelConfig, QkvWeights, RopeStyle as CpuRopeStyle,
    WeightMatrix,
};
use glproc::runner::Runner;

const DIM: usize = 8;
const N_LAYERS: usize = 2;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const HIDDEN: usize = 16;
const VOCAB: usize = 16;
const Q_DIM: usize = N_HEADS * HEAD_DIM;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;

fn gpu() -> Option<(Cuda, KernelSet)> {
    if !cuda_available() {
        eprintln!("SKIP: no CUDA driver/device on this machine");
        return None;
    }
    let cuda = Cuda::probe().expect("driver reported available; probe must succeed");
    let kernels = KernelSet::load(&cuda).expect("PTX must JIT on sm_70+");
    Some((cuda, kernels))
}

/// Deterministic pseudo-random weights in [-0.1, 0.1] — same generator as
/// glproc's runner tests, so a given seed always yields identical tensors
/// for both engines.
fn weights(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
                - 0.5)
                * 0.2
        })
        .collect()
}

/// Norm gains near 1.0 (still deterministic per seed).
fn gain(n: usize, seed: u64) -> Vec<f32> {
    weights(n, seed).iter().map(|w| 1.0 + w).collect()
}

// Per-tensor seeds, shared by both builders.
const S_EMBD: u64 = 99;
const S_ONORM: u64 = 303;
fn seeds(i: u64) -> [u64; 13] {
    [
        11 + i,  // wq
        22 + i,  // wk
        33 + i,  // wv
        44 + i,  // wo
        55 + i,  // gate
        66 + i,  // up
        77 + i,  // down
        101 + i, // bq
        102 + i, // bk
        103 + i, // bv
        201 + i, // q_norm
        202 + i, // k_norm
        301 + i, // attn_norm (ffn_norm uses +1000)
    ]
}

/// The exercise model: GQA + NeoX RoPE + QKV biases + Q/K head norms +
/// SwiGLU — every optional path the runner has.
fn cpu_model() -> GlprocModel {
    let layers = (0..N_LAYERS as u64)
        .map(|i| {
            let s = seeds(i);
            LayerWeights {
                attn_norm: gain(DIM, s[12]),
                qkv: QkvWeights::Split(
                    WeightMatrix::F32(weights(Q_DIM * DIM, s[0])),
                    WeightMatrix::F32(weights(KV_DIM * DIM, s[1])),
                    WeightMatrix::F32(weights(KV_DIM * DIM, s[2])),
                ),
                wo: WeightMatrix::F32(weights(DIM * Q_DIM, s[3])),
                bq: Some(weights(Q_DIM, s[7])),
                bk: Some(weights(KV_DIM, s[8])),
                bv: Some(weights(KV_DIM, s[9])),
                q_norm: Some(gain(HEAD_DIM, s[10])),
                k_norm: Some(gain(HEAD_DIM, s[11])),
                ffn_norm: gain(DIM, s[12] + 1000),
                gate_up: GateUp::Split(
                    WeightMatrix::F32(weights(HIDDEN * DIM, s[4])),
                    WeightMatrix::F32(weights(HIDDEN * DIM, s[5])),
                ),
                w_down: WeightMatrix::F32(weights(DIM * HIDDEN, s[6])),
            }
        })
        .collect();
    GlprocModel {
        config: ModelConfig {
            arch: "qwen2".into(),
            dim: DIM,
            n_layers: N_LAYERS,
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            hidden_dim: HIDDEN,
            vocab_size: VOCAB,
            max_seq: 64,
            rms_eps: 1e-5,
            rope_freq_base: 10_000.0,
            rope_style: CpuRopeStyle::Neox,
        },
        token_embd: WeightMatrix::F32(weights(VOCAB * DIM, S_EMBD)),
        layers,
        output_norm: gain(DIM, S_ONORM),
        output: WeightMatrix::F32(weights(VOCAB * DIM, S_EMBD)), // tied
    }
}

fn host_model() -> HostModel {
    let m = |n_out: usize, n_in: usize, seed: u64| HostMat {
        w: HostWeight::F32(weights(n_out * n_in, seed)),
        out_dim: n_out,
        in_dim: n_in,
    };
    let layers = (0..N_LAYERS as u64)
        .map(|i| {
            let s = seeds(i);
            HostLayer {
                attn_norm: gain(DIM, s[12]),
                wq: m(Q_DIM, DIM, s[0]),
                wk: m(KV_DIM, DIM, s[1]),
                wv: m(KV_DIM, DIM, s[2]),
                wo: m(DIM, Q_DIM, s[3]),
                bq: Some(weights(Q_DIM, s[7])),
                bk: Some(weights(KV_DIM, s[8])),
                bv: Some(weights(KV_DIM, s[9])),
                q_norm: Some(gain(HEAD_DIM, s[10])),
                k_norm: Some(gain(HEAD_DIM, s[11])),
                ffn_norm: gain(DIM, s[12] + 1000),
                w_gate: m(HIDDEN, DIM, s[4]),
                w_up: m(HIDDEN, DIM, s[5]),
                w_down: m(DIM, HIDDEN, s[6]),
            }
        })
        .collect();
    HostModel {
        config: GpuModelConfig {
            arch: "qwen2".into(),
            dim: DIM,
            n_layers: N_LAYERS,
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            hidden_dim: HIDDEN,
            vocab_size: VOCAB,
            max_seq: 64,
            rms_eps: 1e-5,
            rope_freq_base: 10_000.0,
            rope_style: RopeStyle::Neox,
        },
        token_embd: HostWeight::F32(weights(VOCAB * DIM, S_EMBD)),
        layers,
        output_norm: gain(DIM, S_ONORM),
        output: m(VOCAB, DIM, S_EMBD), // tied
    }
}

/// Logit tolerance across 2 full layers. Looser than the per-op ε because
/// the *CPU* dispatched path contributes its own approximations (AVX2
/// fast_exp in attention softmax, ~1e-4 relative) that the GPU does not
/// replicate; the argmax check below is the contract that matters.
const LOGIT_EPS: f32 = 1e-3;

#[test]
fn forward_logits_match_glproc() {
    let Some((cuda, k)) = gpu() else { return };
    let cpu = cpu_model();
    let mut cpu_run = Runner::new(&cpu);
    let mut gpu_model = GpuModel::upload(&cuda, host_model()).unwrap();

    let prompt = [1u32, 2, 3, 7, 4];
    for (pos, &tok) in prompt.iter().enumerate() {
        cpu_run.forward_into(tok, pos).unwrap();
        gpu_model.step(&cuda, &k, tok, pos, pos + 1 == prompt.len()).unwrap();
    }
    let want = cpu_run.logits().to_vec();
    let got = gpu_model.logits_host(&cuda).unwrap().to_vec();
    gpu_model.free(&cuda).unwrap();

    assert_eq!(got.len(), VOCAB);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= LOGIT_EPS,
            "logit {i}: gpu {g} vs cpu {w} (|diff| {})",
            (g - w).abs()
        );
    }
    // The user-facing contract: both engines pick the same next token.
    assert_eq!(
        Sampler::greedy(&got),
        glproc::sampler::Sampler::greedy(&want),
        "argmax diverged between engines"
    );
}

#[test]
fn greedy_generation_matches_glproc() {
    let Some((cuda, k)) = gpu() else { return };
    let cpu = cpu_model();
    let mut cpu_run = Runner::new(&cpu);
    let mut gpu_model = GpuModel::upload(&cuda, host_model()).unwrap();

    let greedy_cfg = || SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repeat_penalty: 1.0,
        seed: Some(1),
    };
    let prompt = [1u32, 2, 3];

    let mut cpu_sampler = glproc::sampler::Sampler::new(glproc::sampler::SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repeat_penalty: 1.0,
        seed: Some(1),
    });
    let (cpu_tokens, _) = cpu_run
        .generate(&prompt, 8, &mut cpu_sampler, |_| false, |_| {})
        .unwrap();

    let mut streamed = Vec::new();
    let (gpu_tokens, timing) = gpu_model
        .generate(&cuda, &k, &prompt, 8, &mut Sampler::new(greedy_cfg()), |_| false, |t| {
            streamed.push(t)
        })
        .unwrap();
    gpu_model.free(&cuda).unwrap();

    assert_eq!(gpu_tokens, cpu_tokens, "greedy continuations diverged");
    assert_eq!(streamed, gpu_tokens, "stream callback must see every token");
    assert_eq!(timing.prompt_tokens, 3);
    assert!(timing.prefill > std::time::Duration::ZERO);
    assert!(timing.decode > std::time::Duration::ZERO);
}

#[test]
fn generate_twice_reuses_cache_deterministically() {
    // The device KV cursor must reset cleanly between conversations.
    let Some((cuda, k)) = gpu() else { return };
    let mut gpu_model = GpuModel::upload(&cuda, host_model()).unwrap();
    let greedy = || {
        Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: Some(1),
        })
    };
    let (a, _) =
        gpu_model.generate(&cuda, &k, &[1, 2, 3], 5, &mut greedy(), |_| false, |_| {}).unwrap();
    let (b, _) =
        gpu_model.generate(&cuda, &k, &[1, 2, 3], 5, &mut greedy(), |_| false, |_| {}).unwrap();
    gpu_model.free(&cuda).unwrap();
    assert_eq!(a, b);
}

#[test]
fn invalid_token_and_empty_prompt_error_cleanly() {
    let Some((cuda, k)) = gpu() else { return };
    let mut gpu_model = GpuModel::upload(&cuda, host_model()).unwrap();
    let mut s = Sampler::new(SamplerConfig::default());
    assert!(gpu_model.generate(&cuda, &k, &[], 5, &mut s, |_| false, |_| {}).is_err());
    assert!(gpu_model.step(&cuda, &k, 9999, 0, true).is_err());
    gpu_model.free(&cuda).unwrap();
}
