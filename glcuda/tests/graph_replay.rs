//! Graph replay against individual launches: the captured decode graph must
//! produce exactly what the same kernels produce when issued one at a time.
//!
//! `parity.rs` already proves the capture/replay *mechanism* works on three
//! `add` kernels, and `forward.rs` proves the graph path agrees with glproc
//! end to end. Neither pins the thing that actually goes wrong when a graph
//! is captured badly: the graph and the individual-launch path silently
//! disagreeing on a real decode step. That is what this file tests.
//!
//! The two arms differ in ONE respect — how the kernels are submitted — so
//! the assertion is bit-exact equality, not a tolerance. Identical kernels
//! over identical buffers in identical order have no licence to differ, and
//! a tolerance here would hide exactly the bugs the test exists to catch
//! (a stale capture, a per-token scalar baked into the graph, a buffer that
//! moved between capture and replay).
//!
//! Two model shapes, because the two paths through `record_forward` are not
//! the same code:
//!   * f32 weights   — no activation quantize; covers the batched per-head
//!                     `rms_norm_rows` call.
//!   * Q8_0 weights  — exercises the ONE shared `quantize_q8` that q/k/v now
//!                     divide between them, inside the captured region.
//!
//! Skips (does not fail) with a note on machines without a CUDA device.

use glcuda::driver::{cuda_available, Cuda};
use glcuda::kernels::KernelSet;
use glcuda::model::{
    GpuModel, GpuModelConfig, HostLayer, HostMat, HostModel, HostWeight, RopeStyle,
};
use glcuda::repack::f32_to_q8_0_soa;

// Every in_dim is a multiple of 32 so the Q8_0 SoA path is legal for every
// projection (the format's own block invariant), and every out_dim is a
// multiple of 8 for the SoA GEMV's row tiling.
const DIM: usize = 32;
const N_LAYERS: usize = 2;
const N_HEADS: usize = 2;
const N_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const HIDDEN: usize = 32;
const VOCAB: usize = 32;
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
/// `forward.rs`, so a seed always yields the same tensor.
fn weights(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                * 0.2
        })
        .collect()
}

/// Norm gains near 1.0 (still deterministic per seed).
fn gain(n: usize, seed: u64) -> Vec<f32> {
    weights(n, seed).iter().map(|w| 1.0 + w).collect()
}

/// Which weight encoding the model under test uses.
#[derive(Clone, Copy, PartialEq)]
enum Enc {
    /// Dense f32: `consumes_q8_act` is false, no activation quantize runs.
    F32,
    /// Q8_0 SoA: every projection reads the shared int8 activation scratch.
    Q8,
}

/// One weight matrix in the requested encoding. The f32 values are identical
/// either way, so the two arms of a test see the same numbers.
fn mat(out_dim: usize, in_dim: usize, seed: u64, enc: Enc) -> HostMat {
    let v = weights(out_dim * in_dim, seed);
    let w = match enc {
        Enc::F32 => HostWeight::F32(v),
        Enc::Q8 => {
            let (qs, scales) = f32_to_q8_0_soa(&v);
            HostWeight::Q8_0Soa { qs, scales }
        }
    };
    HostMat { w, out_dim, in_dim }
}

/// A tiny but structurally complete model: GQA attention, NeoX RoPE, qwen2
/// biases and qwen3 per-head Q/K norms, SwiGLU FFN. The per-head norms are
/// what put the batched `rms_norm_rows` call inside the captured region.
fn host_model(enc: Enc) -> HostModel {
    let layers = (0..N_LAYERS as u64)
        .map(|i| HostLayer {
            attn_norm: gain(DIM, 500 + i),
            wq: mat(Q_DIM, DIM, 11 + i, enc),
            wk: mat(KV_DIM, DIM, 21 + i, enc),
            wv: mat(KV_DIM, DIM, 31 + i, enc),
            wo: mat(DIM, Q_DIM, 41 + i, enc),
            bq: Some(weights(Q_DIM, 51 + i)),
            bk: Some(weights(KV_DIM, 61 + i)),
            bv: Some(weights(KV_DIM, 71 + i)),
            q_norm: Some(gain(HEAD_DIM, 81 + i)),
            k_norm: Some(gain(HEAD_DIM, 91 + i)),
            ffn_norm: gain(DIM, 600 + i),
            w_gate_up: mat(2 * HIDDEN, DIM, 101 + i, enc),
            w_down: mat(DIM, HIDDEN, 111 + i, enc),
        })
        .collect();
    HostModel {
        config: GpuModelConfig {
            arch: "qwen3".into(),
            dim: DIM,
            n_layers: N_LAYERS,
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            head_dim: HEAD_DIM,
            hidden_dim: HIDDEN,
            vocab_size: VOCAB,
            max_seq: 128,
            rms_eps: 1e-5,
            rope_freq_base: 10_000.0,
            rope_style: RopeStyle::Neox,
        },
        // The embedding is dequantized host-side, so it stays f32 in both
        // arms; the encoding under test is the one on the projections.
        token_embd: HostWeight::F32(weights(VOCAB * DIM, 7)),
        layers,
        output_norm: gain(DIM, 700),
        output: mat(VOCAB, DIM, 13, enc),
    }
}

const PROMPT: [u32; 3] = [1, 5, 2];
/// Three decode tokens, so the graph is replayed twice AFTER the capture
/// replay — the repeats are what prove `pos`/`cached_len` are read from
/// device memory each time rather than frozen into the graph.
const DECODE: [u32; 3] = [3, 6, 4];

/// Drive `model` through the prompt and then the decode tokens, returning
/// the logits after each decode token. `graph` picks which submission path
/// the decode steps take; the prompt is always the individual-launch
/// `step`, so both arms enter decode from an identical KV cache.
fn run(cuda: &Cuda, k: &KernelSet, enc: Enc, graph: bool) -> Vec<Vec<f32>> {
    let mut m = GpuModel::upload(cuda, host_model(enc)).unwrap();
    for (pos, &tok) in PROMPT.iter().enumerate() {
        m.step(cuda, k, tok, pos, pos + 1 == PROMPT.len()).unwrap();
    }
    let mut out = Vec::new();
    for (i, &tok) in DECODE.iter().enumerate() {
        let pos = PROMPT.len() + i;
        if graph {
            m.decode_step(cuda, k, tok, pos).unwrap();
        } else {
            m.step(cuda, k, tok, pos, true).unwrap();
        }
        out.push(m.logits_host(cuda).unwrap().to_vec());
    }
    m.free(cuda).unwrap();
    out
}

/// Compare the two arms token by token, bit for bit.
fn assert_arms_agree(individual: &[Vec<f32>], replayed: &[Vec<f32>], what: &str) {
    assert_eq!(individual.len(), replayed.len(), "{what}: token count differs");
    for (t, (a, b)) in individual.iter().zip(replayed).enumerate() {
        assert_eq!(a.len(), VOCAB, "{what}: token {t} logit count");
        assert_eq!(b.len(), VOCAB, "{what}: token {t} logit count");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: decode token {t}, logit {i}: individual {x} vs graph replay {y}. \
                 The two paths run the same kernels over the same buffers in the same \
                 order, so any difference is the capture disagreeing with the code that \
                 was captured -- a stale graph, a per-token scalar baked in at capture \
                 time, or a buffer that moved.",
            );
        }
    }
}

/// The f32 arm: no activation quantize anywhere, so what this pins is the
/// captured region's structure — including the batched per-head Q/K norm.
#[test]
fn graph_replay_matches_individual_launches_f32() {
    let Some((cuda, k)) = gpu() else { return };
    let individual = run(&cuda, &k, Enc::F32, false);
    let replayed = run(&cuda, &k, Enc::F32, true);
    assert_arms_agree(&individual, &replayed, "f32");
}

/// The Q8_0 arm: every projection reads the shared int8 activation scratch,
/// so this is the arm that covers the single hoisted `quantize_q8` feeding
/// q, k and v. If the hoist left the scratch stale for any of the three, the
/// graph and individual paths still agree (both are wrong the same way) --
/// so this arm's real force is combined with `forward.rs`, which checks the
/// numbers against glproc rather than against ourselves.
#[test]
fn graph_replay_matches_individual_launches_q8() {
    let Some((cuda, k)) = gpu() else { return };
    let individual = run(&cuda, &k, Enc::Q8, false);
    let replayed = run(&cuda, &k, Enc::Q8, true);
    assert_arms_agree(&individual, &replayed, "q8_0");
}

/// A replayed graph must advance with the KV cursor. Decoding the same token
/// id three times in a row must NOT give the same logits three times: each
/// replay sees one more cached key/value, which is only true if `pos` and
/// `cached_len` are read from device memory per replay instead of captured.
///
/// This is the direct test for "per-token scalars are not baked into the
/// graph". It needs no reference implementation — a graph with a frozen
/// position produces identical rows and fails here.
#[test]
fn replays_see_updated_position_not_the_captured_one() {
    let Some((cuda, k)) = gpu() else { return };
    let mut m = GpuModel::upload(&cuda, host_model(Enc::F32)).unwrap();
    for (pos, &tok) in PROMPT.iter().enumerate() {
        m.step(&cuda, &k, tok, pos, pos + 1 == PROMPT.len()).unwrap();
    }
    let repeated = 3u32;
    let mut rows = Vec::new();
    for i in 0..3 {
        m.decode_step(&cuda, &k, repeated, PROMPT.len() + i).unwrap();
        rows.push(m.logits_host(&cuda).unwrap().to_vec());
    }
    m.free(&cuda).unwrap();

    for (a, b) in [(0usize, 1usize), (1, 2)] {
        assert!(
            rows[a].iter().zip(&rows[b]).any(|(x, y)| x.to_bits() != y.to_bits()),
            "replay {a} and {b} produced identical logits for the same token id. \
             The KV cache grew between them, so the position the graph used did \
             not -- it was frozen at capture time.",
        );
    }
}

/// Graph capture is an optimization, not a requirement. When the driver
/// exports the graph API the decode path replays; when it does not, it must
/// still decode by issuing kernels individually. This asserts the two agree
/// on which mode is in force, so the fallback cannot rot unnoticed.
#[test]
fn decode_works_in_whichever_mode_the_driver_supports() {
    let Some((cuda, k)) = gpu() else { return };
    let graphs = cuda.graphs_available();
    eprintln!(
        "graph API {} on this driver -- decode takes the {} path",
        if graphs { "available" } else { "MISSING" },
        if graphs { "captured-replay" } else { "individual-launch fallback" },
    );

    // Whichever branch decode_step takes internally, it must produce the
    // same logits as the explicitly individual-launch `step`.
    let individual = run(&cuda, &k, Enc::F32, false);
    let decode_path = run(&cuda, &k, Enc::F32, true);
    assert_arms_agree(&individual, &decode_path, "driver-selected decode path");
}
