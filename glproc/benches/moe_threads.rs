//! Thread-scaling sweep for the MoE layer. Not a criterion bench (crate has
//! zero external deps) — a plain binary that times pool sizes directly.

use std::time::Instant;

use glproc::kernels::bridge::QuantFormat;
use glproc::kernels::dequant::q8_0;
use glproc::model::{GateUp, WeightMatrix};
use glproc::moe::{ExpertWeights, MoEConfig, MoELayer};
use glproc::simd_strategy::SimdStrategy;
use glproc::threading::ThreadPool;

fn prng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() {
    // Qwen3-30B-A3B's MoE block. 128 experts x (2*768*2048 + 2048*768) Q8_0
    // ~= 400 MB of expert weights — big enough that this is a real memory
    // workload, not an L2-resident toy.
    let (ne, k, h, f) = (128usize, 8usize, 2048usize, 768usize);
    let mut seed = 0xBEEFu64;

    eprintln!("building {ne} Q8_0 experts (~400 MB)...");
    let router: Vec<f32> = (0..ne * h).map(|_| prng(&mut seed)).collect();
    let experts: Vec<ExpertWeights> = (0..ne)
        .map(|_| {
            let gate: Vec<f32> = (0..f * h).map(|_| prng(&mut seed)).collect();
            let up: Vec<f32> = (0..f * h).map(|_| prng(&mut seed)).collect();
            let down: Vec<f32> = (0..h * f).map(|_| prng(&mut seed)).collect();
            let (gq, uq) = (q8_0::scalar::quantize(&gate), q8_0::scalar::quantize(&up));
            let row_bytes = gq.len() / f;
            let mut packed = Vec::with_capacity(gq.len() + uq.len());
            for o in 0..f {
                packed.extend_from_slice(&gq[o * row_bytes..(o + 1) * row_bytes]);
                packed.extend_from_slice(&uq[o * row_bytes..(o + 1) * row_bytes]);
            }
            ExpertWeights {
                gate_up: GateUp::FusedQuant(QuantFormat::Q8_0, packed),
                w_down: WeightMatrix::Quant(QuantFormat::Q8_0, q8_0::scalar::quantize(&down)),
            }
        })
        .collect();

    let layer = MoELayer {
        config: MoEConfig {
            num_experts: ne,
            num_experts_per_tok: k,
            expert_ffn_size: f,
            hidden_size: h,
            norm_topk_prob: true,
        },
        router,
        experts,
    };
    let strategy = SimdStrategy::detect();
    eprintln!("simd: {strategy:?}\n");

    let logical = num_cpus::get();
    println!("{:<8} {:>12} {:>12} {:>10}", "threads", "decode ms", "prefill32 ms", "speedup");
    println!("{}", "-".repeat(46));

    let mut base_decode = 0f64;
    for nt in 1..=logical {
        let pool = ThreadPool::new(nt);

        // decode: 1 token, top-8 experts -> matvec path
        let input: Vec<f32> = (0..h).map(|_| prng(&mut seed)).collect();
        let mut out = vec![0f32; h];
        for _ in 0..3 {
            layer.forward(&input, &mut out, 1, &pool, strategy); // warm
        }
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            layer.forward(&input, &mut out, 1, &pool, strategy);
        }
        let decode_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // prefill: 32 tokens -> experts batched, matmul path
        let binput: Vec<f32> = (0..32 * h).map(|_| prng(&mut seed)).collect();
        let mut bout = vec![0f32; 32 * h];
        for _ in 0..2 {
            layer.forward(&binput, &mut bout, 32, &pool, strategy); // warm
        }
        let iters = 5;
        let t = Instant::now();
        for _ in 0..iters {
            layer.forward(&binput, &mut bout, 32, &pool, strategy);
        }
        let prefill_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        if nt == 1 {
            base_decode = decode_ms;
        }
        println!(
            "{nt:<8} {decode_ms:>12.2} {prefill_ms:>12.2} {:>9.2}x",
            base_decode / decode_ms
        );
    }

    println!("\nphysical cores: {}", glproc::topology::physical_core_count());
    println!("logical threads: {logical}");
    println!("\nIf decode keeps improving past the physical-core count, SMT is");
    println!("filling issue slots and pool size should track LOGICAL threads.");
    println!("If it plateaus or regresses at physical, Fix 1 was right here and");
    println!("the i3 result was a 2-core artifact.");
}