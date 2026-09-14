//! Wave 123 in-process production A/B for Q8 no-store and stacked FFN gate/up.
//!
//! Both arms keep the retained Wave 118 deferred-residual path enabled. The
//! only causal switch is Wave 123: skipping f32 scratch writes for Q8-only
//! consumers and consuming the stacked gate/up GEMM layout directly.

use glcore::engine_trait::{GlEngine, InferInput};
use glcuda::{GlcudaConfig, GlcudaEngine};
use glproc::GlprocEngine;

const PROMPT_UNIT: &str = "Measure this deterministic systems prompt carefully. Explain how token-parallel integer matrix multiplication uses shared memory, Tensor Cores, and fixed launch geometry. ";
const QUARTETS: usize = 10;
const WARMUPS_PER_ARM: usize = 5;
const EXPECTED_PROMPT_TOKENS: usize = 244;

fn input(ids: &[u32]) -> InferInput {
    InferInput {
        token_ids: ids.to_vec(),
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 40,
        top_p: 0.95,
        repeat_penalty: 1.1,
        ..InferInput::default()
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn run_arm(
    engine: &GlcudaEngine,
    ids: &[u32],
    candidate: bool,
    oracle: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    engine.set_benchmark_defer_ffn_residual(true)?;
    engine.set_benchmark_q8_nostore(candidate, candidate)?;
    let out = engine.infer(input(ids))?;
    if out.prompt_tokens != EXPECTED_PROMPT_TOKENS || out.token_ids.as_slice() != [oracle] {
        return Err(format!(
            "oracle mismatch: arm={} prompt_tokens={} tokens={:?} expected=[{}]",
            if candidate { "candidate" } else { "retained" },
            out.prompt_tokens,
            out.token_ids,
            oracle,
        )
        .into());
    }
    Ok(out.prefill_ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: wave123_q8_nostore MODEL [a|b|profile]")?;
    let mode = args.next();
    let reverse = matches!(mode.as_deref(), Some("b"));
    let profile_once = matches!(mode.as_deref(), Some("profile"));

    let prompt = PROMPT_UNIT.repeat(8);
    let mut gpu = GlcudaEngine::with_config(GlcudaConfig {
        seed: Some(42),
        benchmark_defer_ffn_residual: Some(true),
        benchmark_q8_nostore: Some(false),
        benchmark_ffn_gate_up_stacked: Some(false),
    });
    gpu.init()?;
    gpu.load_model(&model)?;
    let ids = gpu.encode_chat(&prompt)?;
    if ids.len() != EXPECTED_PROMPT_TOKENS {
        return Err(format!(
            "prompt contract drift: {} != {}",
            ids.len(),
            EXPECTED_PROMPT_TOKENS
        )
        .into());
    }

    let mut cpu = GlprocEngine::new();
    cpu.init()?;
    cpu.load_model(&model)?;
    let oracle_out = cpu.infer(input(&ids))?;
    let oracle = *oracle_out
        .token_ids
        .first()
        .ok_or("glproc oracle generated no token")?;
    cpu.shutdown();

    for candidate in [false, true] {
        for _ in 0..WARMUPS_PER_ARM {
            run_arm(&gpu, &ids, candidate, oracle)?;
        }
    }

    if profile_once {
        let prefill_ms = run_arm(&gpu, &ids, true, oracle)?;
        let telemetry = gpu
            .telemetry()
            .and_then(|t| t.prefill)
            .ok_or("Wave 123 prefill telemetry unavailable")?;
        println!(
            "[wave123-profile] {{\"prompt_tokens\":{},\"host_prefill_ms\":{:.9},\"gpu_prefill_ms\":{:.9},\"gpu_prefill_tps\":{:.6},\"oracle_token\":{}}}",
            ids.len(),
            prefill_ms,
            telemetry.total_ms,
            ids.len() as f64 * 1000.0 / telemetry.total_ms,
            oracle,
        );
        for stage in telemetry.stages {
            println!(
                "[wave123-stage] {{\"name\":\"{}\",\"total_ms\":{:.9},\"calls\":{},\"bytes_read\":{},\"macs\":{}}}",
                stage.name,
                stage.total_ms,
                stage.calls,
                stage.bytes_read.unwrap_or(0),
                stage.macs.unwrap_or(0),
            );
        }
        gpu.shutdown();
        return Ok(());
    }

    let normal = [false, true, true, false];
    let reversed = [true, false, false, true];
    let order = if reverse { reversed } else { normal };
    let invocation = if reverse { "b" } else { "a" };
    let mut retained = Vec::with_capacity(QUARTETS * 2);
    let mut candidate = Vec::with_capacity(QUARTETS * 2);
    for quartet in 0..QUARTETS {
        for (position, arm_candidate) in order.into_iter().enumerate() {
            let prefill_ms = run_arm(&gpu, &ids, arm_candidate, oracle)?;
            let arm = if arm_candidate {
                "candidate"
            } else {
                "retained"
            };
            println!(
                "[wave123-sample] {{\"invocation\":\"{}\",\"quartet\":{},\"position\":{},\"arm\":\"{}\",\"prompt_tokens\":{},\"prefill_ms\":{:.9},\"prefill_tps\":{:.6},\"oracle_token\":{}}}",
                invocation,
                quartet,
                position,
                arm,
                ids.len(),
                prefill_ms,
                ids.len() as f64 * 1000.0 / prefill_ms,
                oracle,
            );
            if arm_candidate {
                candidate.push(prefill_ms);
            } else {
                retained.push(prefill_ms);
            }
        }
    }
    let retained_median = median(retained.clone());
    let candidate_median = median(candidate.clone());
    println!(
        "[wave123-summary] {{\"invocation\":\"{}\",\"quartets\":{},\"samples_per_arm\":{},\"retained_median_ms\":{:.9},\"candidate_median_ms\":{:.9},\"retained_tps\":{:.6},\"candidate_tps\":{:.6},\"speedup\":{:.6},\"oracle_token\":{}}}",
        invocation,
        QUARTETS,
        retained.len(),
        retained_median,
        candidate_median,
        EXPECTED_PROMPT_TOKENS as f64 * 1000.0 / retained_median,
        EXPECTED_PROMPT_TOKENS as f64 * 1000.0 / candidate_median,
        retained_median / candidate_median,
        oracle,
    );
    gpu.shutdown();
    Ok(())
}
