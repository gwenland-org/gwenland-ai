//! Wave 118 in-process production stability harness.
//!
//! One model, runner, CUDA primary context, and allocation alternate the
//! retained and deferred-residual paths. Common production flags are supplied
//! by the caller before process start; the causal switch is explicit here.

use glcore::engine_trait::{GlEngine, InferInput};
use glcuda::{GlcudaConfig, GlcudaEngine};
use glproc::GlprocEngine;

const PROMPT_UNIT: &str = "Measure this deterministic systems prompt carefully. Explain how token-parallel integer matrix multiplication uses shared memory, Tensor Cores, and fixed launch geometry. ";
const QUARTETS: usize = 50;
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

fn run_arm(
    engine: &GlcudaEngine,
    ids: &[u32],
    candidate: bool,
    oracle: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    engine.set_benchmark_defer_ffn_residual(candidate)?;
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
        .ok_or("usage: wave118_in_process_stability MODEL [a|b]")?;
    let mode = args.next();
    let reverse = matches!(mode.as_deref(), Some("b"));
    let profile_once = matches!(mode.as_deref(), Some("profile"));

    let prompt = PROMPT_UNIT.repeat(8);
    let mut gpu = GlcudaEngine::with_config(GlcudaConfig {
        seed: Some(42),
        benchmark_defer_ffn_residual: Some(false),
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

    // Observation-only entry point for Wave 119. Keep one representative
    // candidate inference in the process so an external profiler can select
    // its production kernels without replaying the 200-sample stability run.
    if profile_once {
        let prefill_ms = run_arm(&gpu, &ids, true, oracle)?;
        let telemetry = gpu
            .telemetry()
            .and_then(|t| t.prefill)
            .ok_or("Wave 120 prefill telemetry unavailable")?;
        println!(
            "[wave120-profile] {{\"prompt_tokens\":{},\"host_prefill_ms\":{:.9},\"gpu_prefill_ms\":{:.9},\"gpu_prefill_tps\":{:.6},\"oracle_token\":{}}}",
            ids.len(),
            prefill_ms,
            telemetry.total_ms,
            ids.len() as f64 * 1000.0 / telemetry.total_ms,
            oracle,
        );
        for stage in telemetry.stages {
            println!(
                "[wave120-stage] {{\"name\":\"{}\",\"total_ms\":{:.9},\"calls\":{},\"bytes_read\":{},\"macs\":{}}}",
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
    for quartet in 0..QUARTETS {
        for (position, candidate) in order.into_iter().enumerate() {
            let prefill_ms = run_arm(&gpu, &ids, candidate, oracle)?;
            let arm = if candidate { "candidate" } else { "retained" };
            println!(
                "[wave118-sample] {{\"invocation\":\"{}\",\"quartet\":{},\"position\":{},\"arm\":\"{}\",\"prompt_tokens\":{},\"prefill_ms\":{:.9},\"prefill_tps\":{:.6},\"oracle_token\":{}}}",
                invocation,
                quartet,
                position,
                arm,
                ids.len(),
                prefill_ms,
                ids.len() as f64 * 1000.0 / prefill_ms,
                oracle,
            );
        }
    }
    println!(
        "[wave118-verdict-input] {{\"invocation\":\"{}\",\"quartets\":{},\"samples_per_arm\":{},\"prompt_tokens\":{},\"oracle\":\"{}/{}\"}}",
        invocation,
        QUARTETS,
        QUARTETS * 2,
        ids.len(),
        QUARTETS * 4,
        QUARTETS * 4,
    );
    gpu.shutdown();
    Ok(())
}
