/// Token generation speed benchmark via the native inference proxy.
///
/// Targets POST /gwenland/chat on the default proxy port (1136) — the same
/// endpoint that `gwen chat` and the GUI use. Benchmarking this path measures
/// the throughput users actually observe, including proxy overhead.
///
/// The benchmark returns None when the proxy is not running so the suite
/// can print a skip warning instead of failing the whole run.
///
/// Why a fixed prompt?
/// A fixed prompt ensures the benchmark is reproducible across runs and
/// machines. "Explain recursion in one paragraph" is a standard LLM benchmark
/// prompt that produces 60–200 token responses on most models.
///
/// Why not streaming measurement?
/// Measuring TTFT (time-to-first-token) and inter-token latency would be more
/// accurate, but requires parsing the SSE stream incrementally. The single-shot
/// mode measures total wall time, which is what most users care about for
/// non-interactive generation. Streaming benchmarks can be added later.
use std::time::Instant;

use super::InferenceResult;

const BENCHMARK_PROMPT: &str = "Explain the concept of recursion in one paragraph.";

/// @DANGER: Do not change native inference proxy base URL. Because its a GwenLand identity,
/// it's port is 1136 by default, which the user will input in the GUI.
const PROXY_BASE: &str = "http://127.0.0.1:1136";

const DEFAULT_MODEL: &str = "qwen3-8b-q4_0";

// ── Wire types (native proxy format) ─────────────────────────────────────────

#[derive(serde::Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
}

#[derive(serde::Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<ChunkMessage>,
    #[serde(default)]
    done: bool,
}

#[derive(serde::Deserialize)]
struct ChunkMessage {
    #[serde(default)]
    content: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Attempt one inference request via the native proxy and return throughput stats.
/// Returns None when the proxy is not running (connection refused / timeout).
pub fn run_inference_bench() -> Option<InferenceResult> {
    if !is_proxy_reachable() {
        return None;
    }

    let url = format!("{}/gwenland/chat", PROXY_BASE);
    let body = ChatRequest {
        model: DEFAULT_MODEL,
        messages: vec![ChatMessage {
            role: "user",
            content: BENCHMARK_PROMPT,
        }],
        stream: false,
    };

    let t0 = Instant::now();

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .ok()?;

    let resp = client.post(&url).json(&body).send().ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let text = resp.text().ok()?;
    let elapsed_secs = t0.elapsed().as_secs_f64().max(1e-9);

    // Accumulate tokens from NDJSON chunks (stream=false still emits chunks).
    let mut total_content = String::new();
    for line in text.lines() {
        let data = line.strip_prefix("data: ").unwrap_or(line);
        if let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) {
            if let Some(msg) = chunk.message {
                total_content.push_str(&msg.content);
            }
        }
    }

    // 4-chars/token heuristic — same as eval/metrics.rs.
    let total_tokens = (total_content.len() / 4).max(1);
    let tokens_per_sec = total_tokens as f64 / elapsed_secs;

    Some(InferenceResult {
        tokens_per_sec,
        total_tokens,
        elapsed_secs,
        backend: "proxy".to_string(),
        model_file: None,
    })
}

// ── Mistral.rs in-process path ────────────────────────────────────────────────

#[cfg(feature = "mistralrs-backend")]
pub fn run_mistralrs_bench(model_path: &std::path::Path) -> Option<InferenceResult> {
    use crate::engine::inference::{InferenceBackend, InferParams, MistralRsBackend};

    let backend = MistralRsBackend::new();
    if let Err(e) = backend.load_model(model_path) {
        eprintln!("run_mistralrs_bench: load_model failed: {e}");
        return None;
    }

    let params = InferParams {
        max_tokens: 128,
        temperature: 0.0,
        ..InferParams::default()
    };

    // temperature=0.0 fails InferParams::validate(), but we pass directly to
    // infer() without calling validate() — the backend clamps it internally.
    let t0 = Instant::now();
    let result_text = match backend.infer(BENCHMARK_PROMPT, &params) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("run_mistralrs_bench: infer failed: {e}");
            let _ = backend.unload();
            return None;
        }
    };
    let elapsed_secs = t0.elapsed().as_secs_f64().max(1e-9);

    let total_tokens = (result_text.len() / 4).max(1);
    let tokens_per_sec = total_tokens as f64 / elapsed_secs;

    let basename = model_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned());

    let _ = backend.unload();

    Some(InferenceResult {
        tokens_per_sec,
        total_tokens,
        elapsed_secs,
        backend: "mistralrs".to_string(),
        model_file: basename,
    })
}

fn is_proxy_reachable() -> bool {
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let health_url = format!("{}/health", PROXY_BASE);
    client
        .get(&health_url)
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "mistralrs-backend")]
    #[test]
    fn test_run_mistralrs_bench_missing_model() {
        let result = super::run_mistralrs_bench(std::path::Path::new("nonexistent.gguf"));
        assert!(result.is_none());
    }
}
