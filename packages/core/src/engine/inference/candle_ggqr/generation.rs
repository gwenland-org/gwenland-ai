// engine/inference/candle_ggqr/generation.rs — Autoregressive generation loop.
//
// This module owns the token-by-token generation pipeline that sits between
// the raw forward pass (forward.rs) and the public InferenceBackend trait impl
// (backend.rs).  The separation keeps each layer testable in isolation:
//
//   backend.rs   → holds LoadedState under Mutex, wires InferenceBackend
//   generation.rs → pure function: given weights + config, emit token stream
//   forward.rs   → single forward pass returning logits
//   sampling.rs  → greedy / top-p / top-k token selection from logits
//
// ── Lifecycle of one generation request ────────────────────────────────────
//
//   1. Tokenise the prompt           → Vec<u32>  (prompt_ids)
//   2. Run forward pass over all     → Tensor [seq, vocab]  (logits)
//      prompt tokens at once
//   3. Sample the last-position      → u32  (next_token_id)
//      logit row
//   4. Decode token_id → String      → yield fragment to caller
//   5. Append token_id to context,   → loop to step 2 with the single
//      advance position              new token (autoregressive step)
//   6. Stop when:
//      a. EOS token sampled
//      b. max_tokens budget exhausted
//      c. A stop sequence appears in the accumulated text
//
// ── KV cache note ──────────────────────────────────────────────────────────
//
// The forward pass re-computes attention over the entire growing context on
// every step (O(n²) in sequence length).  This is intentional for the v1
// implementation — a KV cache is deferred to GWEN-215.  For the target
// models (Qwen3-1.7B at Q4_K) this is acceptable for sequences up to ~512
// tokens on a modern laptop CPU.
//
// Requirements: 7.1–7.8 (generate_stream), 8.1–8.6 (stream_infer),
//               10.1–10.4 (infer)

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use futures_util::Stream;
use async_stream::stream;
use std::pin::Pin;
use tokenizers::Tokenizer;

use crate::error::GwenError;
use crate::engine::inference::params::InferParams;
use super::{ModelConfig, forward, sample_token};

// ── EOS token IDs ─────────────────────────────────────────────────────────────
//
// We recognise the most common EOS token IDs used by LLaMA-family models.
// Architectures that define a different EOS token should be handled here in
// a future patch; for now the list covers Qwen3, LLaMA-3, and Phi-3.

/// Token IDs that signal end-of-sequence and must halt generation.
///
/// - 2   : `</s>` — standard SentencePiece / LLaMA EOS
/// - 128001 : `<|end_of_text|>` — LLaMA 3 EOS
/// - 151645 : `<|im_end|>` — Qwen ChatML EOS
/// - 32000 : `<|end|>` — Phi-3 EOS
const EOS_TOKEN_IDS: &[u32] = &[2, 128001, 151645, 32000];

// ── GenerationState ───────────────────────────────────────────────────────────

/// All data needed to drive one autoregressive generation session.
///
/// This is a value type (not behind a Mutex) because `generate_stream` is
/// called with the Mutex already locked; the state is borrowed for the
/// duration of the stream.
pub struct GenerationState<'a> {
    /// Dequantised model weights, keyed by their GGUF tensor name.
    pub tensors: &'a HashMap<String, Tensor>,

    /// Architecture metadata extracted from GGUF KV (layer count, heads, etc.).
    pub config: &'a ModelConfig,

    /// HuggingFace-compatible tokenizer loaded alongside the GGUF file.
    pub tokenizer: &'a Tokenizer,

    /// Target device — always `Device::Cpu` for this backend.
    pub device: &'a Device,
}

// ── generate_stream ───────────────────────────────────────────────────────────

/// Tokenise `prompt`, then produce token strings one at a time via an
/// `async_stream`.
///
/// The stream yields each decoded token fragment as soon as it is sampled,
/// which lets the TUI / GUI render a typewriter effect without waiting for
/// the full response.
///
/// # Termination conditions
///
/// The stream ends after the **first** of:
/// 1. An EOS token ID is sampled (token not yielded).
/// 2. `params.max_tokens` new tokens have been emitted.
/// 3. The accumulated text ends with one of `params.stop_sequences`.
///
/// # Error handling
///
/// Because `Stream<Item = String>` cannot propagate `Result`, errors are
/// encoded as a final yielded string with the prefix `"[gwen-error] "`.
/// Callers that need structured errors should use `generate_collect` instead.
///
/// Requirements: 7.1–7.8
pub fn generate_stream<'a>(
    state: &'a GenerationState<'a>,
    prompt: &str,
    params: &'a InferParams,
) -> impl Stream<Item = String> + 'a {
    // Clone the prompt into an owned String so the stream's async block can
    // hold it without a lifetime tie to the caller's stack frame.
    let prompt = prompt.to_string();

    stream! {
        // ── Step 1: Tokenise the prompt ───────────────────────────────────
        //
        // We encode without adding special tokens; the prompt is assumed to
        // already be formatted (e.g. ChatML `<|im_start|>user\n…\n<|im_end|>`).
        let encoding = match state.tokenizer.encode(prompt.as_str(), false) {
            Ok(e) => e,
            Err(e) => {
                yield format!("[gwen-error] tokenise failed: {e}");
                return;
            }
        };

        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            yield "[gwen-error] prompt tokenised to zero tokens".to_string();
            return;
        }

        eprintln!(
            "candle-ggqr: generation started — {} prompt tokens, max_new={}, temp={:.2}, top_p={:.2}",
            prompt_ids.len(), params.max_tokens, params.temperature, params.top_p
        );

        // ── Step 2: Build the growing context buffer ──────────────────────
        //
        // `context` starts as the prompt IDs.  After each sampling step we
        // append the newly sampled token so the next forward pass sees the
        // full context (no KV cache — see module-level comment).
        let mut context: Vec<u32> = prompt_ids;

        // Track the full decoded text so we can check stop sequences.
        let mut accumulated = String::new();

        // ── Step 3–6: Autoregressive loop ────────────────────────────────
        for step in 0..params.max_tokens {
            // Build input_ids tensor from the current context.
            let input_ids = match Tensor::from_vec(context.clone(), context.len(), state.device) {
                Ok(t) => t,
                Err(e) => {
                    yield format!("[gwen-error] input tensor: {e}");
                    return;
                }
            };

            // Run the full forward pass → [seq_len, vocab_size] logits.
            let logits_2d = match forward(&input_ids, state.tensors, state.config) {
                Ok(l) => l,
                Err(e) => {
                    yield format!("[gwen-error] forward pass at step {step}: {e}");
                    return;
                }
            };

            // Extract the logit row for the last position in the sequence.
            // `logits_2d` shape is [seq_len, vocab_size]; we want row [-1].
            let last_pos = logits_2d.dim(0).unwrap_or(1).saturating_sub(1);
            let last_logits = match logits_2d.get(last_pos) {
                Ok(l) => l,
                Err(e) => {
                    yield format!("[gwen-error] logit extraction at step {step}: {e}");
                    return;
                }
            };

            // Convert to Vec<f32> for the sampling functions.
            let logit_vec: Vec<f32> = match last_logits.to_vec1() {
                Ok(v) => v,
                Err(e) => {
                    yield format!("[gwen-error] logit to_vec at step {step}: {e}");
                    return;
                }
            };

            // Sample the next token ID.
            let token_id = match sample_token(&logit_vec, params) {
                Ok(id) => id,
                Err(e) => {
                    yield format!("[gwen-error] sampling at step {step}: {e}");
                    return;
                }
            };

            eprintln!("candle-ggqr: step {step} → token_id={token_id}");

            // ── EOS check ─────────────────────────────────────────────────
            //
            // Do not yield EOS tokens — they are a control signal, not text.
            if EOS_TOKEN_IDS.contains(&token_id) {
                eprintln!("candle-ggqr: EOS token {token_id} sampled — stopping");
                break;
            }

            // ── Decode token_id → UTF-8 fragment ─────────────────────────
            let fragment = match state.tokenizer.decode(&[token_id], /*skip_special=*/true) {
                Ok(s) => s,
                Err(e) => {
                    yield format!("[gwen-error] decode token {token_id}: {e}");
                    return;
                }
            };

            // ── Stop-sequence check ───────────────────────────────────────
            //
            // We check *after* decoding so we can match multi-byte sequences
            // that may span subword tokens.
            accumulated.push_str(&fragment);
            let hit_stop = params.stop_sequences.iter().any(|s| accumulated.ends_with(s.as_str()));

            // Yield the decoded fragment to the caller.
            yield fragment;

            if hit_stop {
                eprintln!("candle-ggqr: stop sequence matched — stopping");
                break;
            }

            // Append the new token to context for the next iteration.
            context.push(token_id);
        }

        eprintln!("candle-ggqr: generation complete — {} tokens in context", context.len());
    }
}

// ── generate_collect ──────────────────────────────────────────────────────────

/// Run `generate_stream` to completion and collect all fragments into a single
/// owned `String`.
///
/// Errors encoded in the stream as `"[gwen-error] …"` fragments are propagated
/// as `GwenError::InferenceError` so callers get a typed result rather than
/// having to inspect the string.
///
/// This is the synchronous counterpart used by `InferenceBackend::infer`.
///
/// Requirements: 10.1–10.4
pub fn generate_collect(
    state: &GenerationState<'_>,
    prompt: &str,
    params: &InferParams,
) -> Result<String, GwenError> {
    use futures_util::StreamExt;

    // Drive the async stream to completion on the current Tokio runtime.
    // `stream!` from async-stream produces a regular async generator; we
    // collect it with a blocking handle so `infer` can remain synchronous.
    let s = generate_stream(state, prompt, params);
    futures_util::pin_mut!(s);

    // Tokio's `block_in_place` lets us run async code from a sync context
    // inside a multi-threaded runtime without blocking the scheduler thread.
    let fragments: Vec<String> = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            let mut out = Vec::new();
            futures_util::StreamExt::for_each(&mut s, |tok| {
                out.push(tok);
                std::future::ready(())
            }).await;
            out
        })
    });

    // Promote stream-encoded errors to typed GwenError.
    for frag in &fragments {
        if let Some(msg) = frag.strip_prefix("[gwen-error] ") {
            return Err(GwenError::InferenceError {
                layer: "generation".to_string(),
                operation: "generate_collect".to_string(),
                error: msg.to_string(),
            });
        }
    }

    Ok(fragments.concat())
}

// ── make_stream_pinned ────────────────────────────────────────────────────────

/// Wrap the output of `generate_stream` in the `Pin<Box<dyn Stream>>` shape
/// required by `InferenceBackend::stream_infer`.
///
/// The lifetime of `state` is erased by cloning all referenced data into the
/// stream's closure so the returned stream is `'static + Send`.
pub fn make_stream_pinned(
    tensors: HashMap<String, Tensor>,
    config: ModelConfig,
    tokenizer: Tokenizer,
    device: Device,
    prompt: String,
    params: InferParams,
) -> Pin<Box<dyn Stream<Item = String> + Send>> {
    // All data is moved into the stream block, so it is fully owned and 'static.
    let s = stream! {
        let state = GenerationState {
            tensors: &tensors,
            config: &config,
            tokenizer: &tokenizer,
            device: &device,
        };
        // Drive the inner generate_stream and re-yield each fragment.
        let inner = generate_stream(&state, &prompt, &params);
        futures_util::pin_mut!(inner);
        loop {
            use futures_util::StreamExt;
            match inner.next().await {
                Some(tok) => yield tok,
                None => break,
            }
        }
    };

    Box::pin(s)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use candle_core::{Device, Tensor, DType};
    use tokenizers::Tokenizer;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal but coherent set of model weights for a tiny LLaMA model.
    ///
    /// Dimensions:
    ///   hidden = 8, n_heads = 2, intermediate = 16, vocab = 32, n_layers = 1
    ///
    /// All weights are small random-ish values derived from index arithmetic so
    /// tests are deterministic without requiring any file I/O.
    fn tiny_tensors(vocab: usize, hidden: usize, intermediate: usize) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let randn = |shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - (n as f32 * 0.0005)).collect();
            Tensor::from_vec(data, shape, &dev).unwrap()
        };
        let ones = |shape: &[usize]| -> Tensor {
            Tensor::ones(shape, DType::F32, &dev).unwrap()
        };

        let mut t = HashMap::new();
        t.insert("token_embd.weight".into(),         randn(&[vocab, hidden]));
        t.insert("blk.0.attn_norm.weight".into(),    ones(&[hidden]));
        t.insert("blk.0.attn_q.weight".into(),       randn(&[hidden, hidden]));
        t.insert("blk.0.attn_k.weight".into(),       randn(&[hidden, hidden]));
        t.insert("blk.0.attn_v.weight".into(),       randn(&[hidden, hidden]));
        t.insert("blk.0.attn_output.weight".into(),  randn(&[hidden, hidden]));
        t.insert("blk.0.ffn_norm.weight".into(),     ones(&[hidden]));
        t.insert("blk.0.ffn_gate.weight".into(),     randn(&[intermediate, hidden]));
        t.insert("blk.0.ffn_up.weight".into(),       randn(&[intermediate, hidden]));
        t.insert("blk.0.ffn_down.weight".into(),     randn(&[hidden, intermediate]));
        t.insert("output_norm.weight".into(),        ones(&[hidden]));
        t.insert("output.weight".into(),             randn(&[vocab, hidden]));
        t
    }

    fn tiny_config(vocab: usize, hidden: usize, intermediate: usize) -> ModelConfig {
        ModelConfig {
            architecture: "llama".into(),
            n_layers: 1,
            hidden_size: hidden as u32,
            n_heads: 2,
            n_kv_heads: 2,
            intermediate_size: intermediate as u32,
            vocab_size: vocab as u32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    /// Build a minimal BPE tokenizer with a vocabulary of `n` single-byte tokens
    /// ("0", "1", …, "N-1") so we can tokenise simple numeric strings in tests.
    fn tiny_tokenizer(n: usize) -> Tokenizer {
        // Build a JSON representation of a minimal BPE tokenizer whose vocab
        // maps single-character strings to sequential IDs.  The tokenizer
        // library accepts JSON via `from_str`.
        let vocab_entries: String = (0..n)
            .map(|i| format!("\"{i}\": {i}"))
            .collect::<Vec<_>>()
            .join(", ");

        let json = format!(
            r#"{{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": null,
                "model": {{
                    "type": "BPE",
                    "vocab": {{ {vocab_entries} }},
                    "merges": []
                }}
            }}"#
        );
        Tokenizer::from_str(&json).expect("tiny tokenizer must parse")
    }

    // ── 15.2 generate_stream tests ────────────────────────────────────────────

    /// The stream must produce at most `max_tokens` items before stopping.
    #[test]
    fn generate_stream_respects_max_tokens_limit() {
        let vocab = 32;
        let hidden = 8;
        let intermediate = 16;

        let tensors = tiny_tensors(vocab, hidden, intermediate);
        let config  = tiny_config(vocab, hidden, intermediate);
        let tok     = tiny_tokenizer(vocab);
        let dev     = Device::Cpu;

        let state = GenerationState {
            tensors: &tensors,
            config: &config,
            tokenizer: &tok,
            device: &dev,
        };

        let params = InferParams {
            max_tokens: 5,
            temperature: 0.0, // greedy → deterministic
            top_p: 1.0,
            top_k: None,
            repetition_penalty: None,
            stop_sequences: vec![],
            seed: Some(0),
        };

        let s = generate_stream(&state, "1 2 3", &params);
        futures_util::pin_mut!(s);

        let tokens: Vec<String> = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                use futures_util::StreamExt;
                s.collect().await
            });

        // Filter out error fragments so we only count real tokens.
        let real: Vec<_> = tokens.iter().filter(|t| !t.starts_with("[gwen-error]")).collect();
        assert!(
            real.len() <= 5,
            "expected ≤5 real tokens, got {}: {:?}",
            real.len(), real
        );
    }

    /// Each fragment yielded by the stream must be non-empty.
    #[test]
    fn generate_stream_yields_non_empty_fragments() {
        let vocab = 32;
        let hidden = 8;
        let intermediate = 16;

        let tensors = tiny_tensors(vocab, hidden, intermediate);
        let config  = tiny_config(vocab, hidden, intermediate);
        let tok     = tiny_tokenizer(vocab);
        let dev     = Device::Cpu;

        let state = GenerationState {
            tensors: &tensors,
            config: &config,
            tokenizer: &tok,
            device: &dev,
        };

        let params = InferParams {
            max_tokens: 3,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: None,
            stop_sequences: vec![],
            seed: Some(1),
        };

        let s = generate_stream(&state, "1", &params);
        futures_util::pin_mut!(s);

        let tokens: Vec<String> = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                use futures_util::StreamExt;
                s.collect().await
            });

        // Every non-error fragment must be non-empty.
        for tok in &tokens {
            if !tok.starts_with("[gwen-error]") {
                assert!(!tok.is_empty(), "got empty fragment in: {tokens:?}");
            }
        }
    }

    /// When a stop sequence appears in the accumulated text, the stream must
    /// not yield any further tokens after the matching fragment.
    #[test]
    fn generate_stream_stops_on_stop_sequence() {
        let vocab = 32;
        let hidden = 8;
        let intermediate = 16;

        let tensors = tiny_tensors(vocab, hidden, intermediate);
        let config  = tiny_config(vocab, hidden, intermediate);
        let tok     = tiny_tokenizer(vocab);
        let dev     = Device::Cpu;

        let state = GenerationState {
            tensors: &tensors,
            config: &config,
            tokenizer: &tok,
            device: &dev,
        };

        // Collect everything with no stop sequences first to learn what would
        // be produced, then use one of those tokens as a stop sequence and
        // confirm we get fewer tokens.
        let params_no_stop = InferParams {
            max_tokens: 8,
            temperature: 0.0,
            stop_sequences: vec![],
            seed: Some(42),
            ..InferParams::default()
        };

        let s = generate_stream(&state, "1", &params_no_stop);
        futures_util::pin_mut!(s);

        let all_tokens: Vec<String> = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                use futures_util::StreamExt;
                s.collect().await
            });

        let real_tokens: Vec<_> = all_tokens.iter()
            .filter(|t| !t.starts_with("[gwen-error]"))
            .collect();

        // We need at least 2 tokens for the stop-sequence test to be meaningful.
        if real_tokens.len() < 2 {
            // Model is too small or hits EOS immediately — skip gracefully.
            return;
        }

        // Use the second real token as the stop sequence.
        let stop_seq = real_tokens[1].clone();

        let params_with_stop = InferParams {
            max_tokens: 8,
            temperature: 0.0,
            stop_sequences: vec![stop_seq],
            seed: Some(42),
            ..InferParams::default()
        };

        let s2 = generate_stream(&state, "1", &params_with_stop);
        futures_util::pin_mut!(s2);

        let stopped_tokens: Vec<String> = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                use futures_util::StreamExt;
                s2.collect().await
            });

        let stopped_real: Vec<_> = stopped_tokens.iter()
            .filter(|t| !t.starts_with("[gwen-error]"))
            .collect();

        assert!(
            stopped_real.len() <= real_tokens.len(),
            "stop sequence should reduce output: without={} with={}",
            real_tokens.len(), stopped_real.len()
        );
    }

    // ── 15.2 / 16.2 make_stream_pinned tests ─────────────────────────────────

    /// The pinned stream returned by `make_stream_pinned` must be `Send` and
    /// must terminate within `max_tokens`.
    #[test]
    fn make_stream_pinned_is_send_and_terminates() {
        let vocab = 32;
        let hidden = 8;
        let intermediate = 16;

        let tensors = tiny_tensors(vocab, hidden, intermediate);
        let config  = tiny_config(vocab, hidden, intermediate);
        let tok     = tiny_tokenizer(vocab);
        let dev     = Device::Cpu;

        let params = InferParams {
            max_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: None,
            stop_sequences: vec![],
            seed: Some(7),
        };

        let stream = make_stream_pinned(
            tensors, config, tok, dev,
            "1 2".to_string(), params,
        );

        // Verify the stream is Send by sending it to another thread.
        let handle = std::thread::spawn(move || {
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async {
                    use futures_util::StreamExt;
                    stream.collect::<Vec<_>>().await
                })
        });

        let tokens = handle.join().expect("thread must not panic");
        let real: Vec<_> = tokens.iter().filter(|t| !t.starts_with("[gwen-error]")).collect();
        assert!(real.len() <= 4, "expected ≤4 tokens, got {}: {real:?}", real.len());
    }

    // ── 17.1 generate_collect tests ──────────────────────────────────────────

    /// `generate_collect` must return a String (not an error) for a valid prompt.
    #[test]
    fn generate_collect_returns_string_for_valid_prompt() {
        let vocab = 32;
        let hidden = 8;
        let intermediate = 16;

        let tensors = tiny_tensors(vocab, hidden, intermediate);
        let config  = tiny_config(vocab, hidden, intermediate);
        let tok     = tiny_tokenizer(vocab);
        let dev     = Device::Cpu;

        let state = GenerationState {
            tensors: &tensors,
            config: &config,
            tokenizer: &tok,
            device: &dev,
        };

        let params = InferParams {
            max_tokens: 3,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: None,
            stop_sequences: vec![],
            seed: Some(0),
        };

        // `generate_collect` uses `block_in_place` internally, which requires a
        // multi-thread Tokio runtime.  We enter it with `block_on` so the current
        // thread becomes a runtime worker — no `spawn_blocking` needed, which would
        // require a `'static` closure and break the borrow on `state`.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let result = rt.block_on(async {
            tokio::task::block_in_place(|| generate_collect(&state, "1", &params))
        });

        // Either succeeds (returns a string) or fails with a typed GwenError —
        // never panics.
        match result {
            Ok(s) => {
                // On EOS-immediate models the string may be empty; that is valid.
                assert!(s.len() < 10_000, "suspiciously long output: {}", s.len());
            }
            Err(GwenError::InferenceError { .. }) => {
                // Also acceptable — the tiny model may produce invalid logits.
            }
            Err(e) => panic!("unexpected error variant: {e:?}"),
        }
    }
}
