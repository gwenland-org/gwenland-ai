//! Real text-in, text-out generation through a converted `.gllm` package —
//! the E2E gap flagged in `architecture/Pridwen-proposal-v5.md` §14 Phase 2.
//!
//! [`GllmEngine`] already does real embedding lookup, LM head, prefill,
//! decode, and sampling over [`GllmRuntime`] + [`GlprocBackend`] — but it
//! has no tokenizer (`.gllm` does not package one yet, ARTX1 OQ3), so it
//! only accepts/returns token ids. This example supplies the missing half:
//! it tokenizes a real prompt and detokenizes the output using
//! [`glcore::tokenizer::GllmTokenizer`] loaded from the *original* GGUF the
//! package was converted from — the same source-of-tokenizer-truth
//! `glcli`'s `gwen run` uses, just pointed at a sibling `.gllm` package for
//! the actual math instead of running the GGUF directly.
//!
//! ```text
//! cargo run -p glictus-caliburni --release --features glproc-backend \
//!     --example run_package_e2e -- <original.gguf> <gllm_package_dir> \
//!     [--prompt "..."] [--max-tokens N]
//! ```

use glcore::engine_trait::{GlEngine, InferInput};
use glcore::format::gguf::GgufFile;
use glcore::tokenizer::GllmTokenizer;
use glictus_caliburni::runtime::GllmEngine;

struct Args {
    gguf_path: String,
    package_dir: String,
    prompt: String,
    max_tokens: usize,
}

fn parse_args() -> Args {
    let mut positional = Vec::new();
    let mut prompt = "Explain what a hash table is in two sentences.".to_string();
    let mut max_tokens = 64usize;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--prompt" => prompt = it.next().expect("--prompt needs a value"),
            "--max-tokens" => {
                max_tokens = it
                    .next()
                    .expect("--max-tokens needs a value")
                    .parse()
                    .expect("--max-tokens must be an integer");
            }
            other => positional.push(other.to_string()),
        }
    }

    if positional.len() != 2 {
        eprintln!(
            "usage: run_package_e2e <original.gguf> <gllm_package_dir> [--prompt \"...\"] [--max-tokens N]"
        );
        std::process::exit(2);
    }

    Args { gguf_path: positional.remove(0), package_dir: positional.remove(0), prompt, max_tokens }
}

fn main() {
    let args = parse_args();

    eprintln!("loading tokenizer from {} ...", args.gguf_path);
    let gguf = GgufFile::open(&args.gguf_path).expect("open source GGUF for tokenizer");
    let tokenizer = GllmTokenizer::from_gguf_path(&gguf_path).expect("build tokenizer from GGUF metadata");

    eprintln!("loading .gllm package from {} ...", args.package_dir);
    let mut engine = GllmEngine::new();
    engine.load_model(&args.package_dir).expect("load .gllm package");

    let prompt_ids = tokenizer.encode_chat(&args.prompt).unwrap_or_else(|| tokenizer.encode(&args.prompt, true));
    eprintln!("prompt: {:?} -> {} tokens", args.prompt, prompt_ids.len());

    let input = InferInput {
        token_ids: prompt_ids.clone(),
        max_new_tokens: args.max_tokens,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        repeat_penalty: 1.1,
        trace: Default::default(),
        // Unioned inside GllmEngine with whatever the .gllm package's own
        // manifest declares (see gllm_engine.rs's run_request) — supplying
        // it here too matters specifically for older packages converted
        // before manifest-level EOS existed, or ones whose source GGUF
        // metadata was less complete than this tokenizer's own resolve.
        stopping: glcore::stopping::StoppingCriteria::from_tokenizer(&tokenizer),
    };

    let output = engine.infer(input).expect("run request through GllmRuntime + GlprocBackend");

    let text = tokenizer.decode(&output.token_ids, true);
    println!("{text}");

    let prefill_tps = output.prompt_tokens as f64 / (output.prefill_ms / 1000.0);
    let decode_tps = output.tokens_generated as f64 / (output.generation_ms / 1000.0);
    println!(
        "\n[benchmark] prefill: {} tokens @ {:.2} tok/s | generation: {} tokens @ {:.2} tok/s",
        output.prompt_tokens, prefill_tps, output.tokens_generated, decode_tps
    );
}
