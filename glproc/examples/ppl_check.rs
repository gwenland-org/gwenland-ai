//! Teacher-forced perplexity for glproc, over the SAME embedded WikiText-2
//! sample and the SAME sliding-window protocol `glbench ppl` uses for
//! `.gllm` packages (`glbench/src/ppl.rs`) — that command is hardcoded to
//! `GllmEngine`, not glproc/GGUF, so this is a standalone equivalent for
//! glproc, built specifically to answer "is glproc's compute path (Q8_0
//! repack, SIMD kernels) as numerically faithful as llama.cpp's, on the
//! exact same GGUF weights" — a runtime-precision question, not a
//! quantization-format one.
//!
//! Runs the sequential single-token decode path (`Runner::forward_into`) so
//! every position gets real logits — slower than batched prefill (no
//! weight-row reuse across positions), fine for a one-off measurement on a
//! short corpus, not something this file claims is decode-loop-representative
//! of glproc's normal throughput.
//!
//! Run: cargo run --release -p glproc --example ppl_check -- <model.gguf> [context] [stride]

use glcore::format::gguf::GgufFile;
use glcore::tokenizer::GllmTokenizer;
use glproc::loader::load_gguf;
use glproc::runner::Runner;

/// First article of the WikiText-2 test split (Valkyria Chronicles III) —
/// byte-for-byte the same text `glbench/src/ppl.rs::WIKITEXT2_SAMPLE` uses,
/// so a comparison against `.gllm`/glproc/llama.cpp all score the identical
/// corpus.
const WIKITEXT2_SAMPLE: &str = " Valkyria Chronicles III ( Japanese : \u{6226}\u{5834}\u{306e}\u{30f4}\u{30a1}\u{30eb}\u{30ad}\u{30e5}\u{30ea}\u{30a2}3 , lit . Sen j\u{014d} no Valkyria 3 : Gallian Chronicles ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the \" Nameless \" , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against an elite enemy unit known as \" Calamaty Raven \" .\n The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .\n It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 .\n As with the previous Valkyria Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially in voiced speech bubbles and partially in unvoiced , separate text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .\n The game 's battle system , the BliTZ system , is carried over directly from Valkyria Chronicles . During missions , players select each unit using a top @-@ down perspective of the battlefield map : once a character is selected , the player moves the character around the battlefield in third @-@ person . A character can only act once per @-@ turn , but characters can be granted multiple turns at the expense of other characters ' turns . Each character has a field and pursuit range , which determines the area of effect for regular attacks as well as counterattacks performed against them . Each side 's units can move only a limited number of times before a turn is over . If a unit is knocked out during a turn , they are sent back to the barracks and can no longer participate for the rest of the mission . These missions can be replayed as many times as they wish , and a Roman numeral system is used to indicate how far into the mission a mission is .\n";

fn log_softmax_at(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().fold(f32::MIN, |m, &v| m.max(v));
    let sum_exp: f64 = logits.iter().map(|&v| ((v - max) as f64).exp()).sum();
    let log_sum_exp = sum_exp.ln();
    (logits[target] - max) as f64 - log_sum_exp
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).ok_or("usage: ppl_check <model.gguf> [context] [stride]")?;
    let context: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(256);
    let stride: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(128);

    eprintln!("loading {model_path} ...");
    let gguf = GgufFile::open(model_path)?;
    let tokenizer = GllmTokenizer::from_gguf_path(model_path)?;
    let model = load_gguf(&gguf)?;

    // Respect the model's own `tokenizer.ggml.add_bos_token` metadata rather
    // than hardcoding — this is exactly the kind of methodology detail that,
    // if mismatched against whatever llama.cpp does, produces a perplexity
    // gap that looks like a precision difference but is actually two
    // different input sequences being scored.
    let add_bos = tokenizer.add_bos_default();
    eprintln!("add_bos (from tokenizer.ggml.add_bos_token): {add_bos}");
    let tokens = tokenizer.encode(WIKITEXT2_SAMPLE, add_bos)?;
    let total_tokens = tokens.len();
    eprintln!("first 20 token ids: {:?}", &tokens[..20.min(tokens.len())]);

    // ⚠️ The recorded baseline for this model (PPL 36.12, see
    // notes/issues/glproc-precision-gap-vs-llamacpp.md) was measured with the
    // tokenizer that `glcore::tokenizer` replaced. An old-vs-new A/B ran here
    // until step 4 deleted that implementation; its result is recorded rather
    // than re-derivable: on Qwen2.5 the ids came out **identical** (819/819 on
    // this corpus, 201/201 on the glbench prompt) and the perplexity landed at
    // 36.19 against the 36.12 baseline. So this number is comparable *for
    // Qwen2.5*. It is not for Llama-3 or SPM families, where the old
    // implementation scored 65.2%–97.8% and the ids necessarily moved.
    eprintln!("dataset: wikitext2-sample-embedded, {total_tokens} tokens, context={context} stride={stride}");
    if total_tokens <= context {
        return Err(format!("only {total_tokens} tokens, need more than context ({context})").into());
    }

    let mut all_log_probs: Vec<f64> = Vec::new();
    let mut window_start = 0usize;

    while window_start + context <= tokens.len() {
        // Fresh runner per window: independent windows, no cross-window KV
        // cache contamination, matching glbench's `score_sequence` semantics
        // (each window is scored from a clean context).
        let mut runner = Runner::new(&model);
        let extended_end = (window_start + context + 1).min(tokens.len());

        // Index into THIS window (0..context) at which positions stop being
        // "already scored by the previous overlapping window" — matches
        // `glbench::ppl::sliding_window_log_probs`'s `new_start` exactly.
        let new_start = if window_start == 0 { 0 } else { context - stride };
        let mut window_log_probs = Vec::new();

        for i in 0..(extended_end - window_start - 1) {
            let tok = tokens[window_start + i];
            runner.forward_into(tok, i)?;
            if i >= new_start {
                let target = tokens[window_start + i + 1] as usize;
                let lp = log_softmax_at(runner.logits(), target);
                window_log_probs.push(lp);
            }
        }

        all_log_probs.extend(&window_log_probs);
        eprint!(".");
        window_start += stride;
    }
    eprintln!();

    let evaluated_tokens = all_log_probs.len();
    let cross_entropy_mean = -all_log_probs.iter().sum::<f64>() / evaluated_tokens as f64;
    let perplexity = cross_entropy_mean.exp();

    println!("model: {model_path}");
    println!("dataset: wikitext2-sample-embedded ({total_tokens} tokens, {evaluated_tokens} scored)");
    println!("context={context} stride={stride}");
    println!("cross_entropy_mean: {cross_entropy_mean:.6}");
    println!("perplexity: {perplexity:.6}");

    Ok(())
}
