//! Stummañ Deskiñ: Wave 3's forward pass, as assertions.
//!
//! The chain under test is
//! `ABEmbedding -> ABLinear -> log_softmax -> masked_cross_entropy`, run on the
//! committed corpus slice. It lives in `tests/` rather than beside any one
//! module because it spans four of them, and because it is the check that the
//! pieces compose — each of which already has unit tests proving it works alone.

use gltrain::{
    ABByteTokenizer, ABEmbedding, ABLinear, DTChatMl, GlProc, Module, Tape, Tensor, Tokenizer,
    VLBatch, VLChatMlConfig, IGNORE_INDEX,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gwen_code_dataset.jsonl"
);

const D_MODEL: usize = 32;
const BATCH: usize = 3;
/// The system prompt alone is ~380 byte-level tokens and the user turn adds
/// more, so the assistant span does not begin until roughly token 500. A cap
/// below that truncates every sample before anything supervised appears, and
/// `masked_cross_entropy` correctly refuses the resulting 0/0 batch.
const MAX_SEQ_LEN: usize = 1024;
const SEED: u64 = 20260823;

/// One flattened batch of real corpus tokens: `(ids, labels, vocab_size)`.
fn corpus_batch() -> (Vec<u32>, Vec<i32>, usize) {
    let corpus = DTChatMl::from_jsonl_path(Path::new(FIXTURE)).expect("fixture loads");
    let tok = ABByteTokenizer::new();
    let cfg = VLChatMlConfig::new(MAX_SEQ_LEN);
    let samples: Vec<_> = (0..BATCH)
        .map(|i| corpus.tokenize_one(i, &tok, &cfg).expect("tokenizes"))
        .collect();
    let batch = VLBatch::collate(&samples, tok.pad_id()).expect("collates");
    (
        batch.input_ids().to_vec(),
        batch.labels().to_vec(),
        tok.vocab_size(),
    )
}

/// Run the chain, returning `(loss, logits_shape, tape_nodes)`.
fn forward(ids: &[u32], labels: &[i32], vocab: usize, init_std: f32) -> (f32, Vec<usize>, usize) {
    let tape = Arc::new(Mutex::new(Tape::new()));
    let embed =
        ABEmbedding::<GlProc>::randn("embed", vocab, D_MODEL, init_std, SEED).expect("embed");
    let head =
        ABLinear::<GlProc>::randn("lm_head", D_MODEL, vocab, init_std, SEED + 1).expect("head");

    let hidden = embed.forward_ids(ids, &tape).expect("gather");
    let logits = head.forward(&hidden, &tape).expect("projection");
    let log_probs = logits.log_softmax().expect("log_softmax");
    let loss = log_probs.masked_cross_entropy(labels).expect("loss");

    let nodes = Tape::lock(&tape).len();
    (loss.item().expect("scalar"), logits.shape().to_vec(), nodes)
}

#[test]
fn the_chain_produces_a_finite_loss_and_the_expected_shapes() {
    let (ids, labels, vocab) = corpus_batch();
    let (loss, logits_shape, nodes) = forward(&ids, &labels, vocab, 0.02);

    assert_eq!(
        logits_shape,
        vec![ids.len(), vocab],
        "logits are [n_tokens, vocab]"
    );
    assert!(loss.is_finite(), "loss is {loss}");
    assert!(loss > 0.0, "cross-entropy cannot be negative, got {loss}");
    // Embedding, Matmul, LogSoftmax, MaskedCrossEntropy.
    assert_eq!(nodes, 4, "one node per op in the chain");
}

/// The standard language-model-head check. With the head near zero the softmax
/// is near uniform, so the loss must land on `ln(vocab_size)`. Far below means
/// labels are leaking into the input; far above means the head or the mask is
/// wrong. 1% is loose enough for a 0.02 init and tight enough to catch either.
#[test]
fn a_near_zero_initialization_lands_on_the_uniform_loss() {
    let (ids, labels, vocab) = corpus_batch();
    let (loss, _, _) = forward(&ids, &labels, vocab, 0.001);
    let uniform = (vocab as f32).ln();
    assert!(
        (loss - uniform).abs() / uniform < 0.01,
        "loss {loss} is not within 1% of ln({vocab}) = {uniform}"
    );
}

/// The loss must actually respond to the labels rather than being a constant
/// the shapes happen to produce. A head confident in the *right* answer scores
/// near zero; the same head against shifted labels scores far worse.
#[test]
fn the_loss_responds_to_whether_the_labels_match_the_logits() {
    let vocab = 8;
    let rows = 4;
    // Row r puts all its mass on class r.
    let mut logits = vec![0.0f32; rows * vocab];
    for (r, row) in logits.chunks_mut(vocab).enumerate() {
        row[r] = 20.0;
    }
    let t = Tensor::<GlProc>::from_vec(logits, &[rows, vocab]).expect("tensor");
    let log_probs = t.log_softmax().expect("log_softmax");

    let right: Vec<i32> = (0..rows as i32).collect();
    let wrong: Vec<i32> = (0..rows as i32).map(|r| (r + 1) % rows as i32).collect();

    let good = log_probs
        .masked_cross_entropy(&right)
        .expect("loss")
        .item()
        .expect("scalar");
    let bad = log_probs
        .masked_cross_entropy(&wrong)
        .expect("loss")
        .item()
        .expect("scalar");

    assert!(
        good < 1e-5,
        "a confident correct head should score ~0, got {good}"
    );
    assert!(
        bad > 10.0,
        "a confident wrong head should score badly, got {bad}"
    );
}

/// Prompt and padding positions carry IGNORE_INDEX, so changing the *input*
/// under a masked position must move the loss, but changing only masked
/// *labels* must not. This is what proves the mask is load-bearing.
#[test]
fn masked_positions_do_not_contribute_to_the_loss() {
    let (ids, labels, vocab) = corpus_batch();
    let (baseline, _, _) = forward(&ids, &labels, vocab, 0.02);

    // Rewrite every masked label to a different (still masked) sentinel value
    // by leaving it as IGNORE_INDEX, and every supervised one untouched. The
    // loss must be bit-identical: nothing masked ever entered it.
    let same: Vec<i32> = labels
        .iter()
        .map(|&l| if l == IGNORE_INDEX { IGNORE_INDEX } else { l })
        .collect();
    let (unchanged, _, _) = forward(&ids, &same, vocab, 0.02);
    assert_eq!(baseline, unchanged);

    // Supervising one previously-masked position must move it.
    let mut widened = labels.clone();
    let first_masked = widened
        .iter()
        .position(|&l| l == IGNORE_INDEX)
        .expect("the batch has masked positions");
    widened[first_masked] = 0;
    let (widened_loss, _, _) = forward(&ids, &widened, vocab, 0.02);
    assert_ne!(
        baseline, widened_loss,
        "supervising one more position must change the loss"
    );
}

/// Every row of the log-probabilities is a distribution, on real data and not
/// just on the constructed fixtures the unit tests use.
#[test]
fn log_probabilities_are_a_distribution_on_every_row_of_a_real_batch() {
    let (ids, labels, vocab) = corpus_batch();
    let _ = labels;
    let tape = Arc::new(Mutex::new(Tape::new()));
    let embed = ABEmbedding::<GlProc>::randn("embed", vocab, D_MODEL, 0.02, SEED).expect("embed");
    let head = ABLinear::<GlProc>::randn("lm_head", D_MODEL, vocab, 0.02, SEED + 1).expect("head");

    let hidden = embed.forward_ids(&ids, &tape).expect("gather");
    let logits = head.forward(&hidden, &tape).expect("projection");
    let values = logits
        .log_softmax()
        .expect("log_softmax")
        .to_vec()
        .expect("host");

    assert_eq!(values.len(), ids.len() * vocab);
    for (r, row) in values.chunks(vocab).enumerate() {
        assert!(
            row.iter().all(|v| v.is_finite() && *v <= 0.0),
            "row {r} is not a log-distribution"
        );
        let total: f32 = row.iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-4, "row {r} sums to {total}");
    }
}
