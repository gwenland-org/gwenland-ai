//! Stummañ Deskiñ: the ChatML corpus, its tokenization, and the loss mask.
//!
//! This is the SFT data path: a `.jsonl` file of ChatML conversations becomes
//! `input_ids` plus a `labels` vector in which everything the model is *not*
//! asked to produce is [`IGNORE_INDEX`].
//!
//! [`crate::train::dataset::VLMicroDataset`] is unchanged and still owns the
//! synthetic regression task M2's convergence test runs on. The two live side
//! by side because they answer different questions: that one asks "does the
//! loop converge on a problem with a known answer", this one asks "does a real
//! corpus survive the trip to a padded batch intact".
//!
//! # The rendered form
//!
//! ```text
//! <|im_start|>system\n{content}<|im_end|>
//! <|im_start|>user\n{content}<|im_end|>
//! <|im_start|>assistant\n{content}<|im_end|>
//! ```
//!
//! Turns are joined by a single `\n`, with no trailing newline after the last
//! `<|im_end|>`. The newline is a separator *between* turns, so a one-turn
//! sample is not silently given a dangling token that no turn owns.
//!
//! # What is supervised
//!
//! Per turn, `<|im_start|>`, the role name and the newline after it are the
//! **header**. The header is always masked, including the assistant's:
//! `<|im_start|>assistant\n` is what *conditions* generation, it is not what
//! the model is asked to emit. Training on it teaches the model to predict a
//! delimiter it will always be handed for free.
//!
//! The assistant's content is supervised, and so is the `<|im_end|>` that
//! closes it — that token is how the model learns to stop, and masking it
//! produces a model that never terminates. Set
//! [`VLChatMlConfig::supervise_im_end`] to `false` only to reproduce a
//! reference implementation that made the other choice.
//!
//! # Labels are not shifted here
//!
//! `labels[i]` describes position `i` of `input_ids`, not `i + 1`. The
//! next-token shift belongs at the loss, which is where every framework puts
//! it. Shifting in the collator as well would shift twice, and the resulting
//! model would be off by one token in a way that still trains and still lowers
//! the loss.
//!
//! Wave 3's [`crate::tensor::Tensor::masked_cross_entropy`] deliberately does
//! **not** apply it. It is a position-wise cross-entropy: row `i` of the logits
//! is scored against `labels[i]`. The shift is only meaningful once a *causal*
//! model exists, because it is the attention mask that lets position `i` see
//! tokens `0..=i` and nothing later. Wave 3's chain has no attention, so
//! position `i`'s logits depend on token `i` alone, and shifting would ask it
//! to predict token `i + 1` from token `i` with nothing else to go on. The
//! shift lands with the transformer, applied by its caller.

use crate::checkpoint::json::{self, Json};
use crate::error::{GlTrainError, Result};
use crate::train::tokenizer::Tokenizer;
use std::path::Path;

/// The label value for a position that must not contribute to the loss.
///
/// `-100` is not arbitrary: it is PyTorch's `CrossEntropyLoss` default
/// `ignore_index`, and every SFT corpus and script in circulation assumes it.
/// A different sentinel would be correct in isolation and wrong the moment
/// anyone compared a loss curve against a reference run.
pub const IGNORE_INDEX: i32 = -100;

/// Who is speaking in a turn.
///
/// `EN` because a closed set of variants is this type's whole job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ENChatRole {
    /// The system prompt. Never supervised.
    System,
    /// The user's message. Never supervised.
    User,
    /// The response. This is what the model is trained to produce.
    Assistant,
}

impl ENChatRole {
    /// The role's wire name, exactly as it appears in the JSONL and in the
    /// rendered prompt.
    pub fn as_str(&self) -> &'static str {
        match self {
            ENChatRole::System => "system",
            ENChatRole::User => "user",
            ENChatRole::Assistant => "assistant",
        }
    }

    /// Parse a wire name.
    ///
    /// An unknown role is an error rather than a skipped line: a corpus with a
    /// `"tool"` or `"function"` role needs a deliberate decision about whether
    /// it is supervised, and defaulting it to "masked" would quietly discard
    /// training signal.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "system" => Ok(ENChatRole::System),
            "user" => Ok(ENChatRole::User),
            "assistant" => Ok(ENChatRole::Assistant),
            other => Err(GlTrainError::Data(format!(
                "unknown chat role {other:?}; expected system, user or assistant"
            ))),
        }
    }

    /// Whether this role's content contributes to the loss.
    pub fn is_supervised(&self) -> bool {
        matches!(self, ENChatRole::Assistant)
    }
}

/// One `{"role": ..., "content": ...}` message.
///
/// `VL` because it is a plain data bag with derived traits only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLChatTurn {
    /// Who is speaking.
    pub role: ENChatRole,
    /// What they said, verbatim and untokenized.
    pub content: String,
}

/// One conversation: the `messages` array of a single JSONL line.
///
/// `VL` for the same reason as [`VLChatTurn`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLChatSample {
    /// Turns in wire order.
    pub turns: Vec<VLChatTurn>,
}

impl VLChatSample {
    /// Parse one JSONL line.
    fn from_json(v: &Json) -> Result<Self> {
        let messages = v
            .get("messages")
            .and_then(Json::as_arr)
            .ok_or_else(|| GlTrainError::Data("no \"messages\" array".into()))?;

        if messages.is_empty() {
            return Err(GlTrainError::Data("\"messages\" is empty".into()));
        }

        let mut turns = Vec::with_capacity(messages.len());
        for (i, m) in messages.iter().enumerate() {
            let role = m
                .get("role")
                .and_then(Json::as_str)
                .ok_or_else(|| GlTrainError::Data(format!("message {i} has no \"role\"")))?;
            let content = m
                .get("content")
                .and_then(Json::as_str)
                .ok_or_else(|| GlTrainError::Data(format!("message {i} has no \"content\"")))?;
            turns.push(VLChatTurn {
                role: ENChatRole::parse(role)?,
                content: content.to_string(),
            });
        }

        // Nothing to learn from a conversation with no response in it. Caught
        // here rather than at tokenization, where it would surface as a sample
        // whose labels are entirely IGNORE_INDEX and whose loss is 0/0.
        if !turns.iter().any(|t| t.role.is_supervised()) {
            return Err(GlTrainError::Data(
                "no assistant turn, so the sample carries no training signal".into(),
            ));
        }
        Ok(Self { turns })
    }

    /// How many turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether there are no turns. Never true for a parsed sample.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }
}

/// How to turn a [`VLChatSample`] into ids and labels.
///
/// `VL` because it is a plain config bag with derived traits only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLChatMlConfig {
    /// Hard cap on a single sample's length. Longer samples are truncated from
    /// the right and flagged — see [`VLTokenizedSample::is_truncated`].
    pub max_seq_len: usize,
    /// Whether the `<|im_end|>` closing an assistant turn is supervised.
    ///
    /// Defaults to `true`. This is the token that teaches the model to stop.
    pub supervise_im_end: bool,
}

impl Default for VLChatMlConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 1024,
            supervise_im_end: true,
        }
    }
}

impl VLChatMlConfig {
    /// A config capped at `max_seq_len`, supervising `<|im_end|>`.
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            max_seq_len,
            ..Default::default()
        }
    }
}

/// One sample, tokenized, with its loss mask.
///
/// `input_ids` and `labels` are the same length. A label is either a real
/// token id or [`IGNORE_INDEX`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLTokenizedSample {
    input_ids: Vec<u32>,
    labels: Vec<i32>,
    truncated: bool,
}

impl VLTokenizedSample {
    /// The token ids fed to the model.
    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    /// The per-position targets, [`IGNORE_INDEX`] where masked.
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    /// Sequence length. Equal for ids and labels by construction.
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Whether there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    /// Whether `max_seq_len` cut this sample short.
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// How many positions actually contribute to the loss.
    ///
    /// Zero means truncation removed the whole assistant span. Such a sample
    /// contributes a `0/0` to a length-normalized cross-entropy, so Wave 3's
    /// loss must either skip it or normalize over the batch's total supervised
    /// count rather than per-sample. [`VLBatch::supervised_len`] is the number
    /// to divide by.
    pub fn supervised_len(&self) -> usize {
        self.labels.iter().filter(|&&l| l != IGNORE_INDEX).count()
    }

    /// The supervised label ids, in order, with the masked positions dropped.
    ///
    /// This is what the model is being asked to produce. Decoding it is the
    /// fastest way to see whether the mask landed where it should.
    pub fn supervised_ids(&self) -> Vec<u32> {
        self.labels
            .iter()
            .filter(|&&l| l != IGNORE_INDEX)
            .map(|&l| l as u32)
            .collect()
    }
}

/// A ChatML corpus held in memory.
///
/// `DT` because it is a dataset artifact loaded from disk, as distinct from
/// the `VL` value types it holds.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DTChatMl {
    samples: Vec<VLChatSample>,
}

impl DTChatMl {
    /// An empty corpus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load every line of a `.jsonl` file.
    ///
    /// Blank lines are skipped. A malformed line is an error naming its
    /// 1-based line number — a corpus that silently drops the lines it cannot
    /// read is a corpus whose size in the run header is a lie.
    pub fn from_jsonl_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_jsonl_str(&text).map_err(|e| match e {
            GlTrainError::Data(msg) => GlTrainError::Data(format!("{}: {msg}", path.display())),
            other => other,
        })
    }

    /// Parse a whole `.jsonl` document from memory.
    pub fn from_jsonl_str(text: &str) -> Result<Self> {
        let mut samples = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value = json::parse(line)
                .map_err(|e| GlTrainError::Data(format!("line {}: {e}", i + 1)))?;
            let sample = VLChatSample::from_json(&value)
                .map_err(|e| GlTrainError::Data(format!("line {}: {e}", i + 1)))?;
            samples.push(sample);
        }
        Ok(Self { samples })
    }

    /// How many conversations.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether there is nothing to train on.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// One conversation, if `idx` is in range.
    pub fn get(&self, idx: usize) -> Option<&VLChatSample> {
        self.samples.get(idx)
    }

    /// Every conversation.
    pub fn samples(&self) -> &[VLChatSample] {
        &self.samples
    }

    /// Tokenize sample `idx`.
    pub fn tokenize_one<T: Tokenizer>(
        &self,
        idx: usize,
        tok: &T,
        cfg: &VLChatMlConfig,
    ) -> Result<VLTokenizedSample> {
        let sample = self.samples.get(idx).ok_or_else(|| {
            GlTrainError::Data(format!(
                "sample {idx} is out of range for a {}-sample corpus",
                self.samples.len()
            ))
        })?;
        tokenize_sample(sample, tok, cfg)
    }

    /// Tokenize the whole corpus.
    pub fn tokenize<T: Tokenizer>(
        &self,
        tok: &T,
        cfg: &VLChatMlConfig,
    ) -> Result<Vec<VLTokenizedSample>> {
        self.samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                tokenize_sample(s, tok, cfg)
                    .map_err(|e| GlTrainError::Data(format!("sample {i}: {e}")))
            })
            .collect()
    }
}

/// Render one conversation to ids and labels.
///
/// Emission is turn by turn, and each turn contributes its header (masked),
/// its content, and its `<|im_end|>`. `labels` is written in lockstep with
/// `input_ids` so the two cannot drift apart.
fn tokenize_sample<T: Tokenizer>(
    sample: &VLChatSample,
    tok: &T,
    cfg: &VLChatMlConfig,
) -> Result<VLTokenizedSample> {
    if cfg.max_seq_len == 0 {
        return Err(GlTrainError::Data("max_seq_len must be at least 1".into()));
    }

    let im_start = tok.im_start()?;
    let im_end = tok.im_end()?;
    let newline = tok.encode("\n")?;

    let mut input_ids = Vec::new();
    let mut labels = Vec::new();

    // Push `ids`, supervising them or masking them.
    let push = |ids: &[u32], supervised: bool, input_ids: &mut Vec<u32>, labels: &mut Vec<i32>| {
        for &id in ids {
            input_ids.push(id);
            labels.push(if supervised { id as i32 } else { IGNORE_INDEX });
        }
    };

    for (i, turn) in sample.turns.iter().enumerate() {
        if i > 0 {
            // The separator between turns belongs to neither of them.
            push(&newline, false, &mut input_ids, &mut labels);
        }

        // Header: `<|im_start|>{role}\n`. Always masked, assistant included —
        // it conditions the response rather than being part of it.
        push(&[im_start], false, &mut input_ids, &mut labels);
        let header = tok.encode(turn.role.as_str())?;
        push(&header, false, &mut input_ids, &mut labels);
        push(&newline, false, &mut input_ids, &mut labels);

        let supervised = turn.role.is_supervised();
        let content = tok.encode(&turn.content)?;
        push(&content, supervised, &mut input_ids, &mut labels);

        // `<|im_end|>` is how the model learns to stop.
        push(
            &[im_end],
            supervised && cfg.supervise_im_end,
            &mut input_ids,
            &mut labels,
        );
    }

    let truncated = input_ids.len() > cfg.max_seq_len;
    if truncated {
        input_ids.truncate(cfg.max_seq_len);
        labels.truncate(cfg.max_seq_len);
    }

    debug_assert_eq!(input_ids.len(), labels.len());
    Ok(VLTokenizedSample {
        input_ids,
        labels,
        truncated,
    })
}

/// A rectangle of samples, padded to a common length.
///
/// `VL` because it is a plain data bag: three flat vectors and their
/// dimensions, no behaviour beyond indexing.
///
/// All three payloads are `[batch, seq_len]` row-major, so row `b` occupies
/// `[b * seq_len, (b + 1) * seq_len)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLBatch {
    input_ids: Vec<u32>,
    labels: Vec<i32>,
    attention_mask: Vec<u8>,
    batch: usize,
    seq_len: usize,
}

impl VLBatch {
    /// Pad a slice of samples to the length of the longest one.
    ///
    /// Padding is dynamic — to the longest member, not to `max_seq_len`. A
    /// batch of short samples then costs what it should instead of what the
    /// configured ceiling would.
    ///
    /// Every padded position gets `pad_id` in `input_ids`, [`IGNORE_INDEX`] in
    /// `labels`, and `0` in `attention_mask`. The pad id therefore never
    /// reaches a gradient, but it does reach the embedding lookup, which is
    /// why it has to be a real id rather than a sentinel.
    pub fn collate(samples: &[VLTokenizedSample], pad_id: u32) -> Result<Self> {
        if samples.is_empty() {
            return Err(GlTrainError::Data("cannot collate an empty batch".into()));
        }
        let batch = samples.len();
        let seq_len = samples
            .iter()
            .map(VLTokenizedSample::len)
            .max()
            .unwrap_or(0);
        if seq_len == 0 {
            return Err(GlTrainError::Data(
                "every sample in the batch is empty".into(),
            ));
        }

        let mut input_ids = Vec::with_capacity(batch * seq_len);
        let mut labels = Vec::with_capacity(batch * seq_len);
        let mut attention_mask = Vec::with_capacity(batch * seq_len);

        for s in samples {
            let pad = seq_len - s.len();
            input_ids.extend_from_slice(s.input_ids());
            input_ids.extend(std::iter::repeat_n(pad_id, pad));
            labels.extend_from_slice(s.labels());
            labels.extend(std::iter::repeat_n(IGNORE_INDEX, pad));
            attention_mask.extend(std::iter::repeat_n(1u8, s.len()));
            attention_mask.extend(std::iter::repeat_n(0u8, pad));
        }

        Ok(Self {
            input_ids,
            labels,
            attention_mask,
            batch,
            seq_len,
        })
    }

    /// Number of sequences.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Padded sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// `[batch, seq_len]`, the shape all three payloads share.
    pub fn shape(&self) -> [usize; 2] {
        [self.batch, self.seq_len]
    }

    /// Token ids, `[batch, seq_len]` row-major.
    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    /// Targets, `[batch, seq_len]` row-major, [`IGNORE_INDEX`] where masked.
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    /// `1` for a real token, `0` for padding. `[batch, seq_len]` row-major.
    pub fn attention_mask(&self) -> &[u8] {
        &self.attention_mask
    }

    /// Row `b` of `input_ids`, padding included.
    pub fn input_ids_row(&self, b: usize) -> Option<&[u32]> {
        (b < self.batch).then(|| &self.input_ids[b * self.seq_len..(b + 1) * self.seq_len])
    }

    /// Row `b` of `labels`, padding included.
    pub fn labels_row(&self, b: usize) -> Option<&[i32]> {
        (b < self.batch).then(|| &self.labels[b * self.seq_len..(b + 1) * self.seq_len])
    }

    /// How many positions in the whole batch contribute to the loss.
    ///
    /// This is the denominator a length-normalized cross-entropy divides by.
    /// Normalizing per sample instead makes a short response worth as much as
    /// a long one, and divides by zero on a sample truncation left unsupervised.
    pub fn supervised_len(&self) -> usize {
        self.labels.iter().filter(|&&l| l != IGNORE_INDEX).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::tokenizer::ABByteTokenizer;

    /// The committed slice of the real corpus: 1500 single-turn samples.
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gwen_code_dataset.jsonl"
    );

    fn tiny() -> DTChatMl {
        DTChatMl::from_jsonl_str(
            r#"{"messages":[{"role":"system","content":"S"},{"role":"user","content":"U"},{"role":"assistant","content":"A"}]}"#,
        )
        .unwrap()
    }

    // ── Parsing ──────────────────────────────────────────────────────────

    #[test]
    fn a_single_turn_line_parses_into_three_turns() {
        let ds = tiny();
        assert_eq!(ds.len(), 1);
        let s = ds.get(0).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.turns[0].role, ENChatRole::System);
        assert_eq!(s.turns[1].role, ENChatRole::User);
        assert_eq!(s.turns[2].role, ENChatRole::Assistant);
        assert_eq!(s.turns[2].content, "A");
    }

    #[test]
    fn blank_lines_are_skipped_but_malformed_ones_name_their_line_number() {
        let ok = DTChatMl::from_jsonl_str("\n\n").unwrap();
        assert!(ok.is_empty());

        let err = DTChatMl::from_jsonl_str("{\"messages\":[]}\n").unwrap_err();
        assert!(err.to_string().contains("line 1"), "got {err}");
    }

    /// A sample with no assistant turn has no training signal, and its loss
    /// would be a `0/0`. Caught at parse rather than discovered as a NaN.
    #[test]
    fn a_sample_with_no_assistant_turn_is_rejected() {
        let err = DTChatMl::from_jsonl_str(
            r#"{"messages":[{"role":"system","content":"S"},{"role":"user","content":"U"}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("no assistant turn"), "got {err}");
    }

    #[test]
    fn an_unknown_role_is_an_error_rather_than_a_silently_masked_turn() {
        let err = DTChatMl::from_jsonl_str(
            r#"{"messages":[{"role":"tool","content":"x"},{"role":"assistant","content":"y"}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown chat role"), "got {err}");
    }

    // ── The mask ─────────────────────────────────────────────────────────

    /// The property the whole module exists for: the supervised span is the
    /// assistant's content and nothing else.
    #[test]
    fn only_the_assistant_content_and_its_im_end_are_supervised() {
        let t = ABByteTokenizer::new();
        let s = tiny()
            .tokenize_one(0, &t, &VLChatMlConfig::default())
            .unwrap();

        // "A" is one byte, plus the <|im_end|> that closes the turn.
        assert_eq!(s.supervised_len(), 2);
        let sup = s.supervised_ids();
        assert_eq!(t.decode(&sup), "A<|im_end|>");
    }

    #[test]
    fn masking_im_end_leaves_only_the_content_supervised() {
        let t = ABByteTokenizer::new();
        let cfg = VLChatMlConfig {
            supervise_im_end: false,
            ..Default::default()
        };
        let s = tiny().tokenize_one(0, &t, &cfg).unwrap();
        assert_eq!(t.decode(&s.supervised_ids()), "A");
    }

    /// The assistant *header* must be masked. If it were supervised, this
    /// count would include `<|im_start|>assistant\n` as well.
    #[test]
    fn the_assistant_header_is_masked_like_every_other_header() {
        let t = ABByteTokenizer::new();
        let s = tiny()
            .tokenize_one(0, &t, &VLChatMlConfig::default())
            .unwrap();
        let decoded = t.decode(s.input_ids());
        assert!(decoded.contains("<|im_start|>assistant\nA<|im_end|>"));
        assert!(
            !t.decode(&s.supervised_ids()).contains("assistant"),
            "the header leaked into the supervised span"
        );
    }

    /// Every supervised label must equal the id at the same position: the two
    /// vectors are written in lockstep and must not drift.
    #[test]
    fn labels_are_aligned_with_input_ids_and_not_pre_shifted() {
        let t = ABByteTokenizer::new();
        let s = tiny()
            .tokenize_one(0, &t, &VLChatMlConfig::default())
            .unwrap();
        assert_eq!(s.input_ids().len(), s.labels().len());
        for (i, (&id, &label)) in s.input_ids().iter().zip(s.labels()).enumerate() {
            if label != IGNORE_INDEX {
                assert_eq!(label as u32, id, "label {i} disagrees with its input id");
            }
        }
    }

    #[test]
    fn the_rendered_prompt_has_the_chatml_shape() {
        let t = ABByteTokenizer::new();
        let s = tiny()
            .tokenize_one(0, &t, &VLChatMlConfig::default())
            .unwrap();
        assert_eq!(
            t.decode(s.input_ids()),
            "<|im_start|>system\nS<|im_end|>\n\
             <|im_start|>user\nU<|im_end|>\n\
             <|im_start|>assistant\nA<|im_end|>"
        );
    }

    /// Multi-turn is masked the same way: every assistant turn is supervised,
    /// every user turn is not.
    #[test]
    fn a_multi_turn_sample_supervises_every_assistant_turn() {
        let ds = DTChatMl::from_jsonl_str(
            r#"{"messages":[{"role":"system","content":"S"},{"role":"user","content":"U1"},{"role":"assistant","content":"A1"},{"role":"user","content":"U2"},{"role":"assistant","content":"A2"}]}"#,
        )
        .unwrap();
        let t = ABByteTokenizer::new();
        let s = ds.tokenize_one(0, &t, &VLChatMlConfig::default()).unwrap();
        assert_eq!(t.decode(&s.supervised_ids()), "A1<|im_end|>A2<|im_end|>");
    }

    // ── Truncation ───────────────────────────────────────────────────────

    #[test]
    fn truncation_is_flagged_and_caps_the_length() {
        let t = ABByteTokenizer::new();
        let s = tiny().tokenize_one(0, &t, &VLChatMlConfig::new(8)).unwrap();
        assert!(s.is_truncated());
        assert_eq!(s.len(), 8);
        assert_eq!(s.labels().len(), 8);
        // Truncation cut the sample off before the assistant turn began.
        assert_eq!(s.supervised_len(), 0);
    }

    #[test]
    fn a_zero_max_seq_len_is_refused() {
        let t = ABByteTokenizer::new();
        assert!(tiny().tokenize_one(0, &t, &VLChatMlConfig::new(0)).is_err());
    }

    // ── Collation ────────────────────────────────────────────────────────

    #[test]
    fn collation_pads_to_the_longest_sample_and_masks_the_padding() {
        let ds = DTChatMl::from_jsonl_str(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"U\"},{\"role\":\"assistant\",\"content\":\"short\"}]}\n\
             {\"messages\":[{\"role\":\"user\",\"content\":\"U\"},{\"role\":\"assistant\",\"content\":\"much longer answer\"}]}\n",
        )
        .unwrap();
        let t = ABByteTokenizer::new();
        let samples = ds.tokenize(&t, &VLChatMlConfig::default()).unwrap();
        let batch = VLBatch::collate(&samples, t.pad_id()).unwrap();

        assert_eq!(batch.batch(), 2);
        assert_eq!(batch.seq_len(), samples[1].len());
        assert_eq!(batch.shape(), [2, samples[1].len()]);
        assert_eq!(batch.input_ids().len(), 2 * batch.seq_len());
        assert_eq!(batch.labels().len(), 2 * batch.seq_len());
        assert_eq!(batch.attention_mask().len(), 2 * batch.seq_len());

        // Row 0 is the short one, so it carries the padding.
        let pad = batch.seq_len() - samples[0].len();
        assert!(pad > 0, "the two samples must differ in length");
        let ids0 = batch.input_ids_row(0).unwrap();
        let labels0 = batch.labels_row(0).unwrap();
        for i in samples[0].len()..batch.seq_len() {
            assert_eq!(ids0[i], t.pad_id());
            assert_eq!(labels0[i], IGNORE_INDEX);
            assert_eq!(batch.attention_mask()[i], 0);
        }
        // Row 1 is full length, so none of it is padding.
        assert!(batch.attention_mask()[batch.seq_len()..]
            .iter()
            .all(|&m| m == 1));
    }

    /// Padding must not add training signal, and the batch's supervised count
    /// must be exactly the sum of its samples'.
    #[test]
    fn padding_contributes_nothing_to_the_supervised_count() {
        let ds = DTChatMl::from_jsonl_str(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"x\"}]}\n\
             {\"messages\":[{\"role\":\"user\",\"content\":\"b\"},{\"role\":\"assistant\",\"content\":\"yyyy\"}]}\n",
        )
        .unwrap();
        let t = ABByteTokenizer::new();
        let samples = ds.tokenize(&t, &VLChatMlConfig::default()).unwrap();
        let batch = VLBatch::collate(&samples, t.pad_id()).unwrap();
        let expected: usize = samples.iter().map(VLTokenizedSample::supervised_len).sum();
        assert_eq!(batch.supervised_len(), expected);
    }

    #[test]
    fn collating_an_empty_batch_is_refused() {
        assert!(VLBatch::collate(&[], 0).is_err());
    }

    #[test]
    fn a_row_index_past_the_end_is_none_rather_than_a_panic() {
        let t = ABByteTokenizer::new();
        let samples = tiny().tokenize(&t, &VLChatMlConfig::default()).unwrap();
        let batch = VLBatch::collate(&samples, t.pad_id()).unwrap();
        assert!(batch.input_ids_row(1).is_none());
        assert!(batch.labels_row(1).is_none());
    }

    // ── The real corpus ──────────────────────────────────────────────────

    #[test]
    fn the_committed_fixture_loads_all_1500_samples() {
        let ds = DTChatMl::from_jsonl_path(Path::new(FIXTURE)).unwrap();
        assert_eq!(ds.len(), 1500);
        for s in ds.samples() {
            assert_eq!(s.len(), 3, "the slice is single-turn system/user/assistant");
            assert!(s.turns.iter().any(|t| t.role.is_supervised()));
        }
    }

    /// End to end on real text: every sample tokenizes, every one keeps its
    /// mask aligned, and every untruncated one has something to learn from.
    #[test]
    fn the_committed_fixture_tokenizes_with_a_well_formed_mask() {
        let ds = DTChatMl::from_jsonl_path(Path::new(FIXTURE)).unwrap();
        let t = ABByteTokenizer::new();
        // Generous cap: byte-level ids make these ~4x longer than BPE would.
        let cfg = VLChatMlConfig::new(16_384);
        let samples = ds.tokenize(&t, &cfg).unwrap();
        assert_eq!(samples.len(), 1500);

        for (i, s) in samples.iter().enumerate() {
            assert_eq!(s.input_ids().len(), s.labels().len(), "sample {i}");
            assert!(s.len() <= cfg.max_seq_len, "sample {i} exceeds the cap");
            if !s.is_truncated() {
                assert!(s.supervised_len() > 0, "sample {i} has no signal");
            }
            for (&id, &label) in s.input_ids().iter().zip(s.labels()) {
                assert!(
                    label == IGNORE_INDEX || label as u32 == id,
                    "sample {i}: label drifted from its input id"
                );
            }
        }
    }
}
