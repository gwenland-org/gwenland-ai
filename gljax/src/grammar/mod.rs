//! Structured generation via LLGuidance (ARTX15).
//!
//! ARTX15 §1.4's decision, taken as-is: LLGuidance over Outlines (no
//! recursive JSON Schema) or XGrammar (C++ FFI, which ARTX01's pure-Rust
//! posture declines). It computes a grammar mask in ~50 µs/token for a
//! 128k-token vocabulary with negligible per-schema startup — and it is a
//! `cargo` dependency, not a foreign toolchain: confirmed by actually
//! building it into this crate (`llguidance = "1.7"`), not just reading its
//! `Cargo.toml`.
//!
//! ⚠️ **`LlguidanceMaskSource` sits behind [`crate::sample::MaskSource`], not
//! wired in directly** — ARTX15 §1.4's own design decision. Nothing in
//! `sample::` names LLGuidance; this module is the only place that does.
//!
//! # What this wave builds, and what it doesn't
//!
//! Built: [`GrammarSpec`] (JSON Schema / regex / Lark / "any JSON"),
//! [`GrammarFactory`] (one per tokenizer, per LLGuidance's own recommended
//! lifecycle), [`LlguidanceMaskSource`] implementing `MaskSource`, and
//! single-token rollback via [`LlguidanceMaskSource::commit_with_rollback`].
//! All of it is genuinely tested against a real compiled grammar and a real
//! (small, in-memory) vocabulary — unlike most of this sprint, grammar mask
//! computation is pure host-side CPU work with no PJRT dependency, so
//! nothing here is "structurally sound but never run."
//!
//! Not built:
//! - **`GrammarCache`** (ARTX15 §2) — compiling once per distinct grammar
//!   and reusing across requests. This wave compiles a fresh `Constraint`
//!   per [`GrammarFactory::compile`] call; correct, not amortized.
//! - **Jump-forward decoding** (§3.3) and the mask-caching/worker-pool
//!   mitigations in §3.2 — both depend on ARTX07 (continuous batching), which
//!   does not exist in this codebase.
//! - **Multi-token rollback** — see [`LlguidanceMaskSource::commit_with_rollback`]'s
//!   docs. LLGuidance's real `CommitResult::backtrack` can exceed 1; this
//!   wave refuses that case explicitly rather than attempt it, matching the
//!   sprint brief's exact scope ("Multi-token rollback: document as open
//!   issue, do not implement").
//! - **Tool calling / streaming structured output** (§5) — not started.

use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenizerEnv};
use llguidance::{Constraint, ParserFactory};

use crate::sample::mask::{AllowMask, MaskError, MaskSource, SlotId};
use crate::tok::{TokenId, Tokenizer};

/// A grammar source (ARTX15 §2). `GrammarCache`-by-content-hash (compiling
/// once and reusing across requests that share a schema) is not built this
/// wave — see this module's top docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarSpec {
    JsonSchema(String),
    Regex(String),
    Lark(String),
    /// OpenAI-style `response_format: {"type": "json_object"}` — any valid
    /// JSON. An empty JSON Schema (`{}`) is spec-correct for "no constraint."
    JsonAny,
}

#[derive(Debug)]
pub enum GrammarError {
    InvalidJsonSchema(String),
    Compile(String),
    Commit(String),
    /// LLGuidance signaled a backtrack of more than one token. Not
    /// implemented — see this module's top docs and
    /// [`LlguidanceMaskSource::commit_with_rollback`].
    MultiTokenRollbackNotSupported(u32),
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarError::InvalidJsonSchema(e) => write!(f, "invalid JSON schema: {e}"),
            GrammarError::Compile(e) => write!(f, "grammar compile failed: {e}"),
            GrammarError::Commit(e) => write!(f, "grammar commit failed: {e}"),
            GrammarError::MultiTokenRollbackNotSupported(n) => {
                write!(f, "grammar requested a {n}-token rollback; only single-token rollback is supported")
            }
        }
    }
}

impl std::error::Error for GrammarError {}

impl GrammarSpec {
    fn compile(&self) -> Result<TopLevelGrammar, GrammarError> {
        match self {
            GrammarSpec::JsonSchema(s) => {
                let v: serde_json::Value =
                    serde_json::from_str(s).map_err(|e| GrammarError::InvalidJsonSchema(e.to_string()))?;
                Ok(TopLevelGrammar::from_json_schema(v))
            }
            GrammarSpec::JsonAny => Ok(TopLevelGrammar::from_json_schema(serde_json::json!({}))),
            GrammarSpec::Regex(rx) => Ok(TopLevelGrammar::from_regex(rx)),
            GrammarSpec::Lark(src) => Ok(TopLevelGrammar::from_lark(src.clone())),
        }
    }
}

/// Bridges `gljax::tok::Tokenizer` to `toktrie::TokenizerEnv`, which
/// LLGuidance needs to build its own prefix trie over every token's raw
/// bytes for fast mask computation. This struct is the entire integration
/// surface between gljax's tokenizer abstraction and LLGuidance's.
struct GljaxTokEnv {
    trie: TokTrie,
    tok: Arc<dyn Tokenizer>,
}

impl TokenizerEnv for GljaxTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<llguidance::toktrie::TokenId> {
        // Used by LLGuidance for jump-forward re-tokenization (§3.3, not
        // built this wave) and validating forced token sequences — not on
        // compute_mask's hot path. Lossy on invalid UTF-8 is acceptable here:
        // a byte-level vocabulary's *tokens* can be non-UTF-8 (ARTX13 §4.1),
        // but grammar-forced text is always valid UTF-8 by construction.
        let text = String::from_utf8_lossy(s);
        self.tok.encode(&text, false).unwrap_or_default()
    }
}

/// Builds the `toktrie::TokEnv` LLGuidance needs from any gljax `Tokenizer`.
/// ARTX15 §1.4's "sits behind ARTX14's `MaskSource` trait" decision starts
/// here: nothing past this function's call site needs to know LLGuidance
/// exists.
pub fn build_tok_env(tok: Arc<dyn Tokenizer>) -> TokEnv {
    let vocab_size = tok.vocab_size() as u32;
    let info = TokRxInfo::new(vocab_size, tok.eos_id());
    let words: Vec<Vec<u8>> = (0..vocab_size).map(|id| tok.token_bytes(id)).collect();
    let trie = TokTrie::from(&info, &words);
    Arc::new(GljaxTokEnv { trie, tok })
}

/// One per tokenizer, reused across requests — LLGuidance's own documented
/// lifecycle for `ParserFactory` ("typically created once per model/
/// tokenizer and reused across requests").
pub struct GrammarFactory {
    inner: ParserFactory,
    vocab_size: usize,
}

impl GrammarFactory {
    pub fn new(tok: Arc<dyn Tokenizer>) -> Result<Self, GrammarError> {
        let vocab_size = tok.vocab_size();
        let tok_env = build_tok_env(tok);
        let inner = ParserFactory::new_simple(&tok_env).map_err(|e| GrammarError::Compile(e.to_string()))?;
        Ok(GrammarFactory { inner, vocab_size })
    }

    /// Compiles `spec` into a [`LlguidanceMaskSource`] ready to drive one
    /// request's generation. ARTX15 §2's "reject unsupported schema features
    /// at request admission, never mid-generation" is exactly what a
    /// `Compile`-error return here gives a caller — refuse before the first
    /// token, not after a half-emitted response.
    pub fn compile(&self, spec: &GrammarSpec) -> Result<LlguidanceMaskSource, GrammarError> {
        let grammar = spec.compile()?;
        // `max_tokens` is never set — `TopLevelGrammar::from_*`'s own
        // constructors default it to `None`, and nothing here overrides
        // that. This is deliberate: setting it disables LLGuidance's
        // rollback capability, which `commit_with_rollback` depends on.
        debug_assert!(grammar.max_tokens.is_none());
        let parser = self
            .inner
            .create_parser(grammar)
            .map_err(|e| GrammarError::Compile(e.to_string()))?;
        let mut constraint = Constraint::new(parser);
        constraint.start_without_prompt();
        Ok(LlguidanceMaskSource { constraint, vocab_size: self.vocab_size })
    }
}

/// Implements [`crate::sample::MaskSource`] (ARTX14 §3.3's seam) over one
/// compiled LLGuidance grammar. One instance per in-flight structured
/// request — `Constraint` carries mutable parser state, so it cannot be
/// shared across requests the way [`GrammarFactory`] is.
pub struct LlguidanceMaskSource {
    constraint: Constraint,
    vocab_size: usize,
}

impl MaskSource for LlguidanceMaskSource {
    fn mask_for(&mut self, _slot: SlotId, _history: &[TokenId], out: &mut AllowMask) {
        match self.constraint.compute_mask() {
            Ok(step) => {
                *out = AllowMask::none_allowed(self.vocab_size);
                match &step.sample_mask {
                    Some(mask) => {
                        mask.iter_set_entries(|id| out.allow(id as TokenId));
                    }
                    // `sample_mask == None` means no sampling is needed this
                    // step (the grammar has a single forced continuation —
                    // ARTX15 §3.3's jump-forward case). This wave doesn't
                    // consume `step.splices` to emit the forced tokens
                    // without a model forward pass (that's real, separate
                    // work — see this module's top docs), so it degrades to
                    // "allow everything" rather than stalling a caller with
                    // zero allowed tokens.
                    None => *out = AllowMask::all_allowed(self.vocab_size),
                }
            }
            Err(_) => *out = AllowMask::none_allowed(self.vocab_size),
        }
    }

    fn accept(&mut self, slot: SlotId, token: TokenId) -> Result<(), MaskError> {
        self.commit_with_rollback(token)
            .map(|_backtrack| ())
            .map_err(|_| MaskError::TokenNotAllowed(token))?;
        let _ = slot; // no per-slot state yet (ARTX07 doesn't exist) — see sample::mask's SlotId docs.
        Ok(())
    }
}

impl LlguidanceMaskSource {
    /// Commits `token` to the grammar, returning how many previously-emitted
    /// tokens must be rolled back before this one becomes valid.
    ///
    /// ARTX15 §4's actual mechanism, not the sprint brief's original framing:
    /// the brief described rollback as gated by a `max_tokens`
    /// constraint flag ("Constraint: max_tokens disables rollback... Speak:
    /// max_tokens constraint deliberately NOT set (preserves rollback)").
    /// The real LLGuidance API is more direct — `Constraint::commit_token`
    /// returns `CommitResult { backtrack: u32, .. }` unconditionally, and
    /// `max_tokens` (verified: `TopLevelGrammar::from_*`'s own constructors
    /// default it to `None`, checked by a `debug_assert!` in
    /// `GrammarFactory::compile`) simply must not be set for `backtrack` to
    /// ever be meaningful. There is no separate "rollback capability" toggle
    /// to preserve — only a precondition to not violate.
    ///
    /// This wave supports exactly what the sprint brief asked for: **single-
    /// token rollback**. `backtrack > 1` is a real, documented possibility in
    /// LLGuidance's API (multi-token backtracking, e.g. for tokenizer/grammar
    /// boundary mismatches) that this wave explicitly does not implement —
    /// returns `MultiTokenRollbackNotSupported` rather than silently
    /// mis-handling it.
    pub fn commit_with_rollback(&mut self, token: TokenId) -> Result<u32, GrammarError> {
        let result = self
            .constraint
            .commit_token(Some(token))
            .map_err(|e| GrammarError::Commit(e.to_string()))?;
        if result.backtrack > 1 {
            return Err(GrammarError::MultiTokenRollbackNotSupported(result.backtrack));
        }
        Ok(result.backtrack)
    }

    /// Whether the grammar considers generation complete (no further tokens
    /// can be accepted).
    pub fn has_pending_stop(&self) -> bool {
        self.constraint.has_pending_stop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tok::GllmTokenizerAdapter;

    /// A small byte-level vocabulary covering enough ASCII to actually
    /// produce and reject JSON — every printable byte gljax's own tokenizer
    /// tests already establish the GPT-2 byte-to-unicode mapping for
    /// (`gljax::tok::stream`'s test module has the same derivation) plus a
    /// couple of multi-character tokens so a real schema has something to
    /// discriminate between.
    fn json_capable_tokenizer() -> Arc<dyn Tokenizer> {
        let mut vocab = serde_json::Map::new();
        let mut id = 0u32;
        for b in 33u32..127 {
            // Printable ASCII maps to itself under GPT-2 byte-level encoding.
            let ch = char::from_u32(b).unwrap();
            vocab.insert(ch.to_string(), serde_json::json!(id));
            id += 1;
        }
        // A couple of multi-byte "word" tokens, so the vocabulary isn't
        // purely single-character (more representative of a real BPE vocab).
        for word in ["true", "false", "null", "name"] {
            vocab.insert(word.to_string(), serde_json::json!(id));
            id += 1;
        }
        let src = serde_json::json!({
            "model": { "type": "BPE", "vocab": vocab, "merges": [] },
            "pre_tokenizer": { "type": "ByteLevel" },
            "added_tokens": [{"id": id, "content": "<|endoftext|>"}],
        })
        .to_string();
        Arc::new(GllmTokenizerAdapter::from_hf_json(&src).expect("must load"))
    }

    #[test]
    fn factory_and_json_any_grammar_compile_successfully() {
        let tok = json_capable_tokenizer();
        let factory = GrammarFactory::new(tok).expect("factory must build");
        let mask_source = factory.compile(&GrammarSpec::JsonAny);
        assert!(mask_source.is_ok(), "{:?}", mask_source.err());
    }

    #[test]
    fn json_schema_grammar_rejects_a_malformed_schema_string_at_compile_time() {
        let tok = json_capable_tokenizer();
        let factory = GrammarFactory::new(tok).expect("factory must build");
        let bad = GrammarSpec::JsonSchema("not valid json at all {{{".to_string());
        match factory.compile(&bad) {
            Err(GrammarError::InvalidJsonSchema(_)) => {}
            Ok(_) => panic!("must refuse at compile time, not mid-generation"),
            Err(other) => panic!("wrong error variant: {other}"),
        }
    }

    /// ⭐ The whole point of ARTX14 §3.3/ARTX15 §3: a grammar mask must
    /// actually exclude tokens that would produce invalid output. A JSON
    /// value grammar's first token cannot be a letter — valid JSON starts
    /// with `{`, `[`, `"`, a digit, `-`, `t`/`f`/`n` (true/false/null).
    #[test]
    fn grammar_mask_rejects_invalid_first_tokens_for_a_json_value() {
        let tok = json_capable_tokenizer();
        let factory = GrammarFactory::new(tok).expect("factory must build");
        let mut source = factory.compile(&GrammarSpec::JsonAny).expect("compile");

        let mut mask = AllowMask::none_allowed(200);
        source.mask_for(SlotId(0), &[], &mut mask);

        // 'x' is not a valid start of any JSON value.
        let x_id = ('x' as u32) - 33;
        assert!(!mask.is_allowed(x_id as TokenId), "'x' must not start a JSON value");

        // '{' IS a valid start of a JSON value (an object).
        let brace_id = ('{' as u32) - 33;
        assert!(mask.is_allowed(brace_id as TokenId), "'{{' must be allowed to start a JSON value");
    }

    /// The sprint brief's exact second test: "commit_token advances grammar
    /// state" — committing `{` must change what the mask allows next (a
    /// letter/`"` for a key or `}` to close, not another `{`-starting value
    /// in the same position).
    #[test]
    fn commit_token_advances_grammar_state_and_changes_the_next_mask() {
        let tok = json_capable_tokenizer();
        let factory = GrammarFactory::new(tok).expect("factory must build");
        let mut source = factory.compile(&GrammarSpec::JsonAny).expect("compile");

        let mut mask_before = AllowMask::none_allowed(200);
        source.mask_for(SlotId(0), &[], &mut mask_before);

        let brace_id = (('{' as u32) - 33) as TokenId;
        assert!(mask_before.is_allowed(brace_id));
        source.accept(SlotId(0), brace_id).expect("'{' must be accepted");

        let mut mask_after = AllowMask::none_allowed(200);
        source.mask_for(SlotId(0), &[], &mut mask_after);
        assert_ne!(
            mask_before, mask_after,
            "committing a token must change the grammar state, and therefore the mask"
        );
    }

    #[test]
    fn accept_refuses_a_token_the_current_mask_disallows() {
        let tok = json_capable_tokenizer();
        let factory = GrammarFactory::new(tok).expect("factory must build");
        let mut source = factory.compile(&GrammarSpec::JsonAny).expect("compile");

        let mut mask = AllowMask::none_allowed(200);
        source.mask_for(SlotId(0), &[], &mut mask);
        let x_id = (('x' as u32) - 33) as TokenId;
        assert!(!mask.is_allowed(x_id), "test setup: 'x' must actually be disallowed");

        assert!(source.accept(SlotId(0), x_id).is_err(), "accepting a disallowed token must error");
    }

    /// The sprint brief's exact third test, adapted to the real API: rather
    /// than a boolean "rollback capability" preserved by never setting
    /// `max_tokens`, `commit_with_rollback` always reports LLGuidance's real
    /// `backtrack` count. This test pins that the *normal* path (accepting a
    /// token the grammar actually offered) never requires a rollback.
    #[test]
    fn accepting_an_offered_token_never_requires_rollback() {
        let tok = json_capable_tokenizer();
        let factory = GrammarFactory::new(tok).expect("factory must build");
        let mut source = factory.compile(&GrammarSpec::JsonAny).expect("compile");

        let mut mask = AllowMask::none_allowed(200);
        source.mask_for(SlotId(0), &[], &mut mask);
        let brace_id = (('{' as u32) - 33) as TokenId;
        assert!(mask.is_allowed(brace_id));

        let backtrack = source.commit_with_rollback(brace_id).expect("must accept");
        assert_eq!(backtrack, 0, "accepting an offered token must not require rollback");
    }

    #[test]
    fn max_tokens_is_never_set_by_any_grammar_spec_constructor() {
        for spec in [
            GrammarSpec::JsonAny,
            GrammarSpec::JsonSchema("{}".to_string()),
            GrammarSpec::Regex("[a-z]+".to_string()),
            GrammarSpec::Lark("start: \"a\"".to_string()),
        ] {
            let grammar = spec.compile().expect("must compile");
            assert!(
                grammar.max_tokens.is_none(),
                "{spec:?}: max_tokens must stay None -- setting it disables rollback"
            );
        }
    }

    #[test]
    fn regex_and_lark_specs_compile_without_error() {
        let tok = json_capable_tokenizer();
        let factory = GrammarFactory::new(tok).expect("factory must build");
        assert!(factory.compile(&GrammarSpec::Regex("[a-z]+".to_string())).is_ok());
        assert!(factory.compile(&GrammarSpec::Lark("start: \"true\"".to_string())).is_ok());
    }
}
