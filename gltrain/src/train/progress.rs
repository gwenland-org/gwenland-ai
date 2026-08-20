//! Structured progress events emitted to stdout during a candle training run.
//!
//! # Why a dedicated module?
//!
//! `training_loop.rs` already hand-rolled a `println!(r#"{{…}}"#, …)` block.
//! That works for five fields, but it breaks the moment anyone adds a field —
//! the format string must be updated in two places, the escaping is fragile, and
//! there is no type-level guarantee that the output matches what the TUI parser
//! expects.  A typed struct + `serde_json::to_string` solves all three problems.
//!
//! # Why not `tracing` / `log` + a JSON subscriber?
//!
//! The TUI reads stdout synchronously with a line-by-line parser
//! (`parse_train_event` in `tui/train_panel.rs`).  A logging framework would
//! either interleave diagnostic text with the event stream (breaking the parser)
//! or require a custom writer that hides non-JSON lines.  Raw `println!` to
//! stdout is the simplest contract that satisfies the consumer.
//!
//! # Wire format
//!
//! Every event is a **compact single-line JSON object** followed by `\n`.
//! No pretty-print, no framing bytes.  The TUI splits on `\n` and calls
//! `serde_json::from_str` on each line.
//!
//! Step event:
//! ```json
//! {"step":42,"epoch":1,"loss":2.3451,"lr":0.0001,"tokens_per_sec":840.5}
//! ```
//!
//! Done event:
//! ```json
//! {"event":"done","final_loss":1.1200,"total_steps":960,"elapsed_secs":312}
//! ```
//!
//! # Field-name stability
//!
//! The TUI parser (`parse_train_event`) reads `step`, `epoch`, `loss`, and `lr`
//! by name.  `tokens_per_sec` is new — the parser currently ignores unknown
//! fields, so adding it is backwards-compatible.  **Do not rename existing
//! fields** without also updating `parse_train_event`.

use serde::Serialize;

// ── step event ────────────────────────────────────────────────────────────────

/// One training step, emitted after every batch.
///
/// The TUI's `parse_train_event` keys off the presence of `"step"` to
/// distinguish a step frame from a lifecycle frame (`"event":"done"`, etc.).
/// That means `step` **must** be present in every step object; do not make it
/// optional.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    /// Global batch index, 1-based, monotonically increasing across all epochs.
    /// Named `step` (not `global_batch`) to match the TUI parser's key lookup.
    pub step: usize,

    /// Current epoch, 1-based.
    pub epoch: usize,

    /// Running mean cross-entropy loss over the current accumulation window.
    /// Using f32 keeps the JSON compact; the TUI only renders two decimal places.
    pub loss: f32,

    /// Current AdamW learning rate.  f64 to avoid rounding the scheduler output
    /// before it reaches the TUI — the TUI displays it as-is.
    pub lr: f64,

    /// Effective throughput: tokens processed per wall-clock second this step.
    /// Computed by the caller as `seq_len / elapsed_secs_this_batch`.
    /// The TUI does not consume this field yet, but it is logged for offline
    /// analysis of training speed.
    pub tokens_per_sec: f32,
}

impl ProgressEvent {
    /// Serialize to compact JSON and write to stdout.
    ///
    /// One call = one `\n`-terminated line.  `serde_json::to_string` never
    /// pretty-prints; the output is always a single line regardless of value
    /// complexity.
    ///
    /// # Why `unwrap` on `to_string`?
    ///
    /// `serde_json::to_string` can only fail if the value contains a map with
    /// non-string keys or a float that is NaN / ±Inf.  All fields here are
    /// plain integers and finite floats, so the failure path is unreachable in
    /// practice.  Surfacing a `Result` from `emit` would force every call site
    /// to handle an error that never occurs and obscure the real error paths in
    /// the training loop.
    pub fn emit(&self) {
        println!("{}", serde_json::to_string(self).unwrap());
    }
}

// ── done event ────────────────────────────────────────────────────────────────

/// Terminal event emitted once when the training run completes.
///
/// Keyed by `"event":"done"` so the TUI parser can distinguish it from a step
/// frame (which is keyed by the presence of `"step"`).
#[derive(Debug, Clone, Serialize)]
pub struct DoneEvent {
    /// Literal sentinel consumed by the TUI's `match val.get("event")` branch.
    pub event: &'static str,

    /// Loss from the final accumulation window.
    pub final_loss: f32,

    /// Total number of AdamW optimiser steps taken.
    pub total_steps: usize,

    /// Wall-clock seconds from `TrainingLoop::run()` entry to return.
    pub elapsed_secs: u64,
}

impl DoneEvent {
    /// Construct and emit in one call.
    pub fn emit(final_loss: f32, total_steps: usize, elapsed_secs: u64) {
        let ev = Self {
            event: "done",
            final_loss,
            total_steps,
            elapsed_secs,
        };
        println!("{}", serde_json::to_string(&ev).unwrap());
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Capture stdout by serializing directly rather than spawning a subprocess.
    fn step_json(step: usize, epoch: usize, loss: f32, lr: f64, tps: f32) -> String {
        serde_json::to_string(&ProgressEvent { step, epoch, loss, lr, tokens_per_sec: tps })
            .unwrap()
    }

    fn done_json(final_loss: f32, total_steps: usize, elapsed_secs: u64) -> String {
        serde_json::to_string(&DoneEvent {
            event: "done",
            final_loss,
            total_steps,
            elapsed_secs,
        })
        .unwrap()
    }

    #[test]
    fn step_is_single_line() {
        let s = step_json(1, 1, 2.41, 0.0001, 840.0);
        assert!(!s.contains('\n'), "must not contain newlines: {s}");
    }

    #[test]
    fn step_contains_required_tui_fields() {
        let s = step_json(42, 3, 1.5, 5e-5, 1024.0);
        // TUI parser looks for these exact keys
        assert!(s.contains(r#""step":42"#),   "missing step: {s}");
        assert!(s.contains(r#""epoch":3"#),   "missing epoch: {s}");
        assert!(s.contains(r#""loss":"#),      "missing loss: {s}");
        assert!(s.contains(r#""lr":"#),        "missing lr: {s}");
    }

    #[test]
    fn step_contains_tokens_per_sec() {
        let s = step_json(1, 1, 0.0, 0.0, 512.5);
        assert!(s.contains(r#""tokens_per_sec":"#), "missing tokens_per_sec: {s}");
    }

    #[test]
    fn done_event_field_is_done() {
        let s = done_json(1.12, 960, 312);
        assert!(s.contains(r#""event":"done""#), "wrong event sentinel: {s}");
    }

    #[test]
    fn done_contains_required_fields() {
        let s = done_json(0.85, 500, 180);
        assert!(s.contains(r#""final_loss":"#),   "missing final_loss: {s}");
        assert!(s.contains(r#""total_steps":500"#), "missing total_steps: {s}");
        assert!(s.contains(r#""elapsed_secs":180"#), "missing elapsed_secs: {s}");
    }

    #[test]
    fn done_is_single_line() {
        let s = done_json(1.0, 100, 60);
        assert!(!s.contains('\n'), "must not contain newlines: {s}");
    }
}
