//! `MlirEmitter` — the text buffer and SSA name allocator (ARTX02 §1).
//!
//! Pure string formatting. No shapes, no validation, no knowledge of ops. The
//! layer that knows what a `dot_general` is lives in [`super::ops`]; the layer
//! that knows what a *model* is arrives in ARTX02's `FuncBuilder` (Wave A2).

use std::fmt::Write as _;

/// A raw SSA identifier. Displays as `%vN`.
///
/// Name only — [`crate::stablehlo::types::Shape`] is carried separately, and
/// `SsaValue` (Wave A2) will pair the two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SsaName(pub u32);

impl std::fmt::Display for SsaName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%v{}", self.0)
    }
}

/// Emits the lines of a single StableHLO `func.func` body.
///
/// ⚠️ **`!Send` by design** (ARTX02 §1). A trace is single-threaded; making
/// this `Send` would mean a mutex around the SSA counter for no benefit. If
/// you ever want parallel tracing, run two emitters on two threads and compose
/// their outputs — do not share one. The `PhantomData` below makes that a
/// compile error rather than a convention.
pub struct MlirEmitter {
    buf: String,
    ssa_counter: u32,
    /// 1 = inside a `func.func` body. Two spaces per level.
    indent: usize,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl MlirEmitter {
    /// A fresh emitter positioned inside a function body.
    ///
    /// The 64 KiB pre-allocation is ARTX02 §1's figure: a full transformer
    /// block's worth of lines without a realloc in the middle of a trace.
    pub fn new() -> Self {
        MlirEmitter {
            buf: String::with_capacity(64 * 1024),
            ssa_counter: 0,
            indent: 1,
            _not_send: std::marker::PhantomData,
        }
    }

    /// Allocates a fresh SSA name. Emits nothing.
    pub fn fresh(&mut self) -> SsaName {
        let id = self.ssa_counter;
        self.ssa_counter += 1;
        SsaName(id)
    }

    /// Emits one indented line, appending a newline.
    pub fn line(&mut self, s: impl std::fmt::Display) {
        for _ in 0..self.indent {
            self.buf.push_str("  ");
        }
        // `String`'s `fmt::Write` is infallible — the only error path is a
        // `Display` impl that itself fails, and none of ours can.
        let _ = writeln!(self.buf, "{s}");
    }

    pub fn push_indent(&mut self) {
        self.indent += 1;
    }

    pub fn pop_indent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Current indent level (1 = function body).
    pub fn indent_level(&self) -> usize {
        self.indent
    }

    /// How many SSA names have been handed out.
    pub fn ssa_count(&self) -> u32 {
        self.ssa_counter
    }

    /// Read-only view of what has been emitted so far.
    pub fn body(&self) -> &str {
        &self.buf
    }

    /// Consumes the emitter and returns the body lines.
    ///
    /// Does **not** include the `func.func` header, the return, or the
    /// `module` wrapper: those are assembled one layer up (ARTX02 §1), which
    /// is what lets a region — a `while` body, a `reduce` computation — get
    /// its own emitter at a deeper indent.
    pub fn into_body(self) -> String {
        self.buf
    }
}

impl Default for MlirEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssa_names_display_as_percent_v_n() {
        assert_eq!(SsaName(0).to_string(), "%v0");
        assert_eq!(SsaName(4096).to_string(), "%v4096");
    }

    #[test]
    fn fresh_hands_out_monotonic_names_without_emitting() {
        let mut e = MlirEmitter::new();
        assert_eq!(e.fresh(), SsaName(0));
        assert_eq!(e.fresh(), SsaName(1));
        assert_eq!(e.ssa_count(), 2);
        assert!(e.body().is_empty(), "fresh() must not emit anything");
    }

    #[test]
    fn indent_is_two_spaces_per_level_starting_inside_the_func_body() {
        let mut e = MlirEmitter::new();
        e.line("a");
        e.push_indent();
        e.line("b");
        e.pop_indent();
        e.line("c");
        assert_eq!(e.into_body(), "  a\n    b\n  c\n");
    }

    /// `pop_indent` at level 0 would underflow a `usize`. A malformed region
    /// nesting should produce ugly text, not a panic in a release build and a
    /// different failure in debug.
    #[test]
    fn pop_indent_saturates_instead_of_underflowing() {
        let mut e = MlirEmitter::new();
        e.pop_indent();
        e.pop_indent();
        assert_eq!(e.indent_level(), 0);
        e.line("x");
        assert_eq!(e.into_body(), "x\n");
    }
}
