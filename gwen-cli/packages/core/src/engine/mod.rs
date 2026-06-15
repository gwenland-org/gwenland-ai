pub mod chat;
pub mod context_tree;
pub mod gguf_loader;
pub mod inference;
pub mod loader;
pub mod memory_guard;
pub mod runtime;
pub mod stream;
pub mod tokenizer;
pub mod transformer_ops;
pub mod windowing;

pub use gguf_loader::load_gguf;
pub use gguf_loader::load_gguf_with_mode;
pub use gguf_loader::dequant_all;
pub use gguf_loader::load_and_dequant;
pub use loader::LoadMode;

/// Runtime output/interaction mode. Constructed from CLI global flags and
/// auto-detected from stdout (pipe-safe: non-TTY → non_interactive = true).
#[derive(Debug, Clone, Default)]
pub struct GwenMode {
    pub dry_run: bool,
    pub non_interactive: bool,
    pub json: bool,
    pub yes: bool,
}

impl GwenMode {
    /// Build from raw flag values. Enables `non_interactive` automatically
    /// when stdout is not a terminal (pipe-safe behaviour).
    pub fn new(dry_run: bool, non_interactive: bool, json: bool, yes: bool) -> Self {
        use std::io::IsTerminal;
        let piped = !std::io::stdout().is_terminal();
        Self {
            dry_run,
            non_interactive: non_interactive || piped,
            json,
            yes,
        }
    }

    /// True when the TUI / interactive path should be used.
    pub fn is_tui(&self) -> bool {
        !self.non_interactive && !self.json
    }

    /// True when output must be newline-delimited JSON (NDJSON).
    pub fn is_ndjson(&self) -> bool {
        self.non_interactive && self.json
    }
}
