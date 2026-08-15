// @INFO: GwenLand Context Truncator
// Implements truncation rules to ensure context fits within model budgets.

// @EDITABLE
use std::path::Path;
use crate::platform::scanner::FileEntry;
use crate::engine::tokenizer::{ModelBudget, TokenStatus, BUDGET_8K, BUDGET_32K, BUDGET_128K};

pub struct TruncateConfig {
    pub strategy: TruncateStrategy,
    pub reserve_prompt_tokens: usize, // @EDITABLE default: 512
}

impl Default for TruncateConfig {
    fn default() -> Self {
        Self {
            strategy: TruncateStrategy::DropLargest,
            reserve_prompt_tokens: 512,
        }
    }
}

pub enum TruncateStrategy {
    DropLargest,   // drop biggest files first until fits
    DropTailFirst, // drop files from end of list first
    Summarize,     // future: placeholder only for now
}

pub struct TruncateResult {
    pub kept: Vec<FileEntry>,
    pub dropped: Vec<String>,   // paths of dropped files
    pub final_tokens: usize,
    pub budget: usize,
    pub was_truncated: bool,
}

pub fn truncate_to_budget(
    files: Vec<FileEntry>,
    prompt: &str,
    budget: ModelBudget,
    config: &TruncateConfig,
) -> TruncateResult {
    let budget_val = match budget {
        ModelBudget::Small => BUDGET_8K,
        ModelBudget::Medium => BUDGET_32K,
        ModelBudget::Large => BUDGET_128K,
        ModelBudget::Custom(val) => val,
    };

    let initial_estimate = crate::engine::tokenizer::estimate_context(&files, prompt, budget);
    match initial_estimate.status {
        TokenStatus::Safe | TokenStatus::Warning => {
            return TruncateResult {
                kept: files,
                dropped: Vec::new(),
                final_tokens: initial_estimate.total_tokens,
                budget: initial_estimate.budget,
                was_truncated: false,
            };
        }
        TokenStatus::Critical | TokenStatus::Exceeded => {}
    }

    let prompt_tokens = crate::engine::tokenizer::estimate_tokens(prompt);
    // Calculate target context budget
    let target_limit = budget_val.saturating_sub(std::cmp::max(prompt_tokens, config.reserve_prompt_tokens));

    // Calculate token counts for each file
    struct FileWithTokens {
        entry: FileEntry,
        tokens: usize,
        original_index: usize,
    }

    let mut files_with_tokens: Vec<FileWithTokens> = files
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let mut tokens = crate::engine::tokenizer::estimate_file(Path::new(&entry.path));
            if tokens == 0 && entry.size_bytes > 0 && !entry.is_binary {
                tokens = (entry.size_bytes / 4) as usize;
            }
            FileWithTokens {
                entry,
                tokens,
                original_index: idx,
            }
        })
        .collect();

    let mut current_context_tokens: usize = files_with_tokens.iter().map(|f| f.tokens).sum();
    let mut dropped_paths = Vec::new();

    match config.strategy {
        TruncateStrategy::DropLargest => {
            // Sort by token count descending
            files_with_tokens.sort_by(|a, b| b.tokens.cmp(&a.tokens));
            
            let mut kept_files = Vec::new();
            for item in files_with_tokens {
                if current_context_tokens <= target_limit {
                    kept_files.push(item);
                } else {
                    current_context_tokens = current_context_tokens.saturating_sub(item.tokens);
                    dropped_paths.push(item.entry.path.clone());
                }
            }
            // Sort kept files back to their original order
            kept_files.sort_by_key(|item| item.original_index);
            
            let kept_entries: Vec<FileEntry> = kept_files.into_iter().map(|item| item.entry).collect();
            let final_tokens = prompt_tokens + current_context_tokens;

            TruncateResult {
                kept: kept_entries,
                dropped: dropped_paths,
                final_tokens,
                budget: budget_val,
                was_truncated: true,
            }
        }
        TruncateStrategy::DropTailFirst => {
            let mut kept_files = Vec::new();
            let mut prefix_tokens = 0;
            let mut dropped_at_end = false;

            for item in files_with_tokens {
                if !dropped_at_end && prefix_tokens + item.tokens <= target_limit {
                    prefix_tokens += item.tokens;
                    kept_files.push(item.entry);
                } else {
                    dropped_at_end = true;
                    dropped_paths.push(item.entry.path.clone());
                }
            }

            let final_tokens = prompt_tokens + prefix_tokens;

            TruncateResult {
                kept: kept_files,
                dropped: dropped_paths,
                final_tokens,
                budget: budget_val,
                was_truncated: true,
            }
        }
        TruncateStrategy::Summarize => {
            // Summarize is future placeholder: just return all files as-is
            let kept_entries: Vec<FileEntry> = files_with_tokens.into_iter().map(|item| item.entry).collect();
            TruncateResult {
                kept: kept_entries,
                dropped: Vec::new(),
                final_tokens: initial_estimate.total_tokens,
                budget: budget_val,
                was_truncated: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_safe_no_truncation() {
        let files = vec![
            FileEntry {
                path: "a.rs".to_string(),
                size_bytes: 400, // 100 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            }
        ];

        let config = TruncateConfig::default();
        let res = truncate_to_budget(files, "prompt", ModelBudget::Small, &config); // Small budget = 8000. 101 tokens is Safe.
        assert!(!res.was_truncated);
        assert_eq!(res.kept.len(), 1);
        assert!(res.dropped.is_empty());
    }

    #[test]
    fn test_truncate_drop_largest() {
        let files = vec![
            FileEntry {
                path: "small.rs".to_string(),
                size_bytes: 80, // 20 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            },
            FileEntry {
                path: "large.rs".to_string(),
                size_bytes: 400, // 100 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            },
            FileEntry {
                path: "medium.rs".to_string(),
                size_bytes: 200, // 50 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            }
        ];

        // Custom budget: 100 tokens. Reserve: 20 tokens. Prompt: 0 tokens.
        // Target context limit: 100 - 20 = 80 tokens.
        // Total initial context tokens: 170. Exceeded!
        // DropLargest:
        // Sort: large (100), medium (50), small (20).
        // 1. Drop large (100). Remaining context: 70. 70 <= 80 target limit!
        // Keep: medium (50), small (20). Kept are returned in original order: small.rs, medium.rs.
        // Dropped: large.rs.
        let config = TruncateConfig {
            strategy: TruncateStrategy::DropLargest,
            reserve_prompt_tokens: 20,
        };
        let res = truncate_to_budget(files, "", ModelBudget::Custom(100), &config);
        assert!(res.was_truncated);
        assert_eq!(res.kept.len(), 2);
        assert_eq!(res.kept[0].path, "small.rs");
        assert_eq!(res.kept[1].path, "medium.rs");
        assert_eq!(res.dropped, vec!["large.rs"]);
        assert_eq!(res.final_tokens, 70);
    }

    #[test]
    fn test_truncate_drop_tail_first() {
        let files = vec![
            FileEntry {
                path: "first.rs".to_string(),
                size_bytes: 160, // 40 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            },
            FileEntry {
                path: "second.rs".to_string(),
                size_bytes: 160, // 40 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            },
            FileEntry {
                path: "third.rs".to_string(),
                size_bytes: 160, // 40 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            }
        ];

        // Custom budget: 100 tokens. Reserve: 20 tokens. Prompt: 0 tokens.
        // Target context limit: 80 tokens.
        // DropTailFirst:
        // Keep first.rs (40), keep second.rs (40) -> 80 tokens.
        // Drop third.rs (40) -> exceeds.
        let config = TruncateConfig {
            strategy: TruncateStrategy::DropTailFirst,
            reserve_prompt_tokens: 20,
        };
        let res = truncate_to_budget(files, "", ModelBudget::Custom(100), &config);
        assert!(res.was_truncated);
        assert_eq!(res.kept.len(), 2);
        assert_eq!(res.kept[0].path, "first.rs");
        assert_eq!(res.kept[1].path, "second.rs");
        assert_eq!(res.dropped, vec!["third.rs"]);
        assert_eq!(res.final_tokens, 80);
    }
}
