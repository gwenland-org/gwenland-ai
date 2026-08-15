// @INFO: GwenLand Token Estimation Engine
// Counts/estimates tokens before sending context to the AI to prevent context window overflow.
// Use the standard approximation: 1 token ≈ 4 characters.

// @EDITABLE — model context window budgets in tokens
pub const BUDGET_8K: usize   = 8_000;
pub const BUDGET_32K: usize  = 32_000;
pub const BUDGET_128K: usize = 128_000;

use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use colored::Colorize;
use crate::platform::scanner::FileEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelBudget {
    Small,   // 8K  — small local models
    Medium,  // 32K — mid-range models
    Large,   // 128K — large models (Llama 3, Mistral, etc)
    Custom(usize),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenEstimate {
    pub total_tokens: usize,
    pub prompt_tokens: usize,
    pub context_tokens: usize,
    pub budget: usize,
    pub usage_percent: f32,
    pub status: TokenStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenStatus {
    Safe,      // < 70% budget
    Warning,   // 70–90% budget
    Critical,  // 90–100% budget
    Exceeded,  // > 100% budget
}


/// Detects if a file is binary by looking for null bytes (0x00) within the first 512 bytes.
fn is_binary_file(path: &Path) -> bool {
    use std::fs::File;
    use std::io::Read;

    if let Ok(mut file) = File::open(path) {
        let mut buffer = [0u8; 512];
        if let Ok(bytes_read) = file.read(&mut buffer) {
            return buffer[..bytes_read].contains(&0);
        }
    }
    false
}

// @INFO — GPT/Llama tokenizers average ~4 chars per token for code
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

pub fn estimate_file(path: &Path) -> usize {
    if is_binary_file(path) {
        return 0;
    }
    match fs::read_to_string(path) {
        Ok(content) => estimate_tokens(&content),
        Err(_) => 0,
    }
}

pub fn estimate_context(
    files: &[FileEntry],
    prompt: &str,
    budget: ModelBudget,
) -> TokenEstimate {
    let budget_val = match budget {
        ModelBudget::Small => BUDGET_8K,
        ModelBudget::Medium => BUDGET_32K,
        ModelBudget::Large => BUDGET_128K,
        ModelBudget::Custom(val) => val,
    };

    let prompt_tokens = estimate_tokens(prompt);
    let mut context_tokens = 0;

    for file in files {
        if file.is_binary {
            continue;
        }
        let tokens = estimate_file(Path::new(&file.path));
        if tokens == 0 && file.size_bytes > 0 {
            context_tokens += (file.size_bytes / 4) as usize;
        } else {
            context_tokens += tokens;
        }
    }

    let total_tokens = prompt_tokens + context_tokens;
    let usage_percent = if budget_val > 0 {
        (total_tokens as f32 / budget_val as f32) * 100.0
    } else {
        0.0
    };

    let status = if usage_percent < 70.0 {
        TokenStatus::Safe
    } else if usage_percent < 90.0 {
        TokenStatus::Warning
    } else if usage_percent <= 100.0 {
        TokenStatus::Critical
    } else {
        TokenStatus::Exceeded
    };

    TokenEstimate {
        total_tokens,
        prompt_tokens,
        context_tokens,
        budget: budget_val,
        usage_percent,
        status,
    }
}

pub fn print_token_report(estimate: &TokenEstimate) {
    let status_str = match estimate.status {
        TokenStatus::Safe => "🟢 Safe".green(),
        TokenStatus::Warning => "🟡 Warning".yellow(),
        TokenStatus::Critical => "🔴 Critical".red(),
        TokenStatus::Exceeded => "❌ Exceeded".bright_red().bold(),
    };

    println!("=================== TOKEN REPORT ===================");
    println!("  Budget Limit:     {} tokens", estimate.budget);
    println!("  Prompt Tokens:    {} tokens", estimate.prompt_tokens);
    println!("  Context Tokens:   {} tokens", estimate.context_tokens);
    println!("  Total Tokens:     {} tokens", estimate.total_tokens);
    println!("  Usage Percent:    {:.2}%", estimate.usage_percent);
    println!("  Status:           {}", status_str);
    println!("====================================================");
}

pub fn detect_budget_from_model(model_name: &str) -> ModelBudget {
    let lower = model_name.to_lowercase();
    if lower.contains("3b") || lower.contains("7b") || lower.contains("8b") {
        ModelBudget::Small
    } else if lower.contains("13b") || lower.contains("14b") || lower.contains("32b") {
        ModelBudget::Medium
    } else if lower.contains("70b")
        || lower.contains("72b")
        || lower.contains("llama3")
        || lower.contains("mistral")
        || lower.contains("qwen")
    {
        ModelBudget::Large
    } else {
        ModelBudget::Large // @EDITABLE
    }
}

pub fn auto_estimate_context(files: &[FileEntry], prompt: &str) -> TokenEstimate {
    let model_name = get_active_model_from_config().unwrap_or_else(|| "unknown".to_string());
    let budget = detect_budget_from_model(&model_name);
    estimate_context(files, prompt, budget)
}

pub fn get_active_model_from_config() -> Option<String> {
    let model = crate::storage::config::GwenConfig::load().general.last_used_model;
    if model.is_empty() { None } else { Some(model) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn test_estimate_file() {
        let temp_dir = std::env::temp_dir().join("gwen_tokenizer_test");
        fs::create_dir_all(&temp_dir).unwrap();

        let text_file = temp_dir.join("test.txt");
        fs::write(&text_file, "Hello world, testing 1 2 3!").unwrap(); // 27 chars -> 6 tokens
        assert_eq!(estimate_file(&text_file), 6);

        let binary_file = temp_dir.join("test.bin");
        fs::write(&binary_file, b"Hello\x00world").unwrap();
        assert_eq!(estimate_file(&binary_file), 0);

        let non_existent = temp_dir.join("does_not_exist.txt");
        assert_eq!(estimate_file(&non_existent), 0);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_detect_budget_from_model() {
        assert_eq!(detect_budget_from_model("llama-3-8b-instruct"), ModelBudget::Small);
        assert_eq!(detect_budget_from_model("mistral-7b"), ModelBudget::Small);
        assert_eq!(detect_budget_from_model("qwen-32b-chat"), ModelBudget::Medium);
        assert_eq!(detect_budget_from_model("llama-3-70b-instruct"), ModelBudget::Large);
        assert_eq!(detect_budget_from_model("mistral-large"), ModelBudget::Large);
        assert_eq!(detect_budget_from_model("unknown-model-xyz"), ModelBudget::Large);
    }

    #[test]
    fn test_estimate_context() {
        let files = vec![
            FileEntry {
                path: "nonexistent.rs".to_string(),
                size_bytes: 400, // fallback: 400 / 4 = 100 tokens
                extension: Some("rs".to_string()),
                is_binary: false,
            },
            FileEntry {
                path: "binary.bin".to_string(),
                size_bytes: 1000,
                extension: Some("bin".to_string()),
                is_binary: true,
            }
        ];

        // Safe budget (100 context + 20 prompt = 120 tokens, budget 8K (8000))
        let est = estimate_context(&files, "hello world hello wo", ModelBudget::Small); // prompt len 20 -> 5 tokens
        // Wait, prompt len is 20 -> 5 tokens. Total tokens = 100 + 5 = 105 tokens.
        assert_eq!(est.prompt_tokens, 5);
        assert_eq!(est.context_tokens, 100);
        assert_eq!(est.total_tokens, 105);
        assert_eq!(est.budget, 8000);
        assert_eq!(est.status, TokenStatus::Safe);
        assert!(est.usage_percent < 70.0);

        // Warning budget: custom budget 120 -> 105/120 = 87.5%
        let est_warn = estimate_context(&files, "hello world hello wo", ModelBudget::Custom(120));
        assert_eq!(est_warn.status, TokenStatus::Warning);

        // Critical budget: custom budget 110 -> 105/110 = 95.4%
        let est_crit = estimate_context(&files, "hello world hello wo", ModelBudget::Custom(110));
        assert_eq!(est_crit.status, TokenStatus::Critical);

        // Exceeded budget: custom budget 100 -> 105/100 = 105%
        let est_exc = estimate_context(&files, "hello world hello wo", ModelBudget::Custom(100));
        assert_eq!(est_exc.status, TokenStatus::Exceeded);
    }
}
