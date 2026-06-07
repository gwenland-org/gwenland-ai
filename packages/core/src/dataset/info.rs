use crate::dataset::schema::{load_rows_from_path, LoadedLine};
use std::collections::HashMap;
use std::path::Path;

pub struct DatasetInfo {
    pub total: usize,
    pub skipped: usize,
    pub avg_input_tokens: f64,
    pub avg_output_tokens: f64,
    pub token_label: String,
    pub think_ratio: f64,
    pub categories: Vec<(String, usize)>, // sorted desc
    pub warnings: Vec<String>,
}

pub fn run_info(path: &Path, model: Option<&str>) -> Result<DatasetInfo, String> {
    let loaded = load_rows_from_path(path)?;

    let mut total = 0usize;
    let mut skipped = 0usize;
    let mut warnings: Vec<String> = Vec::new();
    let mut input_token_sum = 0usize;
    let mut output_token_sum = 0usize;
    let mut think_count = 0usize;
    let mut category_counts: HashMap<String, usize> = HashMap::new();

    let use_tokenizer = model.is_some();

    // Optionally load HF tokenizer once before iterating.
    let tokenizer = if let Some(model_id) = model {
        match load_hf_tokenizer(model_id) {
            Ok(tok) => Some(tok),
            Err(e) => {
                warnings.push(format!(
                    "⚠ Could not load tokenizer for '{}': {} — falling back to word-count estimate",
                    model_id, e
                ));
                None
            }
        }
    } else {
        None
    };

    let rows: Vec<_> = loaded
        .into_iter()
        .filter_map(|entry| match entry {
            LoadedLine::Row { row, .. } => Some(row),
            LoadedLine::Skipped { line_no, reason } => {
                warnings.push(format!("⚠ Line {}: skipped ({})", line_no, reason));
                skipped += 1;
                None
            }
        })
        .collect();

    for row in &rows {
        total += 1;

        let (in_tok, out_tok) = match &tokenizer {
            Some(tok) => (
                count_tokens_hf(tok, &row.input),
                count_tokens_hf(tok, &row.output),
            ),
            None => (
                estimate_tokens(&row.input),
                estimate_tokens(&row.output),
            ),
        };
        input_token_sum += in_tok;
        output_token_sum += out_tok;

        if row.output.contains("<think>") || row.output.contains("<THINK>") {
            think_count += 1;
        }

        if let Some(cat) = &row.category {
            *category_counts.entry(cat.clone()).or_insert(0) += 1;
        }
    }

    let avg_input_tokens = if total > 0 {
        input_token_sum as f64 / total as f64
    } else {
        0.0
    };
    let avg_output_tokens = if total > 0 {
        output_token_sum as f64 / total as f64
    } else {
        0.0
    };
    let think_ratio = if total > 0 {
        think_count as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    let token_label = if tokenizer.is_some() {
        format!(
            "exact, {} tokenizer",
            model.unwrap_or("unknown")
        )
    } else {
        "estimated via word-count".into()
    };

    let mut categories: Vec<(String, usize)> = category_counts.into_iter().collect();
    categories.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    Ok(DatasetInfo {
        total,
        skipped,
        avg_input_tokens,
        avg_output_tokens,
        token_label,
        think_ratio,
        categories,
        warnings,
    })
}

pub fn estimate_tokens(text: &str) -> usize {
    (text.split_whitespace().count() as f32 / 0.75).ceil() as usize
}

// ── HF tokenizer integration ──────────────────────────────────────────────────

fn load_hf_tokenizer(model_id: &str) -> Result<tokenizers::Tokenizer, String> {
    tokenizers::Tokenizer::from_pretrained(model_id, None)
        .map_err(|e| e.to_string())
}

fn count_tokens_hf(tokenizer: &tokenizers::Tokenizer, text: &str) -> usize {
    tokenizer
        .encode(text, false)
        .map(|enc| enc.len())
        .unwrap_or_else(|_| estimate_tokens(text))
}
