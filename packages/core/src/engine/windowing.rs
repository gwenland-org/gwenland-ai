use std::collections::HashSet;

// ─── Config ───────────────────────────────────────────────────────────────────

pub struct WindowConfig {
    pub enabled: bool,
    pub token_budget: usize,
    pub window_size: usize,
    pub max_windows: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            enabled: false,     // opt-in; disabled by default (privacy-first)
            token_budget: 4096,
            window_size: 20,
            max_windows: 5,
        }
    }
}

impl WindowConfig {
    /// Load from config.toml [ai] section, silently falling back to defaults on any error.
    pub fn load() -> Self {
        let cfg = crate::storage::config::GwenConfig::load();
        Self {
            enabled:      cfg.ai.compression,
            token_budget: cfg.ai.token_budget as usize,
            window_size:  20,
            max_windows:  5,
        }
    }
}

// ─── Output type ─────────────────────────────────────────────────────────────

/// A contiguous slice of a file that was selected as relevant to the query.
pub struct RelevanceWindow {
    /// First included line, 0-indexed.
    pub start_line: usize,
    /// Last included line, 0-indexed, inclusive.
    pub end_line: usize,
    /// Relevance score from TF scoring; higher = more relevant.
    pub score: f32,
    /// The verbatim lines `start_line..=end_line` joined by `\n`.
    pub content: String,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Direction { Up, Down }

fn is_definition_line(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("fn ")          || t.starts_with("pub fn ")
        || t.starts_with("async fn ")     || t.starts_with("pub async fn ")
        || t.starts_with("impl ")         || t.starts_with("pub impl ")
        || t.starts_with("struct ")       || t.starts_with("pub struct ")
        || t.starts_with("enum ")         || t.starts_with("pub enum ")
        || t.starts_with("trait ")        || t.starts_with("pub trait ")
        || t.starts_with("mod ")          || t.starts_with("pub mod ")
}

// ─── TF scoring ───────────────────────────────────────────────────────────────

fn score_line(line: &str, query_terms: &[&str]) -> f32 {
    if line.trim().is_empty() {
        return 0.0;
    }

    let lower = line.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let word_count = words.len().max(1) as f32;

    let mut score = 0.0f32;
    for &term in query_terms {
        let count = lower.matches(term).count() as f32;
        if count > 0.0 {
            score += count / word_count; // TF normalised by line length
            if words.contains(&term) {
                score += 0.5; // bonus for exact whole-word match
            }
        }
    }

    // Bonus for definition lines: likely high-value anchors in the codebase.
    if score > 0.0 && is_definition_line(line) {
        score *= 1.5;
    }

    score
}

/// Tokenise `query` into lower-cased, deduped terms with stopwords removed.
pub fn extract_query_terms(query: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were",
        "what", "why", "how", "when", "where", "which",
        "it", "its", "this", "that", "these", "those",
        "in", "on", "at", "to", "for", "of", "and", "or",
        "do", "does", "did", "not", "be", "been", "being",
        "i", "you", "we", "he", "she", "they", "with",
    ];

    let mut seen: HashSet<String> = HashSet::new();
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 1 && !STOPWORDS.contains(t))
        .filter_map(|t| {
            let s = t.to_string();
            if seen.insert(s.clone()) { Some(s) } else { None }
        })
        .collect()
}

// ─── Function boundary detection ─────────────────────────────────────────────

/// Extend a raw window boundary toward a function-level boundary.
///
/// Direction::Up   — scan backward to the nearest `fn`/`impl`/`struct`/… line.
/// Direction::Down — scan forward until net brace count (from `target`) goes negative,
///                   meaning we have exited the enclosing block.
fn find_function_boundary(
    lines: &[&str],
    target: usize,        // raw_start or raw_end (0-indexed)
    direction: Direction,
    max_scan: usize,      // window_size * 2
) -> usize {
    match direction {
        Direction::Up => {
            let floor = target.saturating_sub(max_scan);
            for i in (floor..=target).rev() {
                if is_definition_line(lines[i]) {
                    return i;
                }
            }
            target // no definition found nearby; keep original boundary
        }
        Direction::Down => {
            let ceiling = (target + max_scan).min(lines.len().saturating_sub(1));
            let mut open: i32 = 0;
            let mut close: i32 = 0;
            for i in target..=ceiling {
                for ch in lines[i].chars() {
                    match ch {
                        '{' => open += 1,
                        '}' => close += 1,
                        _ => {}
                    }
                }
                // More closes than opens from target → exited the enclosing block.
                // Also handle case where a complete block was opened then closed.
                if close > open || (close == open && open > 0) {
                    return i;
                }
            }
            ceiling
        }
    }
}

// ─── Window operations ────────────────────────────────────────────────────────

fn merge_windows(windows: Vec<(usize, usize, f32)>) -> Vec<(usize, usize, f32)> {
    let mut merged: Vec<(usize, usize, f32)> = Vec::new();
    for (start, end, score) in windows {
        match merged.last_mut() {
            Some(last) if start <= last.1 + 1 => {
                last.1 = last.1.max(end);
                last.2 = last.2.max(score);
            }
            _ => merged.push((start, end, score)),
        }
    }
    merged
}

fn estimate_tokens(text: &str) -> usize {
    text.len() / 4 // rough: ~4 chars per token (GPT-style)
}

fn apply_token_budget(
    mut windows: Vec<(usize, usize, f32, String)>,
    budget: usize,
) -> Vec<(usize, usize, f32, String)> {
    // Highest-score window always included; rest pruned if budget exceeded.
    windows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut used: usize = 0;
    let mut kept: Vec<(usize, usize, f32, String)> = Vec::new();

    for w in windows {
        let cost = estimate_tokens(&w.3);
        if kept.is_empty() || used + cost <= budget {
            // @KEEP — always include at least one window (the highest-score)
            used += cost;
            kept.push(w);
        }
    }

    kept.sort_by_key(|w| w.0); // restore start_line order for correct output
    kept
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Extract relevant windows from `file_content` using TF scoring against `query`.
///
/// Returns a **single full-content window** when `config.enabled == false` so
/// callers never need to branch on the feature flag — the return type is always
/// `Vec<RelevanceWindow>`.
pub fn extract_relevant_windows(
    file_content: &str,
    query: &str,
    config: &WindowConfig,
) -> Vec<RelevanceWindow> {
    let lines: Vec<&str> = file_content.lines().collect();
    let total = lines.len();

    // Passthrough: disabled or empty file
    if !config.enabled || total == 0 {
        return vec![RelevanceWindow {
            start_line: 0,
            end_line: total.saturating_sub(1),
            score: 1.0,
            content: file_content.to_string(),
        }];
    }

    let terms: Vec<String> = extract_query_terms(query);
    if terms.is_empty() {
        return vec![RelevanceWindow {
            start_line: 0,
            end_line: total.saturating_sub(1),
            score: 1.0,
            content: file_content.to_string(),
        }];
    }

    let term_refs: Vec<&str> = terms.iter().map(String::as_str).collect();

    // Score each line, collect non-zero hits sorted by score descending
    let mut scored: Vec<(usize, f32)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let s = score_line(line, &term_refs);
            if s > 0.0 { Some((i, s)) } else { None }
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_scan = config.window_size * 2;

    // Expand each seed line to a function-boundary-aware window
    let mut raw: Vec<(usize, usize, f32)> = scored
        .into_iter()
        .take(config.max_windows)
        .map(|(line_idx, score)| {
            let raw_start = line_idx.saturating_sub(config.window_size);
            let raw_end   = (line_idx + config.window_size).min(total.saturating_sub(1));

            let start = find_function_boundary(&lines, raw_start, Direction::Up,   max_scan);
            let end   = find_function_boundary(&lines, raw_end,   Direction::Down, max_scan);

            (start, end, score)
        })
        .collect();

    raw.sort_by_key(|w| w.0);
    let merged = merge_windows(raw);

    // Attach content strings, then apply token budget
    let with_content: Vec<(usize, usize, f32, String)> = merged
        .into_iter()
        .map(|(s, e, sc)| {
            let content = lines[s..=e.min(total - 1)].join("\n");
            (s, e, sc, content)
        })
        .collect();

    apply_token_budget(with_content, config.token_budget)
        .into_iter()
        .map(|(start, end, score, content)| RelevanceWindow { start_line: start, end_line: end, score, content })
        .collect()
}

/// Format windowed output for injection into the API message payload.
///
/// Produces human-readable compressed representation with omission markers,
/// matching the display format expected by mistral.rs context injection.
pub fn format_windowed_output(
    file_path: &str,
    file_content: &str,
    windows: &[RelevanceWindow],
) -> String {
    if windows.is_empty() {
        return format!("[File: {file_path}]\n{file_content}");
    }

    let total_lines = file_content.lines().count();
    let mut out = format!("[File: {file_path} — showing relevant sections]\n\n");
    let mut prev_end: Option<usize> = None; // 0-indexed, last included line

    for window in windows {
        // ── Omission marker before this window ───────────────────────────────
        match prev_end {
            None if window.start_line > 0 => {
                // Before first window: lines 1 .. window.start_line (1-indexed)
                out.push_str(&format!(
                    "... (lines 1-{} omitted) ...\n\n",
                    window.start_line   // 0-indexed == 1-indexed of last omitted line (see note below)
                ));
                // Note: 0-indexed line N is 1-indexed line N+1, but the LAST omitted line
                // is the one before window start: 0-indexed (start-1) = 1-indexed start. ✓
            }
            Some(pe) if window.start_line > pe + 1 => {
                // Between windows: (pe+2) .. window.start_line  (1-indexed)
                out.push_str(&format!(
                    "... (lines {}-{} omitted) ...\n\n",
                    pe + 2,             // first omitted (1-indexed): pe is 0-indexed inclusive → pe+1 is next → pe+2 is 1-indexed
                    window.start_line   // last omitted (same invariant as above)
                ));
            }
            _ => {}
        }

        // ── Window content ────────────────────────────────────────────────────
        out.push_str(&format!(
            "[Line {}-{}]\n",
            window.start_line + 1,
            window.end_line + 1
        ));
        out.push_str(&window.content);
        out.push_str("\n\n");

        prev_end = Some(window.end_line);
    }

    // ── Trailing omission marker ──────────────────────────────────────────────
    if let Some(pe) = prev_end {
        if pe + 1 < total_lines {
            out.push_str("... (rest of file omitted) ...\n");
        }
    }

    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_config() -> WindowConfig {
        WindowConfig { enabled: true, ..Default::default() }
    }

    #[test]
    fn test_relevant_lines_scored_higher() {
        let content = "fn foo() {}\nfn authenticate(u: &str) {}\nfn bar() {}";
        let windows = extract_relevant_windows(content, "authenticate error", &enabled_config());
        assert!(!windows.is_empty(), "should find at least one window");
        assert!(
            windows[0].content.contains("authenticate"),
            "highest-scored window should contain the query term"
        );
    }

    #[test]
    fn test_window_expands_around_match() {
        let mut lines: Vec<String> = (0..100).map(|i| format!("// line {i}")).collect();
        lines[50] = "fn authenticate(u: &str) {}".to_string();
        let content = lines.join("\n");
        let config = WindowConfig { enabled: true, window_size: 20, ..Default::default() };
        let windows = extract_relevant_windows(&content, "authenticate", &config);
        let w = &windows[0];
        assert!(w.start_line <= 30, "start {} should be ≤ 30", w.start_line);
        assert!(w.end_line   >= 50, "end {}   should be ≥ 50", w.end_line);
    }

    #[test]
    fn test_no_cut_inside_function() {
        // Function fn do_auth() spans lines 50-60; query match is on line 55.
        let mut lines: Vec<String> = (0..100).map(|i| format!("// line {i}")).collect();
        lines[50] = "fn do_auth() {".to_string();
        lines[55] = "    let token = authenticate(user);".to_string();
        lines[60] = "}".to_string();
        let content = lines.join("\n");
        let config = WindowConfig {
            enabled: true,
            window_size: 2, // tiny window forces boundary extension
            ..Default::default()
        };
        let windows = extract_relevant_windows(&content, "authenticate", &config);
        assert!(
            windows[0].end_line >= 60,
            "window end {} should extend to or past closing brace at line 60",
            windows[0].end_line
        );
    }

    #[test]
    fn test_token_budget_respected() {
        // Large repetitive file; windowed output should stay within budget.
        let line = "fn process(input: &str) -> String { input.to_string() }\n";
        let content = line.repeat(500);
        let config = WindowConfig {
            enabled: true,
            token_budget: 256,
            window_size: 20,
            max_windows: 5,
        };
        let windows = extract_relevant_windows(&content, "process input", &config);
        let total_chars: usize = windows.iter().map(|w| w.content.len()).sum();
        assert!(
            total_chars <= config.token_budget * 4,
            "total chars {total_chars} exceeds token_budget * 4 = {}",
            config.token_budget * 4
        );
    }

    #[test]
    fn test_disabled_returns_full_content() {
        let content = "fn foo() {}\nfn bar() {}\n".repeat(100);
        let config = WindowConfig { enabled: false, ..Default::default() };
        let windows = extract_relevant_windows(&content, "foo bar", &config);
        assert_eq!(windows.len(), 1, "disabled should return exactly one window");
        assert_eq!(windows[0].content, content, "disabled should return full content unchanged");
    }

    #[test]
    fn test_query_terms_extracted_correctly() {
        let terms = extract_query_terms("why is authenticate() failing?");
        assert!(terms.contains(&"authenticate".to_string()), "should include 'authenticate'");
        assert!(!terms.contains(&"why".to_string()),         "should filter stopword 'why'");
        assert!(!terms.contains(&"is".to_string()),          "should filter stopword 'is'");
    }

    #[test]
    fn test_format_shows_omission_markers() {
        // Window in the middle of a 100-line file should produce omission markers.
        let lines: Vec<String> = (0..100).map(|i| format!("line {i}")).collect();
        let content = lines.join("\n");
        let windows = vec![RelevanceWindow {
            start_line: 45,
            end_line: 55,
            score: 1.0,
            content: lines[45..=55].join("\n"),
        }];
        let out = format_windowed_output("src/foo.rs", &content, &windows);
        assert!(out.contains("omitted"), "should contain omission markers");
        assert!(out.contains("[Line 46-56]"), "should show 1-indexed line range");
    }
}
