use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum SourceFormat {
    GwenStyle,
    ChatML,
    Alpaca,
    ShareGPT,
}

#[derive(Debug, Clone)]
pub struct GwenDatasetRow {
    pub input: String,
    pub output: String,
    pub category: Option<String>,
    pub source_format: SourceFormat,
}

// ── ChatML ──────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// ── Alpaca ───────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RawAlpacaRow {
    pub instruction: String,
    #[serde(default)]
    pub input: String,
    pub output: String,
    pub category: Option<String>,
}

// ── ShareGPT ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ShareGptTurn {
    pub from: String,
    pub value: String,
}

#[derive(Deserialize)]
pub struct RawShareGptRow {
    pub conversations: Vec<ShareGptTurn>,
    pub category: Option<String>,
}

// ── Untagged raw row (GwenStyle + ChatML detected by serde) ──────────────────

#[derive(Deserialize)]
#[serde(untagged)]
pub enum RawDatasetRow {
    GwenStyle {
        input: String,
        output: String,
        category: Option<String>,
    },
    ChatML {
        messages: Vec<ChatMessage>,
    },
}

impl RawDatasetRow {
    pub fn into_gwen_row(self) -> Result<GwenDatasetRow, &'static str> {
        match self {
            RawDatasetRow::GwenStyle { input, output, category } => Ok(GwenDatasetRow {
                input,
                output,
                category,
                source_format: SourceFormat::GwenStyle,
            }),
            RawDatasetRow::ChatML { messages } => {
                let input = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "user")
                    .map(|m| m.content.clone())
                    .unwrap_or_default();

                let output = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "assistant")
                    .map(|m| m.content.clone())
                    .ok_or("no `assistant` message in ChatML messages array")?;

                Ok(GwenDatasetRow {
                    input,
                    output,
                    category: None,
                    source_format: SourceFormat::ChatML,
                })
            }
        }
    }
}

// ── Shared file loader ────────────────────────────────────────────────────────

/// One entry per non-empty line; corrupt lines carry the line number and reason.
pub enum LoadedLine {
    Row { line_no: usize, row: GwenDatasetRow },
    Skipped { line_no: usize, reason: String },
}

/// Detected input format, sniffed from the first non-empty line.
#[derive(Debug, Clone, PartialEq)]
pub enum DetectedFormat {
    GwenStyle,
    ChatML,
    Alpaca,
    ShareGPT,
    Unknown,
}

/// Sniff the format by inspecting keys present in the first valid JSON object.
pub fn detect_format(raw: &str) -> DetectedFormat {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(raw) else {
        return DetectedFormat::Unknown;
    };
    let obj = match val.as_object() {
        Some(o) => o,
        None => return DetectedFormat::Unknown,
    };
    if obj.contains_key("conversations") {
        DetectedFormat::ShareGPT
    } else if obj.contains_key("instruction") {
        DetectedFormat::Alpaca
    } else if obj.contains_key("messages") {
        DetectedFormat::ChatML
    } else if obj.contains_key("input") && obj.contains_key("output") {
        DetectedFormat::GwenStyle
    } else {
        DetectedFormat::Unknown
    }
}

/// Parse one raw JSON line into a GwenDatasetRow under a given detected format.
fn parse_line(raw: &str, fmt: &DetectedFormat) -> Result<GwenDatasetRow, String> {
    match fmt {
        DetectedFormat::GwenStyle | DetectedFormat::ChatML => {
            let row: RawDatasetRow = serde_json::from_str(raw)
                .map_err(|e| e.to_string())?;
            row.into_gwen_row().map_err(|e| e.to_string())
        }
        DetectedFormat::Alpaca => {
            let row: RawAlpacaRow = serde_json::from_str(raw)
                .map_err(|e| e.to_string())?;
            let combined_input = if row.input.trim().is_empty() {
                row.instruction
            } else {
                format!("{}\n{}", row.instruction, row.input)
            };
            Ok(GwenDatasetRow {
                input: combined_input,
                output: row.output,
                category: row.category,
                source_format: SourceFormat::Alpaca,
            })
        }
        DetectedFormat::ShareGPT => {
            let row: RawShareGptRow = serde_json::from_str(raw)
                .map_err(|e| e.to_string())?;
            let input = row
                .conversations
                .iter()
                .rev()
                .find(|t| t.from == "human")
                .map(|t| t.value.clone())
                .unwrap_or_default();
            let output = row
                .conversations
                .iter()
                .rev()
                .find(|t| t.from == "gpt")
                .map(|t| t.value.clone())
                .ok_or_else(|| "no `gpt` turn in ShareGPT conversations".to_string())?;
            Ok(GwenDatasetRow {
                input,
                output,
                category: row.category,
                source_format: SourceFormat::ShareGPT,
            })
        }
        DetectedFormat::Unknown => Err("unknown format".into()),
    }
}

/// Stream a JSONL file and return one `LoadedLine` per non-empty line.
/// Format is auto-detected from the first parseable line.
pub fn load_rows_from_path(path: &Path) -> Result<Vec<LoadedLine>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("cannot open '{}': {}", path.display(), e))?;
    let reader = BufReader::new(file);

    let mut rows: Vec<LoadedLine> = Vec::new();
    let mut fmt: Option<DetectedFormat> = None;

    for (idx, line_res) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let raw = match line_res {
            Ok(l) => l,
            Err(e) => {
                rows.push(LoadedLine::Skipped {
                    line_no,
                    reason: format!("read error: {}", e),
                });
                continue;
            }
        };
        if raw.trim().is_empty() {
            continue;
        }

        // Detect on first parseable line.
        let effective_fmt = match &fmt {
            Some(f) => f.clone(),
            None => {
                let detected = detect_format(&raw);
                fmt = Some(detected.clone());
                detected
            }
        };

        match parse_line(&raw, &effective_fmt) {
            Ok(row) => rows.push(LoadedLine::Row { line_no, row }),
            Err(reason) => rows.push(LoadedLine::Skipped { line_no, reason }),
        }
    }

    Ok(rows)
}
