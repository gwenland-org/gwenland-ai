use crate::dataset::schema::{load_rows_from_path, GwenDatasetRow, LoadedLine};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    GwenStyle,
    ChatML,
    Alpaca,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "gwenstyle" | "gwen" => Ok(OutputFormat::GwenStyle),
            "chatml" => Ok(OutputFormat::ChatML),
            "alpaca" => Ok(OutputFormat::Alpaca),
            "sharegpt" => Err(
                "ShareGPT is supported as an input format only; choose gwenstyle, chatml, or alpaca for output".into()
            ),
            other => Err(format!("unknown output format '{}'; valid: gwenstyle, chatml, alpaca", other)),
        }
    }
}

pub struct ConvertResult {
    pub written: usize,
    pub skipped: usize,
    pub warnings: Vec<String>,
}

pub fn run_convert(
    input_path: &Path,
    output_path: &Path,
    output_format: &OutputFormat,
) -> Result<ConvertResult, String> {
    let loaded = load_rows_from_path(input_path)?;

    let mut written = 0usize;
    let mut skipped = 0usize;
    let mut warnings: Vec<String> = Vec::new();

    let file = std::fs::File::create(output_path)
        .map_err(|e| format!("cannot create output file '{}': {}", output_path.display(), e))?;
    let mut writer = std::io::BufWriter::new(file);

    for entry in loaded {
        match entry {
            LoadedLine::Skipped { line_no, reason } => {
                warnings.push(format!("⚠ Line {}: skipped ({})", line_no, reason));
                skipped += 1;
            }
            LoadedLine::Row { row, .. } => {
                let json = serialize_row(&row, output_format);
                if let Err(e) = writeln!(writer, "{}", json) {
                    return Err(format!("write error: {}", e));
                }
                written += 1;
            }
        }
    }

    writer.flush().map_err(|e| format!("flush error: {}", e))?;
    Ok(ConvertResult { written, skipped, warnings })
}

fn serialize_row(row: &GwenDatasetRow, fmt: &OutputFormat) -> String {
    match fmt {
        OutputFormat::GwenStyle => {
            let mut obj = serde_json::Map::new();
            obj.insert("input".into(), serde_json::Value::String(row.input.clone()));
            obj.insert("output".into(), serde_json::Value::String(row.output.clone()));
            if let Some(cat) = &row.category {
                obj.insert("category".into(), serde_json::Value::String(cat.clone()));
            }
            serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default()
        }
        OutputFormat::ChatML => {
            let messages = serde_json::json!([
                {"role": "user",      "content": row.input},
                {"role": "assistant", "content": row.output},
            ]);
            let mut obj = serde_json::Map::new();
            obj.insert("messages".into(), messages);
            if let Some(cat) = &row.category {
                obj.insert("category".into(), serde_json::Value::String(cat.clone()));
            }
            serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default()
        }
        OutputFormat::Alpaca => {
            let mut obj = serde_json::Map::new();
            obj.insert("instruction".into(), serde_json::Value::String(row.input.clone()));
            obj.insert("input".into(), serde_json::Value::String(String::new()));
            obj.insert("output".into(), serde_json::Value::String(row.output.clone()));
            if let Some(cat) = &row.category {
                obj.insert("category".into(), serde_json::Value::String(cat.clone()));
            }
            serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default()
        }
    }
}
