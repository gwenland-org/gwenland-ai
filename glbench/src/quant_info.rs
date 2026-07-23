//! `glbench quant-info` — static inspection of a `.gllm` package.
//!
//! Zero inference, zero model load: this reads `gllm.json` and tallies tensor
//! dtypes. glbench forbids external dependencies (DESIGN.md), so parsing
//! reuses the crate's existing hand-rolled [`crate::export::json`] reader
//! rather than pulling in `serde_json`.
//!
//! glbench does not import glictus-caliburni, so this is *a* reader of the
//! `.gllm` manifest shape, not *the* reader — it knows only the fields it
//! needs (dtype tally + a few identifying facts) and treats everything else
//! as opaque. The real package layout (`constants.rs`,
//! `manifest::GllmManifest` in glictus-caliburni) is a directory containing
//! `gllm.json`, `GLLMShared.gllm`, `GLLMTensorLayer-NNNN.gllm`, and an
//! optional `GLLMProj.gllm` — there is no ZIP container yet (ARTX06 tracks
//! that work), so `--model` names the package *directory*, not an archive.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::export::json::Json;

/// CLI arguments for `quant-info`.
pub struct QuantInfoArgs {
    pub model: PathBuf,
    pub out: Option<PathBuf>,
}

/// The JSON-serializable summary of one `quant-info` inspection, kept around
/// for a future `quant-compare` (Wave 2+).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantInfoSession {
    pub path: String,
    pub file_size_bytes: u64,
    pub file_size_mb: f64,
    pub model_name: String,
    pub architecture: String,
    pub format_version: String,
    pub num_layers: u32,
    pub total_tensors: usize,
    pub quantized_tensors: usize,
    pub coverage_pct: f64,
    pub dtype_tally: BTreeMap<String, usize>,
    pub timestamp: String,
}

impl QuantInfoSession {
    pub fn to_json(&self) -> Json {
        let tally = self
            .dtype_tally
            .iter()
            .map(|(k, v)| (k.clone(), Json::n(*v as f64)))
            .collect::<BTreeMap<_, _>>();
        Json::obj([
            ("path", Json::s(self.path.clone())),
            ("file_size_bytes", Json::n(self.file_size_bytes as f64)),
            ("file_size_mb", Json::n(self.file_size_mb)),
            ("model_name", Json::s(self.model_name.clone())),
            ("architecture", Json::s(self.architecture.clone())),
            ("format_version", Json::s(self.format_version.clone())),
            ("num_layers", Json::n(self.num_layers as f64)),
            ("total_tensors", Json::n(self.total_tensors as f64)),
            ("quantized_tensors", Json::n(self.quantized_tensors as f64)),
            ("coverage_pct", Json::n(self.coverage_pct)),
            ("dtype_tally", Json::Obj(tally)),
            ("timestamp", Json::s(self.timestamp.clone())),
        ])
    }

    pub fn from_json(v: &Json) -> Result<QuantInfoSession, String> {
        let dtype_tally = v
            .get("dtype_tally")
            .and_then(|t| t.as_obj())
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0) as usize))
                    .collect()
            })
            .unwrap_or_default();

        Ok(QuantInfoSession {
            path: v.get("path").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            file_size_bytes: v.get("file_size_bytes").and_then(|n| n.as_f64()).unwrap_or(0.0) as u64,
            file_size_mb: v.get("file_size_mb").and_then(|n| n.as_f64()).unwrap_or(0.0),
            model_name: v.get("model_name").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            architecture: v.get("architecture").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            format_version: v.get("format_version").and_then(|s| s.as_str()).unwrap_or("").to_string(),
            num_layers: v.get("num_layers").and_then(|n| n.as_f64()).unwrap_or(0.0) as u32,
            total_tensors: v.get("total_tensors").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            quantized_tensors: v.get("quantized_tensors").and_then(|n| n.as_f64()).unwrap_or(0.0) as usize,
            coverage_pct: v.get("coverage_pct").and_then(|n| n.as_f64()).unwrap_or(0.0),
            dtype_tally,
            timestamp: v.get("timestamp").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        })
    }
}

/// One tensor's name + dtype string, as read from `gllm.json` — enough for
/// the tally, nothing more.
struct TensorFact {
    dtype: String,
}

/// dtype strings glbench treats as "kept" (not quantized). Everything else
/// found in a tensor's `dtype` field counts toward `quantized_tensors`.
const UNQUANTIZED_DTYPES: [&str; 2] = ["F32", "F16"];

/// Run `glbench quant-info --model <path> [--out <file.json>]`.
pub fn run_quant_info(args: QuantInfoArgs) -> Result<(), String> {
    let manifest_path = resolve_manifest_path(&args.model)?;
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("reading {}: {e}", manifest_path.display()))?;
    let manifest = crate::export::json::parse(&text)
        .map_err(|e| format!("parsing {}: {e}", manifest_path.display()))?;

    let file_size_bytes = package_size_bytes(&args.model, &manifest_path)?;
    let file_size_mb = file_size_bytes as f64 / 1024.0 / 1024.0;

    let model_name = manifest
        .get("model_id")
        .and_then(|s| s.as_str())
        .unwrap_or("(unknown)")
        .to_string();
    let architecture = manifest
        .get("architecture")
        .and_then(|s| s.as_str())
        .unwrap_or("(unknown)")
        .to_string();
    let format_version = manifest
        .get("gllm_version")
        .and_then(|s| s.as_str())
        .unwrap_or("(unknown)")
        .to_string();
    let num_layers = manifest
        .get("metadata")
        .and_then(|m| m.get("num_layers"))
        .and_then(|n| n.as_f64())
        .unwrap_or(0.0) as u32;

    let tensors = collect_tensors(&manifest);
    let total_tensors = tensors.len();
    let mut dtype_tally: BTreeMap<String, usize> = BTreeMap::new();
    let mut quantized_tensors = 0usize;
    for t in &tensors {
        *dtype_tally.entry(t.dtype.clone()).or_insert(0) += 1;
        if !UNQUANTIZED_DTYPES.contains(&t.dtype.as_str()) {
            quantized_tensors += 1;
        }
    }
    let coverage_pct = if total_tensors == 0 {
        0.0
    } else {
        quantized_tensors as f64 / total_tensors as f64 * 100.0
    };

    let warnings = manifest
        .get("warnings")
        .or_else(|| manifest.get("conversion_warnings"))
        .and_then(|w| w.as_arr())
        .map(|a| a.len());

    let session = QuantInfoSession {
        path: args.model.display().to_string(),
        file_size_bytes,
        file_size_mb,
        model_name,
        architecture,
        format_version,
        num_layers,
        total_tensors,
        quantized_tensors,
        coverage_pct,
        dtype_tally,
        timestamp: iso8601_now(),
    };

    print!("{}", render_table(&session, warnings));

    if let Some(out) = &args.out {
        std::fs::write(out, session.to_json().to_pretty())
            .map_err(|e| format!("writing {}: {e}", out.display()))?;
        eprintln!("wrote {}", out.display());
    }

    Ok(())
}

/// Locate `gllm.json` under `--model`.
///
/// Real `.gllm` packages are directories (see module docs): if `model` is a
/// directory, its `gllm.json` is used directly. A `.gllm`-suffixed *file*
/// starting with the ZIP local-file magic (`PK\x03\x04`) is a real archive —
/// glictus-caliburni itself does not read those yet (ARTX06), so this bails
/// with the same honesty rather than guessing at an unimplemented format.
fn resolve_manifest_path(model: &std::path::Path) -> Result<PathBuf, String> {
    if model.is_dir() {
        let manifest = model.join("gllm.json");
        if !manifest.is_file() {
            return Err(format!("{}: no gllm.json found in package directory", model.display()));
        }
        return Ok(manifest);
    }
    if model.is_file() {
        if let Ok(bytes) = std::fs::read(model) {
            if bytes.starts_with(b"PK\x03\x04") {
                return Err(format!(
                    "{}: ZIP-archive .gllm packages are not yet readable (glictus-caliburni ARTX06); \
                     point --model at the extracted package directory instead",
                    model.display()
                ));
            }
        }
        return Err(format!("{}: expected a .gllm package directory, not a file", model.display()));
    }
    Err(format!("{}: path does not exist", model.display()))
}

/// Total on-disk size of the package: every regular file in the directory,
/// summed, so the report reflects the whole package rather than only the
/// manifest's byte count.
fn package_size_bytes(model: &std::path::Path, manifest_path: &std::path::Path) -> Result<u64, String> {
    if model.is_dir() {
        let mut total = 0u64;
        for entry in std::fs::read_dir(model).map_err(|e| format!("reading {}: {e}", model.display()))? {
            let entry = entry.map_err(|e| format!("reading {}: {e}", model.display()))?;
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
        Ok(total)
    } else {
        std::fs::metadata(manifest_path)
            .map(|m| m.len())
            .map_err(|e| format!("reading {}: {e}", manifest_path.display()))
    }
}

/// Walk `shared.tensors`, every `layers[].tensors`, and `projector.tensors`
/// (when present) — the manifest's complete tensor inventory.
fn collect_tensors(manifest: &Json) -> Vec<TensorFact> {
    let mut out = Vec::new();
    let tensors_of = |unit: Option<&Json>, out: &mut Vec<TensorFact>| {
        let Some(list) = unit.and_then(|u| u.get("tensors")).and_then(|t| t.as_arr()) else {
            return;
        };
        for t in list {
            let dtype = t.get("dtype").and_then(|d| d.as_str()).unwrap_or("(missing)").to_string();
            out.push(TensorFact { dtype });
        }
    };

    tensors_of(manifest.get("shared"), &mut out);
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_arr()) {
        for layer in layers {
            tensors_of(Some(layer), &mut out);
        }
    }
    tensors_of(manifest.get("projector"), &mut out);

    out
}

/// Render the human-readable report.
fn render_table(s: &QuantInfoSession, warnings: Option<usize>) -> String {
    let mut out = String::new();
    let rule = "\u{2500}".repeat(52);

    out.push_str(&format!("glbench quant-info: {}\n", s.path));
    out.push_str(&format!("{rule}\n"));
    out.push_str(&format!("Model            {} ({})\n", s.model_name, s.architecture));
    out.push_str(&format!("Format           GLLM v{}\n", s.format_version));
    out.push_str(&format!("File size        {:.1} MB\n", s.file_size_mb));
    out.push_str(&format!("Active layers    {}\n", s.num_layers));
    out.push_str(&format!("{rule}\n"));
    out.push_str("Tensor coverage\n");
    out.push_str(&format!("Total tensors    {}\n", s.total_tensors));
    out.push_str(&format!(
        "Quantized        {} ({:.1}%)\n",
        s.quantized_tensors, s.coverage_pct
    ));
    for dtype in &UNQUANTIZED_DTYPES {
        if let Some(&count) = s.dtype_tally.get(*dtype) {
            out.push_str(&format!("{:<16} {}\n", format!("{dtype} (kept)"), count));
        }
    }
    out.push_str(&format!("{rule}\n"));
    out.push_str("Dtype tally\n");
    for (dtype, count) in &s.dtype_tally {
        if s.total_tensors == 0 {
            out.push_str(&format!("{dtype:<16} {count}\n"));
            continue;
        }
        let pct = *count as f64 / s.total_tensors as f64 * 100.0;
        out.push_str(&format!("{dtype:<16} {count} ({pct:.1}%)\n"));
    }
    if let Some(n) = warnings {
        out.push_str(&format!("{rule}\n"));
        out.push_str(&format!("Warnings         {n}\n"));
    }
    out.push_str(&format!("{rule}\n"));
    out
}

/// Manual ISO 8601 UTC timestamp (no `chrono` — DESIGN.md's zero-dep rule).
/// Precision to the second is enough for a session label.
fn iso8601_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    civil_from_unix(secs)
}

/// Convert Unix seconds to an ISO 8601 UTC string, using the same
/// days-since-epoch civil calendar algorithm as `environment` (Howard
/// Hinnant's `civil_from_days`), so glbench keeps exactly one implementation
/// of "no `chrono`" date math instead of a second one just for this report.
fn civil_from_unix(unix_secs: u64) -> String {
    let days = (unix_secs / 86400) as i64;
    let secs_of_day = unix_secs % 86400;
    let (hour, minute, second) = (secs_of_day / 3600, (secs_of_day % 3600) / 60, secs_of_day % 60);

    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_manifest_path_missing_dir_errors() {
        let err = resolve_manifest_path(std::path::Path::new("/no/such/path")).unwrap_err();
        assert!(err.contains("does not exist"));
    }

    #[test]
    fn iso8601_epoch_is_1970() {
        assert_eq!(civil_from_unix(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn iso8601_known_date() {
        // 2026-07-23T00:00:00Z
        assert_eq!(civil_from_unix(1_784_764_800), "2026-07-23T00:00:00Z");
    }
}
