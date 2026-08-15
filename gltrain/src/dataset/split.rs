use crate::dataset::schema::{load_rows_from_path, GwenDatasetRow, LoadedLine};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct SplitOptions {
    pub input: PathBuf,
    pub train: f32,
    pub val: f32,
    pub test: Option<f32>,
    pub seed: Option<u64>,
    pub overwrite: bool,
}

pub struct SplitResult {
    pub seed_used: u64,
    pub train_path: PathBuf,
    pub train_count: usize,
    pub val_path: PathBuf,
    pub val_count: usize,
    pub test_path: Option<PathBuf>,
    pub test_count: Option<usize>,
    pub skipped: usize,
    pub warnings: Vec<String>,
}

pub fn run_split(opts: &SplitOptions) -> Result<SplitResult, String> {
    // Validate ratios sum to 1.0 ± 0.001.
    let test_ratio = opts.test.unwrap_or(0.0);
    let total = opts.train + opts.val + test_ratio;
    if (total - 1.0f32).abs() > 0.001 {
        return Err(format!(
            "ratios must sum to 1.0 (got {:.4}); adjust --train / --val / --test",
            total
        ));
    }

    // Derive output paths.
    let parent = opts.input.parent().unwrap_or(Path::new("."));
    let stem = opts
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");

    let train_path = parent.join(format!("{}_train.jsonl", stem));
    let val_path = parent.join(format!("{}_val.jsonl", stem));
    let test_path = opts.test.map(|_| parent.join(format!("{}_test.jsonl", stem)));

    // Guard against overwriting.
    if !opts.overwrite {
        let mut collisions: Vec<String> = Vec::new();
        if train_path.exists() {
            collisions.push(train_path.display().to_string());
        }
        if val_path.exists() {
            collisions.push(val_path.display().to_string());
        }
        if let Some(tp) = &test_path {
            if tp.exists() {
                collisions.push(tp.display().to_string());
            }
        }
        if !collisions.is_empty() {
            return Err(format!(
                "{} already exists.\n  Use --overwrite / -w to force overwrite.",
                collisions[0]
            ));
        }
    }

    // Load rows, collecting warnings for skipped lines.
    let loaded = load_rows_from_path(&opts.input)?;
    let mut warnings: Vec<String> = Vec::new();
    let mut rows: Vec<GwenDatasetRow> = Vec::new();
    let mut skipped = 0usize;

    for entry in loaded {
        match entry {
            LoadedLine::Row { row, .. } => rows.push(row),
            LoadedLine::Skipped { line_no, reason } => {
                warnings.push(format!("⚠ Line {}: skipped ({})", line_no, reason));
                skipped += 1;
            }
        }
    }

    // Shuffle.
    let seed = opts.seed.unwrap_or_else(|| rand::thread_rng().r#gen::<u64>());
    let mut rng = StdRng::seed_from_u64(seed);
    rows.shuffle(&mut rng);

    let n = rows.len();
    let train_end = ((opts.train as f64 * n as f64).round() as usize).min(n);
    let val_end = if opts.test.is_some() {
        let v = train_end + ((opts.val as f64 * n as f64).round() as usize);
        v.min(n)
    } else {
        n // val gets everything remaining when there's no test split
    };

    let train_rows = &rows[..train_end];
    let val_rows = &rows[train_end..val_end];
    let test_rows = &rows[val_end..];

    write_rows(&train_path, train_rows)?;
    write_rows(&val_path, val_rows)?;
    if let Some(tp) = &test_path {
        write_rows(tp, test_rows)?;
    }

    Ok(SplitResult {
        seed_used: seed,
        train_path,
        train_count: train_rows.len(),
        val_path,
        val_count: val_rows.len(),
        test_path,
        test_count: opts.test.map(|_| test_rows.len()),
        skipped,
        warnings,
    })
}

fn write_rows(path: &Path, rows: &[GwenDatasetRow]) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|e| format!("cannot create '{}': {}", path.display(), e))?;
    let mut writer = std::io::BufWriter::new(file);

    for row in rows {
        let mut obj = serde_json::Map::new();
        obj.insert("input".into(), serde_json::Value::String(row.input.clone()));
        obj.insert("output".into(), serde_json::Value::String(row.output.clone()));
        if let Some(cat) = &row.category {
            obj.insert("category".into(), serde_json::Value::String(cat.clone()));
        }
        let line = serde_json::to_string(&serde_json::Value::Object(obj))
            .map_err(|e| format!("serialization error: {}", e))?;
        writeln!(writer, "{}", line).map_err(|e| format!("write error: {}", e))?;
    }

    writer.flush().map_err(|e| format!("flush error: {}", e))?;
    Ok(())
}
