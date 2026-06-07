// @INFO: Bundles base_train.py at compile time and writes it to a temp file at runtime.
// @EDITABLE: The asset path is relative to this file's crate root. Update if assets/ moves.
// @DANGER: Caller must hold the returned NamedTempFile for the lifetime of the subprocess.
//          Dropping it deletes the file on disk.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::Path;

const BASE_TRAIN_PY: &str = include_str!("../../assets/base_train.py");

/// Write the training script to a temp file and return it.
/// If `custom` is provided, uses that file's content instead of the bundled script.
pub fn write_train_script(custom: Option<&Path>) -> Result<tempfile::NamedTempFile> {
    let content = match custom {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("cannot read custom script: {}", path.display()))?,
        None => BASE_TRAIN_PY.to_string(),
    };

    let mut tmp = tempfile::Builder::new()
        .prefix("gwen_train_")
        .suffix(".py")
        .tempfile()
        .context("failed to create temp file for training script")?;

    tmp.write_all(content.as_bytes())
        .context("failed to write training script to temp file")?;

    Ok(tmp)
}
