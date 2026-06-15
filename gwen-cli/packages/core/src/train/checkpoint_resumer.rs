// train/checkpoint_resumer.rs — GWEN-222 checkpoint discovery + load.
//
// Resolves a `ResumeMode` into a concrete checkpoint path + a restored optimiser
// step counter, and loads adapter weights from a SafeTensors checkpoint into an
// existing VarMap.
//
// @INFO Restores LoRA adapter weights ONLY. AdamW optimiser state (momentum /
// variance) is never persisted, so a short momentum warm-up occurs after a
// resume — this is expected, not a bug.
//
// @DANGER `load_checkpoint_into_varmap` relies on `VarMap::load`, which only
// populates Vars that ALREADY EXIST in the map (it iterates the map's vars and
// looks each up in the file). The adapter Vars are created inside
// `LayeredTrainingLoop::new`, so a checkpoint must be loaded AFTER construction,
// not into an empty VarMap.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use candle_nn::VarMap;

use crate::train::config::ResumeMode;

/// Resolve a `ResumeMode` into `(checkpoint_path, initial_step)`.
///
/// - `None`     → `(None, 0)` — fresh start, no discovery.
/// - `Auto`     → newest `checkpoint_*.safetensors` in `output_path`
///                (lexicographic max). Missing dir / empty dir → warn, `(None, 0)`.
/// - `Explicit` → the given path if it exists, else `bail!`.
pub fn resolve_checkpoint(
    mode: &ResumeMode,
    output_path: &Path,
) -> Result<(Option<PathBuf>, usize)> {
    match mode {
        ResumeMode::None => Ok((None, 0)),

        ResumeMode::Auto => {
            let mut checkpoints: Vec<PathBuf> = match std::fs::read_dir(output_path) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| is_checkpoint_file(p))
                    .collect(),
                // A missing output dir is the normal "first run" case — treat it
                // exactly like an empty dir rather than erroring.
                Err(_) => Vec::new(),
            };
            checkpoints.sort();
            match checkpoints.last() {
                None => {
                    eprintln!(
                        "[resume] warning: no checkpoint_*.safetensors found in '{}'; starting from step 0",
                        output_path.display()
                    );
                    Ok((None, 0))
                }
                Some(path) => {
                    let step = parse_step_from_filename(path);
                    eprintln!(
                        "[resume] auto-resume from '{}' (step {})",
                        path.display(),
                        step
                    );
                    Ok((Some(path.clone()), step))
                }
            }
        }

        ResumeMode::Explicit(p) => {
            if p.exists() {
                let step = parse_step_from_filename(p);
                eprintln!("[resume] resume from '{}' (step {})", p.display(), step);
                Ok((Some(p.clone()), step))
            } else {
                bail!("checkpoint path does not exist: {}", p.display());
            }
        }
    }
}

/// True if `path`'s file name matches `checkpoint_*.safetensors`.
fn is_checkpoint_file(path: &Path) -> bool {
    match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => name.starts_with("checkpoint_") && name.ends_with(".safetensors"),
        None => false,
    }
}

/// Parse the optimiser step from a `checkpoint_{NNNNNN}.safetensors` filename.
///
/// Strips the directory, the `.safetensors` suffix, and the `checkpoint_`
/// prefix, then parses the remainder as `usize`. On any failure returns `0`
/// (after a warning) rather than erroring — a non-standard name should not abort
/// a resume, it just restarts the step counter.
pub fn parse_step_from_filename(path: &Path) -> usize {
    let parsed = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(".safetensors"))
        .and_then(|n| n.strip_prefix("checkpoint_"))
        .and_then(|s| s.parse::<usize>().ok());

    match parsed {
        Some(step) => step,
        None => {
            eprintln!(
                "[resume] warning: could not parse step from '{}'; step counter will restart from 0",
                path.display()
            );
            0
        }
    }
}

/// Load a SafeTensors checkpoint into an existing `VarMap`.
///
/// Wraps `VarMap::load`, which matches tensors by name into the VarMap's
/// existing Vars. The VarMap must already contain the adapter Vars (created by
/// `LayeredTrainingLoop::new`); loading into an empty VarMap is a silent no-op.
pub fn load_checkpoint_into_varmap(varmap: &mut VarMap, path: &Path) -> Result<()> {
    varmap
        .load(path)
        .map_err(|e| anyhow::anyhow!("VarMap::load failed for '{}': {}", path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn touch(dir: &Path, name: &str) {
        std::fs::File::create(dir.join(name)).expect("create temp checkpoint file");
    }

    /// Property 3: Auto-discovery selects the lexicographically greatest checkpoint.
    #[test]
    fn test_resolve_auto_picks_lex_max() {
        let dir = TempDir::new().unwrap();
        touch(dir.path(), "checkpoint_000000.safetensors");
        touch(dir.path(), "checkpoint_000500.safetensors");
        touch(dir.path(), "checkpoint_001000.safetensors");

        let (path, step) =
            resolve_checkpoint(&ResumeMode::Auto, dir.path()).expect("auto resolve");
        let path = path.expect("a checkpoint should be found");
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            "checkpoint_001000.safetensors"
        );
        assert_eq!(step, 1000);
    }

    /// Property 5: Step counter parsed correctly from any valid checkpoint filename.
    #[test]
    fn test_parse_step_roundtrip() {
        for n in [0usize, 500, 1000, 999999] {
            let name = format!("checkpoint_{n:06}.safetensors");
            let path = PathBuf::from(name);
            assert_eq!(parse_step_from_filename(&path), n, "round-trip for n={n}");
        }
    }

    #[test]
    fn test_resolve_auto_empty_dir() {
        let dir = TempDir::new().unwrap();
        let result = resolve_checkpoint(&ResumeMode::Auto, dir.path()).expect("empty dir is Ok");
        assert_eq!(result, (None, 0));
    }

    #[test]
    fn test_parse_step_nonstandard() {
        let path = PathBuf::from("arbitrary_name.safetensors");
        // Must not panic, must return 0.
        assert_eq!(parse_step_from_filename(&path), 0);
    }

    /// Property 2: Non-existent explicit checkpoint path fails before training.
    #[test]
    fn test_explicit_path_missing() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("checkpoint_000042.safetensors");
        let result = resolve_checkpoint(&ResumeMode::Explicit(missing), dir.path());
        assert!(result.is_err(), "missing explicit path must Err");
    }

    /// `None` mode never touches the filesystem and always starts fresh.
    #[test]
    fn test_resolve_none_is_fresh_start() {
        let dir = TempDir::new().unwrap();
        touch(dir.path(), "checkpoint_009999.safetensors");
        let result =
            resolve_checkpoint(&ResumeMode::None, dir.path()).expect("None resolve is Ok");
        assert_eq!(result, (None, 0), "None mode ignores existing checkpoints");
    }
}
