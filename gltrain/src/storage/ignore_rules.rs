// @INFO: GwenLand Ignore Rules Parser
// This module implements .gwenignore parsing using gitignore-compatible semantics
// built on top of the `ignore` crate.

// @EDITABLE
use std::fs;
use std::path::{Path, PathBuf};
use ignore::gitignore::{Gitignore, GitignoreBuilder};

/// Wrapper around the gitignore implementation for GwenLand
#[derive(Debug, Clone)]
pub struct GwenIgnore {
    pub root: PathBuf,
    pub gitignore: Gitignore,
}

/// Loads .gwenignore rules from the workspace root directory.
/// If no .gwenignore exists, a default template is generated automatically.
pub fn load_ignore_rules(root: &Path) -> GwenIgnore {
    let ignore_path = root.join(".gwenignore");

    // @DANGER: Automatically writing configuration files back to the workspace.
    // Ensure root is writable and path is valid to prevent silent I/O panics.
    if !ignore_path.exists() {
        let default_rules = b"# GwenLand ignore rules
# Syntax is identical to .gitignore

node_modules/
target/
dist/
.next/
.turbo/
*.lock
*.log
.env*
";
        if let Err(e) = fs::write(&ignore_path, default_rules) {
            eprintln!("[warn] Failed to write default .gwenignore to {}: {}", ignore_path.display(), e);
        }
    }

    let mut builder = GitignoreBuilder::new(root);
    if let Some(err) = builder.add(&ignore_path) {
        eprintln!("[warn] Failed to parse .gwenignore: {}", err);
    }

    let gitignore = builder.build().unwrap_or_else(|err| {
        eprintln!("[error] Failed to build gitignore matcher: {}", err);
        // Fallback to empty matcher on error
        GitignoreBuilder::new(root).build().unwrap()
    });

    GwenIgnore {
        root: root.to_path_buf(),
        gitignore,
    }
}

/// Check if a path should be ignored according to loaded GwenIgnore rules.
/// Handles absolute and relative paths gracefully.
pub fn should_ignore(path: &Path, rules: &GwenIgnore) -> bool {
    let relative_path = if path.is_absolute() {
        match path.strip_prefix(&rules.root) {
            Ok(rel) => rel,
            Err(_) => path,
        }
    } else {
        path
    };

    let is_dir = path.is_dir();
    rules.gitignore.matched(relative_path, is_dir).is_ignore()
}
