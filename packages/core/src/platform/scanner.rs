// @INFO: GwenLand Workspace Scanner
// This module provides file discovery and indexing capabilities. It walks the
// workspace tree, filters out build/dependency directories, and detects binary files.

// @EDITABLE
use std::path::Path;
use std::time::Instant;
use walkdir::WalkDir;
use serde::Serialize;
use crate::storage::ignore_rules::{GwenIgnore, should_ignore};

/// Represents a scanned file in the workspace
#[derive(Debug, Clone, Serialize)]
pub struct FileEntry {
    pub path: String,        // relative path from workspace root
    pub size_bytes: u64,
    pub extension: Option<String>,
    pub is_binary: bool,
}

/// The aggregated result of scanning the workspace
#[derive(Debug, Clone, Serialize)]
pub struct ScanResult {
    pub files: Vec<FileEntry>,
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub scan_duration_ms: u64,
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

/// Recursively scans the workspace starting from the given root path.
/// Skips hidden directories (starting with `.`), build/dependency folders, and gwenignore rules.
pub fn scan_workspace(root: &Path, ignore: Option<&GwenIgnore>) -> ScanResult {
    let start_time = Instant::now();
    let mut files = Vec::new();
    let mut total_size_bytes = 0u64;

    let mut it = WalkDir::new(root).into_iter();
    loop {
        let entry = match it.next() {
            None => break,
            Some(Err(_)) => continue,
            Some(Ok(entry)) => entry,
        };

        let path = entry.path();
        
        // Skip hidden directories and dependency/build artifacts
        if entry.file_type().is_dir() {
            if path != root {
                let mut must_skip = false;
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "node_modules" || name == "target" || name == "dist" {
                        must_skip = true;
                    }
                }

                if !must_skip {
                    if let Some(rules) = ignore {
                        if should_ignore(path, rules) {
                            must_skip = true;
                        }
                    }
                }

                if must_skip {
                    it.skip_current_dir();
                    continue;
                }
            }
            continue;
        }

        // Only process files
        if entry.file_type().is_file() {
            if let Some(rules) = ignore {
                if should_ignore(path, rules) {
                    continue;
                }
            }

            let metadata = match entry.metadata() {
                Ok(meta) => meta,
                Err(_) => continue,
            };

            let size_bytes = metadata.len();
            let extension = path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_lowercase());

            let is_binary = is_binary_file(path);

            // Compute relative path from workspace root
            let relative_path = match path.strip_prefix(root) {
                Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
                Err(_) => path.to_string_lossy().replace('\\', "/"),
            };

            total_size_bytes += size_bytes;
            files.push(FileEntry {
                path: relative_path,
                size_bytes,
                extension,
                is_binary,
            });
        }
    }

    let scan_duration_ms = start_time.elapsed().as_millis() as u64;
    let total_files = files.len();

    ScanResult {
        files,
        total_files,
        total_size_bytes,
        scan_duration_ms,
    }
}

/// Scans the workspace and returns the results formatted as a JSON string.
/// Sorts the files alphabetically by path to ensure deterministic output.
pub fn scan_to_json(root: &Path, ignore: Option<&GwenIgnore>) -> String {
    let mut result = scan_workspace(root, ignore);
    result.files.sort_by(|a, b| a.path.cmp(&b.path));
    serde_json::to_string_pretty(&result).unwrap_or_else(|_| "{}".to_string())
}
