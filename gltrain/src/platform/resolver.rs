// @INFO: GwenLand Recursive Import Resolver
// This module implements recursive dependency discovery for file-context mode.
// It parses imports for Rust, TypeScript/JavaScript, and Python,
// and resolves them relative to the target directory and workspace root.

// @EDITABLE
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use regex::Regex;

use crate::platform::scanner::FileEntry;
use crate::storage::ignore_rules::{load_ignore_rules, should_ignore};

/// Configuration for resolving imports
pub struct ResolveConfig {
    pub max_depth: usize,      // @EDITABLE — default: 3
    pub follow_external: bool, // @EDITABLE — default: false (skip node_modules etc)
}

impl Default for ResolveConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            follow_external: false,
        }
    }
}

/// The result context of the resolved imports
pub struct ResolvedContext {
    pub root_file: String,
    pub resolved_files: Vec<FileEntry>,
    pub depth_reached: usize,
    pub skipped: Vec<String>, // files that couldn't be resolved
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

/// Creates a FileEntry for a given file path relative to the workspace root.
fn create_file_entry(root: &Path, path: &Path) -> Option<FileEntry> {
    let metadata = path.metadata().ok()?;
    let size_bytes = metadata.len();
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase());
    let is_binary = is_binary_file(path);
    let relative_path = match path.strip_prefix(root) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => path.to_string_lossy().replace('\\', "/"),
    };
    Some(FileEntry {
        path: relative_path,
        size_bytes,
        extension,
        is_binary,
    })
}

/// Finds the nearest Rust crate root directory containing Cargo.toml or lib.rs/main.rs.
fn find_crate_root(target: &Path) -> Option<PathBuf> {
    let mut current = target.parent()?;
    loop {
        if current.join("lib.rs").exists() || current.join("main.rs").exists() {
            return Some(current.to_path_buf());
        }
        if current.join("Cargo.toml").exists() {
            let src = current.join("src");
            if src.is_dir() {
                return Some(src);
            }
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

/// Computes the Rust module path of a file relative to its crate root.
fn get_rust_module_path(crate_root: &Path, file_path: &Path) -> Vec<String> {
    let rel_path = match file_path.strip_prefix(crate_root) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    
    let mut segments: Vec<String> = rel_path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
        
    if let Some(last) = segments.last_mut() {
        if let Some(pos) = last.rfind('.') {
            last.truncate(pos);
        }
    }
    
    if let Some(last) = segments.last() {
        if last == "mod" || last == "lib" || last == "main" {
            segments.pop();
        }
    }
    
    segments
}

/// Resolves a Rust module path relative to the crate root.
/// Implements longest prefix matching to handle symbol imports from modules.
fn resolve_rust_module(crate_root: &Path, module_path: &[String]) -> Option<PathBuf> {
    if module_path.is_empty() {
        let lib_rs = crate_root.join("lib.rs");
        if lib_rs.is_file() {
            return Some(lib_rs);
        }
        let main_rs = crate_root.join("main.rs");
        if main_rs.is_file() {
            return Some(main_rs);
        }
        return None;
    }

    let mut current = module_path.to_vec();
    while !current.is_empty() {
        let mut base_path = crate_root.to_path_buf();
        for seg in &current {
            base_path.push(seg);
        }
        
        let file_candidate = base_path.with_extension("rs");
        if file_candidate.is_file() {
            return Some(file_candidate);
        }
        
        let mod_candidate = base_path.join("mod.rs");
        if mod_candidate.is_file() {
            return Some(mod_candidate);
        }
        
        current.pop();
    }
    
    None
}

/// Resolves a JS/TS import path.
/// Handles relative paths, common aliases (@/ and ~/), and extension fallbacks.
fn resolve_js_ts_path(root: &Path, current_file: &Path, import_path: &str, follow_external: bool) -> Option<PathBuf> {
    let parent_dir = current_file.parent()?;
    
    let base_paths = if import_path.starts_with('.') {
        vec![parent_dir.join(import_path)]
    } else if import_path.starts_with("@/") || import_path.starts_with("~/") {
        let rel = &import_path[2..];
        vec![root.join("src").join(rel), root.join(rel)]
    } else {
        if follow_external {
            vec![root.join("node_modules").join(import_path)]
        } else {
            return None;
        }
    };
    
    for base in base_paths {
        // 1. Exact match with extension
        if base.extension().is_some() {
            if base.is_file() {
                return Some(base);
            }
            // If .js target is requested, check if it maps to .ts or .tsx source files
            if let Some(ext) = base.extension().and_then(|e| e.to_str()) {
                if ext == "js" {
                    let ts = base.with_extension("ts");
                    if ts.is_file() { return Some(ts); }
                    let tsx = base.with_extension("tsx");
                    if tsx.is_file() { return Some(tsx); }
                }
            }
        } else {
            // 2. Try implicit extension fallbacks
            let exts = ["ts", "tsx", "js", "jsx", "mjs"];
            for ext in &exts {
                let candidate = base.with_extension(ext);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
            // 3. Try directory index resolution
            for ext in &exts {
                let candidate = base.join(format!("index.{}", ext));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    
    None
}

/// Resolves a Python module path relative to the current file's folder and workspace root.
fn resolve_python_module(
    root: &Path,
    current_file: &Path,
    segments: &[String],
    dots_count: usize,
    _follow_external: bool,
) -> Option<PathBuf> {
    if segments.is_empty() && dots_count == 0 {
        return None;
    }
    
    let current_dir = current_file.parent()?;
    let mut base_dirs = Vec::new();
    
    if dots_count > 0 {
        // Relative imports: . is current folder, .. is parent folder, etc.
        let mut dir = current_dir.to_path_buf();
        for _ in 0..(dots_count - 1) {
            if let Some(parent) = dir.parent() {
                dir = parent.to_path_buf();
            } else {
                break;
            }
        }
        base_dirs.push(dir);
    } else {
        // Absolute imports: check target's folder, then workspace root
        base_dirs.push(current_dir.to_path_buf());
        base_dirs.push(root.to_path_buf());
    }
    
    for base_dir in base_dirs {
        let mut path = base_dir;
        for seg in segments {
            path.push(seg);
        }
        
        // Check for file
        let file_candidate = path.with_extension("py");
        if file_candidate.is_file() {
            return Some(file_candidate);
        }
        
        // Check for package directory
        let init_candidate = path.join("__init__.py");
        if init_candidate.is_file() {
            return Some(init_candidate);
        }
    }
    
    None
}

/// Extract import statements from the content of a file based on its extension.
fn extract_imports(content: &str, extension: Option<&str>) -> Vec<(String, Option<usize>)> {
    let mut imports = Vec::new();
    
    match extension {
        Some("rs") => {
            // Match 'mod name;'
            let mod_regex = Regex::new(r"\bmod\s+([a-zA-Z0-9_]+)\s*;").unwrap();
            // Match 'use crate/super/self::...'
            let use_regex = Regex::new(r"\buse\s+(crate|super|self)\s*::\s*([a-zA-Z0-9_:]+)").unwrap();
            
            for line in content.lines() {
                // Strip inline comments
                let clean_line = match line.split_once("//") {
                    Some((before, _)) => before,
                    None => line,
                };
                
                if let Some(caps) = mod_regex.captures(clean_line) {
                    imports.push((caps[1].to_string(), None)); // None indicates Rust mod import
                }
                
                if let Some(caps) = use_regex.captures(clean_line) {
                    let prefix = caps[1].to_string();
                    let path = caps[2].to_string();
                    // Store as a joint string "prefix::path"
                    imports.push((format!("{}::{}", prefix, path), None));
                }
            }
        }
        Some("ts") | Some("tsx") | Some("js") | Some("mjs") | Some("jsx") => {
            // Match 'import ... from "path"' or 'export ... from "path"'
            let import_export_regex = Regex::new(r#"\b(?:import|export)\s+(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]"#).unwrap();
            // Match 'import "path"'
            let import_simple_regex = Regex::new(r#"\bimport\s+['"]([^'"]+)['"]"#).unwrap();
            // Match 'require("path")'
            let require_regex = Regex::new(r#"\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)"#).unwrap();
            
            for line in content.lines() {
                let clean_line = match line.split_once("//") {
                    Some((before, _)) => before,
                    None => line,
                };
                
                if let Some(caps) = import_export_regex.captures(clean_line) {
                    imports.push((caps[1].to_string(), None));
                } else if let Some(caps) = import_simple_regex.captures(clean_line) {
                    imports.push((caps[1].to_string(), None));
                } else if let Some(caps) = require_regex.captures(clean_line) {
                    imports.push((caps[1].to_string(), None));
                }
            }
        }
        Some("py") => {
            // Match 'import x, y'
            let import_regex = Regex::new(r"^\s*import\s+([a-zA-Z0-9_.,\s]+)").unwrap();
            // Match 'from x import y'
            let from_import_regex = Regex::new(r"^\s*from\s+(\.+[a-zA-Z0-9_.]*|[a-zA-Z0-9_.]+)\s+import\s+([a-zA-Z0-9_.,\s*()]+)").unwrap();
            
            for line in content.lines() {
                let clean_line = match line.split_once('#') {
                    Some((before, _)) => before,
                    None => line,
                };
                
                if let Some(caps) = import_regex.captures(clean_line) {
                    let import_list = &caps[1];
                    for part in import_list.split(',') {
                        let clean_part = part.trim();
                        // Handle 'import A as B'
                        let module_name = match clean_part.split_once(" as ") {
                            Some((before, _)) => before.trim(),
                            None => clean_part,
                        };
                        if !module_name.is_empty() {
                            imports.push((module_name.to_string(), None)); // None indicates absolute import
                        }
                    }
                }
                
                if let Some(caps) = from_import_regex.captures(clean_line) {
                    let from_module = caps[1].trim();
                    let import_list = &caps[2];
                    
                    // Parse leading dots
                    let dots_count = from_module.chars().take_while(|c| *c == '.').count();
                    let rest_of_module = &from_module[dots_count..];
                    
                    // 1. Add the parent module itself if it is not empty (e.g. from X import ...)
                    if !rest_of_module.is_empty() {
                        imports.push((rest_of_module.to_string(), Some(dots_count)));
                    }
                    
                    // 2. Add each imported item relative to X
                    for part in import_list.split(',') {
                        let clean_part = part.trim().trim_matches(|c| c == '(' || c == ')').trim();
                        let imported_item = match clean_part.split_once(" as ") {
                            Some((before, _)) => before.trim(),
                            None => clean_part,
                        };
                        if !imported_item.is_empty() && imported_item != "*" {
                            if rest_of_module.is_empty() {
                                // from . import Y
                                imports.push((imported_item.to_string(), Some(dots_count)));
                            } else {
                                // from X import Y
                                imports.push((
                                    format!("{}.{}", rest_of_module, imported_item),
                                    Some(dots_count),
                                ));
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
    
    imports
}

/// Recursively resolves all local dependency files imported by the target file.
///
/// BFS algorithm is used to crawl file dependencies up to a maximum depth.
/// Non-existent files are cataloged as skipped.
pub fn resolve_imports(
    root: &Path,
    target: &Path,
    config: &ResolveConfig,
) -> ResolvedContext {
    let ignore_rules = load_ignore_rules(root);
    
    let root_relative = match target.strip_prefix(root) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => target.to_string_lossy().replace('\\', "/"),
    };
    
    let mut resolved_files = Vec::new();
    let mut skipped = Vec::new();
    let mut depth_reached = 0;
    
    let root_entry = match create_file_entry(root, target) {
        Some(entry) => entry,
        None => {
            return ResolvedContext {
                root_file: root_relative.clone(),
                resolved_files: Vec::new(),
                depth_reached: 0,
                skipped: vec![target.to_string_lossy().into_owned()],
            };
        }
    };
    
    // Add target file as the first resolved entry
    resolved_files.push(root_entry.clone());
    
    // If target file is binary, don't parse it
    if root_entry.is_binary {
        return ResolvedContext {
            root_file: root_relative,
            resolved_files,
            depth_reached: 0,
            skipped,
        };
    }
    
    // Check extension
    let extension = target.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase());
        
    let is_known = match extension.as_deref() {
        Some("rs") | Some("ts") | Some("tsx") | Some("js") | Some("mjs") | Some("jsx") | Some("py") => true,
        _ => false,
    };
    
    if !is_known {
        // Unknown extension: return just the root file, no resolution
        return ResolvedContext {
            root_file: root_relative,
            resolved_files,
            depth_reached: 0,
            skipped,
        };
    }
    
    // Set up BFS queues and visited sets
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    
    // Insert target file to visited list using its canonical path to avoid duplicate checks
    let canonical_target = fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
    visited.insert(canonical_target);
    
    queue.push_back((target.to_path_buf(), 0));
    
    while let Some((current_path, current_depth)) = queue.pop_front() {
        depth_reached = std::cmp::max(depth_reached, current_depth);
        
        if current_depth >= config.max_depth {
            continue;
        }
        
        // Read file contents
        let content = match fs::read_to_string(&current_path) {
            Ok(c) => c,
            Err(_) => {
                // If it can't be read, skip it
                continue;
            }
        };
        
        let current_ext = current_path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());
            
        let extracted = extract_imports(&content, current_ext.as_deref());
        
        for (import_raw, py_dots) in extracted {
            let mut resolved_path = None;
            let mut is_external_module = false;
            
            match current_ext.as_deref() {
                Some("rs") => {
                    let crate_root = find_crate_root(&current_path).unwrap_or_else(|| {
                        current_path.parent().unwrap_or(&current_path).to_path_buf()
                    });
                    let current_module = get_rust_module_path(&crate_root, &current_path);
                    
                    if import_raw.contains("::") {
                        // e.g. "crate::a::b"
                        let parts: Vec<&str> = import_raw.split("::").collect();
                        if parts.len() >= 2 {
                            let prefix = parts[0];
                            let path_part = parts[1];
                            
                            let mut module_path = match prefix {
                                "crate" => Vec::new(),
                                "self" => current_module.clone(),
                                "super" => {
                                    let mut p = current_module.clone();
                                    p.pop();
                                    p
                                }
                                _ => current_module.clone(),
                            };
                            
                            let segments: Vec<&str> = path_part.split("::").collect();
                            for seg in segments {
                                if seg == "super" {
                                    module_path.pop();
                                } else if seg == "self" || seg.is_empty() {
                                    // ignore
                                } else {
                                    module_path.push(seg.to_string());
                                }
                            }
                            resolved_path = resolve_rust_module(&crate_root, &module_path);
                        }
                    } else {
                        // Mod import, e.g., "mod x;" -> submodule of current file's module path
                        let mut submod_path = current_module.clone();
                        submod_path.push(import_raw.clone());
                        resolved_path = resolve_rust_module(&crate_root, &submod_path);
                    }
                }
                Some("ts") | Some("tsx") | Some("js") | Some("mjs") | Some("jsx") => {
                    // Check if it's external (does not start with . or @ or ~)
                    let is_rel_or_alias = import_raw.starts_with('.') || import_raw.starts_with("@/") || import_raw.starts_with("~/");
                    if !is_rel_or_alias {
                        is_external_module = true;
                    }
                    resolved_path = resolve_js_ts_path(root, &current_path, &import_raw, config.follow_external);
                }
                Some("py") => {
                    let segments: Vec<String> = import_raw.split('.').map(|s| s.to_string()).collect();
                    let dots = py_dots.unwrap_or(0);
                    resolved_path = resolve_python_module(root, &current_path, &segments, dots, config.follow_external);
                    if py_dots.is_none() && resolved_path.is_none() {
                        is_external_module = true;
                    }
                }
                _ => {}
            }
            
            match resolved_path {
                Some(path) => {
                    // Normalize and canonicalize path
                    let canonical_path = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                    
                    if !visited.contains(&canonical_path) {
                        visited.insert(canonical_path.clone());
                        
                        // Check if it is external to the root and follow_external is false
                        let is_node_modules = path.components().any(|c| c.as_os_str() == "node_modules");
                        let is_ext = !path.starts_with(root) || is_node_modules;
                        
                        if is_ext && !config.follow_external {
                            continue;
                        }
                        
                        // Check if file is ignored
                        if should_ignore(&path, &ignore_rules) {
                            continue;
                        }
                        
                        // Create FileEntry
                        if let Some(entry) = create_file_entry(root, &path) {
                            if entry.is_binary {
                                continue;
                            }
                            resolved_files.push(entry);
                            queue.push_back((path, current_depth + 1));
                        } else {
                            skipped.push(import_raw);
                        }
                    }
                }
                None => {
                    // Record it as skipped only if it's not a known external module or if follow_external is true
                    if !is_external_module || config.follow_external {
                        skipped.push(import_raw);
                    }
                }
            }
        }
    }
    
    ResolvedContext {
        root_file: root_relative,
        resolved_files,
        depth_reached,
        skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;

    fn setup_test_env(test_name: &str) -> (PathBuf, PathBuf) {
        let root = env!("CARGO_MANIFEST_DIR");
        let test_dir = PathBuf::from(root)
            .join("target")
            .join("test_resolver")
            .join(test_name);
        
        if test_dir.exists() {
            let _ = fs::remove_dir_all(&test_dir);
        }
        fs::create_dir_all(&test_dir).unwrap();
        
        (test_dir.clone(), test_dir)
    }

    #[test]
    fn test_rust_resolution() {
        let (root, test_dir) = setup_test_env("rust");
        let src_dir = test_dir.join("src");
        fs::create_dir_all(&src_dir).unwrap();

        let main_rs = src_dir.join("main.rs");
        let a_rs = src_dir.join("a.rs");
        let b_mod_rs = src_dir.join("b").join("mod.rs");
        let c_rs = src_dir.join("b").join("c.rs");

        fs::create_dir_all(src_dir.join("b")).unwrap();

        fs::write(&main_rs, "mod a;\nmod b;\n").unwrap();
        fs::write(&a_rs, "use crate::b::c;\n").unwrap();
        fs::write(&b_mod_rs, "pub mod c;\n").unwrap();
        fs::write(&c_rs, "use super::super::a;\n").unwrap();

        let config = ResolveConfig {
            max_depth: 3,
            follow_external: false,
        };

        let result = resolve_imports(&root, &main_rs, &config);
        let paths: Vec<String> = result.resolved_files.iter().map(|f| f.path.clone()).collect();

        assert!(paths.contains(&"src/main.rs".to_string()));
        assert!(paths.contains(&"src/a.rs".to_string()));
        assert!(paths.contains(&"src/b/mod.rs".to_string()));
        assert!(paths.contains(&"src/b/c.rs".to_string()));
        assert_eq!(result.skipped.len(), 0);
        assert!(result.depth_reached > 0);
    }

    #[test]
    fn test_js_ts_resolution() {
        let (root, test_dir) = setup_test_env("js_ts");
        let index_ts = test_dir.join("index.ts");
        let a_ts = test_dir.join("a.ts");
        let b_tsx = test_dir.join("b.tsx");
        let c_js = test_dir.join("c.js");
        let d_dir = test_dir.join("d");
        let d_index_ts = d_dir.join("index.ts");

        fs::create_dir_all(&d_dir).unwrap();

        fs::write(&index_ts, "import { a } from './a';\nimport './b';\n").unwrap();
        fs::write(&a_ts, "import { c } from './c';\nimport d from './d';\n").unwrap();
        fs::write(&b_tsx, "export const b = 2;\n").unwrap();
        fs::write(&c_js, "const d = require('./d');\n").unwrap();
        fs::write(&d_index_ts, "export default 42;\n").unwrap();

        let config = ResolveConfig {
            max_depth: 4,
            follow_external: false,
        };

        let result = resolve_imports(&root, &index_ts, &config);
        let paths: Vec<String> = result.resolved_files.iter().map(|f| f.path.clone()).collect();

        assert!(paths.contains(&"index.ts".to_string()));
        assert!(paths.contains(&"a.ts".to_string()));
        assert!(paths.contains(&"b.tsx".to_string()));
        assert!(paths.contains(&"c.js".to_string()));
        assert!(paths.contains(&"d/index.ts".to_string()));
        assert_eq!(result.skipped.len(), 0);
    }

    #[test]
    fn test_python_resolution() {
        let (root, test_dir) = setup_test_env("python");
        let main_py = test_dir.join("main.py");
        let foo_py = test_dir.join("foo.py");
        let bar_dir = test_dir.join("bar");
        let bar_init_py = bar_dir.join("__init__.py");
        let baz_py = bar_dir.join("baz.py");

        fs::create_dir_all(&bar_dir).unwrap();

        fs::write(&main_py, "import foo\nfrom .bar import baz\n").unwrap();
        fs::write(&foo_py, "pass\n").unwrap();
        fs::write(&bar_init_py, "pass\n").unwrap();
        fs::write(&baz_py, "pass\n").unwrap();

        let config = ResolveConfig {
            max_depth: 3,
            follow_external: false,
        };

        let result = resolve_imports(&root, &main_py, &config);
        let paths: Vec<String> = result.resolved_files.iter().map(|f| f.path.clone()).collect();

        assert!(paths.contains(&"main.py".to_string()));
        assert!(paths.contains(&"foo.py".to_string()));
        assert!(paths.contains(&"bar/__init__.py".to_string()));
        assert!(paths.contains(&"bar/baz.py".to_string()));
        assert_eq!(result.skipped.len(), 0);
    }

    #[test]
    fn test_max_depth_resolution() {
        let (root, test_dir) = setup_test_env("depth");
        let f1 = test_dir.join("f1.ts");
        let f2 = test_dir.join("f2.ts");
        let f3 = test_dir.join("f3.ts");
        let f4 = test_dir.join("f4.ts");

        fs::write(&f1, "import './f2';\n").unwrap();
        fs::write(&f2, "import './f3';\n").unwrap();
        fs::write(&f3, "import './f4';\n").unwrap();
        fs::write(&f4, "// end\n").unwrap();

        let config = ResolveConfig {
            max_depth: 2,
            follow_external: false,
        };

        let result = resolve_imports(&root, &f1, &config);
        let paths: Vec<String> = result.resolved_files.iter().map(|f| f.path.clone()).collect();

        assert!(paths.contains(&"f1.ts".to_string()));
        assert!(paths.contains(&"f2.ts".to_string()));
        assert!(paths.contains(&"f3.ts".to_string()));
        assert!(!paths.contains(&"f4.ts".to_string()));
        assert_eq!(result.depth_reached, 2);
    }

    #[test]
    fn test_ignore_rules_and_binary() {
        let (root, test_dir) = setup_test_env("ignore_rules");
        let main_rs = test_dir.join("main.rs");
        let ignored_rs = test_dir.join("ignored.rs");

        fs::write(test_dir.join(".gwenignore"), "ignored.rs\n").unwrap();

        fs::write(&main_rs, "mod ignored;\nmod data;\n").unwrap();
        fs::write(&ignored_rs, "pub mod something;\n").unwrap();

        let data_rs = test_dir.join("data.rs");
        let mut f = File::create(&data_rs).unwrap();
        f.write_all(&[0, 1, 2, 3]).unwrap();

        let config = ResolveConfig {
            max_depth: 3,
            follow_external: false,
        };

        let result = resolve_imports(&root, &main_rs, &config);
        let paths: Vec<String> = result.resolved_files.iter().map(|f| f.path.clone()).collect();

        assert!(paths.contains(&"main.rs".to_string()));
        assert!(!paths.contains(&"ignored.rs".to_string()));
        assert!(!paths.contains(&"data.rs".to_string()));
    }
}
