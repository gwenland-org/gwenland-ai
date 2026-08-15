// @INFO: GwenLand Workspace Context Tree Generator
// Builds a nested, filtered workspace tree structure that can be serialized
// to JSON and sent to AI models as workspace context. Respects .gwenignore
// rules, skips binary files, and sorts directories-first alphabetically.

// @EDITABLE
use std::collections::BTreeMap;
use std::path::Path;
use chrono::Utc;
use serde::Serialize;
use walkdir::WalkDir;

use crate::storage::ignore_rules::{GwenIgnore, should_ignore};

// @INFO — represents a node in the workspace tree (file or directory)
#[derive(Debug, Clone, Serialize)]
pub struct TreeNode {
    pub name: String,
    pub path: String,            // relative from workspace root
    pub is_dir: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extension: Option<String>,
    pub is_binary: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<TreeNode>, // empty if file
}

// @INFO — final payload sent to AI as workspace context
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceContext {
    pub root: String,
    pub tree: TreeNode,
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub ignored_count: usize,
    pub generated_at: String, // ISO 8601 timestamp
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

/// Checks if a directory name should be unconditionally skipped.
fn is_skip_dir(name: &str) -> bool {
    name.starts_with('.')
        || name == "node_modules"
        || name == "target"
        || name == "dist"
}

/// Sorts TreeNode children: directories first, then files, both alphabetically.
fn sort_children(node: &mut TreeNode) {
    node.children.sort_by(|a, b| {
        match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        }
    });
    for child in &mut node.children {
        sort_children(child);
    }
}

/// Builds a nested workspace context tree from the given root path.
///
/// Walks the directory recursively, applies `.gwenignore` rules,
/// skips hidden/build directories and binary files, and produces a
/// nested `WorkspaceContext` ready for JSON serialization.
///
/// # Arguments
/// * `root` — the workspace root directory to scan
/// * `ignore` — loaded GwenIgnore rules for filtering
pub fn build_context_tree(root: &Path, ignore: &GwenIgnore) -> WorkspaceContext {
    let root_name = root.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace")
        .to_string();

    let mut total_files: usize = 0;
    let mut total_size_bytes: u64 = 0;
    let mut ignored_count: usize = 0;

    // @INFO: We collect all valid relative paths into a BTreeMap keyed by
    // their parent directory path. This lets us reconstruct the nested tree
    // without requiring the walkdir entries to arrive in any particular order.
    // Key: parent relative path, Value: Vec of TreeNode children
    let mut dir_children: BTreeMap<String, Vec<TreeNode>> = BTreeMap::new();
    // Track which directories we've seen so we can create nodes for them
    let mut known_dirs: BTreeMap<String, String> = BTreeMap::new(); // rel_path -> name

    // Seed the root
    dir_children.insert(String::new(), Vec::new());
    known_dirs.insert(String::new(), root_name.clone());

    let mut it = WalkDir::new(root).into_iter();
    loop {
        let entry = match it.next() {
            None => break,
            Some(Err(_)) => continue,
            Some(Ok(entry)) => entry,
        };

        let entry_path = entry.path();

        // Skip the root itself
        if entry_path == root {
            continue;
        }

        let relative = match entry_path.strip_prefix(root) {
            Ok(r) => r.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };

        let name = match entry_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // --- Directory handling ---
        if entry.file_type().is_dir() {
            if is_skip_dir(&name) {
                ignored_count += 1;
                it.skip_current_dir();
                continue;
            }

            if should_ignore(entry_path, ignore) {
                ignored_count += 1;
                it.skip_current_dir();
                continue;
            }

            // Register this directory
            known_dirs.insert(relative.clone(), name.clone());
            dir_children.entry(relative.clone()).or_insert_with(Vec::new);

            // Add a directory node to its parent's children list
            let parent_rel = parent_relative(&relative);
            dir_children.entry(parent_rel).or_insert_with(Vec::new).push(TreeNode {
                name,
                path: relative,
                is_dir: true,
                size_bytes: None,
                extension: None,
                is_binary: false,
                children: Vec::new(), // placeholder — assembled later
            });

            continue;
        }

        // --- File handling ---
        if entry.file_type().is_file() {
            // Check gwenignore
            if should_ignore(entry_path, ignore) {
                ignored_count += 1;
                continue;
            }

            // Check binary
            if is_binary_file(entry_path) {
                ignored_count += 1;
                continue;
            }

            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            let size = metadata.len();
            let extension = entry_path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_lowercase());

            total_files += 1;
            total_size_bytes += size;

            let parent_rel = parent_relative(&relative);
            dir_children.entry(parent_rel).or_insert_with(Vec::new).push(TreeNode {
                name,
                path: relative,
                is_dir: false,
                size_bytes: Some(size),
                extension,
                is_binary: false,
                children: Vec::new(),
            });
        }
    }

    // @INFO: Assemble the nested tree bottom-up.
    // Sort directory keys by depth (deepest first) so children are ready
    // before their parents consume them.
    let mut dir_keys: Vec<String> = dir_children.keys().cloned().collect();
    dir_keys.sort_by(|a, b| {
        let depth_a = if a.is_empty() { 0 } else { a.matches('/').count() + 1 };
        let depth_b = if b.is_empty() { 0 } else { b.matches('/').count() + 1 };
        depth_b.cmp(&depth_a) // deepest first
    });

    for dir_key in &dir_keys {
        if dir_key.is_empty() {
            continue; // skip root, handled last
        }

        let children = dir_children.remove(dir_key).unwrap_or_default();
        let parent_rel = parent_relative(dir_key);

        // Find the directory node in the parent's children and attach
        if let Some(parent_children) = dir_children.get_mut(&parent_rel) {
            for node in parent_children.iter_mut() {
                if node.is_dir && node.path == *dir_key {
                    node.children = children;
                    break;
                }
            }
        }
    }

    // Build the root TreeNode
    let root_children = dir_children.remove("").unwrap_or_default();
    let mut tree = TreeNode {
        name: root_name.clone(),
        path: String::new(),
        is_dir: true,
        size_bytes: None,
        extension: None,
        is_binary: false,
        children: root_children,
    };

    sort_children(&mut tree);

    let generated_at = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    WorkspaceContext {
        root: root_name,
        tree,
        total_files,
        total_size_bytes,
        ignored_count,
        generated_at,
    }
}

/// Serializes the WorkspaceContext to pretty-printed JSON.
pub fn tree_to_json(ctx: &WorkspaceContext) -> String {
    serde_json::to_string_pretty(ctx).unwrap_or_else(|_| "{}".to_string())
}

/// Returns the parent directory's relative path for a given relative path.
/// e.g. "packages/core/src" → "packages/core", "packages" → ""
fn parent_relative(rel: &str) -> String {
    match rel.rfind('/') {
        Some(pos) => rel[..pos].to_string(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn setup_test_dir(name: &str) -> PathBuf {
        let root = env!("CARGO_MANIFEST_DIR");
        let test_dir = PathBuf::from(root)
            .join("target")
            .join("test_context_tree")
            .join(name);

        if test_dir.exists() {
            let _ = fs::remove_dir_all(&test_dir);
        }
        fs::create_dir_all(&test_dir).unwrap();
        test_dir
    }

    #[test]
    fn test_basic_tree_structure() {
        let test_dir = setup_test_dir("basic");

        // Create structure:
        // test_dir/
        //   src/
        //     main.rs
        //     lib.rs
        //   README.md
        //   .gwenignore  (auto-created by load_ignore_rules)
        fs::create_dir_all(test_dir.join("src")).unwrap();
        fs::write(test_dir.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(test_dir.join("src/lib.rs"), "pub mod foo;").unwrap();
        fs::write(test_dir.join("README.md"), "# Hello").unwrap();

        // @INFO: load_ignore_rules auto-creates .gwenignore in the test dir
        let ignore = crate::storage::ignore_rules::load_ignore_rules(&test_dir);
        let ctx = build_context_tree(&test_dir, &ignore);

        // 3 user files + 1 auto-created .gwenignore
        assert_eq!(ctx.total_files, 4);
        assert!(ctx.total_size_bytes > 0);
        assert!(ctx.tree.is_dir);
        assert!(!ctx.tree.children.is_empty());

        // Directories should come before files in sorting
        let names: Vec<&str> = ctx.tree.children.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names[0], "src"); // dir first
        assert!(names.contains(&"README.md"));
    }

    #[test]
    fn test_ignores_hidden_dirs_and_binaries() {
        let test_dir = setup_test_dir("ignores");

        // Create structure with hidden dir and binary file
        fs::create_dir_all(test_dir.join(".git")).unwrap();
        fs::write(test_dir.join(".git/config"), "git stuff").unwrap();
        fs::create_dir_all(test_dir.join("node_modules")).unwrap();
        fs::write(test_dir.join("node_modules/pkg.js"), "module").unwrap();
        fs::write(test_dir.join("hello.rs"), "fn main() {}").unwrap();

        // Write a binary file
        let mut f = std::fs::File::create(test_dir.join("image.bin")).unwrap();
        use std::io::Write;
        f.write_all(&[0u8, 1, 2, 3, 0]).unwrap();

        let ignore = crate::storage::ignore_rules::load_ignore_rules(&test_dir);
        let ctx = build_context_tree(&test_dir, &ignore);

        // hello.rs + auto-created .gwenignore in the tree (not binary, not hidden/node_modules)
        assert_eq!(ctx.total_files, 2);
        assert!(ctx.ignored_count >= 2); // .git + node_modules + binary

        let file_names: Vec<&str> = ctx.tree.children.iter()
            .filter(|n| !n.is_dir)
            .map(|n| n.name.as_str())
            .collect();
        assert!(file_names.contains(&"hello.rs"));
        assert!(!file_names.contains(&"image.bin"));
    }

    #[test]
    fn test_nested_tree() {
        let test_dir = setup_test_dir("nested");

        fs::create_dir_all(test_dir.join("a/b/c")).unwrap();
        fs::write(test_dir.join("a/b/c/deep.txt"), "deep content").unwrap();
        fs::write(test_dir.join("a/top.txt"), "top content").unwrap();

        let ignore = crate::storage::ignore_rules::load_ignore_rules(&test_dir);
        let ctx = build_context_tree(&test_dir, &ignore);

        // 2 user files + 1 auto-created .gwenignore
        assert_eq!(ctx.total_files, 3);

        // Navigate to a/b/c/deep.txt
        let a = ctx.tree.children.iter().find(|n| n.name == "a").unwrap();
        assert!(a.is_dir);
        let b = a.children.iter().find(|n| n.name == "b").unwrap();
        assert!(b.is_dir);
        let c = b.children.iter().find(|n| n.name == "c").unwrap();
        assert!(c.is_dir);
        let deep = c.children.iter().find(|n| n.name == "deep.txt").unwrap();
        assert!(!deep.is_dir);
        assert_eq!(deep.path, "a/b/c/deep.txt");
    }

    #[test]
    fn test_sorting_dirs_first() {
        let test_dir = setup_test_dir("sorting");

        fs::create_dir_all(test_dir.join("zebra")).unwrap();
        fs::write(test_dir.join("zebra/z.txt"), "z").unwrap();
        fs::create_dir_all(test_dir.join("alpha")).unwrap();
        fs::write(test_dir.join("alpha/a.txt"), "a").unwrap();
        fs::write(test_dir.join("middle.txt"), "m").unwrap();
        fs::write(test_dir.join("aaa.txt"), "aaa").unwrap();

        let ignore = crate::storage::ignore_rules::load_ignore_rules(&test_dir);
        let ctx = build_context_tree(&test_dir, &ignore);

        let names: Vec<&str> = ctx.tree.children.iter().map(|n| n.name.as_str()).collect();
        // Dirs first alphabetically, then files alphabetically
        // .gwenignore is auto-created by load_ignore_rules
        assert_eq!(names, vec!["alpha", "zebra", ".gwenignore", "aaa.txt", "middle.txt"]);
    }

    #[test]
    fn test_tree_to_json() {
        let test_dir = setup_test_dir("json");

        fs::write(test_dir.join("test.rs"), "fn main() {}").unwrap();

        let ignore = crate::storage::ignore_rules::load_ignore_rules(&test_dir);
        let ctx = build_context_tree(&test_dir, &ignore);
        let json = tree_to_json(&ctx);

        assert!(json.contains("\"root\""));
        assert!(json.contains("\"total_files\""));
        assert!(json.contains("\"generated_at\""));
        assert!(json.contains("\"tree\""));
        assert!(json.contains("test.rs"));
    }

    #[test]
    fn test_generated_at_is_iso8601() {
        let test_dir = setup_test_dir("timestamp");
        fs::write(test_dir.join("f.txt"), "x").unwrap();

        let ignore = crate::storage::ignore_rules::load_ignore_rules(&test_dir);
        let ctx = build_context_tree(&test_dir, &ignore);

        // Verify it parses as a valid ISO 8601 timestamp
        assert!(chrono::DateTime::parse_from_rfc3339(&ctx.generated_at).is_ok());
    }
}
