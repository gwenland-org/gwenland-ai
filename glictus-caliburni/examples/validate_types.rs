//! Check a package's layer types against the ARTX8 plugin registry.
//!
//! ```text
//! cargo run -p glictus-caliburni --release --example validate_types -- <package_dir>
//! ```

use glictus_caliburni::{GllmPackage, PluginRegistry};

fn main() {
    let root = std::env::args().nth(1).expect("arg1: <package_dir>");
    let pkg = GllmPackage::open(std::path::Path::new(&root)).expect("package opens");
    let registry = PluginRegistry::with_builtins();

    println!("model    : {}", pkg.manifest().model_id);
    println!("layers   : {}", pkg.layer_count());
    println!("registry : {}", registry.registered_uris().join(", "));

    let missing = registry.missing_for_manifest(pkg.manifest());
    if missing.is_empty() {
        println!("types    : all declared types have a plugin");
    } else {
        println!("types    : UNSUPPORTED -> {}", missing.join(", "));
    }

    let findings = pkg.validate_layer_types(&registry);
    if findings.is_empty() {
        println!("\nOK: every layer matches its declared type's layout");
    } else {
        println!("\n{} layer(s) do not match their declared type:", findings.len());
        for (file, problem) in findings.iter().take(3) {
            println!("  {file}: {problem}");
        }
        if findings.len() > 3 {
            println!("  ... and {} more", findings.len() - 3);
        }
    }
}
