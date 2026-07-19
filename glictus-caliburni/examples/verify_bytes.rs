//! Byte-exactness check for a converted GLLM package: every tensor must be
//! byte-identical to its GGUF source.
//!
//! ```text
//! cargo run -p glictus-caliburni --features converter --release \
//!     --example verify_bytes -- <model.gguf> <package_dir>
//! ```

use std::path::Path;

use glcore::format::gguf::GgufFile;
use glictus_caliburni::package::GllmPackage;
use glictus_caliburni::types::layer::LayerFile;

/// Mirror of the converter's name mapping: `(is_shared, gllm_name)`.
fn gllm_name_for(gguf_name: &str) -> (bool, String) {
    match gguf_name {
        "token_embd.weight" => (true, "token_embeddings".into()),
        "output_norm.weight" => (true, "output_norm.weight".into()),
        "output.weight" => (true, "output_head.weight".into()),
        _ => match gguf_name.strip_prefix("blk.") {
            Some(rest) => {
                let (_, tensor) = rest.split_once('.').expect("blk.N.<name>");
                (false, tensor.to_string())
            }
            None => (true, gguf_name.to_string()),
        },
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf_path = args.next().expect("arg1: <model.gguf>");
    let pkg_path = args.next().expect("arg2: <package_dir>");

    let gguf = GgufFile::open(&gguf_path).expect("gguf opens");
    let pkg = GllmPackage::open(Path::new(&pkg_path)).expect("package opens");

    let shared_bytes = std::fs::read(&pkg.layout.shared_path).expect("read shared");
    let shared_index = LayerFile::read(&pkg.layout.shared_path).expect("parse shared");
    let layer_files: Vec<(Vec<u8>, LayerFile)> = pkg
        .layout
        .layer_paths
        .iter()
        .map(|lp| {
            (
                std::fs::read(&lp.path).expect("read layer"),
                LayerFile::read(&lp.path).expect("parse layer"),
            )
        })
        .collect();

    let mut checked = 0usize;
    let mut total_bytes = 0u64;
    for info in &gguf.tensors {
        let src = gguf.tensor_data(info).expect("gguf tensor data");
        let (is_shared, name) = gllm_name_for(&info.name);

        let (bytes, index) = if is_shared {
            (&shared_bytes, &shared_index)
        } else {
            let idx: usize = info.name.strip_prefix("blk.").expect("blk. prefix")
                .split_once('.').expect("blk.N.<name>")
                .0.parse().expect("numeric layer index");
            let (b, l) = &layer_files[idx];
            (b, l)
        };

        let (offset, size) = index
            .absolute_range(&name)
            .unwrap_or_else(|| panic!("{} -> {name}: not in unit tensor index", info.name));
        let dst = &bytes[offset as usize..(offset + size) as usize];

        assert_eq!(src.len(), dst.len(), "{}: size differs", info.name);
        assert_eq!(src, dst, "{}: BYTES DIFFER", info.name);
        checked += 1;
        total_bytes += size;
    }

    println!(
        "OK: {checked} tensors byte-identical ({:.2} MB verified)",
        total_bytes as f64 / (1024.0 * 1024.0)
    );
}
