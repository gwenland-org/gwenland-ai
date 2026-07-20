//! Drive the ARTX05 runtime over a real converted package.
//!
//! Executes one forward pass with [`NullBackend`], which computes nothing —
//! this exercises the *orchestration* (map → resolve → execute → unmap, KV
//! sizing, adaptive prefetch), not any numerics. A pass here proves the
//! runtime walks a real model end to end; it does not produce logits.
//!
//! ```text
//! cargo run -p glictus-caliburni --release --example run_package -- <package_dir>
//! ```

use std::sync::Arc;

use glictus_caliburni::runtime::{
    ActivationBuffer, GllmRuntime, NullBackend, RuntimeConfig, RuntimeLogLevel, RuntimeLogger,
};

fn main() {
    let root = std::env::args()
        .nth(1)
        .expect("arg1: <package_dir> (a converted .gllm package)");
    let path = std::path::Path::new(&root);

    let logger = RuntimeLogger::shared_silent(RuntimeLogLevel::Info);
    let config = RuntimeConfig::default();

    let open_start = std::time::Instant::now();
    let mut rt = GllmRuntime::open_with_logger(
        path,
        config,
        Arc::new(NullBackend::new()),
        Arc::clone(&logger),
    )
    .expect("package opens");
    let open_ms = open_start.elapsed().as_secs_f64() * 1000.0;

    let meta = &rt.manifest().metadata;
    println!("model      : {}", rt.manifest().model_id);
    println!("layers     : {}", rt.layer_count());
    println!(
        "shape      : dim {}, {} heads / {} kv heads, ctx {}",
        meta.embedding_length, meta.num_heads, meta.head_count_kv, meta.context_length
    );
    println!(
        "kv cache   : {:.1} MiB across {} slots (elem {} B, seq {})",
        rt.kv_cache().total_allocated_bytes() as f64 / (1024.0 * 1024.0),
        rt.kv_cache().len(),
        rt.config().kv_element_size(),
        rt.kv_cache().config().max_seq_len,
    );
    println!("open       : {open_ms:.1} ms");

    let mut activation = ActivationBuffer::zeros(vec![meta.embedding_length as usize]);
    let stats = rt.run(&mut activation).expect("forward pass");

    println!("\n--- pass ---");
    println!("executed   : {} layers", stats.layers_executed);
    println!("unmapped   : {}", stats.layers_unmapped);
    println!("fallbacks  : {}", stats.recoverable_errors);
    println!("wall time  : {} ms", stats.total_exec_time_ms);
    println!("state      : {:?}", rt.state());
    println!("errors     : {}", rt.error_count());

    assert_eq!(
        stats.layers_executed,
        rt.layer_count(),
        "every layer must have executed"
    );
    println!("\nOK: orchestration walked all {} layers", rt.layer_count());
}
