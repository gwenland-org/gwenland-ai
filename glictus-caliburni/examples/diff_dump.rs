//! Differential dump: compare `glproc::runner::Runner`'s (known-good)
//! per-layer hidden states against `GlprocBackend`'s (still producing
//! garbage output after 3 confirmed-and-fixed bugs — RMSNorm epsilon, GQ4A
//! dequant ruled out, missing QKV bias) on the identical single token, to
//! localize the remaining divergence.
//!
//! Both paths load the same original GGUF's weights — Path A directly,
//! Path B via the `.gllm` package converted from it (use the `--quant F32`
//! diagnostic package: zero quantization anywhere, so any divergence here
//! cannot be a dequant artifact).
//!
//! ```text
//! cargo run -p glictus-caliburni --release --features glproc-backend \
//!     --example diff_dump -- <original.gguf> <gllm_package_dir> [--token N]
//! ```

use std::path::Path;
use std::sync::{Arc, Mutex};

use glcore::format::gguf::GgufFile;
use glictus_caliburni::manifest::{DType, LayerManifest, ModelMetadata};
use glictus_caliburni::package::GllmPackage;
use glictus_caliburni::runtime::{
    ActivationBuffer, ExecutionBackend, GllmRuntime, GlprocBackend, KvCacheSlot, LayerMapping,
    RuntimeConfig,
};
use glictus_caliburni::GllmResult;

/// Wraps a real `GlprocBackend`, forwarding every call unchanged but also
/// recording the output activation after each layer — the only way to get
/// per-layer hidden states out of `GllmRuntime` without adding new
/// production API surface. `ExecutionBackend` is already the swappable seam
/// `GllmRuntime` was designed around, so this needs zero changes to
/// `glproc_backend.rs`.
struct CapturingBackend {
    inner: GlprocBackend,
    captured: Mutex<Vec<Vec<f32>>>,
}

impl ExecutionBackend for CapturingBackend {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }
    fn kv_element_size(&self) -> usize {
        self.inner.kv_element_size()
    }
    fn execute_layer(
        &self,
        layer: &LayerMapping,
        manifest: &LayerManifest,
        kv_cache: &mut KvCacheSlot,
        input: &ActivationBuffer,
        output: &mut ActivationBuffer,
    ) -> GllmResult<()> {
        self.inner.execute_layer(layer, manifest, kv_cache, input, output)?;
        self.captured.lock().unwrap().push(output.data.clone());
        Ok(())
    }
}

/// The shared-component tensors `GllmEngine::load_shared` reads internally
/// (private to `gllm_engine.rs`) — duplicated here for this diagnostic tool
/// rather than exposing new production API just to reuse ~20 lines.
struct SharedTensors {
    embed: Vec<f32>,
    output_norm: Vec<f32>,
    lm_head: Vec<f32>,
    dim: usize,
    vocab_size: usize,
    rms_eps: f32,
}

fn dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::Bf16 => "BF16",
        DType::Q4_0 => "Q4_0",
        DType::Q4_1 => "Q4_1",
        DType::Q4K => "Q4_K",
        DType::Q4Km => "Q4_K_M",
        DType::Q4Ks => "Q4_K_S",
        DType::Q5_0 => "Q5_0",
        DType::Q6K => "Q6_K",
        DType::Q8_0 => "Q8_0",
        DType::Q8K => "Q8_K",
        DType::I32 => "I32",
        DType::Fp8E4m3 => "F8",
        DType::Fp8E5m2 => "FP8_E5M2",
        DType::GQ4A => "GQ4A",
        DType::GQ2A => "GQ2A",
        DType::Unknown => "UNKNOWN",
    }
}

fn load_shared_tensors(pkg: &GllmPackage) -> SharedTensors {
    let metadata: &ModelMetadata = &pkg.manifest().metadata;
    let dim = metadata.embedding_length as usize;
    let vocab_size = metadata.vocab_size as usize;
    let rms_eps = metadata.effective_rms_eps() as f32;

    let shared_path = &pkg.layout.shared_path;
    let mapping = LayerMapping::open(shared_path, None, false).expect("open shared component");

    let decode = |name: &str| -> Vec<f32> {
        let entry = pkg
            .manifest()
            .shared
            .tensor(name)
            .unwrap_or_else(|| panic!("shared component missing tensor {name:?}"));
        let bytes = mapping
            .tensor_bytes(name)
            .unwrap_or_else(|| panic!("shared component has no data for {name:?}"));
        match entry.dtype {
            DType::GQ4A => glproc::kernels::gquant::dequant_gq4a_stream(bytes),
            DType::GQ2A => glproc::kernels::gquant::dequant_gq2a_stream(bytes),
            _ => glcore::format::decode_tensor(name, &entry.shape, dtype_str(entry.dtype), bytes)
                .unwrap_or_else(|e| panic!("decode {name:?}: {e}"))
                .data,
        }
    };

    let embed = decode("token_embeddings");
    let output_norm = decode("output_norm.weight");
    let lm_head = if pkg.manifest().shared.tensor("output_head.weight").is_some() {
        decode("output_head.weight")
    } else {
        embed.clone()
    };

    SharedTensors { embed, output_norm, lm_head, dim, vocab_size, rms_eps }
}

/// A layer's F32-dequantized tensors, fetched from the `.gllm` package —
/// enough to manually replay the attention block for one layer.
struct LayerTensors {
    attn_norm: Vec<f32>,
    attn_v: Vec<f32>,
    attn_v_bias: Option<Vec<f32>>,
    attn_output: Vec<f32>,
}

fn load_layer_tensors(pkg: &GllmPackage, gllm_dir: &str, index: u32) -> LayerTensors {
    let manifest = pkg.layer_manifest(index).unwrap_or_else(|| panic!("layer {index} manifest"));
    let file = manifest.file.clone();
    let path = Path::new(gllm_dir).join(&file);
    let mapping = LayerMapping::open(&path, Some(index), false).unwrap_or_else(|e| panic!("open layer {index}: {e}"));

    let decode = |name: &str| -> Vec<f32> {
        let entry = manifest.tensor(name).unwrap_or_else(|| panic!("layer {index} missing tensor {name:?}"));
        let bytes = mapping.tensor_bytes(name).unwrap_or_else(|| panic!("layer {index} has no data for {name:?}"));
        match entry.dtype {
            DType::GQ4A => glproc::kernels::gquant::dequant_gq4a_stream(bytes),
            DType::GQ2A => glproc::kernels::gquant::dequant_gq2a_stream(bytes),
            _ => glcore::format::decode_tensor(name, &entry.shape, dtype_str(entry.dtype), bytes)
                .unwrap_or_else(|e| panic!("decode {name:?}: {e}"))
                .data,
        }
    };
    let decode_opt = |name: &str| -> Option<Vec<f32>> {
        manifest.tensor(name).map(|_| decode(name))
    };

    LayerTensors {
        attn_norm: decode("attn_norm.weight"),
        attn_v: decode("attn_v.weight"),
        attn_v_bias: decode_opt("attn_v.bias"),
        attn_output: decode("attn_output.weight"),
    }
}

/// Manually replay ONLY the attention block of one layer, for Path B's own
/// real weights and real input — no RoPE, no `attention_one_into`, no KV
/// cache needed at all. This is not an approximation: with exactly one
/// cached position (`cached_len == 1`), softmax over a single score is
/// mathematically forced to be exactly `[1.0]` regardless of that score's
/// value, so `attn_out` for every head equals `V` for that head exactly —
/// Q, K, and RoPE cannot change the result. This isolates whether a
/// layer's massive-activation dimension is seeded by the attention block or
/// only appears after the FFN block runs.
fn replay_attention_only(t: &LayerTensors, input: &[f32], dim: usize, kv_dim: usize, n_heads: usize, n_kv_heads: usize, rms_eps: f32) -> Vec<f32> {
    let mut xn = vec![0.0f32; dim];
    glproc::kernels::rms_norm_into(input, &t.attn_norm, rms_eps, &mut xn);

    let mut v = vec![0.0f32; kv_dim];
    glproc::kernels::matvec(&t.attn_v, &xn, &mut v, kv_dim, dim);
    if let Some(bias) = &t.attn_v_bias {
        for (vi, bi) in v.iter_mut().zip(bias) {
            *vi += bi;
        }
    }

    let head_dim = kv_dim / n_kv_heads;
    let heads_per_kv = n_heads / n_kv_heads.max(1);
    let mut attn_out = vec![0.0f32; dim];
    for h in 0..n_heads {
        let kv_head = h / heads_per_kv.max(1);
        attn_out[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&v[kv_head * head_dim..(kv_head + 1) * head_dim]);
    }

    let mut proj = vec![0.0f32; dim];
    glproc::kernels::matvec(&t.attn_output, &attn_out, &mut proj, dim, dim);
    let mut out = input.to_vec();
    for (oi, pi) in out.iter_mut().zip(&proj) {
        *oi += pi;
    }
    out
}

/// A layer's F32-dequantized tensors, fetched directly from the ORIGINAL
/// GGUF — for replaying Path A's layer with F32 math instead of glproc's
/// native quantized (bridge/repack) kernels, to isolate whether a
/// divergence is a quantization-computation-path artifact rather than a
/// logic difference.
struct GgufLayerTensors {
    attn_norm: Vec<f32>,
    attn_q: Vec<f32>,
    attn_k: Vec<f32>,
    attn_v: Vec<f32>,
    attn_q_bias: Option<Vec<f32>>,
    attn_k_bias: Option<Vec<f32>>,
    attn_v_bias: Option<Vec<f32>>,
    attn_output: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

fn dequant_gguf_tensor(gguf: &GgufFile, name: &str) -> Vec<f32> {
    let info = gguf.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("find {name}"));
    let raw = gguf.tensor_data(info).unwrap_or_else(|e| panic!("read {name}: {e}"));
    // Q4_K/Q5_0/Q6_K all route through glproc, never glcore::GgufFile::dequantize
    // — glcore rejects Q4_K/Q5_0 outright, but *silently accepts and
    // mis-decodes* Q6_K (wrong nibble order; see
    // architecture/gl-stack-audit-2026-07/ARTX2-Quant.md). This diagnostic tool
    // is exactly what found that bug, and until this fix it was itself still
    // using the wrong Q6_K reference for every sanity check below.
    match info.dtype {
        glcore::format::gguf::GgufDType::Q4_K => {
            glproc::kernels::dequant::q4_k::scalar::run(raw).unwrap_or_else(|e| panic!("dequant Q4_K {name}: {e}"))
        }
        glcore::format::gguf::GgufDType::Q5_0 => {
            glproc::kernels::dequant::q5_0::scalar::run(raw).unwrap_or_else(|e| panic!("dequant Q5_0 {name}: {e}"))
        }
        glcore::format::gguf::GgufDType::Q6_K => {
            glproc::kernels::dequant::q6_k::scalar::run(raw).unwrap_or_else(|e| panic!("dequant Q6_K {name}: {e}"))
        }
        _ => gguf.dequantize(info).unwrap_or_else(|e| panic!("dequant {name}: {e}")),
    }
}

fn dequant_gguf_tensor_opt(gguf: &GgufFile, name: &str) -> Option<Vec<f32>> {
    gguf.tensors.iter().any(|t| t.name == name).then(|| dequant_gguf_tensor(gguf, name))
}

fn load_gguf_layer_tensors(gguf: &GgufFile, layer: u32) -> GgufLayerTensors {
    let n = |suffix: &str| format!("blk.{layer}.{suffix}");
    GgufLayerTensors {
        attn_norm: dequant_gguf_tensor(gguf, &n("attn_norm.weight")),
        attn_q: dequant_gguf_tensor(gguf, &n("attn_q.weight")),
        attn_k: dequant_gguf_tensor(gguf, &n("attn_k.weight")),
        attn_v: dequant_gguf_tensor(gguf, &n("attn_v.weight")),
        attn_q_bias: dequant_gguf_tensor_opt(gguf, &n("attn_q.bias")),
        attn_k_bias: dequant_gguf_tensor_opt(gguf, &n("attn_k.bias")),
        attn_v_bias: dequant_gguf_tensor_opt(gguf, &n("attn_v.bias")),
        attn_output: dequant_gguf_tensor(gguf, &n("attn_output.weight")),
        ffn_norm: dequant_gguf_tensor(gguf, &n("ffn_norm.weight")),
        ffn_gate: dequant_gguf_tensor(gguf, &n("ffn_gate.weight")),
        ffn_up: dequant_gguf_tensor(gguf, &n("ffn_up.weight")),
        ffn_down: dequant_gguf_tensor(gguf, &n("ffn_down.weight")),
    }
}

/// Full layer replay (attention + FFN) in PURE F32 math — no RoPE needed
/// for the same `cached_len == 1` reason as `replay_attention_only`, and no
/// native quantized/bridge kernels at all, even though the source weights
/// were originally Q4_K/Q5_0. If this reproduces the real (native-quantized)
/// Path A output, the divergence is not a quantization-computation-path
/// artifact. If it does NOT (i.e. this F32 replay stays small like Path B,
/// while the real native-quantized Path A explodes), that pinpoints the
/// divergence as specifically a native-quantized-kernel effect.
#[allow(clippy::too_many_arguments)]
fn replay_full_layer_f32(
    t: &GgufLayerTensors,
    input: &[f32],
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    hidden_dim: usize,
    rms_eps: f32,
) -> Vec<f32> {
    let mut xn = vec![0.0f32; dim];
    glproc::kernels::rms_norm_into(input, &t.attn_norm, rms_eps, &mut xn);

    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    glproc::kernels::matvec(&t.attn_q, &xn, &mut q, q_dim, dim);
    glproc::kernels::matvec(&t.attn_k, &xn, &mut k, kv_dim, dim);
    glproc::kernels::matvec(&t.attn_v, &xn, &mut v, kv_dim, dim);
    if let Some(b) = &t.attn_q_bias {
        for (qi, bi) in q.iter_mut().zip(b) { *qi += bi; }
    }
    if let Some(b) = &t.attn_k_bias {
        for (ki, bi) in k.iter_mut().zip(b) { *ki += bi; }
    }
    if let Some(b) = &t.attn_v_bias {
        for (vi, bi) in v.iter_mut().zip(b) { *vi += bi; }
    }
    // Q and K (and thus RoPE) cannot affect the result when cached_len == 1
    // — softmax over a single score is always exactly 1.0 — so only V feeds
    // attn_out, same shortcut as replay_attention_only.
    let _ = (&q, &k);

    let head_dim = kv_dim / n_kv_heads;
    let heads_per_kv = n_heads / n_kv_heads.max(1);
    let mut attn_out = vec![0.0f32; q_dim];
    for h in 0..n_heads {
        let kv_head = h / heads_per_kv.max(1);
        attn_out[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&v[kv_head * head_dim..(kv_head + 1) * head_dim]);
    }

    let mut proj = vec![0.0f32; dim];
    glproc::kernels::matvec(&t.attn_output, &attn_out, &mut proj, dim, dim);
    let mut x = input.to_vec();
    for (xi, pi) in x.iter_mut().zip(&proj) { *xi += pi; }

    glproc::kernels::rms_norm_into(&x, &t.ffn_norm, rms_eps, &mut xn);
    let mut gate = vec![0.0f32; hidden_dim];
    let mut up = vec![0.0f32; hidden_dim];
    glproc::kernels::matvec(&t.ffn_gate, &xn, &mut gate, hidden_dim, dim);
    glproc::kernels::matvec(&t.ffn_up, &xn, &mut up, hidden_dim, dim);
    glproc::kernels::silu_mul(&mut gate, &up);
    let mut down = vec![0.0f32; dim];
    glproc::kernels::matvec(&t.ffn_down, &gate, &mut down, dim, hidden_dim);
    for (xi, di) in x.iter_mut().zip(&down) { *xi += di; }
    x
}

fn l2_norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.into_iter().take(k).map(|i| (i, logits[i])).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: diff_dump <original.gguf> <gllm_package_dir> [--token N]";
    let gguf_path = args.next().unwrap_or_else(|| { eprintln!("{usage}"); std::process::exit(2); });
    let gllm_dir = args.next().unwrap_or_else(|| { eprintln!("{usage}"); std::process::exit(2); });
    let mut token: u32 = 1;
    while let Some(flag) = args.next() {
        if flag == "--token" {
            token = args.next().expect("--token needs a value").parse().expect("--token must be a u32");
        }
    }

    // GLPROC_ATTN_SEQ=1 removes threaded-reduction float-order noise from
    // Path A as a confound. GlprocBackend (Path B) has no threading at all
    // (see its own module docs: "Threading is deliberately absent"), so
    // only Path A needs this forced.
    // SAFETY: single-threaded at this point in main(), before any thread
    // pool is spawned by either path's engine construction below.
    unsafe {
        std::env::set_var("GLPROC_ATTN_SEQ", "1");
    }

    eprintln!("=== Path A: glproc::runner::Runner (known-good), loading {gguf_path} ===");
    let gguf = GgufFile::open(&gguf_path).expect("open source GGUF");
    let model = glproc::loader::load_gguf(&gguf).expect("load GlprocModel from GGUF");
    let mut runner = glproc::runner::Runner::new(&model);
    runner.set_capture_hidden(true);
    // forward_chunk_into (the batched prefill path), not forward_into: real
    // generation always processes position 0 through step_chunk, even for a
    // one-token prompt. forward_into's single-token step() is only ever
    // reached for pos >= 1 in actual use — using it here would compare
    // against a code path position 0 never really exercises.
    runner.forward_chunk_into(&[token], 0).expect("Path A forward pass");
    let hidden_a: Vec<Vec<f32>> = runner.hidden_dump().to_vec();
    let logits_a: Vec<f32> = runner.logits().to_vec();

    eprintln!("=== Path B: GlprocBackend (still broken), loading {gllm_dir} ===");
    let pkg = GllmPackage::open(Path::new(&gllm_dir)).expect("open .gllm package");
    let shared = load_shared_tensors(&pkg);
    let metadata = pkg.manifest().metadata.clone();

    println!(
        "config: A rms_eps={:e} rope_freq_base={} | B rms_eps={:e} rope_freq_base={}",
        model.config.rms_eps,
        model.config.rope_freq_base,
        metadata.effective_rms_eps(),
        metadata.effective_rope_freq_base(),
    );
    println!(
        "config: A n_heads={} n_kv_heads={} head_dim={} dim={} | B n_heads={} n_kv_heads={} head_dim={:?} dim={}",
        model.config.n_heads,
        model.config.n_kv_heads,
        model.config.head_dim,
        model.config.dim,
        metadata.num_heads,
        metadata.head_count_kv,
        metadata.head_dim(),
        metadata.embedding_length,
    );
    let inner_backend = GlprocBackend::new(&metadata).expect("build GlprocBackend");
    let capturing = Arc::new(CapturingBackend { inner: inner_backend, captured: Mutex::new(Vec::new()) });
    let backend_for_runtime: Arc<dyn ExecutionBackend> = capturing.clone();
    let mut runtime = GllmRuntime::open(Path::new(&gllm_dir), RuntimeConfig::default(), backend_for_runtime)
        .expect("open GllmRuntime");

    let embed_row = shared.embed[(token as usize) * shared.dim..(token as usize + 1) * shared.dim].to_vec();
    let mut activation = ActivationBuffer { data: embed_row, shape: vec![shared.dim] };
    runtime.run(&mut activation).expect("Path B forward pass");
    let hidden_b: Vec<Vec<f32>> = capturing.captured.lock().unwrap().clone();

    let mut normed = vec![0.0f32; shared.dim];
    glproc::kernels::rms_norm_into(&activation.data, &shared.output_norm, shared.rms_eps, &mut normed);
    let mut logits_b = vec![0.0f32; shared.vocab_size];
    glproc::kernels::matvec(&shared.lm_head, &normed, &mut logits_b, shared.vocab_size, shared.dim);

    // --- sanity check: is the F32 package's layer-0 attn_q.weight a
    // faithful dequant of the ORIGINAL Q4_K_M GGUF's same tensor? Path A
    // runs the GGUF's weights NATIVELY QUANTIZED (Q4_K), Path B runs the
    // .gllm package's F32-dequantized copy — an apples-to-oranges precision
    // mismatch unless this holds. ---
    {
        let info = gguf.tensors.iter().find(|t| t.name == "blk.0.attn_q.weight").expect("find blk.0.attn_q.weight in source GGUF");
        println!("blk.0.attn_q.weight source dtype = {:?}", info.dtype);
        let dequant_from_gguf = dequant_gguf_tensor(&gguf, "blk.0.attn_q.weight");

        let layer0_path = Path::new(&gllm_dir).join("GLLMTensorLayer-0000.gllm");
        let layer0_mapping = LayerMapping::open(&layer0_path, Some(0), false).expect("open layer 0");
        let layer0_manifest = pkg.layer_manifest(0).expect("layer 0 manifest");
        let entry = layer0_manifest.tensor("attn_q.weight").expect("attn_q.weight in layer 0 manifest");
        let bytes = layer0_mapping.tensor_bytes("attn_q.weight").expect("attn_q.weight bytes");
        let from_package: Vec<f32> = glcore::format::decode_tensor("attn_q.weight", &entry.shape, dtype_str(entry.dtype), bytes)
            .expect("decode attn_q.weight")
            .data;

        println!(
            "attn_q.weight[layer0] dequant sanity: GGUF-dequant norm={:.4} package-F32 norm={:.4} max_diff={:.8}",
            l2_norm(&dequant_from_gguf),
            l2_norm(&from_package),
            max_abs_diff(&dequant_from_gguf, &from_package)
        );
        println!();
    }

    // --- compare the raw embedding itself, before any layer runs ---
    let mut embed_a = vec![0.0f32; model.config.dim];
    model.embed_into(token, &mut embed_a).expect("Path A embed_into");
    let embed_b = &shared.embed[(token as usize) * shared.dim..(token as usize + 1) * shared.dim];
    println!(
        "embedding: A_norm={:.4} B_norm={:.4} max_diff={:.6}",
        l2_norm(&embed_a),
        l2_norm(embed_b),
        max_abs_diff(&embed_a, embed_b)
    );
    println!("  first8_A = {:?}", &embed_a[..8.min(embed_a.len())]);
    println!("  first8_B = {:?}", &embed_b[..8.min(embed_b.len())]);
    println!();

    // --- compare ---
    println!("token = {token}");
    println!("{:<7} {:>12} {:>12} {:>12}", "layer", "A_norm", "B_norm", "max_diff");
    let n_layers = hidden_a.len().min(hidden_b.len());
    if hidden_a.len() != hidden_b.len() {
        println!(
            "!! layer count differs: Path A = {} layers, Path B = {} layers",
            hidden_a.len(),
            hidden_b.len()
        );
    }
    let mut first_divergence: Option<usize> = None;
    for i in 0..n_layers {
        let a = &hidden_a[i];
        let b = &hidden_b[i];
        let a_norm = l2_norm(a);
        let b_norm = l2_norm(b);
        let max_diff = max_abs_diff(a, b);
        let marker = if max_diff > 0.01 && first_divergence.is_none() {
            first_divergence = Some(i);
            " <-- DIVERGE HERE"
        } else {
            ""
        };
        println!("{i:<7} {a_norm:>12.4} {b_norm:>12.4} {max_diff:>12.6}{marker}");
        if i < 2 || Some(i) == first_divergence {
            println!("        first8_A = {:?}", &a[..8.min(a.len())]);
            println!("        first8_B = {:?}", &b[..8.min(b.len())]);
        }
        // "Massive activations" (Sun et al. and follow-ups) concentrate in a
        // handful of specific dimensions, not spread evenly — find Path A's
        // top-magnitude dimension at this layer and print BOTH paths' value
        // there, to see whether B is missing the same channel entirely (a
        // real bug) or has it but smaller (a scale/precision difference).
        if let Some((dim_idx, _)) = a
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap())
        {
            println!(
                "        top-dim A[{dim_idx}]={:.4}  B[{dim_idx}]={:.4}",
                a[dim_idx], b[dim_idx]
            );
        }
    }

    // --- the SAME two checks, but at layer 0 — where the divergence
    // actually starts (embedding is proven identical, so any difference at
    // layer 0's output is created entirely within layer 0's own math). ---
    {
        let layer_idx = 0u32;
        let t = load_layer_tensors(&pkg, &gllm_dir, layer_idx);
        let kv_dim = metadata.head_count_kv as usize * metadata.head_dim().expect("head_dim") as usize;
        let post_attn = replay_attention_only(
            &t,
            embed_b,
            shared.dim,
            kv_dim,
            metadata.num_heads as usize,
            metadata.head_count_kv as usize,
            shared.rms_eps,
        );
        let full_layer0_output = &hidden_b[layer_idx as usize];
        println!("layer {layer_idx} attention-only replay (Path B, real weights, exact for cached_len=1):");
        println!(
            "  post-attention norm={:.4}  |  real full-layer(post-FFN) norm={:.4}",
            l2_norm(&post_attn),
            l2_norm(full_layer0_output),
        );

        let gt = load_gguf_layer_tensors(&gguf, layer_idx);
        let q_dim = model.config.n_heads * model.config.head_dim;
        let kv_dim_a = model.config.n_kv_heads * model.config.head_dim;
        let replay = replay_full_layer_f32(
            &gt,
            &embed_a,
            model.config.dim,
            q_dim,
            kv_dim_a,
            model.config.n_heads,
            model.config.n_kv_heads,
            model.config.hidden_dim,
            model.config.rms_eps,
        );
        let real_a_layer0 = &hidden_a[layer_idx as usize];
        println!("layer {layer_idx} F32-only replay of Path A (no native-quantized kernels):");
        println!(
            "  F32-replay norm={:.4}  |  Path A REAL (native-quantized) norm={:.4}",
            l2_norm(&replay),
            l2_norm(real_a_layer0),
        );
        println!("  F32-replay first8  = {:?}", &replay[..8]);
        println!("  Path A REAL first8 = {:?}", &real_a_layer0[..8]);
        println!("  Path B REAL first8 = {:?}", &full_layer0_output[..8]);
        println!("  max_diff(F32-replay-A, real-A)={:.6}  max_diff(post-attn-B, real-B)={:.6}",
            max_abs_diff(&replay, real_a_layer0),
            max_abs_diff(&post_attn, full_layer0_output),
        );
        println!();
    }

    // --- localize layer 2's divergence: attention block alone, or does the
    // spike only appear after FFN runs? Path B only (pure F32, so this
    // replay is exactly faithful to what GlprocBackend really computes —
    // unlike a Path A replay, which would need to reproduce its native
    // quantized kernels to be trustworthy). ---
    {
        let layer_idx = 2u32;
        let t = load_layer_tensors(&pkg, &gllm_dir, layer_idx);
        let input = &hidden_b[(layer_idx as usize) - 1]; // layer 2's real input = layer 1's real output
        let kv_dim = metadata.head_count_kv as usize * metadata.head_dim().expect("head_dim") as usize;
        let post_attn = replay_attention_only(
            &t,
            input,
            shared.dim,
            kv_dim,
            metadata.num_heads as usize,
            metadata.head_count_kv as usize,
            shared.rms_eps,
        );
        let full_layer2_output = &hidden_b[layer_idx as usize];
        println!(
            "layer {layer_idx} attention-only replay (Path B, real weights, exact for cached_len=1):"
        );
        println!(
            "  post-attention norm={:.4} dim62={:.4}  |  real full-layer(post-FFN) norm={:.4} dim62={:.4}",
            l2_norm(&post_attn),
            post_attn[62],
            l2_norm(full_layer2_output),
            full_layer2_output[62],
        );
        println!();
    }

    // --- is layer 2's ffn_down.weight (the tensor "massive activations"
    // literature most often implicates) actually byte-equivalent between a
    // fresh GGUF dequant and the F32 package's copy? Same check as the
    // layer-0 attn_q.weight sanity check above, for the tensor most likely
    // to matter here specifically. ---
    for tensor_name in ["ffn_down.weight", "ffn_gate.weight", "ffn_up.weight", "attn_output.weight"] {
        let gguf_name = format!("blk.2.{tensor_name}");
        let info = gguf.tensors.iter().find(|t| t.name == gguf_name).unwrap_or_else(|| panic!("find {gguf_name}"));
        let dequant_from_gguf = dequant_gguf_tensor(&gguf, &gguf_name);

        let layer2_manifest = pkg.layer_manifest(2).expect("layer 2 manifest");
        let layer2_path = Path::new(&gllm_dir).join(&layer2_manifest.file);
        let layer2_mapping = LayerMapping::open(&layer2_path, Some(2), false).expect("open layer 2");
        let entry = layer2_manifest.tensor(tensor_name).unwrap_or_else(|| panic!("{tensor_name} in layer 2 manifest"));
        let bytes = layer2_mapping.tensor_bytes(tensor_name).unwrap_or_else(|| panic!("{tensor_name} bytes"));
        let from_package: Vec<f32> = glcore::format::decode_tensor(tensor_name, &entry.shape, dtype_str(entry.dtype), bytes)
            .unwrap_or_else(|e| panic!("decode {tensor_name}: {e}"))
            .data;

        println!(
            "blk.2.{tensor_name} (source dtype {:?}): GGUF-dequant norm={:.4} package-F32 norm={:.4} max_diff={:.8}",
            info.dtype,
            l2_norm(&dequant_from_gguf),
            l2_norm(&from_package),
            max_abs_diff(&dequant_from_gguf, &from_package)
        );
    }
    println!();

    // --- decisive check: replay Path A's layer 2 in PURE F32 (no native
    // quantized/bridge kernels), using Path A's own real layer-1 output as
    // input. If this reproduces Path A's REAL (native-quantized) layer-2
    // output, the divergence is not a quantization-computation-path
    // artifact. If it does NOT — i.e. this F32 replay stays modest like
    // Path B, while the real native-quantized Path A explodes — that
    // pinpoints the whole divergence as a native-quantized-kernel effect,
    // not a GlprocBackend bug at all. ---
    {
        let layer_idx = 2u32;
        let gt = load_gguf_layer_tensors(&gguf, layer_idx);
        let input = &hidden_a[(layer_idx as usize) - 1]; // Path A's REAL layer-1 output
        let q_dim = model.config.n_heads * model.config.head_dim;
        let kv_dim = model.config.n_kv_heads * model.config.head_dim;
        let replay = replay_full_layer_f32(
            &gt,
            input,
            model.config.dim,
            q_dim,
            kv_dim,
            model.config.n_heads,
            model.config.n_kv_heads,
            model.config.hidden_dim,
            model.config.rms_eps,
        );
        let real_a_layer2 = &hidden_a[layer_idx as usize];
        println!("layer {layer_idx} F32-only replay of Path A (no native-quantized kernels):");
        println!(
            "  F32-replay norm={:.4} dim62={:.4}  |  Path A REAL (native-quantized) norm={:.4} dim62={:.4}",
            l2_norm(&replay),
            replay[62],
            l2_norm(real_a_layer2),
            real_a_layer2[62],
        );
        println!();
    }

    println!();
    println!("logits: A_len={} B_len={}", logits_a.len(), logits_b.len());
    println!("top5 A: {:?}", top_k(&logits_a, 5));
    println!("top5 B: {:?}", top_k(&logits_b, 5));
    let logit_diff: Vec<f32> = logits_a.iter().zip(&logits_b).map(|(x, y)| x - y).collect();
    println!("logits L2 diff: {:.4}", l2_norm(&logit_diff));

    match first_divergence {
        Some(i) => println!("\n=== first meaningful divergence (max_diff > 0.01) at layer {i} ==="),
        None => println!("\n=== no layer showed max_diff > 0.01 — hidden states matched at every layer ==="),
    }

    // --- MULTI-TOKEN extension ---
    //
    // Everything above tested exactly one isolated token at position 0,
    // where `cached_len == 1` makes attention mathematically trivial
    // (softmax over one score is always 1.0). The REAL garbage-output
    // symptom was only ever observed on multi-token generation (real
    // prompts, 10-20+ tokens) — a regime this single-token test cannot see
    // at all: position 1 onward has a real, non-empty KV cache and
    // non-trivial attention. Feed a short sequence through both paths,
    // sequentially (matching how real generation actually calls each path,
    // one token at a time, KV cache persisting across calls), and compare
    // at position 1 — the first position where attention is not trivial.
    println!("\n\n=== multi-token extension (position 1 has a real, non-trivial KV cache) ===");
    let seq: Vec<u32> = vec![token, token, token]; // same id 3x: isolates position/KV effects from token-content effects
    println!("token sequence = {seq:?}");

    let mut runner2 = glproc::runner::Runner::new(&model);
    runner2.set_capture_hidden(true);
    let mut hidden_a_multi: Vec<Vec<Vec<f32>>> = Vec::new(); // [pos][layer] -> hidden state
    for (pos, &tok) in seq.iter().enumerate() {
        runner2.forward_chunk_into(&[tok], pos).expect("Path A multi-token forward pass");
        hidden_a_multi.push(runner2.hidden_dump().to_vec());
    }

    let inner_backend2 = GlprocBackend::new(&metadata).expect("build GlprocBackend (multi-token)");
    let capturing2 = Arc::new(CapturingBackend { inner: inner_backend2, captured: Mutex::new(Vec::new()) });
    let backend_for_runtime2: Arc<dyn ExecutionBackend> = capturing2.clone();
    let mut runtime2 = GllmRuntime::open(Path::new(&gllm_dir), RuntimeConfig::default(), backend_for_runtime2)
        .expect("open GllmRuntime (multi-token)");
    let n_layers = model.config.n_layers;
    let mut hidden_b_multi: Vec<Vec<Vec<f32>>> = Vec::new();
    for &tok in &seq {
        let row = shared.embed[(tok as usize) * shared.dim..(tok as usize + 1) * shared.dim].to_vec();
        let mut act = ActivationBuffer { data: row, shape: vec![shared.dim] };
        runtime2.run(&mut act).expect("Path B multi-token forward pass");
    }
    let all_captured = capturing2.captured.lock().unwrap().clone();
    for pos in 0..seq.len() {
        hidden_b_multi.push(all_captured[pos * n_layers..(pos + 1) * n_layers].to_vec());
    }

    for pos in 0..seq.len() {
        println!("\n-- position {pos} (cached_len={}) --", pos + 1);
        println!("{:<7} {:>12} {:>12} {:>12}", "layer", "A_norm", "B_norm", "max_diff");
        for l in 0..n_layers {
            let a = &hidden_a_multi[pos][l];
            let b = &hidden_b_multi[pos][l];
            let marker = if max_abs_diff(a, b) > 0.01 { " <-- DIVERGE" } else { "" };
            println!(
                "{l:<7} {:>12.4} {:>12.4} {:>12.6}{marker}",
                l2_norm(a),
                l2_norm(b),
                max_abs_diff(a, b)
            );
        }
    }
}
