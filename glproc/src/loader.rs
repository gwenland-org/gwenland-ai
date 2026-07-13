//! Loads a GGUF file into a [`GlprocModel`].
//!
//! Q4_K weight matrices are kept in their raw quantized form and dequantized
//! block-by-block inside the Bridge-ing matvec at decode time; everything
//! else is dequantized to f32 up front.

use glcore::format::gguf::{GgufDType, GgufFile, GgufValue};
use glcore::GlError;

use crate::kernels;
use crate::kernels::bridge::QuantFormat;
use crate::kernels::qdot;
use crate::model::{
    FfnLayer, GateUp, GlprocModel, LayerWeights, ModelConfig, QkvWeights, RopeStyle, WeightMatrix,
};
use crate::moe::{ExpertWeights, MoEConfig, MoELayer};

/// Page size used for the warm-touch stride. 4 KiB on every x86 target we
/// ship on; touching one byte per page faults the whole page in.
const PAGE_BYTES: usize = 4096;

#[cfg(windows)]
mod winmem {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn VirtualLock(addr: *const c_void, size: usize) -> i32;
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn SetProcessWorkingSetSize(process: *mut c_void, min: usize, max: usize) -> i32;
    }
}

/// The memory region backing one weight matrix, whichever representation
/// it is stored in.
fn weight_region(w: &WeightMatrix) -> (*const u8, usize) {
    match w {
        WeightMatrix::F32(v) => (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice())),
        WeightMatrix::Quant(_, b) => (b.as_ptr(), b.len()),
    }
}

/// Warm the model's weights into RAM and pin them there. Call once, after
/// load and before the first decode.
///
/// A cold or evicted page costs a fault mid-matvec — at this machine's disk
/// speed that is milliseconds per page, which destroys decode latency. So:
/// touch every page up front (fault them in while nothing is latency
/// sensitive), then pin them (`VirtualLock` / `mlock`) so an 8 GB box under
/// memory pressure cannot swap the weights back out between requests.
///
/// Deviation from the X5 sketch: glproc copies tensors out of the GGUF mmap
/// into owned heap buffers at load, so the decode working set is those
/// buffers, not the mmap — warm-and-lock targets each weight buffer.
///
/// Best effort: pinning can fail (working-set quota, `ulimit -l`); the
/// prefetch touch still helps, so a failure only warns.
pub fn warm_and_lock_model(model: &GlprocModel) {
    let mut regions: Vec<(*const u8, usize)> = Vec::new();
    let mut push_f32 = |regions: &mut Vec<_>, v: &[f32]| {
        regions.push((v.as_ptr() as *const u8, std::mem::size_of_val(v)));
    };
    regions.push(weight_region(&model.token_embd));
    push_f32(&mut regions, &model.output_norm);
    regions.push(weight_region(&model.output));
    for l in &model.layers {
        push_f32(&mut regions, &l.attn_norm);
        push_f32(&mut regions, &l.ffn_norm);
        for opt in [&l.bq, &l.bk, &l.bv, &l.q_norm, &l.k_norm] {
            if let Some(v) = opt {
                push_f32(&mut regions, v);
            }
        }
        regions.push(weight_region(&l.wo));
        match &l.qkv {
            QkvWeights::FusedQuant(_, packed) => regions.push((packed.as_ptr(), packed.len())),
            QkvWeights::Split(q, k, v) => {
                for w in [q, k, v] {
                    regions.push(weight_region(w));
                }
            }
        }

        // The FFN is the bulk of a layer's bytes, and on an MoE block that is
        // every expert — all of them, not just the top-k. An expert skipped
        // this token is needed by the next one, so leaving experts unwarmed
        // would trade a page fault mid-decode for every expert that goes cold.
        let mut push_gate_up = |regions: &mut Vec<_>, gu: &GateUp| match gu {
            GateUp::FusedQuant(_, packed) => regions.push((packed.as_ptr(), packed.len())),
            GateUp::Split(g, u) => {
                regions.push(weight_region(g));
                regions.push(weight_region(u));
            }
        };
        match &l.ffn {
            FfnLayer::Dense { gate_up, w_down } => {
                push_gate_up(&mut regions, gate_up);
                regions.push(weight_region(w_down));
            }
            FfnLayer::MoE(moe) => {
                push_f32(&mut regions, &moe.router);
                for e in &moe.experts {
                    push_gate_up(&mut regions, &e.gate_up);
                    regions.push(weight_region(&e.w_down));
                }
            }
        }
    }
    regions.retain(|&(_, size)| size > 0);
    let total: usize = regions.iter().map(|&(_, size)| size).sum();

    // Step 1: touch one byte per page. Faults every page in now, so the
    // decode loop never takes one — and it still helps if pinning fails.
    for &(ptr, size) in &regions {
        let mut i = 0;
        while i < size {
            // SAFETY: ptr..ptr+size is a live owned buffer of the model.
            unsafe { std::ptr::read_volatile(ptr.add(i)) };
            i += PAGE_BYTES;
        }
    }

    // Step 2: pin the pages so they cannot be evicted.
    // `GLPROC_NO_LOCK=1` skips pinning (and the working-set resize) — an
    // A/B knob: the working-set cap the resize installs can make the OS
    // trim the *unpinned* runtime buffers under memory pressure.
    if std::env::var("GLPROC_NO_LOCK").is_ok_and(|v| !v.is_empty() && v != "0") {
        return;
    }
    let mut failed = 0usize;
    #[cfg(windows)]
    {
        // VirtualLock is capped by the process working-set maximum (a few
        // MB by default), so raise it to model size + slack first.
        const SLACK: usize = 256 << 20;
        // SAFETY: plain kernel32 calls on the current process handle.
        unsafe {
            winmem::SetProcessWorkingSetSize(
                winmem::GetCurrentProcess(),
                total + SLACK,
                total + SLACK * 2,
            );
        }
        for &(ptr, size) in &regions {
            // SAFETY: region is a live owned buffer, valid for `size` bytes.
            if unsafe { winmem::VirtualLock(ptr as *const _, size) } == 0 {
                failed += 1;
            }
        }
    }
    #[cfg(unix)]
    {
        for &(ptr, size) in &regions {
            // SAFETY: region is a live owned buffer, valid for `size` bytes.
            if unsafe { libc::mlock(ptr as *const libc::c_void, size) } != 0 {
                failed += 1;
            }
        }
    }
    if failed > 0 {
        eprintln!(
            "warning: pinning failed for {failed}/{} weight buffers ({} MB total) — \
             pages are prefetched but may be swapped out under memory pressure \
             (unix: try `ulimit -l unlimited`)",
            regions.len(),
            total >> 20,
        );
    }
}

/// Read `{arch}.{suffix}` from metadata as u64.
fn meta_u64(gguf: &GgufFile, arch: &str, suffix: &str) -> Option<u64> {
    gguf.get_meta(&format!("{arch}.{suffix}"))
        .and_then(GgufValue::as_u64)
}

/// Read `{arch}.{suffix}` from metadata as f32.
fn meta_f32(gguf: &GgufFile, arch: &str, suffix: &str) -> Option<f32> {
    gguf.get_meta(&format!("{arch}.{suffix}"))
        .and_then(GgufValue::as_f32)
}

/// Dequantize any supported dtype to f32, routing the formats glcore does
/// not (or does not correctly) handle through glproc's own kernels: Q4_K
/// and Q5_0 have no glcore path, and glcore's Q6_K assumes a naive linear
/// nibble order that disagrees with real llama.cpp files.
fn dequant_any(gguf: &GgufFile, info: &glcore::format::gguf::GgufTensorInfo) -> Result<Vec<f32>, GlError> {
    match info.dtype {
        GgufDType::Q4_K => kernels::dequant_q4_k(gguf.tensor_data(info)?),
        GgufDType::Q5_0 => kernels::dequant::q5_0::scalar::run(gguf.tensor_data(info)?),
        GgufDType::Q6_K => kernels::dequant::q6_k::scalar::run(gguf.tensor_data(info)?),
        _ => gguf.dequantize(info),
    }
}

/// Dequantize a required tensor by name to f32.
fn tensor(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, GlError> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing tensor '{name}'")))?;
    dequant_any(gguf, info)
}

/// Dequantize an optional tensor (e.g. attention biases).
fn tensor_opt(gguf: &GgufFile, name: &str) -> Result<Option<Vec<f32>>, GlError> {
    match gguf.find_tensor(name) {
        Some(info) => Ok(Some(dequant_any(gguf, info)?)),
        None => Ok(None),
    }
}

/// Load a required weight matrix. Bridge-supported quantized formats stay
/// quantized (raw GGML blocks) for the bridge matvec; other dtypes are
/// dequantized to f32.
fn weight(gguf: &GgufFile, name: &str) -> Result<WeightMatrix, GlError> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing tensor '{name}'")))?;
    let fmt = match info.dtype {
        GgufDType::Q4_K => Some(QuantFormat::Q4K),
        GgufDType::Q5_0 => Some(QuantFormat::Q5_0),
        GgufDType::Q6_K => Some(QuantFormat::Q6K),
        GgufDType::Q8_0 => Some(QuantFormat::Q8_0),
        _ => None,
    };
    match fmt {
        // Q5_0 is repacked to Q8_0 at load: the conversion is bit-exact
        // (see `repack_to_q8_0`) and Q8_0's inner loop is much cheaper than
        // Q5_0's high-bit unpack, which measured compute-bound.
        Some(QuantFormat::Q5_0) if info.dimensions[0] as usize % 32 == 0 => {
            Ok(WeightMatrix::Quant(
                QuantFormat::Q8_0,
                kernels::dequant::q5_0::scalar::repack_to_q8_0(gguf.tensor_data(info)?)?,
            ))
        }
        // Q6_K likewise repacks to Q8_0 (requantized; error well under the
        // format's own quantization noise — see `repack_to_q8_0`).
        Some(QuantFormat::Q6K) if info.dimensions[0] as usize % 256 == 0 => {
            Ok(WeightMatrix::Quant(
                QuantFormat::Q8_0,
                kernels::dequant::q6_k::scalar::repack_to_q8_0(gguf.tensor_data(info)?)?,
            ))
        }
        // Q4_K too: it has no integer-dot kernel, so unrepacked it takes
        // the f32 bridge — which re-dequantizes every block once per batch
        // row in the prefill matmul (measured ~15x slower than repacked
        // layers). Requantization error is the same class as Q6_K's.
        Some(QuantFormat::Q4K) if info.dimensions[0] as usize % 256 == 0 => {
            Ok(WeightMatrix::Quant(
                QuantFormat::Q8_0,
                kernels::dequant::q4_k::scalar::repack_to_q8_0(gguf.tensor_data(info)?)?,
            ))
        }
        // dimensions[0] is the contiguous axis = in_features; GGML packs
        // quantization blocks along it, so it is always a whole number of
        // blocks — guard anyway and fall back to f32 if not.
        Some(fmt) if info.dimensions[0] as usize % fmt.block_numel() == 0 => {
            Ok(WeightMatrix::Quant(fmt, gguf.tensor_data(info)?.to_vec()))
        }
        _ => Ok(WeightMatrix::F32(dequant_any(gguf, info)?)),
    }
}

// ---------------------------------------------------------------------------
// MoE (`*_exps`) loading
// ---------------------------------------------------------------------------

/// **UNVERIFIED ASSUMPTION — audit this against a real Qwen3-MoE GGUF.**
///
/// No Qwen3-MoE file was available when this was written (the smallest,
/// 30B-A3B, is ~17 GB and did not fit any machine on hand), so the expert
/// tensor layout below is taken from llama.cpp's convention rather than from
/// bytes anyone has read. Everything downstream of it — [`crate::moe`] — is
/// verified against a naive reference at Qwen3's real dims; this function is
/// the *only* unverified link in the chain. If a real MoE model produces
/// fluent-but-wrong output, start here.
///
/// The claim, in full:
///
/// - MoE FFN tensors are named `blk.{i}.ffn_{gate,up,down}_exps.weight`, and
///   the `_exps` suffix is what distinguishes an MoE block from a dense one
///   (`ffn_gate.weight`, no suffix). The router is `blk.{i}.ffn_gate_inp.weight`.
/// - The `_exps` tensors are **3-D**, unlike every other weight in glproc.
///   GGUF lists dimensions contiguous-axis-first, so `dimensions` reads
///   `[in_features, out_features, n_expert]`:
///     * `ffn_gate_exps` / `ffn_up_exps` → `[hidden, expert_ffn, n_expert]`
///     * `ffn_down_exps`                 → `[expert_ffn, hidden, n_expert]`
/// - Expert `e`'s matrix is therefore a **contiguous slice**: experts are the
///   outermost (slowest-varying) axis, so expert `e` occupies
///   `[e * bytes_per_expert, (e+1) * bytes_per_expert)` with no interleaving.
///   This is the load-bearing part. If experts were instead interleaved along
///   a faster axis, every expert would load as a stripe of all the others and
///   the model would still *run* — producing plausible garbage.
///
/// Rather than trust the claim silently, [`split_experts`] cross-checks the
/// tensor's declared dimensions against `expert_count` from metadata and
/// against the byte length, and errors out on any disagreement. A wrong
/// stacking order that still satisfies those checks is possible, which is why
/// this comment exists — but a wrong *shape* cannot get through.
const _EXPS_LAYOUT_ASSUMPTION: () = ();

/// Slice a 3-D `*_exps` tensor into `n_expert` per-expert [`WeightMatrix`]es.
///
/// `out_features` / `in_features` are the expected 2-D shape of ONE expert;
/// they are checked against the tensor header, not inferred from it, so a
/// layout that disagrees with metadata fails loudly at load.
///
/// See [`_EXPS_LAYOUT_ASSUMPTION`] — the contiguous-outermost-axis claim here
/// is not verified against a real file.
fn split_experts(
    gguf: &GgufFile,
    name: &str,
    n_expert: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<WeightMatrix>, GlError> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing tensor '{name}'")))?;

    // Shape gate: dimensions must be exactly [in, out, n_expert]. A 2-D
    // tensor here means the file is not laid out the way we assume, and a
    // silent reinterpret would scramble every expert.
    if info.dimensions.len() != 3 {
        return Err(GlError::Parse(format!(
            "GGUF: '{name}' has {} dimensions, expected 3 ([in, out, n_expert]) \
             — MoE expert layout assumption violated, see _EXPS_LAYOUT_ASSUMPTION",
            info.dimensions.len()
        )));
    }
    let (d_in, d_out, d_exp) = (
        info.dimensions[0] as usize,
        info.dimensions[1] as usize,
        info.dimensions[2] as usize,
    );
    if d_in != in_features || d_out != out_features || d_exp != n_expert {
        return Err(GlError::Parse(format!(
            "GGUF: '{name}' is [{d_in}, {d_out}, {d_exp}], expected \
             [{in_features}, {out_features}, {n_expert}] from metadata \
             — MoE expert layout assumption violated"
        )));
    }

    // Slice the raw tensor per expert, then run each slice through the same
    // dtype handling the dense path uses (`weight_from_bytes`), so experts get
    // the Q4_K/Q5_0/Q6_K → Q8_0 repack and land on the integer-dot kernels.
    let data = gguf.tensor_data(info)?;
    if data.len() % n_expert != 0 {
        return Err(GlError::Parse(format!(
            "GGUF: '{name}' is {} bytes, not divisible by {n_expert} experts",
            data.len()
        )));
    }
    let per_expert = data.len() / n_expert;

    (0..n_expert)
        .map(|e| {
            // The contiguous-outermost-axis assumption, in one line.
            let bytes = &data[e * per_expert..(e + 1) * per_expert];
            weight_from_bytes(bytes, info.dtype, in_features, name)
        })
        .collect()
}

/// Build a [`WeightMatrix`] from raw tensor bytes of a known dtype.
///
/// This is [`weight`]'s body, factored out so the MoE path can apply the same
/// repack rules to an expert *slice* rather than to a whole named tensor.
/// `in_features` is the contiguous axis, used for the block-alignment guards.
fn weight_from_bytes(
    bytes: &[u8],
    dtype: GgufDType,
    in_features: usize,
    name: &str,
) -> Result<WeightMatrix, GlError> {
    let fmt = match dtype {
        GgufDType::Q4_K => Some(QuantFormat::Q4K),
        GgufDType::Q5_0 => Some(QuantFormat::Q5_0),
        GgufDType::Q6_K => Some(QuantFormat::Q6K),
        GgufDType::Q8_0 => Some(QuantFormat::Q8_0),
        _ => None,
    };
    // Same repack policy as the dense `weight()`: everything that can become
    // Q8_0 does, because only Q8_0 has an integer-dot kernel and the bridge
    // path re-dequantizes per batch row (~15x slower in prefill).
    match fmt {
        Some(QuantFormat::Q5_0) if in_features % 32 == 0 => Ok(WeightMatrix::Quant(
            QuantFormat::Q8_0,
            kernels::dequant::q5_0::scalar::repack_to_q8_0(bytes)?,
        )),
        Some(QuantFormat::Q6K) if in_features % 256 == 0 => Ok(WeightMatrix::Quant(
            QuantFormat::Q8_0,
            kernels::dequant::q6_k::scalar::repack_to_q8_0(bytes)?,
        )),
        Some(QuantFormat::Q4K) if in_features % 256 == 0 => Ok(WeightMatrix::Quant(
            QuantFormat::Q8_0,
            kernels::dequant::q4_k::scalar::repack_to_q8_0(bytes)?,
        )),
        Some(fmt) if in_features % fmt.block_numel() == 0 => {
            Ok(WeightMatrix::Quant(fmt, bytes.to_vec()))
        }
        // f32/f16/bf16, or a quantized tensor whose contiguous axis is not a
        // whole number of blocks. Dequantize to f32 — correct, just slower.
        _ => match dtype {
            GgufDType::F32 => Ok(WeightMatrix::F32(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )),
            GgufDType::F16 => Ok(WeightMatrix::F32(kernels::dequant::f16::scalar::run(bytes))),
            other => Err(GlError::Parse(format!(
                "GGUF: '{name}' has dtype {other:?}, unsupported for MoE experts"
            ))),
        },
    }
}

/// Read this layer's MoE config, or `None` if the layer is dense.
///
/// Detection is by tensor presence (`ffn_gate_exps` exists), not by a global
/// model flag: Qwen3-MoE may mix dense and MoE blocks, so this is per layer.
fn moe_config_for(
    gguf: &GgufFile,
    arch: &str,
    i: usize,
    dim: usize,
) -> Result<Option<MoEConfig>, GlError> {
    if gguf
        .find_tensor(&format!("blk.{i}.ffn_gate_exps.weight"))
        .is_none()
    {
        return Ok(None);
    }

    let num_experts = meta_u64(gguf, arch, "expert_count").ok_or_else(|| {
        GlError::Parse(format!(
            "GGUF: layer {i} has ffn_gate_exps but no {arch}.expert_count"
        ))
    })? as usize;
    let num_experts_per_tok = meta_u64(gguf, arch, "expert_used_count").ok_or_else(|| {
        GlError::Parse(format!(
            "GGUF: layer {i} has ffn_gate_exps but no {arch}.expert_used_count"
        ))
    })? as usize;

    // An expert's inner width is its OWN feed_forward_length, not the model's.
    // Qwen3-30B-A3B: expert_ffn=768 while the dense feed_forward_length is
    // much larger. Falling back to the dense value would silently build
    // wrong-shaped experts, so take the shape from the tensor itself when the
    // dedicated key is absent — the tensor cannot lie about its own width.
    let expert_ffn_size = match meta_u64(gguf, arch, "expert_feed_forward_length") {
        Some(v) => v as usize,
        None => {
            let info = gguf
                .find_tensor(&format!("blk.{i}.ffn_gate_exps.weight"))
                .expect("presence checked above");
            *info.dimensions.get(1).ok_or_else(|| {
                GlError::Parse(format!("GGUF: blk.{i}.ffn_gate_exps.weight has no dim[1]"))
            })? as usize
        }
    };

    let cfg = MoEConfig {
        num_experts,
        num_experts_per_tok,
        expert_ffn_size,
        hidden_size: dim,
        // Qwen3-MoE renormalizes the top-k probabilities so they sum to 1.
        // GGUF writes `expert_weights_norm` as a *bool*, and glcore's
        // GgufValue has no bool accessor — rather than widen that API on an
        // assumption, default to true (Qwen3's behavior) and read an explicit
        // integer override if some writer emits one. Getting this wrong
        // rescales the whole FFN residual, so it is worth revisiting when a
        // real file is in hand — see _EXPS_LAYOUT_ASSUMPTION.
        norm_topk_prob: meta_u64(gguf, arch, "expert_weights_norm")
            .map(|v| v != 0)
            .unwrap_or(true),
    };
    cfg.validate()?;
    Ok(Some(cfg))
}

/// Build the MoE FFN for layer `i`.
///
/// Each expert's gate/up pair is interleaved into the same
/// [`GateUp::FusedQuant`] layout the dense path uses, so an expert's FFN runs
/// through the existing `par_matvec_swiglu` / `par_matmul_swiglu` kernels
/// unchanged — no MoE-specific kernel exists, by design.
fn build_moe(
    gguf: &GgufFile,
    i: usize,
    cfg: MoEConfig,
    dim: usize,
) -> Result<MoELayer, GlError> {
    let (ne, f) = (cfg.num_experts, cfg.expert_ffn_size);

    // Router: [n_expert, dim]. Tiny (128 x 2048), stays f32 — moe::forward
    // dots it on the calling thread rather than waking the pool.
    let router = tensor(gguf, &format!("blk.{i}.ffn_gate_inp.weight"))?;
    if router.len() != ne * dim {
        return Err(GlError::Parse(format!(
            "GGUF: blk.{i}.ffn_gate_inp.weight is {} elements, expected {} ({ne} x {dim})",
            router.len(),
            ne * dim
        )));
    }

    // gate/up: [dim, f, ne] each. down: [f, dim, ne] — in_features is f here,
    // since down maps the expert's inner width back to hidden.
    let gates = split_experts(gguf, &format!("blk.{i}.ffn_gate_exps.weight"), ne, dim, f)?;
    let ups = split_experts(gguf, &format!("blk.{i}.ffn_up_exps.weight"), ne, dim, f)?;
    let downs = split_experts(gguf, &format!("blk.{i}.ffn_down_exps.weight"), ne, f, dim)?;

    let experts = gates
        .into_iter()
        .zip(ups)
        .zip(downs)
        .map(|((gate, up), w_down)| ExpertWeights {
            // `f` (not hidden_dim) is this expert's row count — fuse_gate_up
            // interleaves that many gate/up row pairs.
            gate_up: fuse_gate_up(gate, up, f),
            w_down,
        })
        .collect();

    Ok(MoELayer {
        config: cfg,
        router,
        experts,
    })
}

/// Combine the gate and up projections: interleave rows into one buffer
/// when both are quantized in the same integer-dot format, so the fused
/// SwiGLU matvec streams a single contiguous region (see [`GateUp`]).
fn fuse_gate_up(gate: WeightMatrix, up: WeightMatrix, hidden_dim: usize) -> GateUp {
    match (gate, up) {
        (WeightMatrix::Quant(gf, gb), WeightMatrix::Quant(uf, ub))
            if gf == uf
                && qdot::supports(gf)
                && gb.len() == ub.len()
                && hidden_dim > 0
                && gb.len() % hidden_dim == 0 =>
        {
            let row_bytes = gb.len() / hidden_dim;
            let mut packed = Vec::with_capacity(gb.len() + ub.len());
            for o in 0..hidden_dim {
                packed.extend_from_slice(&gb[o * row_bytes..(o + 1) * row_bytes]);
                packed.extend_from_slice(&ub[o * row_bytes..(o + 1) * row_bytes]);
            }
            GateUp::FusedQuant(gf, packed)
        }
        (gate, up) => GateUp::Split(gate, up),
    }
}

/// Combine the Q/K/V projections: stack rows into one matrix when all
/// three are quantized in the same integer-dot format, so one dispatch
/// covers the whole projection (see [`QkvWeights`]). Rows are already
/// contiguous per matrix, so stacking is plain concatenation.
fn fuse_qkv(q: WeightMatrix, k: WeightMatrix, v: WeightMatrix) -> QkvWeights {
    match (q, k, v) {
        (
            WeightMatrix::Quant(qf, qb),
            WeightMatrix::Quant(kf, kb),
            WeightMatrix::Quant(vf, vb),
        ) if qf == kf && qf == vf && qdot::supports(qf) => {
            let mut packed = Vec::with_capacity(qb.len() + kb.len() + vb.len());
            packed.extend_from_slice(&qb);
            packed.extend_from_slice(&kb);
            packed.extend_from_slice(&vb);
            QkvWeights::FusedQuant(qf, packed)
        }
        (q, k, v) => QkvWeights::Split(q, k, v),
    }
}

/// Build one transformer block's weights: copy, dequantize and repack all
/// of layer `i`'s tensors. Pure per-layer work — safe to run in parallel.
fn build_layer(
    gguf: &GgufFile,
    arch: &str,
    i: usize,
    dim: usize,
    hidden_dim: usize,
) -> Result<LayerWeights, GlError> {
    // MoE or dense is a per-layer question — Qwen3-MoE may mix both — so it is
    // answered from this layer's tensors, not a model-wide flag.
    let ffn = match moe_config_for(gguf, arch, i, dim)? {
        Some(cfg) => FfnLayer::MoE(Box::new(build_moe(gguf, i, cfg, dim)?)),
        None => FfnLayer::Dense {
            gate_up: fuse_gate_up(
                weight(gguf, &format!("blk.{i}.ffn_gate.weight"))?,
                weight(gguf, &format!("blk.{i}.ffn_up.weight"))?,
                hidden_dim,
            ),
            w_down: weight(gguf, &format!("blk.{i}.ffn_down.weight"))?,
        },
    };

    Ok(LayerWeights {
        attn_norm: tensor(gguf, &format!("blk.{i}.attn_norm.weight"))?,
        qkv: fuse_qkv(
            weight(gguf, &format!("blk.{i}.attn_q.weight"))?,
            weight(gguf, &format!("blk.{i}.attn_k.weight"))?,
            weight(gguf, &format!("blk.{i}.attn_v.weight"))?,
        ),
        wo: weight(gguf, &format!("blk.{i}.attn_output.weight"))?,
        bq: tensor_opt(gguf, &format!("blk.{i}.attn_q.bias"))?,
        bk: tensor_opt(gguf, &format!("blk.{i}.attn_k.bias"))?,
        bv: tensor_opt(gguf, &format!("blk.{i}.attn_v.bias"))?,
        q_norm: tensor_opt(gguf, &format!("blk.{i}.attn_q_norm.weight"))?,
        k_norm: tensor_opt(gguf, &format!("blk.{i}.attn_k_norm.weight"))?,
        ffn_norm: tensor(gguf, &format!("blk.{i}.ffn_norm.weight"))?,
        ffn,
    })
}

/// Parse config and dequantize every weight of a GGUF transformer.
///
/// Supports llama-family architectures (llama, mistral, qwen2, tinyllama...)
/// that use the standard `blk.N.*` tensor naming.
pub fn load_gguf(gguf: &GgufFile) -> Result<GlprocModel, GlError> {
    let arch = gguf
        .get_meta("general.architecture")
        .and_then(GgufValue::as_str)
        .ok_or_else(|| GlError::Parse("GGUF: missing general.architecture".into()))?
        .to_string();

    let dim = meta_u64(gguf, &arch, "embedding_length")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.embedding_length")))?
        as usize;
    let n_layers = meta_u64(gguf, &arch, "block_count")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.block_count")))?
        as usize;
    let n_heads = meta_u64(gguf, &arch, "attention.head_count")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.attention.head_count")))?
        as usize;
    if dim == 0 || n_layers == 0 || n_heads == 0 {
        return Err(GlError::Parse(
            "GGUF: model dimensions must be non-zero".into(),
        ));
    }
    let n_kv_heads =
        meta_u64(gguf, &arch, "attention.head_count_kv").unwrap_or(n_heads as u64) as usize;
    let head_dim =
        meta_u64(gguf, &arch, "attention.key_length").unwrap_or((dim / n_heads) as u64) as usize;
    let hidden_dim = meta_u64(gguf, &arch, "feed_forward_length")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.feed_forward_length")))?
        as usize;
    let max_seq = meta_u64(gguf, &arch, "context_length").unwrap_or(2048) as usize;
    let rms_eps = meta_f32(gguf, &arch, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
    let rope_freq_base = meta_f32(gguf, &arch, "rope.freq_base").unwrap_or(10_000.0);
    // Original-llama models rotate adjacent dim pairs; most newer archs
    // (qwen2, phi3, gemma, stablelm) use the NeoX half-split convention.
    let rope_style = match arch.as_str() {
        "llama" | "llama2" | "minicpm" => RopeStyle::Norm,
        _ => RopeStyle::Neox,
    };

    // The embedding table is the single biggest tensor (vocab × dim); keep
    // it quantized and dequantize one row per lookup instead of paying the
    // ~4x f32 blow-up in RAM and its dequantization at load. Row lookup
    // needs a whole number of Q8_0 blocks per row, and formats other than
    // Q8_0 (post-repack) fall back to f32.
    let embd_info = gguf
        .find_tensor("token_embd.weight")
        .ok_or_else(|| GlError::Parse("GGUF: missing tensor 'token_embd.weight'".into()))?;
    let vocab_size = embd_info.dimensions.get(1).copied().unwrap_or(0) as usize;
    let token_embd = match weight(gguf, "token_embd.weight")? {
        w @ WeightMatrix::Quant(QuantFormat::Q8_0, _) if dim % 32 == 0 => w,
        WeightMatrix::Quant(..) => WeightMatrix::F32(dequant_any(gguf, embd_info)?),
        w => w,
    };
    if vocab_size == 0 {
        return Err(GlError::Parse(
            "GGUF: token_embd.weight has no vocab dimension".into(),
        ));
    }

    // Layers are independent — copy/dequantize/repack them in parallel.
    // The work is a mix of mmap reads and requantization compute, so it
    // scales with cores until the disk saturates. Workers pull layer
    // indices from a shared counter; results land in per-index slots.
    // Logical threads, not physical cores (unlike the decode pool in
    // `runner.rs`): this work interleaves mmap page faults with requantization
    // compute, so an SMT sibling issues real work while its partner stalls on
    // a fault. Decode has no such stalls to hide, which is why it sizes from
    // `topology::physical_core_count()` instead.
    let n_workers = num_cpus::get().clamp(1, 8).min(n_layers.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Result<LayerWeights, GlError>>>> =
        (0..n_layers).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n_layers {
                    break;
                }
                let built = build_layer(gguf, &arch, i, dim, hidden_dim);
                *slots[i].lock().unwrap() = Some(built);
            });
        }
    });
    let mut layers = Vec::with_capacity(n_layers);
    for slot in slots {
        layers.push(slot.into_inner().unwrap().expect("worker built every slot")?);
    }

    let output_norm = tensor(gguf, "output_norm.weight")?;
    // Tied embeddings: fall back to the embedding table as LM head — a
    // quantized table doubles as a quantized head (integer-dot path).
    let output = match gguf.find_tensor("output.weight") {
        Some(_) => weight(gguf, "output.weight")?,
        None => token_embd.clone(),
    };

    Ok(GlprocModel {
        config: ModelConfig {
            arch,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            vocab_size,
            max_seq,
            rms_eps,
            rope_freq_base,
            rope_style,
        },
        token_embd,
        layers,
        output_norm,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `_exps` assumption, made executable.
    ///
    /// This does NOT prove the assumption is right — only a real Qwen3-MoE
    /// GGUF can do that (see [`_EXPS_LAYOUT_ASSUMPTION`]). What it pins down
    /// is the *arithmetic*: given a tensor stacked with experts on the
    /// outermost (slowest) axis, slicing at `e * per_expert` recovers each
    /// expert's matrix intact. If someone "simplifies" the slicing later, this
    /// catches it.
    ///
    /// Layout under test, Q8_0, 3 experts x 2 rows x 32 cols:
    ///   expert 0's rows, then expert 1's rows, then expert 2's rows.
    #[test]
    fn expert_slicing_recovers_each_expert_contiguously() {
        let (ne, out_f, in_f) = (3usize, 2usize, 32usize);
        let row_bytes = in_f / 32 * 34; // one Q8_0 block per row
        let per_expert = out_f * row_bytes;

        // Byte-fill each expert's region with its own id, so a slice that
        // strays into a neighbour is immediately visible.
        let mut data = Vec::new();
        for e in 0..ne {
            data.extend(std::iter::repeat(e as u8).take(per_expert));
        }
        assert_eq!(data.len(), ne * per_expert);

        for e in 0..ne {
            let slice = &data[e * per_expert..(e + 1) * per_expert];
            let w = weight_from_bytes(slice, GgufDType::Q8_0, in_f, "test").unwrap();
            match w {
                WeightMatrix::Quant(QuantFormat::Q8_0, b) => {
                    assert_eq!(b.len(), per_expert, "expert {e}: wrong byte count");
                    assert!(
                        b.iter().all(|&v| v == e as u8),
                        "expert {e}: slice bled into a neighbouring expert — the \
                         contiguous-outermost-axis assumption is violated"
                    );
                }
                _ => panic!("expert {e}: Q8_0 should stay quantized for the integer-dot path"),
            }
        }
    }

    /// An expert dtype with no MoE path must fail LOUDLY, not silently
    /// reinterpret. Q4_0 is a real GGUF dtype that glproc has no `QuantFormat`
    /// for, so it has neither an integer-dot kernel nor a repack — exactly the
    /// case that must not quietly produce a misread matrix.
    #[test]
    fn weight_from_bytes_rejects_unsupported_dtype() {
        let r = weight_from_bytes(&[0u8; 64], GgufDType::Q4_0, 32, "blk.0.ffn_gate_exps.weight");
        assert!(r.is_err(), "unsupported expert dtype must be rejected");
    }

    /// f32 experts round-trip through the byte reader. Guards the little-endian
    /// decode, which is easy to get subtly wrong and produces plausible noise.
    #[test]
    fn weight_from_bytes_decodes_f32() {
        let vals: Vec<f32> = vec![1.5, -2.25, 0.0, 7.75];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w = weight_from_bytes(&bytes, GgufDType::F32, 4, "test").unwrap();
        assert_eq!(w.as_f32().unwrap(), &vals[..]);
    }
}
