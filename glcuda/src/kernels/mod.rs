//! Typed launch wrappers around the PTX kernel suite.
//!
//! Each method mirrors one row of the ArchGLML_X2 §16 kernel inventory and
//! encodes that kernel's launch geometry, so callers never repeat grid
//! math. All launches go to the default stream; the caller synchronizes
//! once per forward pass (or per test).

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use glcore::GlError;

use crate::driver::{Cuda, Kernel, Module};
use crate::ffi::CUdeviceptr;

/// The PTX image embedded in the binary (ADR-004 — no JIT of our own, the
/// driver compiles this for the actual device at module load).
pub const PTX: &str = include_str!("glcuda.ptx");

/// Turing tensor-core kernels (M2.1 Task B). A separate module because the
/// main image targets sm_70 and ptxas rejects instructions above a module's
/// `.target` — loaded only when the device reports sm_75+.
pub const PTX_SM75: &str = include_str!("glcuda_sm75.ptx");

/// Wave 59's isolated narrow-grid candidate. Keeping this in a separate
/// module means the retained sm_75 image and its default JIT cost do not
/// change unless the experiment is explicitly enabled.
pub const PTX_SM75_WAVE59: &str = include_str!("glcuda_sm75_wave59.ptx");

/// Wave 88's isolated N16 register-prefetch candidate. It is loaded only for
/// the explicit experiment so the retained sm_75 module and JIT stay fixed.
pub const PTX_SM75_WAVE88: &str = include_str!("glcuda_sm75_wave88.ptx");

/// Threads per block for element-wise and one-block-reduction kernels.
const BLOCK: u32 = 256;
/// Warp size — grid geometry for the one-warp-per-row GEMV.
const WARP: u32 = 32;
/// Four warp partials plus one broadcast slot. The PTX reserves 36 bytes so
/// the same reduction layout also remains valid if the block grows to eight
/// warps later.
const ATTN_ROWS_REDUCTION_BYTES: u32 = 36;
/// Wave 11 GQA7 packs seven query heads that share one KV head into a CTA.
/// Keep the diagnostic candidate in the >=24-resident-warp tier on T4.
const GQA7_MAX_SCORE_CAPACITY: u32 = 288;
/// Wave 20 keeps sixteen score rows in dynamic shared memory. The f16 hi/lo Q
/// tile is a separate 4 KiB static allocation in the sm_75 kernel, so 640
/// scores leaves the complete CTA below Turing's default 48 KiB/block limit.
const MMA4_ATTN_MAX_SCORE_CAPACITY: u32 = 640;
/// Wave 48 aliases the compensated Q image with the dynamic score tile, so a
/// short prompt must still reserve enough bytes for the 16x64 hi/lo staging.
const MMA4_REGQ_STAGE_BYTES: u32 = 4_096;

fn ceil_div(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}

/// Wave 12 may widen the output tile only when the resulting launch still
/// exposes at least one CTA per SM for the real token-slab count.
fn ntile128_covers(out_dim: u32, ntok: u32, sm_count: u32) -> bool {
    ceil_div(out_dim, 128).saturating_mul(ceil_div(ntok, 64)) >= sm_count.max(1)
}

/// Wave 27's N16 warp tile halves activation staging and barrier count per
/// output column at N128, so its experimental arm measures N128 directly
/// whenever the retained Wave 12 opt-in is set. The direct gate decides
/// whether that trade is worthwhile on narrow grids.
fn n16_threads(ntile128: bool) -> u32 {
    if ntile128 {
        256
    } else {
        128
    }
}

/// Wave 27 hybrid repair: wide N128 grids retain the M64/N16 warp entry while
/// under-covered N128 grids use paired M32 warps and N64 CTAs. Wave 28 changes
/// only the shared-to-register fragment load inside those two entries.
fn n16_uses_m32(ntile128: bool, out_dim: u32, ntok: u32, sm_count: u32) -> bool {
    ntile128 && !ntile128_covers(out_dim, ntok, sm_count)
}

/// Wave 88 deliberately covers only the pinned FFN gate/up projection. Wave
/// 123 may launch the two stacked halves at once, so both the split half and
/// the full stacked output are in scope. The exact shape guard prevents the
/// experiment from leaking into QKV or down.
fn n16_prefetch_shape(out_dim: u32, in_dim: u32, ntok: u32) -> bool {
    matches!(out_dim, 4_864 | 9_728) && in_dim == 896 && ntok == 244
}

/// Dynamic shared memory for prefill attention: one f32 score per causal row
/// plus the fixed block-reduction scratch. Returning `None` makes an invalid
/// zero/overflow capacity a launch error rather than an undersized buffer.
fn attn_rows_shared_bytes(score_capacity: u32) -> Option<u32> {
    if score_capacity == 0 {
        return None;
    }
    score_capacity
        .checked_mul(4)?
        .checked_add(ATTN_ROWS_REDUCTION_BYTES)
}

/// Dynamic shared memory for the Wave 20 16-query tile: sixteen padded f32
/// score rows. The launch value deliberately excludes the kernel's fixed
/// 4 KiB compensated-Q tile, exactly as CUDA's dynamic-smem argument requires.
fn attn_mma4_shared_bytes(score_capacity: u32) -> Option<u32> {
    if score_capacity == 0 || score_capacity > MMA4_ATTN_MAX_SCORE_CAPACITY {
        return None;
    }
    let padded = score_capacity.checked_add(3)? & !3;
    padded.checked_mul(16)?.checked_mul(4)
}

/// Dynamic shared memory for Wave 48. Production capacities already exceed
/// 4 KiB; the maximum only matters for parity's short-tail shapes.
fn attn_mma4_regq_shared_bytes(score_capacity: u32) -> Option<u32> {
    attn_mma4_shared_bytes(score_capacity).map(|bytes| bytes.max(MMA4_REGQ_STAGE_BYTES))
}

/// Seven padded score rows, 64 B of per-head reduction/broadcast state, and
/// one reusable 8x64 f32 K/V tile. This helper is deliberately shape-specific:
/// unsupported model shapes take the retained attention path.
/// Wave 15D: the same layout with a FOUR-row K/V tile, which is 1024 B of
/// staging instead of 2048.
///
/// At the pinned 244-token prompt that is 7920 B against GQA7's 8944. The T4
/// allocates shared memory on a granule, so 8944 occupies 8960 and fits seven
/// blocks per SM, while 7920 occupies 7936 and fits **eight**. Wave 15C
/// measured what one tier is worth here by accident: 418 bytes of padding cost
/// 30% on an otherwise identical kernel.
fn attn_rows_gqa7_t4_shared_bytes(score_capacity: u32) -> Option<u32> {
    if score_capacity == 0 || score_capacity > GQA7_MAX_SCORE_CAPACITY {
        return None;
    }
    let padded = score_capacity.checked_add(3)? & !3;
    padded.checked_mul(28)?.checked_add(1088)
}

fn attn_rows_gqa7_shared_bytes(score_capacity: u32) -> Option<u32> {
    if score_capacity == 0 || score_capacity > GQA7_MAX_SCORE_CAPACITY {
        return None;
    }
    let padded = score_capacity.checked_add(3)? & !3;
    padded.checked_mul(28)?.checked_add(2112)
}

/// The sm_75 tensor-core module and every entry resolved from it.
///
/// This was a tuple until Wave 17 made it seven wide, at which point
/// `let (_, _, _, _, _, _, f)` stopped being readable and started being a
/// place to put a bug. Names cost nothing here.
struct MmaModule {
    /// Owned so every `Kernel` below stays valid for as long as this does.
    _module: Module,
    /// The retained direct GEMM.
    direct: Kernel,
    /// Opt-in 256-row variant; never selected by production dispatch.
    r256: Kernel,
    /// Wave 12's cooperative B staging, which production runs.
    bstage: Kernel,
    /// Wave 16 mainloop ablation probe over `direct` (diagnostic only).
    probe: Kernel,
    /// Wave 16B mainloop ablation probe over `bstage` (diagnostic only).
    bstage_probe: Kernel,
    /// Wave 17: `bstage` with the next k-block prefetched into registers.
    bstage_pipe: Kernel,
    /// Wave 27: one warp owns two adjacent N8 output fragments.
    bstage_n16: Kernel,
    /// Wave 27 repair: paired warps split M64 while sharing one N16 fragment.
    bstage_n16_m32: Kernel,
    /// Wave 20: compensated-f16 MMA QK fused with causal softmax and AV.
    attn_mma4: Kernel,
    /// Wave 48: Wave 20 arithmetic with Q fragments resident in registers.
    attn_mma4_regq: Kernel,
    /// Wave 78: register-Q QK plus cooperative compensated-MMA AV.
    attn_mma4_regq_avmma: Kernel,
}

/// Wave 59 is deliberately isolated from the retained tensor-core module.
struct Wave59Module {
    _module: Module,
    n32_m32: Kernel,
}

/// Wave 88 is isolated so enabling it cannot mutate the retained module.
struct Wave88Module {
    _module: Module,
    n16_prefetch: Kernel,
}

/// One loaded module plus resolved handles for every kernel. Handles stay
/// valid while `_module` lives — the struct owns it for exactly that.
pub struct KernelSet {
    _module: Module,
    /// The sm_75+ tensor-core module and its three GEMM entries, present only on
    /// capable devices (and absent under `GLCUDA_NO_MMA=1`, the benchmark
    /// A/B switch). The `Option` IS the runtime kernel selection: callers
    /// ask [`KernelSet::has_mma`] and fall back to `gl_gemm_q8_0_soa`.
    /// Tuple: (module, direct 8-m-tile GEMM, 32-m-tile r256 GEMM, Wave 12
    /// cooperative-B GEMM). The r256 entry is the Phase B weight-reuse kernel
    /// (256 rows/weight-read); the bench A/B picks which design net-wins on the
    /// bandwidth-bound FFN GEMMs.
    mma: Option<MmaModule>,
    /// Opt-in Wave 59 N32 x M32 narrow-grid candidate.
    wave59: Option<Wave59Module>,
    /// Opt-in Wave 88 wide-grid N16 register-prefetch candidate.
    wave88: Option<Wave88Module>,
    /// Whether prefill should drive the r256 (256-row) GEMM instead of the
    /// 64-row one. Read once at load from `GLCUDA_R256`; see
    /// [`KernelSet::r256_enabled`].
    r256: bool,
    /// Whether the 8-m-tile kernel should cover all token slabs in one 2-D
    /// launch. Read once from `GLCUDA_GRID2D`; see
    /// [`KernelSet::grid2d_enabled`].
    grid2d: bool,
    /// Wave 11 exact adjacent-chain fusion; prefill-only and opt-in.
    fuse_q8_glue: bool,
    /// Wave 123: skip f32 scratch writes when the downstream prefill GEMM reads
    /// only the Q8 activation copy.
    q8_nostore: AtomicBool,
    /// Wave 123: compute the stacked FFN gate+up projection in one GEMM and
    /// consume its row-major [gate, up] layout directly.
    ffn_gate_up_stacked: AtomicBool,
    /// Wave 11 Qwen GQA7 K/V-reuse attention; prefill-only and opt-in.
    gqa_group: bool,
    /// Wave 12: 512-thread N128 CTA when coverage remains >= one CTA/SM.
    ntile128: bool,
    /// Wave 12: use the exact prepacked cooperative-B staging kernel.
    bstage: bool,
    /// Wave 27: reuse each A fragment across an N16 per-warp output tile.
    gemm_n16: bool,
    /// Whether narrow N16/M32 launches should use Wave 59's N32/M32 entry.
    gemm_n32: bool,
    /// Whether the pinned FFN gate/up shape should use Wave 88 prefetch.
    gemm_n16_prefetch: bool,
    /// Wave 111: defer each non-final FFN residual into the next layer's
    /// already-fused attention RMS+Q8 pass. Opt-in until production A/B.
    defer_ffn_residual: AtomicBool,
    /// Wave 15A is retained and default; this forces the row kernel back,
    /// which is what an A/B against it needs.
    rows_forced: bool,
    /// Wave 15B: how many independent QK chains the GQA7 kernel runs.
    ///
    /// 1 is the retained kernel. The count is the experiment's causal
    /// variable, so it is one dial rather than two flags, and every value
    /// shares the same eight-row K tile and the same shared-memory
    /// footprint.
    gqa7_chains: u8,
    /// Opt-in Wave 20 fused compensated-MMA attention candidate.
    mma4_attention: bool,
    /// Opt-in Wave 48 register-resident-Q schedule on the Wave 20 path.
    mma4_regq_attention: bool,
    /// Opt-in Wave 78 compensated-MMA AV on top of register-resident Q.
    mma4_regq_avmma_attention: bool,
    /// Device SM count used by the N128 coverage guard.
    sm_count: u32,
    f_add: Kernel,
    f_silu_mul: Kernel,
    f_silu_mul_quantize_q8: Kernel,
    f_silu_mul_quantize_q8_stacked_nostore: Kernel,
    f_rope: Kernel,
    f_gemv: Kernel,
    f_quantize_q8: Kernel,
    f_rms_quantize_q8_rows: Kernel,
    f_gemv_q8_0: Kernel,
    f_gemv_q8_0_soa: Kernel,
    f_gemm_q8_0_soa: Kernel,
    f_gemv_q4_k_soa: Kernel,
    f_gemv_q4_0_soa: Kernel,
    f_gemv_q6_k_soa: Kernel,
    f_gemv_q4_0: Kernel,
    f_gemv_t: Kernel,
    f_rms_norm: Kernel,
    f_softmax_scale: Kernel,
    f_attn_decode: Kernel,
    f_kv_write: Kernel,
    // Batched-over-tokens prefill variants (M2.3 Stage 1b). The single-token
    // originals above stay untouched — the decode graph is captured against
    // them; these exist so one launch covers a whole prefill chunk.
    f_rms_norm_rows: Kernel,
    f_add_bias_rows: Kernel,
    f_rope_rows: Kernel,
    f_kv_write_rows: Kernel,
    f_attn_decode_rows: Kernel,
    f_attn_decode_rows_gqa7: Kernel,
    /// Wave 15A: the same attention with four independent QK chains per warp.
    f_attn_rows_qk4: Kernel,
    /// Wave 15B: GQA7 with two independent QK chains per warp.
    f_attn_gqa7_qk2: Kernel,
    /// Wave 15B: GQA7 with four (two tile rows x two query heads).
    f_attn_gqa7_qk4: Kernel,
    /// Wave 15C: GQA7 with early exits, for the pass split.
    f_attn_gqa7_probe: Kernel,
    /// Wave 15D: GQA7 with a four-row tile, trading barriers for occupancy.
    f_attn_gqa7_t4: Kernel,
    /// Diagnostic pass-split copy of `gl_attn_decode_rows_f32` (bench-only —
    /// the engine never launches it; see [`Self::attn_rows_probe`]).
    f_attn_rows_probe: Kernel,
    /// Diagnostic pass-split copy of `gl_attn_rows_qk4_f32`, the kernel the
    /// production dispatcher actually selects (bench-only, never launched by
    /// the engine; see [`Self::attn_rows_qk4_probe`]). `gl_attn_rows_probe`
    /// splits the OLD default instead, and qk4's Pass 1 runs four independent
    /// key chains, so its split cannot be assumed to match.
    f_attn_rows_qk4_probe: Kernel,
}

fn resolve_defer_ffn_residual(
    fuse_q8_glue: bool,
    environment_enabled: bool,
    benchmark_override: Option<bool>,
) -> bool {
    fuse_q8_glue && benchmark_override.unwrap_or(environment_enabled)
}

impl KernelSet {
    /// JIT the embedded PTX and resolve every entry point. On sm_75+ the
    /// tensor-core module is loaded too (`GLCUDA_NO_MMA=1` opts out, for
    /// A/B benchmarking against the sm_70 dp4a GEMM).
    pub fn load(cuda: &Cuda) -> Result<KernelSet, GlError> {
        Self::load_with_defer_override(cuda, None)
    }

    /// Load the kernel suite with a narrow benchmark-only override for Wave
    /// 111. `None` is byte-for-byte the production environment policy.
    #[doc(hidden)]
    pub fn load_with_defer_override(
        cuda: &Cuda,
        defer_ffn_residual_override: Option<bool>,
    ) -> Result<KernelSet, GlError> {
        Self::load_with_benchmark_overrides(cuda, defer_ffn_residual_override, None, None)
    }

    /// Load the kernel suite with benchmark-only overrides for paths that must
    /// be A/B'd inside one process. `None` keeps the production environment
    /// contract for that switch.
    #[doc(hidden)]
    pub fn load_with_benchmark_overrides(
        cuda: &Cuda,
        defer_ffn_residual_override: Option<bool>,
        q8_nostore_override: Option<bool>,
        ffn_gate_up_stacked_override: Option<bool>,
    ) -> Result<KernelSet, GlError> {
        let module = cuda.load_module(PTX)?;
        let sm = (cuda.info.sm_major, cuda.info.sm_minor);
        let mma = if sm >= (7, 5) && std::env::var_os("GLCUDA_NO_MMA").is_none() {
            let m75 = cuda.load_module(PTX_SM75)?;
            let f = m75.get_function("gl_gemm_mma_q8")?;
            let f256 = m75.get_function("gl_gemm_mma_q8_r256")?;
            let f_bstage = m75.get_function("gl_gemm_mma_q8_bstage")?;
            let f_probe = m75.get_function("gl_gemm_mma_q8_probe")?;
            let f_bsprobe = m75.get_function("gl_gemm_mma_q8_bstage_probe")?;
            let f_bspipe = m75.get_function("gl_gemm_mma_q8_bstage_pipe")?;
            let f_bsn16 = m75.get_function("gl_gemm_mma_q8_bstage_n16")?;
            let f_bsn16_m32 = m75.get_function("gl_gemm_mma_q8_bstage_n16_m32")?;
            let f_attn_mma4 = m75.get_function("gl_attn_mma4_fused_f32")?;
            let f_attn_mma4_regq = m75.get_function("gl_attn_mma4_regq_fused_f32")?;
            let f_attn_mma4_regq_avmma = m75.get_function("gl_attn_mma4_regq_avmma_fused_f32")?;
            eprintln!(
                "[glcuda] tensor-core MMA GEMM enabled (sm_{}{})",
                cuda.info.sm_major, cuda.info.sm_minor
            );
            Some(MmaModule {
                _module: m75,
                direct: f,
                r256: f256,
                bstage: f_bstage,
                probe: f_probe,
                bstage_probe: f_bsprobe,
                bstage_pipe: f_bspipe,
                bstage_n16: f_bsn16,
                bstage_n16_m32: f_bsn16_m32,
                attn_mma4: f_attn_mma4,
                attn_mma4_regq: f_attn_mma4_regq,
                attn_mma4_regq_avmma: f_attn_mma4_regq_avmma,
            })
        } else {
            if sm >= (7, 5) {
                eprintln!("[glcuda] GLCUDA_NO_MMA set: prefill GEMM on the sm_70 dp4a path");
            }
            None
        };
        // Opt-in, and off by default on purpose. r256 is measured correct
        // (parity green on a T4, max_abs_diff 0.00e0 against gemm_mma_q8 at
        // real shapes) and measured 31% faster at 512-row chunks -- but the
        // last attempt to wire it into prefill crashed the engine with
        // CUDA_ERROR_MISALIGNED_ADDRESS, and a kernel-level win does not have
        // to survive the trip into production: the multi-stream prefill
        // experiment was 4x the blocks in flight for -0.6%.
        //
        // So this ships as an A/B switch until a production run says to flip
        // the default.
        let r256 = mma.is_some() && std::env::var_os("GLCUDA_R256").is_some();
        if r256 {
            eprintln!("[glcuda] r256 prefill GEMM enabled (256-row weight reuse)");
        }
        // Wave 3 candidate. This keeps the arithmetic kernel unchanged and
        // replaces the host's serial 64-row launch loop with grid.y. It stays
        // opt-in until the production glbench gate reproduces on T4.
        let grid2d = mma.is_some() && std::env::var_os("GLCUDA_GRID2D").is_some();
        if grid2d {
            eprintln!("[glcuda] 2-D token-grid prefill GEMM enabled");
        }
        let fuse_q8_glue = std::env::var_os("GLCUDA_FUSE_Q8_GLUE").is_some();
        let q8_nostore = fuse_q8_glue
            && q8_nostore_override
                .unwrap_or_else(|| std::env::var_os("GLCUDA_Q8_NOSTORE").is_some());
        let ffn_gate_up_stacked = fuse_q8_glue
            && ffn_gate_up_stacked_override
                .unwrap_or_else(|| std::env::var_os("GLCUDA_FFN_GATE_UP_STACKED").is_some());
        let gqa_group = std::env::var_os("GLCUDA_GQA_GROUP").is_some();
        let ntile128 = mma.is_some() && std::env::var_os("GLCUDA_NTILE128").is_some();
        let bstage = mma.is_some() && std::env::var_os("GLCUDA_BSTAGE").is_some();
        let gemm_n16 =
            mma.is_some() && grid2d && bstage && std::env::var_os("GLCUDA_GEMM_N16").is_some();
        let gemm_n32 = gemm_n16 && std::env::var_os("GLCUDA_GEMM_N32").is_some();
        let wave59 = if gemm_n32 {
            let module = cuda.load_module(PTX_SM75_WAVE59)?;
            let n32_m32 = module.get_function("gl_gemm_mma_q8_bstage_n32_m32")?;
            eprintln!("[glcuda] Wave 59 N32/M32 narrow-grid GEMM enabled");
            Some(Wave59Module {
                _module: module,
                n32_m32,
            })
        } else {
            None
        };
        let gemm_n16_prefetch = gemm_n16 && std::env::var_os("GLCUDA_GEMM_N16_PREFETCH").is_some();
        let wave88 = if gemm_n16_prefetch {
            let module = cuda.load_module(PTX_SM75_WAVE88)?;
            let n16_prefetch = module.get_function("gl_gemm_mma_q8_bstage_n16_prefetch")?;
            eprintln!("[glcuda] Wave 88 N16 register-prefetch GEMM enabled");
            Some(Wave88Module {
                _module: module,
                n16_prefetch,
            })
        } else {
            None
        };
        let defer_ffn_residual = resolve_defer_ffn_residual(
            fuse_q8_glue,
            std::env::var_os("GLCUDA_DEFER_FFN_RESIDUAL").is_some(),
            defer_ffn_residual_override,
        );
        let rows_forced = std::env::var_os("GLCUDA_ATTN_ROWS").is_some();
        let mma4_attention = mma.is_some() && std::env::var_os("GLCUDA_ATTN_MMA4").is_some();
        let mma4_regq_attention =
            mma4_attention && std::env::var_os("GLCUDA_ATTN_MMA4_REGQ").is_some();
        let mma4_regq_avmma_attention =
            mma4_regq_attention && std::env::var_os("GLCUDA_ATTN_MMA4_AV").is_some();
        let gqa7_chains = match std::env::var("GLCUDA_GQA7_CHAINS").as_deref() {
            Ok("2") => 2,
            Ok("4") => 4,
            _ => 1,
        };
        eprintln!(
            "[glcuda-contract] {{\"exact_fusion\":{},\"q8_nostore\":{},\"ffn_gate_up_stacked\":{},\"defer_ffn_residual\":{},\"gqa_group\":{},\"grid2d\":{},\"r256\":{},\"ntile128\":{},\"bstage\":{},\"gemm_n16\":{},\"gemm_n32\":{},\"gemm_n16_prefetch\":{},\"attn_rows_forced\":{},\"gqa7_chains\":{},\"attn_mma4\":{},\"attn_mma4_regq\":{},\"attn_mma4_av\":{}}}",
            fuse_q8_glue, q8_nostore, ffn_gate_up_stacked, defer_ffn_residual, gqa_group, grid2d, r256, ntile128, bstage, gemm_n16, gemm_n32, gemm_n16_prefetch, rows_forced, gqa7_chains, mma4_attention, mma4_regq_attention, mma4_regq_avmma_attention
        );
        eprintln!("[glcuda] dynamic-shared prefill attention enabled");
        Ok(KernelSet {
            mma,
            wave59,
            wave88,
            r256,
            grid2d,
            fuse_q8_glue,
            q8_nostore: AtomicBool::new(q8_nostore),
            ffn_gate_up_stacked: AtomicBool::new(ffn_gate_up_stacked),
            gqa_group,
            ntile128,
            bstage,
            gemm_n16,
            gemm_n32,
            gemm_n16_prefetch,
            defer_ffn_residual: AtomicBool::new(defer_ffn_residual),
            rows_forced,
            gqa7_chains,
            mma4_attention,
            mma4_regq_attention,
            mma4_regq_avmma_attention,
            sm_count: cuda.info.sm_count.max(1) as u32,
            f_add: module.get_function("gl_add_f32")?,
            f_silu_mul: module.get_function("gl_silu_mul_f32")?,
            f_silu_mul_quantize_q8: module.get_function("gl_silu_mul_quantize_q8")?,
            f_silu_mul_quantize_q8_stacked_nostore: module
                .get_function("gl_silu_mul_quantize_q8_stacked_nostore")?,
            f_rope: module.get_function("gl_rope_f32")?,
            f_gemv: module.get_function("gl_gemv_f32")?,
            f_quantize_q8: module.get_function("gl_quantize_q8")?,
            f_rms_quantize_q8_rows: module.get_function("gl_rms_quantize_q8_rows")?,
            f_gemv_q8_0: module.get_function("gl_gemv_q8_0")?,
            f_gemv_q8_0_soa: module.get_function("gl_gemv_q8_0_soa")?,
            f_gemm_q8_0_soa: module.get_function("gl_gemm_q8_0_soa")?,
            f_gemv_q4_k_soa: module.get_function("gl_gemv_q4_k_soa")?,
            f_gemv_q4_0_soa: module.get_function("gl_gemv_q4_0_soa")?,
            f_gemv_q6_k_soa: module.get_function("gl_gemv_q6_k_soa")?,
            f_gemv_q4_0: module.get_function("gl_gemv_q4_0")?,
            f_gemv_t: module.get_function("gl_gemv_t_f32")?,
            f_rms_norm: module.get_function("gl_rms_norm_f32")?,
            f_softmax_scale: module.get_function("gl_softmax_scale_f32")?,
            f_attn_decode: module.get_function("gl_attn_decode_f32")?,
            f_kv_write: module.get_function("gl_kv_write")?,
            f_rms_norm_rows: module.get_function("gl_rms_norm_rows_f32")?,
            f_add_bias_rows: module.get_function("gl_add_bias_rows_f32")?,
            f_rope_rows: module.get_function("gl_rope_rows_f32")?,
            f_kv_write_rows: module.get_function("gl_kv_write_rows")?,
            f_attn_decode_rows: module.get_function("gl_attn_decode_rows_f32")?,
            f_attn_decode_rows_gqa7: module.get_function("gl_attn_decode_rows_gqa7_f32")?,
            f_attn_rows_qk4: module.get_function("gl_attn_rows_qk4_f32")?,
            f_attn_gqa7_qk2: module.get_function("gl_attn_gqa7_qk2_f32")?,
            f_attn_gqa7_qk4: module.get_function("gl_attn_gqa7_qk4_f32")?,
            f_attn_gqa7_probe: module.get_function("gl_attn_gqa7_probe_f32")?,
            f_attn_gqa7_t4: module.get_function("gl_attn_gqa7_t4_f32")?,
            f_attn_rows_probe: module.get_function("gl_attn_rows_probe")?,
            f_attn_rows_qk4_probe: module.get_function("gl_attn_rows_qk4_probe")?,
            _module: module,
        })
    }

    /// `y[i] += x[i]` over `n` elements (residual add).
    pub fn add(&self, cuda: &Cuda, y: CUdeviceptr, x: CUdeviceptr, n: u32) -> Result<(), GlError> {
        let (mut y, mut x, mut n_) = (y, x, n);
        let mut params = [
            &mut y as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut n_ as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_add,
            (ceil_div(n, BLOCK), 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Fused SwiGLU gating: `gate[i] = silu(gate[i]) * up[i]`.
    pub fn silu_mul(
        &self,
        cuda: &Cuda,
        gate: CUdeviceptr,
        up: CUdeviceptr,
        n: u32,
    ) -> Result<(), GlError> {
        let (mut gate, mut up, mut n_) = (gate, up, n);
        let mut params = [
            &mut gate as *mut _ as *mut c_void,
            &mut up as *mut _ as *mut c_void,
            &mut n_ as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_silu_mul,
            (ceil_div(n, BLOCK), 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Rotary embedding over all heads of `x` (`[n_heads * head_dim]`).
    /// `cos`/`sin` are the FULL device tables covering every position
    /// (`[max_ctx * head_dim/2]`), computed on the host (host owns
    /// transcendental precision — the RoPE ε is 1e-7). `pos` is a device
    /// pointer to the current position (a `u32` in device memory); the
    /// kernel reads it and indexes row `pos`. Passing `pos` by device
    /// pointer rather than value keeps the launch arguments token-invariant
    /// so the per-token graph can be captured once (M2.2).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        cos: CUdeviceptr,
        sin: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        neox: bool,
        pos: CUdeviceptr,
    ) -> Result<(), GlError> {
        let (mut x, mut cos, mut sin) = (x, cos, sin);
        let (mut h, mut hd, mut nx, mut p) = (n_heads, head_dim, neox as u32, pos);
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nx as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
        ];
        let pairs = n_heads * (head_dim / 2);
        cuda.launch(
            self.f_rope,
            (ceil_div(pairs, BLOCK), 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Write this token's K (or V) rows for all KV heads into the cache at
    /// device-side position `pos` (M2.2 graph-static replacement for the
    /// per-head `cuMemcpyDtoD`). `dst_base` is the layer's cache region for
    /// head 0; `src` is the contiguous `[n_kv * head_dim]` workspace rows.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_write(
        &self,
        cuda: &Cuda,
        dst_base: CUdeviceptr,
        src: CUdeviceptr,
        pos: CUdeviceptr,
        head_dim: u32,
        n_kv: u32,
        head_stride: u32,
    ) -> Result<(), GlError> {
        let (mut d, mut s, mut p) = (dst_base, src, pos);
        let (mut hd, mut nk, mut hs) = (head_dim, n_kv, head_stride);
        let mut params = [
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
        ];
        let n = n_kv * head_dim;
        cuda.launch(
            self.f_kv_write,
            (ceil_div(n, BLOCK), 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Batched RMSNorm over `rows` contiguous rows of `dim` (M2.3 prefill):
    /// one launch replaces the per-token loop. Also serves the per-head
    /// q/k-norms — a `[n, heads*head_dim]` block is `n*heads` contiguous
    /// rows of `head_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_rows(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        w: CUdeviceptr,
        out: CUdeviceptr,
        dim: u32,
        eps: f32,
        rows: u32,
    ) -> Result<(), GlError> {
        let (mut x, mut w, mut out) = (x, w, out);
        let (mut d, mut e) = (dim, eps);
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut e as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_rms_norm_rows,
            (rows, 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Whether Wave 11's exact prefill glue fusion was selected at module load.
    pub fn fuse_q8_glue_enabled(&self) -> bool {
        self.fuse_q8_glue
    }

    /// Whether Wave 123 may skip f32 scratch writes for Q8-only prefill consumers.
    pub fn q8_nostore_enabled(&self) -> bool {
        self.q8_nostore.load(Ordering::Relaxed)
    }

    /// Whether Wave 123 may consume the stacked gate/up prefill GEMM layout.
    pub fn ffn_gate_up_stacked_enabled(&self) -> bool {
        self.ffn_gate_up_stacked.load(Ordering::Relaxed)
    }

    /// RMSNorm followed by the byte-compatible Q8 activation quantizer, with
    /// an optional in-place residual add before the unchanged RMS reduction.
    /// This is prefill-only: decode graph geometry and entry points stay fixed.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_quantize_q8_rows(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        residual: Option<CUdeviceptr>,
        w: CUdeviceptr,
        out: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        dim: u32,
        eps: f32,
        rows: u32,
    ) -> Result<(), GlError> {
        self.rms_quantize_q8_rows_impl(cuda, x, residual, w, out, qs, scales, dim, eps, rows, true)
    }

    /// Wave 123 no-store variant of [`Self::rms_quantize_q8_rows`]. Use only
    /// when the following prefill GEMM consumes `qs`/`scales` and never the f32
    /// `out` scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_quantize_q8_rows_nostore(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        residual: Option<CUdeviceptr>,
        w: CUdeviceptr,
        out: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        dim: u32,
        eps: f32,
        rows: u32,
    ) -> Result<(), GlError> {
        self.rms_quantize_q8_rows_impl(cuda, x, residual, w, out, qs, scales, dim, eps, rows, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_quantize_q8_rows_impl(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        residual: Option<CUdeviceptr>,
        w: CUdeviceptr,
        out: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        dim: u32,
        eps: f32,
        rows: u32,
        store_out: bool,
    ) -> Result<(), GlError> {
        debug_assert_eq!(dim % 32, 0, "fused RMS+Q8 dim must be a multiple of 32");
        let (mut x, mut residual_ptr, mut w, mut out, mut qs, mut scales) =
            (x, residual.unwrap_or(x), w, out, qs, scales);
        let (mut d, mut e, mut add, mut store) = (
            dim,
            eps,
            u32::from(residual.is_some()),
            u32::from(store_out),
        );
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut residual_ptr as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut scales as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut e as *mut _ as *mut c_void,
            &mut add as *mut _ as *mut c_void,
            &mut store as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_rms_quantize_q8_rows,
            (rows, 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Exact SwiGLU followed by the byte-compatible per-K32 Q8 quantizer.
    pub fn silu_mul_quantize_q8(
        &self,
        cuda: &Cuda,
        gate: CUdeviceptr,
        up: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        n: u32,
    ) -> Result<(), GlError> {
        self.silu_mul_quantize_q8_impl(cuda, gate, up, qs, scales, n, true)
    }

    /// Wave 123 no-store variant of [`Self::silu_mul_quantize_q8`]. Use only
    /// when the following prefill down projection consumes the Q8 activation.
    pub fn silu_mul_quantize_q8_nostore(
        &self,
        cuda: &Cuda,
        gate: CUdeviceptr,
        up: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        n: u32,
    ) -> Result<(), GlError> {
        self.silu_mul_quantize_q8_impl(cuda, gate, up, qs, scales, n, false)
    }

    fn silu_mul_quantize_q8_impl(
        &self,
        cuda: &Cuda,
        gate: CUdeviceptr,
        up: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        n: u32,
        store_gate: bool,
    ) -> Result<(), GlError> {
        debug_assert_eq!(n % 32, 0, "fused SwiGLU+Q8 n must be a multiple of 32");
        let (mut gate, mut up, mut qs, mut scales, mut n_, mut store) =
            (gate, up, qs, scales, n, u32::from(store_gate));
        let mut params = [
            &mut gate as *mut _ as *mut c_void,
            &mut up as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut scales as *mut _ as *mut c_void,
            &mut n_ as *mut _ as *mut c_void,
            &mut store as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_silu_mul_quantize_q8,
            (ceil_div(n, BLOCK), 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Wave 123 stacked gate/up SwiGLU followed by the byte-compatible Q8
    /// activation quantizer. The input is `[ntok, 2*hidden]`, laid out as
    /// gate columns then up columns for each token row. It does not write a f32
    /// product scratch; the following down GEMM must consume Q8.
    pub fn silu_mul_quantize_q8_stacked_nostore(
        &self,
        cuda: &Cuda,
        gate_up: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        hidden: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(
            hidden % 32,
            0,
            "stacked fused SwiGLU+Q8 hidden must be a multiple of 32"
        );
        let (mut gate_up, mut qs, mut scales, mut h, mut n) = (gate_up, qs, scales, hidden, ntok);
        let hidden_blocks = hidden / 32;
        let mut params = [
            &mut gate_up as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut scales as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_silu_mul_quantize_q8_stacked_nostore,
            (ceil_div(hidden_blocks, BLOCK / WARP), ntok, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Broadcast bias add over a `[rows, dim]` activation block in one
    /// launch: `y[i] += b[i % dim]` for `i < total` (M2.3 prefill).
    pub fn add_bias_rows(
        &self,
        cuda: &Cuda,
        y: CUdeviceptr,
        b: CUdeviceptr,
        dim: u32,
        total: u32,
        row_stride: u32,
    ) -> Result<(), GlError> {
        let (mut y, mut b) = (y, b);
        let (mut d, mut t, mut rs) = (dim, total, row_stride);
        let mut params = [
            &mut y as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut t as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_add_bias_rows,
            (ceil_div(total, BLOCK), 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Batched RoPE over `ntok` token rows in one launch (M2.3 prefill).
    /// Row `t` rotates `x + t*heads*head_dim` at position `pos_seq[t]` —
    /// pass `pos_seq` already offset to the chunk's base position.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_rows(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        cos: CUdeviceptr,
        sin: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        neox: bool,
        pos_seq: CUdeviceptr,
        ntok: u32,
        row_stride: u32,
    ) -> Result<(), GlError> {
        let (mut x, mut cos, mut sin) = (x, cos, sin);
        let (mut h, mut hd, mut nx, mut p) = (n_heads, head_dim, neox as u32, pos_seq);
        let mut rs = row_stride;
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nx as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void,
        ];
        let pairs = n_heads * (head_dim / 2);
        cuda.launch(
            self.f_rope_rows,
            (ceil_div(pairs, BLOCK), ntok, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Batched KV write over `ntok` token rows in one launch (M2.3
    /// prefill): row `t` (at `src + t*n_kv*head_dim`) lands at cache
    /// position `pos_seq[t]`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_write_rows(
        &self,
        cuda: &Cuda,
        dst_base: CUdeviceptr,
        src: CUdeviceptr,
        pos_seq: CUdeviceptr,
        head_dim: u32,
        n_kv: u32,
        head_stride: u32,
        ntok: u32,
        src_stride: u32,
    ) -> Result<(), GlError> {
        let (mut d, mut s, mut p) = (dst_base, src, pos_seq);
        let (mut hd, mut nk, mut hs) = (head_dim, n_kv, head_stride);
        let mut ss = src_stride;
        let mut params = [
            &mut d as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut ss as *mut _ as *mut c_void,
        ];
        let n = n_kv * head_dim;
        cuda.launch(
            self.f_kv_write_rows,
            (ceil_div(n, BLOCK), ntok, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Whether Wave 11's grouped-GQA attention was selected at module load.
    ///
    /// The attention module reads this to choose a path. This type deliberately
    /// no longer chooses one itself: a dispatcher here and a dispatcher there
    /// is how the two drift.
    pub fn gqa_group_enabled(&self) -> bool {
        self.gqa_group
    }

    /// Can the GQA7 kernel hold `score_capacity` scores in shared memory?
    ///
    /// Shape support is a launch-geometry fact, so it stays with the launch
    /// geometry; the attention module asks rather than re-deriving the bound.
    pub fn gqa7_capacity_supported(&self, score_capacity: u32) -> bool {
        attn_rows_gqa7_shared_bytes(score_capacity).is_some()
    }

    /// Whether the opt-in fused compensated-MMA attention path is available
    /// for this process. Shape and shared-memory capacity are checked by the
    /// attention dispatcher separately.
    pub fn mma4_attention_enabled(&self) -> bool {
        self.mma4_attention
    }

    /// Whether Wave 48's register-resident-Q schedule was selected on top of
    /// the compensated MMA4 attention path.
    pub fn mma4_regq_attention_enabled(&self) -> bool {
        self.mma4_regq_attention
    }

    /// Whether Wave 78's compensated-MMA AV candidate was selected on top of
    /// the register-resident-Q path.
    pub fn mma4_regq_avmma_attention_enabled(&self) -> bool {
        self.mma4_regq_avmma_attention
    }

    /// Whether one 16-query score tile fits the Wave 20 launch contract.
    pub fn mma4_attention_capacity_supported(&self, score_capacity: u32) -> bool {
        attn_mma4_shared_bytes(score_capacity).is_some()
    }

    /// Whether Wave 48 can provide both its aliased Q stage and score tile.
    pub fn mma4_regq_attention_capacity_supported(&self, score_capacity: u32) -> bool {
        attn_mma4_regq_shared_bytes(score_capacity).is_some()
    }

    /// Batched causal decode-attention over `ntok` token rows in one launch
    /// (M2.3 prefill): block (h, t) runs head h of row t with
    /// `cached_len = pos_seq[t] + 1`, so each row attends to exactly its own
    /// prefix (rows after it exist in the cache but are never read). Requires
    /// the chunk's KV rows to be written first (kv_write_rows on the same
    /// stream). `score_capacity` is the largest causal length in this launch
    /// (`chunk_base + ntok`); it sizes dynamic shared memory to the real
    /// prompt prefix instead of reserving 4096 scores for every CTA.
    ///
    /// This is the retained Wave 4 path. Callers reach it through
    /// `crate::attention::prefill`, which owns the choice between this and
    /// [`Self::attn_decode_rows_gqa7`].
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_rows_legacy(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        let shared_bytes = attn_rows_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "invalid prefill attention score capacity {score_capacity}"
            ))
        })?;
        cuda.launch(
            self.f_attn_decode_rows,
            (n_heads, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Wave 15A: the retained row attention with four independent QK chains.
    ///
    /// Identical contract, identical launch geometry, identical shared memory:
    /// only Pass 1 differs, and each score is reduced in exactly the retained
    /// order, so the output is bit-identical rather than merely close.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_rows_qk4(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        let shared_bytes = attn_rows_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "invalid prefill attention score capacity {score_capacity}"
            ))
        })?;
        cuda.launch(
            self.f_attn_rows_qk4,
            (n_heads, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Wave 20 production candidate. One 128-thread CTA owns one query head
    /// and a 16-row query tile: warp 0 computes compensated-f16 QK with four
    /// sm_75 MMA products, then all four warps finish causal softmax and AV in
    /// shared memory. Scores never make a global-memory round trip.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_mma4_fused(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        if head_dim != 64 {
            return Err(GlError::Engine(format!(
                "Wave 20 MMA4 attention requires head_dim=64; got {head_dim}"
            )));
        }
        let mma = self.mma.as_ref().ok_or_else(|| {
            GlError::Engine("Wave 20 MMA4 attention requires an sm_75 device".into())
        })?;
        let shared_bytes = attn_mma4_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "Wave 20 MMA4 attention score capacity must be in 1..={MMA4_ATTN_MAX_SCORE_CAPACITY}; got {score_capacity}"
            ))
        })?;
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            mma.attn_mma4,
            (ceil_div(ntok, 16), n_heads, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Wave 48 candidate. Arithmetic, grid, score layout, softmax, and AV are
    /// identical to Wave 20; only Q-fragment lifetime and shared-memory
    /// residency differ.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_mma4_regq_fused(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        if head_dim != 64 {
            return Err(GlError::Engine(format!(
                "Wave 48 register-Q MMA4 attention requires head_dim=64; got {head_dim}"
            )));
        }
        let mma = self.mma.as_ref().ok_or_else(|| {
            GlError::Engine("Wave 48 register-Q MMA4 attention requires an sm_75 device".into())
        })?;
        let shared_bytes = attn_mma4_regq_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "Wave 48 register-Q MMA4 attention score capacity must be in 1..={MMA4_ATTN_MAX_SCORE_CAPACITY}; got {score_capacity}"
            ))
        })?;
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            mma.attn_mma4_regq,
            (ceil_div(ntok, 16), n_heads, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Wave 78 candidate. QK, causal masking, and softmax retain Wave 48's
    /// arithmetic; only the final normalized P@V is computed cooperatively by
    /// four compensated-f16 MMA warps over the retained row-major V cache.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_mma4_regq_avmma_fused(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        if head_dim != 64 {
            return Err(GlError::Engine(format!(
                "Wave 78 MMA AV attention requires head_dim=64; got {head_dim}"
            )));
        }
        let mma = self.mma.as_ref().ok_or_else(|| {
            GlError::Engine("Wave 78 MMA AV attention requires an sm_75 device".into())
        })?;
        let shared_bytes = attn_mma4_regq_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "Wave 78 MMA AV attention score capacity must be in 1..={MMA4_ATTN_MAX_SCORE_CAPACITY}; got {score_capacity}"
            ))
        })?;
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            mma.attn_mma4_regq_avmma,
            (ceil_div(ntok, 16), n_heads, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Explicit Wave 11 GQA7 path. The caller must provide the supported
    /// 7:1, head-dim-64 shape; the production dispatcher checks that contract.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_rows_gqa7(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        if !n_heads.is_multiple_of(7) || heads_per_kv != 7 || head_dim != 64 {
            return Err(GlError::Engine(format!(
                "GQA7 attention requires n_heads%7=0, heads_per_kv=7, head_dim=64; got {n_heads}/{heads_per_kv}/{head_dim}"
            )));
        }
        let shared_bytes = attn_rows_gqa7_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "GQA7 attention score capacity must be in 1..={GQA7_MAX_SCORE_CAPACITY}; got {score_capacity}"
            ))
        })?;
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_attn_decode_rows_gqa7,
            (n_heads / 7, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Wave 15B: GQA7 with `chains` independent QK chains per warp.
    ///
    /// Identical launch geometry, identical shared memory and identical
    /// per-score arithmetic to [`Self::attn_decode_rows_gqa7`] - each chain
    /// reduces in exactly the retained order - so the output is
    /// bit-identical and only the schedule differs. That is deliberate:
    /// the factorial exists to isolate chain count from every other
    /// resource, so nothing else is allowed to move.
    ///
    /// `chains` of 1 selects the retained kernel, so one call site can
    /// sweep the whole factorial. `smem_pad` is diagnostic-only padding
    /// added to the dynamic shared request: the kernel never reads it,
    /// and it exists so an audit can lower resident blocks per SM
    /// without touching a line of the kernel - the only way to vary
    /// occupancy and nothing else. Pass 0 everywhere but an audit.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_gqa7_chained(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
        chains: u8,
        smem_pad: u32,
    ) -> Result<(), GlError> {
        if !n_heads.is_multiple_of(7) || heads_per_kv != 7 || head_dim != 64 {
            return Err(GlError::Engine(format!(
                "GQA7 attention requires n_heads%7=0, heads_per_kv=7, head_dim=64; got {n_heads}/{heads_per_kv}/{head_dim}"
            )));
        }
        let shared_bytes = attn_rows_gqa7_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "GQA7 attention score capacity must be in 1..={GQA7_MAX_SCORE_CAPACITY}; got {score_capacity}"
            ))
        })?;
        let shared_bytes = shared_bytes + smem_pad;
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            match chains {
                1 => self.f_attn_decode_rows_gqa7,
                2 => self.f_attn_gqa7_qk2,
                _ => self.f_attn_gqa7_qk4,
            },
            (n_heads / 7, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Wave 15D: GQA7 with a four-row K/V tile.
    ///
    /// Identical arithmetic and identical launch geometry; the only difference
    /// is that the staged tile is half as tall, so the kernel asks for 1024 B
    /// less shared memory and twice as many tiles. Warp `w` still owns the same
    /// rows in the same order, so the output is bit-identical to
    /// [`Self::attn_decode_rows_gqa7`] rather than merely close.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_gqa7_t4(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        if !n_heads.is_multiple_of(7) || heads_per_kv != 7 || head_dim != 64 {
            return Err(GlError::Engine(format!(
                "GQA7 attention requires n_heads%7=0, heads_per_kv=7, head_dim=64; got {n_heads}/{heads_per_kv}/{head_dim}"
            )));
        }
        let shared_bytes = attn_rows_gqa7_t4_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "GQA7 attention score capacity must be in 1..={GQA7_MAX_SCORE_CAPACITY}; got {score_capacity}"
            ))
        })?;
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut qrs) = (score_capacity, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_attn_gqa7_t4,
            (n_heads / 7, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// The attention entries a diagnostic can ask the driver about, with the
    /// dynamic shared memory each one actually launches with.
    ///
    /// Wave 15C computed occupancy from bytes and mislabelled every point on
    /// its sweep. This exists so the next one asks
    /// [`Cuda::max_active_blocks_per_sm`] instead.
    pub fn attention_entries(&self, score_capacity: u32) -> Vec<(&'static str, Kernel, u32)> {
        let rows = attn_rows_shared_bytes(score_capacity).unwrap_or(0);
        let gqa7 = attn_rows_gqa7_shared_bytes(score_capacity).unwrap_or(0);
        let t4 = attn_rows_gqa7_t4_shared_bytes(score_capacity).unwrap_or(0);
        let mut entries = vec![
            ("rows", self.f_attn_decode_rows, rows),
            ("rows_qk4", self.f_attn_rows_qk4, rows),
            ("gqa7", self.f_attn_decode_rows_gqa7, gqa7),
            ("gqa7_qk2", self.f_attn_gqa7_qk2, gqa7),
            ("gqa7_qk4", self.f_attn_gqa7_qk4, gqa7),
            ("gqa7_t4", self.f_attn_gqa7_t4, t4),
        ];
        if let (Some(mma), Some(shared)) =
            (self.mma.as_ref(), attn_mma4_shared_bytes(score_capacity))
        {
            entries.push(("mma4_fused", mma.attn_mma4, shared));
        }
        if let (Some(mma), Some(shared)) = (
            self.mma.as_ref(),
            attn_mma4_regq_shared_bytes(score_capacity),
        ) {
            entries.push(("mma4_regq", mma.attn_mma4_regq, shared));
            entries.push(("mma4_regq_avmma", mma.attn_mma4_regq_avmma, shared));
        }
        entries
    }

    /// Wave 15C: pass-split launch of the GQA7 kernel (diagnostic only).
    ///
    /// The 71% QK share every Wave 15 ratio rests on was measured on the ROW
    /// kernel. GQA7 reads K out of shared memory and runs seven heads per CTA,
    /// so its split has never actually been measured, and if it differs then
    /// those derived "QK itself" numbers are wrong. `stop` selects how far the
    /// kernel runs: 1 = the tile loop and both `bar.sync`s with no score math
    /// (the floor neither Wave 15 lever touches), 2 = plus QK, 3 = plus
    /// softmax, 0 = the whole kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_gqa7_probe(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        stop: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        let shared_bytes = attn_rows_gqa7_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "GQA7 attention score capacity must be in 1..={GQA7_MAX_SCORE_CAPACITY}; got {score_capacity}"
            ))
        })?;
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let (mut cap, mut st, mut qrs) = (score_capacity, stop, q_row_stride);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_attn_gqa7_probe,
            (n_heads / 7, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Diagnostic pass-split launch of the prefill attention kernel (bench
    /// `[attn]` section only — never on the inference path). `stop` selects
    /// how much of the kernel runs: 0 = full (identical work to
    /// [`Self::attn_decode_rows`]), 1 = return after Pass 1 (QK scores),
    /// 2 = return after Pass 2 (softmax). The three passes are separated by
    /// `bar.sync` inside one launch, so this early-exit copy is the only way
    /// to attribute time to them without an external profiler.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_rows_probe(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        stop: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs) = (head_dim, pos_seq, heads_per_kv, head_stride);
        let (mut sc, mut cap, mut st) = (scale, score_capacity, stop);
        let mut qrs = q_row_stride;
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        let shared_bytes = attn_rows_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!(
                "invalid attention probe score capacity {score_capacity}"
            ))
        })?;
        cuda.launch(
            self.f_attn_rows_probe,
            (n_heads, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Pass-split launch of the PRODUCTION prefill attention kernel.
    ///
    /// Same grid, block and shared bytes as [`Self::attn_rows_qk4`], so the
    /// only difference from the shipped path is where it stops: 0 = full,
    /// 1 = return after Pass 1 (QK scores), 2 = return after Pass 2 (softmax).
    /// Thread 0 publishes a value derived from the completed pass before each
    /// early exit, so the work cannot be eliminated as dead.
    ///
    /// Diagnostic only. [`Self::attn_rows_probe`] splits `gl_attn_decode_rows`,
    /// which stopped being the default when `select()` moved to Qk4.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_rows_qk4_probe(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        pos_seq: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
        ntok: u32,
        score_capacity: u32,
        stop: u32,
        q_row_stride: u32,
    ) -> Result<(), GlError> {
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs) = (head_dim, pos_seq, heads_per_kv, head_stride);
        let (mut sc, mut cap, mut st) = (scale, score_capacity, stop);
        let mut qrs = q_row_stride;
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut cap as *mut _ as *mut c_void,
            &mut st as *mut _ as *mut c_void,
            &mut qrs as *mut _ as *mut c_void,
        ];
        let shared_bytes = attn_rows_shared_bytes(score_capacity).ok_or_else(|| {
            GlError::Engine(format!("invalid qk4 probe score capacity {score_capacity}"))
        })?;
        cuda.launch(
            self.f_attn_rows_qk4_probe,
            (n_heads, ntok, 1),
            (128, 1, 1),
            shared_bytes,
            &mut params,
        )
    }

    /// Decode GEMV: `y = W @ x`, `W` row-major `[out_dim, in_dim]`.
    /// One warp per output row, warp-shuffle reduction, FP32 accumulation.
    pub fn gemv(
        &self,
        cuda: &Cuda,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        let (mut w, mut x, mut y) = (w, x, y);
        let (mut o, mut i) = (out_dim, in_dim);
        let mut params = [
            &mut w as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
        ];
        cuda.launch(self.f_gemv, (out_dim, 1, 1), (WARP, 1, 1), 0, &mut params)
    }

    /// `y = x * w^T` for Q8_0 weights (row-major). `w` is `[out_dim, in_dim]`.
    /// `x` must be pre-quantized using `quantize_q8`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8_0(
        &self,
        cuda: &Cuda,
        w: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 32, 0, "Q8_0 rows are whole blocks");
        debug_assert_eq!(
            out_dim % 4,
            0,
            "Q8_0 out_dim must be multiple of 4 for Thread Coarsening"
        );
        let (mut w, mut x_qs, mut x_scales, mut y) = (w, x_qs, x_scales, y);
        let (mut o, mut i) = (out_dim, in_dim);
        let mut params = [
            &mut w as *mut _ as *mut c_void,
            &mut x_qs as *mut _ as *mut c_void,
            &mut x_scales as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_gemv_q8_0,
            (ceil_div(out_dim, 16), 1, 1),
            (128, 1, 1),
            0,
            &mut params,
        )
    }

    /// `y = W @ x` for Q8_0 weights in Structure-of-Arrays layout: `w_qs`
    /// contiguous int8 `[out_dim, in_dim]`, `w_scales` contiguous f16
    /// `[out_dim, in_dim/32]`. One warp per row (256 threads = 8 rows/block)
    /// reads 128 contiguous qs bytes per iteration — a coalesced transaction
    /// with no padding, unlike the AoS `gemv_q8_0`. `x` pre-quantized.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8_0_soa(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 32, 0, "Q8_0 rows are whole blocks");
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scales, y);
        let (mut o, mut i) = (out_dim, in_dim);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
        ];
        // 256 threads = 8 warps = 8 rows/block.
        cuda.launch(
            self.f_gemv_q8_0_soa,
            (ceil_div(out_dim, 8), 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )
    }

    /// Batched GEMM `Y[ntok, out] = X[ntok, in] @ W[out, in]^T` for Q8_0 SoA
    /// weights + int8 activations — the prefill path. The weight row is streamed
    /// once and reused across a tile of 4 tokens. `in_dim % 128 == 0` is
    /// required; `ntok` is the real token count but `x_qs`/`x_scales` must be
    /// allocated for `ntok` rounded up to a multiple of 4 (extra rows are read
    /// but never written).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q8_0_soa(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 128, 0, "gemm_q8_0_soa requires in_dim % 128 == 0");
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_gemm_q8_0_soa,
            (ceil_div(out_dim, 8), 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )
    }

    /// True when the tensor-core GEMM is available (device is sm_75+ and
    /// `GLCUDA_NO_MMA` is unset) — the runtime kernel selection callers use
    /// before [`Self::gemm_mma_q8`].
    pub fn has_mma(&self) -> bool {
        self.mma.is_some()
    }

    /// True when the Wave 12 N128 arm was requested. Individual launches may
    /// still fall back to N64 to preserve one-CTA-per-SM grid coverage.
    pub fn ntile128_enabled(&self) -> bool {
        self.ntile128
    }

    /// True when the Wave 12 exact cooperative-B image and kernel are active.
    pub fn bstage_enabled(&self) -> bool {
        self.bstage
    }

    /// True when the retained Wave 27 N16 per-warp output tile is selected.
    pub fn gemm_n16_enabled(&self) -> bool {
        self.gemm_n16
    }

    /// True when Wave 59 should replace only the retained narrow-grid arm.
    pub fn gemm_n32_enabled(&self) -> bool {
        self.gemm_n32
    }

    /// Whether Wave 88 is requested and this exact launch is in its scope.
    pub fn gemm_n16_prefetch_enabled(&self, out_dim: u32, in_dim: u32, ntok: u32) -> bool {
        self.gemm_n16_prefetch && n16_prefetch_shape(out_dim, in_dim, ntok)
    }

    /// Whether non-final prefill layers may defer their FFN residual add into
    /// the next attention RMS+Q8 pass.
    pub fn defer_ffn_residual_enabled(&self) -> bool {
        self.defer_ffn_residual.load(Ordering::Relaxed)
    }

    /// Switch Wave 111 between synchronized benchmark iterations.
    ///
    /// Production construction never calls this. The Wave 118 harness owns a
    /// single runner/model/context and uses this instead of process-global
    /// environment mutation.
    #[doc(hidden)]
    pub fn set_benchmark_defer_ffn_residual(&self, enabled: bool) {
        self.defer_ffn_residual
            .store(self.fuse_q8_glue && enabled, Ordering::Relaxed);
    }

    /// Switch Wave 123 no-store and stacked gate/up paths between synchronized
    /// benchmark iterations.
    #[doc(hidden)]
    pub fn set_benchmark_q8_nostore(&self, no_store: bool, stacked_gate_up: bool) {
        self.q8_nostore
            .store(self.fuse_q8_glue && no_store, Ordering::Relaxed);
        self.ffn_gate_up_stacked
            .store(self.fuse_q8_glue && stacked_gate_up, Ordering::Relaxed);
    }

    /// The Wave 27 wide-grid entry used by the driver's occupancy query.
    pub fn wave27_n16_resource_kernel(&self) -> Option<Kernel> {
        self.mma.as_ref().map(|module| module.bstage_n16)
    }

    /// The Wave 27 narrow-grid M32 entry used by the driver's occupancy query.
    pub fn wave27_n16_m32_resource_kernel(&self) -> Option<Kernel> {
        self.mma.as_ref().map(|module| module.bstage_n16_m32)
    }

    /// Wave 59 entry used by the direct resource and occupancy gate.
    pub fn wave59_n32_m32_resource_kernel(&self) -> Option<Kernel> {
        self.wave59.as_ref().map(|module| module.n32_m32)
    }

    /// Wave 88 entry used by the direct resource and occupancy gate.
    pub fn wave88_n16_prefetch_resource_kernel(&self) -> Option<Kernel> {
        self.wave88.as_ref().map(|module| module.n16_prefetch)
    }

    /// Whether this launch uses the narrow-grid M32 entry.
    pub fn gemm_n16_uses_m32(&self, out_dim: u32, ntok: u32) -> bool {
        n16_uses_m32(self.ntile128, out_dim, ntok, self.sm_count)
    }

    /// True when the retained row kernel was forced back over the default
    /// four-chain QK, which only an A/B should want.
    pub fn rows_forced(&self) -> bool {
        self.rows_forced
    }

    /// How many independent QK chains the GQA7 path should run: 1, 2 or 4.
    pub fn gqa7_chains(&self) -> u8 {
        self.gqa7_chains
    }

    fn mma_threads(&self, out_dim: u32, ntok: u32) -> u32 {
        if self.ntile128 && ntile128_covers(out_dim, ntok, self.sm_count) {
            512
        } else {
            256
        }
    }

    /// Should prefill use the 256-row GEMM? Requires the sm_75 module and
    /// `GLCUDA_R256` in the environment.
    pub fn r256_enabled(&self) -> bool {
        self.r256
    }

    /// Should prefill submit all 64-row slabs in one 2-D MMA launch? Requires
    /// the sm_75 module and `GLCUDA_GRID2D` in the environment.
    pub fn grid2d_enabled(&self) -> bool {
        self.grid2d
    }

    /// Batched GEMM `Y[ntok, out] = X[ntok, in] @ W[out, in]^T` on the INT8
    /// tensor cores (M2.1 Task B, sm_75+). Same operands as
    /// [`Self::gemm_q8_0_soa`] — the row-major Q8_0 SoA qs stream is already
    /// the col-major B fragment layout `mma.row.col` wants (W row-major ==
    /// B^T col-major), so the two kernels share one weight image. One warp
    /// per 8x8 output tile, 8-token tiles (vs the fallback's 4), fused
    /// per-32-K dequant epilogue in registers.
    ///
    /// M2.3 Stage 2a: the k-loop is outer and each weight fragment feeds up
    /// to eight 8-token m-tiles from registers — weights stream once per 64
    /// tokens instead of once per 8 (the llama.cpp head-to-head showed
    /// weight re-streaming was the prefill ceiling).
    ///
    /// Requires `out_dim % 8 == 0`, `in_dim % 32 == 0`, and
    /// `x_qs`/`x_scales` allocated for `ntok` rounded up to a multiple of 8
    /// (extra rows are read, never written). A 2-D grid partitions arbitrary
    /// `ntok` into independent 64-row CTAs. Errors if the module is not loaded — gate on
    /// [`Self::has_mma`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(out_dim % 8, 0, "gemm_mma_q8 requires out_dim % 8 == 0");
        debug_assert_eq!(
            in_dim % 32,
            0,
            "gemm_mma_q8 requires whole 32-K scale blocks"
        );
        debug_assert!(ntok > 0, "gemm_mma_q8 requires at least one token row");
        let f = &self
            .mma
            .as_ref()
            .ok_or_else(|| GlError::Engine("gemm_mma_q8 called without sm_75 module".into()))?
            .direct;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let threads = self.mma_threads(out_dim, ntok);
        let n_tile = threads / 4; // one 8-column MMA tile per 32-thread warp
        cuda.launch(
            *f,
            (ceil_div(out_dim, n_tile), ceil_div(ntok, 64), 1),
            (threads, 1, 1),
            0,
            &mut params,
        )
    }

    /// Wave 16: the MMA GEMM with one piece of its mainloop removed.
    ///
    /// ⛔ DIAGNOSTIC ONLY, and every ablated arm computes the WRONG ANSWER by
    /// construction. It is never on the inference path; what it produces is a
    /// time, never an output.
    ///
    /// The FFN GEMM runs at 8.2% of the T4's int8 peak and prefill is 5.85x
    /// behind llama.cpp. The mainloop stages, barriers, issues 16 `mma.sync`,
    /// barriers again, and repeats every 32 K-elements, so load and math
    /// strictly serialise - which is exactly what CUTLASS's sm_75 int8 example
    /// spends `NumStages = 2` to avoid. Before writing a pipelined kernel this
    /// prices the pieces, because a wave built on a guess about which piece
    /// binds is a wave spent either way.
    ///
    /// `ablate` is a bitmask: **1** skips the math block (leaving staging and
    /// barriers), **2** skips both `bar.sync`s (leaving staging and math),
    /// **4** skips the staging stores. The predicates come from a kernel
    /// parameter, so they are block-uniform and skipping `bar.sync` under them
    /// stays legal.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_probe(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
        ablate: u32,
    ) -> Result<(), GlError> {
        let f = &self
            .mma
            .as_ref()
            .ok_or_else(|| GlError::Engine("gemm_mma_q8_probe called without sm_75 module".into()))?
            .probe;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n, mut ab) = (out_dim, in_dim, ntok, ablate);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
        ];
        let threads = self.mma_threads(out_dim, ntok);
        let n_tile = threads / 4;
        cuda.launch(
            *f,
            (ceil_div(out_dim, n_tile), ceil_div(ntok, 64), 1),
            (threads, 1, 1),
            0,
            &mut params,
        )
    }

    /// Wave 12 exact Q8_0 MMA using a K32-major, N128-padded weight image.
    /// B bytes and f16 scale bits are staged cooperatively into shared memory;
    /// the MMA/dequant/write sequence remains identical to the retained kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_bstage(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert!(self.bstage, "B-stage launch requires GLCUDA_BSTAGE");
        debug_assert_eq!(out_dim % 8, 0);
        debug_assert_eq!(in_dim % 32, 0);
        let threads = self.mma_threads(out_dim, ntok);
        self.launch_mma_q8_bstage_with(
            cuda,
            tiled_w_qs,
            tiled_w_scales,
            x_qs,
            x_scales,
            y,
            out_dim,
            in_dim,
            ntok,
            threads,
            false,
        )
    }

    /// Wave 17: the B-stage GEMM with the next k-block prefetched.
    ///
    /// Wave 16B measured the retained B-stage mainloop and found staging worth
    /// **25.9%** unhidden, against 2.4% for barriers and 2.3% for the f32
    /// epilogue — so the exposed global-load latency is the largest single
    /// thing in that kernel that is not arithmetic. This issues those loads one
    /// k-block early, which is what CUTLASS's sm_75 int8 example spends
    /// `NumStages = 2` on.
    ///
    /// The stage is four registers per thread rather than a second shared
    /// buffer, so the shared footprint — and with it the 6-blocks-per-SM
    /// occupancy tier — is exactly the retained kernel's. Doubling shared would
    /// have dropped it to 3, and Wave 15C priced one tier down at 30%.
    ///
    /// Arithmetic, operand order and accumulation order are untouched, so the
    /// output is **bit-identical** to [`Self::gemm_mma_q8_bstage`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_bstage_pipe(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(out_dim % 8, 0);
        debug_assert_eq!(in_dim % 32, 0);
        let threads = self.mma_threads(out_dim, ntok);
        self.launch_mma_q8_bstage_with(
            cuda,
            tiled_w_qs,
            tiled_w_scales,
            x_qs,
            x_scales,
            y,
            out_dim,
            in_dim,
            ntok,
            threads,
            true,
        )
    }

    /// Hardware-correctness hook that forces either legal Wave 12 N tile.
    /// Production dispatch uses [`Self::gemm_mma_q8_bstage`] and its SM
    /// coverage guard; this entry exists so the notebook can prove both
    /// launch geometries against the retained direct kernel before timing.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_bstage_diagnostic(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
        n_tile: u32,
    ) -> Result<(), GlError> {
        let threads = match n_tile {
            64 => 256,
            128 => 512,
            _ => {
                return Err(GlError::Engine(format!(
                    "unsupported B-stage diagnostic N tile {n_tile}"
                )))
            }
        };
        self.launch_mma_q8_bstage_with(
            cuda,
            tiled_w_qs,
            tiled_w_scales,
            x_qs,
            x_scales,
            y,
            out_dim,
            in_dim,
            ntok,
            threads,
            false,
        )
    }

    /// Wave 27 exact N16 tile. Wide grids use one M64 warp per N16 fragment;
    /// narrow grids use paired M32 warps so the CTA returns to N64 without
    /// losing eight resident warps of work.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_bstage_n16(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert!(self.gemm_n16, "N16 launch requires GLCUDA_GEMM_N16");
        debug_assert_eq!(out_dim % 8, 0);
        debug_assert_eq!(in_dim % 32, 0);
        let use_m32 = self.gemm_n16_uses_m32(out_dim, ntok);
        let mma = self
            .mma
            .as_ref()
            .ok_or_else(|| GlError::Engine("N16 GEMM called without sm_75 module".into()))?;
        let (f, threads, n_tile) = if use_m32 {
            (&mma.bstage_n16_m32, 256, 64)
        } else {
            let threads = n16_threads(self.ntile128);
            (&mma.bstage_n16, threads, threads / 2)
        };
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) =
            (tiled_w_qs, tiled_w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        cuda.launch(
            *f,
            (ceil_div(out_dim, n_tile), ceil_div(ntok, 64), 1),
            (threads, 1, 1),
            0,
            &mut params,
        )
    }

    /// Wave 88 exact N16 register-prefetch candidate. Its output tile,
    /// arithmetic, shared image and launch geometry match the retained wide
    /// N16 entry; only the next K32 global-load issue point changes.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_bstage_n16_prefetch(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert!(self.gemm_n16_prefetch_enabled(out_dim, in_dim, ntok));
        debug_assert!(!self.gemm_n16_uses_m32(out_dim, ntok));
        let f = self
            .wave88
            .as_ref()
            .map(|module| module.n16_prefetch)
            .ok_or_else(|| GlError::Engine("Wave 88 GEMM called without its module".into()))?;
        let threads = n16_threads(self.ntile128);
        let n_tile = threads / 2;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) =
            (tiled_w_qs, tiled_w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        cuda.launch(
            f,
            (ceil_div(out_dim, n_tile), ceil_div(ntok, 64), 1),
            (threads, 1, 1),
            0,
            &mut params,
        )
    }

    /// Wave 59 N32 x M32 narrow-grid candidate. One 128-thread CTA covers
    /// exactly the same 4096 output elements as the retained N16/M32 CTA,
    /// while four warps reuse each activation fragment across four N8 tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_bstage_n32_m32(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert!(self.gemm_n32, "N32 launch requires GLCUDA_GEMM_N32");
        debug_assert_eq!(out_dim % 8, 0);
        debug_assert_eq!(in_dim % 32, 0);
        let f = self
            .wave59
            .as_ref()
            .map(|module| module.n32_m32)
            .ok_or_else(|| GlError::Engine("N32 GEMM called without Wave 59 module".into()))?;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) =
            (tiled_w_qs, tiled_w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        cuda.launch(
            f,
            (ceil_div(out_dim, 128), ceil_div(ntok, 32), 1),
            (128, 1, 1),
            0,
            &mut params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// `pipelined` picks Wave 17's prefetching variant. Both kernels take the
    /// same arguments, the same grid and the same shared memory; the only
    /// difference is when their global loads are issued, which is why one
    /// helper can launch either.
    #[allow(clippy::too_many_arguments)]
    fn launch_mma_q8_bstage_with(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
        threads: u32,
        pipelined: bool,
    ) -> Result<(), GlError> {
        debug_assert!(matches!(threads, 256 | 512));
        let m = self.mma.as_ref().ok_or_else(|| {
            GlError::Engine("gemm_mma_q8_bstage called without sm_75 module".into())
        })?;
        let f = if pipelined { &m.bstage_pipe } else { &m.bstage };
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) =
            (tiled_w_qs, tiled_w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let n_tile = threads / 4;
        cuda.launch(
            *f,
            (ceil_div(out_dim, n_tile), ceil_div(ntok, 64), 1),
            (threads, 1, 1),
            0,
            &mut params,
        )
    }

    /// Wave 16B: the B-STAGE MMA GEMM with one piece of its mainloop removed.
    ///
    /// ⛔ DIAGNOSTIC ONLY, and every ablated arm computes the WRONG ANSWER by
    /// construction. It produces times, never outputs, and is never on the
    /// inference path.
    ///
    /// The first Wave 16 probe copied the DIRECT kernel, but every benchmark
    /// sets `GLCUDA_BSTAGE=1`, so this is the one production actually runs.
    /// It found barriers to be the smallest bucket (2.9-8.4%), staging ~26%,
    /// tensor-core issue only ~7%, and **~60% of the kernel inside the math
    /// block yet not tensor-core issue**. The PTX says what is in there: per
    /// m-tile the mainloop issues 2 `mma.sync` against 11 non-MMA
    /// instructions, six of which are an f32 epilogue that depends on the MMA
    /// it just consumed and runs every 32 K because Q8_0 scales are per-32-K.
    ///
    /// `ablate` is a bitmask: **1** skips the math block, **2** skips both
    /// `bar.sync`s, **4** skips the staging stores, and **8** skips exactly
    /// that six-instruction epilogue in all eight m-tiles while keeping the
    /// loads and the MMAs. The skip is a predicated branch rather than a
    /// deletion, so the untaken path still consumes the MMA results and ptxas
    /// cannot dead-code them away.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_bstage_probe(
        &self,
        cuda: &Cuda,
        tiled_w_qs: CUdeviceptr,
        tiled_w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
        ablate: u32,
    ) -> Result<(), GlError> {
        let f = &self
            .mma
            .as_ref()
            .ok_or_else(|| {
                GlError::Engine("gemm_mma_q8_bstage_probe called without sm_75 module".into())
            })?
            .bstage_probe;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) =
            (tiled_w_qs, tiled_w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n, mut ab) = (out_dim, in_dim, ntok, ablate);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut ab as *mut _ as *mut c_void,
        ];
        let threads = self.mma_threads(out_dim, ntok);
        let n_tile = threads / 4;
        cuda.launch(
            *f,
            (ceil_div(out_dim, n_tile), ceil_div(ntok, 64), 1),
            (threads, 1, 1),
            0,
            &mut params,
        )
    }

    /// Diagnostic-only direct kernel launch for comparing N64/N128/N256.
    /// Production dispatch never selects N256 because its small-layer grid
    /// cannot cover all T4 SMs at the pinned prompt shape.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_diagnostic_ntile(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
        n_tile: u32,
    ) -> Result<(), GlError> {
        let threads = match n_tile {
            64 => 256,
            128 => 512,
            256 => 1024,
            _ => {
                return Err(GlError::Engine(format!(
                    "unsupported diagnostic N tile {n_tile}"
                )))
            }
        };
        let f = &self
            .mma
            .as_ref()
            .ok_or_else(|| GlError::Engine("diagnostic N-tile called without sm_75 module".into()))?
            .direct;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        cuda.launch(
            *f,
            (ceil_div(out_dim, n_tile), ceil_div(ntok, 64), 1),
            (threads, 1, 1),
            0,
            &mut params,
        )
    }

    /// Phase B weight-reuse GEMM: identical contract to [`Self::gemm_mma_q8`]
    /// but 32 m-tiles (up to 256 token rows per weight-fragment read) instead
    /// of 8. Requires `ntok <= 256` and x rows allocated to `round8(ntok)`.
    /// The weight walker/grid are the same; only the per-block row span grows,
    /// so the launch geometry is unchanged. Gate on [`Self::has_mma`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_r256(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
        ntok: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(out_dim % 8, 0, "gemm_mma_q8_r256 requires out_dim % 8 == 0");
        debug_assert_eq!(
            in_dim % 32,
            0,
            "gemm_mma_q8_r256 requires whole 32-K scale blocks"
        );
        debug_assert!(
            ntok <= 256,
            "gemm_mma_q8_r256 covers at most 32 m-tiles (256 rows)"
        );
        let f256 = &self
            .mma
            .as_ref()
            .ok_or_else(|| GlError::Engine("gemm_mma_q8_r256 called without sm_75 module".into()))?
            .r256;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n) = (out_dim, in_dim, ntok);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        cuda.launch(
            *f256,
            (ceil_div(out_dim, 64), 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )
    }

    /// `y = W @ x` for Q4_K weights in Structure-of-Arrays layout (M2.1
    /// Task A): `w_qs` packed nibbles `[out, in/2]`, `w_scales`/`w_mins`
    /// pre-multiplied f16 sub-block pairs `[out, in/32]` (see
    /// `repack::q4_k_to_soa` for the exact packing). `x` pre-quantized with
    /// [`Self::quantize_q8`] — the 32-value activation block matches the
    /// Q4_K sub-block, so the integer dot decomposes per sub-block as
    /// `(d*sc)*xs*dot(q,xq) - (dmin*m)*xs*sum(xq)`, both dp4a chains.
    /// One warp per row; one loop iteration streams one 256-weight
    /// super-block (128 coalesced qs bytes).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_soa(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        w_mins: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 256, 0, "Q4_K rows are whole super-blocks");
        let (mut wqs, mut wsc, mut wmn) = (w_qs, w_scales, w_mins);
        let (mut xqs, mut xsc, mut y) = (x_qs, x_scales, y);
        let (mut o, mut i) = (out_dim, in_dim);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut wmn as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
        ];
        // 256 threads = 8 warps = 8 rows/block, same geometry as the Q8_0 SoA GEMV.
        cuda.launch(
            self.f_gemv_q4_k_soa,
            (ceil_div(out_dim, 8), 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )
    }

    /// `y = W @ x` for Q4_0 weights in Structure-of-Arrays layout (M2.2
    /// Task C-2): `w_qs` packed nibbles `[out, in/2]` (Q4_K's kernel
    /// order), `w_scales` verbatim f16 block scales `[out, in/32]`. `x`
    /// pre-quantized with [`Self::quantize_q8`]. Per 32-value block the dot
    /// is `d*xs*(dot(q,xq) - 8*sum(xq))`, both terms dp4a chains with the
    /// -8 centering folded into the integer domain. One warp per row, one
    /// iteration per 256 values, guarded tail for `in % 256 != 0`
    /// (`in % 32 == 0` is the only requirement).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_0_soa(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 32, 0, "Q4_0 rows are whole blocks");
        let (mut wqs, mut wsc) = (w_qs, w_scales);
        let (mut xqs, mut xsc, mut y) = (x_qs, x_scales, y);
        let (mut o, mut i) = (out_dim, in_dim);
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
        ];
        // 256 threads = 8 warps = 8 rows/block, same geometry as the other SoA GEMVs.
        cuda.launch(
            self.f_gemv_q4_0_soa,
            (ceil_div(out_dim, 8), 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )
    }

    /// `y = W @ x` for Q6_K weights in Structure-of-Arrays layout (M2.2
    /// Task C-1): `w_ql` packed low nibbles `[out, in/2]`, `w_qh` 2-bit
    /// highs `[out, in/4]`, `w_scales` verbatim i8 sub-block scales
    /// `[out, in/16]`, `w_d` verbatim f16 super-block scales
    /// `[out, in/256]` (see `repack::q6_k_to_soa`). `x` pre-quantized with
    /// [`Self::quantize_q8`]. Per 16-value sub-block the dot is
    /// `d*sc*xs*(dot(q6,xq) - 32*sum(xq))` with q6 assembled from ql|qh<<4
    /// in registers. One warp per row, one iteration per super-block.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_soa(
        &self,
        cuda: &Cuda,
        w_ql: CUdeviceptr,
        w_qh: CUdeviceptr,
        w_scales: CUdeviceptr,
        w_d: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scales: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 256, 0, "Q6_K rows are whole super-blocks");
        let (mut wql, mut wqh, mut wsc, mut wd) = (w_ql, w_qh, w_scales, w_d);
        let (mut xqs, mut xsc, mut y) = (x_qs, x_scales, y);
        let (mut o, mut i) = (out_dim, in_dim);
        let mut params = [
            &mut wql as *mut _ as *mut c_void,
            &mut wqh as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut wd as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
        ];
        // 256 threads = 8 warps = 8 rows/block, same geometry as the other SoA GEMVs.
        cuda.launch(
            self.f_gemv_q6_k_soa,
            (ceil_div(out_dim, 8), 1, 1),
            (256, 1, 1),
            0,
            &mut params,
        )
    }

    /// Dynamically quantize `x` into `qs` and `scales`.
    pub fn quantize_q8(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        n: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(n % 32, 0, "quantize_q8 n must be a multiple of 32");
        let (mut x, mut qs, mut scales, mut n_) = (x, qs, scales, n);
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut scales as *mut _ as *mut c_void,
            &mut n_ as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_quantize_q8,
            (ceil_div(n, 32), 1, 1),
            (WARP, 1, 1),
            0,
            &mut params,
        )
    }

    /// `y = x * w^T` for Q4_0 weights (row-major). `w` is `[out_dim, in_dim]`.
    /// `in_dim` must be a multiple of 32 (Q4_0 block size).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_0(
        &self,
        cuda: &Cuda,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 32, 0, "Q4_0 rows are whole blocks");
        let (mut w, mut x, mut y) = (w, x, y);
        let (mut o, mut i) = (out_dim, in_dim);
        let mut params = [
            &mut w as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_gemv_q4_0,
            (out_dim, 1, 1),
            (WARP, 1, 1),
            0,
            &mut params,
        )
    }

    /// Transposed-access GEMV: `y[c] = Σ_r x[r] * a[r*cols + c]` — the
    /// attention weighted-V sum (`a` = V cache rows, `x` = scores).
    pub fn gemv_t(
        &self,
        cuda: &Cuda,
        a: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        rows: u32,
        cols: u32,
    ) -> Result<(), GlError> {
        let (mut a, mut x, mut y) = (a, x, y);
        let (mut r, mut c) = (rows, cols);
        let mut params = [
            &mut a as *mut _ as *mut c_void,
            &mut x as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut c as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_gemv_t,
            (ceil_div(cols, BLOCK), 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// RMSNorm: `out[i] = x[i] * rsqrt(mean(x²) + eps) * w[i]`, one block.
    pub fn rms_norm(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        w: CUdeviceptr,
        out: CUdeviceptr,
        dim: u32,
        eps: f32,
    ) -> Result<(), GlError> {
        let (mut x, mut w, mut out) = (x, w, out);
        let (mut d, mut e) = (dim, eps);
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut e as *mut _ as *mut c_void,
        ];
        cuda.launch(self.f_rms_norm, (1, 1, 1), (BLOCK, 1, 1), 0, &mut params)
    }

    /// In-place scaled softmax over `s[0..n]`: `s = softmax(s * scale)`.
    /// Scale is folded in so attention can feed raw Q·K dots straight from
    /// the GEMV kernel.
    pub fn softmax_scale(
        &self,
        cuda: &Cuda,
        s: CUdeviceptr,
        n: u32,
        scale: f32,
    ) -> Result<(), GlError> {
        let (mut s, mut n_, mut sc) = (s, n, scale);
        let mut params = [
            &mut s as *mut _ as *mut c_void,
            &mut n_ as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_softmax_scale,
            (1, 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Fused decode attention over ALL query heads in one launch (M2.1).
    /// One block per query head does Q·K, scaled softmax and the weighted-V
    /// sum in shared memory — replacing the per-head gemv+softmax+gemv_t
    /// triple (from `3 * n_heads` launches to 1).
    ///
    /// * `q` — all heads' query vectors, `[n_heads * head_dim]`
    /// * `k_base`/`v_base` — this layer's K/V region start (head 0); the
    ///   kernel offsets by `kv_head * head_stride` internally
    /// * `out` — all heads' attention output, `[n_heads * head_dim]`
    /// * `head_stride` — elements between consecutive KV heads' `[seq][dim]`
    ///   regions (`max_context * head_dim`)
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        cuda: &Cuda,
        q: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        out: CUdeviceptr,
        n_heads: u32,
        head_dim: u32,
        cached_len: CUdeviceptr,
        heads_per_kv: u32,
        head_stride: u32,
        scale: f32,
    ) -> Result<(), GlError> {
        // `cached_len` is a device pointer read at launch (token-invariant
        // args for M2.2 graph capture). Scores live in a fixed 16 KiB shared
        // array, so the caller must keep cached_len <= 4096.
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut cl, mut hpk, mut hs, mut sc) =
            (head_dim, cached_len, heads_per_kv, head_stride, scale);
        let mut params = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut cl as *mut _ as *mut c_void,
            &mut hpk as *mut _ as *mut c_void,
            &mut hs as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // One block per head, 128 threads (4 warps). Shared scores are
        // declared statically in the kernel, so shared_bytes here is 0.
        cuda.launch(
            self.f_attn_decode,
            (n_heads, 1, 1),
            (128, 1, 1),
            0,
            &mut params,
        )
    }
}

/// Host-side cos/sin tables for [`KernelSet::rope`] at one position —
/// exactly glproc's frequency formula, so the device rotation is
/// bit-compatible with the CPU reference.
pub fn rope_tables(pos: usize, head_dim: usize, freq_base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for i in 0..half {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let (s, c) = theta.sin_cos();
        cos.push(c);
        sin.push(s);
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave118_deferred_residual_override_is_explicit_and_fusion_gated() {
        assert!(!resolve_defer_ffn_residual(true, false, None));
        assert!(resolve_defer_ffn_residual(true, true, None));
        assert!(resolve_defer_ffn_residual(true, false, Some(true)));
        assert!(!resolve_defer_ffn_residual(true, true, Some(false)));
        assert!(!resolve_defer_ffn_residual(false, true, Some(true)));
    }

    /// The PTX image must declare exactly the entry points KernelSet
    /// resolves — catches drift between the .ptx file and this module
    /// without needing a GPU.
    #[test]
    fn ptx_declares_all_entries() {
        for entry in [
            "gl_add_f32",
            "gl_silu_mul_f32",
            "gl_silu_mul_quantize_q8",
            "gl_silu_mul_quantize_q8_stacked_nostore",
            "gl_rope_f32",
            "gl_gemv_f32",
            "gl_quantize_q8",
            "gl_rms_quantize_q8_rows",
            "gl_gemv_q8_0",
            "gl_gemv_q8_0_soa",
            "gl_gemm_q8_0_soa",
            "gl_gemv_q4_k_soa",
            "gl_gemv_q4_0_soa",
            "gl_gemv_q6_k_soa",
            "gl_gemv_t_f32",
            "gl_rms_norm_f32",
            "gl_softmax_scale_f32",
            "gl_attn_decode_f32",
            "gl_kv_write",
            "gl_rms_norm_rows_f32",
            "gl_add_bias_rows_f32",
            "gl_rope_rows_f32",
            "gl_kv_write_rows",
            "gl_attn_decode_rows_f32",
            "gl_attn_decode_rows_gqa7_f32",
            "gl_attn_rows_probe",
            "gl_attn_rows_qk4_probe",
        ] {
            assert!(
                PTX.contains(&format!(".visible .entry {entry}(")),
                "PTX is missing entry {entry}"
            );
        }
    }

    #[test]
    fn ptx_is_structurally_balanced() {
        let opens = PTX.matches('{').count();
        let closes = PTX.matches('}').count();
        assert_eq!(opens, closes, "unbalanced braces in PTX");
        assert!(
            PTX.starts_with("//"),
            "PTX must start with its header comment"
        );
        assert!(PTX.contains(".version 7.0"));
        assert!(PTX.contains(".target sm_70"));
        assert!(PTX.contains(".extern .shared .align 4 .b8 sm_attn_rows[];"));
        // rows, gqa7, the row probe, Wave 15A's four-chain QK, Wave 15B's two-
        // and four-chain GQA7, Wave 15C's GQA7 pass-split probe, and Wave 18's
        // pass-split probe for the shipped qk4 path.
        assert_eq!(PTX.matches(".param .u32 p_score_capacity").count(), 9);
        // Two probes now, and both must keep their stop parameter.
        // The three pass-split probes: rows, GQA7, and qk4.
        assert_eq!(PTX.matches(".param .u32 p_stop").count(), 3);
        // Wave 15B isolates chain count, so the chained GQA7 kernels must not
        // move the shared-memory footprint: all three read the same eight-row
        // tile, which is 512 f32 staged by the same cooperative load.
        for entry in [
            "gl_attn_decode_rows_gqa7_f32",
            "gl_attn_gqa7_qk2_f32",
            "gl_attn_gqa7_qk4_f32",
        ] {
            let body = &PTX[PTX.find(entry).expect("entry present")..];
            let body = &body[..body.find("// ---- Pass 2").unwrap_or(body.len())];
            assert!(
                body.contains("setp.ge.u32 %p2, %r18, 512;"),
                "{entry} changed its K tile; the chain-count factorial would be confounded"
            );
        }
        assert!(!PTX.contains("sm_scr[16384]"));
        assert!(!PTX.contains("sm_scp[16384]"));
        assert!(!PTX.contains('\0'), "NUL would truncate cuModuleLoadData");
        // ptxas rejects any non-ASCII byte with a fatal "Unexpected non-ASCII
        // character" before it parses a single instruction — a stray em-dash
        // in a comment kills the whole module. Catch it here, not on the GPU.
        if let Some(line) = PTX.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
            panic!("PTX line {} contains non-ASCII: {:?}", line.0 + 1, line.1);
        }
    }

    /// Wave 15D exists to cross one shared-memory granule, so the budget is
    /// the contract: 8944 B allocates as 8960 and fits seven blocks per SM,
    /// while 7920 allocates as 7936 and fits eight. If either number moves, the
    /// kernel stops buying the tier it was built for.
    #[test]
    fn wave15d_four_row_tile_crosses_the_occupancy_granule() {
        let eight_row = attn_rows_gqa7_shared_bytes(244).unwrap();
        let four_row = attn_rows_gqa7_t4_shared_bytes(244).unwrap();
        assert_eq!(eight_row, 8_944);
        assert_eq!(four_row, 7_920);
        assert_eq!(eight_row - four_row, 1_024, "exactly one 4x64 f32 tile");
        // The T4 allocates shared memory on a granule and has 64 KB per SM.
        // Both granules in use round these the same way, so the tier is not an
        // artifact of which one the driver picks.
        for granule in [128u32, 256u32] {
            let alloc = |b: u32| b.div_ceil(granule) * granule;
            assert_eq!(65_536 / alloc(eight_row), 7, "granule {granule}");
            assert_eq!(65_536 / alloc(four_row), 8, "granule {granule}");
        }
    }

    #[test]
    fn wave11_gqa7_shared_layout_and_residency_contract() {
        assert_eq!(attn_rows_gqa7_shared_bytes(244), Some(8_944));
        assert_eq!(attn_rows_gqa7_shared_bytes(288), Some(10_176));
        assert_eq!(attn_rows_gqa7_shared_bytes(0), None);
        assert_eq!(attn_rows_gqa7_shared_bytes(289), None);
        // Six 128-thread CTAs are 24 warps and fit the T4's 64 KiB SMEM.
        assert!(attn_rows_gqa7_shared_bytes(288).unwrap() * 6 <= 65_536);
    }

    #[test]
    fn wave11_ptx_keeps_exact_arithmetic_contract() {
        assert!(PTX.contains("W11_RMSQ_ACC:"));
        assert!(PTX.contains("@%p8 add.f32 %f3, %f3, %f4;"));
        assert!(PTX.contains("@%p8 st.global.f32 [%rd17], %f3;"));
        assert!(PTX.contains("GQA7_SCORE_TILE:"));
        assert!(PTX.contains("GQA7_V_TILE:"));
        assert!(!PTX.contains("cp.async"));
        assert!(!PTX
            .lines()
            .any(|line| line.trim_start().starts_with("rsqrt.approx")));
    }

    #[test]
    fn prefill_attention_shared_memory_tracks_real_context() {
        assert_eq!(attn_rows_shared_bytes(244), Some(1_012));
        assert_eq!(attn_rows_shared_bytes(512), Some(2_084));
        assert_eq!(attn_rows_shared_bytes(4_096), Some(16_420));
        assert_eq!(attn_rows_shared_bytes(0), None);
        assert_eq!(attn_rows_shared_bytes(u32::MAX), None);
    }

    #[test]
    fn wave20_mma4_smem_is_padded_bounded_and_excludes_static_q() {
        assert_eq!(attn_mma4_shared_bytes(244), Some(15_616));
        assert_eq!(attn_mma4_shared_bytes(245), Some(15_872));
        assert_eq!(attn_mma4_shared_bytes(640), Some(40_960));
        assert_eq!(attn_mma4_shared_bytes(0), None);
        assert_eq!(attn_mma4_shared_bytes(641), None);
        assert!(attn_mma4_shared_bytes(640).unwrap() + 4_096 <= 48 * 1024);
        assert_eq!(attn_mma4_regq_shared_bytes(1), Some(4_096));
        assert_eq!(attn_mma4_regq_shared_bytes(3), Some(4_096));
        assert_eq!(attn_mma4_regq_shared_bytes(244), Some(15_616));
        assert_eq!(attn_mma4_regq_shared_bytes(640), Some(40_960));
        assert_eq!(attn_mma4_regq_shared_bytes(0), None);
        assert_eq!(attn_mma4_regq_shared_bytes(641), None);
    }

    #[test]
    fn wave12_ntile128_never_reduces_a_full_t4_grid_below_40_ctas() {
        // Qwen2.5-0.5B prompt=244: q/o/down would expose only 28-32 CTAs
        // at N128 and must keep N64; fused gate+up remains amply wide.
        assert!(!ntile128_covers(896, 244, 40));
        assert!(!ntile128_covers(1_024, 244, 40));
        assert!(ntile128_covers(1_280, 244, 40));
        assert!(ntile128_covers(9_728, 244, 40));
        assert!(!ntile128_covers(4_992, 64, 40));
        assert!(ntile128_covers(5_120, 64, 40));
        // The hybrid keeps M64/N16 for covered wide grids and sends only
        // under-covered N128 launches through the paired-warp M32 entry.
        assert_eq!(n16_threads(false), 128);
        assert_eq!(n16_threads(true), 256);
        assert!(!n16_uses_m32(true, 9_728, 244, 40));
        assert!(n16_uses_m32(true, 896, 244, 40));
        assert!(n16_uses_m32(true, 136, 17, 40));
        assert!(!n16_uses_m32(false, 896, 244, 40));
    }

    /// Same structural gate for the sm_75 tensor-core module — it JITs on
    /// far fewer machines, so catching a stray byte here matters more.
    #[test]
    fn sm75_ptx_is_structurally_sound() {
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8("),
            "sm_75 PTX is missing gl_gemm_mma_q8"
        );
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8_bstage("),
            "sm_75 PTX is missing gl_gemm_mma_q8_bstage (Wave 12 cooperative B stage)"
        );
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8_bstage_n16("),
            "sm_75 PTX is missing the Wave 27 N16 per-warp entry"
        );
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8_bstage_n16_m32("),
            "sm_75 PTX is missing the Wave 27 narrow-grid M32 entry"
        );
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8_r256("),
            "sm_75 PTX is missing gl_gemm_mma_q8_r256 (Phase B weight-reuse GEMM)"
        );
        // r256 unrolls 32 m-tiles = 64 mma.sync ops; the base kernel has 16.
        // A regenerated body with the wrong tile count trips this.
        assert!(
            PTX_SM75.matches("mma.sync.aligned.m8n8k16").count() >= 64 + 16,
            "sm_75 PTX mma.sync count too low — r256 unroll incomplete?"
        );
        assert_eq!(PTX_SM75.matches('{').count(), PTX_SM75.matches('}').count());
        assert!(PTX_SM75.contains(".target sm_75"));
        assert!(PTX_SM75.contains("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32"));
        assert_eq!(
            PTX_SM75
                .matches("ldmatrix.sync.aligned.x2.m8n8.shared.b16")
                .count(),
            12,
            "retained N16 entries must keep eight wide and four M32 A-fragment loads"
        );
        assert_eq!(
            PTX_SM75
                .matches("ldmatrix.sync.aligned.x4.m8n8.shared.b16")
                .count(),
            2,
            "retained N16 entries must each keep one B-fragment load"
        );
        assert!(!PTX_SM75.contains("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"));
        // Wave 3 token-grid rebasing belongs only to the 8-m-tile kernel.
        // r256 already covers 256 rows internally and must stay a 1-D launch.
        // Five GEMM entries share this module and the file order has already
        // shifted once under me, slicing one region over its neighbour. So the
        // regions are derived rather than assumed: find every entry, sort by
        // position, and cut each one at whichever entry follows it.
        const GEMM_ENTRIES: [&str; 11] = [
            "gl_gemm_mma_q8(",
            // A non-GEMM entry sits between the direct kernel and its probe;
            // include it as a region boundary so the direct-kernel assertions
            // cannot accidentally inspect Wave 20's attention body.
            "gl_attn_mma4_fused_f32(",
            "gl_attn_mma4_regq_fused_f32(",
            "gl_attn_mma4_regq_avmma_fused_f32(",
            "gl_gemm_mma_q8_probe(",
            "gl_gemm_mma_q8_bstage(",
            "gl_gemm_mma_q8_bstage_n16(",
            "gl_gemm_mma_q8_bstage_n16_m32(",
            "gl_gemm_mma_q8_bstage_pipe(",
            "gl_gemm_mma_q8_bstage_probe(",
            "gl_gemm_mma_q8_r256(",
        ];
        let mut starts: Vec<(usize, &str)> = GEMM_ENTRIES
            .iter()
            .map(|name| {
                let needle = format!(".visible .entry {name}");
                (
                    PTX_SM75
                        .find(&needle)
                        .unwrap_or_else(|| panic!("sm_75 PTX is missing {name}")),
                    *name,
                )
            })
            .collect();
        starts.sort_unstable();
        let region = |name: &str| -> &str {
            let i = starts.iter().position(|(_, n)| *n == name).expect("entry");
            let from = starts[i].0;
            let to = starts.get(i + 1).map_or(PTX_SM75.len(), |(p, _)| *p);
            &PTX_SM75[from..to]
        };
        let base_kernel = region("gl_gemm_mma_q8(");
        let probe_kernel = region("gl_gemm_mma_q8_probe(");
        let bstage_kernel = region("gl_gemm_mma_q8_bstage(");
        let n16_kernel = region("gl_gemm_mma_q8_bstage_n16(");
        let n16_m32_kernel = region("gl_gemm_mma_q8_bstage_n16_m32(");
        let bspipe_kernel = region("gl_gemm_mma_q8_bstage_pipe(");
        let bsprobe_kernel = region("gl_gemm_mma_q8_bstage_probe(");
        let r256_kernel = region("gl_gemm_mma_q8_r256(");
        let mma4_attention = region("gl_attn_mma4_fused_f32(");
        let mma4_regq_attention = region("gl_attn_mma4_regq_fused_f32(");
        let mma4_regq_avmma_attention = region("gl_attn_mma4_regq_avmma_fused_f32(");
        assert_eq!(base_kernel.matches("%ctaid.y").count(), 1);
        // The probe is a copy of the base kernel plus predicated skips, so
        // it must keep the base geometry exactly; if it drifts, it stops
        // pricing the kernel it claims to.
        assert_eq!(probe_kernel.matches("%ctaid.y").count(), 1);
        assert_eq!(probe_kernel.matches("mma.sync.aligned").count(), 16);
        assert_eq!(probe_kernel.matches("bar.sync 0;").count(), 2);
        assert!(probe_kernel.contains("p_ablate"));
        // The B-stage probe is the one production's kernel is priced by, so
        // it carries the same contract plus the eight epilogue skips.
        assert_eq!(bsprobe_kernel.matches("%ctaid.y").count(), 1);
        assert_eq!(bsprobe_kernel.matches("mma.sync.aligned").count(), 16);
        assert_eq!(bsprobe_kernel.matches("bar.sync 0;").count(), 2);
        assert_eq!(bsprobe_kernel.matches("MMA_AB_NOEPI").count(), 16);
        assert!(bsprobe_kernel.contains("p_ablate"));
        assert_eq!(n16_kernel.matches("mma.sync.aligned").count(), 32);
        assert_eq!(n16_kernel.matches("bar.sync 0;").count(), 2);
        assert!(n16_kernel.contains("sm_a[3072]"));
        assert!(n16_kernel.contains("sm_xs[256]"));
        assert!(n16_kernel.contains("sm_b[6144]"));
        assert!(n16_kernel.contains("sm_bs[256]"));
        assert!(n16_kernel.contains("shl.b32 %r11, %r10, 4;"));
        assert!(n16_kernel.contains("shr.u32 %r_bntile, %r7, 1;"));
        assert!(n16_kernel.contains(".maxnreg 72"));
        assert_eq!(
            n16_kernel
                .matches("ldmatrix.sync.aligned.x2.m8n8.shared.b16")
                .count(),
            8
        );
        assert_eq!(
            n16_kernel
                .matches("ldmatrix.sync.aligned.x4.m8n8.shared.b16")
                .count(),
            1
        );
        assert!(!n16_kernel.contains("ld.shared.u32 %r24"));
        assert!(!n16_kernel.contains("ld.shared.u32 %r26"));
        assert!(!n16_kernel.contains("wmma."));
        assert_eq!(n16_m32_kernel.matches("mma.sync.aligned").count(), 16);
        assert_eq!(n16_m32_kernel.matches("bar.sync 0;").count(), 2);
        assert_eq!(n16_m32_kernel.matches("st.global.v2.f32").count(), 8);
        assert_eq!(n16_m32_kernel.matches("st.shared.u64").count(), 3);
        assert!(n16_m32_kernel.contains("sm_a[3072]"));
        assert!(n16_m32_kernel.contains("sm_xs[256]"));
        assert!(n16_m32_kernel.contains("sm_b[6144]"));
        assert!(n16_m32_kernel.contains("sm_bs[256]"));
        assert!(n16_m32_kernel.contains("shr.u32 %r_m32_ngroup, %r5, 1;"));
        assert!(n16_m32_kernel.contains("shl.b32 %r_m32_base, %r_m32_half, 5;"));
        assert!(n16_m32_kernel.contains("add.s32 %r_m32_row, %r_m32_base, %r12;"));
        assert!(n16_m32_kernel.contains("shr.u32 %r_bntile, %r7, 2;"));
        assert!(n16_m32_kernel.contains(".maxnreg 72"));
        assert_eq!(
            n16_m32_kernel
                .matches("ldmatrix.sync.aligned.x2.m8n8.shared.b16")
                .count(),
            4
        );
        assert_eq!(
            n16_m32_kernel
                .matches("ldmatrix.sync.aligned.x4.m8n8.shared.b16")
                .count(),
            1
        );
        assert!(!n16_m32_kernel.contains("ld.shared.u32 %r24"));
        assert!(!n16_m32_kernel.contains("ld.shared.u32 %r26"));
        assert!(!n16_m32_kernel.contains("wmma."));
        // Wave 17 changes WHEN the loads happen, never what is computed:
        // same MMAs, same barriers, and no global load left in the stage
        // block (all four moved into the prefetch).
        assert_eq!(bspipe_kernel.matches("mma.sync.aligned").count(), 16);
        assert_eq!(bspipe_kernel.matches("bar.sync 0;").count(), 2);
        assert_eq!(bspipe_kernel.matches("ld.global.u64 %rdP_").count(), 4);
        assert!(!bspipe_kernel.contains("ld.global.u64 %rd24"));
        assert!(!bspipe_kernel.contains("p_ablate"));
        assert_eq!(bstage_kernel.matches("%ctaid.y").count(), 1);
        assert_eq!(r256_kernel.matches("%ctaid.y").count(), 0);
        assert_eq!(
            mma4_attention.matches("mma.sync.aligned.m16n8k8").count(),
            4
        );
        assert_eq!(mma4_attention.matches("bar.sync 0;").count(), 2);
        assert!(mma4_attention.contains("wave20_q_smem[4096]"));
        assert!(mma4_attention.contains("sm_wave20_scores"));
        assert_eq!(
            mma4_regq_attention
                .matches("mma.sync.aligned.m16n8k8")
                .count(),
            32
        );
        assert_eq!(mma4_regq_attention.matches("bar.sync 0;").count(), 3);
        assert_eq!(
            mma4_regq_attention.matches("ld.shared.u32 %qa_").count(),
            32
        );
        assert_eq!(
            mma4_regq_attention
                .matches("mov.u32 %r14, sm_wave20_scores;")
                .count(),
            1
        );
        assert_eq!(
            mma4_regq_attention
                .matches("mov.u32 %r15, sm_wave20_scores;")
                .count(),
            1
        );
        assert!(mma4_regq_attention.contains("%qa_hi00, %qa_hi01"));
        assert!(!mma4_regq_attention.contains("%qa_hi0<8>"));
        assert!(mma4_regq_attention.contains("W48_Q_PRELOAD_WAIT:"));
        assert!(!mma4_regq_attention.contains("wave20_q_smem[4096]"));
        assert!(!mma4_regq_attention.contains("W48_K_CHUNK:"));
        assert_eq!(
            mma4_regq_avmma_attention
                .matches("mma.sync.aligned.m16n8k8")
                .count(),
            40
        );
        assert_eq!(mma4_regq_avmma_attention.matches("bar.sync 0;").count(), 4);
        assert_eq!(
            mma4_regq_avmma_attention
                .matches("ld.shared.u32 %qa_")
                .count(),
            32
        );
        assert!(mma4_regq_avmma_attention.contains("W78_NORM_LOOP:"));
        assert!(mma4_regq_avmma_attention.contains("W78_AV_MMA_K:"));
        assert!(mma4_regq_avmma_attention.contains("W78_AV_MMA_STORE:"));
        assert!(!mma4_regq_avmma_attention.contains("W78_AV_LOOP:"));
        assert!(!mma4_regq_avmma_attention.contains("wave20_q_smem[4096]"));
        assert!(base_kernel.contains("min.s32 %r3, %r_gy_rem, 64"));
        assert!(bstage_kernel.contains("min.s32 %r3, %r_gy_rem, 64"));
        // Every D lane pair is adjacent and 8-byte aligned, so both kernels
        // must retain one vector store per m-tile and no scalar pair stores.
        // Existing six retained entries plus N16-wide and N16-M32.
        assert_eq!(
            PTX_SM75.matches("st.global.v2.f32").count(),
            8 + 8 + 8 + 32 + 8 + 8 + 16 + 8
        );
        assert!(!PTX_SM75.contains("st.global.f32 [%rd32]"));

        // A 48-byte A-row pitch is both conflict-free for the MMA lane map
        // and naturally aligned for the required 64-bit staging stores. The
        // tempting 36-byte pitch satisfies neither property.
        assert!(PTX_SM75.contains("sm_a[3072]"));
        assert!(PTX_SM75.contains("sm_a[12288]"));
        // Existing retained entries plus N16's four and M32's three sites.
        assert_eq!(
            PTX_SM75.matches("st.shared.u64").count(),
            1 + 1 + 2 + 4 + 2 + 2 + 4 + 3
        );
        assert!(bstage_kernel.contains("sm_b[6144]"));
        assert!(bstage_kernel.contains("sm_bs[256]"));
        assert!(bstage_kernel.contains("shl.b64 %rd_bqbase, %rd_btileidx, 12"));
        assert!(bstage_kernel.contains("shl.b64 %rd_bsbase, %rd_btileidx, 8"));
        assert!(bstage_kernel.contains("add.s64 %rd_bqptr, %rd_bqptr, 4096"));
        assert!(bstage_kernel.contains("add.s64 %rd_bsptr, %rd_bsptr, 256"));
        assert!(bstage_kernel.contains("mov.u32 %r15, sm_b"));
        assert!(bstage_kernel.contains("mov.u32 %r23, sm_bs"));
        assert!(!bstage_kernel.contains("mov.u32 %r_baddr, sm_b"));
        assert!(!bstage_kernel.contains("mov.u32 %r_bsaddr, sm_bs"));
        assert!(!bstage_kernel.contains("ld.global.u32 %r26, [%rd11]"));
        let mut banks = Vec::with_capacity(32);
        for group_id in 0..8 {
            for tig in 0..4 {
                banks.push(((group_id * 48 + tig * 4) / 4) % 32);
            }
        }
        banks.sort_unstable();
        banks.dedup();
        assert_eq!(banks.len(), 32, "sm_a lane map must touch every bank once");
        for row in 0..256 {
            for byte_offset in [0, 8, 16, 24] {
                assert_eq!((row * 48 + byte_offset) % 8, 0);
            }
        }

        // The direct kernel, its Wave 16 probe copy, and the B-stage kernel
        // each keep a named NEXT B fragment and prefetch kb+1 before rotating
        // it into the current MMA operands at the existing barrier.
        assert_eq!(PTX_SM75.matches(".reg .b32 %bfrag0n, %bfrag1n;").count(), 3);
        assert_eq!(
            PTX_SM75
                .matches("ld.global.u32 %bfrag0n, [%rd11+32]")
                .count(),
            3
        );
        assert!(!PTX_SM75.contains('\0'));
        assert!(!PTX_SM75.contains('\r'), "CRLF would be rejected by ptxas");
        if let Some(line) = PTX_SM75.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
            panic!(
                "sm_75 PTX line {} contains non-ASCII: {:?}",
                line.0 + 1,
                line.1
            );
        }
    }

    #[test]
    fn wave59_n32_ptx_keeps_the_isolated_tile_contract() {
        assert!(PTX_SM75_WAVE59.starts_with(".version 6.5\n.target sm_75\n"));
        assert!(PTX_SM75_WAVE59.contains(".visible .entry gl_gemm_mma_q8_bstage_n32_m32("));
        assert_eq!(
            PTX_SM75_WAVE59.matches('{').count(),
            PTX_SM75_WAVE59.matches('}').count()
        );
        assert_eq!(
            PTX_SM75_WAVE59
                .matches("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32")
                .count(),
            32
        );
        assert_eq!(
            PTX_SM75_WAVE59
                .matches("ldmatrix.sync.aligned.x2.m8n8.shared.b16")
                .count(),
            4
        );
        assert_eq!(
            PTX_SM75_WAVE59
                .matches("ldmatrix.sync.aligned.x4.m8n8.shared.b16")
                .count(),
            2
        );
        assert_eq!(PTX_SM75_WAVE59.matches("bar.sync 0;").count(), 2);
        assert_eq!(PTX_SM75_WAVE59.matches("st.global.v2.f32").count(), 16);
        assert_eq!(PTX_SM75_WAVE59.matches("st.shared.u64").count(), 5);
        assert!(PTX_SM75_WAVE59.contains("sm_a[1536]"));
        assert!(PTX_SM75_WAVE59.contains("sm_xs[128]"));
        assert!(PTX_SM75_WAVE59.contains("sm_b[6144]"));
        assert!(PTX_SM75_WAVE59.contains("sm_bs[256]"));
        assert!(PTX_SM75_WAVE59.contains(".maxnreg 80"));
        assert!(PTX_SM75_WAVE59.contains("add.s64 %rd14, %rd14, 4096"));
        assert!(PTX_SM75_WAVE59.contains("add.s64 %rd18, %rd18, 256"));
        assert!(!PTX_SM75_WAVE59.contains("wmma."));
        assert!(!PTX_SM75_WAVE59.contains('\0'));
        assert!(
            !PTX_SM75_WAVE59.contains('\r'),
            "CRLF would be rejected by ptxas"
        );
        if let Some(line) = PTX_SM75_WAVE59
            .lines()
            .enumerate()
            .find(|(_, line)| !line.is_ascii())
        {
            panic!(
                "Wave 59 PTX line {} contains non-ASCII: {:?}",
                line.0 + 1,
                line.1
            );
        }
        // Retained narrow CTA: N64 x M64 at 256 threads. Candidate CTA:
        // N128 x M32 at 128 threads. Only the work decomposition changes.
        assert_eq!(64 * 64, 128 * 32);
    }

    #[test]
    fn wave88_n16_prefetch_keeps_the_isolated_exact_tile_contract() {
        assert!(PTX_SM75_WAVE88.starts_with(".version 6.5\n.target sm_75\n"));
        assert!(PTX_SM75_WAVE88.contains(".visible .entry gl_gemm_mma_q8_bstage_n16_prefetch("));
        assert_eq!(
            PTX_SM75_WAVE88.matches('{').count(),
            PTX_SM75_WAVE88.matches('}').count()
        );
        assert_eq!(PTX_SM75_WAVE88.matches("mma.sync.aligned").count(), 32);
        assert_eq!(PTX_SM75_WAVE88.matches("bar.sync 0;").count(), 2);
        assert_eq!(
            PTX_SM75_WAVE88
                .matches("ldmatrix.sync.aligned.x2.m8n8.shared.b16")
                .count(),
            8
        );
        assert_eq!(
            PTX_SM75_WAVE88
                .matches("ldmatrix.sync.aligned.x4.m8n8.shared.b16")
                .count(),
            1
        );
        assert!(PTX_SM75_WAVE88.contains("sm_a[3072]"));
        assert!(PTX_SM75_WAVE88.contains("sm_xs[256]"));
        assert!(PTX_SM75_WAVE88.contains("sm_b[6144]"));
        assert!(PTX_SM75_WAVE88.contains("sm_bs[256]"));
        assert!(PTX_SM75_WAVE88.contains(".maxnreg 80"));
        assert!(PTX_SM75_WAVE88.contains("%rdP_a0"));
        assert!(PTX_SM75_WAVE88.contains("%rdP_a1"));
        assert!(PTX_SM75_WAVE88.contains("%rdP_b0"));
        assert!(PTX_SM75_WAVE88.contains("%rdP_b1"));
        assert!(!PTX_SM75_WAVE88.contains("wmma."));
        assert!(!PTX_SM75_WAVE88.contains('\0'));
        assert!(!PTX_SM75_WAVE88.contains('\r'));
        assert!(n16_prefetch_shape(4_864, 896, 244));
        assert!(!n16_prefetch_shape(896, 4_864, 244));
        assert!(!n16_prefetch_shape(4_864, 896, 243));
    }

    #[test]
    fn rope_tables_match_glproc_formula() {
        let (cos, sin) = rope_tables(7, 8, 10_000.0);
        assert_eq!(cos.len(), 4);
        for i in 0..4 {
            let freq = 1.0f32 / 10_000f32.powf(2.0 * i as f32 / 8.0);
            let theta = 7.0 * freq;
            assert_eq!(cos[i], theta.cos());
            assert_eq!(sin[i], theta.sin());
        }
    }
}
