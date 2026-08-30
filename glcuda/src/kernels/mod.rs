//! Typed launch wrappers around the PTX kernel suite.
//!
//! Each method mirrors one row of the ArchGLML_X2 §16 kernel inventory and
//! encodes that kernel's launch geometry, so callers never repeat grid
//! math. All launches go to the default stream; the caller synchronizes
//! once per forward pass (or per test).

use std::ffi::c_void;

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

/// Threads per block for element-wise and one-block-reduction kernels.
const BLOCK: u32 = 256;
/// Warp size — grid geometry for the one-warp-per-row GEMV.
const WARP: u32 = 32;
/// Four warp partials plus one broadcast slot. The PTX reserves 36 bytes so
/// the same reduction layout also remains valid if the block grows to eight
/// warps later.
const ATTN_ROWS_REDUCTION_BYTES: u32 = 36;

fn ceil_div(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
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

/// One loaded module plus resolved handles for every kernel. Handles stay
/// valid while `_module` lives — the struct owns it for exactly that.
pub struct KernelSet {
    _module: Module,
    /// The sm_75+ tensor-core module and its three GEMM entries, present only on
    /// capable devices (and absent under `GLCUDA_NO_MMA=1`, the benchmark
    /// A/B switch). The `Option` IS the runtime kernel selection: callers
    /// ask [`KernelSet::has_mma`] and fall back to `gl_gemm_q8_0_soa`.
    /// Tuple: (module, grid64, r128, r256, w8pc). Wave 6's r128 entry is the measured
    /// midpoint: 128 rows/weight-read with 128-thread CTAs to retain enough
    /// grid width for the narrow down/o projections on a 40-SM T4.
    mma: Option<(Module, Kernel, Kernel, Kernel, Kernel)>,
    /// Wave 7 W8A8 row-scaled contract. Read once from `GLCUDA_W8PC` and
    /// enabled only when the sm_75 module is available.
    w8pc: bool,
    /// Wave 8 byte-identical Q8_0 activation quantizer. When enabled, prefill
    /// uses one 256-thread CTA per token row instead of one 32-thread CTA per
    /// K32 block. Decode stays on the original kernel.
    q8_rowcta: bool,
    /// Whether prefill may use the Wave 6 128-row GEMM when it strictly saves
    /// a weight read. Read once from `GLCUDA_R128`.
    r128: bool,
    /// Whether prefill should drive the r256 (256-row) GEMM instead of the
    /// 64-row one. Read once at load from `GLCUDA_R256`; see
    /// [`KernelSet::r256_enabled`].
    r256: bool,
    /// Whether the 8-m-tile kernel should cover all token slabs in one 2-D
    /// launch. Read once from `GLCUDA_GRID2D`; see
    /// [`KernelSet::grid2d_enabled`].
    grid2d: bool,
    /// Wave 9 grouped CTA ordering. This swaps the grid axes so token slabs
    /// that consume one output weight tile are adjacent in linear CTA order.
    l2_raster: bool,
    f_add: Kernel,
    f_silu_mul: Kernel,
    f_rope: Kernel,
    f_gemv: Kernel,
    f_quantize_q8: Kernel,
    f_quantize_q8_rowcta: Kernel,
    f_quantize_q8_rows: Kernel,
    f_gemv_q8_0: Kernel,
    f_gemv_q8_0_soa: Kernel,
    f_gemv_w8pc: Kernel,
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
    /// Diagnostic pass-split copy of `gl_attn_decode_rows_f32` (bench-only —
    /// the engine never launches it; see [`Self::attn_rows_probe`]).
    f_attn_rows_probe: Kernel,
}

impl KernelSet {
    /// JIT the embedded PTX and resolve every entry point. On sm_75+ the
    /// tensor-core module is loaded too (`GLCUDA_NO_MMA=1` opts out, for
    /// A/B benchmarking against the sm_70 dp4a GEMM).
    pub fn load(cuda: &Cuda) -> Result<KernelSet, GlError> {
        let module = cuda.load_module(PTX)?;
        let sm = (cuda.info.sm_major, cuda.info.sm_minor);
        let mma = if sm >= (7, 5) && std::env::var_os("GLCUDA_NO_MMA").is_none() {
            let m75 = cuda.load_module(PTX_SM75)?;
            let f = m75.get_function("gl_gemm_mma_q8")?;
            let f128 = m75.get_function("gl_gemm_mma_q8_r128")?;
            let f256 = m75.get_function("gl_gemm_mma_q8_r256")?;
            let fw8pc = m75.get_function("gl_gemm_mma_w8pc")?;
            eprintln!(
                "[glcuda] tensor-core MMA GEMM enabled (sm_{}{})",
                cuda.info.sm_major, cuda.info.sm_minor
            );
            Some((m75, f, f128, f256, fw8pc))
        } else {
            if sm >= (7, 5) {
                eprintln!("[glcuda] GLCUDA_NO_MMA set: prefill GEMM on the sm_70 dp4a path");
            }
            None
        };
        // Wave 6 A/B switch. r128 is selected only when ceil(n/128) is
        // strictly below ceil(n/64); short prompts keep the retained grid64
        // path even when the switch is present.
        let r128 = mma.is_some() && std::env::var_os("GLCUDA_R128").is_some();
        if r128 {
            eprintln!("[glcuda] r128 prefill GEMM enabled (128-row weight reuse, 128-thread CTA)");
        }
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
        let l2_raster = mma.is_some() && std::env::var_os("GLCUDA_L2_RASTER").is_some();
        if l2_raster {
            eprintln!(
                "[glcuda] L2-grouped GEMM raster enabled (token slabs adjacent per weight tile)"
            );
        }
        let w8pc = mma.is_some() && std::env::var_os("GLCUDA_W8PC").is_some();
        if w8pc {
            eprintln!("[glcuda] W8PC MMA enabled (full-K s32 accumulation, one dequant epilogue)");
        }
        let q8_rowcta = std::env::var_os("GLCUDA_Q8_ROWCTA").is_some();
        if q8_rowcta {
            eprintln!(
                "[glcuda] Q8_0 row-CTA prefill quantizer enabled (byte-identical K32 contract)"
            );
        }
        eprintln!("[glcuda] dynamic-shared prefill attention enabled");
        Ok(KernelSet {
            mma,
            w8pc,
            q8_rowcta,
            r128,
            r256,
            grid2d,
            l2_raster,
            f_add: module.get_function("gl_add_f32")?,
            f_silu_mul: module.get_function("gl_silu_mul_f32")?,
            f_rope: module.get_function("gl_rope_f32")?,
            f_gemv: module.get_function("gl_gemv_f32")?,
            f_quantize_q8: module.get_function("gl_quantize_q8")?,
            f_quantize_q8_rowcta: module.get_function("gl_quantize_q8_rowcta")?,
            f_quantize_q8_rows: module.get_function("gl_quantize_q8_rows")?,
            f_gemv_q8_0: module.get_function("gl_gemv_q8_0")?,
            f_gemv_q8_0_soa: module.get_function("gl_gemv_q8_0_soa")?,
            f_gemv_w8pc: module.get_function("gl_gemv_w8pc")?,
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
            f_attn_rows_probe: module.get_function("gl_attn_rows_probe")?,
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

    /// Broadcast bias add over a `[rows, dim]` activation block in one
    /// launch: `y[i] += b[i % dim]` for `i < total` (M2.3 prefill).
    pub fn add_bias_rows(
        &self,
        cuda: &Cuda,
        y: CUdeviceptr,
        b: CUdeviceptr,
        dim: u32,
        total: u32,
    ) -> Result<(), GlError> {
        let (mut y, mut b) = (y, b);
        let (mut d, mut t) = (dim, total);
        let mut params = [
            &mut y as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut t as *mut _ as *mut c_void,
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
    ) -> Result<(), GlError> {
        let (mut x, mut cos, mut sin) = (x, cos, sin);
        let (mut h, mut hd, mut nx, mut p) = (n_heads, head_dim, neox as u32, pos_seq);
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
    ) -> Result<(), GlError> {
        let (mut d, mut s, mut p) = (dst_base, src, pos_seq);
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
            self.f_kv_write_rows,
            (ceil_div(n, BLOCK), ntok, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Batched causal decode-attention over `ntok` token rows in one launch
    /// (M2.3 prefill): block (h, t) runs head h of row t with
    /// `cached_len = pos_seq[t] + 1`, so each row attends to exactly its
    /// own prefix (rows after it exist in the cache but are never read).
    /// Requires the chunk's KV rows to be written first (kv_write_rows on
    /// the same stream). `score_capacity` is the largest causal length in
    /// this launch (`chunk_base + ntok`); it sizes dynamic shared memory to
    /// the real prompt prefix instead of reserving 4096 scores for every CTA.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_rows(
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
    ) -> Result<(), GlError> {
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs, mut sc) =
            (head_dim, pos_seq, heads_per_kv, head_stride, scale);
        let mut cap = score_capacity;
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
    ) -> Result<(), GlError> {
        let (mut q, mut k, mut v, mut o) = (q, k_base, v_base, out);
        let (mut hd, mut ps, mut hpk, mut hs) = (head_dim, pos_seq, heads_per_kv, head_stride);
        let (mut sc, mut cap, mut st) = (scale, score_capacity, stop);
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

    /// Decode/fallback GEMV for Wave 7 W8PC weights and a row-quantized
    /// activation. Both operands have one f32 scale for the complete K row,
    /// so the warp reduces an s32 dot and applies the two scales once.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_w8pc(
        &self,
        cuda: &Cuda,
        w_qs: CUdeviceptr,
        w_scales: CUdeviceptr,
        x_qs: CUdeviceptr,
        x_scale: CUdeviceptr,
        y: CUdeviceptr,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), GlError> {
        debug_assert_eq!(in_dim % 4, 0, "gemv_w8pc requires packed int8x4 K");
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scale, y);
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
        cuda.launch(
            self.f_gemv_w8pc,
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

    /// True when the Wave 7 row-scaled W8A8 path was explicitly requested
    /// and the device can execute its sm_75 MMA kernel.
    pub fn w8pc_enabled(&self) -> bool {
        self.w8pc
    }

    /// Should prefill use the 128-row Wave 6 GEMM when it reduces weight
    /// streams? Requires the sm_75 module and `GLCUDA_R128`.
    pub fn r128_enabled(&self) -> bool {
        self.r128
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

    /// Should prefill group token-slab CTAs by output weight tile? Requires
    /// the sm_75 module and `GLCUDA_L2_RASTER` in the environment.
    pub fn l2_raster_enabled(&self) -> bool {
        self.l2_raster
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
        self.gemm_mma_q8_raster(
            cuda, w_qs, w_scales, x_qs, x_scales, y, out_dim, in_dim, ntok, false,
        )
    }

    /// Wave 9 L2-grouped launch of [`Self::gemm_mma_q8`]. The same PTX entry
    /// and arithmetic are used; only the uniform CTA coordinate mapping and
    /// launch-axis order differ.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_l2(
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
        self.gemm_mma_q8_raster(
            cuda, w_qs, w_scales, x_qs, x_scales, y, out_dim, in_dim, ntok, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_mma_q8_raster(
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
        l2_raster: bool,
    ) -> Result<(), GlError> {
        debug_assert_eq!(out_dim % 8, 0, "gemm_mma_q8 requires out_dim % 8 == 0");
        debug_assert_eq!(
            in_dim % 32,
            0,
            "gemm_mma_q8 requires whole 32-K scale blocks"
        );
        debug_assert!(ntok > 0, "gemm_mma_q8 requires at least one token row");
        let (_, f, _, _, _) = self
            .mma
            .as_ref()
            .ok_or_else(|| GlError::Engine("gemm_mma_q8 called without sm_75 module".into()))?;
        let (mut wqs, mut wsc, mut xqs, mut xsc, mut y) = (w_qs, w_scales, x_qs, x_scales, y);
        let (mut o, mut i, mut n, mut raster) = (out_dim, in_dim, ntok, u32::from(l2_raster));
        let mut params = [
            &mut wqs as *mut _ as *mut c_void,
            &mut wsc as *mut _ as *mut c_void,
            &mut xqs as *mut _ as *mut c_void,
            &mut xsc as *mut _ as *mut c_void,
            &mut y as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void,
            &mut i as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut raster as *mut _ as *mut c_void,
        ];
        // 256 threads = 8 warps = 8 output tiles of 8 rows per block. grid.y
        // replaces the serial host slab loop; for ntok <= 64 it is exactly 1.
        let output_tiles = ceil_div(out_dim, 64);
        let token_slabs = ceil_div(ntok, 64);
        let grid = if l2_raster {
            (token_slabs, output_tiles, 1)
        } else {
            (output_tiles, token_slabs, 1)
        };
        cuda.launch(*f, grid, (256, 1, 1), 0, &mut params)
    }

    /// Wave 6 midpoint GEMM: 16 m-tiles per warp, four warps per CTA, and a
    /// 2-D grid of 128-token slabs. At 244 tokens the narrow 896-row matrices
    /// launch 56 CTAs instead of r256's 14, while each weight row is streamed
    /// twice instead of grid64's four times.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_q8_r128(
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
        debug_assert_eq!(out_dim % 8, 0, "gemm_mma_q8_r128 requires out_dim % 8 == 0");
        debug_assert_eq!(
            in_dim % 32,
            0,
            "gemm_mma_q8_r128 requires whole 32-K scale blocks"
        );
        debug_assert!(ntok > 0, "gemm_mma_q8_r128 requires at least one token row");
        let (_, _, f128, _, _) = self.mma.as_ref().ok_or_else(|| {
            GlError::Engine("gemm_mma_q8_r128 called without sm_75 module".into())
        })?;
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
            *f128,
            (ceil_div(out_dim, 32), ceil_div(ntok, 128), 1),
            (128, 1, 1),
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
        let (_, _, _, f256, _) = self.mma.as_ref().ok_or_else(|| {
            GlError::Engine("gemm_mma_q8_r256 called without sm_75 module".into())
        })?;
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

    /// Wave 7 W8A8 GEMM. Geometry and MMA shape match the retained grid64
    /// kernel, but weight and activation scales are invariant across K. Each
    /// D fragment therefore accumulates in s32 for the full dot and is
    /// converted/scaled once at the output epilogue.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mma_w8pc(
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
        debug_assert_eq!(out_dim % 8, 0, "gemm_mma_w8pc requires out_dim % 8 == 0");
        debug_assert_eq!(in_dim % 32, 0, "gemm_mma_w8pc requires in_dim % 32 == 0");
        debug_assert!(ntok > 0, "gemm_mma_w8pc requires token rows");
        let (_, _, _, _, f) = self
            .mma
            .as_ref()
            .ok_or_else(|| GlError::Engine("gemm_mma_w8pc called without sm_75 module".into()))?;
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
            (ceil_div(out_dim, 64), ceil_div(ntok, 64), 1),
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

    /// Quantize a Q8_0 activation matrix with one 256-thread CTA per row.
    /// Eight warps independently execute the original K32 algorithm, so the
    /// output bytes and f32 scale bits must match [`Self::quantize_q8`].
    pub fn quantize_q8_rowcta(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        rows: u32,
        cols: u32,
    ) -> Result<(), GlError> {
        debug_assert!(rows > 0, "quantize_q8_rowcta needs at least one row");
        debug_assert_eq!(
            cols % 32,
            0,
            "quantize_q8_rowcta rows must contain whole K32 blocks"
        );
        let (mut x, mut qs, mut scales) = (x, qs, scales);
        let (mut rows_, mut cols_) = (rows, cols);
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut scales as *mut _ as *mut c_void,
            &mut rows_ as *mut _ as *mut c_void,
            &mut cols_ as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_quantize_q8_rowcta,
            (rows, 1, 1),
            (BLOCK, 1, 1),
            0,
            &mut params,
        )
    }

    /// Production Q8_0 matrix quantizer. Wave 8 only changes batched prefill:
    /// decode-shaped rows remain on the captured one-warp-per-K32 kernel.
    pub fn quantize_q8_matrix(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        rows: u32,
        cols: u32,
    ) -> Result<(), GlError> {
        if self.q8_rowcta && rows > 1 {
            self.quantize_q8_rowcta(cuda, x, qs, scales, rows, cols)
        } else {
            let n = rows.checked_mul(cols).ok_or_else(|| {
                GlError::Engine(format!("Q8_0 activation shape {rows}x{cols} overflows u32"))
            })?;
            self.quantize_q8(cuda, x, qs, scales, n)
        }
    }

    /// Dynamically quantize `rows` activation rows with one f32 scale per
    /// row. Launches one 256-thread CTA per row; each CTA performs a block
    /// max reduction, then a second coalesced pass that writes signed INT8.
    pub fn quantize_q8_rows(
        &self,
        cuda: &Cuda,
        x: CUdeviceptr,
        qs: CUdeviceptr,
        scales: CUdeviceptr,
        rows: u32,
        cols: u32,
    ) -> Result<(), GlError> {
        debug_assert!(
            rows > 0 && cols > 0,
            "quantize_q8_rows needs a non-empty matrix"
        );
        let (mut x, mut qs, mut scales) = (x, qs, scales);
        let (mut r, mut c) = (rows, cols);
        let mut params = [
            &mut x as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
            &mut scales as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut c as *mut _ as *mut c_void,
        ];
        cuda.launch(
            self.f_quantize_q8_rows,
            (rows, 1, 1),
            (256, 1, 1),
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

    /// The PTX image must declare exactly the entry points KernelSet
    /// resolves — catches drift between the .ptx file and this module
    /// without needing a GPU.
    #[test]
    fn ptx_declares_all_entries() {
        for entry in [
            "gl_add_f32",
            "gl_silu_mul_f32",
            "gl_rope_f32",
            "gl_gemv_f32",
            "gl_quantize_q8",
            "gl_quantize_q8_rowcta",
            "gl_quantize_q8_rows",
            "gl_gemv_q8_0",
            "gl_gemv_q8_0_soa",
            "gl_gemv_w8pc",
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
            "gl_attn_rows_probe",
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
        assert_eq!(PTX.matches(".param .u32 p_score_capacity").count(), 2);
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

    #[test]
    fn prefill_attention_shared_memory_tracks_real_context() {
        assert_eq!(attn_rows_shared_bytes(244), Some(1_012));
        assert_eq!(attn_rows_shared_bytes(512), Some(2_084));
        assert_eq!(attn_rows_shared_bytes(4_096), Some(16_420));
        assert_eq!(attn_rows_shared_bytes(0), None);
        assert_eq!(attn_rows_shared_bytes(u32::MAX), None);
    }

    /// Same structural gate for the sm_75 tensor-core module — it JITs on
    /// far fewer machines, so catching a stray byte here matters more.
    #[test]
    fn row_cta_quantizer_preserves_the_k32_warp_contract() {
        let start = PTX
            .find(".visible .entry gl_quantize_q8_rowcta(")
            .expect("row-CTA quantizer entry");
        let end = PTX[start..]
            .find("\n}\n")
            .map(|offset| start + offset + 2)
            .expect("row-CTA quantizer end");
        let body = &PTX[start..end];
        assert!(body.contains("shr.u32 %r7, %r2, 5"));
        assert!(body.contains("add.s32 %r9, %r9, 8"));
        assert_eq!(body.matches("shfl.sync.down.b32").count(), 5);
        assert_eq!(body.matches("shfl.sync.idx.b32").count(), 1);
        assert!(body.contains("div.rn.f32 %f5, %f4, 127.0"));
        assert!(body.contains("cvt.rni.s32.f32 %r11, %f7"));
        assert!(!body.contains("bar.sync"));
        assert!(!body.contains(".shared"));
    }

    #[test]
    fn sm75_ptx_is_structurally_sound() {
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8("),
            "sm_75 PTX is missing gl_gemm_mma_q8"
        );
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8_r128("),
            "sm_75 PTX is missing the Wave 6 r128 GEMM"
        );
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_q8_r256("),
            "sm_75 PTX is missing gl_gemm_mma_q8_r256 (Phase B weight-reuse GEMM)"
        );
        assert!(
            PTX_SM75.contains(".visible .entry gl_gemm_mma_w8pc("),
            "sm_75 PTX is missing the Wave 7 W8PC GEMM"
        );
        // Exact instruction counts: grid64=16, W8PC=16, r128=32, r256=64.
        assert_eq!(
            PTX_SM75.matches("mma.sync.aligned.m8n8k16").count(),
            16 + 16 + 32 + 64
        );
        assert_eq!(PTX_SM75.matches('{').count(), PTX_SM75.matches('}').count());
        assert!(PTX_SM75.contains(".target sm_75"));
        assert!(PTX_SM75.contains("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32"));
        // Wave 3 grid64 and Wave 6 r128 rebase token slabs through grid.y.
        // r256 covers its token span internally and remains a 1-D launch.
        let w8pc_entry = PTX_SM75
            .find(".visible .entry gl_gemm_mma_w8pc(")
            .expect("w8pc entry");
        let r128_entry = PTX_SM75
            .find(".visible .entry gl_gemm_mma_q8_r128(")
            .expect("r128 entry");
        let r256_entry = PTX_SM75
            .find(".visible .entry gl_gemm_mma_q8_r256(")
            .expect("r256 entry");
        let base_kernel = &PTX_SM75[..w8pc_entry];
        let w8pc_kernel = &PTX_SM75[w8pc_entry..r128_entry];
        let r128_kernel = &PTX_SM75[r128_entry..r256_entry];
        let r256_kernel = &PTX_SM75[r256_entry..];
        // Wave 9 names both launch coordinates and uniformly swaps them when
        // p_l2_raster is set. The other kernels retain their old mapping.
        assert_eq!(base_kernel.matches("%ctaid.y").count(), 2);
        assert_eq!(base_kernel.matches("%ctaid.x").count(), 2);
        assert_eq!(w8pc_kernel.matches("%ctaid.y").count(), 1);
        assert_eq!(r128_kernel.matches("%ctaid.y").count(), 1);
        assert_eq!(r256_kernel.matches("%ctaid.y").count(), 0);
        assert!(base_kernel.contains("min.s32 %r3, %r_gy_rem, 64"));
        assert_eq!(base_kernel.matches(".param .u32 p_l2_raster").count(), 1);
        assert!(base_kernel.contains("setp.ne.u32 %p_l2, %r_l2_mode, 0"));
        assert!(base_kernel.contains("@%p_l2 mov.u32 %r_l2_out, %ctaid.y"));
        assert!(base_kernel.contains("@%p_l2 mov.u32 %r_l2_slab, %ctaid.x"));
        assert_eq!(base_kernel.matches("bar.sync 0").count(), 2);
        assert_eq!(base_kernel.matches("mma.sync.aligned.m8n8k16").count(), 16);
        assert!(w8pc_kernel.contains("min.s32 %r3, %r_w8_rem, 64"));
        assert!(r128_kernel.contains("min.s32 %r3, %r_gy_rem, 128"));
        // Every D lane pair is adjacent and 8-byte aligned, so all kernels
        // must retain one vector store per m-tile and no scalar pair stores.
        assert_eq!(
            PTX_SM75.matches("st.global.v2.f32").count(),
            8 + 8 + 16 + 32
        );
        assert!(!PTX_SM75.contains("st.global.f32 [%rd32]"));

        // A 48-byte A-row pitch is both conflict-free for the MMA lane map
        // and naturally aligned for the required 64-bit staging stores. The
        // tempting 36-byte pitch satisfies neither property.
        assert!(PTX_SM75.contains("sm_a[3072]"));
        assert!(PTX_SM75.contains("sm_w8a[3072]"));
        assert!(PTX_SM75.contains("sm_a[6144]"));
        assert!(PTX_SM75.contains("sm_a[12288]"));
        assert_eq!(PTX_SM75.matches("st.shared.u64").count(), 1 + 1 + 4 + 4);
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

        // All kernels keep a named NEXT B fragment and prefetch kb+1 before
        // rotating it into the current MMA operands at the existing barrier.
        assert_eq!(PTX_SM75.matches(".reg .b32 %bfrag0n, %bfrag1n;").count(), 4);
        assert_eq!(
            PTX_SM75
                .matches("ld.global.u32 %bfrag0n, [%rd11+32]")
                .count(),
            4
        );
        assert_eq!(
            PTX_SM75
                .lines()
                .filter(|line| line.trim_start().starts_with("cp.async"))
                .count(),
            0
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
