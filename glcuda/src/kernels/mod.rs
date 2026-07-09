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

/// Threads per block for element-wise and one-block-reduction kernels.
const BLOCK: u32 = 256;
/// Warp size — grid geometry for the one-warp-per-row GEMV.
const WARP: u32 = 32;

fn ceil_div(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}

/// One loaded module plus resolved handles for every kernel. Handles stay
/// valid while `_module` lives — the struct owns it for exactly that.
pub struct KernelSet {
    _module: Module,
    f_add: Kernel,
    f_silu_mul: Kernel,
    f_rope: Kernel,
    f_gemv: Kernel,
    f_quantize_q8: Kernel,
    f_gemv_q8_0: Kernel,
    f_gemv_q4_0: Kernel,
    f_gemv_t: Kernel,
    f_rms_norm: Kernel,
    f_softmax_scale: Kernel,
    f_attn_decode: Kernel,
    f_kv_write: Kernel,
}

impl KernelSet {
    /// JIT the embedded PTX and resolve every entry point.
    pub fn load(cuda: &Cuda) -> Result<KernelSet, GlError> {
        let module = cuda.load_module(PTX)?;
        Ok(KernelSet {
            f_add: module.get_function("gl_add_f32")?,
            f_silu_mul: module.get_function("gl_silu_mul_f32")?,
            f_rope: module.get_function("gl_rope_f32")?,
            f_gemv: module.get_function("gl_gemv_f32")?,
            f_quantize_q8: module.get_function("gl_quantize_q8")?,
            f_gemv_q8_0: module.get_function("gl_gemv_q8_0")?,
            f_gemv_q4_0: module.get_function("gl_gemv_q4_0")?,
            f_gemv_t: module.get_function("gl_gemv_t_f32")?,
            f_rms_norm: module.get_function("gl_rms_norm_f32")?,
            f_softmax_scale: module.get_function("gl_softmax_scale_f32")?,
            f_attn_decode: module.get_function("gl_attn_decode_f32")?,
            f_kv_write: module.get_function("gl_kv_write")?,
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
        cuda.launch(self.f_add, (ceil_div(n, BLOCK), 1, 1), (BLOCK, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_silu_mul, (ceil_div(n, BLOCK), 1, 1), (BLOCK, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_rope, (ceil_div(pairs, BLOCK), 1, 1), (BLOCK, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_kv_write, (ceil_div(n, BLOCK), 1, 1), (BLOCK, 1, 1), 0, &mut params)
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
        debug_assert_eq!(out_dim % 4, 0, "Q8_0 out_dim must be multiple of 4 for Thread Coarsening");
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
        cuda.launch(self.f_gemv_q8_0, (ceil_div(out_dim, 16), 1, 1), (128, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_quantize_q8, (ceil_div(n, 32), 1, 1), (WARP, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_gemv_q4_0, (out_dim, 1, 1), (WARP, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_gemv_t, (ceil_div(cols, BLOCK), 1, 1), (BLOCK, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_softmax_scale, (1, 1, 1), (BLOCK, 1, 1), 0, &mut params)
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
        cuda.launch(self.f_attn_decode, (n_heads, 1, 1), (128, 1, 1), 0, &mut params)
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
            "gl_gemv_q8_0",
            "gl_gemv_t_f32",
            "gl_rms_norm_f32",
            "gl_softmax_scale_f32",
            "gl_attn_decode_f32",
            "gl_kv_write",
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
        assert!(PTX.starts_with("//"), "PTX must start with its header comment");
        assert!(PTX.contains(".version 7.0"));
        assert!(PTX.contains(".target sm_70"));
        assert!(!PTX.contains('\0'), "NUL would truncate cuModuleLoadData");
        // ptxas rejects any non-ASCII byte with a fatal "Unexpected non-ASCII
        // character" before it parses a single instruction — a stray em-dash
        // in a comment kills the whole module. Catch it here, not on the GPU.
        if let Some(line) = PTX.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
            panic!("PTX line {} contains non-ASCII: {:?}", line.0 + 1, line.1);
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
