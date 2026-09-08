//! Attention as its own backend path.
//!
//! Wave 12 measured why this deserves a boundary. Prefill attention holds
//! about 1.7% of the model's arithmetic and about 30% of prefill wall time: it
//! ran at 130 GMAC/s while the FFN GEMM beside it ran at 5162. A stage that
//! far off the rest of the engine is a performance domain, not an operator.
//!
//! So the runner says what it wants computed and this module decides how. The
//! three types split that responsibility:
//!
//! * [`VLAttentionCall`] is semantics. Tokens, heads, positions, scale. It
//!   says nothing about tiles, warps, or shared memory.
//! * [`ENAttentionPath`] is the strategy, chosen in [`select`] and nowhere
//!   else, and returned so callers know what actually ran.
//! * [`VLAttentionCost`] is what the call had to multiply and move. Wave 12
//!   could not answer "which part of attention" because the stage reported
//!   zero MACs and zero bytes; this closes that hole.
//!
//! Prefill and decode are deliberately not unified here. Many queries against
//! a cache and one query against a cache share these semantics, but they are
//! different shape regimes and must not be forced to share a kernel strategy.
//! Only prefill lives in this module today; decode still runs its own path.
//!
//! [`reference`] holds the oracle every path is graded against: the same
//! semantics in plain host code, slow on purpose, and unit-testable without a
//! GPU.

pub mod reference;

use glcore::GlError;

use crate::driver::Cuda;
use crate::ffi::CUdeviceptr;
use crate::kernels::KernelSet;

/// One prefill attention call, in the model's terms.
///
/// Everything here is shape and semantics. Device addresses are passed
/// alongside it, so this type stays cheap to build in a test that has no GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VLAttentionCall {
    /// Query rows in this chunk.
    pub n_tokens: u32,
    /// KV rows already cached before this chunk. Row `t` attends to
    /// `pos_base + t + 1` of them, which is the whole causal contract.
    pub pos_base: u32,
    /// Query heads.
    pub n_heads: u32,
    /// Key/value heads. Equal to `n_heads` when the model is not GQA.
    pub n_kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Distance between two heads in the KV cache, in elements.
    pub head_stride: u32,
    /// Softmax scale, passed through rather than recomputed per kernel.
    pub scale: f32,
}

impl VLAttentionCall {
    /// Query heads sharing one KV head. 1 when the model is not GQA.
    ///
    /// Both clamps are the runner's retained expression: the kernels divide by
    /// this, so a degenerate config must still land on 1 rather than 0.
    pub fn heads_per_kv(&self) -> u32 {
        (self.n_heads / self.n_kv_heads.max(1)).max(1)
    }

    /// KV rows the last query row of this chunk sees, which is also the exact
    /// score-buffer capacity every CTA in the launch must hold.
    pub fn score_capacity(&self) -> u32 {
        self.pos_base + self.n_tokens
    }

    /// Sum of `cached_len` over this chunk's query rows.
    ///
    /// This is the causal triangle, and it is why attention is O(n^2) while
    /// every GEMM around it is O(n): row `t` attends to `pos_base + t + 1` KV
    /// rows, so the total is `n*pos_base + n(n+1)/2`.
    pub fn causal_kv_rows(&self) -> u64 {
        let n = u64::from(self.n_tokens);
        n * u64::from(self.pos_base) + n * (n + 1) / 2
    }
}

/// Which attention implementation ran.
///
/// Returned rather than assumed, because it changes the traffic a caller must
/// charge: the GQA7 path loads one K/V history per seven query heads, the row
/// path loads one per head. Guessing wrong is a 7x error in the byte column of
/// the roofline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENAttentionPath {
    /// The retained Wave 4 kernel: one CTA per (head, query row).
    Rows,
    /// The Wave 11 kernel: the seven query heads that share a KV head ride in
    /// one CTA, so their K/V tile is loaded once for the group.
    Gqa7,
    /// Wave 15B: GQA7 with two independent QK chains per warp.
    Gqa7Qk2,
    /// Wave 15B: GQA7 with four (two tile rows times two query heads).
    Gqa7Qk4,
    /// Wave 15A: the row kernel with four independent QK chains per warp.
    ///
    /// Same memory pattern and same per-score arithmetic as [`Self::Rows`], so
    /// its output is bit-identical; what differs is that four chains are in
    /// flight at once, which is the lever Wave 14 pointed at.
    Qk4,
    /// Wave 20 opt-in: a 16-query compensated-f16 MMA QK tile fused directly
    /// into causal softmax and AV. The shipped qk4 path remains the fallback.
    Mma4,
    /// Wave 48: Wave 20's exact arithmetic with Q fragments captured once in
    /// registers so the static 4 KiB shared tile disappears.
    Mma4RegQ,
    /// Wave 78: Wave 48 QK/softmax with compensated-f16 MMA for the final AV.
    Mma4RegQAvMma,
}

impl ENAttentionPath {
    /// Short name for telemetry and logs.
    pub fn name(&self) -> &'static str {
        match self {
            ENAttentionPath::Rows => "rows",
            ENAttentionPath::Gqa7 => "gqa7",
            ENAttentionPath::Qk4 => "qk4",
            ENAttentionPath::Mma4 => "mma4-fused",
            ENAttentionPath::Mma4RegQ => "mma4-regq",
            ENAttentionPath::Mma4RegQAvMma => "mma4-regq-avmma",
            ENAttentionPath::Gqa7Qk2 => "gqa7+qk2",
            ENAttentionPath::Gqa7Qk4 => "gqa7+qk4",
        }
    }

    /// How many independent K/V histories this path streams for one call.
    fn kv_histories(&self, call: &VLAttentionCall) -> u64 {
        match self {
            // Qk4 keeps the row kernel's one-CTA-per-(head, row) shape, so it
            // streams a history per head exactly as Rows does.
            ENAttentionPath::Rows
            | ENAttentionPath::Qk4
            | ENAttentionPath::Mma4
            | ENAttentionPath::Mma4RegQ
            | ENAttentionPath::Mma4RegQAvMma => u64::from(call.n_heads),
            // The chained GQA7 kernels share GQA7's one-CTA-per-group shape,
            // so they stream exactly the same K/V history.
            ENAttentionPath::Gqa7 | ENAttentionPath::Gqa7Qk2 | ENAttentionPath::Gqa7Qk4 => {
                u64::from(call.n_kv_heads)
            }
        }
    }
}

/// f32 activations and an f32 KV cache, so every element is four bytes.
const ELEM_BYTES: u64 = 4;

/// What one attention call multiplies and moves.
///
/// Modeled from shapes rather than counted by the kernel, exactly like the
/// GEMM stages next to it. Bytes are *logical* loads: L2 may serve some of
/// them, but a cache hit is a cache effect, not a load the kernel never
/// issued.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VLAttentionCost {
    /// Multiply-accumulates in `Q @ K^T`.
    pub qk_macs: u64,
    /// Multiply-accumulates in `P @ V`.
    pub pv_macs: u64,
    /// Query bytes read.
    pub bytes_q: u64,
    /// Key bytes read, over every history this path streams.
    pub bytes_k: u64,
    /// Value bytes read, same accounting as [`Self::bytes_k`].
    pub bytes_v: u64,
    /// Output bytes written. Deliberately not part of [`Self::read_bytes`].
    pub bytes_out: u64,
}

impl VLAttentionCost {
    /// Model one call's cost. `path` matters, see
    /// [`ENAttentionPath::kv_histories`].
    pub fn of(call: &VLAttentionCall, path: ENAttentionPath) -> VLAttentionCost {
        let head_dim = u64::from(call.head_dim);
        let kv_rows = call.causal_kv_rows();
        let dots = kv_rows * u64::from(call.n_heads) * head_dim;
        let stream = path.kv_histories(call) * kv_rows * head_dim * ELEM_BYTES;
        let activations =
            u64::from(call.n_tokens) * u64::from(call.n_heads) * head_dim * ELEM_BYTES;
        VLAttentionCost {
            qk_macs: dots,
            pv_macs: dots,
            bytes_q: activations,
            bytes_k: stream,
            bytes_v: stream,
            bytes_out: activations,
        }
    }

    /// Total multiply-accumulates.
    pub fn macs(&self) -> u64 {
        self.qk_macs + self.pv_macs
    }

    /// Total bytes read. Writes are excluded so this matches how the GEMM
    /// stages report their traffic.
    pub fn read_bytes(&self) -> u64 {
        self.bytes_q + self.bytes_k + self.bytes_v
    }
}

/// Pick the implementation for this call. The one place that decision lives.
/// The inference path never pads shared memory; only an audit does, to move
/// resident blocks per SM without touching the kernel.
///
/// This is a named constant rather than a bare `0` so the contract survives
/// in code: the wave notebooks strip comments before they grep the source,
/// so a promise written in a comment is a promise nothing can check.
const NO_SMEM_PAD: u32 = 0;

pub fn select(kernels: &KernelSet, call: &VLAttentionCall) -> ENAttentionPath {
    // Wave 15A is RETAINED and is now the default.
    //
    // Measured on a T4 in one interleaved session, three arms, two repeats,
    // every arm audited by the path the engine announces: **+8.59% over the
    // row kernel and +3.65% over GQA7**, both repeats positive. It sits under
    // the repo's 5% retention bar, so this is a judgement call JinXSuper made
    // rather than a bar it cleared, and two facts made the call cheap: the run
    // that first "failed" on tail latency was re-audited from the raw ten
    // samples and the tail turned out to be a warm-up artifact that hit a
    // different arm each repeat, and every one of these kernels is
    // **bit-identical** to the others. Switching the default therefore changes
    // the schedule and not one output bit.
    //
    // `GLCUDA_ATTN_ROWS` forces the retained row kernel back, which is what an
    // A/B needs; `GLCUDA_GQA_GROUP` still selects the KV-sharing family.
    //
    // ⚠️ The +3.65% is measured at the pinned 244-token prompt. GQA7 shares one
    // K/V tile across seven heads, so its advantage grows with context while
    // Qk4's shrank from 1.349x to 1.299x between 244 and 1024 in screening.
    // Qk4 against GQA7 at long context has never been measured, and that is the
    // regime to watch before trusting this default there.
    if kernels.rows_forced() {
        return ENAttentionPath::Rows;
    }
    if kernels.mma4_regq_avmma_attention_enabled()
        && call.head_dim == 64
        && kernels.mma4_regq_attention_capacity_supported(call.score_capacity())
    {
        return ENAttentionPath::Mma4RegQAvMma;
    }
    if kernels.mma4_regq_attention_enabled()
        && call.head_dim == 64
        && kernels.mma4_regq_attention_capacity_supported(call.score_capacity())
    {
        return ENAttentionPath::Mma4RegQ;
    }
    if kernels.mma4_attention_enabled()
        && call.head_dim == 64
        && kernels.mma4_attention_capacity_supported(call.score_capacity())
    {
        return ENAttentionPath::Mma4;
    }
    if kernels.gqa_group_enabled()
        && call.n_heads.is_multiple_of(7)
        && call.heads_per_kv() == 7
        && call.head_dim == 64
        && kernels.gqa7_capacity_supported(call.score_capacity())
    {
        // Wave 15B: the chain count is one dial on the SAME kernel shape, so
        // it is chosen here rather than by a second flag that could silently
        // combine with something else.
        match kernels.gqa7_chains() {
            2 => ENAttentionPath::Gqa7Qk2,
            4 => ENAttentionPath::Gqa7Qk4,
            _ => ENAttentionPath::Gqa7,
        }
    } else {
        ENAttentionPath::Qk4
    }
}

/// Run one prefill attention call and report which path did it.
///
/// `q` is this chunk's query rows and `q_row_stride` is the distance between
/// them, which is `n_heads * head_dim` for a packed buffer and wider when `q`
/// is a column slice of a stacked projection. `k_cache`/`v_cache` are the
/// layer's cache bases, `pos_seq` is the device position table the kernels
/// index, and `out` receives a packed `[n_tokens, n_heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    cuda: &Cuda,
    kernels: &KernelSet,
    q: CUdeviceptr,
    q_row_stride: u32,
    k_cache: CUdeviceptr,
    v_cache: CUdeviceptr,
    out: CUdeviceptr,
    pos_seq: CUdeviceptr,
    call: &VLAttentionCall,
) -> Result<ENAttentionPath, GlError> {
    let path = select(kernels, call);
    // Wave 13B's precedent: state which path ran, so an A/B arm is auditable
    // rather than inferred from its own timing. Once per process, because this
    // is called 24 times a prefill.
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    ANNOUNCED.call_once(|| {
        eprintln!(
            "[glcuda-attn] {{\"path\":\"{}\",\"heads\":{},\"kv_heads\":{},\"head_dim\":{},\"ntok\":{}}}",
            path.name(),
            call.n_heads,
            call.n_kv_heads,
            call.head_dim,
            call.n_tokens,
        );
    });
    let launch = match path {
        ENAttentionPath::Gqa7 => KernelSet::attn_decode_rows_gqa7,
        ENAttentionPath::Rows => KernelSet::attn_decode_rows_legacy,
        ENAttentionPath::Qk4 => KernelSet::attn_rows_qk4,
        ENAttentionPath::Mma4 => {
            kernels.attn_mma4_fused(
                cuda,
                q,
                k_cache,
                v_cache,
                out,
                call.n_heads,
                call.head_dim,
                pos_seq,
                call.heads_per_kv(),
                call.head_stride,
                call.scale,
                call.n_tokens,
                call.score_capacity(),
                q_row_stride,
            )?;
            return Ok(path);
        }
        ENAttentionPath::Mma4RegQ => {
            kernels.attn_mma4_regq_fused(
                cuda,
                q,
                k_cache,
                v_cache,
                out,
                call.n_heads,
                call.head_dim,
                pos_seq,
                call.heads_per_kv(),
                call.head_stride,
                call.scale,
                call.n_tokens,
                call.score_capacity(),
                q_row_stride,
            )?;
            return Ok(path);
        }
        ENAttentionPath::Mma4RegQAvMma => {
            kernels.attn_mma4_regq_avmma_fused(
                cuda,
                q,
                k_cache,
                v_cache,
                out,
                call.n_heads,
                call.head_dim,
                pos_seq,
                call.heads_per_kv(),
                call.head_stride,
                call.scale,
                call.n_tokens,
                call.score_capacity(),
                q_row_stride,
            )?;
            return Ok(path);
        }
        // The chained GQA7 kernels take the chain count as an extra argument,
        // so they are dispatched directly rather than through the shared
        // function pointer the other three share.
        ENAttentionPath::Gqa7Qk2 | ENAttentionPath::Gqa7Qk4 => {
            let chains = if path == ENAttentionPath::Gqa7Qk2 {
                2
            } else {
                4
            };
            kernels.attn_gqa7_chained(
                cuda,
                q,
                k_cache,
                v_cache,
                out,
                call.n_heads,
                call.head_dim,
                pos_seq,
                call.heads_per_kv(),
                call.head_stride,
                call.scale,
                call.n_tokens,
                call.score_capacity(),
                q_row_stride,
                chains,
                NO_SMEM_PAD,
            )?;
            return Ok(path);
        }
    };
    launch(
        kernels,
        cuda,
        q,
        k_cache,
        v_cache,
        out,
        call.n_heads,
        call.head_dim,
        pos_seq,
        call.heads_per_kv(),
        call.head_stride,
        call.scale,
        call.n_tokens,
        call.score_capacity(),
        q_row_stride,
    )?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen2.5-0.5B attending over the pinned 244-token prompt, one layer.
    fn qwen_call() -> VLAttentionCall {
        VLAttentionCall {
            n_tokens: 244,
            pos_base: 0,
            n_heads: 14,
            n_kv_heads: 2,
            head_dim: 64,
            head_stride: 64,
            scale: 0.125,
        }
    }

    #[test]
    fn causal_kv_rows_is_the_triangle_not_the_square() {
        let call = qwen_call();
        assert_eq!(call.causal_kv_rows(), 244 * 245 / 2);
        // A later chunk carries the whole cache in front of it.
        let tail = VLAttentionCall {
            n_tokens: 44,
            pos_base: 200,
            ..call
        };
        assert_eq!(tail.causal_kv_rows(), 44 * 200 + 44 * 45 / 2);
        // Chunking must not change the work one prompt costs.
        let head = VLAttentionCall {
            n_tokens: 200,
            pos_base: 0,
            ..call
        };
        assert_eq!(
            head.causal_kv_rows() + tail.causal_kv_rows(),
            call.causal_kv_rows()
        );
    }

    /// The figure Wave 12 could not see, because the stage reported zero MACs.
    /// Attention holds a sliver of the model's arithmetic, so when it costs a
    /// third of prefill the answer is in the byte column, not this one.
    #[test]
    fn qwen_attention_macs_match_the_hand_computed_wave12_figure() {
        let cost = VLAttentionCost::of(&qwen_call(), ENAttentionPath::Gqa7);
        assert_eq!(cost.macs(), 2 * 29_890 * 14 * 64);
        // All 24 layers, the whole prompt.
        assert_eq!(cost.macs() * 24, 1_285_509_120);
    }

    /// Grouping is a traffic decision before it is a speed decision: seven
    /// query heads sharing a KV head should not stream it seven times.
    #[test]
    fn gqa7_streams_one_seventh_of_the_row_paths_kv_bytes() {
        let call = qwen_call();
        let rows = VLAttentionCost::of(&call, ENAttentionPath::Rows);
        let gqa7 = VLAttentionCost::of(&call, ENAttentionPath::Gqa7);
        assert_eq!(rows.macs(), gqa7.macs(), "same arithmetic either way");
        assert_eq!(rows.bytes_k, gqa7.bytes_k * 7);
        assert_eq!(
            rows.bytes_q, gqa7.bytes_q,
            "queries are read once regardless"
        );
        // The KV stream dominates even after grouping: 15.3 MB against 0.87 MB
        // for one layer of the pinned prompt.
        assert!(gqa7.bytes_k > gqa7.bytes_q * 17);
    }

    #[test]
    fn read_bytes_excludes_the_output_write() {
        let cost = VLAttentionCost::of(&qwen_call(), ENAttentionPath::Gqa7);
        assert_eq!(
            cost.read_bytes(),
            cost.bytes_q + cost.bytes_k + cost.bytes_v
        );
        assert_eq!(cost.bytes_out, cost.bytes_q);
    }

    #[test]
    fn heads_per_kv_survives_a_model_that_is_not_gqa() {
        let call = VLAttentionCall {
            n_heads: 8,
            n_kv_heads: 8,
            ..qwen_call()
        };
        assert_eq!(call.heads_per_kv(), 1);
        assert_eq!(
            VLAttentionCost::of(&call, ENAttentionPath::Rows),
            VLAttentionCost::of(&call, ENAttentionPath::Gqa7),
            "one head per KV head means the two paths move identical bytes"
        );
    }
}
