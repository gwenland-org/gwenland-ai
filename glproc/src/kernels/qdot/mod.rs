//! Integer-domain fused dot: quantized weights × Q8-quantized activations.
//!
//! The f32 bridge dequantizes every weight to f32 before the dot — correct,
//! but the dequant instructions dominate on 4/5-bit formats. This module
//! instead quantizes the *activation vector* to int8 once per matvec
//! (~`in_dim` ops, amortized over `out_dim` rows) and keeps the inner loop
//! in the integer domain: `_mm256_maddubs_epi16` does 32 multiply-adds per
//! instruction, and the result is scaled back to f32 once per block.
//!
//! Accuracy: activations get one 8-bit scale per 32 values (the same scheme
//! llama.cpp uses for all its k-quant matvecs); relative error per dot is
//! ~1e-3, well under the quantization noise of the weights themselves.
//!
//! Layout note: every weight format here packs its blocks so that a block's
//! weights are contiguous in the *logical* row; the activation is quantized
//! in matching 32-element groups, so block `j` of a row always pairs with
//! `q[j*32..j*32+32]`, `scales[j]`, `sums[j]`.

pub mod q4_k;
pub mod q8_k;
pub mod q5_0;
pub mod q6_k;
pub mod q8_0;

use crate::simd_strategy::SimdStrategy;

/// f16 → f32 through the F16C `vcvtph2ps` instruction. The software
/// conversion is a branchy ~15-op routine and the AVX2 kernels burn one
/// conversion per weight block (millions per token) — this is 1 instruction.
///
/// # Safety
/// CPU must support F16C. Guaranteed by `SimdStrategy::detect`, which only
/// selects a wide backend when `f16c` is present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
#[inline]
pub(crate) unsafe fn f16_hw(bits: u16) -> f32 {
    use std::arch::x86_64::*;
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

/// An activation vector quantized to int8 in 32-element groups.
/// Buffers are pre-allocated once (runner workspace) and reused per matvec —
/// zero allocation in the decode loop after warm-up.
pub struct QuantizedActivation {
    /// int8 quants, `len >= in_dim`.
    pub q: Vec<i8>,
    /// One f32 scale per 32-element group: `x ≈ scale * q`.
    pub scales: Vec<f32>,
    /// Sum of the 32 int8 quants per group (for offset formats: Q5_0's -16).
    pub sums: Vec<i32>,
    /// Sum of each 16-element half-group (for Q6_K's per-16 sub-scales).
    pub sums16: Vec<i32>,
    /// Number of valid elements from the last `quantize` call.
    pub len: usize,
}

impl QuantizedActivation {
    /// Pre-allocate for activations up to `max_len` elements (`% 32 == 0`).
    pub fn with_capacity(max_len: usize) -> Self {
        let groups = max_len / 32;
        QuantizedActivation {
            q: vec![0; max_len],
            scales: vec![0.0; groups],
            sums: vec![0; groups],
            sums16: vec![0; groups * 2],
            len: 0,
        }
    }

    /// Quantize `x` (length a multiple of 32, within capacity) into the
    /// pre-allocated buffers: per 32-group, `scale = max|x| / 127` and
    /// `q_i = round(x_i / scale)`.
    pub fn quantize(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len() % 32, 0);
        debug_assert!(x.len() <= self.q.len());
        self.len = x.len();
        for (g, group) in x.chunks_exact(32).enumerate() {
            let amax = group.iter().fold(0f32, |m, &v| m.max(v.abs()));
            if amax == 0.0 {
                self.scales[g] = 0.0;
                self.q[g * 32..g * 32 + 32].fill(0);
                self.sums[g] = 0;
                self.sums16[g * 2] = 0;
                self.sums16[g * 2 + 1] = 0;
                continue;
            }
            let scale = amax / 127.0;
            let inv = 127.0 / amax;
            let mut sum16 = [0i32; 2];
            for (i, &v) in group.iter().enumerate() {
                // Round half away from zero, branchless: `f32::round()` is a
                // libm call, and this loop runs for every element of every
                // activation vector in the decode loop.
                let scaled = v * inv;
                let qv = (scaled + 0.5f32.copysign(scaled)) as i32;
                self.q[g * 32 + i] = qv as i8;
                sum16[i / 16] += qv;
            }
            self.scales[g] = scale;
            self.sums[g] = sum16[0] + sum16[1];
            self.sums16[g * 2] = sum16[0];
            self.sums16[g * 2 + 1] = sum16[1];
        }
    }
}

/// True when the 256-bit EVEX VNNI dot (`vpdpbusd` on ymm) is available.
/// Detected once — this is AVX512VL+VNNI encoding-wise, but it is a 256-bit
/// datapath running at the AVX2 frequency license, so the X5 AVX-512
/// thermal ban does not apply (explicitly approved for use).
pub fn has_vnni_256() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// True when genuine 512-bit-wide AVX-512VNNI (`vpdpbusd` on zmm) is
/// available. Separate from [`has_vnni_256`] on purpose: this is the exact
/// datapath `gl-agent-skills/cpu-skills/rejected-optimizations.md` entry 3
/// closed for thermal/downclock reasons on the i3-1115G4 reference tier.
/// Detecting it is not the same as using it — see [`vnni512_enabled`].
pub fn has_vnni_512() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Opt-in gate for the 512-bit VNNI qdot path (`GLPROC_VNNI512=1`), default
/// **false**. This is a revisit of a closed entry in `gl-agent-skills/
/// cpu-skills/rejected-optimizations.md` (#3, "AVX-512F ... Includes 'at
/// least use AVX-512VNNI-512' — declined"), under JinXSuper's explicit
/// authorization to re-measure it — see `benches/vnni512_probe.rs`'s doc for
/// the isolated-kernel result that motivated this production A/B (VNNI-512
/// beat VNNI-256 by 20-26% GMAC/s in that probe, but the probe did not
/// measure the thermal/downclock question the rejection was actually about).
///
/// **Do not flip this default without a production glbench A/B (warm+cold,
/// decode AND prefill) that also checks `environment.hardware.cpu.thermal`
/// for throttling** — per the rejected-optimizations anti-pattern note, an
/// isolated probe alone is not sufficient grounds, and this project's own
/// probes have disagreed with production by 0.07x-2.40x before.
///
/// Cached in a `OnceLock`, unlike `q4k_native`'s deliberately-uncached env
/// lookup — that one is fine uncached because it is read only ~200 times per
/// model *load*. This function is read from inside `row_dot_q8`/
/// `row_dot_q8_packed8`, the per-row hot loop called millions of times per
/// token; an uncached `std::env::var` there was measured to cost far more
/// than any kernel-width effect (production decode dropped from ~37 to ~11
/// tok/s with the flag unset — see this session's A/B notes), so the lookup
/// itself must happen at most once per process.
pub fn vnni512_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        has_vnni_512() && matches!(std::env::var("GLPROC_VNNI512"), Ok(v) if !v.is_empty() && v != "0")
    })
}

/// Opt-in gate for the row-tiled qdot path (`GLPROC_ROW_TILE=1`), default
/// **false**. Different axis than [`vnni512_enabled`]: that one widened a
/// single row's instruction (256->512-bit) and came up neutral in
/// production — see `gl-agent-skills/cpu-skills/rejected-optimizations.md`
/// entry 3. This gates `q8_0::row_tile::row_tile_dot`, which instead tiles
/// R=8 *output rows* against one shared activation with 8 independent
/// accumulator chains, per `architecture/percival/CPU/ARTX02-IceLake.md`
/// Finding F05 (llama.cpp's real IceLake win is many independent chains
/// across rows, not a wider per-row dot). Isolated probe
/// (`benches/row_tile_probe.rs`) measured **2x GMAC/s** over the current
/// sequential dispatch — same anti-pattern warning applies: an isolated
/// probe alone is not sufficient grounds to trust in production, hence the
/// flag, hence the required A/B before flipping the default.
///
/// Cached in a `OnceLock` for the same reason `vnni512_enabled` is — this is
/// read from `threading::par_matvec_qdot`'s per-call hot path.
pub fn row_tile_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(std::env::var("GLPROC_ROW_TILE"), Ok(v) if !v.is_empty() && v != "0")
    })
}

/// One quantized weight row · Q8 activation, integer inner loop.
/// `fmt`-specific kernels; scalar is the parity ground truth for AVX2.
pub fn row_dot_q8(
    fmt: crate::kernels::bridge::QuantFormat,
    row: &[u8],
    act: &QuantizedActivation,
    strategy: SimdStrategy,
) -> f32 {
    use crate::kernels::bridge::QuantFormat;
    // SAFETY (both arms): strategy comes from SimdStrategy::detect(), so the
    // required CPU features are present. No AVX-512 qdot kernels yet — AVX2
    // covers both wide backends.
    match fmt {
        QuantFormat::Q5_0 => match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { q5_0::avx2::row_dot(row, act) },
            SimdStrategy::Scalar => q5_0::scalar::row_dot(row, act),
        },
        QuantFormat::Q8_0 => match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
                if vnni512_enabled() {
                    q8_0::vnni512::row_dot(row, act)
                } else if has_vnni_256() {
                    q8_0::vnni::row_dot(row, act)
                } else {
                    q8_0::avx2::row_dot(row, act)
                }
            },
            SimdStrategy::Scalar => q8_0::scalar::row_dot(row, act),
        },
        QuantFormat::Q6K => match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { q6_k::avx2::row_dot(row, act) },
            SimdStrategy::Scalar => q6_k::scalar::row_dot(row, act),
        },
        QuantFormat::Q4K => match strategy {
            SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { q4_k::avx2::row_dot(row, act) },
            SimdStrategy::Scalar => q4_k::scalar::row_dot(row, act),
        },
    }
}

thread_local! {
    /// Set only by `loader::load_gguf_with_gate` for the duration of one
    /// `load_gguf` call, to let a GATE-selected plan override the env-var
    /// default without threading a new parameter through every `weight()`
    /// call site. `None` = no override, fall through to the env var (every
    /// existing caller of `load_gguf` is unaffected). Thread-local, not a
    /// process-global `OnceLock`, so concurrent loads on different threads
    /// (as `q4k_e2e.rs`'s A/B test and any future multi-model host would
    /// do) cannot clobber each other's override.
    static GATE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Set this thread's `q4k_native()` override for the duration of `f`,
/// restoring the previous value (including `None`) afterward on every exit
/// path, including a panic unwind through `f`.
pub(crate) fn with_q4k_native_override<T>(value: Option<bool>, f: impl FnOnce() -> T) -> T {
    let previous = GATE_OVERRIDE.with(|cell| cell.replace(value));
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GATE_OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(previous);
    f()
}

/// Should the loader keep Q4_K tensors native (Wave 3) instead of repacking
/// them to Q8_0?
///
/// Default: **false** — opt in with `GLPROC_Q4K_NATIVE=1` (still needs AVX2).
///
/// The kernel is correct (E2E: top-1 identical, top-5 5/5 vs the repack path;
/// isolated probe: per-MAC parity with Q8_0). But routing dense gate/up
/// weights through it **loses ~30% end-to-end**, and the reason is not the
/// kernel — it is that native Q4_K forces `GateUp::Split`, giving up the
/// **fused SwiGLU** path (`par_matvec_swiglu`: one dispatch, gate+up
/// interleaved into a single DRAM stream, SiLU inline). Measured on
/// Qwen2.5-1.5B-q4_k_m decode, gate_up:
///
/// | path                    | GMAC/s | %ceiling |
/// |-------------------------|--------|----------|
/// | Q8_0 repack, fused      | 19.6   | 86%      |
/// | Q4_K native, un-fused   | 7.5    | 13%      |
///
/// Same weights, half the throughput — from losing fusion, not from the dot.
/// A fused Q4_K SwiGLU kernel would close this; that is Wave-4 work. Until it
/// exists, repack-to-Q8_0 stays the production default.
///
/// **`GLPROC_Q4K_NATIVE` is superseded whenever a caller goes through
/// `GlprocEngine::load_model`.** That path always calls
/// `gate::resolve_prefer_q4k_native` and sets
/// [`with_q4k_native_override`]'s thread-local before loading a single
/// tensor — GATE's session-init answer is checked first and, when set,
/// wins unconditionally over the env var below. The env var still works
/// exactly as documented for any caller that reaches [`crate::loader::
/// load_gguf`] directly (bypassing `GlprocEngine`, e.g. `q4k_e2e.rs`'s A/B
/// test) — this is not a bug, it is GATE actually being the decision-maker
/// now that one exists (see `architecture/GATE/GATE-algorithm.md`).
///
/// Deliberately NOT cached in a OnceLock: consulted only at load time (~200
/// getenv calls per model), and the E2E test flips it between two loads in one
/// process.
pub fn q4k_native() -> bool {
    let wide = matches!(
        SimdStrategy::detect(),
        SimdStrategy::Avx2 | SimdStrategy::Avx512
    );
    if let Some(gate_choice) = GATE_OVERRIDE.with(|cell| cell.get()) {
        // GATE's answer still respects the hardware gate: it selected
        // between two measured tok/s numbers, not between "native works"
        // and "native doesn't" — a non-wide backend can't run the native
        // kernel regardless of what GATE picked.
        return gate_choice && wide;
    }
    match std::env::var("GLPROC_Q4K_NATIVE") {
        Ok(v) if !v.is_empty() && v != "0" => wide, // opt-in, still needs AVX2
        _ => false,
    }
}

/// True if `fmt` should be consumed through the integer-dot path.
///
/// Q4_K is **excluded on purpose, despite having a working kernel**
/// ([`q4_k`]). The loader repacks it to Q8_0 instead, and that is the faster
/// choice — counter-intuitively, since the repack inflates per-token DRAM
/// traffic 1.70x on a real Q4_K model.
///
/// Measured on Qwen2.5-1.5B-q4_k_m (75.7% Q4_K by weight), decode:
///
/// | path                  | tok/s (3 runs)      |
/// |-----------------------|---------------------|
/// | repack to Q8_0        | **14,1 · 14,2 · 14,1** |
/// | native Q4_K integer-dot | 9,4 · 9,6 · 9,5   |
///
/// A 33% regression, with no overlap between the groups. The reason is in the
/// kernel, not the loader: isolated on identical work, Q4_K runs at **1.5–2.0
/// GMAC/s against Q8_0's 3.3** — 1.7–2.2x slower per MAC. Crucially the gap is
/// **the same when the data is L2-resident**, so it is not a memory effect: the
/// nibble unpack genuinely costs more compute than the bytes it saves. Q4_K
/// reaches only 0.8–1.1 GB/s where Q8_0 sustains 3.5, so it never gets close to
/// being bandwidth-bound in the first place.
///
/// This is the exact failure mode ARTX04 warned about — quantization only pays
/// "asalkan overhead dequantisasi tidak melebihi penghematan bandwidth". Here it
/// does. See `benches/q4k_probe.rs` for the measurement.
///
/// The kernel is kept (parity-tested, correct) so a future AVX-512 / VNNI-512
/// path, or a wider unpack, can be evaluated against a working baseline rather
/// than written from scratch.
pub fn supports(fmt: crate::kernels::bridge::QuantFormat) -> bool {
    !matches!(fmt, crate::kernels::bridge::QuantFormat::Q4K)
}

/// One Q8_0 row · a packed panel of 8 activations (quants `[block][act][32]`,
/// scales `[block][act]`). Caller must have checked `fmt == Q8_0` and a wide
/// strategy — this only dispatches between the VNNI and AVX2 kernels.
///
/// # Safety-relevant contract
/// `strategy` must be a wide backend from `SimdStrategy::detect()`.
pub fn row_dot_q8_packed8(row: &[u8], pq: &[u8], ps: &[f32]) -> [f32; 8] {
    // SAFETY: only called on wide backends (AVX2+FMA+F16C present); the
    // VNNI branches additionally check vnni_256/vnni512.
    unsafe {
        // vnni512's packed8 requires an even block count (see its doc) —
        // every real model dimension this project has loaded satisfies
        // that, but fall back rather than risk an out-of-bounds slice on
        // a future odd shape instead of asserting production down.
        if vnni512_enabled() && (row.len() / 34) % 2 == 0 {
            q8_0::vnni512::row_dot_packed8(row, pq, ps)
        } else if has_vnni_256() {
            q8_0::vnni::row_dot_packed8(row, pq, ps)
        } else {
            q8_0::avx2::row_dot_packed8(row, pq, ps)
        }
    }
}

/// One quantized weight row · `G` Q8 activations — the batched-prefill
/// fast path. Q8_0 on a wide backend shares the weight-side work across
/// the group; every other format/backend combination falls back to single
/// dots (correct, just unamortized). `G` must stay ≤ 8 so the wide kernels'
/// accumulators fit the 16 ymm registers.
pub fn row_dot_q8_xn<const G: usize>(
    fmt: crate::kernels::bridge::QuantFormat,
    row: &[u8],
    acts: [&QuantizedActivation; G],
    strategy: SimdStrategy,
) -> [f32; G] {
    use crate::kernels::bridge::QuantFormat;
    if matches!(fmt, QuantFormat::Q8_0)
        && matches!(strategy, SimdStrategy::Avx2 | SimdStrategy::Avx512)
    {
        // SAFETY: strategy comes from SimdStrategy::detect(), so AVX2/FMA/
        // F16C are present; the VNNI branch additionally checks vnni_256.
        unsafe {
            return if has_vnni_256() {
                q8_0::vnni::row_dot_xn::<G>(row, acts)
            } else {
                q8_0::avx2::row_dot_xn::<G>(row, acts)
            };
        }
    }
    std::array::from_fn(|g| row_dot_q8(fmt, row, acts[g], strategy))
}
