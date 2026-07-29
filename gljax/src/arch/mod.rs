//! `Architecture` — a model-shape descriptor (ARTX11 §4, "Wave A11.0").
//!
//! ⛔ **Scope note, read before extending this module.** ARTX11 §4 specifies
//! this as a retrofit that ultimately makes `gljax`'s tracing generic over
//! model family — `model::qwen2`'s `trace_forward`/`trace_prefill_with_cache`/
//! `trace_decode` would eventually be *driven by* an `Architecture` value
//! instead of hardcoding SwiGLU/plain-RMSNorm/`1/sqrt(head_dim)`. That
//! rewiring is **not done here**. `model::qwen2` is the one path Gate A5 has
//! actually run against a real PJRT plugin (CI runs `30447306245` and
//! `30453269580`); this machine has no PJRT plugin to re-verify a retrofit
//! against (see `gljax/README.md`), and P4 makes silently-wrong output the
//! bug class this whole project is organized around. Rewiring the verified
//! path blind, without a CI run to check it against, is a correctness gamble
//! this wave declines to take.
//!
//! What *is* here, and is real, additive, unit-tested capability:
//! - The descriptor types themselves (`Architecture`, `FfnKind`, `NormKind`,
//!   `AttentionKind`, `LayerPattern`, `RopeKind`, `EmbeddingKind`) — data,
//!   inspectable and diffable, exactly as ARTX11 §4.3 specifies ("a data
//!   descriptor, not a trait hierarchy" — variation here is configuration).
//! - The op-layer primitives Gemma's seven behavioral differences need:
//!   [`crate::ops::ffn::geglu_ffn`], [`crate::ops::norm::rms_norm_zero_centered`],
//!   [`crate::ops::attention::apply_qk_norm`],
//!   [`crate::ops::attention::gqa_attention_with_scale`],
//!   [`crate::ops::embedding::gather_embed_scaled`] — each independently unit
//!   tested against the same shape/op-count/numeric-pin style the rest of
//!   `ops/` uses.
//! - [`Architecture::from_qwen2`], expressing today's *actual* traced Qwen2
//!   behavior as a descriptor value — proof the descriptor can represent the
//!   one architecture that has run for real, not just a paper design.
//! - [`Architecture::arch_hash`] — every field, hashed. ARTX11 §4.3 files this
//!   under "joins the compile-cache key," so a Qwen draft and a Gemma target
//!   don't collide. That pairing needs the speculative-decoding session
//!   (ARTX11 §5–6, a `spec/` module) this sprint does not build — and without
//!   it there is exactly one `Architecture` loaded per process, so nothing
//!   can collide yet. `CompileKey` (`runtime/cache.rs`) already disambiguates
//!   every architecture difference that reaches the trace, because the
//!   difference shows up in the MLIR text `mlir_sha256` covers — verified by
//!   `arch_hash_differs_wherever_the_mlir_the_architecture_would_produce_differs`
//!   below, which is the honest substitute for "wire it into `CompileKey`
//!   today": proving the two hashes move together, not bolting an unused
//!   field onto a struct nothing yet produces two values for.
//!
//! Gemma parity itself — an actual Gemma checkpoint traced, compiled, and run
//! through PJRT — is **not claimed here**. ARTX11 §4.4 recommends validating
//! against "a real Gemma checkpoint"; this environment has neither a PJRT
//! plugin nor a downloaded Gemma checkpoint. [`Architecture::gemma3_shaped`]
//! exists for structural use (ARTX12's model zoo, `arch_hash` diffability
//! tests) and is explicitly documented as unverified against real weights.

use crate::model::qwen2::Qwen2Config;
use crate::runtime::digest::{hex, sha256};

/// A model's structure, independent of weights. ARTX11 §4.3: "a data
/// descriptor, not a trait hierarchy" — every field here is configuration
/// shared tracing code branches on, not a per-model implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct Architecture {
    /// Human-readable family name: `"qwen2"`, `"gemma3"`, etc. Not part of
    /// `arch_hash` — two configs with the same shape but different names
    /// should still collide (the name is a label, not a distinguishing
    /// property; see `feedback_key_lookup_tables_on_the_axis_that_varies`
    /// in this project's own history of exactly this mistake elsewhere).
    pub name: &'static str,
    pub ffn: FfnKind,
    pub norm: NormKind,
    pub attention: AttentionKind,
    pub rope: RopeKind,
    pub embedding: EmbeddingKind,
    /// Some models clamp final logits through `softcap * tanh(logits / softcap)`.
    /// `None` means no softcap — today's Qwen2 path.
    pub logit_softcap: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnKind {
    /// gate/up/down with SiLU on the gate branch. Qwen, LLaMA, Mistral.
    SwiGlu,
    /// gate/up/down with GeLU on the gate branch. Gemma.
    /// `tanh_approx` selects `gelu_pytorch_tanh` vs exact erf-based GELU —
    /// see [`crate::ops::ffn::geglu_ffn`]'s docs for why `false` is refused,
    /// not approximated.
    GeGlu { tanh_approx: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormKind {
    /// `out = x/rms(x) * weight`.
    RmsNorm { eps: f32 },
    /// `out = x/rms(x) * (1 + weight)` — Gemma's zero-centered convention.
    RmsNormZeroCentered { eps: f32 },
}

impl NormKind {
    pub fn eps(&self) -> f32 {
        match self {
            NormKind::RmsNorm { eps } | NormKind::RmsNormZeroCentered { eps } => *eps,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionKind {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// `None` -> `1/sqrt(head_dim)`. `Some(s)` -> Gemma's custom query
    /// pre-attention scalar, unrelated to `head_dim`.
    pub query_scale: Option<f32>,
    pub qk_norm: bool,
    /// Per-layer attention pattern. Uniform for Qwen; 5-local:1-global for
    /// Gemma 3/4.
    pub layer_pattern: LayerPattern,
    /// Extra RMSNorms Gemma places after attention and after the FFN (4 norms
    /// per block total, vs. Qwen's 2).
    pub post_norms: bool,
}

impl AttentionKind {
    /// `query_scale`, defaulting to `1/sqrt(head_dim)` when the descriptor
    /// doesn't override it.
    pub fn effective_query_scale(&self) -> f64 {
        self.query_scale
            .map(|s| s as f64)
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt())
    }

    /// How many query heads share each KV head.
    pub fn gqa_repeat(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerPattern {
    /// Every layer attends over the full cached window.
    Uniform,
    /// `period`-layer cycle; `global_at` (0-indexed within the cycle) is
    /// full-window causal attention, every other layer in the cycle is a
    /// `local_window`-wide sliding causal window. Gemma 3/4: `period=6`,
    /// `global_at=5` (5 local, then 1 global).
    ///
    /// ⚠️ **What this wave does NOT do.** ARTX11 §4.2 flags this as the most
    /// invasive difference: local layers only *need* a `local_window`-sized
    /// KV allocation, smaller than global layers' full window — but that
    /// saving is a slab-sizing change in ARTX5/ARTX7, and ARTX7 (continuous
    /// batching / slot accounting) does not exist in this codebase yet.
    /// `LayerPattern` here is data — inspectable, diffable, ready to drive a
    /// sliding-window *mask* over the existing full-size cache buffer (which
    /// is correct, just not memory-optimal) whenever tracing is rewired to
    /// consume it. The allocation-size optimization is real follow-up work,
    /// gated on ARTX7 existing at all.
    LocalGlobal {
        local_window: usize,
        period: usize,
        global_at: usize,
    },
}

impl LayerPattern {
    /// Whether layer `layer_idx` (0-indexed from the start of the model) is a
    /// global-attention layer.
    pub fn is_global(&self, layer_idx: usize) -> bool {
        match self {
            LayerPattern::Uniform => true,
            LayerPattern::LocalGlobal { period, global_at, .. } => {
                period > &0 && layer_idx % period == *global_at
            }
        }
    }
}

/// RoPE parameterization. Not one of ARTX11 §4.2's seven table rows, but
/// present in the descriptor (`Architecture::rope`) because Gemma 3 uses two
/// different `theta` values for local vs. global layers — a detail that pairs
/// naturally with [`LayerPattern::LocalGlobal`] and belongs in the same
/// descriptor. See the Gemma 3 Technical Report, cited in ARTX11's own
/// sources (`gljax/architecture/ARTX11-speculative-decoding.md` line 856).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeKind {
    /// One theta for every layer. Qwen, LLaMA, Mistral.
    Uniform { theta: f32 },
    /// A different theta for local vs. global layers. Gemma 3/4.
    LocalGlobal { local_theta: f32, global_theta: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingKind {
    /// Plain lookup. Qwen, LLaMA.
    Plain,
    /// Lookup scaled by `sqrt(hidden_size)` — Gemma's "scaled word embedding"
    /// (ARTX11 §4.2). See [`crate::ops::embedding::gather_embed_scaled`].
    ScaledBySqrtHidden,
}

impl Architecture {
    /// Expresses `Qwen2Config`'s actual traced behavior (`model::qwen2`) as
    /// an `Architecture` value — every field here matches what
    /// `trace_forward`/`trace_decode` really do today, not an aspiration.
    pub fn from_qwen2(cfg: &Qwen2Config) -> Self {
        Architecture {
            name: "qwen2",
            ffn: FfnKind::SwiGlu,
            norm: NormKind::RmsNorm { eps: cfg.rms_eps as f32 },
            attention: AttentionKind {
                n_heads: cfg.n_heads,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
                query_scale: None,
                qk_norm: false,
                layer_pattern: LayerPattern::Uniform,
                post_norms: false,
            },
            rope: RopeKind::Uniform { theta: cfg.rope_base },
            embedding: EmbeddingKind::Plain,
            logit_softcap: None,
        }
    }

    /// A Gemma-3-shaped descriptor, for structural use only (ARTX12's model
    /// zoo, `arch_hash` diffability tests). ⛔ **Not verified against a real
    /// Gemma checkpoint** — no PJRT plugin and no downloaded checkpoint exist
    /// in this environment. The field values follow the Gemma 3 Technical
    /// Report as cited in ARTX11 §"Sources"; they are believed correct, not
    /// measured correct.
    ///
    /// No `hidden_size` parameter: [`EmbeddingKind::ScaledBySqrtHidden`]'s
    /// scale factor is derived at trace time from the embedding table's own
    /// shape ([`crate::ops::embedding::gather_embed_scaled`] takes the scale
    /// as a plain `f64`), not stored redundantly in the descriptor.
    pub fn gemma3_shaped(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Architecture {
            name: "gemma3",
            ffn: FfnKind::GeGlu { tanh_approx: true },
            norm: NormKind::RmsNormZeroCentered { eps: 1e-6 },
            attention: AttentionKind {
                n_heads,
                n_kv_heads,
                head_dim,
                query_scale: Some(1.0 / (256f32).sqrt()),
                qk_norm: true,
                layer_pattern: LayerPattern::LocalGlobal {
                    local_window: 1024,
                    period: 6,
                    global_at: 5,
                },
                post_norms: true,
            },
            rope: RopeKind::LocalGlobal { local_theta: 10_000.0, global_theta: 1_000_000.0 },
            embedding: EmbeddingKind::ScaledBySqrtHidden,
            logit_softcap: None,
        }
    }

    /// A deterministic digest over every field that can change what gets
    /// traced. ARTX11 §4.3: "inspectable, serializable into the ARTX5
    /// compile-cache key, and diffable between draft and target."
    ///
    /// Uses [`crate::runtime::digest::sha256`] — the same primitive
    /// `CompileKey::digest` uses — rather than `std::hash::Hash`, so this
    /// follows the codebase's existing convention for anything that becomes
    /// part of a cache key (a `Hash`-based `u64` is not what `CompileKey`
    /// uses anywhere else, and mixing hashing schemes for the same purpose is
    /// its own source of confusion).
    ///
    /// `name` is deliberately excluded — see its field doc.
    pub fn arch_hash(&self) -> String {
        let mut buf = Vec::with_capacity(128);
        self.hash_ffn(&mut buf);
        self.hash_norm(&mut buf);
        self.hash_attention(&mut buf);
        self.hash_rope(&mut buf);
        self.hash_embedding(&mut buf);
        push_tagged_f32(&mut buf, 90, self.logit_softcap);
        hex(&sha256(&buf))
    }

    fn hash_ffn(&self, buf: &mut Vec<u8>) {
        match self.ffn {
            FfnKind::SwiGlu => buf.push(0),
            FfnKind::GeGlu { tanh_approx } => {
                buf.push(1);
                buf.push(tanh_approx as u8);
            }
        }
    }

    fn hash_norm(&self, buf: &mut Vec<u8>) {
        match self.norm {
            NormKind::RmsNorm { eps } => {
                buf.push(0);
                buf.extend_from_slice(&eps.to_le_bytes());
            }
            NormKind::RmsNormZeroCentered { eps } => {
                buf.push(1);
                buf.extend_from_slice(&eps.to_le_bytes());
            }
        }
    }

    fn hash_attention(&self, buf: &mut Vec<u8>) {
        let a = &self.attention;
        buf.extend_from_slice(&(a.n_heads as u64).to_le_bytes());
        buf.extend_from_slice(&(a.n_kv_heads as u64).to_le_bytes());
        buf.extend_from_slice(&(a.head_dim as u64).to_le_bytes());
        push_tagged_f32(buf, 91, a.query_scale);
        buf.push(a.qk_norm as u8);
        buf.push(a.post_norms as u8);
        match a.layer_pattern {
            LayerPattern::Uniform => buf.push(0),
            LayerPattern::LocalGlobal { local_window, period, global_at } => {
                buf.push(1);
                buf.extend_from_slice(&(local_window as u64).to_le_bytes());
                buf.extend_from_slice(&(period as u64).to_le_bytes());
                buf.extend_from_slice(&(global_at as u64).to_le_bytes());
            }
        }
    }

    fn hash_rope(&self, buf: &mut Vec<u8>) {
        match self.rope {
            RopeKind::Uniform { theta } => {
                buf.push(0);
                buf.extend_from_slice(&theta.to_le_bytes());
            }
            RopeKind::LocalGlobal { local_theta, global_theta } => {
                buf.push(1);
                buf.extend_from_slice(&local_theta.to_le_bytes());
                buf.extend_from_slice(&global_theta.to_le_bytes());
            }
        }
    }

    fn hash_embedding(&self, buf: &mut Vec<u8>) {
        match self.embedding {
            EmbeddingKind::Plain => buf.push(0),
            EmbeddingKind::ScaledBySqrtHidden => buf.push(1),
        }
    }
}

/// Pushes a tag byte then an f32's bytes for `Some`, or just a distinct tag
/// for `None` — so `None` and `Some(0.0)` cannot collide.
fn push_tagged_f32(buf: &mut Vec<u8>, tag: u8, value: Option<f32>) {
    match value {
        None => buf.push(tag),
        Some(v) => {
            buf.push(tag + 1);
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_qwen2_matches_the_actual_traced_defaults() {
        let cfg = Qwen2Config::qwen2_0_5b();
        let arch = Architecture::from_qwen2(&cfg);
        assert_eq!(arch.ffn, FfnKind::SwiGlu);
        assert_eq!(arch.norm, NormKind::RmsNorm { eps: 1e-6 });
        assert_eq!(arch.attention.query_scale, None);
        assert!(!arch.attention.qk_norm);
        assert_eq!(arch.attention.layer_pattern, LayerPattern::Uniform);
        assert_eq!(arch.embedding, EmbeddingKind::Plain);
        assert_eq!(arch.attention.gqa_repeat(), cfg.n_heads / cfg.n_kv_heads);
    }

    #[test]
    fn effective_query_scale_defaults_to_one_over_sqrt_head_dim() {
        let attn = AttentionKind {
            n_heads: 14,
            n_kv_heads: 2,
            head_dim: 64,
            query_scale: None,
            qk_norm: false,
            layer_pattern: LayerPattern::Uniform,
            post_norms: false,
        };
        assert!((attn.effective_query_scale() - 0.125).abs() < 1e-12);
    }

    #[test]
    fn effective_query_scale_honors_an_explicit_override() {
        let attn = AttentionKind {
            n_heads: 14,
            n_kv_heads: 2,
            head_dim: 64,
            query_scale: Some(0.0625),
            qk_norm: false,
            layer_pattern: LayerPattern::Uniform,
            post_norms: false,
        };
        assert_eq!(attn.effective_query_scale(), 0.0625);
    }

    #[test]
    fn uniform_layer_pattern_is_global_everywhere() {
        let p = LayerPattern::Uniform;
        for i in 0..12 {
            assert!(p.is_global(i), "layer {i}");
        }
    }

    #[test]
    fn local_global_pattern_matches_gemma3s_five_to_one_ratio() {
        // period=6, global_at=5 -> layers 5, 11, 17, ... are global; the rest local.
        let p = LayerPattern::LocalGlobal { local_window: 1024, period: 6, global_at: 5 };
        let global_layers: Vec<usize> = (0..18).filter(|&i| p.is_global(i)).collect();
        assert_eq!(global_layers, vec![5, 11, 17]);
        let local_count = (0..18).filter(|&i| !p.is_global(i)).count();
        assert_eq!(local_count, 15, "3 cycles of 6 = 18 layers, 3 global, 15 local");
    }

    #[test]
    fn arch_hash_is_stable_across_equal_descriptors() {
        let cfg = Qwen2Config::qwen2_0_5b();
        let a = Architecture::from_qwen2(&cfg);
        let b = Architecture::from_qwen2(&cfg);
        assert_eq!(a.arch_hash(), b.arch_hash());
    }

    #[test]
    fn arch_hash_differs_when_ffn_kind_differs() {
        let cfg = Qwen2Config::qwen2_0_5b();
        let swiglu = Architecture::from_qwen2(&cfg);
        let mut geglu = swiglu.clone();
        geglu.ffn = FfnKind::GeGlu { tanh_approx: true };
        assert_ne!(swiglu.arch_hash(), geglu.arch_hash());
    }

    #[test]
    fn arch_hash_ignores_the_name_field() {
        let cfg = Qwen2Config::qwen2_0_5b();
        let mut a = Architecture::from_qwen2(&cfg);
        let mut b = a.clone();
        a.name = "qwen2";
        b.name = "totally-different-label";
        assert_eq!(
            a.arch_hash(),
            b.arch_hash(),
            "name is a label, not a distinguishing property — see the field doc"
        );
    }

    /// A draft (Qwen-shaped) and a target (Gemma-shaped) descriptor at the
    /// exact same head geometry must still hash differently — this is
    /// literally the ARTX11 §4.3 collision scenario the hash exists to
    /// prevent, reproduced without needing the (not-yet-built) speculative
    /// session that would actually load both at once.
    #[test]
    fn qwen_and_gemma_shaped_descriptors_at_identical_head_geometry_hash_differently() {
        let mut cfg = Qwen2Config::qwen2_0_5b();
        cfg.n_heads = 8;
        cfg.n_kv_heads = 8;
        cfg.head_dim = 32;
        let qwen = Architecture::from_qwen2(&cfg);
        let gemma = Architecture::gemma3_shaped(8, 8, 32);
        assert_eq!(
            (qwen.attention.n_heads, qwen.attention.n_kv_heads, qwen.attention.head_dim),
            (gemma.attention.n_heads, gemma.attention.n_kv_heads, gemma.attention.head_dim),
            "the test setup must hold geometry fixed to isolate architecture"
        );
        assert_ne!(qwen.arch_hash(), gemma.arch_hash());
    }

    /// The honest substitute for wiring `arch_hash` into `CompileKey` today
    /// (see this module's top-level doc comment): every field `arch_hash`
    /// covers either already reaches the trace (so `CompileKey::mlir_sha256`
    /// already disambiguates it) or doesn't yet drive tracing at all (so
    /// there is nothing today for it to disambiguate). This pins the first
    /// half — differing precision-affecting behavior already produces a
    /// differing `mlir_sha256`, which is `different_precision_policies_get_different_cache_keys`'s
    /// exact pattern (`wave_a45_runtime.rs`), reproduced here for `eps`.
    #[test]
    fn norm_eps_difference_that_reaches_rms_norm_changes_the_emitted_mlir() {
        use crate::graph::TraceCx;
        use crate::ops::norm::rms_norm;
        use crate::stablehlo::types::{DType, Shape};

        let mut cx_a = TraceCx::new("main", "rms");
        let xa = cx_a.input("x", Shape::new([2, 8], DType::F32));
        let wa = cx_a.weight("w", Shape::new([8], DType::F32));
        let ya = rms_norm(&xa, &wa, 1e-6);
        let mlir_a = cx_a.finish(&[&ya]).mlir;

        let mut cx_b = TraceCx::new("main", "rms");
        let xb = cx_b.input("x", Shape::new([2, 8], DType::F32));
        let wb = cx_b.weight("w", Shape::new([8], DType::F32));
        let yb = rms_norm(&xb, &wb, 1e-5);
        let mlir_b = cx_b.finish(&[&yb]).mlir;

        assert_ne!(mlir_a, mlir_b, "a different eps must reach the emitted MLIR");
    }
}
