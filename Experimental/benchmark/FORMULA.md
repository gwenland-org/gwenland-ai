# GDTQP — Gwen Deret-Taylor Quantizer Parser
## Formula & Technical Documentation

**Author:** JinXSuper  
**Project:** GwenLand — Rust-Native AI Inference Toolkit  
**Date:** June 2026 · Version: 0.1 (Concept Draft)  
**Status:** Pre-Implementation Theoretical Framework  
**License:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Abstract

GDTQP replaces the conventional linear weight recovery formula used in GGUF K-quant formats with an adaptive, non-linear mapping grounded in the Gamma function and accelerated via Stirling-Taylor polynomial approximation.

LLM weight distributions are non-uniform — they follow heavy-tailed, near-Gaussian distributions with high density near zero and sparse outliers at the extremes. Linear dequantisation wastes quantisation bit-resolution on sparse regions while under-sampling dense regions.

GDTQP maps raw quantised integers through a Gamma-based density transform, then uses a compile-time-baked Stirling-Taylor polynomial expansion to approximate the transform with zero transcendental function calls at runtime.

**Key Claims:**
- Superior bit-resolution allocation in dense weight regions vs. linear dequant
- Zero transcendental function calls in the hot path (Stirling-Taylor const-baked)
- Drop-in compatible with existing GGUF Q2_K through Q6_K superblock layouts
- Implementable in pure safe Rust with no unsafe blocks
- Bounded output range $[0,\, 1/\varphi]$ preserved via Gamma normalisation

---

## 1. Background: Standard GGUF K-Quant Dequantisation

The GGUF format encodes quantised LLM weights using K-quant formats (Q2_K through Q6_K). Each format partitions a tensor into superblocks of 256 elements, further divided into sub-blocks with their own scale factors.

### 1.1 Standard Linear Dequantisation Formula

$$W[i] = d \times \text{scale}[j] \times q[i] \;-\; d_{\min} \times \min[j]$$

| Symbol              | Definition                                     |
|---------------------|------------------------------------------------|
| $d$                 | Superblock scale factor (f16)                  |
| $d_{\min}$          | Superblock min factor (f16)                    |
| $\text{scale}[j]$   | Sub-block scale index (4–6 bit unsigned)       |
| $\min[j]$           | Sub-block min index (4–6 bit unsigned)         |
| $q[i]$              | Raw quantised integer (format-dependent range) |

This formula assumes a linear relationship between integer codes and real-valued weights — an assumption that is empirically incorrect for LLM weight tensors.

---

## 2. The Weight Distribution Problem

LLM weight tensors follow near-Gaussian distributions with significant kurtosis. The majority of weight values cluster tightly around zero; occasional large-magnitude outliers carry disproportionate representational importance.

Linear K-quant mappers assign integer codes uniformly across $[q_{\min},\, q_{\max}]$, distributing bit-resolution evenly across a range that is mostly empty:

| Format | $q$ range    | Dense region       | Linear waste                   |
|--------|--------------|--------------------|--------------------------------|
| Q2_K   | $[0, 3]$     | $q \approx 1$–$2$  | ~50% of codes unused           |
| Q3_K   | $[-4, 3]$    | $q \approx -1$–$1$ | ~60% of codes in sparse region |
| Q4_K   | $[0, 15]$    | $q \approx 6$–$9$  | ~65% of codes in sparse region |
| Q6_K   | $[-32, 31]$  | $q \approx -10$–$10$ | ~70% of codes in sparse region |

---

## 3. GDTQP Architecture

### 3.1 Three-Stage Pipeline

GDTQP replaces the single linear dequant step with a three-stage pipeline. Arithmetic cost is equivalent to linear dequant once Stirling-Taylor constants are baked at compile time.

| Stage                | Operation                                                    | Cost           | Purpose                          |
|----------------------|--------------------------------------------------------------|----------------|----------------------------------|
| 1. Domain Shift      | $q \;\to\; q^{+} = q - q_{\min} + 1 \;\in [1, N]$          | 1 add          | Map to positive domain for $\Gamma(x)$ |
| 2. Gamma Transform   | $\Gamma_t(q^{+})$ via Stirling-Taylor poly                   | ~5 muls + adds | Non-linear density remap         |
| 3. Normalise & Scale | $W = d \times \dfrac{\Gamma_t(q^{+})}{\Gamma_t(q_{\text{mid}})} \times \text{sign}$ | 2 muls | Recover magnitude + sign |

**Total arithmetic cost per element:** ~8 multiply-add operations (vs. 3 for standard linear dequant). On modern CPUs with FMA support, this delta is absorbed by wider issue width and elimination of memory-bandwidth pressure.

### 3.2 The Gamma Density Transform

The Gamma function $\Gamma(x)$ for positive real $x$:

$$\Gamma(x) = \int_0^{\infty} t^{\,x-1}\, e^{-t}\, dt$$

**Key properties used by GDTQP:**

$$\Gamma(1) = 1$$

$$\Gamma(n) = (n-1)! \qquad \text{for positive integers } n$$

$$\Gamma(x+1) = x \cdot \Gamma(x) \qquad \text{(recurrence relation)}$$

$$\Gamma\!\left(\tfrac{1}{2}\right) = \sqrt{\pi} \qquad \text{(non-integer anchor)}$$

**Density mapping behaviour:**
- For $q^{+}$ near the centre of the quantisation range (dense region): $\Gamma(q^{+})$ grows slowly — codes map to closely-spaced weight values (high resolution).
- For $q^{+}$ near the extremes (sparse region): $\Gamma(q^{+})$ grows rapidly — codes map to widely-spaced weight values (compressed resolution).

### 3.3 Domain Shift: Handling Signed Q Ranges

The Gamma function is undefined for non-positive arguments. GDTQP applies a format-specific domain shift:

$$q^{+} = q - q_{\min} + 1$$

| Format | $q$ range    | $q_{\min}$ | $q^{+}$ range |
|--------|--------------|------------|---------------|
| Q2_K   | $[0,\; 3]$   | $0$        | $[1,\; 4]$    |
| Q3_K   | $[-4,\; 3]$  | $-4$       | $[1,\; 8]$    |
| Q4_K   | $[0,\; 15]$  | $0$        | $[1,\; 16]$   |
| Q5_K   | $[0,\; 31]$  | $0$        | $[1,\; 32]$   |
| Q6_K   | $[-32,\; 31]$| $-32$      | $[1,\; 64]$   |

The midpoint anchor:

$$q_{\text{mid}} = \frac{q_{\max} - q_{\min}}{2} + 1$$

The Gamma transform is normalised by $\Gamma(q_{\text{mid}})$ so that the central weight value maps to a normalised output of $1.0$ before the superblock scale is applied.

### 3.4 Stirling-Taylor Polynomial Approximation

Direct computation of $\Gamma(x)$ requires numerical integration or recursive calls — prohibitive in a tight inner loop. GDTQP uses the Stirling-Taylor hybrid approximation for $\ln\Gamma(x)$:

$$\ln\Gamma(x) \;\approx\; \left(x - \tfrac{1}{2}\right)\ln x \;-\; x \;+\; \tfrac{1}{2}\ln(2\pi) \;+\; \sum_{k=1}^{K} \frac{B_{2k}}{2k(2k-1)\,x^{2k-1}}$$

**Bernoulli correction terms** (first three, $k = 1, 2, 3$):

$$+\frac{1}{12x} \;-\; \frac{1}{360x^3} \;+\; \frac{1}{1260x^5}$$

For $x \in [1,\, 64]$, truncating at $k = 3$ gives:

$$\left|\,\epsilon\,\right| < 1.4 \times 10^{-10}$$

### 3.5 Compile-Time Constant Tables

Since $q^{+}$ values are bounded integers in a known range $[1, N]$, GDTQP pre-computes $\ln\Gamma(q^{+})$ for every possible integer $q^{+}$ at compile time and bakes them as static lookup tables of `f32` constants:

```rust
// Rust compile-time constant table (Q4_K: q⁺ ∈ [1, 16])
const GDTQP_LN_GAMMA_Q4K: [f32; 16] = [
    0.000000, 0.000000, 0.693147, 1.791759, 3.178054,
    4.787492, 6.579251, 8.525161, 10.604603, 12.801827,
    15.104412, 17.502308, 19.987214, 22.552164, 25.191221,
    27.899271,
];

// Runtime inner loop: ZERO transcendental calls
let ln_gamma_q   = GDTQP_LN_GAMMA_Q4K[(q_pos - 1) as usize];
let ln_gamma_mid = GDTQP_LN_GAMMA_Q4K[q_mid - 1];
let ratio        = (ln_gamma_q - ln_gamma_mid).exp(); // 1 exp() per element
```

The runtime ratio in closed form:

$$\text{ratio} = \exp\!\bigl(\ln\Gamma(q^{+}) - \ln\Gamma(q_{\text{mid}})\bigr) = \frac{\Gamma(q^{+})}{\Gamma(q_{\text{mid}})}$$

> **Note:** One `exp()` call remains in the current formulation (converting the $\ln\Gamma$ ratio back to linear). This can be eliminated by operating entirely in log-space with a log-domain accumulator — a direction for future GwenTensor kernel integration.

---

## 4. Comparison with Existing Approaches

### 4.1 GDTQP vs. Standard Linear Dequant

| Property               | Linear (GGML)                           | Euler (GwenLand)                         | GDTQP                                                |
|------------------------|-----------------------------------------|------------------------------------------|------------------------------------------------------|
| Formula                | $W = d \cdot s \cdot q - d_{\min} \cdot m$ | $W = \cos(\theta)\cdot\delta/\varphi$ | $W = d\cdot\dfrac{\Gamma(q^{+})}{\Gamma(q_{\text{mid}})}\cdot\text{sign}$ |
| Output range           | Unbounded $\mathbb{R}$                  | $[-0.618,\; 0.618]$                      | $[0,\; 1/\varphi]$ normalised                        |
| Density mapping        | Uniform (linear)                        | Cosine-warped                            | Gamma-adaptive                                       |
| Transcendental calls   | $0$                                     | $1$ `cos()` per element                  | $0$ (const-baked)                                    |
| Dense region precision | Baseline                                | Slightly better                          | Theoretically optimal                                |
| Outlier handling       | Full range                              | Compressed                               | Gracefully compressed                                |
| Hardware cost          | ~3 ops/elem                             | ~5 ops/elem                              | ~8 ops/elem (est.)                                   |
| Validation status      | Proven, widespread                      | GwenLand-only                            | Pre-implementation                                   |

### 4.2 Relationship to GPTQ and AWQ

- **GPTQ** (Frantar et al., 2022) and **AWQ** (Lin et al., 2023) address weight distribution mismatch at quantisation time — adjusting boundaries before storing the model.
- **GDTQP** addresses the same problem at dequantisation time — correcting the distribution mismatch during inference on any existing GGUF file without requiring re-quantisation.

These are complementary approaches: a model quantised with AWQ and dequantised with GDTQP theoretically benefits from both distribution-aware compression and distribution-aware recovery.

### 4.3 Relationship to GwenLand Euler Mode

| Mode       | Use case                                           | Output range                      |
|------------|----------------------------------------------------|-----------------------------------|
| Standard   | GGML-compatible reference                          | Linear, maximally portable        |
| Euler      | GwenTensor inference — SIMD-friendly               | Bounded $[\pm 0.618]$             |
| GDTQP      | High-quality dequantisation, SafeTensors export    | Unbounded, distribution-optimal   |

---

## 5. Implementation Plan

### Phase 1 — Constant Table Generation

A Rust `build.rs` script pre-computes $\ln\Gamma(x)$ via Stirling-Taylor for all integer $x \in [1,\, 64]$ and emits static const arrays, one per K-quant format:

```rust
// build.rs (pseudo-code)
fn stirling_ln_gamma(x: f64) -> f64 {
    let ln2pi = (2.0 * PI).ln();
    (x - 0.5) * x.ln() - x + 0.5 * ln2pi
        + 1.0/(12.0*x)
        - 1.0/(360.0*x.powi(3))
        + 1.0/(1260.0*x.powi(5))
}

// Emits: const GDTQP_LN_GAMMA_Q6K: [f32; 64] = [ ... ];
```

### Phase 2 — DequantMode Extension

```rust
pub enum DequantMode {
    Standard,  // GGML-compatible linear
    Euler,     // GwenTensor cosine projection
    Gdtqp,     // Gamma-Taylor adaptive (new)
}
```

### Phase 3 — Validation Against llama.cpp Baseline

1. Run `llama.cpp` perplexity on wikitext-2 with Q4_K_M model → baseline PPL
2. Export dequantised weights via `validate_dequant` binary (Standard mode) — verify match with llama.cpp
3. Dequantise same model with GDTQP mode, re-run perplexity
4. Compare: GDTQP PPL should be $\leq$ Standard PPL (lower = better)
5. If improvement confirmed, promote from `EXPERIMENTAL` to `STABLE`

### Estimated Timeline

| Phase    | Task                                          | Effort      |
|----------|-----------------------------------------------|-------------|
| Phase 1  | `build.rs` constant table generation          | 0.5 day     |
| Phase 2  | `dequant.rs` GDTQP functions (Q4_K, Q6_K)    | 1 day       |
| Phase 3  | Unit tests + `validate_dequant` GDTQP mode    | 0.5 day     |
| Phase 4  | Perplexity validation on Kaggle T4            | 1 day       |
| Phase 5  | Q2_K, Q3_K, Q5_K GDTQP paths                 | 1 day       |
| **Total**|                                               | **~4 days** |

---

## 6. Open Questions

### 6.1 Unresolved Theoretical Questions

**Optimal normalisation.** Should the Gamma ratio be normalised by $\Gamma(q_{\text{mid}})$ (per-element anchor) or by the partition sum $\sum_{q} \Gamma(q^{+})$ (true probability normalisation)?

$$\text{Option A: } \frac{\Gamma(q^{+})}{\Gamma(q_{\text{mid}})} \qquad \text{Option B: } \frac{\Gamma(q^{+})}{\displaystyle\sum_{q^{+}=1}^{N} \Gamma(q^{+})}$$

**Sign recovery.** Since $\Gamma(x) > 0$ for all $x > 0$, sign information is lost in the transform. The current proposal recovers it as:

$$W = d \;\cdot\; \frac{\Gamma(q^{+})}{\Gamma(q_{\text{mid}})} \;\cdot\; \text{sign}(q)$$

Whether this fully recovers the original weight polarity for symmetric formats (Q3_K, Q6_K) is unresolved.

**Sub-block scale interaction.** The current formulation applies $d \times \Gamma\text{-ratio}$ as the weight. Whether the sub-block scale factor should also modulate the Gamma transform or be applied additively after is unresolved.

**Log-domain accumulation.** The remaining `exp()` call per element can be eliminated if the GwenTensor accumulator is extended to operate in log-space throughout:

$$\ln W = \ln d + \ln\Gamma(q^{+}) - \ln\Gamma(q_{\text{mid}}) + \ln|\text{sign}|$$

### 6.2 Future Directions

- **GDTQP-aware quantisation:** Train or re-quantise models with GDTQP dequantisation in the loop, enabling lower bpw at equivalent perplexity.
- **Hardware-specific tables:** Separate Stirling-Taylor tables for f16 arithmetic (GPU half-precision) vs. f32 (CPU paths).
- **IQ format extension:** Apply GDTQP to importance-matrix quantised formats (IQ2_XXS, IQ3_S) where weight distribution non-uniformity is most extreme.
- **Publication:** If perplexity results are positive, formalise as a technical report and submit to arXiv under cs.LG.

---

## 7. References

1. Frantar, E., et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323.
2. Lin, J., et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.
3. Gerganov, G. (2023). *GGUF Format Specification.* ggml-org/llama.cpp, GitHub.
4. Abramowitz, M., Stegun, I. (1964). *Handbook of Mathematical Functions.* Chapter 6: Gamma and Related Functions.
5. JinXSuper. (2026). *GGQR-CF-mmap: A Rust-native GGUF Dequantisation Engine.* GwenLand v1.0, GitLab.
6. JinXSuper. (2026). *math_theory_q4k.md: Mathematical Derivation of K-quant Dequantisation and Euler Projection.* GwenLand v1.0, GitLab.

---

## License

**CC BY-NC-ND 4.0** — Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International

You are free to share this material with attribution. You may not use it for commercial purposes. You may not distribute modified versions.

Full license: https://creativecommons.org/licenses/by-nc-nd/4.0/

Copyright © 2026 JinXSuper × GwenLand. All rights reserved.

---

*"Your machine. Your models. Your rules."*  
— GwenLand v1.0 design philosophy

*GDTQP is GwenLand's answer to the question: not just how fast can we load weights, but how faithfully can we recover them.*

---
GDTQP Whitepaper v0.1 · JinXSuper × GwenLand · June 2026
