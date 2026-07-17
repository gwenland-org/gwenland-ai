# GDTQP — Gwen Deret-Taylor Quantizer Parser

**Project:** GwenLand — Rust-Native AI Inference Toolkit  
**Author:** JinXSuper  
**Version:** 0.1 (Concept Draft) · June 2026  
**License:** [CC BY-NC-ND 4.0](LICENSE)

---

## Documents

- [EXPLANATION.md](EXPLANATION.md) — What GDTQP is and why it exists, explained in plain language.
- [FORMULA.md](FORMULA.md) — Full technical documentation: formulas, derivations, implementation plan, and open questions.

---

## Disclaimer

**This is a pre-implementation theoretical framework. No empirical validation has been performed.**

The formulas, constants, and architectural decisions described in these documents have not yet been tested against real model perplexity benchmarks. All claims about quality-per-bit improvement, density mapping behaviour, and runtime cost are derived from mathematical reasoning — not from measured results.

Specifically:

- GDTQP has **not** been validated against any llama.cpp or other inference baseline.
- The Gamma-based density mapping is theoretically motivated but **not yet proven** to improve perplexity on real GGUF models.
- The implementation plan, phase estimates, and proposed Rust APIs are **subject to change** as development progresses.
- Open questions around sign recovery, normalisation choice, and sub-block scale interaction remain **unresolved**.

Everything here should be treated as a working hypothesis. The math may be revised, the approach may be abandoned, or new findings may require significant changes to the design.

Validation against `llama.cpp` perplexity baselines on wikitext-2 is the critical next step. Until that is complete, do not use GDTQP in any production or evaluation context.

---

*"Your machine. Your models. Your rules."*  
— GwenLand v1.0 design philosophy
