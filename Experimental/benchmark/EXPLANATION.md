# GDTQP — What Is It and Why Does It Exist?

**Author:** JinXSuper  
**Project:** GwenLand — Rust-Native AI Inference Toolkit  
**Date:** June 2026  
**License:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## The Simple Version First

When you run an AI model locally, the model's weights are stored in a compressed format to save disk space and RAM. To actually use those weights, your software has to "decompress" them back into numbers the model can do math with. That decompression step is called **dequantisation**.

GDTQP is a new way to do that decompression step — one that's smarter about where it puts its precision.

---

## Why Does This Even Matter?

Here's the thing about AI model weights: they're not random. Almost all of the weight values in a large language model cluster tightly near zero. There are a few big outliers at the extremes, but the overwhelming majority of the action happens in a narrow band around the middle.

The problem is that the standard decompression method — used by virtually every LLM runtime including llama.cpp — doesn't know or care about this. It treats every possible value as equally likely and spreads its precision evenly across the entire range. That's like a photographer using the same focus setting for the foreground subject and the empty sky — technically covering everything, but wasting sharpness on the part that doesn't matter.

When you're working with a 4-bit or 2-bit quantised model, every bit of precision is precious. Wasting resolution on sparse regions of the weight distribution directly translates to worse model quality. The numbers come back a little wrong. Those errors stack up across millions of weights. Your model gets dumber.

---

## What Does GDTQP Do Differently?

GDTQP uses a mathematical function called the **Gamma function** to reshape how integer codes map to weight values.

Instead of spreading resolution evenly, the Gamma function naturally stretches the mapping around the dense central region and compresses it at the sparse extremes. The result: more precision where the weights actually live, less precision where they don't.

Think of it like a rubber band stretched between two pegs. Standard dequantisation stretches it evenly. GDTQP pulls the middle tighter, giving you finer gradations where you need them.

This isn't a hack or an approximation trick — it's a mathematically principled density-aware mapping. The Gamma function has exactly the growth rate behaviour you want for this problem.

---

## Okay, But the Gamma Function Is Slow. Won't This Kill Performance?

This is the clever part.

The Gamma function normally requires heavy computation — logarithms, exponentials, recursive calls. You absolutely cannot afford that in a hot loop processing millions of weights.

GDTQP sidesteps this entirely. Here's how: the quantised integer values in a GGUF file are small bounded integers — at most 64 distinct values for the most precise format. So instead of computing the Gamma function at runtime, GDTQP computes it for every possible input value **once, at compile time**, and bakes the results into a tiny lookup table.

At runtime, dequantising a weight is just: look up a pre-computed constant, do a couple of multiplications, done. The mathematical sophistication happens at build time, not inference time. The running code is just fast arithmetic.

The Stirling-Taylor approximation (a well-known technique from numerical mathematics) is what makes the compile-time computation accurate enough — the error is less than one part in ten billion for the ranges GDTQP uses.

---

## How Does It Fit Into GwenLand?

GwenLand already has three dequantisation modes:

**Standard** is the GGML-compatible mode — exactly what llama.cpp uses. Kept around for compatibility and as a correctness baseline.

**Euler** is GwenLand's existing custom mode — it uses a cosine projection to pin weights within a bounded range that the GwenTensor fixed-point accumulator needs. It's optimised for GwenLand's own inference pipeline.

**GDTQP** is the new third mode. Its goal is different from both: maximum weight recovery quality. It's aimed at cases where you want to export dequantised weights to a higher-precision format (like SafeTensors) or where you care deeply about the accuracy of the recovered floating-point values rather than inference speed.

Each mode exists for a reason. They're not competing — they serve different jobs.

---

## Is This Proven to Work?

Honestly, not yet. That's the point of the whitepaper being labelled a "concept draft."

The theory is solid — the Gamma function's behaviour really does match what you want for this problem, and the compile-time baking approach is genuinely zero-cost at runtime. But "theoretically better" and "actually better on real models" are different things, and GDTQP has not yet been validated against real perplexity benchmarks.

The planned validation is straightforward: run a standard model through llama.cpp, measure how well it predicts text (perplexity). Then run the same model through GDTQP dequantisation and measure again. If GDTQP perplexity is lower (better), the theory holds. If not, there's a bug or the theory has a flaw.

That step hasn't happened yet. GDTQP is an idea backed by solid math — not yet a proven improvement. The whitepaper is transparent about this.

---

## What Are the Remaining Unknowns?

A few things aren't fully resolved yet:

**Sign recovery.** The Gamma function only produces positive values. For formats that use negative integer codes to represent negative weights, GDTQP currently recovers the sign separately by multiplying by `sign(q)`. It's not yet proven whether this fully preserves the weight polarity that the original quantisation intended.

**Normalisation choice.** There are two reasonable ways to normalise the Gamma output — by the centre point (fast, per-element) or by the total sum (a true probability normalisation). The current design uses the centre-point approach, but this is still an open question.

**The last transcendental call.** After looking up the pre-computed `lnΓ` values, GDTQP still calls `exp()` once per element to convert back from log-space to linear. This can theoretically be eliminated by keeping everything in log-space throughout — but that requires changes to the GwenTensor accumulator, which is future work.

---

## Why Build This at All?

The framing behind GwenLand is "your machine, your models, your rules." Part of that means not just running models fast, but running them well — recovering the original researcher's intent as faithfully as possible from whatever compressed format the model ships in.

Standard dequantisation is fast and correct in a narrow sense: it's a faithful reconstruction of the GGUF encoding. But it's not optimal for what the encoding is trying to represent. GDTQP is an attempt to close that gap — to ask not just "what did the file say?" but "what did the original weights probably look like, and how do we get as close as possible with the bits we have?"

If the perplexity validation confirms the theory, it becomes a meaningful contribution to how local AI inference works. If it doesn't, the process of finding out why will teach something useful anyway.

---

## The One-Line Version

GDTQP is a smarter decompression method for AI model files that concentrates its limited precision where the weights actually are, rather than spreading it evenly — and it does this with zero runtime cost by baking all the hard math into the compiled binary ahead of time.

---

## License

**CC BY-NC-ND 4.0** — Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International

You are free to share this document with proper attribution. You may not use it for commercial purposes. You may not distribute modified versions.

Full license: https://creativecommons.org/licenses/by-nc-nd/4.0/

Copyright © 2026 JinXSuper × GwenLand. All rights reserved.
