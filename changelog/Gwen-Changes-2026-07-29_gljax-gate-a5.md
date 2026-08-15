**Type:** Gate A5 — gljax runs a real model and produces coherent English.
**Status:** ⭐ **PASSED.** `"The capital of France is"` →
`" Paris. It is the largest city in France and the second largest in Europe…"`
Measured in CI on `Qwen/Qwen2-0.5B`.

---

## Executive Summary

gljax went from "compiles and emits valid IR" to "loads a real checkpoint and
writes readable English". The gate is closed on coherence.

Three real bugs were found, and **every one of them was found by running the
thing**, not by a test:

1. `rope_theta` is **1e6**, not the Llama-family 1e4.
2. Weights are HuggingFace **`[out_features, in_features]`**, not `[in, out]`.
3. `glcore`'s HF `tokenizer.json` loader dropped `added_tokens` from the
   vocabulary — so no modern `tokenizer.json` could load at all.

A fourth finding is about the tests rather than the code: glcore's
"14 vocabulary families exact" guarantee covers the **GGUF** loader only.

## The result

CI run `30433602366`, model pinned at revision
`91d2aff3f957f99e4c74c962f2f408dcc88a18d8`:

```
prompt encodes to 5 tokens: [785, 6722, 315, 9625, 374]

tokens  bucket    wall_s     tok/s  verdict  output
     8     128       6.8      1.18  PASS     " Paris. It is the largest city in"
    16     128      12.6      1.27  PASS     " Paris. It is the largest city in France and the
                                              second largest in Europe."
    32     128      25.1      1.28  PASS
    64     128      50.2      1.28  PASS
   128     256     190.9      0.67  PASS

compile + weight upload: 1.6 s per bucket
```

The first token is ` Paris`. The chain that produced it: `dlopen` →
`GetPjrtApi` → StableHLO text → XLA compile → safetensors weights → tokenizer
→ argmax → text.

## ⛔ Bug 1 — `rope_theta` is 1e6

`Qwen2Config::qwen2_0_5b()` shipped with `rope_base: 10_000.0`.
`Qwen/Qwen2-0.5B/config.json` says `"rope_theta": 1000000.0`.

A hundred times off. No shape error, no crash — every position rotated by the
wrong angle, with the damage growing further into the sequence. It would have
been blamed on the hand-written vtable or on the NeoX half-split, both of
which were fine.

Found by *reading the published config* rather than assuming it. That is why
`model/hf_config.rs` now exists: hyperparameters are **read, never defaulted**,
and a missing field is refused with a message explaining why, because a silent
default is exactly what produces this failure.

## ⛔ Bug 2 — weights are `[out, in]`

PyTorch's `nn.Linear` stores its weight transposed relative to the maths: the
forward is `x @ W.T`, and safetensors stores `W`. The checkpoint binder from
Wave A4 caught it on first load:

```
model.layers.0.mlp.gate_proj.weight: trace wants [896, 4864],
  checkpoint has [4864, 896] (transposed — same element count, different layout)
```

120 tensors reported.

⭐ **The square projections hid it.** `q_proj` and `o_proj` are both
`[896, 896]`, so they matched on shape while being equally wrong — 48 more
tensors that no shape check could ever flag. Had the FFN happened to be square
too, the model would have loaded cleanly and produced fluent garbage.

That is precisely why the binder reports *every* disagreement rather than the
first, and why it names transposition specifically. That design turned a
numerics hunt into a two-line diagnosis.

Fixed by `ops::linear`, contracting rhs axis 1 instead of 0. `dot_general` can
contract whichever axis it is told to, so reading the weight as stored costs
nothing; materialising `W.T` for 24 layers would be pure waste.

## ⛔ Bug 3 — glcore dropped `added_tokens`

```
tokenizer: eos id 151645 is outside a vocabulary of 151643
```

`Vocab::from_hf_json` built `id_to_token` from `model.vocab` (151,643 entries)
and then read `added_tokens` for their **ids only**, discarding `content`. The
three Qwen2 specials at 151643..=151645 ended up registered but textless, so
the vocabulary never grew, every added id was out of range, and
`<|endoftext|>` — the EOS of every Qwen2 base model — could neither be encoded
nor decoded.

⚠️ **This path had no test coverage.** `tokenizer_parity.rs`'s 14-vocabulary
reference suite exercises `from_gguf_path` only, so "14 vocabulary families
exact" was never a statement about `from_hf_json`. Its only other caller is
`glcore/src/runtime.rs:68`.

Same shape as the round-trip lesson: a suite that reads as comprehensive
covered one loader while the other was broken for essentially every modern
model. **When quoting the parity guarantee, say which loader.**

⚠️ Left unfixed on purpose: `eos_id = added.last()` is positional and picks
`<|im_end|>` (chat) over `<|endoftext|>` (base). `tokenizer.json` has no
unambiguous EOS field so it cannot be resolved there, and keying off token
*names* would repeat the 13-of-24 pre-tokenizer table mistake. Resolved at the
caller instead: `gljax::runtime::hf` treats `config.json`'s `eos_token_id` as
authoritative and logs the disagreement.

## ⚠️ Throughput: I was 5.4× pessimistic

Measured **163 GFLOP/s**, consistent across buckets 128 and 256
(151/163/163/163/174). I had estimated 15–60 GFLOP/s from "4 vCPUs" and
predicted the 7-length sweep would take 6.9 h against GitHub's 6 h limit. The
real figure is **~76 min**.

The *shape* of the argument held — with no KV cache the cost is superlinear in
`n`, because the bucket grows with `n` — but the absolute numbers did not.
⭐ Do not quote a FLOP/s figure for a machine without measuring it.

The cost model that did hold: Qwen2-0.5B is **494 M MACs/position** (357.8 M
across 24 layers + 136.1 M for the lm_head), so a forward is 128 GFLOP at
bucket 128 and 1102 GFLOP at bucket 1024.

## ⛔ Not via `glconv`, contrary to the brief

The brief's step 2 was `glconv --input <hf dir> --output ... --format gllm`.
`glconv`'s real CLI is `glconv <input.gguf> <output_dir>` — **GGUF input
only**, positional arguments. A HuggingFace download is safetensors, so that
path does not exist.

It is also unnecessary: `glcore::format::SafetensorsFile` already reads
safetensors and `GllmTokenizer::from_hf_json_path` already reads
`tokenizer.json`. Converting would have added a `glictus-caliburni`
dependency and a lossy step for nothing.

## What the gate does and does not prove

✅ The whole stack runs: FFI, IR emission, the Qwen2 graph, real weights, the
tokenizer, sampling. The output is on-topic English with the right first token.

⛔ It is **not** a numerical claim. Coherence is a strong signal, not a proof;
token-for-token agreement with glproc is ARTX12 Part B.

⚠️ The coherence check therefore tests **content**, not bytes: every failure
this gate exists to catch — wrong RoPE base, a transposed weight, an
off-by-one sampled row — still produces valid UTF-8 that looks like words.
"Non-empty and decodes" would have passed all three of the bugs above. So it
also requires the output to mention Paris/France/French and rejects repetition
loops, the signature of a broken position encoding.

## Still open

* **No KV cache.** `generate` re-runs the whole padded sequence per token —
  correct by construction, and the reason 512 tokens costs an hour instead of
  seconds. This is now the single biggest lever.
* **Bucket 2048 still does not trace** (dense O(S²) causal mask).
* `PJRT_Executable_Serialize`/`DeserializeAndLoad` bound but not wrapped, so
  the compile cache is written and never read.
* Sampling is argmax only; MoE is `unimplemented!()`.
