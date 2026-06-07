// engine/inference/mod.rs — Native candle-transformers inference pipeline.
//
// Module layout:
//   loader         — load GGUF weights + tokenizer from disk, auto-detect device
//   model_dispatch — map GGUF architecture metadata to the correct candle model
//   sampler        — temperature / top-p / repetition-penalty token sampling
//   runner         — main generation loop with RAM guard and SSE streaming

pub mod loader;
pub mod model_dispatch;
pub mod runner;
pub mod sampler;
