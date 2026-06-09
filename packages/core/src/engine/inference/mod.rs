// engine/inference/mod.rs — Native candle-transformers inference pipeline.
//
// Module layout:
//   loader         — load GGUF weights + tokenizer from disk, auto-detect device
//   model_dispatch — map GGUF architecture metadata to the correct candle model
//   params         — InferParams struct and validation
//   sampler        — temperature / top-p / repetition-penalty token sampling
//   runner         — main generation loop with RAM guard and SSE streaming

pub mod arch_detect;
pub mod backend;
pub mod config;
pub mod loader;
pub mod model_dispatch;
pub mod params;
pub mod registry;
pub mod runner;
pub mod sampler;

#[cfg(feature = "mistralrs-backend")]
pub mod mistralrs_backend;

#[cfg(feature = "candle-backend")]
pub mod candle_ggqr;
#[cfg(feature = "candle-backend")]
pub use candle_ggqr::ModelConfig;

pub mod selector;

pub use arch_detect::detect_architecture;
pub use backend::InferenceBackend;
pub use config::InferenceConfig;
pub use params::InferParams;
pub use selector::select_backend;

#[cfg(feature = "mistralrs-backend")]
pub use mistralrs_backend::MistralRsBackend;

#[cfg(any(test, feature = "test-utils"))]
pub mod mock_backend;
#[cfg(any(test, feature = "test-utils"))]
pub use mock_backend::MockBackend;
pub use registry::BackendRegistry;
