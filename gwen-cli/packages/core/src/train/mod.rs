// train/mod.rs — Training pipeline modules.
//
// Candle is now an unconditional dependency (no feature gate), so all
// submodules compile on every build.

pub mod checkpoint_resumer;
pub mod config;
pub mod layer_loader;
pub mod layered_training_loop;
pub use layer_loader::{LayerIndex, LayerLoader, LayerSlice, LoadedLayer};
pub use layered_training_loop::LayeredTrainingLoop;
pub mod dataset;
pub mod lora_bridge;
pub mod lora_cli;
pub mod lora_merger;
pub mod dry_run;
pub mod lora;
pub mod native_runner;
pub mod progress;
pub mod runner;
pub mod samples;
pub mod script;
pub mod transformer_layer;
pub mod training_loop;
pub mod vram;
