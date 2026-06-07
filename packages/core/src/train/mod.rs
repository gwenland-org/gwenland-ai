// train/mod.rs — Training pipeline modules.
//
// Candle is now an unconditional dependency (no feature gate), so all
// submodules compile on every build.

pub mod config;
pub mod dataset;
pub mod dry_run;
pub mod lora;
pub mod native_runner;
pub mod progress;
pub mod runner;
pub mod samples;
pub mod script;
pub mod training_loop;
pub mod vram;
