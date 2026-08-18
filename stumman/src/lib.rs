//! # Stummañ — GwenLand Training Framework
//!
//! Codename: Stummañ (Breton: "to train, to form")
//! Version: M1 Wave 1 — Core Tensor Abstraction
//!
//! Sub-systems:
//! - Kevrin    (tensor):     [`tensor`] module
//! - Karg      (backend):    [`backend`] module
//! - Kevskrid  (autograd):   [`autograd`] module
//! - Gwiskadur (model):      [`nn`] module
//! - Gwellaer  (optimizer):  [`optim`] module
//! - Pik       (checkpoint): [`checkpoint`] module
//! - Deskiñ    (training):   [`train`] module
//!
//! `nn::adapter` landed ahead of `optim`/`checkpoint`: LRLora is a full
//! implementation, and the other five adapters (DoRA/QLoRA/LoHa/VeRA/LoCon) are
//! researched stubs with real parameter shapes and a capability registry. The
//! `optim` and `checkpoint` sub-systems are documented in
//! `gl-agent-skills/stumman-m2-skills/` (implementation skills for AdamW, LoRA
//! integration, and checkpoints) rather than implemented yet in this tree.

pub mod autograd;
pub mod backend;
pub mod checkpoint;
pub mod error;
pub mod nn;
pub mod optim;
pub mod rng;
pub mod tensor;
pub mod train;

// Convenient top-level re-exports
pub use autograd::{NodeId, Tape, TensorId, VLGradStore};
pub use backend::{GlProc, SisdBackend};
pub use error::{GlTrainError, Result};
pub use nn::{
    trainable_parameters, trainable_parameters_mut, ABLinear, Adapter, AdapterRegistry,
    ENSkillStatus, LRDora, LRLoCon, LRLoHa, LRLora, LRQLora, LRVeRA, Module, TPParameter,
    VLAdapterCapability, VLAdapterSpec, VLLoraConfig,
};
pub use checkpoint::{
    CheckpointRegistry, CheckpointStore, ENSegment, ENTensorEntry, ENVersionCompatibility,
    Exporter, CPFull, CPIncremental, CPLora, CPSharded, PLGgufMerge, VLCheckpoint,
    VLCheckpointFormat, VLFormatVersion, VLManifest, VLValidation,
};
pub use optim::{
    ENAdafactorMoment, ENOptimizerStateShape, OPAdafactor, OPAdamW, OPAdamW8bit, OPAdamWMoments,
    OPLion, Optimizer, OptimizerRegistry, VLAdamWConfig, VLLionConfig, VLNamedTensor,
    VLOptimizerCapability, VLOptimizerSpec, VLParamGroup,
};
pub use rng::Xorshift64Star;
pub use tensor::{Backend, Tensor};
pub use train::{mse_loss, Trainer, VLMicroDataset, VLTrainerConfig};
