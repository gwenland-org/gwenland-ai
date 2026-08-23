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
//! researched stubs with real parameter shapes and a capability registry.
//! `optim` and `checkpoint` shipped after that: OPAdamW and CPLora are full,
//! and OPLion/OPAdafactor/OPAdamW8bit/CPFull/CPIncremental/CPSharded remain
//! registered stubs. Design notes for all of them live in
//! `gl-agent-skills/gltrain-m2-skills/`.

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
pub use checkpoint::{
    CPFull, CPIncremental, CPLora, CPSharded, CheckpointRegistry, CheckpointStore, ENSegment,
    ENTensorEntry, ENVersionCompatibility, Exporter, PLGgufMerge, VLCheckpoint, VLCheckpointFormat,
    VLFormatVersion, VLManifest, VLValidation,
};
pub use error::{GlTrainError, Result};
pub use nn::{
    trainable_parameters, trainable_parameters_mut, ABLinear, Adapter, AdapterRegistry,
    ENSkillStatus, LRDora, LRLoCon, LRLoHa, LRLora, LRQLora, LRVeRA, Module, TPParameter,
    VLAdapterCapability, VLAdapterSpec, VLLoraConfig,
};
pub use optim::{
    ENAdafactorMoment, ENOptimizerStateShape, OPAdafactor, OPAdamW, OPAdamW8bit, OPAdamWMoments,
    OPLion, Optimizer, OptimizerRegistry, VLAdamWConfig, VLLionConfig, VLNamedTensor,
    VLOptimizerCapability, VLOptimizerSpec, VLParamGroup,
};
pub use rng::Xorshift64Star;
pub use tensor::{Backend, Tensor};
pub use train::{
    mse_loss, ABByteTokenizer, ABGllmTokenizer, DTChatMl, ENChatRole, StepObserver, Tokenizer,
    Trainer, VLBatch, VLChatMlConfig, VLChatSample, VLChatTurn, VLMicroDataset, VLTokenizedSample,
    VLTrainerConfig, VLTrainingStep, IGNORE_INDEX,
};
