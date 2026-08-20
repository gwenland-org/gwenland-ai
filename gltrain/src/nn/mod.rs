//! Stummañ Gwiskadur: the model sub-system.
//!
//! [`param`] is the named trainable tensor, [`module`] the forward/parameters
//! contract, [`linear`] the one concrete layer M2 needs, and [`adapter`] the
//! LoRA family plus its registry.
//!
//! The shape convention for every weight in this sub-system is stated in
//! [`linear`], and it is the transpose of PyTorch's. Read it before adding a
//! layer.

pub mod adapter;
pub mod linear;
pub mod module;
pub mod param;

pub use adapter::{
    Adapter, AdapterRegistry, ENSkillStatus, LRDora, LRLoCon, LRLoHa, LRLora, LRQLora, LRVeRA,
    VLAdapterCapability, VLAdapterSpec, VLLoraConfig,
};
pub use linear::ABLinear;
pub use module::{trainable_parameters, trainable_parameters_mut, Module};
pub use param::TPParameter;
