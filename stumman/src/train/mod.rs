//! Stummañ Deskiñ: the training loop and the data it runs on.
//!
//! [`Trainer`] closes the loop M1 and M2 built the pieces for: forward through
//! a frozen base plus a LoRA adapter, MSE against a target, backward through
//! the tape, one AdamW step, checkpoint. [`VLMicroDataset`] is the small
//! synthetic task M2's exit criterion is measured on.
//!
//! Deskiñ is Breton for "to learn". The five sub-systems M1 named (Kevrin,
//! Karg, Kevskrid, Gwiskadur, Gwellaer, Pik) each own a piece; this module owns
//! the order they run in, which is where KL-006 lives. See
//! [`Trainer::train_step`].

pub mod dataset;
pub mod trainer;

pub use dataset::VLMicroDataset;
pub use trainer::{mse_loss, Trainer, VLTrainerConfig};
