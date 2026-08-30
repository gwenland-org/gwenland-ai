//! Stummañ Gwiskadur: the Module contract.
//!
//! A module is something with a forward pass and a set of named parameters.

use crate::autograd::tape::Tape;
use crate::error::Result;
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// One layer, or a tree of them.
///
/// Traits take no prefix (naming rule 2).
///
/// # Why `forward` takes the tape explicitly
///
/// A parameter has to re-register with the tape on every step, because
/// `Tape::clear()` drops registrations between steps. Passing the tape in means
/// the module does that at the one moment it can be sure the tape is live. The
/// alternative, storing an `Arc<Mutex<Tape>>` inside every module, would make a
/// module's parameters silently belong to whichever tape happened to be current
/// when it was constructed, which is the failure KL-002 exists to prevent.
pub trait Module<B: Backend> {
    /// Run the forward pass, recording onto `tape`.
    fn forward(&self, x: &Tensor<B>, tape: &Arc<Mutex<Tape>>) -> Result<Tensor<B>>;

    /// Every parameter this module owns, trainable or frozen, in a stable order.
    ///
    /// # Duplicates must be deduped by the implementor
    ///
    /// VeRA (M3) shares one frozen `A`/`B` pair across every adapted layer, so a
    /// naive tree walk returns the same parameter many times. An optimizer that
    /// received it twice would apply the update twice, which is a silent 2x on
    /// the learning rate for exactly those parameters.
    /// [`trainable_parameters`] dedupes by name as a backstop, but an
    /// implementor that shares parameters should not rely on it.
    fn parameters(&self) -> Vec<&TPParameter<B>>;

    /// Mutable access, for the optimizer's update.
    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>>;

    /// Total trainable scalar count. Useful for a run's header line.
    fn trainable_param_count(&self) -> usize {
        trainable_parameters(self).iter().map(|p| p.n_elems()).sum()
    }
}

/// The trainable subset of a module's parameters, deduplicated by name.
///
/// Deduplication is by **name**, not by `TensorId`: two parameters that share a
/// name are the same logical weight by this crate's serialization rules, and
/// a checkpoint could not tell them apart anyway.
pub fn trainable_parameters<B: Backend, M: Module<B> + ?Sized>(
    module: &M,
) -> Vec<&TPParameter<B>> {
    let mut seen = BTreeSet::new();
    module
        .parameters()
        .into_iter()
        .filter(|p| p.is_trainable())
        .filter(|p| seen.insert(p.name().to_string()))
        .collect()
}

/// The trainable subset, mutably, deduplicated by name.
///
/// The optimizer's input. Dedup matters more here than on the immutable side:
/// a shared parameter returned twice would be *updated* twice in one
/// [`crate::optim::Optimizer::step`], which is a silent 2x on its learning
/// rate rather than merely a double count.
///
/// Rust's borrow checker cannot express "these `&mut` are distinct", so a
/// [`Module::parameters_mut`] implementor that genuinely aliases one parameter
/// could not compile in the first place. The dedup here is against an
/// implementor that returns two *different* `&mut` under the same name, which
/// is the shape a VeRA-style tree walk produces.
pub fn trainable_parameters_mut<B: Backend, M: Module<B> + ?Sized>(
    module: &mut M,
) -> Vec<&mut TPParameter<B>> {
    let mut seen = BTreeSet::new();
    module
        .parameters_mut()
        .into_iter()
        .filter(|p| p.is_trainable())
        .filter(|p| seen.insert(p.name().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// A module that deliberately returns the same parameter twice, standing in
    /// for VeRA's cross-layer sharing.
    struct SharingModule {
        shared: TPParameter<GlProc>,
        own: TPParameter<GlProc>,
        frozen: TPParameter<GlProc>,
    }

    impl Module<GlProc> for SharingModule {
        fn forward(&self, x: &Tensor<GlProc>, _: &Arc<Mutex<Tape>>) -> Result<Tensor<GlProc>> {
            Ok(x.clone())
        }
        fn parameters(&self) -> Vec<&TPParameter<GlProc>> {
            // `shared` appears twice, as a two-layer VeRA model would report it.
            vec![&self.shared, &self.own, &self.shared, &self.frozen]
        }
        fn parameters_mut(&mut self) -> Vec<&mut TPParameter<GlProc>> {
            vec![&mut self.shared, &mut self.own, &mut self.frozen]
        }
    }

    fn module() -> SharingModule {
        let mk = |n: usize| Tensor::<GlProc>::zeros(&[n, 1]).unwrap();
        SharingModule {
            shared: TPParameter::trainable("shared", mk(4)),
            own: TPParameter::trainable("own", mk(3)),
            frozen: TPParameter::frozen("frozen", mk(100)),
        }
    }

    #[test]
    fn trainable_parameters_drops_frozen_ones() {
        let m = module();
        let names: Vec<&str> = trainable_parameters(&m).iter().map(|p| p.name()).collect();
        assert!(!names.contains(&"frozen"));
    }

    #[test]
    fn trainable_parameters_dedupes_a_shared_parameter() {
        // Without this, an optimizer would apply the update to `shared` twice,
        // doubling its effective learning rate with no error anywhere.
        let m = module();
        let names: Vec<&str> = trainable_parameters(&m).iter().map(|p| p.name()).collect();
        assert_eq!(names.iter().filter(|n| **n == "shared").count(), 1);
        assert_eq!(names.len(), 2, "got {names:?}");
    }

    #[test]
    fn trainable_param_count_counts_each_shared_parameter_once() {
        // shared = 4, own = 3, frozen excluded.
        assert_eq!(module().trainable_param_count(), 7);
    }

    #[test]
    fn trainable_parameters_mut_drops_frozen_ones() {
        let mut m = module();
        let names: Vec<String> = trainable_parameters_mut(&mut m)
            .iter()
            .map(|p| p.name().to_string())
            .collect();
        assert!(!names.contains(&"frozen".to_string()));
    }

    /// The optimizer consumes this list. A duplicate here applies the update
    /// twice in one step, which is a silent 2x learning rate.
    #[test]
    fn trainable_parameters_mut_dedupes_a_shared_parameter() {
        let mut m = module();
        let names: Vec<String> = trainable_parameters_mut(&mut m)
            .iter()
            .map(|p| p.name().to_string())
            .collect();
        assert_eq!(names.iter().filter(|n| *n == "shared").count(), 1);
        assert_eq!(names.len(), 2, "got {names:?}");
    }
}
