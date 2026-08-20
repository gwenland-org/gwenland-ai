//! Stummañ Gwiskadur: adapter parameterizations and their registry.
//!
//! # Why adapters are not a subtype tree
//!
//! The obvious shape is `Adapter -> {LoRA, LoRA+, DoRA, QLoRA, LoCon, LoHa,
//! VeRA}`. Reading the papers kills it (full write-up in `M2_RESEARCH.md` §7-A);
//! four of those seven are not adapter parameterizations at all:
//!
//! - **LoRA+** changes only the learning-rate ratio between the A and B
//!   parameter groups. Its architecture is byte-for-byte LoRA. It lives in
//!   [`crate::optim`] as a policy, and there is deliberately no `LRLoraPlus`.
//! - **QLoRA**'s adapter term is `+ X L1 L2`, which is plain LoRA. What changes
//!   is how the *base weight* is stored (NF4 + double quantization). It composes
//!   `BaseWeightSource x LRLora`, so it is a stub here only to hold the
//!   researched contract, not because it is a new parameterization.
//! - **DoRA** computes `m * (W0 + BA) / ||W0 + BA||_c`. That is not
//!   `base_out + delta_out`: it renormalizes the *combined* weight, so it needs
//!   the base weight's **values**, not the base layer's output. This is why
//!   [`Adapter::forward`] takes `&Tensor<B>` for the base weight rather than a
//!   base output. One signature choice, and DoRA stops needing a trait change.
//! - **VeRA** shares one frozen random `A`/`B` pair across *every* adapted
//!   layer, so its parameters are not owned per layer. See
//!   [`crate::nn::module::trainable_parameters`], which dedupes for that reason.
//!
//! The three that genuinely are alternative parameterizations of `ΔW` are LoRA,
//! LoHa and LoCon, plus VeRA's scaled-random form.

use crate::autograd::tape::Tape;
use crate::error::Result;
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub mod dora;
pub mod locon;
pub mod loha;
pub mod lora;
pub mod qlora;
pub mod vera;

pub use dora::LRDora;
pub use locon::LRLoCon;
pub use loha::LRLoHa;
pub use lora::{LRLora, VLLoraConfig};
pub use qlora::LRQLora;
pub use vera::LRVeRA;

/// Whether a registered skill actually computes, or only describes itself.
///
/// `EN` because a closed set of variants is this type's whole job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENSkillStatus {
    /// Implemented and tested.
    Full,
    /// Registered, introspectable, constructible, and refuses to compute.
    ///
    /// Carries the reason and the milestone that owns it, so the error a caller
    /// sees names the blocking work rather than saying "todo".
    Stub {
        /// What has to exist first, in one line.
        reason: &'static str,
        /// Milestone that owns the implementation.
        milestone: &'static str,
    },
}

impl ENSkillStatus {
    /// True when the skill will actually compute.
    pub fn is_full(&self) -> bool {
        matches!(self, ENSkillStatus::Full)
    }
}

/// Machine-readable facts about an adapter.
///
/// Every field records something the research established, so this is a
/// structured summary of `M2_RESEARCH.md` §5 rather than invented ceremony. The
/// three flags in the middle exist because each one is a property that a caller
/// would otherwise get wrong by assuming all adapters behave like LoRA.
#[derive(Debug, Clone)]
pub struct VLAdapterCapability {
    /// Registry key, e.g. `"lora"`.
    pub id: &'static str,
    /// Implemented, or a stub with a reason.
    pub status: ENSkillStatus,
    /// Trainable parameter count as an expression, for a run's header line.
    pub trainable_params: &'static str,
    /// Can `ΔW` be folded into the base weight for inference?
    pub mergeable: bool,
    /// Does the forward pass need the base weight's **values**?
    ///
    /// `false` for every additive adapter: they only add to the base output.
    /// `true` for DoRA, which normalizes by `||W0 + BA||_c`. A caller that
    /// streams base weights layer by layer and drops them must keep them
    /// resident when this is set.
    pub requires_base_values: bool,
    /// Does the forward pass materialize the full `[d_in, d_out]` delta?
    ///
    /// `false` for LoRA, whose whole memory argument is that it never forms
    /// `ΔW`. **`true` for LoHa**: `(B1A1) ⊙ (B2A2)` does not factor through `x`,
    /// so the full matrix has to exist. Anything sizing a memory budget from
    /// "adapters are low-rank, therefore cheap" is wrong for LoHa.
    pub materializes_delta: bool,
    /// Are parameters shared across layers rather than owned per layer?
    ///
    /// `true` only for VeRA. See the dedup note on
    /// [`crate::nn::module::Module::parameters`].
    pub shares_params_across_layers: bool,
    /// The paper this was implemented from.
    pub source: &'static str,
}

/// One adapter parameterization.
///
/// Traits take no prefix (naming rule 2). Object-safe, because
/// [`AdapterRegistry`] hands out `Box<dyn Adapter<B>>`.
pub trait Adapter<B: Backend> {
    /// The adapter's contribution, given the input and the **frozen base weight**.
    ///
    /// Returns the full layer output, not just the delta, because DoRA cannot
    /// express its result as a delta (see the module docs). An additive adapter
    /// computes `base_out + scale * delta` here and is none the worse for it.
    fn forward(
        &self,
        x: &Tensor<B>,
        base_weight: &Tensor<B>,
        tape: &Arc<Mutex<Tape>>,
    ) -> Result<Tensor<B>>;

    /// Parameters this adapter owns.
    fn parameters(&self) -> Vec<&TPParameter<B>>;

    /// Mutable parameter access, for the optimizer.
    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>>;

    /// Fold the adapter into `base_weight`, in place.
    fn merge_into(&self, base_weight: &mut Tensor<B>) -> Result<()>;

    /// This adapter's capability record.
    fn capability(&self) -> &'static VLAdapterCapability;
}

/// How to build one adapter.
///
/// Shared by every adapter so the registry can construct any of them from the
/// same call. Fields a given adapter ignores are documented on that adapter.
#[derive(Debug, Clone)]
pub struct VLAdapterSpec {
    /// Input dimension of the layer being adapted.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Rank.
    pub r: usize,
    /// LoRA alpha. Scaling is `alpha / r`, or `alpha / sqrt(r)` under rslora.
    pub alpha: f32,
    /// Use `alpha / sqrt(r)` instead of `alpha / r`.
    pub rslora: bool,
    /// Seed for the random half of the initialization.
    pub seed: u64,
}

impl VLAdapterSpec {
    /// A spec with the usual defaults: `alpha = r`, no rslora.
    pub fn new(d_in: usize, d_out: usize, r: usize, seed: u64) -> Self {
        Self {
            d_in,
            d_out,
            r,
            alpha: r as f32,
            rslora: false,
            seed,
        }
    }

    /// The scaling factor applied to the adapter delta.
    ///
    /// LoRA: "We then scale ΔWx by α/r, where α is a constant in r." PEFT adds
    /// `use_rslora`, which switches to `α/√r`. Both are offered because the two
    /// reference implementations disagree, and picking one silently would make
    /// a checkpoint's numbers untranslatable.
    pub fn scale(&self) -> f32 {
        if self.rslora {
            self.alpha / (self.r as f32).sqrt()
        } else {
            self.alpha / self.r as f32
        }
    }
}

type BuildFn<B> = fn(&VLAdapterSpec) -> Result<Box<dyn Adapter<B>>>;

/// Maps adapter ids to their capability record and constructor.
///
/// Copies the decisions in `glictus-caliburni/src/plugin.rs`, which solved this
/// already: registration **refuses** on a duplicate id rather than overwriting,
/// [`resolve`](Self::resolve) returns an `Option` and
/// [`require`](Self::require) an error, and `with_builtins` preloads what the
/// crate ships. Silent last-one-wins registration would make behaviour depend on
/// registration order.
pub struct AdapterRegistry<B: Backend> {
    entries: BTreeMap<&'static str, (&'static VLAdapterCapability, BuildFn<B>)>,
}

impl<B: Backend> Default for AdapterRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> AdapterRegistry<B> {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Every adapter this crate ships: one real, five stubs.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        // Built-in ids are distinct by construction, so these cannot collide.
        let builtins: [(&'static VLAdapterCapability, BuildFn<B>); 6] = [
            (lora::CAPABILITY, lora::build),
            (dora::CAPABILITY, dora::build),
            (qlora::CAPABILITY, qlora::build),
            (loha::CAPABILITY, loha::build),
            (vera::CAPABILITY, vera::build),
            (locon::CAPABILITY, locon::build),
        ];
        for (cap, build) in builtins {
            r.register(cap, build)
                .expect("built-in adapter ids are unique");
        }
        r
    }

    /// Add an adapter.
    ///
    /// Returns [`crate::error::GlTrainError::InvalidOp`] if the id is taken. Two
    /// implementations claiming one id is a wiring bug, and keeping the last
    /// registered would make behaviour depend on registration order.
    pub fn register(
        &mut self,
        capability: &'static VLAdapterCapability,
        build: BuildFn<B>,
    ) -> Result<()> {
        if self.entries.contains_key(capability.id) {
            return Err(crate::error::GlTrainError::InvalidOp(format!(
                "adapter '{}' is already registered; an id maps to exactly one implementation",
                capability.id
            )));
        }
        self.entries.insert(capability.id, (capability, build));
        Ok(())
    }

    /// Look up a capability record. `None` means nothing claims this id.
    pub fn resolve(&self, id: &str) -> Option<&'static VLAdapterCapability> {
        self.entries.get(id).map(|(cap, _)| *cap)
    }

    /// Look up a capability record, erroring when absent.
    pub fn require(&self, id: &str) -> Result<&'static VLAdapterCapability> {
        self.resolve(id).ok_or_else(|| {
            crate::error::GlTrainError::InvalidOp(format!(
                "unknown adapter '{id}'; registered: {}",
                self.ids().join(", ")
            ))
        })
    }

    /// Build an adapter.
    ///
    /// A stub constructs successfully and fails on `forward`. That is deliberate:
    /// the caller learns the difference between "no such adapter" (a typo, at
    /// construction) and "this adapter is not implemented yet" (a real request,
    /// at the point of use), and never gets a different adapter than it asked
    /// for.
    pub fn build(&self, id: &str, spec: &VLAdapterSpec) -> Result<Box<dyn Adapter<B>>> {
        let (_, build) = self.entries.get(id).ok_or_else(|| {
            crate::error::GlTrainError::InvalidOp(format!(
                "unknown adapter '{id}'; registered: {}",
                self.ids().join(", ")
            ))
        })?;
        build(spec)
    }

    /// Every registered id, sorted.
    pub fn ids(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    /// Ids that actually compute.
    pub fn implemented_ids(&self) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|(_, (cap, _))| cap.status.is_full())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Whether an id is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    /// Number of registered adapters.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// Scale is a single division, so it is exact for these inputs.
    const TOL_EXACT: f32 = 1e-9;

    fn registry() -> AdapterRegistry<GlProc> {
        AdapterRegistry::with_builtins()
    }

    #[test]
    fn all_six_researched_adapters_are_registered() {
        let r = registry();
        for id in ["lora", "dora", "qlora", "loha", "vera", "locon"] {
            assert!(r.contains(id), "'{id}' must be registered");
        }
        assert_eq!(r.len(), 6);
    }

    #[test]
    fn only_lora_is_implemented_in_m2() {
        assert_eq!(registry().implemented_ids(), vec!["lora"]);
    }

    #[test]
    fn there_is_no_lora_plus_adapter() {
        // LoRA+ is an optimizer policy, not a parameterization. Registering it
        // here would create a type structurally identical to LRLora.
        let r = registry();
        assert!(!r.contains("lora+"));
        assert!(!r.contains("loraplus"));
    }

    #[test]
    fn an_unknown_id_is_an_error_that_lists_what_exists() {
        let err = registry().require("dora2").unwrap_err().to_string();
        assert!(err.contains("unknown adapter"), "{err}");
        assert!(err.contains("dora"), "the error should list real ids: {err}");
    }

    #[test]
    fn registering_a_duplicate_id_is_refused() {
        let mut r = registry();
        let err = r.register(lora::CAPABILITY, lora::build);
        assert!(err.is_err(), "a duplicate id must be refused, not overwritten");
    }

    #[test]
    fn capability_records_the_loha_delta_materialization() {
        // The single most surprising research finding: LoHa cannot avoid
        // forming the full delta, so "low-rank means cheap" is false for it.
        let r = registry();
        assert!(!r.require("lora").unwrap().materializes_delta);
        assert!(r.require("loha").unwrap().materializes_delta);
    }

    #[test]
    fn capability_records_that_dora_needs_base_values() {
        let r = registry();
        assert!(!r.require("lora").unwrap().requires_base_values);
        assert!(r.require("dora").unwrap().requires_base_values);
    }

    #[test]
    fn capability_records_veras_cross_layer_sharing() {
        let r = registry();
        assert!(r.require("vera").unwrap().shares_params_across_layers);
        assert!(!r.require("lora").unwrap().shares_params_across_layers);
    }

    #[test]
    fn every_stub_names_a_reason_and_a_milestone() {
        let r = registry();
        for id in r.ids() {
            let cap = r.require(id).unwrap();
            if let ENSkillStatus::Stub { reason, milestone } = cap.status {
                assert!(!reason.is_empty(), "{id}: empty reason");
                assert!(!milestone.is_empty(), "{id}: empty milestone");
            }
        }
    }

    #[test]
    fn every_capability_cites_a_source() {
        let r = registry();
        for id in r.ids() {
            assert!(!r.require(id).unwrap().source.is_empty(), "{id}");
        }
    }

    #[test]
    fn spec_scale_is_alpha_over_r_by_default() {
        let s = VLAdapterSpec {
            alpha: 16.0,
            ..VLAdapterSpec::new(4, 4, 8, 0)
        };
        assert!((s.scale() - 2.0).abs() < TOL_EXACT);
    }

    #[test]
    fn spec_scale_switches_to_alpha_over_sqrt_r_under_rslora() {
        let s = VLAdapterSpec {
            alpha: 16.0,
            rslora: true,
            ..VLAdapterSpec::new(4, 4, 4, 0)
        };
        // 16 / sqrt(4) = 8
        assert!((s.scale() - 8.0).abs() < TOL_EXACT);
    }

    #[test]
    fn default_spec_alpha_equals_r_so_scale_is_one() {
        let s = VLAdapterSpec::new(8, 8, 4, 0);
        assert!((s.scale() - 1.0).abs() < TOL_EXACT);
    }
}
