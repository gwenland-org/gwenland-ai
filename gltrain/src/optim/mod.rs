//! Stummañ Gwellaer: optimizers and their registry.
//!
//! Four update rules, one implemented. [`OPAdamW`] is full; [`OPLion`],
//! [`OPAdafactor`] and [`OPAdamW8bit`] are stubs that allocate their **real**
//! state shape and refuse to compute.
//!
//! # Why `step` takes a gradient store and not a tape
//!
//! This is the KL-006 resolution, and it is the load-bearing decision in this
//! module. `matmul`/`mul`/`div`/`sqrt`/`relu` snapshot their forward operands
//! when the node is recorded. The optimizer writes weights back **in place**
//! (through [`TPParameter::set_data`], so the parameter keeps its `TensorId`
//! and the state keyed on that ID stays valid). If a tape were still live at
//! that moment, its captures would hold pre-update values and the next backward
//! pass would compute gradients against weights that no longer exist: no error,
//! no crash, a plausible loss curve that is quietly wrong.
//!
//! [`crate::autograd::Tape::finish_step`] hands over the gradients and clears
//! the tape in one call, so a [`VLGradStore`] cannot be obtained while the tape
//! is still live. `step` takes that store. There is no ordering for a caller to
//! get wrong. See `KNOWN_ISSUES.md` KL-006.
//!
//! # Why the arithmetic never touches `Tensor`
//!
//! `Tensor::mul_scalar` and friends record to whatever tape their operands
//! carry, and a parameter is tracked by definition. Running the update through
//! them would append a node per parameter per step, forever, and pollute the
//! next backward pass. Every optimizer here works on `B::Storage` directly.

use crate::autograd::grad_store::VLGradStore;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use std::collections::BTreeMap;

pub mod adafactor;
pub mod adamw;
pub mod adamw8bit;
pub mod lion;

pub use adafactor::{ENAdafactorMoment, OPAdafactor};
pub use adamw::{OPAdamW, OPAdamWMoments, VLAdamWConfig};
pub use adamw8bit::OPAdamW8bit;
pub use lion::{OPLion, VLLionConfig};

/// A named parameter group with a learning-rate multiplier.
///
/// # This exists in M2 so LoRA+ is a config change in M3, not a rewrite
///
/// Hayou et al. 2024 (arXiv:2402.12354) changes exactly one thing about LoRA:
/// `eta_B = lambda * eta_A`. No shape, no init, no forward-pass difference, and
/// nothing new in a checkpoint. So there is deliberately no `LRLoraPlus`
/// adapter: the whole method is a per-group learning rate, and this is the
/// struct that carries it.
///
/// Every parameter starts in a group named `"default"` at multiplier 1.0.
#[derive(Debug, Clone, PartialEq)]
pub struct VLParamGroup {
    /// Group name. `"default"` is the implicit group every parameter falls into.
    pub name: String,
    /// Multiplies the optimizer's base learning rate for members of this group.
    pub lr_multiplier: f64,
}

impl VLParamGroup {
    /// A group with the given name and multiplier.
    pub fn new(name: impl Into<String>, lr_multiplier: f64) -> Self {
        Self {
            name: name.into(),
            lr_multiplier,
        }
    }
}

/// The name of the group every parameter belongs to unless assigned otherwise.
pub const DEFAULT_GROUP: &str = "default";

/// How an optimizer's per-parameter state is shaped.
///
/// `EN` because a closed set of variants is this type's whole job. The point of
/// the distinction: a caller sizing a memory budget cannot treat these the
/// same, and Adafactor in particular is not "AdamW with smaller numbers".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENOptimizerStateShape {
    /// N full-size buffers per parameter, whatever the parameter's rank.
    /// AdamW is 2 (`m` and `v`), Lion is 1 (`m` only).
    Fixed {
        /// Buffers of the parameter's own size, per parameter.
        buffers_per_param: usize,
    },
    /// The state's shape depends on the parameter's rank. Adafactor keeps
    /// `O(n+m)` row/column sums for a rank-2 parameter but a full `O(n)` second
    /// moment for a rank-1 one, so no single multiplier describes it.
    RankDependent,
    /// AdamW's two buffers, stored quantized. The update still runs in 32-bit.
    Quantized {
        /// Bits per stored element.
        bits: u8,
        /// Elements per independently-scaled block.
        block: usize,
    },
}

/// Machine-readable facts about an optimizer.
///
/// Mirrors [`crate::nn::adapter::VLAdapterCapability`]. Every field records
/// something the research established (`M2_RESEARCH.md` §4, §6) rather than
/// invented ceremony.
#[derive(Debug, Clone)]
pub struct VLOptimizerCapability {
    /// Registry key, e.g. `"adamw"`.
    pub id: &'static str,
    /// Implemented, or a stub with a reason and the milestone that owns it.
    pub status: ENSkillStatus,
    /// The shape of the per-parameter state.
    pub state_shape: ENOptimizerStateShape,
    /// State bytes as a multiple of the parameter's own size, for a run's
    /// header line. Approximate for [`ENOptimizerStateShape::RankDependent`],
    /// where the true figure is `O(n+m)/O(n*m)` and depends on the shape.
    pub memory_multiplier: f32,
    /// The paper this was implemented from.
    pub source: &'static str,
}

/// One update rule.
///
/// Traits take no prefix (naming rule 2). Object-safe, because
/// [`OptimizerRegistry`] hands out `Box<dyn Optimizer<B>>`.
pub trait Optimizer<B: Backend> {
    /// Apply one update to every parameter that has a gradient.
    ///
    /// `grads` comes from [`crate::autograd::Tape::finish_step`], never from a
    /// live tape. See the module docs for why that is the whole KL-006 story.
    ///
    /// A parameter with no entry in `grads` is **skipped**, not an error: a
    /// frozen base weight never receives one, and that is the ordinary LoRA
    /// shape rather than an edge case.
    fn step(&mut self, params: &mut [&mut TPParameter<B>], grads: &VLGradStore) -> Result<()>;

    /// Parameter groups, in registration order. `"default"` is always first.
    fn groups(&self) -> &[VLParamGroup];

    /// Add a group. Errors if the name is already taken.
    fn add_group(&mut self, group: VLParamGroup) -> Result<()>;

    /// Put a parameter in a group by name. Errors if the group does not exist.
    fn assign_group(&mut self, param_name: &str, group_name: &str) -> Result<()>;

    /// Optimizer state as named tensors, keyed by **parameter name**.
    ///
    /// State is keyed by `TensorId` in memory, which is process-global and
    /// explicitly not persistable. `params` is what resolves ID back to name,
    /// and this is the only place that translation happens.
    fn state_tensors(&self, params: &[&TPParameter<B>]) -> Result<Vec<VLNamedTensor>>;

    /// Restore state saved by [`Optimizer::state_tensors`], re-keying name back
    /// to the current process's `TensorId`s via `params`.
    fn load_state(&mut self, params: &[&TPParameter<B>], named: &[VLNamedTensor]) -> Result<()>;

    /// This optimizer's capability record.
    fn capability(&self) -> &'static VLOptimizerCapability;
}

/// One named tensor as it crosses the serialization boundary.
///
/// `VL` because it is a plain data bag. Not generic over `B`: by the time state
/// is being saved it is host data, and a checkpoint has no backend.
#[derive(Debug, Clone, PartialEq)]
pub struct VLNamedTensor {
    /// Key. For optimizer state this is `"{param_name}.{slot}"`, e.g.
    /// `"lora_a.m"`.
    pub name: String,
    /// Row-major values.
    pub data: Vec<f32>,
    /// Shape. Stored so a validator can catch a transposed load, which an
    /// element count cannot.
    pub shape: Vec<usize>,
}

impl VLNamedTensor {
    /// A named tensor.
    pub fn new(name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            data,
            shape,
        }
    }
}

/// How to build one optimizer.
///
/// # Why every field is an `Option`
///
/// There is no set of defaults that is correct for all four. Lion's
/// `beta2` is 0.99 where AdamW's is 0.999, and its learning rate should be
/// 3-10x smaller with `weight_decay` correspondingly larger, because the
/// effective decay is `lr * lambda` and the two have to move together. A spec
/// that carried one default set would quietly impose AdamW's on Lion, which is
/// exactly the "same knob wearing two names" mistake `adamw.md` Rule 8 warns
/// about. `None` means "use this optimizer's own default".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VLOptimizerSpec {
    /// Base learning rate.
    pub lr: Option<f64>,
    /// First moment decay.
    pub beta1: Option<f64>,
    /// Second moment decay, or Lion's momentum decay. Not the same quantity.
    pub beta2: Option<f64>,
    /// Denominator epsilon. Not meaningful for Lion, which has no denominator.
    pub eps: Option<f64>,
    /// Decoupled weight decay coefficient.
    pub weight_decay: Option<f64>,
}

impl VLOptimizerSpec {
    /// A spec that overrides nothing: every optimizer gets its own defaults.
    pub fn defaults() -> Self {
        Self::default()
    }

    /// A spec overriding only the learning rate.
    pub fn with_lr(lr: f64) -> Self {
        Self {
            lr: Some(lr),
            ..Self::default()
        }
    }
}

type BuildFn<B> = fn(&VLOptimizerSpec) -> Result<Box<dyn Optimizer<B>>>;

/// Maps optimizer ids to their capability record and constructor.
///
/// Copies `glictus-caliburni/src/plugin.rs`'s `PluginRegistry`, the same way
/// [`crate::nn::adapter::AdapterRegistry`] does: registration **refuses** on a
/// duplicate id rather than overwriting, [`resolve`](Self::resolve) returns an
/// `Option` and [`require`](Self::require) an error, and `with_builtins`
/// preloads what the crate ships.
pub struct OptimizerRegistry<B: Backend> {
    entries: BTreeMap<&'static str, (&'static VLOptimizerCapability, BuildFn<B>)>,
}

impl<B: Backend> Default for OptimizerRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> OptimizerRegistry<B> {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Every optimizer this crate ships: one real, three stubs.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        // Built-in ids are distinct by construction, so these cannot collide.
        let builtins: [(&'static VLOptimizerCapability, BuildFn<B>); 4] = [
            (adamw::CAPABILITY, adamw::build),
            (lion::CAPABILITY, lion::build),
            (adafactor::CAPABILITY, adafactor::build),
            (adamw8bit::CAPABILITY, adamw8bit::build),
        ];
        for (cap, build) in builtins {
            r.register(cap, build)
                .expect("built-in optimizer ids are unique");
        }
        r
    }

    /// Add an optimizer.
    ///
    /// Returns [`GlTrainError::InvalidOp`] if the id is taken. Two
    /// implementations claiming one id is a wiring bug, and keeping the last
    /// registered would make behaviour depend on registration order.
    pub fn register(
        &mut self,
        capability: &'static VLOptimizerCapability,
        build: BuildFn<B>,
    ) -> Result<()> {
        if self.entries.contains_key(capability.id) {
            return Err(GlTrainError::InvalidOp(format!(
                "optimizer '{}' is already registered; an id maps to exactly one implementation",
                capability.id
            )));
        }
        self.entries.insert(capability.id, (capability, build));
        Ok(())
    }

    /// Look up a capability record. `None` means nothing claims this id.
    pub fn resolve(&self, id: &str) -> Option<&'static VLOptimizerCapability> {
        self.entries.get(id).map(|(cap, _)| *cap)
    }

    /// Look up a capability record, erroring when absent.
    pub fn require(&self, id: &str) -> Result<&'static VLOptimizerCapability> {
        self.resolve(id).ok_or_else(|| {
            GlTrainError::InvalidOp(format!(
                "unknown optimizer '{id}'; registered: {}",
                self.ids().join(", ")
            ))
        })
    }

    /// Build an optimizer.
    ///
    /// A stub constructs successfully and fails on `step`. The caller learns
    /// the difference between "no such optimizer" (a typo, at construction) and
    /// "this optimizer is not implemented yet" (a real request, at the point of
    /// use), and never silently gets a different update rule than it asked for.
    pub fn build(&self, id: &str, spec: &VLOptimizerSpec) -> Result<Box<dyn Optimizer<B>>> {
        let (_, build) = self.entries.get(id).ok_or_else(|| {
            GlTrainError::InvalidOp(format!(
                "unknown optimizer '{id}'; registered: {}",
                self.ids().join(", ")
            ))
        })?;
        build(spec)
    }

    /// Every registered id, sorted.
    pub fn ids(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    /// Ids whose `step` actually computes.
    pub fn implemented_ids(&self) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|(_, (cap, _))| cap.status.is_full())
            .map(|(id, _)| *id)
            .collect()
    }
}

/// Shared group bookkeeping, so the four optimizers do not each reimplement it.
///
/// Not public: it is an implementation detail of this module's optimizers, and
/// a caller reaches groups through the [`Optimizer`] trait.
#[derive(Debug, Clone)]
pub(crate) struct GroupTable {
    groups: Vec<VLParamGroup>,
    group_of: BTreeMap<String, usize>,
}

impl GroupTable {
    /// A table holding only the default group.
    pub(crate) fn new() -> Self {
        Self {
            groups: vec![VLParamGroup::new(DEFAULT_GROUP, 1.0)],
            group_of: BTreeMap::new(),
        }
    }

    pub(crate) fn groups(&self) -> &[VLParamGroup] {
        &self.groups
    }

    pub(crate) fn add_group(&mut self, group: VLParamGroup) -> Result<()> {
        if self.groups.iter().any(|g| g.name == group.name) {
            return Err(GlTrainError::InvalidOp(format!(
                "parameter group '{}' already exists",
                group.name
            )));
        }
        self.groups.push(group);
        Ok(())
    }

    pub(crate) fn assign(&mut self, param_name: &str, group_name: &str) -> Result<()> {
        let idx = self
            .groups
            .iter()
            .position(|g| g.name == group_name)
            .ok_or_else(|| {
                GlTrainError::InvalidOp(format!(
                    "no parameter group named '{group_name}'; add it before assigning to it"
                ))
            })?;
        self.group_of.insert(param_name.to_string(), idx);
        Ok(())
    }

    /// The learning rate for `param_name`, base rate times its group's
    /// multiplier. An unassigned parameter is in the default group at 1.0.
    pub(crate) fn effective_lr(&self, base_lr: f64, param_name: &str) -> f64 {
        let idx = self.group_of.get(param_name).copied().unwrap_or(0);
        base_lr * self.groups[idx].lr_multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    #[test]
    fn builtins_register_all_four_optimizers() {
        let r = OptimizerRegistry::<GlProc>::with_builtins();
        assert_eq!(r.ids(), vec!["adafactor", "adamw", "adamw8bit", "lion"]);
    }

    /// Only AdamW computes on M2. A caller reading this list knows what it can
    /// actually train with, without constructing anything.
    #[test]
    fn only_adamw_reports_itself_as_implemented() {
        let r = OptimizerRegistry::<GlProc>::with_builtins();
        assert_eq!(r.implemented_ids(), vec!["adamw"]);
    }

    /// Two implementations claiming one id is a wiring bug. Keeping the last
    /// registered would make behaviour depend on registration order.
    #[test]
    fn optimizer_registry_refuses_a_duplicate_id() {
        let mut r = OptimizerRegistry::<GlProc>::with_builtins();
        let err = r.register(adamw::CAPABILITY, adamw::build);
        assert!(err.is_err(), "re-registering 'adamw' must be refused");
    }

    #[test]
    fn require_names_the_registered_ids_when_the_lookup_fails() {
        let r = OptimizerRegistry::<GlProc>::with_builtins();
        assert!(r.resolve("adamw").is_some());
        assert!(r.resolve("adamwww").is_none());
        let msg = r.require("adamwww").unwrap_err().to_string();
        assert!(msg.contains("adamw"), "error should list what is available: {msg}");
    }

    /// A stub builds. It only refuses when asked to compute, so a caller can
    /// introspect it and gets a precise error at the point of use.
    #[test]
    fn every_builtin_including_the_stubs_can_be_built() {
        let r = OptimizerRegistry::<GlProc>::with_builtins();
        for id in r.ids() {
            assert!(
                r.build(id, &VLOptimizerSpec::defaults()).is_ok(),
                "'{id}' failed to construct"
            );
        }
    }

    #[test]
    fn an_unassigned_parameter_lands_in_the_default_group_at_multiplier_one() {
        let t = GroupTable::new();
        assert_eq!(t.effective_lr(1e-3, "anything"), 1e-3);
        assert_eq!(t.groups().len(), 1);
        assert_eq!(t.groups()[0].name, DEFAULT_GROUP);
    }

    /// The LoRA+ shape: `lora_b` gets a larger rate than `lora_a`, and nothing
    /// about the adapter changes.
    #[test]
    fn an_assigned_parameter_gets_its_groups_multiplier() {
        let mut t = GroupTable::new();
        t.add_group(VLParamGroup::new("lora_b", 4.0)).unwrap();
        t.assign("lora_b", "lora_b").unwrap();
        assert_eq!(t.effective_lr(1e-3, "lora_b"), 4e-3);
        assert_eq!(t.effective_lr(1e-3, "lora_a"), 1e-3);
    }

    #[test]
    fn adding_a_group_twice_is_refused() {
        let mut t = GroupTable::new();
        t.add_group(VLParamGroup::new("g", 2.0)).unwrap();
        assert!(t.add_group(VLParamGroup::new("g", 3.0)).is_err());
    }

    /// Assigning to a group nobody created is a typo, and a silent fallback to
    /// the default group would hide it behind a plausible loss curve.
    #[test]
    fn assigning_to_an_unknown_group_is_refused() {
        let mut t = GroupTable::new();
        assert!(t.assign("lora_b", "typo").is_err());
    }
}
