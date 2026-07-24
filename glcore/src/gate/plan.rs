//! Execution plans and their constituent domains — paper §4.1 (domains)
//! and §4.2 (`ExecutionPlan`). See `architecture/GATE/GATE-concepts.md`
//! for the full mapping from the paper's mathematical definitions to
//! these types.

use std::collections::HashMap;

use crate::gate::metrics::MetricVector;

/// One tensor operation — a member of the paper's `𝒪` domain (§4.1).
///
/// Marker stub only: GwenLand has no op-DAG representation yet (see
/// `architecture/GATE/GATE-mapping.md` Gap 1) — this type carries no
/// shape/dtype information or behavior. It exists so
/// [`ExecutionPlan::ordering`] has something concrete to be a `Vec` of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TensorOp;

/// A tensor computation graph — the paper's `𝒢 = (V, E)` (§4.1): a DAG of
/// operations and data dependencies.
///
/// Marker stub only, matching [`TensorOp`]: no dependency edges are
/// tracked. Exists so `Planner::generate_candidates`'s signature (see
/// `planner.rs`) matches the paper's reference interface (§5.2, where
/// `ShapeConstraint { graph: graph.ops.clone() }` reads a `.ops` field) —
/// see `architecture/GATE/GATE-mapping.md` Gap 1 for why a real graph
/// representation is deliberately not built this sprint.
#[derive(Debug, Clone, Default)]
pub struct TensorGraph {
    /// This graph's operations. Unordered: no edges/dependencies tracked.
    pub ops: Vec<TensorOp>,
}

/// Index of an operation within a plan's ordering — a reference to a
/// vertex `V` of the paper's `𝒢`, used as the key into
/// [`ExecutionPlan::layouts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub usize);

/// A target hardware backend — the paper's `β ∈ ℬ` (§4.1), restricted to
/// GwenLand's four engines rather than an open registry (see
/// `architecture/GATE/GATE-mapping.md` Gap 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// The `glproc` CPU engine — the fallback chain's unconditional floor
    /// (see `gl-agent-skills/architecture-skills/fallback-chain.md`), and
    /// so the only defensible default.
    #[default]
    Glproc,
    /// The `glcuda` NVIDIA CUDA engine.
    Glcuda,
    /// The `glvulkan` cross-vendor GPU engine.
    Glvulkan,
    /// The `glmetal` Apple Metal engine.
    Glmetal,
}

/// A memory layout assignment — the paper's `𝓛` domain (§4.1): a mapping
/// from logical tensor indices to physical offsets.
///
/// Stub: every GwenLand tensor is row-major today (see `crate::tensor`),
/// so there is only one variant — no layout choice exists yet to validate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// The only layout GwenLand tensors use today.
    RowMajor,
}

/// An execution plan — the paper's `P = (σ, β, 𝓛map, m)` (Definition,
/// §4.2).
#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    /// `σ` — the topological ordering of operations this plan executes.
    pub ordering: Vec<TensorOp>,
    /// `β` — the target backend.
    pub backend: BackendKind,
    /// `𝓛map` — the memory layout assigned to each intermediate tensor.
    pub layouts: HashMap<OpId, MemoryLayout>,
    /// `m` — this plan's 5-dimensional metric vector.
    pub metrics: MetricVector,
}
