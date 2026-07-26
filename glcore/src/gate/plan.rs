//! Execution plans and their constituent domains — paper §4.1 (domains)
//! and §4.2 (`ExecutionPlan`). See `architecture/GATE/GATE-concepts.md`
//! for the full mapping from the paper's mathematical definitions to
//! these types.

use std::collections::HashMap;

use crate::gate::metrics::MetricVector;

/// One tensor operation — a member of the paper's `𝒪` domain (§4.1).
///
/// Named by the weight tensor it operates on (GGUF tensor-name convention,
/// e.g. `"blk.0.ffn_gate.weight"`); no dependency edges or shape/dtype
/// beyond the name are tracked. This is deliberately not a full op-DAG —
/// GwenLand's engines execute a fixed, hand-written layer walk
/// (`Runner::generate`, `GpuModel::generate`), not a graph a planner walks
/// generically, so a richer representation would describe a compiler this
/// codebase does not have. The name is enough to let a backend crate (e.g.
/// `glproc`) generate real per-tensor format candidates against it — see
/// `Planner::generate_candidates`'s glproc caller for a worked example.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorOp {
    /// The weight tensor this op reads, by name.
    pub tensor_name: String,
}

impl TensorOp {
    /// A tensor op named after the weight tensor it operates on.
    pub fn new(tensor_name: impl Into<String>) -> Self {
        TensorOp { tensor_name: tensor_name.into() }
    }
}

/// A tensor computation graph — the paper's `𝒢 = (V, E)` (§4.1): a DAG of
/// operations and data dependencies.
///
/// Unordered: no edges are tracked, per [`TensorOp`]'s doc — GwenLand has
/// no op-DAG representation to derive edges from, only a fixed per-layer
/// walk. `ops` is the graph's vertex set; a caller building candidates
/// (e.g. `Planner::generate_candidates`) treats each op independently.
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
