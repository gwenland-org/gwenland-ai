//! `FuncBuilder` — shape inference and module assembly (ARTX02 §5).
//!
//! The middle layer between model code and the text emitter. It owns the
//! [`MlirEmitter`], infers every output shape, records parameters in
//! declaration order (the PJRT argument ABI), and assembles the final
//! `module { func.func @main }`.
//!
//! # Shape errors panic
//!
//! ⚠️ ARTX02 §5's decision, kept: a shape mismatch is a bug in the *model
//! definition*, caught at trace time before anything is compiled, and it
//! mirrors JAX, where a shape error is a Python exception during tracing.
//! Threading `Result` through every op call site would buy nothing — there is
//! no caller that can recover from "your matmul dimensions disagree".
//!
//! ⛔ This is **not** a licence to panic elsewhere. `rust-skills/error-handling.md`
//! exists because GwenLand parses untrusted model files, and that path — the
//! checkpoint loader matching real weights against this signature (ARTX04) —
//! must return `Result`. The line is: model definition panics, file contents
//! do not.

use crate::graph::value::SsaValue;
use crate::matrix::MatmulOpts;
use crate::stablehlo::emitter::{MlirEmitter, SsaName};
use crate::stablehlo::ops::{self, DotDimensionNumbers};
use crate::stablehlo::types::{DType, ParamDesc, ParamKind, Shape};

/// A finished trace: the MLIR text plus the ABI needed to call it.
#[derive(Debug, Clone)]
pub struct BuiltFunc {
    pub mlir: String,
    pub signature: Signature,
}

/// What the compiled program expects, in PJRT argument order.
///
/// ⚠️ `inputs` then `weights` is the order [`FuncBuilder::finish`] declares
/// parameters in **only if** they were declared in that order during tracing.
/// [`Signature::param_order`] is the authority; the split lists are for the
/// checkpoint loader's convenience.
#[derive(Debug, Clone)]
pub struct Signature {
    /// Runtime inputs (tokens, positions, masks) in declaration order.
    pub inputs: Vec<ParamDesc>,
    /// Checkpoint weights in declaration order. `name` is the checkpoint key.
    pub weights: Vec<ParamDesc>,
    /// KV cache buffers (ARTX05) in declaration order. `alias_output` on each
    /// says which output it is donated to.
    pub kv_caches: Vec<ParamDesc>,
    /// Every parameter in the exact order `func.func` declares them — this is
    /// the order PJRT will read `argument_lists` in.
    pub param_order: Vec<ParamDesc>,
    pub outputs: Vec<Shape>,
}

pub struct FuncBuilder {
    emitter: MlirEmitter,
    params: Vec<ParamDesc>,
    outputs: Vec<SsaValue>,
    func_name: String,
    module_name: String,
}

impl FuncBuilder {
    pub fn new(func_name: impl Into<String>, module_name: impl Into<String>) -> Self {
        let mut emitter = MlirEmitter::new();
        // The emitter starts at one level (inside *a* body). Body lines live
        // inside `module { func.func { ... } }`, which is two levels, so push
        // one more. Purely cosmetic — MLIR ignores whitespace — but a dumped
        // module is read by a human when something goes wrong.
        emitter.push_indent();
        FuncBuilder {
            emitter,
            params: Vec::new(),
            outputs: Vec::new(),
            func_name: func_name.into(),
            module_name: module_name.into(),
        }
    }

    // ── Parameters ─────────────────────────────────────────────────────────

    pub fn param(&mut self, name: impl Into<String>, shape: Shape, kind: ParamKind) -> SsaValue {
        let ssa = self.emitter.fresh();
        let name = name.into();
        assert!(
            !self.params.iter().any(|p| p.name == name),
            "duplicate parameter name {name:?} — weight names are checkpoint keys \
             and must be unique, or the loader silently binds one tensor twice"
        );
        self.params.push(ParamDesc {
            name,
            shape: shape.clone(),
            kind,
            ssa,
            alias_output: None,
        });
        SsaValue::new(ssa, shape)
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> SsaValue {
        self.param(name, shape, ParamKind::Input)
    }

    pub fn weight(&mut self, name: impl Into<String>, shape: Shape) -> SsaValue {
        self.param(name, shape, ParamKind::Weight)
    }

    /// Declares a KV cache buffer: a resident, per-call input that a decode
    /// step writes to (ARTX05). Not scope-qualified — like [`FuncBuilder::input`],
    /// this is the caller's ABI, not a checkpoint key.
    pub fn kv_cache(&mut self, name: impl Into<String>, shape: Shape) -> SsaValue {
        self.param(name, shape, ParamKind::KvCache)
    }

    /// Declares that `param` — which must have been created by
    /// [`FuncBuilder::kv_cache`] — is donated to output `output_index`: XLA
    /// aliases the same buffer instead of copying (ARTX05 §6).
    ///
    /// Emitted as a `tf.aliasing_output` argument attribute on `func.func`,
    /// which is the syntax `jax.jit(..., donate_argnums=...)` itself lowers
    /// to — confirmed by round-tripping a hand-written module through
    /// jaxlib's MLIR parser, since there is no PJRT plugin on this machine to
    /// check it against directly.
    ///
    /// Restricted to `KvCache` params on purpose: aliasing a weight or a plain
    /// input would let PJRT overwrite it in place, corrupting a checkpoint
    /// tensor or a caller-owned buffer that was never meant to be mutated.
    ///
    /// The output-index bound and the shape match against the aliased output
    /// are checked in [`FuncBuilder::finish`], where the output list is final.
    ///
    /// # Panics
    /// If `param` was not declared via `kv_cache`, or is already aliased.
    pub fn alias_output(&mut self, param: &SsaValue, output_index: usize) {
        let p = self
            .params
            .iter_mut()
            .find(|p| p.ssa == param.ssa())
            .unwrap_or_else(|| panic!("alias_output: {} is not a declared parameter", param.ssa()));
        assert_eq!(
            p.kind,
            ParamKind::KvCache,
            "alias_output: {:?} ({}) is not a kv-cache parameter — only kv-cache \
             buffers may be donated",
            p.kind,
            p.name
        );
        assert!(
            p.alias_output.is_none(),
            "alias_output: {} is already aliased to output {}",
            p.name,
            p.alias_output.unwrap()
        );
        p.alias_output = Some(output_index);
    }

    /// Direct emitter access, for ops that need to emit a region body.
    pub fn emitter_mut(&mut self) -> &mut MlirEmitter {
        &mut self.emitter
    }

    pub fn set_outputs(&mut self, outputs: Vec<SsaValue>) {
        self.outputs = outputs;
    }

    // ── Elementwise ────────────────────────────────────────────────────────

    pub fn add(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        self.binary(lhs, rhs, "add", ops::emit_add)
    }

    pub fn subtract(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        self.binary(lhs, rhs, "subtract", ops::emit_subtract)
    }

    pub fn multiply(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        self.binary(lhs, rhs, "multiply", ops::emit_multiply)
    }

    pub fn divide(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        self.binary(lhs, rhs, "divide", ops::emit_divide)
    }

    pub fn maximum(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        self.binary(lhs, rhs, "maximum", ops::emit_maximum)
    }

    pub fn minimum(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        self.binary(lhs, rhs, "minimum", ops::emit_minimum)
    }

    fn binary(
        &mut self,
        lhs: &SsaValue,
        rhs: &SsaValue,
        what: &str,
        emit: fn(&mut MlirEmitter, SsaName, SsaName, &Shape) -> SsaName,
    ) -> SsaValue {
        assert_eq!(
            lhs.shape(),
            rhs.shape(),
            "{what}: operand shapes must be identical — StableHLO elementwise ops \
             do not broadcast. Got {lhs} and {rhs}"
        );
        let name = emit(&mut self.emitter, lhs.ssa(), rhs.ssa(), lhs.shape());
        SsaValue::new(name, lhs.shape.clone())
    }

    pub fn negate(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_negate)
    }

    pub fn rsqrt(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_rsqrt)
    }

    pub fn sqrt(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_sqrt)
    }

    /// Sigmoid. The building block of SiLU (ARTX01 §7.5).
    pub fn logistic(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_logistic)
    }

    pub fn exponential(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_exponential)
    }

    pub fn log(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_log)
    }

    pub fn tanh(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_tanh)
    }

    pub fn abs(&mut self, x: &SsaValue) -> SsaValue {
        self.unary(x, ops::emit_abs)
    }

    fn unary(
        &mut self,
        x: &SsaValue,
        emit: fn(&mut MlirEmitter, SsaName, &Shape) -> SsaName,
    ) -> SsaValue {
        let name = emit(&mut self.emitter, x.ssa(), x.shape());
        SsaValue::new(name, x.shape.clone())
    }

    // ── Precision ──────────────────────────────────────────────────────────

    /// Inserts a `stablehlo.convert`, or returns the operand unchanged if it is
    /// already at `target`.
    ///
    /// The no-op short-circuit is what makes `to_dtype` free to call
    /// defensively in ops — under `PrecisionPolicy::f32()` the upcasts in
    /// RMSNorm vanish instead of emitting `f32 -> f32` conversions.
    pub fn convert(&mut self, x: &SsaValue, target: DType) -> SsaValue {
        if x.dtype() == target {
            return x.clone();
        }
        let out_shape = Shape::new(x.shape().dims.clone(), target);
        let name = ops::emit_convert(&mut self.emitter, x.ssa(), x.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    // ── Constants ──────────────────────────────────────────────────────────

    /// # Panics
    /// On a value with no plain MLIR literal (non-finite float, fractional
    /// integer). Trace-time programming error — see the module docs.
    pub fn constant_scalar(&mut self, value: f64, dtype: DType) -> SsaValue {
        self.constant_splat(value, Shape::scalar(dtype))
    }

    /// # Panics
    /// As [`FuncBuilder::constant_scalar`].
    pub fn constant_splat(&mut self, value: f64, shape: Shape) -> SsaValue {
        let name = ops::emit_constant_splat(&mut self.emitter, value, &shape)
            .unwrap_or_else(|e| panic!("constant: {e}"));
        SsaValue::new(name, shape)
    }

    // ── Shape ops ──────────────────────────────────────────────────────────

    pub fn reshape(&mut self, x: &SsaValue, new_dims: Vec<usize>) -> SsaValue {
        let numel_out: usize = new_dims.iter().product();
        assert_eq!(
            x.shape().numel(),
            numel_out,
            "reshape: element count must be preserved — {} has {} elements, \
             requested {new_dims:?} has {numel_out}",
            x.shape().mlir_type(),
            x.shape().numel()
        );
        let out_shape = Shape::new(new_dims, x.dtype());
        let name = ops::emit_reshape(&mut self.emitter, x.ssa(), x.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn transpose(&mut self, x: &SsaValue, permutation: Vec<usize>) -> SsaValue {
        assert_eq!(
            permutation.len(),
            x.rank(),
            "transpose: permutation {permutation:?} has {} entries but {} is rank {}",
            permutation.len(),
            x.shape().mlir_type(),
            x.rank()
        );
        let mut seen = vec![false; x.rank()];
        for &p in &permutation {
            assert!(p < x.rank(), "transpose: index {p} out of range");
            assert!(
                !std::mem::replace(&mut seen[p], true),
                "transpose: {permutation:?} repeats index {p} — a permutation must \
                 use every axis exactly once"
            );
        }
        let out_dims: Vec<usize> = permutation.iter().map(|&i| x.dim(i)).collect();
        let out_shape = Shape::new(out_dims, x.dtype());
        let name = ops::emit_transpose(
            &mut self.emitter,
            x.ssa(),
            &permutation,
            x.shape(),
            &out_shape,
        );
        SsaValue::new(name, out_shape)
    }

    pub fn slice(
        &mut self,
        x: &SsaValue,
        starts: Vec<usize>,
        limits: Vec<usize>,
        strides: Vec<usize>,
    ) -> SsaValue {
        assert!(
            starts.len() == x.rank() && limits.len() == x.rank() && strides.len() == x.rank(),
            "slice: starts/limits/strides must each have {} entries for {}",
            x.rank(),
            x.shape().mlir_type()
        );
        let mut out_dims = Vec::with_capacity(x.rank());
        for i in 0..x.rank() {
            assert!(strides[i] > 0, "slice: stride {} must be positive", i);
            assert!(
                starts[i] <= limits[i] && limits[i] <= x.dim(i),
                "slice: dim {i} bounds [{}:{}] out of range for size {}",
                starts[i],
                limits[i],
                x.dim(i)
            );
            out_dims.push((limits[i] - starts[i]).div_ceil(strides[i]));
        }
        let out_shape = Shape::new(out_dims, x.dtype());
        let name = ops::emit_slice(
            &mut self.emitter,
            x.ssa(),
            &starts,
            &limits,
            &strides,
            x.shape(),
            &out_shape,
        );
        SsaValue::new(name, out_shape)
    }

    pub fn concatenate(&mut self, operands: &[&SsaValue], dimension: usize) -> SsaValue {
        assert!(!operands.is_empty(), "concatenate: no operands");
        let first = operands[0];
        assert!(
            dimension < first.rank(),
            "concatenate: dimension {dimension} out of range for rank {}",
            first.rank()
        );
        let mut out_dims = first.shape().dims.clone();
        for op in &operands[1..] {
            assert_eq!(op.rank(), first.rank(), "concatenate: rank mismatch");
            assert_eq!(op.dtype(), first.dtype(), "concatenate: dtype mismatch");
            for i in 0..first.rank() {
                if i != dimension {
                    assert_eq!(
                        op.dim(i),
                        first.dim(i),
                        "concatenate: dim {i} must match on every operand \
                         (only dim {dimension} may differ)"
                    );
                }
            }
            out_dims[dimension] += op.dim(dimension);
        }
        let out_shape = Shape::new(out_dims, first.dtype());
        let tagged: Vec<(SsaName, Shape)> = operands
            .iter()
            .map(|v| (v.ssa(), v.shape().clone()))
            .collect();
        let name = ops::emit_concatenate(&mut self.emitter, &tagged, dimension, &out_shape);
        SsaValue::new(name, out_shape)
    }

    /// `broadcast_dims[i]` is the **output** dimension that input dimension `i`
    /// maps to.
    pub fn broadcast_in_dim(
        &mut self,
        x: &SsaValue,
        broadcast_dims: Vec<usize>,
        out_dims: Vec<usize>,
    ) -> SsaValue {
        assert_eq!(
            broadcast_dims.len(),
            x.rank(),
            "broadcast_in_dim: need one output axis per input axis — {} is rank {} \
             but broadcast_dimensions is {broadcast_dims:?}",
            x.shape().mlir_type(),
            x.rank()
        );
        for (i, &o) in broadcast_dims.iter().enumerate() {
            assert!(
                o < out_dims.len(),
                "broadcast_in_dim: input axis {i} maps to output axis {o}, \
                 but the result is rank {}",
                out_dims.len()
            );
            assert!(
                x.dim(i) == out_dims[o] || x.dim(i) == 1,
                "broadcast_in_dim: input axis {i} has size {} and maps to output axis {o} \
                 of size {} — a broadcast axis must match or be 1",
                x.dim(i),
                out_dims[o]
            );
        }
        let out_shape = Shape::new(out_dims, x.dtype());
        let name = ops::emit_broadcast_in_dim(
            &mut self.emitter,
            x.ssa(),
            &broadcast_dims,
            x.shape(),
            &out_shape,
        );
        SsaValue::new(name, out_shape)
    }

    // ── dot_general ────────────────────────────────────────────────────────

    /// Emits `stablehlo.dot_general`, reconciling operand dtypes first.
    ///
    /// ⛔ **Addition to ARTX02 §5.** The sketch there passes both operands
    /// straight through and takes the output dtype from `lhs`. StableHLO
    /// requires `dot_general` operands to share an element type, so a BF16
    /// weight meeting an F32 activation would emit a module the verifier
    /// rejects — and under a mixed-precision policy that is the *normal* case,
    /// not an edge case.
    ///
    /// The reconciliation always **widens** (see [`promote`]). Narrowing would
    /// silently drop precision to make a shape check pass, which is exactly
    /// what P5 forbids. Code that wants BF16×BF16 should convert explicitly —
    /// which is what ARTX03's ops do, via the precision policy.
    pub fn dot_general(
        &mut self,
        lhs: &SsaValue,
        rhs: &SsaValue,
        dnums: &DotDimensionNumbers,
    ) -> SsaValue {
        self.dot_general_with(lhs, rhs, dnums, &MatmulOpts::default())
    }

    /// `dot_general` with explicit numerics (ARTX08 §A8.α). `MatmulOpts::default()`
    /// makes this byte-identical to [`FuncBuilder::dot_general`].
    pub fn dot_general_with(
        &mut self,
        lhs: &SsaValue,
        rhs: &SsaValue,
        dnums: &DotDimensionNumbers,
        opts: &MatmulOpts,
    ) -> SsaValue {
        let (lhs, rhs) = self.reconcile_dtypes(lhs, rhs);
        let out_shape = infer_dot_general_shape(lhs.shape(), rhs.shape(), dnums, opts.accumulate);
        let name = ops::emit_dot_general(
            &mut self.emitter,
            lhs.ssa(),
            rhs.ssa(),
            dnums,
            lhs.shape(),
            rhs.shape(),
            &out_shape,
            opts.numerics,
            opts.accumulate,
        );
        SsaValue::new(name, out_shape)
    }

    /// Batched matmul over the trailing two dimensions.
    ///
    /// ⛔ **Correction to ARTX02 §7.** That sketch derives the batch dimensions
    /// from `self.rank()` alone, so `[B,S,D] @ [D,F]` — every projection in the
    /// model — comes out as `lhs_batching=[0], rhs_batching=[0]`, batching the
    /// weight over its contraction axis. ARTX02 §9's own expected output for
    /// that exact matmul shows `lhs_batching = []`, so the document contradicts
    /// itself; §9 is the correct one.
    ///
    /// The batch count is taken from the **lower-rank** operand, which gives
    /// the right answer for both real shapes:
    ///
    /// | lhs | rhs | batch dims | meaning |
    /// |---|---|---|---|
    /// | `[B,S,D]` | `[D,F]` | none | projection against a weight matrix |
    /// | `[B,H,S,D]` | `[B,H,D,S]` | `[0,1]` | attention scores |
    pub fn matmul(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        self.matmul_with(lhs, rhs, &MatmulOpts::default())
    }

    /// [`FuncBuilder::matmul`] with explicit numerics (ARTX08 §A8.α).
    pub fn matmul_with(&mut self, lhs: &SsaValue, rhs: &SsaValue, opts: &MatmulOpts) -> SsaValue {
        assert!(
            lhs.rank() >= 2 && rhs.rank() >= 2,
            "matmul: both operands must be at least rank 2, got {lhs} and {rhs}"
        );
        let n_batch = lhs.rank().min(rhs.rank()) - 2;
        for i in 0..n_batch {
            assert_eq!(
                lhs.dim(i),
                rhs.dim(i),
                "matmul: batch dim {i} disagrees between {lhs} and {rhs}"
            );
        }
        let dnums = DotDimensionNumbers {
            lhs_batching: (0..n_batch).collect(),
            rhs_batching: (0..n_batch).collect(),
            lhs_contracting: vec![lhs.rank() - 1],
            rhs_contracting: vec![rhs.rank() - 2],
        };
        self.dot_general_with(lhs, rhs, &dnums, opts)
    }

    /// Widens whichever operand is narrower so both share an element type.
    fn reconcile_dtypes(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> (SsaValue, SsaValue) {
        if lhs.dtype() == rhs.dtype() {
            return (lhs.clone(), rhs.clone());
        }
        let target = promote(lhs.dtype(), rhs.dtype()).unwrap_or_else(|| {
            panic!(
                "dot_general: cannot reconcile {:?} with {:?} — convert one operand \
                 explicitly, because any implicit choice here would lose precision",
                lhs.dtype(),
                rhs.dtype()
            )
        });
        (self.convert(lhs, target), self.convert(rhs, target))
    }

    // ── reduce ─────────────────────────────────────────────────────────────

    /// Emits `stablehlo.reduce` with an `add` combiner.
    ///
    /// `init` must be a rank-0 tensor of the operand's element type.
    pub fn reduce_add(&mut self, x: &SsaValue, init: &SsaValue, dims: &[usize]) -> SsaValue {
        let out_shape = self.check_reduce(x, init, dims, "reduce_add");
        let name = ops::emit_reduce_add(
            &mut self.emitter,
            x.ssa(),
            init.ssa(),
            dims,
            x.shape(),
            &out_shape,
        );
        SsaValue::new(name, out_shape)
    }

    /// Emits `stablehlo.reduce` with a `maximum` combiner.
    pub fn reduce_max(&mut self, x: &SsaValue, init: &SsaValue, dims: &[usize]) -> SsaValue {
        let out_shape = self.check_reduce(x, init, dims, "reduce_max");
        let name = ops::emit_reduce_max(
            &mut self.emitter,
            x.ssa(),
            init.ssa(),
            dims,
            x.shape(),
            &out_shape,
        );
        SsaValue::new(name, out_shape)
    }

    /// Validates a reduce and returns the output shape (reduced dims dropped).
    fn check_reduce(
        &self,
        x: &SsaValue,
        init: &SsaValue,
        dims: &[usize],
        what: &str,
    ) -> Shape {
        assert!(!dims.is_empty(), "{what}: no dimensions to reduce");
        assert_eq!(
            init.rank(),
            0,
            "{what}: init must be a rank-0 tensor, got {init}"
        );
        assert_eq!(
            init.dtype(),
            x.dtype(),
            "{what}: init dtype {:?} does not match operand dtype {:?}",
            init.dtype(),
            x.dtype()
        );
        let mut seen = vec![false; x.rank()];
        for &d in dims {
            assert!(d < x.rank(), "{what}: dimension {d} out of range");
            assert!(
                !std::mem::replace(&mut seen[d], true),
                "{what}: dimension {d} listed twice"
            );
        }
        let out_dims: Vec<usize> = (0..x.rank())
            .filter(|i| !seen[*i])
            .map(|i| x.dim(i))
            .collect();
        Shape::new(out_dims, x.dtype())
    }

    // ── finish ─────────────────────────────────────────────────────────────

    /// Assembles the complete module.
    ///
    /// ⛔ **Deviation from ARTX02 §5**, which has this consume `self` so it can
    /// call `MlirEmitter::into_body()`. That forces [`crate::graph::TraceCx`]
    /// to pull the builder back out of its `Rc`, which cannot succeed while any
    /// [`crate::tensor::Tensor`] is alive — including the output tensors being
    /// passed in. ARTX02 §9's own example (`cx.finish(vec![&out])`) would panic
    /// every time. Borrowing instead removes the failure mode entirely.
    ///
    /// # Panics
    /// If no outputs were declared — a program that returns nothing is never
    /// what the caller meant.
    pub fn finish(&self) -> BuiltFunc {
        assert!(
            !self.outputs.is_empty(),
            "FuncBuilder::finish: no outputs declared — call set_outputs first"
        );

        // Buffer donation (ARTX05 §6): the output list is only final here, so
        // this is the first point that can check an alias against it.
        for p in &self.params {
            if let Some(idx) = p.alias_output {
                assert!(
                    idx < self.outputs.len(),
                    "finish: {} aliases output {idx}, but only {} outputs are declared",
                    p.name,
                    self.outputs.len()
                );
                assert_eq!(
                    self.outputs[idx].shape(),
                    &p.shape,
                    "finish: {} (shape {}) aliases output {idx} (shape {}) — a donated \
                     buffer must have identical shape on both sides, or XLA would be \
                     aliasing mismatched memory",
                    p.name,
                    p.shape.mlir_type(),
                    self.outputs[idx].shape().mlir_type()
                );
            }
        }

        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| match p.alias_output {
                Some(idx) => format!(
                    "{}: {} {{tf.aliasing_output = {idx} : i32}}",
                    p.ssa,
                    p.shape.mlir_type()
                ),
                None => format!("{}: {}", p.ssa, p.shape.mlir_type()),
            })
            .collect();
        let out_types: Vec<String> = self
            .outputs
            .iter()
            .map(|v| v.shape().mlir_type())
            .collect();
        let ret_names: Vec<String> = self.outputs.iter().map(|v| v.name.to_string()).collect();
        let ret_types = out_types.join(", ");

        let mut mlir = String::with_capacity(self.emitter.body().len() + 512);
        mlir.push_str(&format!("module @{} {{\n", self.module_name));
        mlir.push_str(&format!(
            "  func.func @{}({}) -> ({ret_types}) {{\n",
            self.func_name,
            params.join(", ")
        ));
        mlir.push_str(self.emitter.body());
        mlir.push_str(&format!(
            "    \"func.return\"({}) : ({ret_types}) -> ()\n",
            ret_names.join(", ")
        ));
        mlir.push_str("  }\n}\n");

        let signature = Signature {
            inputs: self
                .params
                .iter()
                .filter(|p| p.kind == ParamKind::Input)
                .cloned()
                .collect(),
            weights: self
                .params
                .iter()
                .filter(|p| p.kind == ParamKind::Weight)
                .cloned()
                .collect(),
            kv_caches: self
                .params
                .iter()
                .filter(|p| p.kind == ParamKind::KvCache)
                .cloned()
                .collect(),
            param_order: self.params.clone(),
            outputs: self.outputs.iter().map(|v| v.shape().clone()).collect(),
        };
        BuiltFunc { mlir, signature }
    }
}

/// The dtype both operands should be converted to, or `None` if no conversion
/// is safe.
///
/// Widening only. Two same-width floats (BF16 vs F16) have different exponent
/// ranges and neither converts to the other losslessly, so that pair refuses
/// rather than picking one — P5.
fn promote(a: DType, b: DType) -> Option<DType> {
    if a == b {
        return Some(a);
    }
    let float = |d: DType| matches!(d, DType::F64 | DType::F32 | DType::BF16 | DType::F16);
    let int = |d: DType| matches!(d, DType::I64 | DType::I32 | DType::I16 | DType::I8);
    // Mixing a float with an integer is never an accident worth guessing at.
    if !((float(a) && float(b)) || (int(a) && int(b))) {
        return None;
    }
    match a.byte_size().cmp(&b.byte_size()) {
        std::cmp::Ordering::Greater => Some(a),
        std::cmp::Ordering::Less => Some(b),
        std::cmp::Ordering::Equal => None,
    }
}

/// Output shape of a `dot_general`: batch dims, then lhs free dims, then rhs
/// free dims. Output dtype is `accumulate` (`preferred_element_type`) if the
/// caller named one, else the (already reconciled) lhs dtype — ARTX08 §A8.α:
/// before this, `preferred_element_type` could be emitted but the shape
/// inference ignored it, so the *stated* accumulation type and the *inferred*
/// result type could silently disagree.
fn infer_dot_general_shape(
    lhs: &Shape,
    rhs: &Shape,
    dnums: &DotDimensionNumbers,
    accumulate: Option<DType>,
) -> Shape {
    assert_eq!(
        dnums.lhs_batching.len(),
        dnums.rhs_batching.len(),
        "dot_general: batching dimension lists must be the same length"
    );
    assert_eq!(
        dnums.lhs_contracting.len(),
        dnums.rhs_contracting.len(),
        "dot_general: contracting dimension lists must be the same length"
    );
    for (&li, &ri) in dnums.lhs_batching.iter().zip(&dnums.rhs_batching) {
        assert_eq!(
            lhs.dims[li], rhs.dims[ri],
            "dot_general: batch dim size mismatch lhs[{li}]={} vs rhs[{ri}]={}",
            lhs.dims[li], rhs.dims[ri]
        );
    }
    for (&li, &ri) in dnums.lhs_contracting.iter().zip(&dnums.rhs_contracting) {
        assert_eq!(
            lhs.dims[li], rhs.dims[ri],
            "dot_general: contracting dim size mismatch lhs[{li}]={} vs rhs[{ri}]={}",
            lhs.dims[li], rhs.dims[ri]
        );
    }

    let batch = dnums.lhs_batching.iter().map(|&i| lhs.dims[i]);
    let lhs_free = (0..lhs.rank())
        .filter(|i| !dnums.lhs_batching.contains(i) && !dnums.lhs_contracting.contains(i))
        .map(|i| lhs.dims[i]);
    let rhs_free = (0..rhs.rank())
        .filter(|i| !dnums.rhs_batching.contains(i) && !dnums.rhs_contracting.contains(i))
        .map(|i| rhs.dims[i]);

    Shape::new(
        batch.chain(lhs_free).chain(rhs_free).collect::<Vec<_>>(),
        accumulate.unwrap_or(lhs.dtype),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision;

    fn f32(dims: impl Into<Vec<usize>>) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn dot_general_shape_is_batch_then_lhs_free_then_rhs_free() {
        // [B,H,S,D] @ [B,H,D,S'] -> [B,H,S,S'] — the attention score product.
        let dnums = DotDimensionNumbers {
            lhs_batching: vec![0, 1],
            rhs_batching: vec![0, 1],
            lhs_contracting: vec![3],
            rhs_contracting: vec![2],
        };
        let out = infer_dot_general_shape(
            &f32([1, 16, 512, 128]),
            &f32([1, 16, 128, 512]),
            &dnums,
            None,
        );
        assert_eq!(out.dims, vec![1, 16, 512, 512]);
    }

    #[test]
    #[should_panic(expected = "contracting dim size mismatch")]
    fn dot_general_rejects_a_contraction_over_unequal_sizes() {
        let dnums = DotDimensionNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            ..Default::default()
        };
        let _ = infer_dot_general_shape(&f32([4, 8]), &f32([9, 16]), &dnums, None);
    }

    #[test]
    fn preferred_element_type_overrides_the_inferred_output_dtype() {
        let dnums = DotDimensionNumbers {
            lhs_contracting: vec![1],
            rhs_contracting: vec![0],
            ..Default::default()
        };
        let out = infer_dot_general_shape(
            &f32([4, 8]),
            &f32([8, 16]),
            &dnums,
            Some(DType::F64),
        );
        assert_eq!(out.dtype, DType::F64, "accumulate must win over lhs's dtype");
    }

    #[test]
    fn matmul_with_default_opts_matches_plain_matmul() {
        let mut default_b = FuncBuilder::new("main", "t");
        let x = default_b.input("x", f32([1, 512, 2048]));
        let w = default_b.weight("w", f32([2048, 5504]));
        let plain = default_b.matmul(&x, &w);

        let mut opts_b = FuncBuilder::new("main", "t");
        let x2 = opts_b.input("x", f32([1, 512, 2048]));
        let w2 = opts_b.weight("w", f32([2048, 5504]));
        let with_opts = opts_b.matmul_with(&x2, &w2, &MatmulOpts::default());

        assert_eq!(plain.shape(), with_opts.shape());
        assert_eq!(default_b.emitter.body(), opts_b.emitter.body());
    }

    #[test]
    fn matmul_with_algorithm_emits_a_mutually_exclusive_algorithm_attribute() {
        use crate::matrix::{DotAlgorithm, DotNumerics};

        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", Shape::new([4, 8], DType::BF16));
        let w = b.weight("w", Shape::new([8, 16], DType::BF16));
        let opts = MatmulOpts {
            numerics: DotNumerics::Algorithm(DotAlgorithm::Bf16Bf16F32X3),
            accumulate: Some(DType::F32),
        };
        let y = b.matmul_with(&x, &w, &opts);

        assert_eq!(y.dtype(), DType::F32, "preferred_element_type must drive the output dtype");
        let body = b.emitter.body();
        assert!(body.contains("algorithm = #stablehlo.dot_algorithm<"), "{body}");
        assert!(body.contains("num_primitive_operations = 3"), "{body}");
        assert!(
            !body.contains("precision_config"),
            "algorithm and precision_config are mutually exclusive:\n{body}"
        );
        assert!(body.contains("preferred_element_type = f32"), "{body}");
    }

    /// ⛔ Regression test for ARTX02 §7's `matmul`, which would batch a rank-2
    /// weight over its own contraction axis.
    #[test]
    fn matmul_against_a_rank2_weight_uses_no_batch_dimensions() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", f32([1, 512, 2048]));
        let w = b.weight("w", f32([2048, 5504]));
        let y = b.matmul(&x, &w);

        assert_eq!(y.shape().dims, vec![1, 512, 5504]);
        let body = b.emitter.body();
        assert!(
            body.contains("lhs_batching_dimensions = [],"),
            "a [B,S,D] @ [D,F] projection must not batch:\n{body}"
        );
        assert!(body.contains("rhs_batching_dimensions = [],"), "{body}");
        assert!(body.contains("lhs_contracting_dimensions = [2],"), "{body}");
        assert!(body.contains("rhs_contracting_dimensions = [0]"), "{body}");
    }

    #[test]
    fn matmul_between_equal_ranks_batches_the_leading_dimensions() {
        let mut b = FuncBuilder::new("main", "t");
        let q = b.input("q", f32([1, 16, 512, 128]));
        let k = b.input("k", f32([1, 16, 128, 512]));
        let scores = b.matmul(&q, &k);

        assert_eq!(scores.shape().dims, vec![1, 16, 512, 512]);
        let body = b.emitter.body();
        assert!(body.contains("lhs_batching_dimensions = [0, 1],"), "{body}");
        assert!(body.contains("rhs_batching_dimensions = [0, 1],"), "{body}");
    }

    #[test]
    fn mixed_precision_matmul_widens_the_narrower_operand() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", Shape::new([4, 8], DType::F32));
        let w = b.weight("w", Shape::new([8, 16], DType::BF16));
        let y = b.matmul(&x, &w);

        assert_eq!(y.dtype(), DType::F32, "widening must not narrow to bf16");
        let body = b.emitter.body();
        assert!(
            body.contains(r#""stablehlo.convert"(%v1) : (tensor<8x16xbf16>) -> tensor<8x16xf32>"#),
            "the bf16 weight must be converted before the dot:\n{body}"
        );
    }

    #[test]
    fn promote_widens_within_a_family_and_refuses_across_families() {
        assert_eq!(promote(DType::F32, DType::BF16), Some(DType::F32));
        assert_eq!(promote(DType::BF16, DType::F64), Some(DType::F64));
        assert_eq!(promote(DType::I64, DType::I32), Some(DType::I64));
        assert_eq!(promote(DType::F32, DType::F32), Some(DType::F32));
        // Same width, different exponent range — neither is lossless.
        assert_eq!(promote(DType::BF16, DType::F16), None);
        // Float vs int.
        assert_eq!(promote(DType::F32, DType::I32), None);
    }

    #[test]
    fn convert_to_the_same_dtype_emits_nothing() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", f32([4]));
        let y = b.convert(&x, DType::F32);
        assert_eq!(y, x);
        assert!(b.emitter.body().is_empty(), "{}", b.emitter.body());
    }

    #[test]
    fn reduce_drops_the_reduced_dimensions() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", f32([1, 512, 2048]));
        let zero = b.constant_scalar(0.0, DType::F32);
        let sum = b.reduce_add(&x, &zero, &[2]);
        assert_eq!(sum.shape().dims, vec![1, 512]);
    }

    #[test]
    #[should_panic(expected = "init must be a rank-0 tensor")]
    fn reduce_rejects_a_non_scalar_init() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", f32([4, 8]));
        let bad_init = b.input("i", f32([8]));
        let _ = b.reduce_add(&x, &bad_init, &[1]);
    }

    #[test]
    #[should_panic(expected = "element count must be preserved")]
    fn reshape_rejects_a_different_element_count() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", f32([4, 8]));
        let _ = b.reshape(&x, vec![4, 9]);
    }

    #[test]
    #[should_panic(expected = "repeats index")]
    fn transpose_rejects_a_non_permutation() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", f32([2, 3, 4]));
        let _ = b.transpose(&x, vec![0, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "do not broadcast")]
    fn elementwise_add_rejects_mismatched_shapes() {
        let mut b = FuncBuilder::new("main", "t");
        let a = b.input("a", f32([4, 8]));
        let c = b.input("c", f32([4, 1]));
        let _ = b.add(&a, &c);
    }

    #[test]
    #[should_panic(expected = "duplicate parameter name")]
    fn duplicate_weight_names_are_rejected() {
        // Two weights with one checkpoint key means the loader binds one
        // tensor twice and never notices.
        let mut b = FuncBuilder::new("main", "t");
        let _ = b.weight("model.layers.0.mlp.up_proj.weight", f32([8]));
        let _ = b.weight("model.layers.0.mlp.up_proj.weight", f32([8]));
    }

    #[test]
    fn finish_declares_parameters_in_trace_order() {
        let mut b = FuncBuilder::new("main", "m");
        let x = b.input("x", f32([2, 4]));
        let w = b.weight("w", f32([4, 3]));
        let y = b.matmul(&x, &w);
        b.set_outputs(vec![y]);
        let built = b.finish();

        assert!(
            built
                .mlir
                .contains("func.func @main(%v0: tensor<2x4xf32>, %v1: tensor<4x3xf32>) -> (tensor<2x3xf32>)"),
            "{}",
            built.mlir
        );
        assert!(built.mlir.starts_with("module @m {\n"), "{}", built.mlir);
        assert!(
            built.mlir.contains(r#""func.return"(%v2) : (tensor<2x3xf32>) -> ()"#),
            "{}",
            built.mlir
        );
        assert_eq!(built.signature.inputs.len(), 1);
        assert_eq!(built.signature.weights.len(), 1);
        assert_eq!(
            built
                .signature
                .param_order
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec!["x", "w"]
        );
    }

    #[test]
    #[should_panic(expected = "no outputs declared")]
    fn finish_without_outputs_panics() {
        let b = FuncBuilder::new("main", "m");
        let _ = b.finish();
    }

    // ── Buffer donation (ARTX05 §6) ──────────────────────────────────────────

    #[test]
    fn alias_output_emits_a_tf_aliasing_output_attribute() {
        let mut b = FuncBuilder::new("main", "t");
        let cache = b.kv_cache("k_cache", f32([2, 4]));
        let other = b.input("x", f32([2, 4]));
        let updated = b.add(&cache, &other);
        b.alias_output(&cache, 0);
        b.set_outputs(vec![updated]);
        let built = b.finish();

        assert!(
            built
                .mlir
                .contains("tensor<2x4xf32> {tf.aliasing_output = 0 : i32}"),
            "{}",
            built.mlir
        );
        // Every other parameter must be unaffected.
        assert!(
            !built.mlir.contains("%v1: tensor<2x4xf32> {"),
            "only the kv-cache param may carry the attribute:\n{}",
            built.mlir
        );
        assert_eq!(built.signature.kv_caches.len(), 1);
        assert_eq!(built.signature.kv_caches[0].alias_output, Some(0));
        assert_eq!(built.signature.inputs.len(), 1, "kv_cache is not an input");
    }

    #[test]
    #[should_panic(expected = "is not a kv-cache parameter")]
    fn alias_output_rejects_a_plain_input() {
        let mut b = FuncBuilder::new("main", "t");
        let x = b.input("x", f32([2, 4]));
        b.alias_output(&x, 0);
    }

    #[test]
    #[should_panic(expected = "is not a kv-cache parameter")]
    fn alias_output_rejects_a_weight() {
        let mut b = FuncBuilder::new("main", "t");
        let w = b.weight("w", f32([2, 4]));
        b.alias_output(&w, 0);
    }

    #[test]
    #[should_panic(expected = "already aliased")]
    fn alias_output_rejects_a_second_alias_on_the_same_param() {
        let mut b = FuncBuilder::new("main", "t");
        let cache = b.kv_cache("k_cache", f32([2, 4]));
        b.alias_output(&cache, 0);
        b.alias_output(&cache, 1);
    }

    #[test]
    #[should_panic(expected = "but only")]
    fn finish_rejects_an_alias_output_index_out_of_range() {
        let mut b = FuncBuilder::new("main", "t");
        let cache = b.kv_cache("k_cache", f32([2, 4]));
        b.alias_output(&cache, 3);
        b.set_outputs(vec![cache]);
        let _ = b.finish();
    }

    #[test]
    #[should_panic(expected = "must have identical shape")]
    fn finish_rejects_an_alias_output_shape_mismatch() {
        let mut b = FuncBuilder::new("main", "t");
        let cache = b.kv_cache("k_cache", f32([2, 4]));
        let other = b.input("y", f32([4, 2]));
        b.alias_output(&cache, 0);
        b.set_outputs(vec![other]);
        let _ = b.finish();
    }

    /// The policy is read by ops, not by the builder — but a wave that ships
    /// the policy without anything reading it would be dead code, so this
    /// pins the wiring the ARTX03 ops will rely on.
    #[test]
    fn precision_policy_is_readable_during_a_trace() {
        precision::with_policy(precision::PrecisionPolicy::f64_oracle(), || {
            let mut b = FuncBuilder::new("main", "t");
            let x = b.input("x", Shape::new([4], precision::current().activation));
            assert_eq!(x.dtype(), DType::F64);
        });
    }
}
