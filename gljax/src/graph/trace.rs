//! `TraceCx` — the tracing context and the scope stack (ARTX02 §6).

use std::cell::RefCell;
use std::rc::Rc;

use crate::graph::builder::{BuiltFunc, FuncBuilder};
use crate::graph::value::SsaValue;
use crate::stablehlo::types::Shape;
use crate::tensor::Tensor;

/// The user-facing tracing context.
///
/// Wraps [`FuncBuilder`] in `Rc<RefCell<_>>` so every [`Tensor`] can push ops
/// without the caller managing borrows.
///
/// ⚠️ **`Rc<RefCell<_>>`, not `Arc<Mutex<_>>`** (ARTX02 §6). Tracing is
/// single-threaded by construction — [`crate::stablehlo::MlirEmitter`] is
/// `!Send` precisely so this cannot quietly become a shared-mutable-state
/// problem. Parallel tracing means two independent `TraceCx`es, not one shared
/// one.
pub struct TraceCx {
    builder: Rc<RefCell<FuncBuilder>>,
    scope_stack: Vec<String>,
}

impl TraceCx {
    pub fn new(func_name: impl Into<String>, module_name: impl Into<String>) -> Self {
        TraceCx {
            builder: Rc::new(RefCell::new(FuncBuilder::new(func_name, module_name))),
            scope_stack: Vec::new(),
        }
    }

    // ── Scopes ─────────────────────────────────────────────────────────────

    /// Runs `f` inside a named scope.
    ///
    /// ⭐ The scope stack is not decoration — it is how weight names are
    /// produced, and those names **are** the checkpoint keys (ARTX02 §6). A
    /// trace that walks `model.layers.0` → `self_attn` → `q_proj.weight`
    /// yields `model.layers.0.self_attn.q_proj.weight`, which is the
    /// safetensors key verbatim, so ARTX04's loader needs no mapping table.
    /// A mapping table is a thing that can drift out of sync; a name that is
    /// already right cannot.
    ///
    /// The closure form is deliberate: it cannot leak a scope the way a
    /// push/pop pair can when the body returns early.
    pub fn scope<T>(&mut self, name: impl Into<String>, f: impl FnOnce(&mut TraceCx) -> T) -> T {
        self.scope_stack.push(name.into());
        let result = f(self);
        self.scope_stack
            .pop()
            .expect("scope stack was emptied inside a scope body");
        result
    }

    /// The current scope path, e.g. `model.layers.0.self_attn`.
    pub fn scope_path(&self) -> String {
        self.scope_stack.join(".")
    }

    /// Qualifies `name` with the current scope.
    pub fn qualify(&self, name: &str) -> String {
        if self.scope_stack.is_empty() {
            return name.to_owned();
        }
        format!("{}.{name}", self.scope_stack.join("."))
    }

    // ── Parameters ─────────────────────────────────────────────────────────

    /// Declares a runtime input (tokens, positions, masks).
    ///
    /// Input names are **not** scope-qualified: they are the caller's ABI, not
    /// checkpoint keys, and a caller should not have to know where in the model
    /// an input happened to be declared.
    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        let value = self.builder.borrow_mut().input(name, shape);
        Tensor::new(value, Rc::clone(&self.builder))
    }

    /// Declares a checkpoint weight in the current scope. The name **is**
    /// scope-qualified.
    pub fn weight(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        let qualified = self.qualify(&name.into());
        let value = self.builder.borrow_mut().weight(qualified, shape);
        Tensor::new(value, Rc::clone(&self.builder))
    }

    /// Declares a KV cache buffer (ARTX05). Not scope-qualified — see
    /// [`FuncBuilder::kv_cache`](crate::graph::builder::FuncBuilder::kv_cache).
    pub fn kv_cache(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        let value = self.builder.borrow_mut().kv_cache(name, shape);
        Tensor::new(value, Rc::clone(&self.builder))
    }

    /// Donates `param` to output `output_index` — see
    /// [`FuncBuilder::alias_output`](crate::graph::builder::FuncBuilder::alias_output).
    pub fn alias_output(&mut self, param: &Tensor, output_index: usize) {
        self.builder
            .borrow_mut()
            .alias_output(param.value(), output_index);
    }

    /// The builder, for ops that need to emit directly.
    pub fn builder(&self) -> &Rc<RefCell<FuncBuilder>> {
        &self.builder
    }

    // ── Finish ─────────────────────────────────────────────────────────────

    /// Assembles the module with the given tensors as outputs.
    ///
    /// ⛔ **Deviation from ARTX02 §6**, which does
    /// `Rc::try_unwrap(self.builder)`. That can only succeed when no [`Tensor`]
    /// is alive — but the outputs being passed in *are* live tensors, so
    /// ARTX02 §9's own `cx.finish(vec![&out])` would panic every time. gljax
    /// borrows the builder instead, which works no matter how many handles
    /// are outstanding.
    pub fn finish(self, outputs: &[&Tensor]) -> BuiltFunc {
        let out_values: Vec<SsaValue> = outputs.iter().map(|t| t.value().clone()).collect();
        self.builder.borrow_mut().set_outputs(out_values);
        self.builder.borrow().finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stablehlo::types::DType;

    #[test]
    fn nested_scopes_produce_dotted_checkpoint_keys() {
        let mut cx = TraceCx::new("main", "m");
        let w = cx.scope("model", |cx| {
            cx.scope("layers.0", |cx| {
                cx.scope("self_attn", |cx| {
                    cx.weight("q_proj.weight", Shape::new([2048, 2048], DType::BF16))
                })
            })
        });
        let built = cx.finish(&[&w]);
        assert_eq!(
            built.signature.weights[0].name,
            "model.layers.0.self_attn.q_proj.weight",
            "the traced name must be the safetensors key verbatim"
        );
    }

    /// ⛔ Regression test for ARTX02 §6's `Rc::try_unwrap`, which made
    /// `finish` panic whenever an output tensor was still alive — i.e. always.
    #[test]
    fn finish_works_while_intermediate_tensors_are_still_alive() {
        let mut cx = TraceCx::new("main", "m");
        let x = cx.input("x", Shape::new([2, 4], DType::F32));
        let w = cx.weight("w", Shape::new([4, 3], DType::F32));
        let y = x.matmul(&w);
        // x, w and y are all live handles here — three `Rc` clones plus the
        // context's own.
        let built = cx.finish(&[&y]);
        assert!(built.mlir.contains("stablehlo.dot_general"), "{}", built.mlir);
        assert_eq!(built.signature.outputs[0].dims, vec![2, 3]);
    }

    #[test]
    fn scopes_pop_when_the_body_returns() {
        let mut cx = TraceCx::new("main", "m");
        cx.scope("a", |cx| {
            cx.scope("b", |cx| {
                assert_eq!(cx.scope_path(), "a.b");
            });
            assert_eq!(cx.scope_path(), "a");
        });
        assert_eq!(cx.scope_path(), "");
        assert_eq!(cx.qualify("w"), "w", "unqualified at the root scope");
    }

    #[test]
    fn inputs_are_not_scope_qualified() {
        let mut cx = TraceCx::new("main", "m");
        let x = cx.scope("model", |cx| {
            cx.scope("layers.0", |cx| {
                cx.input("hidden_states", Shape::new([1, 4], DType::F32))
            })
        });
        let built = cx.finish(&[&x]);
        assert_eq!(built.signature.inputs[0].name, "hidden_states");
    }
}
