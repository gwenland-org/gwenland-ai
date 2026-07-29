//! Binding safetensors weights to a traced signature.

use std::collections::BTreeSet;

use glcore::format::SafetensorsFile;

use crate::graph::Signature;
use crate::stablehlo::types::{DType, Shape};
use crate::GlError;

/// Somewhere weights can be read from by checkpoint key.
///
/// A trait rather than a concrete type so a `.gllm` reader can be added later
/// without touching the binding logic — see [`super::GLLM_STATUS`].
pub trait WeightSource {
    /// Every key the source can provide.
    fn keys(&self) -> BTreeSet<String>;
    /// Shape of a tensor, or `None` if absent.
    fn shape_of(&self, key: &str) -> Option<Vec<usize>>;
    /// The tensor's values, converted to `f32`.
    fn read_f32(&self, key: &str) -> Result<Vec<f32>, GlError>;
}

impl WeightSource for SafetensorsFile {
    fn keys(&self) -> BTreeSet<String> {
        self.tensors.keys().cloned().collect()
    }

    fn shape_of(&self, key: &str) -> Option<Vec<usize>> {
        self.tensors.get(key).map(|m| m.shape.clone())
    }

    fn read_f32(&self, key: &str) -> Result<Vec<f32>, GlError> {
        self.to_f32(key)
    }
}

/// One bound weight, ready to upload.
#[derive(Debug, Clone)]
pub struct BoundWeight {
    pub name: String,
    pub shape: Shape,
    pub data: Vec<f32>,
}

/// Matches a traced signature against a weight source and reads every weight,
/// in declaration order.
///
/// ⛔ **Refuses on any discrepancy** (P5), and reports *all* of them rather
/// than the first: a checkpoint that is one key away from fitting and a
/// checkpoint that is a different model entirely produce very different reports,
/// and the difference is what tells you which it is.
///
/// The four ways this fails, in the order they are checked:
///
/// 1. a traced weight the checkpoint does not have;
/// 2. a shape disagreement — including a **transposed** one, which is the
///    interesting case: `[896, 4864]` vs `[4864, 896]` has the same element
///    count and would load without complaint if only sizes were compared;
/// 3. a length that disagrees with the shape (a truncated file);
/// 4. checkpoint keys nothing in the trace consumes — a warning, not an error,
///    since tied embeddings and unused buffers legitimately produce these.
pub fn bind_safetensors(
    sig: &Signature,
    source: &dyn WeightSource,
) -> Result<Vec<BoundWeight>, GlError> {
    let available = source.keys();
    let mut missing = Vec::new();
    let mut mismatched = Vec::new();

    for w in &sig.weights {
        match source.shape_of(&w.name) {
            None => missing.push(w.name.clone()),
            Some(dims) if dims != w.shape.dims => mismatched.push(format!(
                "{}: trace wants {:?}, checkpoint has {:?}{}",
                w.name,
                w.shape.dims,
                dims,
                if is_transpose(&w.shape.dims, &dims) {
                    " (transposed — same element count, different layout)"
                } else {
                    ""
                }
            )),
            Some(_) => {}
        }
    }

    if !missing.is_empty() || !mismatched.is_empty() {
        let mut msg = String::from("checkpoint does not match the traced model");
        if !missing.is_empty() {
            msg.push_str(&format!(
                "\n  missing {} weight(s): {}",
                missing.len(),
                preview(&missing)
            ));
        }
        if !mismatched.is_empty() {
            msg.push_str(&format!(
                "\n  {} shape disagreement(s):\n    {}",
                mismatched.len(),
                mismatched.join("\n    ")
            ));
        }
        return Err(GlError::Engine(msg));
    }

    let traced: BTreeSet<&str> = sig.weights.iter().map(|w| w.name.as_str()).collect();
    let extra: Vec<String> = available
        .iter()
        .filter(|k| !traced.contains(k.as_str()))
        .cloned()
        .collect();
    if !extra.is_empty() {
        // Not an error: a tied-embedding checkpoint carries `lm_head.weight`
        // that the trace never declares, and optimiser state is common too.
        log::warn!(
            "checkpoint has {} tensor(s) the trace does not consume: {}",
            extra.len(),
            preview(&extra)
        );
    }

    let mut bound = Vec::with_capacity(sig.weights.len());
    for w in &sig.weights {
        let data = source.read_f32(&w.name)?;
        if data.len() != w.shape.numel() {
            return Err(GlError::ShapeMismatch {
                expected: w.shape.dims.clone(),
                got: vec![data.len()],
            });
        }
        bound.push(BoundWeight {
            name: w.name.clone(),
            shape: w.shape.clone(),
            data,
        });
    }
    Ok(bound)
}

/// True when `b` is `a` with two axes swapped and the same element count.
fn is_transpose(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() || a == b {
        return false;
    }
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_unstable();
    sb.sort_unstable();
    sa == sb
}

/// First few names, so an error about a 300-tensor model stays readable.
fn preview(names: &[String]) -> String {
    const MAX: usize = 5;
    if names.len() <= MAX {
        return names.join(", ");
    }
    format!("{}, … (+{} more)", names[..MAX].join(", "), names.len() - MAX)
}

/// Which dtype a bound weight should be uploaded as.
///
/// ⚠️ Weights are dequantized to `f32` on the host and uploaded as F32 or
/// BF16 — ARTX10 is explicitly **parked** for this sprint, so there is no
/// quantized upload path and none should be added here.
pub fn upload_dtype(traced: DType) -> DType {
    match traced {
        DType::BF16 => DType::BF16,
        _ => DType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{trace_forward, Qwen2Config};
    use std::collections::BTreeMap;

    /// A synthetic weight source — real safetensors files are `.gitignore`d
    /// and a test that needs a 1 GB download is a test that never runs.
    struct FakeSource(BTreeMap<String, (Vec<usize>, Vec<f32>)>);

    impl FakeSource {
        /// Every weight the trace declares, filled with zeros.
        fn matching(sig: &Signature) -> Self {
            FakeSource(
                sig.weights
                    .iter()
                    .map(|w| {
                        (
                            w.name.clone(),
                            (w.shape.dims.clone(), vec![0.0; w.shape.numel()]),
                        )
                    })
                    .collect(),
            )
        }
    }

    impl WeightSource for FakeSource {
        fn keys(&self) -> BTreeSet<String> {
            self.0.keys().cloned().collect()
        }
        fn shape_of(&self, key: &str) -> Option<Vec<usize>> {
            self.0.get(key).map(|(d, _)| d.clone())
        }
        fn read_f32(&self, key: &str) -> Result<Vec<f32>, GlError> {
            self.0
                .get(key)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| GlError::Parse(format!("no tensor {key}")))
        }
    }

    fn tiny_signature() -> Signature {
        crate::with_policy(crate::PrecisionPolicy::f32(), || {
            trace_forward(&Qwen2Config::tiny(), 4, 0)
        })
        .expect("trace")
        .signature
    }

    #[test]
    fn a_matching_checkpoint_binds_every_weight_in_order() {
        let sig = tiny_signature();
        let src = FakeSource::matching(&sig);
        let bound = bind_safetensors(&sig, &src).expect("must bind");

        assert_eq!(bound.len(), sig.weights.len());
        for (b, w) in bound.iter().zip(&sig.weights) {
            assert_eq!(b.name, w.name, "order must follow the trace");
            assert_eq!(b.shape, w.shape);
            assert_eq!(b.data.len(), w.shape.numel());
        }
    }

    #[test]
    fn a_missing_weight_is_refused_and_named() {
        let sig = tiny_signature();
        let mut src = FakeSource::matching(&sig);
        src.0.remove("model.layers.0.self_attn.q_proj.bias");
        let err = bind_safetensors(&sig, &src).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("missing 1 weight"), "{msg}");
        assert!(msg.contains("q_proj.bias"), "{msg}");
    }

    /// ⛔ The interesting failure: `[D, FFN]` vs `[FFN, D]` has the same
    /// element count, so a length check alone would accept it and every FFN in
    /// the model would compute against a transposed matrix.
    #[test]
    fn a_transposed_weight_is_refused_and_called_out_as_transposed() {
        let sig = tiny_signature();
        let mut src = FakeSource::matching(&sig);
        let key = "model.layers.0.mlp.gate_proj.weight";
        let (dims, data) = src.0.get(key).cloned().expect("present");
        src.0
            .insert(key.into(), (vec![dims[1], dims[0]], data));

        let err = bind_safetensors(&sig, &src).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("gate_proj.weight"), "{msg}");
        assert!(
            msg.contains("transposed"),
            "the report must say what kind of mismatch it is: {msg}"
        );
    }

    #[test]
    fn every_problem_is_reported_not_just_the_first() {
        let sig = tiny_signature();
        let mut src = FakeSource::matching(&sig);
        src.0.remove("model.norm.weight");
        src.0.remove("model.layers.1.mlp.up_proj.weight");
        let key = "model.layers.0.mlp.down_proj.weight";
        let (dims, data) = src.0.get(key).cloned().expect("present");
        src.0.insert(key.into(), (vec![dims[1], dims[0]], data));

        let err = bind_safetensors(&sig, &src).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("missing 2 weight"), "{msg}");
        assert!(msg.contains("1 shape disagreement"), "{msg}");
    }

    #[test]
    fn a_truncated_tensor_is_refused() {
        let sig = tiny_signature();
        let mut src = FakeSource::matching(&sig);
        let key = "model.norm.weight";
        let (dims, mut data) = src.0.get(key).cloned().expect("present");
        data.pop(); // shape still says N, data has N-1
        src.0.insert(key.into(), (dims, data));

        let err = bind_safetensors(&sig, &src).expect_err("must refuse");
        assert!(matches!(err, GlError::ShapeMismatch { .. }), "{err:?}");
    }

    /// Extra checkpoint tensors are normal — a tied-embedding model still ships
    /// `lm_head.weight` — so they warn rather than fail.
    #[test]
    fn unused_checkpoint_tensors_do_not_fail_the_load() {
        let sig = tiny_signature();
        let mut src = FakeSource::matching(&sig);
        src.0.insert("lm_head.weight".into(), (vec![2, 2], vec![0.0; 4]));
        src.0
            .insert("optimizer.step".into(), (vec![1], vec![0.0]));
        bind_safetensors(&sig, &src).expect("extra tensors must not fail the load");
    }

    #[test]
    fn transpose_detection_does_not_fire_on_unrelated_shapes() {
        assert!(is_transpose(&[2, 3], &[3, 2]));
        assert!(is_transpose(&[1, 4, 8], &[4, 1, 8]));
        assert!(!is_transpose(&[2, 3], &[2, 3]));
        assert!(!is_transpose(&[2, 3], &[6]));
        assert!(!is_transpose(&[2, 3], &[2, 4]));
    }
}
