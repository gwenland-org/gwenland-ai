//! Static KV cache primitives (ARTX05).
//!
//! # Where this sits
//!
//! ARTX03 §4 makes an explicit design decision: **gljax v1 has no KV cache**,
//! and a decode step recomputes the whole sequence. ARTX05 replaces that with a
//! pre-allocated cache written by `dynamic_update_slice` and read by
//! `dynamic_slice`.
//!
//! What is here is the second half's primitives, emitted and parse-verified.
//! Buffer donation (`FuncBuilder::kv_cache` + `FuncBuilder::alias_output`,
//! emitting a `tf.aliasing_output` argument attribute) is implemented and
//! parse-verified too — see `graph::builder`. What is **not** here is a
//! cache-aware Qwen2 trace: wiring the cache through attention changes the
//! compiled signature (two extra in/out parameters per layer, a dynamic RoPE
//! offset, a runtime position mask instead of the static causal one), and that
//! cannot be validated without a plugin. Landing it unmeasured would be
//! ARTX05's headline risk — a cache that reads one position off produces
//! fluent, wrong text.
//!
//! [`Session::generate`](crate::runtime::Session::generate) therefore uses the
//! ARTX03 recomputation path, which is correct-by-construction and slow.
//!
//! # Layout
//!
//! ```text
//! k_cache, v_cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
//! ```
//!
//! Sequence-major *within* a layer so a decode write is one contiguous slice at
//! `[layer, pos, .., ..]`, and a read is `[layer, 0..pos+1, .., ..]`.

use std::rc::Rc;

use crate::graph::value::SsaValue;
use crate::stablehlo::ops::{emit_dynamic_slice, emit_dynamic_update_slice};
use crate::stablehlo::types::{DType, Shape};
use crate::tensor::Tensor;

/// Shape of one cache tensor for a model.
pub fn cache_shape(
    n_layers: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
) -> Shape {
    Shape::new([n_layers, max_seq_len, n_kv_heads, head_dim], dtype)
}

/// Writes one layer's K (or V) for a single position.
///
/// * `cache`: `[n_layers, max_seq_len, n_kv_heads, head_dim]`
/// * `update`: `[1, 1, n_kv_heads, head_dim]`
/// * `layer`, `position`: rank-0 `i32` tensors — `position` is a **runtime**
///   value, which is the whole point.
///
/// ⛔ Returns a new cache value. That is not a copy in the compiled program
/// *if* the caller declares the cache donated: pass a `cache` tensor obtained
/// from `FuncBuilder::kv_cache`/`TraceCx::kv_cache`, and alias it to this
/// call's output index via `alias_output` before `finish()`. XLA then turns
/// the functional update into an in-place store. Without that declaration it
/// *is* a full copy of the cache every step, which at Qwen2-0.5B's 24 layers ×
/// 2048 positions is the difference between a working decode and an unusable
/// one.
pub fn write_at(
    cache: &Tensor,
    update: &Tensor,
    layer: &Tensor,
    position: &Tensor,
    zero: &Tensor,
) -> Tensor {
    assert_eq!(
        update.dim(1),
        1,
        "write_at writes one position at a time, got {} — use write_range for a \
         prefill's bulk fill",
        update.dim(1)
    );
    write_impl(cache, update, layer, position, zero)
}

/// Writes one layer's K (or V) for a **contiguous range** of positions —
/// prefill's bulk cache fill, as opposed to [`write_at`]'s single-position
/// decode write. `update`'s position axis (dim 1) is the range width; `start`
/// is where it begins.
///
/// Used once per layer per prefill call: the whole prompt's K/V (already
/// computed for the ordinary, uncached attention pass) is written into the
/// cache in one `dynamic_update_slice`, so decode steps afterward can read it
/// back.
pub fn write_range(
    cache: &Tensor,
    update: &Tensor,
    layer: &Tensor,
    start: &Tensor,
    zero: &Tensor,
) -> Tensor {
    write_impl(cache, update, layer, start, zero)
}

fn write_impl(
    cache: &Tensor,
    update: &Tensor,
    layer: &Tensor,
    position: &Tensor,
    zero: &Tensor,
) -> Tensor {
    assert_eq!(cache.rank(), 4, "kv cache must be rank 4");
    assert_eq!(update.rank(), 4, "kv update must be rank 4");
    assert_eq!(
        update.dim(0),
        1,
        "write_at/write_range write one layer at a time, got {}",
        update.dim(0)
    );
    assert!(
        update.dim(1) <= cache.dim(1),
        "kv update spans {} positions, which exceeds the cache's {}",
        update.dim(1),
        cache.dim(1)
    );
    assert_eq!(
        (update.dim(2), update.dim(3)),
        (cache.dim(2), cache.dim(3)),
        "kv update head layout must match the cache"
    );

    let idx = |t: &Tensor| (t.value().ssa(), t.shape().clone());
    let name = {
        let mut b = cache.builder().borrow_mut();
        emit_dynamic_update_slice(
            b.emitter_mut(),
            cache.value().ssa(),
            update.value().ssa(),
            &[idx(layer), idx(position), idx(zero), idx(zero)],
            cache.shape(),
            update.shape(),
        )
    };
    Tensor::new(
        SsaValue::new(name, cache.shape().clone()),
        Rc::clone(cache.builder()),
    )
}

/// Reads a fixed-size window of one layer's cache.
///
/// * returns `[1, window, n_kv_heads, head_dim]`
///
/// ⚠️ `window` is **static** — it is a compiled shape, so a decode step reads
/// the whole bucket, not just the filled prefix. Positions past the write head
/// hold whatever the cache was initialised with, and it is the causal mask's
/// job to make them harmless. Reading "only what is filled" would be a dynamic
/// shape, which P3 forbids.
pub fn read_window(
    cache: &Tensor,
    layer: &Tensor,
    start: &Tensor,
    zero: &Tensor,
    window: usize,
) -> Tensor {
    assert_eq!(cache.rank(), 4, "kv cache must be rank 4");
    assert!(
        window <= cache.dim(1),
        "read window {window} exceeds the cache's {} positions",
        cache.dim(1)
    );

    let out_shape = Shape::new(
        [1, window, cache.dim(2), cache.dim(3)],
        cache.shape().dtype,
    );
    let idx = |t: &Tensor| (t.value().ssa(), t.shape().clone());
    let name = {
        let mut b = cache.builder().borrow_mut();
        emit_dynamic_slice(
            b.emitter_mut(),
            cache.value().ssa(),
            &[idx(layer), idx(start), idx(zero), idx(zero)],
            &[1, window, cache.dim(2), cache.dim(3)],
            cache.shape(),
            &out_shape,
        )
    };
    Tensor::new(
        SsaValue::new(name, out_shape),
        Rc::clone(cache.builder()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;

    fn i32_scalar(cx: &mut TraceCx, name: &str) -> Tensor {
        cx.input(name, Shape::scalar(DType::I32))
    }

    #[test]
    fn cache_shape_is_layer_major_then_position() {
        let s = cache_shape(24, 2048, 2, 64, DType::F32);
        assert_eq!(s.dims, vec![24, 2048, 2, 64]);
        // Qwen2-0.5B full-context KV, both tensors, f32.
        assert_eq!(s.byte_len() * 2, 24 * 2048 * 2 * 64 * 4 * 2);
    }

    #[test]
    fn a_decode_write_targets_one_layer_and_one_position() {
        let mut cx = TraceCx::new("main", "kv");
        let cache = cx.input("k_cache", cache_shape(2, 16, 2, 8, DType::F32));
        let update = cx.input("k_new", Shape::new([1, 1, 2, 8], DType::F32));
        let layer = i32_scalar(&mut cx, "layer");
        let pos = i32_scalar(&mut cx, "pos");
        let zero = i32_scalar(&mut cx, "zero");

        let out = write_at(&cache, &update, &layer, &pos, &zero);
        assert_eq!(out.shape().dims, cache.shape().dims, "shape is preserved");

        let mlir = cx.finish(&[&out]).mlir;
        assert!(mlir.contains(r#""stablehlo.dynamic_update_slice""#), "{mlir}");
        // Four runtime start indices, one per cache axis.
        assert!(
            mlir.contains(
                "(tensor<2x16x2x8xf32>, tensor<1x1x2x8xf32>, tensor<i32>, tensor<i32>, \
                 tensor<i32>, tensor<i32>) -> tensor<2x16x2x8xf32>"
            ),
            "{mlir}"
        );
    }

    #[test]
    fn a_read_returns_a_static_window() {
        let mut cx = TraceCx::new("main", "kv");
        let cache = cx.input("k_cache", cache_shape(2, 16, 2, 8, DType::F32));
        let layer = i32_scalar(&mut cx, "layer");
        let start = i32_scalar(&mut cx, "start");
        let zero = i32_scalar(&mut cx, "zero");

        let win = read_window(&cache, &layer, &start, &zero, 16);
        assert_eq!(win.shape().dims, vec![1, 16, 2, 8]);

        let mlir = cx.finish(&[&win]).mlir;
        assert!(mlir.contains(r#""stablehlo.dynamic_slice""#), "{mlir}");
        assert!(
            mlir.contains("slice_sizes = array<i64: 1, 16, 2, 8>"),
            "{mlir}"
        );
    }

    #[test]
    #[should_panic(expected = "one position at a time")]
    fn a_multi_position_update_is_refused() {
        let mut cx = TraceCx::new("main", "kv");
        let cache = cx.input("k_cache", cache_shape(2, 16, 2, 8, DType::F32));
        let update = cx.input("k_new", Shape::new([1, 4, 2, 8], DType::F32));
        let layer = i32_scalar(&mut cx, "layer");
        let pos = i32_scalar(&mut cx, "pos");
        let zero = i32_scalar(&mut cx, "zero");
        let _ = write_at(&cache, &update, &layer, &pos, &zero);
    }

    #[test]
    fn write_range_allows_a_multi_position_bulk_update() {
        let mut cx = TraceCx::new("main", "kv");
        let cache = cx.input("k_cache", cache_shape(2, 16, 2, 8, DType::F32));
        let update = cx.input("k_new", Shape::new([1, 4, 2, 8], DType::F32));
        let layer = i32_scalar(&mut cx, "layer");
        let start = i32_scalar(&mut cx, "start");
        let zero = i32_scalar(&mut cx, "zero");

        let out = write_range(&cache, &update, &layer, &start, &zero);
        assert_eq!(out.shape().dims, cache.shape().dims, "shape is preserved");

        let mlir = cx.finish(&[&out]).mlir;
        assert!(mlir.contains(r#""stablehlo.dynamic_update_slice""#), "{mlir}");
        assert!(
            mlir.contains(
                "(tensor<2x16x2x8xf32>, tensor<1x4x2x8xf32>, tensor<i32>, tensor<i32>, \
                 tensor<i32>, tensor<i32>) -> tensor<2x16x2x8xf32>"
            ),
            "{mlir}"
        );
    }

    #[test]
    #[should_panic(expected = "exceeds the cache's")]
    fn write_range_refuses_a_span_wider_than_the_cache() {
        let mut cx = TraceCx::new("main", "kv");
        let cache = cx.input("k_cache", cache_shape(2, 16, 2, 8, DType::F32));
        let update = cx.input("k_new", Shape::new([1, 17, 2, 8], DType::F32));
        let layer = i32_scalar(&mut cx, "layer");
        let start = i32_scalar(&mut cx, "start");
        let zero = i32_scalar(&mut cx, "zero");
        let _ = write_range(&cache, &update, &layer, &start, &zero);
    }

    #[test]
    #[should_panic(expected = "exceeds the cache's")]
    fn a_window_larger_than_the_cache_is_refused() {
        let mut cx = TraceCx::new("main", "kv");
        let cache = cx.input("k_cache", cache_shape(2, 16, 2, 8, DType::F32));
        let layer = i32_scalar(&mut cx, "layer");
        let start = i32_scalar(&mut cx, "start");
        let zero = i32_scalar(&mut cx, "zero");
        let _ = read_window(&cache, &layer, &start, &zero, 17);
    }
}
