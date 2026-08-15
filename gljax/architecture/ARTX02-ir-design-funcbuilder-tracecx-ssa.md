# ARTX2 — gljax IR Design: FuncBuilder, TraceCx, and SSA Value System

**Date**: July 2026  
**Codename**: Sanctum Visibilia  
**Depends on**: ARTX1 (PJRT + StableHLO Research)  
**Scope**: Complete design for `src/graph/`, `src/stablehlo/`, and `src/tensor/tensor.rs`

## Research Basis

All StableHLO op syntax in this document is derived directly from the official
**StableHLO Specification** at [openxla.org/stablehlo/spec](https://openxla.org/stablehlo/spec)
(fetched July 2026, last updated 2026-04-10 UTC). Rust architecture decisions are
grounded in the `fusebox`/ZML pattern (Section 8 of ARTX1) and standard Rust
compiler IR patterns. The `gomlx/stablehlo` Go builder (github.com/gomlx/stablehlo)
was also consulted as a cross-reference for attribute format patterns.

### Critical spec finding: Two valid MLIR text formats

The StableHLO spec defines a **generic** op syntax used in all spec examples:
```mlir
%result = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
```
And a **pretty** (sugar) syntax accepted by the MLIR parser:
```mlir
%result = stablehlo.add %lhs, %rhs : tensor<2x2xi32>
```
PJRT's `PJRT_Client_Compile` accepts both. The generic form is more explicit and
harder to get wrong. The pretty form is shorter. gljax emits **generic form** to
avoid ambiguity — the MLIR parser never rejects it and it maps 1:1 to the spec.

---

## Table of Contents

1. [MlirEmitter — `src/stablehlo/emitter.rs`](#1-mliremitter)
2. [Type Formatting — `src/stablehlo/types.rs`](#2-type-formatting)
3. [Op Emitters — `src/stablehlo/ops.rs`](#3-op-emitters)
4. [SsaValue — `src/graph/value.rs`](#4-ssavalue)
5. [FuncBuilder — `src/graph/builder.rs`](#5-funcbuilder)
6. [TraceCx — `src/graph/trace.rs`](#6-tracecx)
7. [Tensor — `src/tensor/tensor.rs`](#7-tensor)
8. [Precision Integration](#8-precision-integration)
9. [End-to-End Trace Example](#9-end-to-end-trace-example)
10. [What ARTX3 Should Cover](#10-what-artx3-should-cover)
11. [Design Decision Summary](#11-design-decision-summary)

---

## 1. MlirEmitter

### Role

`MlirEmitter` owns the output `String` buffer and an SSA counter. It is a
pure text-emission layer: no knowledge of tensor shapes, no logic — just string
formatting and SSA name allocation.

### Design Decisions

⚠️ **DESIGN DECISION — !Send by design**  
A single forward-pass trace is single-threaded. Making `MlirEmitter` `Send` would
require `Arc<Mutex<>>` around the SSA counter and buffer for zero benefit. If you
ever need parallel tracing (pipeline parallelism), do it by running two separate
`TraceCx` instances on separate threads and composing their `BuiltFunc` outputs.

⚠️ **DESIGN DECISION — Emitter owns one func.func body only**  
`MlirEmitter` emits the body lines (indented at level 1). The `func.func @main(...)`
header, return statement, and `module {}` wrapper are assembled by `FuncBuilder::finish()`.
This separation allows nested emitters for regions (e.g., `while` body, `reduce`
computation body) — each region gets its own `MlirEmitter` at a deeper indent.

```rust
// src/stablehlo/emitter.rs

/// Emits lines of a single StableHLO func.func body.
/// Not Send. Single-threaded use only.
pub struct MlirEmitter {
    buf:         String,
    ssa_counter: u32,
    indent:      usize,  // 1 = inside func body (2 spaces per level)
}

/// A raw SSA identifier. Displays as `%vN`.
/// This is the name-only type. SsaValue (in graph/) adds Shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SsaName(pub u32);

impl std::fmt::Display for SsaName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%v{}", self.0)
    }
}

impl MlirEmitter {
    pub fn new() -> Self {
        Self {
            buf:         String::with_capacity(64 * 1024),
            ssa_counter: 0,
            indent:      1,
        }
    }

    /// Allocate a fresh SSA name. Does NOT emit anything.
    pub fn fresh(&mut self) -> SsaName {
        let id = self.ssa_counter;
        self.ssa_counter += 1;
        SsaName(id)
    }

    /// Emit an indented line, appending '\n'.
    pub fn line(&mut self, s: impl std::fmt::Display) {
        use std::fmt::Write;
        for _ in 0..self.indent { self.buf.push_str("  "); }
        writeln!(self.buf, "{s}").unwrap();
    }

    pub fn push_indent(&mut self) { self.indent += 1; }
    pub fn pop_indent(&mut self)  { self.indent = self.indent.saturating_sub(1); }

    /// Consume the emitter and return the accumulated body lines.
    /// Does NOT include func.func header or module wrapper.
    pub fn into_body(self) -> String { self.buf }

    /// Peek at counter (for tests).
    pub fn ssa_count(&self) -> u32 { self.ssa_counter }
}
```

### Module structure assembled by FuncBuilder::finish()

The final text passed to PJRT looks like:
```mlir
module @model {
  func.func @main(%v0: tensor<...>, %v1: tensor<...>) -> (tensor<...>) {
    [lines from MlirEmitter::into_body()]
    "func.return"(%vN) : (tensor<...>) -> ()
  }
}
```

---

## 2. Type Formatting

Source: [StableHLO spec §Types](https://openxla.org/stablehlo/spec#types)

The spec defines:
```
TensorType       ::= 'tensor' '<' Shape TensorElementType '>'
Shape            ::= {DimensionSize 'x'}
DimensionSize    ::= digit {digit} | '?'
TensorElementType ::= BooleanType | IntegerType | FloatType | ...
BooleanType      ::= 'i1'
SignedIntegerType ::= 'si2' | 'si4' | 'si8' | 'si16' | 'si32' | 'si64'
FloatType        ::= 'bf16' | 'f16' | 'f32' | 'f64' | ...
```

Note: The spec uses **signed integer** types (`si8`, `si32`, `si64`) for index/shape
computation. For token ids and general integer tensors gljax uses `si32` or `si64`.
`i32` / `i64` (no sign prefix) is accepted by the MLIR parser as unsigned by default
in practice, but the canonical spec form is `si`/`ui` prefixed. For simplicity and
broad compatibility, gljax uses `i32` / `i64` in emission (matching what JAX/PyTorch
actually emit and what PJRT accepts).

⚠️ **DESIGN DECISION — Static dims only in v1**  
`Shape::dims` is `Vec<usize>`. Dynamic dims (`?`) deferred to v2.

```rust
// src/stablehlo/types.rs

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F64, F32, BF16, F16,
    I64, I32, I16, I8,
    Bool,  // i1 in MLIR
}

impl DType {
    pub const fn mlir_str(self) -> &'static str {
        match self {
            DType::F64  => "f64",
            DType::F32  => "f32",
            DType::BF16 => "bf16",
            DType::F16  => "f16",
            DType::I64  => "i64",
            DType::I32  => "i32",
            DType::I16  => "i16",
            DType::I8   => "i8",
            DType::Bool => "i1",
        }
    }

    pub const fn byte_size(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::BF16 | DType::F16 | DType::I16 => 2,
            DType::I8 | DType::Bool => 1,
        }
    }
}

/// A static tensor shape: dims + element dtype.
/// gljax v1 — no dynamic dims.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dims:  Vec<usize>,
    pub dtype: DType,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Self { dims: dims.into(), dtype }
    }

    /// 0-dimensional tensor ("scalar tensor" in StableHLO).
    /// Spec says scalars are NOT first-class; use 0-dim tensors.
    /// MLIR text: tensor<f32>
    pub fn scalar(dtype: DType) -> Self {
        Self { dims: vec![], dtype }
    }

    pub fn numel(&self) -> usize { self.dims.iter().product() }
    pub fn byte_len(&self) -> usize { self.numel() * self.dtype.byte_size() }
    pub fn rank(&self) -> usize { self.dims.len() }

    /// Formats as MLIR TensorType string.
    /// Spec: tensor<2x4xf32>, tensor<f32> (0-dim), tensor<128xbf16>
    pub fn mlir_type(&self) -> String {
        if self.dims.is_empty() {
            return format!("tensor<{}>", self.dtype.mlir_str());
        }
        let dim_str: Vec<String> = self.dims.iter().map(|d| d.to_string()).collect();
        format!("tensor<{}x{}>", dim_str.join("x"), self.dtype.mlir_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind { Input, Weight }

#[derive(Clone, Debug)]
pub struct ParamDesc {
    pub name:  String,
    pub shape: Shape,
    pub kind:  ParamKind,
    pub ssa:   crate::stablehlo::emitter::SsaName,
}
```

---

## 3. Op Emitters

Source: [StableHLO spec §Ops](https://openxla.org/stablehlo/spec#ops)

All op examples below are in the **generic form** used by the spec. This is what
gljax emits. The spec defines the signature as:
```
Op ::= [OpOutputs] OpName OpInputs ':' OpSignature
OpName ::= '"' 'stablehlo' '.' OpMnemonic '"'
OpSignature ::= '(' [ValueType {',' ValueType}] ')' '->' '(' [ValueType {',' ValueType}] ')'
```

Attributes are provided in curly-brace blocks after inputs.

### Elementwise ops

From the spec:
```mlir
// add (spec example)
%result = "stablehlo.add"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>

// multiply (spec example)  
%result = "stablehlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>

// subtract (spec example)
%result = "stablehlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)

// negate (spec example)
%result = "stablehlo.negate"(%operand) : (tensor<2xi32>) -> tensor<2xi32>

// rsqrt (spec example)
%result = "stablehlo.rsqrt"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>

// logistic = sigmoid (spec example)
%result = "stablehlo.logistic"(%operand) : (tensor<2x2xf64>) -> tensor<2x2xf64>

// exponential (spec example — note: NOT "exp", it's "exponential")
%result = "stablehlo.exponential"(%operand) : (tensor<2x2xf64>) -> tensor<2x2xf64>

// maximum (spec example)
%result = "stablehlo.maximum"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>

// convert (spec example)
%result = "stablehlo.convert"(%operand) : (tensor<3xi64>) -> tensor<3xcomplex<f64>>
```

```rust
// src/stablehlo/ops.rs
// All function signatures: take SsaName inputs + Shape refs, emit to &mut MlirEmitter,
// return output SsaName.

use crate::stablehlo::{emitter::{MlirEmitter, SsaName}, types::Shape};

// ── Elementwise binary ──────────────────────────────────────────────────────

macro_rules! emit_binary_op {
    ($fn_name:ident, $mnemonic:literal) => {
        pub fn $fn_name(
            e: &mut MlirEmitter,
            lhs: SsaName, rhs: SsaName, ty: &Shape,
        ) -> SsaName {
            let out = e.fresh();
            e.line(format!(
                r#"{out} = "stablehlo.{mnemonic}"({lhs}, {rhs}) : ({t}, {t}) -> {t}"#,
                mnemonic = $mnemonic, t = ty.mlir_type(),
            ));
            out
        }
    };
}

emit_binary_op!(emit_add,      "add");
emit_binary_op!(emit_subtract, "subtract");
emit_binary_op!(emit_multiply, "multiply");
emit_binary_op!(emit_divide,   "divide");
emit_binary_op!(emit_maximum,  "maximum");
emit_binary_op!(emit_minimum,  "minimum");

// ── Elementwise unary ───────────────────────────────────────────────────────

macro_rules! emit_unary_op {
    ($fn_name:ident, $mnemonic:literal) => {
        pub fn $fn_name(e: &mut MlirEmitter, operand: SsaName, ty: &Shape) -> SsaName {
            let out = e.fresh();
            e.line(format!(
                r#"{out} = "stablehlo.{mnemonic}"({operand}) : ({t}) -> {t}"#,
                mnemonic = $mnemonic, t = ty.mlir_type(),
            ));
            out
        }
    };
}

emit_unary_op!(emit_negate,      "negate");
emit_unary_op!(emit_rsqrt,       "rsqrt");
emit_unary_op!(emit_sqrt,        "sqrt");
emit_unary_op!(emit_logistic,    "logistic");   // sigmoid
emit_unary_op!(emit_exponential, "exponential"); // exp
emit_unary_op!(emit_log,         "log");
emit_unary_op!(emit_tanh,        "tanh");
emit_unary_op!(emit_abs,         "abs");
```


### Convert

Spec: `"stablehlo.convert"(%operand) : (TIn) -> TOut`

```rust
/// Precision cast. In and out shapes must have same dims, different dtype.
pub fn emit_convert(
    e: &mut MlirEmitter,
    operand: SsaName, in_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.convert"({operand}) : ({}) -> {}"#,
        in_shape.mlir_type(), out_shape.mlir_type()
    ));
    out
}
```

### Constant

Spec example:
```mlir
%output = "stablehlo.constant"() {
  value = dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
} : () -> tensor<2x2xf32>
```

Scalars are 0-dim tensors. `dense<0.0> : tensor<f32>` is the canonical form for
a scalar constant. For a splat (all elements equal), `dense<VALUE>` works for any
shape.

```rust
/// Scalar constant: dense<VALUE> : tensor<dtype>
pub fn emit_constant_scalar(e: &mut MlirEmitter, value: f64, dtype: DType) -> SsaName {
    use crate::stablehlo::types::DType;
    let out = e.fresh();
    let ty = Shape::scalar(dtype);
    let val_str = match dtype {
        DType::F64 | DType::F32 | DType::BF16 | DType::F16 => format!("{value}"),
        DType::I64 | DType::I32 | DType::I16 | DType::I8   => format!("{}", value as i64),
        DType::Bool => if value != 0.0 { "true".to_string() } else { "false".to_string() },
    };
    e.line(format!(
        r#"{out} = "stablehlo.constant"() {{value = dense<{val_str}> : {}}} : () -> {}"#,
        ty.mlir_type(), ty.mlir_type()
    ));
    out
}

/// Splat constant: all elements = value. Works for any shape.
pub fn emit_constant_splat(e: &mut MlirEmitter, value: f64, shape: &Shape) -> SsaName {
    use crate::stablehlo::types::DType;
    let out = e.fresh();
    let val_str = match shape.dtype {
        DType::F64 | DType::F32 | DType::BF16 | DType::F16 => format!("{value}"),
        DType::I64 | DType::I32 | DType::I16 | DType::I8   => format!("{}", value as i64),
        DType::Bool => if value != 0.0 { "true".to_string() } else { "false".to_string() },
    };
    e.line(format!(
        r#"{out} = "stablehlo.constant"() {{value = dense<{val_str}> : {}}} : () -> {}"#,
        shape.mlir_type(), shape.mlir_type()
    ));
    out
}
```

### Reshape

Spec example:
```mlir
%result = "stablehlo.reshape"(%operand) : (tensor<2x3xi32>) -> tensor<3x2xi32>
```
Note: no attributes. The output shape is entirely expressed in the type signature.

```rust
pub fn emit_reshape(
    e: &mut MlirEmitter,
    operand: SsaName, in_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.reshape"({operand}) : ({}) -> {}"#,
        in_shape.mlir_type(), out_shape.mlir_type()
    ));
    out
}
```

### Transpose

Spec example:
```mlir
%result = "stablehlo.transpose"(%operand) {
  permutation = array<i64: 2, 1, 0>
} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
```

```rust
pub fn emit_transpose(
    e: &mut MlirEmitter,
    operand: SsaName, permutation: &[usize],
    in_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let perm: Vec<String> = permutation.iter().map(|x| x.to_string()).collect();
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.transpose"({operand}) {{permutation = array<i64: {}>}} : ({}) -> {}"#,
        perm.join(", "), in_shape.mlir_type(), out_shape.mlir_type()
    ));
    out
}
```

### Slice

Spec example:
```mlir
%result = "stablehlo.slice"(%operand) {
  start_indices = array<i64: 1, 2>,
  limit_indices = array<i64: 3, 4>,
  strides = array<i64: 1, 1>
} : (tensor<3x4xi64>) -> tensor<2x2xi64>
```

```rust
pub fn emit_slice(
    e: &mut MlirEmitter,
    operand: SsaName,
    starts: &[usize], limits: &[usize], strides: &[usize],
    in_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let fmt_arr = |v: &[usize]| -> String {
        v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
    };
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.slice"({operand}) {{start_indices = array<i64: {}>, limit_indices = array<i64: {}>, strides = array<i64: {}>}} : ({}) -> {}"#,
        fmt_arr(starts), fmt_arr(limits), fmt_arr(strides),
        in_shape.mlir_type(), out_shape.mlir_type()
    ));
    out
}
```

### Concatenate

Spec example:
```mlir
%result = "stablehlo.concatenate"(%input0, %input1) {
  dimension = 0 : i64
} : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
```

```rust
pub fn emit_concatenate(
    e: &mut MlirEmitter,
    operands: &[(SsaName, Shape)],
    dimension: usize,
    out_shape: &Shape,
) -> SsaName {
    let op_names: Vec<String> = operands.iter().map(|(n, _)| n.to_string()).collect();
    let in_types: Vec<String> = operands.iter().map(|(_, s)| s.mlir_type()).collect();
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.concatenate"({}) {{dimension = {} : i64}} : ({}) -> {}"#,
        op_names.join(", "), dimension,
        in_types.join(", "), out_shape.mlir_type()
    ));
    out
}
```

### Broadcast in Dim

Spec example:
```mlir
%result = "stablehlo.broadcast_in_dim"(%operand) {
  broadcast_dimensions = array<i64: 2, 1>
} : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
```

```rust
pub fn emit_broadcast_in_dim(
    e: &mut MlirEmitter,
    operand: SsaName, broadcast_dims: &[usize],
    in_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let dims: Vec<String> = broadcast_dims.iter().map(|x| x.to_string()).collect();
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.broadcast_in_dim"({operand}) {{broadcast_dimensions = array<i64: {}>}} : ({}) -> {}"#,
        dims.join(", "), in_shape.mlir_type(), out_shape.mlir_type()
    ));
    out
}
```

### Dot General

Spec example:
```mlir
%result = "stablehlo.dot_general"(%lhs, %rhs) {
  dot_dimension_numbers = #stablehlo.dot<
    lhs_batching_dimensions = [0],
    rhs_batching_dimensions = [0],
    lhs_contracting_dimensions = [2],
    rhs_contracting_dimensions = [1]
  >,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
} : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
```

```rust
#[derive(Debug, Clone, Default)]
pub struct DotDimensionNumbers {
    pub lhs_batching:    Vec<usize>,
    pub rhs_batching:    Vec<usize>,
    pub lhs_contracting: Vec<usize>,
    pub rhs_contracting: Vec<usize>,
}

fn fmt_i64_arr(v: &[usize]) -> String {
    let s: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", s.join(", "))
}

pub fn emit_dot_general(
    e: &mut MlirEmitter,
    lhs: SsaName, rhs: SsaName,
    dnums: &DotDimensionNumbers,
    lhs_shape: &Shape, rhs_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    e.line(format!(
        r#"{out} = "stablehlo.dot_general"({lhs}, {rhs}) {{"#
    ));
    e.push_indent();
    e.line(format!(
        "dot_dimension_numbers = #stablehlo.dot<"
    ));
    e.push_indent();
    e.line(format!("lhs_batching_dimensions = {},",    fmt_i64_arr(&dnums.lhs_batching)));
    e.line(format!("rhs_batching_dimensions = {},",    fmt_i64_arr(&dnums.rhs_batching)));
    e.line(format!("lhs_contracting_dimensions = {},", fmt_i64_arr(&dnums.lhs_contracting)));
    e.line(format!("rhs_contracting_dimensions = {}",  fmt_i64_arr(&dnums.rhs_contracting)));
    e.pop_indent();
    e.line(">,".to_string());
    e.line(r#"precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]"#.to_string());
    e.pop_indent();
    e.line(format!(
        r#"}} : ({}, {}) -> {}"#,
        lhs_shape.mlir_type(), rhs_shape.mlir_type(), out_shape.mlir_type()
    ));
    out
}
```


### Reduce

Spec example:
```mlir
%result = "stablehlo.reduce"(%input, %init_value) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  dimensions = array<i64: 1>
} : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
```

Key spec points:
- Region uses `^bb0` block label
- Inner ops use `"stablehlo.return"` (not `return`)
- Region args are 0-dim tensors of the element type (`tensor<i64>`, not `i64`)
- Output shape drops the reduced dimensions

```rust
/// Emit a reduce op with a callback for the combiner body.
/// The combiner receives two SsaName args (0-dim tensor type) and returns one SsaName.
pub fn emit_reduce(
    e: &mut MlirEmitter,
    operand: SsaName, init: SsaName,
    reduce_dims: &[usize],
    in_shape: &Shape,    // full input shape
    out_shape: &Shape,   // shape with reduced dims removed
    combiner: impl FnOnce(&mut MlirEmitter, SsaName, SsaName) -> SsaName,
) -> SsaName {
    let out = e.fresh();
    let scalar_ty = Shape::scalar(in_shape.dtype);
    let arg_a = e.fresh();
    let arg_b = e.fresh();
    let dims: Vec<String> = reduce_dims.iter().map(|x| x.to_string()).collect();

    e.line(format!(r#"{out} = "stablehlo.reduce"({operand}, {init}) ("#));
    e.push_indent();
    e.line(format!("^bb0({arg_a}: {st}, {arg_b}: {st}):", st = scalar_ty.mlir_type()));
    e.push_indent();
    let result = combiner(e, arg_a, arg_b);
    e.line(format!(
        r#""stablehlo.return"({result}) : ({}) -> ()"#,
        scalar_ty.mlir_type()
    ));
    e.pop_indent();
    e.pop_indent();
    e.line(format!(
        r#") {{dimensions = array<i64: {}>}} : ({}, {}) -> {}"#,
        dims.join(", "),
        in_shape.mlir_type(), scalar_ty.mlir_type(),
        out_shape.mlir_type()
    ));
    out
}

/// Convenience: reduce-add (sum). Used in RMSNorm, softmax.
pub fn emit_reduce_add(
    e: &mut MlirEmitter,
    operand: SsaName, zero: SsaName,
    reduce_dims: &[usize],
    in_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let dtype = in_shape.dtype;
    emit_reduce(e, operand, zero, reduce_dims, in_shape, out_shape,
        move |inner, a, b| emit_add(inner, a, b, &Shape::scalar(dtype)))
}

/// Convenience: reduce-max. Used in numerically stable softmax.
pub fn emit_reduce_max(
    e: &mut MlirEmitter,
    operand: SsaName, neg_inf: SsaName,
    reduce_dims: &[usize],
    in_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let dtype = in_shape.dtype;
    emit_reduce(e, operand, neg_inf, reduce_dims, in_shape, out_shape,
        move |inner, a, b| emit_maximum(inner, a, b, &Shape::scalar(dtype)))
}
```

### Gather

Spec example:
```mlir
%result = "stablehlo.gather"(%operand, %start_indices) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [3, 4],
    collapsed_slice_dims = [1],
    operand_batching_dims = [0],
    start_indices_batching_dims = [1],
    start_index_map = [2, 1],
    index_vector_dim = 3>,
  slice_sizes = array<i64: 1, 1, 2, 2>,
  indices_are_sorted = false
} : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
```

Note: `operand_batching_dims` and `start_indices_batching_dims` are newer fields
(batched gather). For simple embedding lookup use empty batching dims.

```rust
#[derive(Debug, Clone, Default)]
pub struct GatherDimensionNumbers {
    pub offset_dims:                  Vec<usize>,
    pub collapsed_slice_dims:         Vec<usize>,
    pub operand_batching_dims:        Vec<usize>,
    pub start_indices_batching_dims:  Vec<usize>,
    pub start_index_map:              Vec<usize>,
    pub index_vector_dim:             usize,
}

pub fn emit_gather(
    e: &mut MlirEmitter,
    operand: SsaName, indices: SsaName,
    dnums: &GatherDimensionNumbers,
    slice_sizes: &[usize],
    operand_shape: &Shape, indices_shape: &Shape, out_shape: &Shape,
) -> SsaName {
    let out = e.fresh();
    let slices: Vec<String> = slice_sizes.iter().map(|x| x.to_string()).collect();
    e.line(format!(r#"{out} = "stablehlo.gather"({operand}, {indices}) {{"#));
    e.push_indent();
    e.line("dimension_numbers = #stablehlo.gather<".to_string());
    e.push_indent();
    e.line(format!("offset_dims = {},",                 fmt_i64_arr(&dnums.offset_dims)));
    e.line(format!("collapsed_slice_dims = {},",        fmt_i64_arr(&dnums.collapsed_slice_dims)));
    if !dnums.operand_batching_dims.is_empty() {
        e.line(format!("operand_batching_dims = {},",       fmt_i64_arr(&dnums.operand_batching_dims)));
        e.line(format!("start_indices_batching_dims = {},", fmt_i64_arr(&dnums.start_indices_batching_dims)));
    }
    e.line(format!("start_index_map = {},",             fmt_i64_arr(&dnums.start_index_map)));
    e.line(format!("index_vector_dim = {}>,",           dnums.index_vector_dim));
    e.pop_indent();
    e.line(format!("slice_sizes = array<i64: {}>,", slices.join(", ")));
    e.line("indices_are_sorted = false".to_string());
    e.pop_indent();
    e.line(format!(
        r#"}} : ({}, {}) -> {}"#,
        operand_shape.mlir_type(), indices_shape.mlir_type(), out_shape.mlir_type()
    ));
    out
}
```


### Scatter

Spec example:
```mlir
%result = "stablehlo.scatter"(%input, %scatter_indices, %update) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  scatter_dimension_numbers = #stablehlo.scatter<
    update_window_dims = [3, 4],
    inserted_window_dims = [1],
    input_batching_dims = [0],
    scatter_indices_batching_dims = [1],
    scatter_dims_to_operand_dims = [2, 1],
    index_vector_dim = 3>,
  indices_are_sorted = false,
  unique_indices = false
} : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
```

Region args are 0-dim tensors of the element type. Use `unique_indices = true`
for KV cache writes where each position is written at most once per step.

```rust
#[derive(Debug, Clone, Default)]
pub struct ScatterDimensionNumbers {
    pub update_window_dims:             Vec<usize>,
    pub inserted_window_dims:           Vec<usize>,
    pub input_batching_dims:            Vec<usize>,
    pub scatter_indices_batching_dims:  Vec<usize>,
    pub scatter_dims_to_operand_dims:   Vec<usize>,
    pub index_vector_dim:               usize,
}

pub fn emit_scatter(
    e: &mut MlirEmitter,
    base: SsaName, indices: SsaName, updates: SsaName,
    dnums: &ScatterDimensionNumbers,
    unique_indices: bool,
    base_shape: &Shape, indices_shape: &Shape, updates_shape: &Shape,
    combiner: impl FnOnce(&mut MlirEmitter, SsaName, SsaName) -> SsaName,
) -> SsaName {
    let out = e.fresh();
    let scalar_ty = Shape::scalar(base_shape.dtype);
    let arg_old = e.fresh();
    let arg_new = e.fresh();

    e.line(format!(r#"{out} = "stablehlo.scatter"({base}, {indices}, {updates}) ("#));
    e.push_indent();
    e.line(format!("^bb0({arg_old}: {st}, {arg_new}: {st}):", st = scalar_ty.mlir_type()));
    e.push_indent();
    let result = combiner(e, arg_old, arg_new);
    e.line(format!(r#""stablehlo.return"({result}) : ({}) -> ()"#, scalar_ty.mlir_type()));
    e.pop_indent();
    e.pop_indent();

    e.line(") {".to_string());
    e.push_indent();
    e.line("scatter_dimension_numbers = #stablehlo.scatter<".to_string());
    e.push_indent();
    e.line(format!("update_window_dims = {},",            fmt_i64_arr(&dnums.update_window_dims)));
    e.line(format!("inserted_window_dims = {},",          fmt_i64_arr(&dnums.inserted_window_dims)));
    if !dnums.input_batching_dims.is_empty() {
        e.line(format!("input_batching_dims = {},",           fmt_i64_arr(&dnums.input_batching_dims)));
        e.line(format!("scatter_indices_batching_dims = {},", fmt_i64_arr(&dnums.scatter_indices_batching_dims)));
    }
    e.line(format!("scatter_dims_to_operand_dims = {},",  fmt_i64_arr(&dnums.scatter_dims_to_operand_dims)));
    e.line(format!("index_vector_dim = {}>,",             dnums.index_vector_dim));
    e.pop_indent();
    e.line(format!("indices_are_sorted = false,"));
    e.line(format!("unique_indices = {}", unique_indices));
    e.pop_indent();
    e.line(format!(
        r#"}} : ({}, {}, {}) -> {}"#,
        base_shape.mlir_type(), indices_shape.mlir_type(),
        updates_shape.mlir_type(), base_shape.mlir_type()
    ));
    out
}
```

### All Reduce

Spec example:
```mlir
%result:2 = "stablehlo.all_reduce"(%operand0, %operand0) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "stablehlo.return"(%0) : (tensor<i64>) -> ()
}) {
  replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
} : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
```

For single-operand all-reduce (the common LLM case):

```rust
pub fn emit_all_reduce(
    e: &mut MlirEmitter,
    operand: SsaName,
    // replica_groups: e.g. [[0,1,2,3]] for 4-device ring
    replica_groups: &[Vec<usize>],
    shape: &Shape,
    combiner: impl FnOnce(&mut MlirEmitter, SsaName, SsaName) -> SsaName,
) -> SsaName {
    let out = e.fresh();
    let scalar_ty = Shape::scalar(shape.dtype);
    let arg_a = e.fresh();
    let arg_b = e.fresh();

    e.line(format!(r#"{out} = "stablehlo.all_reduce"({operand}) ("#));
    e.push_indent();
    e.line(format!("^bb0({arg_a}: {st}, {arg_b}: {st}):", st = scalar_ty.mlir_type()));
    e.push_indent();
    let result = combiner(e, arg_a, arg_b);
    e.line(format!(r#""stablehlo.return"({result}) : ({}) -> ()"#, scalar_ty.mlir_type()));
    e.pop_indent();
    e.pop_indent();

    // Format replica groups as dense tensor: dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
    let n_groups = replica_groups.len();
    let group_size = replica_groups.first().map(|g| g.len()).unwrap_or(0);
    let groups_flat: Vec<String> = replica_groups.iter()
        .map(|g| format!("[{}]", g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")))
        .collect();
    e.line(format!(
        r#") {{replica_groups = dense<[{}]> : tensor<{}x{}xi64>, channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>}} : ({}) -> {}"#,
        groups_flat.join(", "), n_groups, group_size,
        shape.mlir_type(), shape.mlir_type()
    ));
    out
}
```

### Custom Call

Spec: `"stablehlo.custom_call"` with `call_target_name`, optional `backend_config`.

```rust
pub fn emit_custom_call(
    e: &mut MlirEmitter,
    call_target_name: &str,
    operands: &[(SsaName, Shape)],
    out_shape: &Shape,
    backend_config: Option<&str>,
) -> SsaName {
    let out = e.fresh();
    let ops_str: Vec<String> = operands.iter().map(|(n, _)| n.to_string()).collect();
    let in_types: Vec<String> = operands.iter().map(|(_, s)| s.mlir_type()).collect();
    let cfg = match backend_config {
        // backend_config is a string attribute in the spec
        Some(c) => format!(r#", backend_config = "{}""#, c.replace('"', "\\\"")),
        None    => String::new(),
    };
    e.line(format!(
        r#"{out} = "stablehlo.custom_call"({}) {{call_target_name = "{}"{cfg}}} : ({}) -> {}"#,
        ops_str.join(", "), call_target_name,
        in_types.join(", "), out_shape.mlir_type()
    ));
    out
}
```

### While

Spec example:
```mlir
%results0, %results1 = "stablehlo.while"(%init_i, %init_sum) ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %cond = "stablehlo.compare"(%arg0, %ten) {
      comparison_direction = #stablehlo<comparison_direction LT>
    } : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %cond : tensor<i1>
  }, {
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
    %new_sum = stablehlo.add %arg1, %one : tensor<i64>
    %new_i = stablehlo.add %arg0, %one : tensor<i64>
    stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
}) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
```

Note: `while` takes two regions (cond, body) as inline function literals.
The output types match the init operand types exactly.

For gljax v1 the generation loop is expressed statically (unrolled or bucket-
compiled), so `stablehlo.while` is deferred to v2. Include the struct now,
implement in ARTX3.

```rust
/// Placeholder struct for while loop — implementation in ARTX3 (ops/ layer).
pub struct WhileRegions {
    pub cond_body: String,  // pre-rendered region text
    pub loop_body: String,  // pre-rendered region text
}
```

---

## 4. SsaValue

`SsaValue` = `SsaName` (the `%vN` id) + `Shape` (output type of the op that produced it).
It lives in `graph/` and is what `FuncBuilder` returns from every op method.

```rust
// src/graph/value.rs

use crate::stablehlo::{emitter::SsaName, types::Shape};

/// A typed SSA value. Not Copy (Shape has Vec<usize>).
#[derive(Clone, Debug)]
pub struct SsaValue {
    pub name:  SsaName,
    pub shape: Shape,
}

impl SsaValue {
    pub fn new(name: SsaName, shape: Shape) -> Self { Self { name, shape } }
    pub fn ssa(&self)   -> SsaName { self.name }
    pub fn shape(&self) -> &Shape  { &self.shape }
    pub fn dtype(&self) -> crate::stablehlo::types::DType { self.shape.dtype }
    pub fn rank(&self)  -> usize   { self.shape.rank() }
    pub fn dim(&self, i: usize) -> usize { self.shape.dims[i] }
}

impl std::fmt::Display for SsaValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} : {}", self.name, self.shape.mlir_type())
    }
}
```

⚠️ **DESIGN DECISION — SsaValue is Clone, not Copy**  
`Shape::dims` is `Vec<usize>`, which is not `Copy`. Clones are cheap in practice
(typical rank ≤ 5 means a small heap allocation). If profiling shows clone overhead,
change `dims` to `SmallVec<[usize; 6]>` — all LLM tensors are rank ≤ 5, making
this stack-allocated. Defer until there is measurement.

**Relationship to Tensor**: `Tensor` (in `src/tensor/`) wraps `SsaValue` plus an
`Rc<RefCell<FuncBuilder>>`. The separation matters: `SsaValue` can appear in shape
inference logic without touching the builder.

---

## 5. FuncBuilder

### Role

`FuncBuilder` is the middle layer between model code and the emitter. It:
1. Owns `MlirEmitter` and forwards op calls to it.
2. Performs **shape inference** — validates input shapes, computes output shapes,
   returns `SsaValue` with the inferred output shape.
3. Tracks declared parameters in declaration order (needed for the PJRT ABI).
4. `finish()` assembles the complete `module { func.func @main(...) }` text.

⚠️ **DESIGN DECISION — Shape errors are panics**  
Shape mismatches are programming errors, not user input errors. They fire at
trace time (before any PJRT compilation). Panicking with a clear message is more
debuggable than threading `Result` through every op call site. This mirrors JAX:
shape errors in JAX are Python exceptions at trace time.

```rust
// src/graph/builder.rs

use crate::{
    graph::value::SsaValue,
    stablehlo::{
        emitter::{MlirEmitter, SsaName},
        ops::{self, DotDimensionNumbers, GatherDimensionNumbers},
        types::{DType, ParamDesc, ParamKind, Shape},
    },
};

pub struct BuiltFunc {
    pub mlir:      String,
    pub signature: Signature,
}

pub struct Signature {
    pub inputs:  Vec<ParamDesc>,
    pub weights: Vec<ParamDesc>,
    pub outputs: Vec<Shape>,
}

pub struct FuncBuilder {
    emitter:     MlirEmitter,
    params:      Vec<ParamDesc>,
    outputs:     Vec<SsaValue>,
    func_name:   String,
    module_name: String,
}

impl FuncBuilder {
    pub fn new(func_name: impl Into<String>, module_name: impl Into<String>) -> Self {
        Self {
            emitter:     MlirEmitter::new(),
            params:      Vec::new(),
            outputs:     Vec::new(),
            func_name:   func_name.into(),
            module_name: module_name.into(),
        }
    }

    pub fn param(&mut self, name: impl Into<String>, shape: Shape, kind: ParamKind) -> SsaValue {
        let ssa = self.emitter.fresh();
        let desc = ParamDesc { name: name.into(), shape: shape.clone(), kind, ssa };
        self.params.push(desc);
        SsaValue::new(ssa, shape)
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> SsaValue {
        self.param(name, shape, ParamKind::Input)
    }

    pub fn weight(&mut self, name: impl Into<String>, shape: Shape) -> SsaValue {
        self.param(name, shape, ParamKind::Weight)
    }

    pub fn emitter_mut(&mut self) -> &mut MlirEmitter { &mut self.emitter }

    pub fn set_outputs(&mut self, outputs: Vec<SsaValue>) { self.outputs = outputs; }

    // ── Shape-inferring op methods ─────────────────────────────────────────

    pub fn add(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        assert_eq!(lhs.shape(), rhs.shape(), "add: shape mismatch {} vs {}", lhs, rhs);
        let name = ops::emit_add(&mut self.emitter, lhs.ssa(), rhs.ssa(), lhs.shape());
        SsaValue::new(name, lhs.shape.clone())
    }

    pub fn multiply(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        assert_eq!(lhs.shape(), rhs.shape(), "multiply: shape mismatch");
        let name = ops::emit_multiply(&mut self.emitter, lhs.ssa(), rhs.ssa(), lhs.shape());
        SsaValue::new(name, lhs.shape.clone())
    }

    pub fn subtract(&mut self, lhs: &SsaValue, rhs: &SsaValue) -> SsaValue {
        assert_eq!(lhs.shape(), rhs.shape(), "subtract: shape mismatch");
        let name = ops::emit_subtract(&mut self.emitter, lhs.ssa(), rhs.ssa(), lhs.shape());
        SsaValue::new(name, lhs.shape.clone())
    }

    pub fn convert(&mut self, operand: &SsaValue, target_dtype: DType) -> SsaValue {
        if operand.dtype() == target_dtype { return operand.clone(); }
        let out_shape = Shape::new(operand.shape().dims.clone(), target_dtype);
        let name = ops::emit_convert(&mut self.emitter, operand.ssa(), operand.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn reshape(&mut self, operand: &SsaValue, new_dims: Vec<usize>) -> SsaValue {
        let numel_in = operand.shape().numel();
        let numel_out: usize = new_dims.iter().product();
        assert_eq!(numel_in, numel_out, "reshape: numel mismatch {} vs {}", numel_in, numel_out);
        let out_shape = Shape::new(new_dims, operand.dtype());
        let name = ops::emit_reshape(&mut self.emitter, operand.ssa(), operand.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn transpose(&mut self, operand: &SsaValue, permutation: Vec<usize>) -> SsaValue {
        assert_eq!(permutation.len(), operand.rank(), "transpose: permutation rank mismatch");
        let out_dims: Vec<usize> = permutation.iter().map(|&i| operand.dim(i)).collect();
        let out_shape = Shape::new(out_dims, operand.dtype());
        let name = ops::emit_transpose(
            &mut self.emitter, operand.ssa(), &permutation, operand.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn slice(
        &mut self, operand: &SsaValue,
        starts: Vec<usize>, limits: Vec<usize>, strides: Vec<usize>,
    ) -> SsaValue {
        let out_dims: Vec<usize> = starts.iter().zip(&limits).zip(&strides)
            .map(|((s, l), st)| (l - s).div_ceil(*st))
            .collect();
        let out_shape = Shape::new(out_dims, operand.dtype());
        let name = ops::emit_slice(
            &mut self.emitter, operand.ssa(), &starts, &limits, &strides,
            operand.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn concatenate(&mut self, operands: &[&SsaValue], dimension: usize) -> SsaValue {
        assert!(!operands.is_empty(), "concatenate: no operands");
        let dtype = operands[0].dtype();
        let rank = operands[0].rank();
        let mut out_dims = operands[0].shape().dims.clone();
        for op in &operands[1..] {
            assert_eq!(op.rank(), rank, "concatenate: rank mismatch");
            assert_eq!(op.dtype(), dtype, "concatenate: dtype mismatch");
            out_dims[dimension] += op.dim(dimension);
        }
        let out_shape = Shape::new(out_dims, dtype);
        let tagged: Vec<(SsaName, Shape)> = operands.iter()
            .map(|v| (v.ssa(), v.shape().clone())).collect();
        let name = ops::emit_concatenate(&mut self.emitter, &tagged, dimension, &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn broadcast_in_dim(
        &mut self, operand: &SsaValue,
        broadcast_dims: Vec<usize>, out_dims: Vec<usize>,
    ) -> SsaValue {
        let out_shape = Shape::new(out_dims, operand.dtype());
        let name = ops::emit_broadcast_in_dim(
            &mut self.emitter, operand.ssa(), &broadcast_dims, operand.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn dot_general(
        &mut self, lhs: &SsaValue, rhs: &SsaValue,
        dnums: &DotDimensionNumbers,
    ) -> SsaValue {
        let out_shape = infer_dot_general_shape(lhs.shape(), rhs.shape(), dnums);
        let name = ops::emit_dot_general(
            &mut self.emitter, lhs.ssa(), rhs.ssa(), dnums,
            lhs.shape(), rhs.shape(), &out_shape);
        SsaValue::new(name, out_shape)
    }

    pub fn constant_scalar(&mut self, value: f64, dtype: DType) -> SsaValue {
        let shape = Shape::scalar(dtype);
        let name = ops::emit_constant_scalar(&mut self.emitter, value, dtype);
        SsaValue::new(name, shape)
    }

    pub fn constant_splat(&mut self, value: f64, shape: Shape) -> SsaValue {
        let name = ops::emit_constant_splat(&mut self.emitter, value, &shape);
        SsaValue::new(name, shape)
    }

    pub fn rsqrt(&mut self, operand: &SsaValue) -> SsaValue {
        let name = ops::emit_rsqrt(&mut self.emitter, operand.ssa(), operand.shape());
        SsaValue::new(name, operand.shape.clone())
    }

    pub fn logistic(&mut self, operand: &SsaValue) -> SsaValue {
        let name = ops::emit_logistic(&mut self.emitter, operand.ssa(), operand.shape());
        SsaValue::new(name, operand.shape.clone())
    }

    pub fn exponential(&mut self, operand: &SsaValue) -> SsaValue {
        let name = ops::emit_exponential(&mut self.emitter, operand.ssa(), operand.shape());
        SsaValue::new(name, operand.shape.clone())
    }

    // ── finish ─────────────────────────────────────────────────────────────

    pub fn finish(self) -> BuiltFunc {
        assert!(!self.outputs.is_empty(), "FuncBuilder::finish: no outputs declared");

        let param_list: Vec<String> = self.params.iter()
            .map(|p| format!("{}: {}", p.ssa, p.shape.mlir_type()))
            .collect();
        let out_types: Vec<String> = self.outputs.iter()
            .map(|v| v.shape().mlir_type()).collect();
        let ret_names: Vec<String> = self.outputs.iter()
            .map(|v| v.name.to_string()).collect();
        let ret_types = out_types.join(", ");

        let mut s = String::new();
        s.push_str(&format!("module @{} {{\n", self.module_name));
        s.push_str(&format!("  func.func @{}({}) -> ({}) {{\n",
            self.func_name, param_list.join(", "), ret_types));
        s.push_str(&self.emitter.into_body());
        s.push_str(&format!("    \"func.return\"({}) : ({}) -> ()\n",
            ret_names.join(", "), ret_types));
        s.push_str("  }\n}\n");

        let sig = Signature {
            inputs:  self.params.iter().filter(|p| p.kind == ParamKind::Input).cloned().collect(),
            weights: self.params.iter().filter(|p| p.kind == ParamKind::Weight).cloned().collect(),
            outputs: self.outputs.iter().map(|v| v.shape().clone()).collect(),
        };
        BuiltFunc { mlir: s, signature: sig }
    }
}

fn infer_dot_general_shape(
    lhs: &Shape, rhs: &Shape, dnums: &DotDimensionNumbers,
) -> Shape {
    // Batch dims from lhs
    let batch: Vec<usize> = dnums.lhs_batching.iter().map(|&i| lhs.dims[i]).collect();
    // Free dims of lhs (not batch, not contracting)
    let lhs_free: Vec<usize> = (0..lhs.rank())
        .filter(|i| !dnums.lhs_batching.contains(i) && !dnums.lhs_contracting.contains(i))
        .map(|i| lhs.dims[i]).collect();
    // Free dims of rhs
    let rhs_free: Vec<usize> = (0..rhs.rank())
        .filter(|i| !dnums.rhs_batching.contains(i) && !dnums.rhs_contracting.contains(i))
        .map(|i| rhs.dims[i]).collect();
    // Validate contracting dims match
    for (&li, &ri) in dnums.lhs_contracting.iter().zip(&dnums.rhs_contracting) {
        assert_eq!(lhs.dims[li], rhs.dims[ri],
            "dot_general: contracting dim size mismatch: lhs[{li}]={} != rhs[{ri}]={}",
            lhs.dims[li], rhs.dims[ri]);
    }
    let out_dims: Vec<usize> = batch.into_iter().chain(lhs_free).chain(rhs_free).collect();
    Shape::new(out_dims, lhs.dtype) // output dtype = lhs dtype
}
```

---

## 6. TraceCx

`TraceCx` is the user-facing tracing context. It wraps `FuncBuilder` in
`Rc<RefCell<>>` so `Tensor` objects can hold a reference and push ops via
operator overloads without the user managing borrows.

⚠️ **DESIGN DECISION — Rc<RefCell<>> not Arc<Mutex<>>**  
Tracing is single-threaded. `Rc<RefCell<>>` is zero-overhead vs `Arc<Mutex<>>`.
See §1 for rationale on !Send.

⚠️ **DESIGN DECISION — Scope stack as the source of truth for weight names**  
Weight names produced during tracing must match checkpoint keys exactly (e.g.,
`"model.layers.0.self_attn.q_proj.weight"` in Qwen2 safetensors). The scope
stack produces this naming automatically. The `Signature::weights` list from
`finish()` gives the checkpoint loader both the shape and the checkpoint key for
each weight in declaration order — no separate registry needed.

```rust
// src/graph/trace.rs

use std::{cell::RefCell, rc::Rc};
use crate::{
    graph::{builder::FuncBuilder, value::SsaValue},
    stablehlo::types::{ParamKind, Shape},
    tensor::tensor::Tensor,
};

pub struct TraceCx {
    builder:     Rc<RefCell<FuncBuilder>>,
    scope_stack: Vec<String>,
}

impl TraceCx {
    pub fn new(func_name: impl Into<String>, module_name: impl Into<String>) -> Self {
        Self {
            builder:     Rc::new(RefCell::new(FuncBuilder::new(func_name, module_name))),
            scope_stack: Vec::new(),
        }
    }

    // ── Scope management ──────────────────────────────────────────────────

    pub fn push_scope(&mut self, name: impl Into<String>) {
        self.scope_stack.push(name.into());
    }

    pub fn pop_scope(&mut self) {
        self.scope_stack.pop().expect("TraceCx::pop_scope: stack underflow");
    }

    /// Full qualified name for a weight in current scope.
    /// e.g. scope=["model","layers.0","self_attn"], name="q_proj.weight"
    ///   -> "model.layers.0.self_attn.q_proj.weight"
    pub fn qualify(&self, name: &str) -> String {
        if self.scope_stack.is_empty() {
            return name.to_string();
        }
        format!("{}.{}", self.scope_stack.join("."), name)
    }

    /// Run a closure in a named scope. Automatically pushes/pops.
    pub fn scope<T>(&mut self, name: impl Into<String>, f: impl FnOnce(&mut TraceCx) -> T) -> T {
        self.push_scope(name);
        let result = f(self);
        self.pop_scope();
        result
    }

    // ── Parameter declaration ─────────────────────────────────────────────

    /// Runtime input (tokens, position ids, masks).
    /// Name is NOT scope-qualified — inputs have flat names.
    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        let value = self.builder.borrow_mut().input(name, shape);
        Tensor::new(value, Rc::clone(&self.builder))
    }

    /// Checkpoint weight in the current scope.
    /// Name IS scope-qualified.
    pub fn weight(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        let qname = self.qualify(&name.into());
        let value = self.builder.borrow_mut().weight(qname, shape);
        Tensor::new(value, Rc::clone(&self.builder))
    }

    // ── Finish ────────────────────────────────────────────────────────────

    pub fn finish(self, outputs: Vec<&Tensor>) -> crate::graph::builder::BuiltFunc {
        let out_values: Vec<SsaValue> = outputs.iter().map(|t| t.value().clone()).collect();
        let builder = Rc::try_unwrap(self.builder)
            .unwrap_or_else(|_| panic!("TraceCx::finish: builder still borrowed"))
            .into_inner();
        let mut b = builder;
        b.set_outputs(out_values);
        b.finish()
    }
}
```

---

## 7. Tensor

`Tensor` is the public API type. Model code only interacts with `Tensor`.
It combines `SsaValue` (its identity in the graph) with
`Rc<RefCell<FuncBuilder>>` (how to push new ops).

```rust
// src/tensor/tensor.rs

use std::{cell::RefCell, rc::Rc};
use crate::{
    graph::{builder::FuncBuilder, value::SsaValue},
    stablehlo::{
        ops::DotDimensionNumbers,
        types::{DType, Shape},
    },
};

pub struct Tensor {
    value:   SsaValue,
    builder: Rc<RefCell<FuncBuilder>>,
}

impl Tensor {
    pub fn new(value: SsaValue, builder: Rc<RefCell<FuncBuilder>>) -> Self {
        Self { value, builder }
    }

    pub fn value(&self)  -> &SsaValue { &self.value }
    pub fn shape(&self)  -> &Shape    { self.value.shape() }
    pub fn dtype(&self)  -> DType     { self.value.dtype() }
    pub fn rank(&self)   -> usize     { self.value.rank() }
    pub fn dim(&self, i: usize) -> usize { self.value.dim(i) }

    fn b(&self) -> std::cell::RefMut<'_, FuncBuilder> {
        self.builder.borrow_mut()
    }

    // ── Elementwise ───────────────────────────────────────────────────────

    pub fn add(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().add(&self.value, &rhs.value);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    pub fn mul(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().multiply(&self.value, &rhs.value);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    pub fn sub(&self, rhs: &Tensor) -> Tensor {
        let v = self.b().subtract(&self.value, &rhs.value);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    // ── Shape ops ─────────────────────────────────────────────────────────

    pub fn reshape(&self, new_dims: Vec<usize>) -> Tensor {
        let v = self.b().reshape(&self.value, new_dims);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    pub fn transpose(&self, permutation: Vec<usize>) -> Tensor {
        let v = self.b().transpose(&self.value, permutation);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    pub fn slice(&self, starts: Vec<usize>, limits: Vec<usize>, strides: Vec<usize>) -> Tensor {
        let v = self.b().slice(&self.value, starts, limits, strides);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    pub fn concat(tensors: &[&Tensor], dimension: usize) -> Tensor {
        assert!(!tensors.is_empty(), "concat: no tensors");
        let values: Vec<&SsaValue> = tensors.iter().map(|t| &t.value).collect();
        let v = tensors[0].b().concatenate(&values, dimension);
        Tensor::new(v, Rc::clone(&tensors[0].builder))
    }

    pub fn broadcast_to(&self, broadcast_dims: Vec<usize>, out_dims: Vec<usize>) -> Tensor {
        let v = self.b().broadcast_in_dim(&self.value, broadcast_dims, out_dims);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    // ── Precision ─────────────────────────────────────────────────────────

    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if self.dtype() == dtype { return self.clone_ref(); }
        let v = self.b().convert(&self.value, dtype);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    // ── Matmul ────────────────────────────────────────────────────────────

    pub fn dot_general(&self, rhs: &Tensor, dnums: DotDimensionNumbers) -> Tensor {
        let v = self.b().dot_general(&self.value, &rhs.value, &dnums);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    /// Convenience: last two dims matmul, with batch dims [0..rank-2].
    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        let r = self.rank();
        let dnums = DotDimensionNumbers {
            lhs_batching:    (0..r.saturating_sub(2)).collect(),
            rhs_batching:    (0..r.saturating_sub(2)).collect(),
            lhs_contracting: vec![r - 1],
            rhs_contracting: vec![r - 2],
        };
        self.dot_general(rhs, dnums)
    }

    // ── High-level ops (dispatch to ops/ module) ──────────────────────────

    pub fn silu(&self) -> Tensor {
        // SiLU = x * sigmoid(x). sigmoid = stablehlo.logistic per the spec.
        let sigmoid_v = self.b().logistic(&self.value);
        let sigmoid = Tensor::new(sigmoid_v, Rc::clone(&self.builder));
        self.mul(&sigmoid)
    }

    pub fn rsqrt(&self) -> Tensor {
        let v = self.b().rsqrt(&self.value);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    pub fn exp(&self) -> Tensor {
        let v = self.b().exponential(&self.value);
        Tensor::new(v, Rc::clone(&self.builder))
    }

    // Delegates for high-level ops implemented in ops/:
    pub fn rms_norm(&self, weight: &Tensor, eps: f64) -> Tensor {
        crate::ops::norm::rms_norm(self, weight, eps)
    }
    pub fn softmax(&self, dim: usize) -> Tensor {
        crate::ops::attention::softmax(self, dim)
    }

    // ── Internal ──────────────────────────────────────────────────────────

    /// Clone the handle: copies the Rc pointer and SsaValue. No graph mutation.
    pub fn clone_ref(&self) -> Tensor {
        Tensor::new(self.value.clone(), Rc::clone(&self.builder))
    }

    /// Expose builder Rc for use in ops/ functions that need to emit ops.
    pub fn builder(&self) -> &Rc<RefCell<FuncBuilder>> { &self.builder }
}

// ── Operator overloads ────────────────────────────────────────────────────────

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor { Tensor::add(self, rhs) }
}
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor { Tensor::mul(self, rhs) }
}
impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor { Tensor::sub(self, rhs) }
}
```

---

## 8. Precision Integration

```rust
// src/precision/policy.rs

#[derive(Clone, Copy, Debug)]
pub struct PrecisionPolicy {
    pub activation:     DType,  // activations default dtype
    pub weight:         DType,  // weights default dtype
    pub norm_reduce:    DType,  // RMSNorm reduce dtype (upcast for stability)
    pub softmax_reduce: DType,  // softmax sum/max reduce dtype
    pub rope:           DType,  // RoPE cos/sin application dtype
}

impl PrecisionPolicy {
    /// BF16 activations, F32 reductions. Standard for A100/TPU v5e production.
    pub fn bf16() -> Self {
        Self {
            activation:     DType::BF16,
            weight:         DType::BF16,
            norm_reduce:    DType::F32,
            softmax_reduce: DType::F32,
            rope:           DType::F32,
        }
    }

    pub fn f32() -> Self {
        Self {
            activation:     DType::F32,
            weight:         DType::F32,
            norm_reduce:    DType::F32,
            softmax_reduce: DType::F32,
            rope:           DType::F32,
        }
    }

    /// All FP64. Only valid on CPU and A100/H100 PJRT plugins.
    /// Use for cross-checking glproc FP32 output (the oracle pattern from ARTX1 §3.4).
    pub fn f64_oracle() -> Self {
        Self {
            activation:     DType::F64,
            weight:         DType::F64,
            norm_reduce:    DType::F64,
            softmax_reduce: DType::F64,
            rope:           DType::F64,
        }
    }
}
```

### Thread-local policy (no function signature pollution)

```rust
// src/precision/mod.rs
use std::cell::Cell;

thread_local! {
    static CURRENT: Cell<PrecisionPolicy> = Cell::new(PrecisionPolicy::bf16());
}

pub fn current() -> PrecisionPolicy { CURRENT.with(|p| p.get()) }

pub fn with_policy<T>(policy: PrecisionPolicy, f: impl FnOnce() -> T) -> T {
    CURRENT.with(|p| {
        let prev = p.get();
        p.set(policy);
        let result = f();
        p.set(prev);
        result
    })
}
```

### How ops use the policy

In `ops/norm.rs` (to be written in ARTX3), RMSNorm upcasts before the reduce:

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let policy = crate::precision::current();
    // Upcast to norm_reduce dtype for numerical stability
    let x_acc = x.to_dtype(policy.norm_reduce);  // stablehlo.convert
    // ... reduce(x^2) / D, rsqrt, multiply ...
    // Downcast back to activation dtype
    let normed = result_in_reduce_dtype.to_dtype(x.dtype());  // stablehlo.convert
    &normed * &weight.to_dtype(x.dtype())
}
```

With `PrecisionPolicy::f64_oracle()`, every `to_dtype` call inserts `stablehlo.convert`
ops to F64. The model code is unchanged — precision is entirely driven by the policy.

---

## 9. End-to-End Trace Example

Constants: B=1, S=512, D=2048, H=16, KV=4, Dh=128, FFN=5504 (Qwen2-1.5B layer).

```rust
use gljax::{
    graph::trace::TraceCx,
    precision::{PrecisionPolicy, with_policy},
    stablehlo::types::{DType, Shape},
    ops::attention::DotDimensionNumbers,
};

const B: usize = 1; const S: usize = 512; const D: usize = 2048;
const H: usize = 16; const KV: usize = 4; const DH: usize = 128;
const FFN: usize = 5504;

fn trace_layer(cx: &mut TraceCx, x: &gljax::tensor::Tensor, layer: usize)
    -> gljax::tensor::Tensor
{
    cx.scope(format!("model.layers.{layer}"), |cx| {
        // ── input_layernorm ──────────────────────────────────────────────
        let ln1_w = cx.scope("input_layernorm", |cx|
            cx.weight("weight", Shape::new([D], DType::BF16))
        );
        let residual = x.clone_ref();
        let h = x.rms_norm(&ln1_w, 1e-6);

        // ── self_attn ─────────────────────────────────────────────────────
        let attn_out = cx.scope("self_attn", |cx| {
            let q_w = cx.weight("q_proj.weight", Shape::new([D, H*DH], DType::BF16));
            let k_w = cx.weight("k_proj.weight", Shape::new([D, KV*DH], DType::BF16));
            let v_w = cx.weight("v_proj.weight", Shape::new([D, KV*DH], DType::BF16));
            let o_w = cx.weight("o_proj.weight", Shape::new([H*DH, D], DType::BF16));

            // [B,S,D] x [D, H*DH] -> [B,S,H*DH]
            let q = h.matmul(&q_w).reshape(vec![B,S,H,DH]).transpose(vec![0,2,1,3]);
            let k = h.matmul(&k_w).reshape(vec![B,S,KV,DH]).transpose(vec![0,2,1,3]);
            let v = h.matmul(&v_w).reshape(vec![B,S,KV,DH]).transpose(vec![0,2,1,3]);

            // GQA + causal attention (implementation in ops/attention.rs, ARTX3)
            let attn = gljax::ops::attention::gqa_causal_attention(&q, &k, &v);
            // [B,H,S,DH] -> [B,S,H*DH] -> [B,S,D]
            attn.transpose(vec![0,2,1,3]).reshape(vec![B,S,H*DH]).matmul(&o_w)
        });

        let h2 = &residual + &attn_out;

        // ── post_attention_layernorm ──────────────────────────────────────
        let ln2_w = cx.scope("post_attention_layernorm", |cx|
            cx.weight("weight", Shape::new([D], DType::BF16))
        );
        let residual2 = h2.clone_ref();
        let h3 = h2.rms_norm(&ln2_w, 1e-6);

        // ── mlp (SwiGLU) ─────────────────────────────────────────────────
        let mlp_out = cx.scope("mlp", |cx| {
            let gate_w = cx.weight("gate_proj.weight", Shape::new([D, FFN], DType::BF16));
            let up_w   = cx.weight("up_proj.weight",   Shape::new([D, FFN], DType::BF16));
            let down_w = cx.weight("down_proj.weight", Shape::new([FFN, D], DType::BF16));
            let gated = &h3.matmul(&gate_w).silu() * &h3.matmul(&up_w);
            gated.matmul(&down_w)
        });

        &residual2 + &mlp_out
    })
}

fn main() {
    let built = with_policy(PrecisionPolicy::bf16(), || {
        let mut cx = TraceCx::new("main", "qwen2_layer_0");
        let x = cx.input("hidden_states", Shape::new([B, S, D], DType::BF16));
        let out = trace_layer(&mut cx, &x, 0);
        cx.finish(vec![&out])
    });

    println!("{}", built.mlir);
    println!("inputs:  {}", built.signature.inputs.len());   // 1 (hidden_states)
    println!("weights: {}", built.signature.weights.len());  // 9 per layer
}
```

### Expected MLIR output (abbreviated)

The emitted text follows the generic form from the spec. Key fragments:

```mlir
module @qwen2_layer_0 {
  func.func @main(%v0: tensor<1x512x2048xbf16>) -> (tensor<1x512x2048xbf16>) {
    // weight params declared inline
    %v1: tensor<2048xbf16>    // model.layers.0.input_layernorm.weight

    // RMSNorm: upcast to f32 for reduce
    %v2 = "stablehlo.convert"(%v0) : (tensor<1x512x2048xbf16>) -> tensor<1x512x2048xf32>
    %v3 = "stablehlo.multiply"(%v2, %v2) : (tensor<1x512x2048xf32>, tensor<1x512x2048xf32>) -> tensor<1x512x2048xf32>
    %v4 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %v5 = "stablehlo.reduce"(%v3, %v4) ({
      ^bb0(%v6: tensor<f32>, %v7: tensor<f32>):
        %v8 = "stablehlo.add"(%v6, %v7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%v8) : (tensor<f32>) -> ()
    }) {dimensions = array<i64: 2>} : (tensor<1x512x2048xf32>, tensor<f32>) -> tensor<1x512xf32>
    // ... scale by 1/D, add eps, rsqrt, broadcast, multiply, convert back to bf16 ...

    // Q projection: [1,512,2048] x [2048,2048] -> [1,512,2048]
    %vN = "stablehlo.dot_general"(%vRMSout, %v_q_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x512x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<1x512x2048xbf16>

    // reshape [1,512,16,128] -> transpose [0,2,1,3] -> [1,16,512,128]
    // ... K, V, attention, output proj, residual, ln2, mlp, residual ...

    "func.return"(%v_final) : (tensor<1x512x2048xbf16>) -> ()
  }
}
```

Weight names in `signature.weights` (in declaration order):
```
model.layers.0.input_layernorm.weight          [2048] bf16
model.layers.0.self_attn.q_proj.weight         [2048, 2048] bf16
model.layers.0.self_attn.k_proj.weight         [2048, 512] bf16
model.layers.0.self_attn.v_proj.weight         [2048, 512] bf16
model.layers.0.self_attn.o_proj.weight         [2048, 2048] bf16
model.layers.0.post_attention_layernorm.weight [2048] bf16
model.layers.0.mlp.gate_proj.weight            [2048, 5504] bf16
model.layers.0.mlp.up_proj.weight              [2048, 5504] bf16
model.layers.0.mlp.down_proj.weight            [5504, 2048] bf16
```
These keys match Qwen2 safetensors exactly. No translation needed in the checkpoint loader.

---

## 10. What ARTX3 Should Cover

**ARTX3 — gljax ops/ Layer: LLM Op Implementations**

Full Rust implementations of all high-level ops in `src/ops/`, using
`FuncBuilder` from ARTX2:

1. `ops/norm.rs` — RMSNorm: full sequence with upcast/downcast per `PrecisionPolicy`.
   Test: compare to glproc FP32 via FP64 oracle.
2. `ops/rope.rs` — RoPE NeoX (Qwen2/Qwen3): precomputed cos/sin table, even/odd
   split via `emit_slice`, rotate, concatenate. Verify against MaxText reference.
3. `ops/attention.rs` — GQA causal attention: KV head repeat (broadcast+reshape),
   scaled dot-product, causal mask as a constant tensor, numerically stable softmax,
   AV product. Static KV cache via emit_scatter/emit_slice.
4. `ops/ffn.rs` — SwiGLU: single fused `[D, 2*FFN]` weight matmul + split +
   silu + gated multiply + down projection.
5. `ops/embedding.rs` — Token embedding lookup via emit_gather with
   correct GatherDimensionNumbers for the [V, D] table.
6. `ops/softmax.rs` — Numerically stable: subtract max, exp, sum, divide.
   Uses emit_reduce_max + emit_reduce_add.
7. `ops/moe.rs` — Top-2 routing + expert dispatch (CPU/single-device v1).

Each section: full Rust code, annotated MLIR output for a small shape, performance notes.

**ARTX4 — runtime/ and checkpoint/**

After ops are complete: Session, compile cache, safetensors streaming loader,
PJRT execute loop, and a full integration test (trace → compile → load weights → run).

---

## 11. Design Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| MLIR text format | Generic form `"stablehlo.op"(...)` | Matches spec; never ambiguous; same as what the spec verifier expects |
| `MlirEmitter` thread safety | `!Send` | Single-threaded trace; `Arc<Mutex<>>` adds overhead for zero benefit |
| Module/func assembly | `FuncBuilder::finish()` | Separates body from header; allows nested region emitters |
| Shape errors | Panic | Programming errors; fire at trace time; better than `Result` proliferation |
| `SsaValue` Copy | No — Clone only | `Shape::dims` is `Vec<usize>`, not `Copy` |
| `TraceCx`/`FuncBuilder` sharing | `Rc<RefCell<>>` | Single-threaded tracing only |
| Dynamic shapes | Deferred to v2 | Static shapes + bucketing sufficient (ARTX1 §9.3) |
| Weight naming | Scope stack → qualified key | Direct match to safetensors keys; no registry |
| Precision control | Thread-local `PrecisionPolicy` | No function signature pollution; restores on exit |
| Autograd | None | Inference-only crate |
| Op return type | `SsaValue` from builder | Shape inference in builder; ops.rs is pure text formatting |
| `stablehlo.return` | `"stablehlo.return"` in regions | Spec uses quoted form in region bodies |
| Scalar type | `tensor<dtype>` (0-dim tensor) | Spec: scalars are not first-class; 0-dim tensors are the canonical form |

---

*End of ARTX2 — gljax IR Design: FuncBuilder, TraceCx, and SSA Value System*  
*Research sources: [openxla.org/stablehlo/spec](https://openxla.org/stablehlo/spec) (fetched July 2026),*  
*[github.com/gomlx/stablehlo](https://github.com/gomlx/stablehlo) (cross-reference for attribute format),*  
*ARTX1 §8 (fusebox/ZML prior art).*  
*Next: ARTX3 — gljax ops/ Layer: LLM Op Implementations*
