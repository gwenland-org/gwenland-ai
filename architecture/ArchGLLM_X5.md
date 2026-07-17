# MEGAPROMPT — GwenLand AI M1.5: Bridge-ing AVX2 + Threading (ArchGLLM-X5)

> **Milestone**: M1.5 — between M1 (done, 1.83 TPS scalar) and M2 (GPU engines)
> **Architecture**: ArchGLLM-X5 (final, locked)
> **Target**: 80 TPS decode on Qwen2.5-0.5B Q4_K_M, Intel i3-1115G4 (2P/4T, AVX2, 8GB DDR4)
> **Paradigm**: Bridge-ing (fragmented files + L1 pipeline). NOT fused. NEVER fused.
> **Agent**: Claude Code
> **Repo**: gwenland-ai (Cargo workspace monorepo)

---

## 0. Context: What M1 Built (Already Done)

M1 is complete. The following exists and is working at 1.83 TPS scalar:

- Cargo workspace: `glcore`, `glproc`, `glcuda`, `glvulkan`, `glmetal`, `glcli`, `gltui`
- `glcore`: shared tensor types, `GlError`, engine trait (`init`, `load_model`, `infer`, `stream`, `shutdown`, `capabilities`)
- `glproc`: skeleton with `loader.rs` (GGUF mmap), `matmul.rs` (scalar), `attention.rs` (scaled dot-product), `kv_cache.rs` (basic), `sampler.rs` (greedy/top-k/top-p/temperature), `runner.rs` (decode loop)
- `glcli`: `gwen run model.gguf --prompt "Hello"` works, coherent text output confirmed
- Benchmark confirmed: 1.83 tok/s on Qwen2.5-0.5B Q4_K_M (or Qwen3-1.7B-Q8_0), i3-1115G4, 1 thread, no SIMD
- 435 tests passing

**M1.5 goal**: Take the existing M1 codebase and upgrade `glproc` with AVX2 SIMD, 4-thread interleaved execution, Bridge-ing kernel architecture, arena allocator, cursor-based KV cache, and warm-model-to-RAM loading. Target: 80 TPS stable.

---

## 1. Philosophy (Non-Negotiable)

### Bridge-ing, Not Fused

```
dequant/q4_k/avx2.rs ──► [f32; 256] stack buffer (L1 cache) ──► matmul/avx2.rs
         │                                                                │
         └──────────── NO `use` between them. Zero coupling. ────────────┘
                       bridge/mod.rs is the ONLY file that knows both exist.
```

- `dequant/` converts quantized bytes → f32. One format = one subfolder.
- `matmul/` does f32 × f32 dot product. One hardware = one file.
- `bridge/mod.rs` orchestrates: call dequant → write to stack buffer → call matmul. That's it.
- **No cross-file `use` between dequant and matmul.**
- Buffer = `[f32; 256]` on stack = 1 KB = fits in L1 cache (32–64 KB). No heap write between stages.

Why not fused (llama.cpp style)?
- Fused = 1 file, 500+ lines, dequant+matmul+activation entangled
- Bug in fused kernel = debug 500 lines = stress for a week
- Bridge-ing = bug in dequant = isolated 1 file, testable in isolation
- Trade-off: ~80% of fused performance. **100% of fragmented maintainability. Worth it.**

### Stability First

- **AVX2 only. NO AVX-512F.** i3-1115G4 AIO = small fan, passive cooling, TDP 28W
- AVX-512F @ throttled 2.4 GHz = 38.4 f32/GHz effective
- AVX2 @ sustained 3.0 GHz = 24 f32/GHz effective
- AVX-512F is only 60% faster but thermal risk = device damage. Not worth it.
- **Detect AVX-512F at startup for logging/warning only. Never use it.**

### Zero Dynamic Dispatch in Hot Path

- `SimdBackend` enum: `Scalar`, `Avx2`, `Avx512`
- `detect_backend()` runs once at startup, stored in Engine struct
- Decode loop uses `match backend { Avx2 => dot_f32_avx2(...), ... }`. No `Box<dyn>`. No `dyn Trait`.
- Vtable lookup = cache miss = jitter. Unacceptable.

### One Function = One Job

- `dequant_block_avx2()` = dequant 1 Q4_K block. Nothing else.
- `dot_f32_avx2()` = dot product of two f32 arrays. Nothing else.
- `matmul_q4k_bridge()` = call dequant + buffer + call matmul. Nothing else.

### Comment the Why, Not the What

- Bad: `// Extract low nibble`
- Good: `// Each byte packs 2 weights: low nibble = weight[i], high nibble = weight[i+1]`

### No Allocation in Decode Loop

- No `Vec::push`, no `String::new()`, no `malloc` in decode loop
- Arena allocator pre-allocated at startup. Reset per token.
- KV cache pre-allocated. Cursor-based. No realloc.
- Bridge buffer = stack (`let mut buffer = [0f32; 256]`). Not heap.

---

## 2. Target Hardware

| Spec | Value |
|------|-------|
| CPU | Intel Core i3-1115G4 |
| Cores | 2 physical (P), 4 logical (T, SMT) |
| Base clock | 3.0 GHz sustained under AVX2 load |
| Turbo | 4.1 GHz burst only |
| L1 cache | 32–48 KB per core |
| L3 cache | 6 MB shared |
| RAM | 8 GB DDR4, single channel, ~25 GB/s |
| SIMD | AVX2 (256-bit, 8 f32/reg, FMA) |
| AVX-512F | Present but NOT used (thermal risk) |
| TDP | 28W (AIO cooling, limited headroom) |
| GPU | None |

**Threading**: 4 threads (2P + 2T). Leave 0–1 logical cores for OS.

---

## 3. File Topology (glproc only — M1.5 scope)

Do not touch `glcuda`, `glvulkan`, `glmetal`, `glcli`, `gltui`, `glcore`. M1.5 = `glproc` only.

```
glproc/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Module re-exports (update to include new modules)
│   ├── engine.rs               # Engine trait impl (M1 — may need SimdBackend integration)
│   ├── runner.rs               # Decode loop (UPDATE: call warm_and_lock_model, use bridge, use threading)
│   ├── model.rs                # Model struct (M1 — likely unchanged)
│   ├── loader.rs               # UPDATE: add warm_and_lock_model() + mlock
│   ├── attention.rs            # M1 — may need KV cache cursor integration
│   ├── kv_cache.rs             # REPLACE: cursor-based, [layer][kv][head][seq][dim] layout
│   ├── sampler.rs              # M1 — unchanged
│   ├── simd_strategy.rs        # NEW: SimdBackend enum + detect_backend()
│   ├── threading.rs            # NEW: run_layers_interleaved()
│   ├── memory.rs               # NEW: Arena allocator
│   └── kernels/
│       ├── mod.rs              # NEW: re-export all kernels
│       ├── dequant/
│       │   ├── mod.rs          # NEW: re-export dequant formats
│       │   └── q4_k/
│       │       ├── mod.rs      # NEW: public API, run() for load-time
│       │       ├── scalar.rs   # NEW: ground truth dequant (no SIMD)
│       │       └── avx2.rs     # NEW: AVX2 SIMD dequant → L1 buffer
│       ├── matmul/
│       │   ├── mod.rs          # NEW: SimdBackend dispatch, dot_f32()
│       │   ├── scalar.rs       # NEW: ground truth dot product
│       │   └── avx2.rs         # NEW: AVX2 FMA dot product
│       ├── ops/
│       │   ├── mod.rs          # NEW
│       │   ├── fast_exp/       # NEW: AVX2 Taylor degree-5 poly (replaces f32::exp in hot loop)
│       │   └── rms_norm/       # NEW: in-place RMS normalization
│       └── bridge/
│           └── mod.rs          # NEW: Bridge-ing orchestrator
└── tests/
    └── kernel_parity.rs        # NEW: all parity tests
```

---

## 4. Deliverable Specs

### 4.1 `kernels/dequant/q4_k/scalar.rs` — Ground Truth

**Q4_K_M block layout** (144 bytes per 256 weights):

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 2 | `d` | f16 LE. Global scale. |
| 2 | 2 | `dmin` | f16 LE. Global minimum. |
| 4 | 12 | `scales` | 12 bytes. Per-sub-block scales and mins. |
| 16 | 128 | `qs` | 128 bytes. 2 nibbles per byte. |

**Constants** (named, no magic numbers):
```rust
pub const BLOCK_BYTES: usize = 144;
pub const BLOCK_NUMEL: usize = 256;
```

**Dequant formula**: `weight = dmin + (nibble * d * scale)`

**Function**:
```rust
/// Dequantize one Q4_K super-block (144 bytes) into 256 f32 values.
/// This is the scalar ground truth. Every SIMD path is validated against this.
pub fn dequant_block_scalar(data: &[u8], output: &mut [f32; 256]) {
    // data.len() must be >= BLOCK_BYTES (144)
    // Read d and dmin as f16 LE, convert to f32
    // Read 12 scale bytes, compute per-sub-block scale and min
    // For each nibble in qs[128]: weight = dmin_sub + nibble * d_sub
    // Comment the byte-packing: "Each byte packs 2 weights: low nibble = weight[i], high nibble = weight[i+1]"
}
```

No `unsafe`. No SIMD. Pure correctness.

---

### 4.2 `kernels/dequant/q4_k/avx2.rs` — AVX2 SIMD

**Target**: Same output as `dequant_block_scalar` but using AVX2 intrinsics.

```rust
#[target_feature(enable = "avx2", enable = "fma")]
#[inline(always)]
pub unsafe fn dequant_block_avx2(data: &[u8], output: &mut [f32; 256]) {
    // SAFETY: caller must ensure data.len() >= BLOCK_BYTES and CPU supports AVX2+FMA
    
    // AVX2 approach:
    // 1. Load d and dmin from f16 LE → f32
    // 2. Process qs[128] in 4 chunks of 32 bytes each
    //    - Each chunk: 2-pass unroll (two 16-byte halves)
    //    - _mm256_cvtepu8_epi32 → _mm256_cvtepi32_ps → _mm256_fmadd_ps
    // 3. Apply per-sub-block scale and min via FMA
}
```

**Parity requirement**: `assert!((avx2[i] - scalar[i]).abs() < 1e-5)` for all 256 values.

---

### 4.3 `kernels/dequant/q4_k/mod.rs` — Public API

```rust
pub mod scalar;
pub mod avx2;

pub use scalar::{dequant_block_scalar, BLOCK_BYTES, BLOCK_NUMEL};
pub use avx2::dequant_block_avx2;

/// Dequantize a full tensor (multiple Q4_K blocks) to f32.
/// LOAD-TIME ONLY. Not for use in decode loop (no hot path).
pub fn run(data: &[u8]) -> Result<Vec<f32>, glcore::GlError> {
    // Iterate over data in chunks of BLOCK_BYTES
    // Call dequant_block_scalar (or avx2 if available) per block
    // Collect into Vec<f32>
}
```

---

### 4.4 `kernels/matmul/scalar.rs` — Ground Truth

```rust
/// Dot product of two f32 arrays. Scalar, no SIMD.
/// Ground truth reference. Used in parity tests.
pub fn dot_f32_scalar(a: *const f32, b: *const f32, len: usize) -> f32 {
    // Simple for loop over len elements
    // No unsafe required
}
```

---

### 4.5 `kernels/matmul/avx2.rs` — AVX2 FMA

```rust
#[target_feature(enable = "avx2", enable = "fma")]
#[inline(always)]
pub unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    // SAFETY: caller ensures len % 8 == 0, CPU supports AVX2+FMA
    
    // 1. acc = _mm256_setzero_ps()
    // 2. Loop i in (0..len).step_by(8):
    //    a_vec = _mm256_loadu_ps(a.add(i))   // loadu = unaligned safe
    //    b_vec = _mm256_loadu_ps(b.add(i))
    //    acc   = _mm256_fmadd_ps(a_vec, b_vec, acc)
    // 3. Horizontal sum of acc register → f32
    //    Use _mm256_hadd_ps or extract + add
}
```

---

### 4.6 `kernels/matmul/mod.rs` — SimdBackend + Dispatch

```rust
// SimdBackend enum lives ONLY in simd_strategy.rs.
// kernels/matmul/mod.rs should re-export it:
pub use crate::simd_strategy::SimdBackend;

/// Dispatch dot product. Static dispatch via match — zero vtable overhead.
#[inline(always)]
pub fn dot_f32(a: *const f32, b: *const f32, len: usize, backend: SimdBackend) -> f32 {
    match backend {
        SimdBackend::Avx2 | SimdBackend::Avx512 => unsafe { avx2::dot_f32_avx2(a, b, len) },
        SimdBackend::Scalar => scalar::dot_f32_scalar(a, b, len),
    }
}
```

**Critical**: `SimdBackend` enum lives ONLY in `simd_strategy.rs`. `kernels/matmul/mod.rs` re-exports it: `pub use crate::simd_strategy::SimdBackend;` to avoid duplication.

And in `simd_strategy.rs`:
```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimdBackend {
    Scalar,
    Avx2,
    Avx512, // Detected at startup but NEVER used (thermal throttle risk on i3-1115G4)
}

/// Detect CPU SIMD capabilities. Call ONCE at startup. Store in Engine struct.
pub fn detect_backend() -> SimdBackend {
    if is_x86_feature_detected!("avx2") {
        if is_x86_feature_detected!("avx512f") {
            // Log warning: AVX-512F detected but disabled (thermal risk, AIO cooling)
            // Return Avx2 anyway
        }
        SimdBackend::Avx2
    } else {
        SimdBackend::Scalar
    }
}
```

---

### 4.7 `kernels/bridge/mod.rs` — Bridge-ing Orchestrator

```rust
use crate::kernels::dequant::q4_k::{dequant_block_avx2, dequant_block_scalar, BLOCK_BYTES, BLOCK_NUMEL};
use crate::kernels::matmul::{dot_f32, SimdBackend};

/// Bridge-ing: dequant → L1 stack buffer → matmul. No RAM write between stages.
///
/// weight_ptr: pointer to Q4_K raw bytes (n_blocks * BLOCK_BYTES)
/// input_ptr:  pointer to f32 input vector (n_blocks * BLOCK_NUMEL)
/// n_blocks:   number of Q4_K super-blocks
/// backend:    detected SIMD backend (stored in Engine, passed in)
pub fn matmul_q4k_bridge(
    weight_ptr: *const u8,
    input_ptr: *const f32,
    n_blocks: usize,
    backend: SimdBackend,
) -> f32 {
    // Stack buffer: 256 f32 = 1 KB. Stays in L1 cache (32–64 KB per core).
    // NEVER allocate this on heap. Stack = L1 hot.
    let mut buffer = [0f32; BLOCK_NUMEL];
    let mut acc = 0f32;

    for i in 0..n_blocks {
        // Step 1: dequant one block into L1 stack buffer
        unsafe {
            // SAFETY: weight_ptr is valid for n_blocks * BLOCK_BYTES, AVX2 confirmed by backend
            dequant_block_avx2(
                std::slice::from_raw_parts(weight_ptr.add(i * BLOCK_BYTES), BLOCK_BYTES),
                &mut buffer,
            );
        }

        // Step 2: dot product of buffer vs input slice
        // Buffer is hot in L1. No RAM write between dequant and matmul.
        acc += dot_f32(
            buffer.as_ptr(),
            unsafe { input_ptr.add(i * BLOCK_NUMEL) },
            BLOCK_NUMEL,
            backend,
        );
    }

    acc
}
```

**Key rule**: `dequant` and `matmul` modules have zero knowledge of each other. Only `bridge` knows both exist.

---

### 4.8 `kernels/ops/fast_exp/` — Fast Exponential

**Why**: `f32::exp()` and `f32::sqrt()` in hot loops are slow. Replace with AVX2 Taylor polynomial.

**Approach**: AVX2 degree-5 polynomial approximation for exp(x). 4× faster than `f32::exp()` in tight loops.

```rust
// Taylor degree-5 polynomial for exp(x), AVX2
// Coefficients: c0=1.0, c1=1.0, c2=0.5, c3=1/6, c4=1/24, c5=1/120
// Valid range: [-88.0, 88.0] (f32 exp domain)
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fast_exp_avx2(x: __m256) -> __m256;

// Scalar fallback
pub fn fast_exp_scalar(x: f32) -> f32;
```

Integrate into attention softmax (replace `f32::exp()` in score normalization).

---

### 4.9 `kernels/ops/rms_norm/` — RMS Normalization

**Why**: RMSNorm is used between every transformer layer. Must be fast.

```rust
/// In-place RMS normalization.
/// output[i] = input[i] / sqrt(mean(input^2) + eps) * weight[i]
pub fn rms_norm_scalar(input: &mut [f32], weight: &[f32], eps: f32);

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rms_norm_avx2(input: &mut [f32], weight: &[f32], eps: f32);
```

---

### 4.10 `threading.rs` — 4-Thread Interleaved Execution

**Strategy**: Interleaved layer assignment (NOT sequential).

```
Thread 0: layers  0,  4,  8, 12, 16, 20   (pinned: Physical Core 0)
Thread 1: layers  1,  5,  9, 13, 17, 21   (pinned: Physical Core 1)
Thread 2: layers  2,  6, 10, 14, 18, 22   (pinned: Logical Core 2, SMT of Core 0)
Thread 3: layers  3,  7, 11, 15, 19, 23   (pinned: Logical Core 3, SMT of Core 1)
```

**Why interleaved?**
- Sequential (thread 0 = layers 0–5) = all from different model file regions = cache thrash
- Interleaved = each thread's 6 layers spread out = better cache distribution across L3

```rust
/// Run transformer layers using interleaved multi-thread assignment.
/// Falls back to single-thread if spawn fails (no panic, no crash).
pub fn run_layers_interleaved(
    layers: &[Layer],
    input: &mut Tensor,
    n_threads: usize,  // typically 4 for i3-1115G4
) {
    // Use std::thread::scope for lifetime safety
    // Barrier after each wave (all threads must finish before next wave)
    // Example:
    // use std::sync::{Arc, Barrier};
    // let barrier = Arc::new(Barrier::new(n_threads));
    // Each thread calls barrier.wait() after finishing its layer wave
    // 
    // Fallback: if thread spawn fails → run single-thread
}
```

**Why `std::thread::scope`?** Lifetime safety, no `move` closures, no `Arc<Mutex<>>`. Simple and correct.

---

### 4.11 `memory.rs` — Arena Allocator

**Why**: `malloc` in decode loop = jitter. Arena = O(1) bump pointer, zero fragmentation.

```rust
pub struct Arena {
    base: *mut u8,
    size: usize,
    used: usize,
}

impl Arena {
    /// Allocate arena once at startup. Never reallocate.
    pub fn new(size: usize) -> Self;

    /// Bump pointer allocation. 64-byte aligned.
    pub fn allocate(&mut self, size: usize, align: usize) -> *mut u8;

    /// Reset cursor to 0. O(1). No free, no zeroing.
    /// Inference workspace is identical every token — reuse, don't free.
    pub fn reset(&mut self);
}
```

**Call site in decode loop**:
```rust
// At start of each token decode:
arena.reset();
// Then allocate workspace slices from arena
```

---

### 4.12 `kv_cache.rs` — Cursor-Based KV Cache (Replace M1 version)

**Layout**: `[layer][kv][head][seq][dim]`

Why this order? Attention reads `seq` sequentially. This layout = `seq` is contiguous = cache-friendly.

**Stride calculation**:
```
stride_dim   = 4                              // bytes, sizeof f32
stride_seq   = head_dim * stride_dim
stride_head  = max_context * stride_seq
stride_kv    = n_kv_heads * stride_head
stride_layer = 2 * stride_kv
```

**API**:
```rust
pub struct KvCache {
    data: Vec<f32>,            // pre-allocated at init, never reallocated
    current_pos: usize,        // cursor: advances per token
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_context: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_context: usize) -> Self;
    pub fn write_k(&mut self, layer: usize, head: usize, data: &[f32]);
    pub fn write_v(&mut self, layer: usize, head: usize, data: &[f32]);
    pub fn read_k(&self, layer: usize, head: usize, seq_len: usize) -> &[f32];
    pub fn read_v(&self, layer: usize, head: usize, seq_len: usize) -> &[f32];
    pub fn advance(&mut self);      // current_pos += 1
    pub fn reset(&mut self);        // current_pos = 0, no munmap, no zeroing
}
```

---

### 4.13 `loader.rs` — Warm Model to RAM

**Problem**: Cold mmap = disk read at 206 MB/s = ~2s per token. Must be solved before decode.

**Solution**: Touch all pages before decode to trigger kernel page faults upfront, then `mlock` to pin.

```rust
// In glproc/Cargo.toml:
// [target.'cfg(unix)'.dependencies]
// libc = "0.2"

/// Warm model pages into RAM and lock them. Call before decode loop.
/// Best effort — handle mlock errors gracefully.
#[cfg(unix)]
pub fn warm_and_lock_model(ptr: *mut u8, size: usize) {
    // Step 1: spawn thread to touch every 4096-byte page (triggers page faults)
    std::thread::scope(|s| {
        s.spawn(|| {
            for i in (0..size).step_by(4096) {
                unsafe { ptr.add(i).read_volatile(); }
            }
        });
    });

    // Step 2: mlock — pin pages, prevent eviction
    unsafe {
        if libc::mlock(ptr as *const libc::c_void, size) != 0 {
            // If mlock fails, at minimum ensure pages are touched (prefetch still helps)
            // Log: "mlock failed (EPERM) — pages prefetched but not pinned. 
            //       Consider: ulimit -l unlimited"
            eprintln!("Warning: mlock failed. Pages prefetched but not pinned. Consider: ulimit -l unlimited");
        }
    }
}

#[cfg(windows)]
pub fn warm_and_lock_model(ptr: *mut u8, size: usize) {
    // Same prefetch approach
    // VirtualLock instead of mlock
}
```

**Call site**: In `runner.rs`, call `warm_and_lock_model` BEFORE the decode loop starts. After GGUF mmap, before first token.

**Expected gain**: 2–4× TPS improvement (disk-cold 12 TPS → RAM-warm 50–80 TPS).

---

### 4.14 `tests/kernel_parity.rs` — All Parity Tests

```rust
// Test 1: Scalar dequant known values
#[test]
fn dequant_q4k_known_values() {
    // Build synthetic Q4_K block: d=1.0, dmin=0.0, all scales=1, qs=[0,1,2,3,...15,0,1,...]
    // Expected output: [0.0, 1.0, 2.0, ..., 15.0, 0.0, ...]
}

// Test 2: AVX2 dequant parity vs scalar
#[test]
fn dequant_avx2_parity_vs_scalar() {
    // Random Q4_K block bytes
    // Run scalar and avx2 on same data
    // assert!((scalar[i] - avx2[i]).abs() < 1e-5) for all 256
}

// Test 3: Scalar dot product correctness
#[test]
fn dot_scalar_known_values() {
    // a = [1.0, 2.0, 3.0, 4.0, ...], b = [1.0, 1.0, 1.0, 1.0, ...]
    // assert_eq!(dot_scalar(a, b, n), n*(n+1)/2)
}

// Test 4: AVX2 dot parity vs scalar
#[test]
fn dot_avx2_parity_vs_scalar() {
    // Random f32 arrays, length 256 (multiple of 8)
    // assert!((avx2 - scalar).abs() < 1e-5)
}

// Test 5: Bridge parity vs full scalar path
#[test]
fn bridge_matmul_q4k_parity() {
    // Random Q4_K blocks + input vector
    // Compare: bridge path (avx2 dequant → avx2 matmul) vs scalar path
    // assert diff < 1e-4
}

// Test 6: KV cache cursor correctness
#[test]
fn kv_cache_cursor_read_write() {
    // Write K/V at pos 0, advance, write at pos 1
    // Read back pos 0, assert matches original write
    // Read back pos 0..1 seq, assert both values correct
}

// Test 7: Arena alignment
#[test]
fn arena_64byte_alignment() {
    // Allocate several slots
    // Assert (ptr as usize) % 64 == 0 for each
}

// Test 8: Threading output matches single-thread
#[test]
fn threading_output_matches_single_thread() {
    // Run same layers single-thread vs 4-thread interleaved
    // Assert outputs identical (or within f32 tolerance)
}

// Test 9: Warm RAM (no major page fault after warm)
#[test]
fn warm_model_pages_loaded() {
    // Allocate test buffer, warm_and_lock_model()
    // Access all pages, assert accessible (no SIGSEGV)
}
```

**Rule**: ALL tests must pass before any deliverable is considered done. No exceptions. Correct first.

---

## 5. Integration: Updating runner.rs

After all kernels are built, integrate into the decode loop in `runner.rs`:

```rust
// Pseudocode for M1.5 decode loop integration:

fn run(model: &Model, prompt: &str) {
    // 1. Warm model to RAM (critical — before any compute)
    warm_and_lock_model(model.mmap_ptr, model.mmap_size);

    // 2. Detect SIMD backend once
    let backend = detect_backend(); // Avx2 on i3-1115G4

    // 3. Initialize arena (pre-alloc workspace)
    let mut arena = Arena::new(WORKSPACE_SIZE); // size = enough for all activations

    // 4. Initialize KV cache
    let mut kv_cache = KvCache::new(n_layers, n_kv_heads, head_dim, max_context);

    // 5. Tokenize prompt
    let tokens = tokenizer.encode(prompt);

    // 6. Decode loop (no alloc, no disk read, all RAM-warm)
    for token in tokens {
        arena.reset(); // O(1) workspace reset

        // Forward pass through layers
        run_layers_interleaved(&model.layers, &mut hidden_state, 4);

        // Each layer uses matmul_q4k_bridge(weight_ptr, input_ptr, n_blocks, backend)
        // Each layer uses rms_norm_avx2 for normalization
        // Attention uses fast_exp_avx2 for softmax scores
        // KV cache uses write_k/write_v + advance

        // Sample next token
        let next_token = sampler.sample(&logits);
        kv_cache.advance();

        print!("{}", tokenizer.decode(next_token));
    }
}
```

---

## 6. Performance Math (For Reference)

| Optimization | Multiplier | Status |
|---|---|---|
| Bridge-ing AVX2 (dequant + matmul) | 8× | NEW in M1.5 |
| 4-thread (2P+2T) | 3.0× | NEW in M1.5 |
| L1 cache pipeline (bridge buffer) | 1.5× | NEW in M1.5 |
| Memory layout (KV cache) | 1.3× | NEW in M1.5 |
| Prefetch (layer N+1 while computing N) | 1.1× | NEW in M1.5 |
| Warm RAM (mmap pinned, no disk) | 2.0× | **HIGHEST PRIORITY** |
| Scalar opt (loop unroll, branch hints) | 1.2× | NEW in M1.5 |

**Theoretical**: 1.83 × 8 × 3.0 × 1.5 × 1.3 × 1.1 × 2.0 × 1.2 = ~227 TPS
**Realistic** (35% efficiency): ~80 TPS
**Target**: **80 TPS steady state**

**Why warm_and_lock_model is highest priority**: Disk bottleneck is the biggest single gain (2× alone). Implement this first and benchmark before anything else.

---

## 7. Implementation Order (Recommended)

Follow this order. Do not proceed to next step until current tests pass.

1. **`warm_and_lock_model()`** in `loader.rs` → benchmark immediately. Expected: 12 → 50+ TPS from this alone.
   ⚠️ **STOP AFTER STEP 1.** Run benchmark. Report TPS.
   Do NOT proceed to step 2 until `warm_and_lock_model` is confirmed working and disk reads drop to ~0.
2. **`kernels/dequant/q4_k/scalar.rs`** → pass known-value test.
3. **`kernels/dequant/q4_k/avx2.rs`** → pass parity test vs scalar.
4. **`kernels/matmul/scalar.rs`** → pass known-value test.
5. **`kernels/matmul/avx2.rs`** → pass parity test vs scalar.
6. **`kernels/matmul/mod.rs`** (SimdBackend + dispatch) → compiles, detect_backend() returns Avx2.
7. **`kernels/bridge/mod.rs`** → pass bridge parity test.
8. **`kv_cache.rs`** (cursor-based replacement) → pass KV cache cursor test.
9. **`memory.rs`** (Arena) → pass alignment test.
10. **`threading.rs`** → pass threading output = single-thread test.
11. **`kernels/ops/fast_exp/`** → integrate into attention softmax.
12. **`kernels/ops/rms_norm/`** → integrate into layer norm.
13. **Integrate all into `runner.rs`** → full decode loop.
14. **Benchmark**: `gwen run model.gguf --prompt "Hello" --benchmark`. Target: 80 TPS steady state.

---

## 8. Constraints Summary

| Rule | What it means |
|------|---------------|
| Bridge-ing, not fused | dequant and matmul never `use` each other. Bridge is the only coupling point. |
| AVX2 only | NO AVX-512F. Detect it, log it, ignore it. |
| Zero dynamic dispatch | `SimdBackend` enum + `match`. No `dyn Trait` in hot path. |
| Zero alloc in decode | Arena pre-alloc. KV cache pre-alloc. Bridge buffer on stack. No `Vec::push` per token. |
| Scalar ground truth | Every SIMD function has a scalar counterpart. Parity test must pass. |
| One function = one job | `dequant_block_avx2` = dequant. `dot_f32_avx2` = dot. `matmul_q4k_bridge` = orchestrate. |
| No unsafe without SAFETY comment | Every `unsafe` block has `// SAFETY: ...` explaining the invariants. |
| Comment the why | "Each byte packs 2 weights" not "extract nibble". |
| Stability > peak | If CPU temp > 85°C, reduce thread count. No thermal risk. |
| Tests first | Write test. Run test. Pass test. Proceed. If stuck >30 min, write scalar version first. |

---

## 9. Exit Criteria (M1.5 Done When)

- [ ] All 12 deliverables implemented
- [ ] All 9 parity/correctness tests pass
- [ ] `cargo test --workspace` → all pass (M1 tests still passing + new M1.5 tests)
- [ ] No dynamic dispatch in decode loop (`dyn`, `Box<dyn>` absent from hot path)
- [ ] No allocation in decode loop (`Vec::push`, `String::new()` absent after init)
- [ ] Threading fallback works (single-thread if spawn fails, no panic)
- [ ] Bridge buffer = stack (no heap alloc between dequant and matmul)
- [ ] Model warm in RAM before decode (no page fault during benchmark)
- [ ] **Benchmark: 80 TPS steady state on Qwen2.5-0.5B Q4_K_M, i3-1115G4**
- [ ] **CPU temp stable under sustained 10-minute load (< 85°C, no throttle)**

---

## 10. Benchmark Protocol

**Before running**:
- Close all apps (browser, IDE, terminal except one)
- Plug in power (not battery)
- Let system idle 5 minutes

**Command**:
```bash
gwen run qwen2.5-0.5b-q4_k_m.gguf --prompt "Tell me about the universe" --benchmark
```

**Measure**:
1. First 10 tokens: cold TPS (warm RAM, cold KV cache)
2. Tokens 11–110: steady-state TPS
3. RAM usage (should be ~static after warm)
4. Disk read activity (should be zero after warm)

**Linux: monitor disk read during benchmark**
```bash
iostat -x 1 & gwen run model.gguf --benchmark
# Confirm: disk read rate drops to ~0 after warmup phase
```

**Abort if**: CPU temp > 90°C or clock throttling detected.

**Target**: 80 TPS steady state, < 85°C, 0 disk reads during decode.

---

*"Bridge-ing: 80% of fused performance, 100% of fragmented maintainability."*
*"Correct first, fast second, fused never."*
*"Stability first. Device rusak = habis duit."*
*"80 TPS. Not 100, not 50. 80. Stable. Sustained."*
