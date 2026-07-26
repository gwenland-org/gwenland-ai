# ARTX23 — Backend Dispatch: Registry, Dynamic Loader, Device Discovery

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux (ARTX23)
**Target GwenLand module:** `GATE` (cross-backend scheduler), `glproc` / `glcuda` / `glmetal` / `glvulkan` (all backends, since every backend must register through this layer)

---

## 1. Executive Summary

The ggml backend dispatch layer is the **contract layer** between the rest of
the engine and the per-backend code. It is small (~1.2k lines across three
files) but architecturally dense: it defines how backends are *found*, *loaded*,
*scored*, *enumerated*, and *asked to expose functions to each other*.

Three files do the work:

1. **`ggml-backend-reg.cpp`** (593 lines) — the registry. A Meyers singleton
   (`static ggml_backend_registry reg` in `get_reg()`, line 293) holds two
   flat vectors: `backends` (`ggml_backend_reg_entry` = reg + dl handle) and
   `devices` (`ggml_backend_dev_t`). The constructor statically links every
   backend compiled in via `#ifdef GGML_USE_*` + `register_backend(...)`.
   `ggml_backend_load_best` enumerates a directory of `[lib]ggml-<name>-*.{so,dll}`
   files, calls `ggml_backend_score()` in each, and keeps the highest score.

2. **`ggml-backend-dl.cpp`** (48 lines) — the dynamic loader. Two `#ifdef`
   branches: Win32 uses `LoadLibraryW` + `GetProcAddress`; POSIX uses
   `dlopen(RTLD_NOW | RTLD_LOCAL)` + `dlsym`. Errors are swallowed silently
   in `NDEBUG` builds.

3. **`ggml-backend-impl.h`** (275 lines) — the `ggml_backend_reg_i`,
   `ggml_backend_device_i`, `ggml_backend_i`, and `ggml_backend_buffer_type_i`
   vtables, plus the `GGML_BACKEND_DL_IMPL` and `GGML_BACKEND_DL_SCORE_IMPL`
   macros that backends use to emit the `ggml_backend_init` / `ggml_backend_score`
   C symbols the loader expects. Defines `GGML_BACKEND_API_VERSION = 2`.

For GwenLand, the architectural decisions worth **ADOPT**ing are the
registry/loader/score split (the multi-binary ISA dispatch pattern from
ARTX01-F12 lifted to a cross-backend mechanism), the `get_proc_address`
string-keyed extension hook (lets one backend ask another for a function the
ABI doesn't formalize), and the optional `offload_op` device hook (lets a GPU
opportunistically claim a CPU-resident `MUL_MAT` when batch size crosses a
threshold). The decisions worth **REJECT**ing are the missing thread-safety
on runtime `register_backend` / `unload_backend` (registry relies on
C++11 static-init thread-safety only), the hardcoded 15-name list in
`ggml_backend_load_all_from_path`, and the lack of any cross-device scoring
in `ggml_backend_init_best` (first GPU wins, even if a higher-tier GPU exists
further down the device list).

ARTX22 already audited the scheduler that *consumes* this registry; this
document zooms in on the registry itself, the loader, the score function, the
device enumeration contract, and the proc-address extension mechanism. It does
not duplicate ARTX22's coverage of `ggml_backend_sched` split heuristics.

---

## 2. Purpose

Provide a uniform, ABI-versioned, cross-platform backend discovery and loading
system that:

* statically links every backend compiled into the binary,
* dynamically loads additional `[lib]ggml-<name>-<variant>.so` files at runtime,
* scores each candidate so that, e.g., a `libggml-cpu-haswell.so` and a
  `libggml-cpu-skylake.so` can co-exist and the right one wins,
* enumerates devices across all backends in a single flat namespace,
* exposes a `get_proc_address` mechanism so backends can publish functions
  to each other without bumping the C ABI,
* verifies that any loaded backend was compiled against a compatible
  `GGML_BACKEND_API_VERSION`.

It is **not** responsible for: op-level kernel selection (that's the
scheduler, ARTX22), per-op `supports_op` decisions (that's the backend device
interface, audited per-backend in ARTX01/08/15/18), or buffer allocation
(that's `ggml_backend_buffer_type_i`).

---

## 3. Source Files

| File                                         | Lines | Role                                                          |
| -------------------------------------------- | ----- | ------------------------------------------------------------- |
| `ggml/src/ggml-backend-reg.cpp`              | 593   | The registry: ctor wires static backends; `load_best`, `load_all` |
| `ggml/src/ggml-backend-dl.cpp`               | 48    | `dlopen`/`dlsym` (POSIX) and `LoadLibraryW`/`GetProcAddress` (Win32) |
| `ggml/src/ggml-backend-impl.h`               | 275   | Vtables: `buffer_type_i`, `buffer_i`, `backend_i`, `device_i`, `reg_i`; `GGML_BACKEND_DL_IMPL` macros; `GGML_BACKEND_API_VERSION` |
| `ggml/src/ggml-backend-meta.cpp`             | large | Meta backend: virtual device wrapping multiple physical devices for tensor parallelism |
| `ggml/src/ggml-backend.cpp`                  | 2371  | Front-end wrappers (`ggml_backend_dev_get_props`, `reg_get_proc_address`, …) and the scheduler (audited in ARTX22) |
| `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`   | 328   | `ggml_backend_cpu_x86_score` (lines 263-325); the score-function pattern (cited, not re-audited — see ARTX01-F12) |
| `ggml/src/ggml-cpu/ggml-cpu.cpp`             | 708   | CPU `reg_i` vtable; CPU `get_proc_address` exposes `set_n_threads`, `set_abort_callback`, `numa_init`, `set_use_ref`, `threadpool_*`, `get_features`, `dev_get_extra_bufts` |
| `ggml/src/ggml-cuda/ggml-cuda.cu`            | 5426  | CUDA `reg_i` vtable; `get_proc_address` exposes comm_init/allreduce/host_buffer register/get_features |
| `ggml/src/ggml-metal/ggml-metal.cpp`         | 951   | Metal `reg_i` vtable; `get_proc_address` exposes only `get_features` |
| `ggml/include/ggml-backend.h`                | 436   | Public API: device types, dev_props struct, dev/reg enumeration, init_by_name/by_type/best |
| `ggml/src/ggml-backend-dl.h`                 | 44    | `dl_handle`, `dl_handle_ptr` (unique_ptr with `dlclose` deleter), `dl_load_library`, `dl_get_sym`, `dl_error` |

---

## 4. Architecture Overview

```
                          ┌─────────────────────────────────────────┐
                          │  ggml-backend-reg.cpp :                  │
                          │  ggml_backend_registry (Meyers singleton)│
                          │  ├─ vector<reg_entry> backends           │
                          │  └─ vector<dev_t>    devices             │
                          │                                          │
                          │  ctor (compile-time):                    │
                          │   #ifdef GGML_USE_CUDA register_cuda()   │
                          │   #ifdef GGML_USE_METAL register_metal() │
                          │   … 14 backends …                        │
                          │   #ifdef GGML_USE_CPU register_cpu()     │
                          │                                          │
                          │  load_best(name):                        │
                          │   dir-iter libggml-<name>-*.so           │
                          │   call ggml_backend_score() in each      │
                          │   keep highest score                     │
                          └─────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼─────────────────────┐
              ▼                        ▼                     ▼
   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
   │ ggml-backend-dl   │   │ ggml-backend-impl │   │ per-backend       │
   │ dlopen / dlsym    │   │ .h : vtables +    │   │ reg_i vtable +    │
   │ LoadLibraryW /    │   │ GGML_BACKEND_     │   │ GGML_BACKEND_     │
   │ GetProcAddress    │   │ DL_IMPL macro     │   │ DL_IMPL(reg_fn)   │
   └───────────────────┘   └───────────────────┘   └───────────────────┘
                                                              │
                                                              ▼
                                              ┌───────────────────────────┐
                                              │ ggml_backend_reg {        │
                                              │   api_version: int        │
                                              │   iface: reg_i {          │
                                              │     get_name              │
                                              │     get_device_count      │
                                              │     get_device            │
                                              │     get_proc_address      │
                                              │   }                       │
                                              │   context: void*          │
                                              │ }                         │
                                              └───────────────────────────┘
                                                              │
                                                              ▼
                                              ┌───────────────────────────┐
                                              │ ggml_backend_device {     │
                                              │   iface: device_i {       │
                                              │     get_name / get_descr  │
                                              │     get_memory            │
                                              │     get_type              │
                                              │     get_props             │
                                              │     init_backend          │
                                              │     get_buffer_type       │
                                              │     get_host_buffer_type  │
                                              │     buffer_from_host_ptr  │
                                              │     supports_op           │
                                              │     supports_buft         │
                                              │     offload_op (optional) │
                                              │     event_new/free/sync   │
                                              │   }                       │
                                              │   reg: back-pointer       │
                                              │   context: void*          │
                                              │ }                         │
                                              └───────────────────────────┘
```

Key design points:

* **Three-level vtable hierarchy.** A backend publishes a `reg_i` (the
  *registry* — name + device count + device getter + proc-address hook);
  each device publishes a `device_i` (capabilities, supports_op, buffer
  types); each backend *instance* (one per device, per stream) publishes a
  `backend_i` (graph_compute, async ops, events).
* **CPU is registered last.** The constructor at `ggml-backend-reg.cpp:120-173`
  hardcodes CUDA, Metal, SYCL, Vulkan, WebGPU, ZDNN, VirtGPU, OpenCL, ZenDNN,
  Hexagon, CANN, BLAS, RPC, OpenVINO, ET, then CPU. This means `ggml_backend_dev_by_type(GPU)`
  returns a GPU device before the CPU device is even in the list —
  `ggml_backend_init_best` relies on this.
* **Static linking is the default; dynamic loading is opt-in.** A binary
  with `GGML_USE_CUDA` has CUDA statically linked and `ggml_backend_cuda_reg()`
  is called from the registry constructor. A binary without `GGML_USE_CUDA`
  but with `libggml-cuda-*.so` on the search path can load it dynamically
  via `ggml_backend_load_best("cuda", ...)`.
* **No polymorphism in C.** Every vtable is a plain struct of function
  pointers; every "method" is a free function that takes the struct pointer
  as its first arg. The C++ wrapper at `ggml-backend.cpp:570-665` is pure
  syntactic sugar.

---

## 5. Execution Flow

### 5.1 Registry construction (process startup)

`get_reg()` (`ggml-backend-reg.cpp:292-295`) returns a reference to a
function-local `static ggml_backend_registry reg`. The first call constructs
the registry; C++11 guarantees this construction is thread-safe. The
constructor (lines 119-174) walks the `#ifdef GGML_USE_*` ladder and calls
`register_backend(ggml_backend_<name>_reg())` for every compiled-in backend.

`register_backend` (lines 186-205) dedups by `reg` pointer, appends to
`backends`, then calls `register_device` for every device the reg reports
(via `ggml_backend_reg_dev_count` / `dev_get`). `register_device` (lines
207-218) dedups by device pointer and appends to `devices`.

### 5.2 Dynamic loading (`ggml_backend_load_best`)

`ggml_backend_load_best(name, silent, user_search_path)` (lines 480-560):

1. Compute the file prefix `[lib]ggml-<name>-` and extension `.so` / `.dll`.
2. Assemble `search_paths`:
   * if `user_search_path == nullptr`: prepend `GGML_BACKEND_DIR` (compile-time)
     then `get_executable_path()` then `fs::current_path()`.
   * else: just `user_search_path`.
3. For each search path: `directory_iterator` over the directory; for every
   regular file whose name starts with the prefix and has the right extension,
   `dl_load_library` it and look up `ggml_backend_score`. If `score_fn() > best_score`,
   remember the path. The handle is *closed* (unique_ptr deleter) — only
   the score is collected at this stage.
4. If `best_score == 0`: fall back to the bare name `[lib]ggml-<name>.so`
   (no ISA suffix) via `load_backend(path, silent)`.
5. Otherwise: `load_backend(best_path, silent)` — re-`dlopen`, call
   `ggml_backend_init`, verify `reg->api_version == GGML_BACKEND_API_VERSION`,
   then `register_backend(reg, handle)`.

`ggml_backend_load_all_from_path(dir_path)` (lines 566-593) calls
`load_best` for each of 15 hardcoded backend names (see Finding F12), then
checks `GGML_BACKEND_PATH` env var for an out-of-tree backend.

### 5.3 Device enumeration

* `ggml_backend_dev_count()` returns `get_reg().devices.size()` (line 337).
* `ggml_backend_dev_get(i)` returns the i-th device (line 341).
* `ggml_backend_dev_by_name(name)` linear-scans, case-insensitive match
  against `dev_name(dev)` (lines 345-353).
* `ggml_backend_dev_by_type(type)` returns the **first** device whose
  `dev_type(dev)` matches (lines 355-363). There is no scoring across
  multiple devices of the same type.

### 5.4 Backend init

* `ggml_backend_init_by_name(name, params)` → `dev_by_name` → `dev_init`
  (lines 366-372).
* `ggml_backend_init_by_type(type, params)` → `dev_by_type` → `dev_init`
  (lines 374-380).
* `ggml_backend_init_best()` (lines 382-390):
  ```
  dev = dev_by_type(GPU) ?? dev_by_type(IGPU) ?? dev_by_type(CPU)
  return dev_init(dev, nullptr)
  ```
  No ACCEL fallback. No scoring across multiple GPUs.

### 5.5 The `get_proc_address` mechanism

`ggml_backend_reg_get_proc_address(reg, name)` (ggml-backend.cpp:659-665)
calls `reg->iface.get_proc_address(reg, name)` if non-NULL, else returns NULL.
Each backend implements its own string-keyed dispatch. Example (CPU,
`ggml-cpu.cpp:646-685`):

```
"ggml_backend_set_n_threads"        → ggml_backend_cpu_set_n_threads
"ggml_backend_dev_get_extra_bufts"  → ggml_backend_cpu_device_get_extra_buffers_type
"ggml_backend_get_features"         → ggml_backend_cpu_get_features
"ggml_backend_set_abort_callback"   → ggml_backend_cpu_set_abort_callback
"ggml_backend_cpu_numa_init"        → ggml_numa_init
"ggml_backend_cpu_is_numa"          → ggml_is_numa
"ggml_backend_cpu_set_use_ref"      → ggml_backend_cpu_set_use_ref
"ggml_threadpool_new"               → ggml_threadpool_new
"ggml_threadpool_free"              → ggml_threadpool_free
"ggml_backend_cpu_set_threadpool"   → ggml_backend_cpu_set_threadpool
```

CUDA exposes a smaller set (`comm_init`, `comm_free`, `comm_allreduce_tensor`,
`register_host_buffer`, `unregister_host_buffer`, `get_features`).
Metal exposes only `get_features`. The set is **not** standardized — the
header (`ggml-backend.h:203-2223`) lists "common functions that may be
obtained using `ggml_backend_reg_get_proc_address`" and typedefs the
function pointer types, but each backend picks which subset to publish.

---

## 6. Data Layout

### 6.1 The `ggml_backend_reg` struct

```c
struct ggml_backend_reg {
    int api_version;             // = GGML_BACKEND_API_VERSION = 2
    struct ggml_backend_reg_i iface;
    void * context;
};
```
(`ggml-backend-impl.h:226-230`)

The `api_version` field is the only versioning signal. It is checked in
`load_backend` (line 246) immediately after `backend_init_fn()` returns;
mismatch rejects the backend with a log message but does not abort.

### 6.2 The `ggml_backend_device` struct

```c
struct ggml_backend_device {
    struct ggml_backend_device_i iface;
    ggml_backend_reg_t reg;       // back-pointer to owning registry
    void * context;
};
```
(`ggml-backend-impl.h:204-208`)

The `reg` back-pointer lets `ggml_backend_dev_backend_reg(dev)` (line 594)
return the owning registry without a lookup. It also lets `unload_backend`
find every device belonging to a reg via `dev->reg == reg`.

### 6.3 The registry's two vectors

```c
struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;         // nullptr for statically-linked backends
};
struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t>     devices;
};
```
(`ggml-backend-reg.cpp:110-117`)

The `devices` vector is a *denormalized* flat list: every device of every
backend appears once, in registration order. There is no per-backend
sub-listing at the public-API level; `ggml_backend_reg_dev_get(reg, i)` is
the only way to enumerate a single backend's devices.

### 6.4 Device properties

```c
struct ggml_backend_dev_props {
    const char * name;
    const char * description;
    size_t memory_free;
    size_t memory_total;
    enum ggml_backend_dev_type type;   // CPU | GPU | IGPU | ACCEL | META
    const char * device_id;            // PCI bus id "domain:bus:device.function"
    struct ggml_backend_dev_caps caps; // async, host_buffer, buffer_from_host_ptr, events
};
```
(`ggml-backend.h:160-177`)

`ggml_backend_dev_get_props` (`ggml-backend.cpp:588-592`) `memset`s the
struct to zero then dispatches to `dev->iface.get_props`. Each backend
fills what it knows; e.g., the CPU backend fills `caps = { async=false,
host_buffer=false, buffer_from_host_ptr=true, events=false }`
(`ggml-cpu.cpp:395-400`).

---

## 7. Memory Layout

The registry's two vectors live in the static-local `ggml_backend_registry`
singleton. Both vectors grow monotonically during process startup (static
ctor) and during any `ggml_backend_load_*` call; they only shrink on
`unload_backend`. There is no per-device arena or pool — each backend
allocates its own contexts.

The `dl_handle_ptr` (`unique_ptr<dl_handle, dl_handle_deleter>` from
`ggml-backend-dl.h:40`) owns the .so handle for dynamically-loaded backends.
Statically-linked backends set `handle = nullptr` in `register_backend` and
the deleter never runs.

The destructor `~ggml_backend_registry()` (lines 176-184) explicitly
**leaks** every dl handle by calling `entry.handle.release()`. The comment
on line 177 explains why:

> FIXME: backends cannot be safely unloaded without a function to destroy
> all the backend resources, since backend threads may still be running and
> accessing resources from the dynamic library

This is a known design limitation. See Finding F10.

---

## 8. Parallelism Strategy

The registry is **single-threaded by construction**. There is no mutex
around `register_backend`, `register_device`, `load_backend`, or
`unload_backend`. The only thread-safety guarantee is C++11's rule that a
function-local `static` is constructed exactly once, thread-safely
([stmt.dcl]/4). After construction, all mutations are unprotected.

In practice this is OK because:

* Static backend registration happens in the registry constructor, which
  runs on the first `get_reg()` call — typically from the main thread during
  `llama_backend_init()`.
* Dynamic loading (`ggml_backend_load_all`) is also called from the main
  thread, before any worker threads are spawned.
* The public API documentation does not promise that runtime registration
  is thread-safe; no caller attempts it.

But it is a latent footgun. See Finding F01.

The registry's hot paths (device enumeration, `dev_get_props`,
`reg_get_proc_address`) are read-only after startup and therefore safe to
call from worker threads. The scheduler (ARTX22) does this constantly.

---

## 9. SIMD / GPU Strategy

The dispatch layer itself contains no SIMD and no GPU code. Its only
"performance" lever is the **score function**, which selects among multiple
compiled variants of the same backend.

### 9.1 The score contract

```c
typedef int (*ggml_backend_score_t)(void);
```
(`ggml-backend-impl.h:238`)

A score of `0` means "this backend does not support the current system";
`> 0` means "supported, higher is better". `load_best` keeps the highest
score across all `libggml-<name>-*.so` files it finds.

### 9.2 The CPU score pattern (x86)

`ggml_backend_cpu_x86_score` (`arch/x86/cpu-feats.cpp:263-325`, audited in
ARTX01-F12) builds a bitmask:

```
score = 1
+ (1 << 0) if compiled with GGML_FMA        and CPU has FMA
+ (1 << 1) if compiled with GGML_F16C       and CPU has F16C
+ (1 << 2) if compiled with GGML_SSE42      and CPU has SSE42
+ (1 << 3) if compiled with GGML_BMI2       and CPU has BMI2
+ (1 << 4) if compiled with GGML_AVX        and CPU has AVX
+ (1 << 5) if compiled with GGML_AVX2       and CPU has AVX2
+ (1 << 6) if compiled with GGML_AVX_VNNI   and CPU has AVX_VNNI
+ (1 << 7) if compiled with GGML_AVX512     and CPU has AVX512F/CD/VL/DQ/BW
+ (1 << 8) if compiled with GGML_AVX512_VBMI and CPU has VBMI
+ (1 << 9) if compiled with GGML_AVX512_BF16 and CPU has BF16
+ (1 << 10) if compiled with GGML_AVX512_VNNI and CPU has VNNI
+ (1 << 11) if compiled with GGML_AMX_INT8   and CPU has AMX_INT8
```

If the CPU lacks *any* feature the .so was compiled for, score returns 0
and the loader skips it. This is multi-binary ISA dispatch: ship N .so
files, one per ISA target, the loader picks the best matching one. See
Finding F04.

### 9.3 Other backends' scores

CUDA, Metal, Vulkan, etc. each ship a `ggml_backend_score` that returns a
positive constant (or 0 if no device is present). They do not have
multi-binary variants because they are already device-specific — the
runtime CUDA/Metal/Vulkan driver handles dispatch internally.

---

## 10. Quantization Strategy

Not applicable at this layer. Quantization is a per-backend concern
(audited in ARTX06 for CPU, ARTX10 for CUDA, etc.). The dispatch layer
treats every backend as opaque.

The one exception: the registry exposes
`ggml_backend_cpu_buffer_type()` and `ggml_backend_cpu_buffer_from_ptr()`
at `ggml-backend.cpp:2328, 2368` for legacy callers. These are not part of
the registry per se; they are convenience functions in the same TU.

---

## 11. Correctness Analysis

### 11.1 API version check

`load_backend` checks `reg->api_version != GGML_BACKEND_API_VERSION`
(line 246) immediately after `backend_init_fn()` returns. On mismatch it
logs and returns nullptr. **No ABI hash, no struct-layout check, no symbol
fingerprint** — just an integer. A backend compiled against API v1 will
load and silently misbehave if the loader is v2 *and* the v2 changes don't
touch any struct this backend touches. The integer is currently `2`
(`ggml-backend-impl.h:11`), bumped from 1 at some point in the past (not
auditable from this commit alone).

### 11.2 Stale device pointers after unload

`unload_backend` (lines 266-289) erases devices from the `devices` vector
and erases the reg entry from `backends`. It does **not** call any
"shutdown" function on the reg or its devices. Any tensor or buffer
allocated by the backend before unload becomes a dangling pointer. The
FIXME at line 177 acknowledges this: backends cannot be safely unloaded
because their threads may still be running.

### 11.3 Race on static-init

The C++11 static-init guarantee covers the *constructor* only. If two
threads simultaneously call `ggml_backend_register(reg)` (the public API,
line 298) after construction, both will mutate `backends` and `devices`
without a lock. Undefined behavior. The registry's only defense is that
no caller actually does this — but nothing in the public API or the
documentation forbids it.

### 11.4 Case-insensitive name matching

`striequals` (lines 307-314) does ASCII tolower compare. Backend names are
hardcoded by each backend's `reg_get_name` (e.g., CPU returns `"CPU"`,
CUDA returns `"CUDA"`, Metal returns `"Metal"`). The case-insensitivity is
a courtesy to callers who type `"cpu"` or `"cuda"`. No ambiguity is
possible because no two backends share a name (case-folded).

### 11.5 `GGML_DISABLE_VULKAN` env var

The Vulkan branch of the registry constructor (lines 129-136) is the only
backend with a runtime-disable env var. Every other backend is gated by
compile-time `#ifdef GGML_USE_*`. This is an inconsistency: a binary with
`GGML_USE_CUDA` defined cannot be told to skip CUDA at runtime even if the
user knows the driver is broken. See Finding F11.

### 11.6 `op_params` sanity

Not applicable at this layer.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                          | Where                                  | Notes                                                                  |
| ------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| Meyers singleton                      | `ggml-backend-reg.cpp:292-295`         | Thread-safe construction; no mutex needed for the static-init path.    |
| Score-based multi-binary dispatch     | `load_best`, `cpu-feats.cpp:263-325`   | Picks the best .so among many; survives partial-link builds.           |
| Flat device vector                    | `ggml-backend-reg.cpp:117`             | O(1) indexing, but O(N) `by_name`/`by_type` linear scan.               |
| `dl_handle_ptr` unique_ptr            | `ggml-backend-dl.h:40`                 | RAII for dlopen; dlclose runs automatically on unload.                 |
| `RTLD_NOW | RTLD_LOCAL`               | `ggml-backend-dl.cpp:35`               | Resolve all symbols at load (fail fast); don't pollute global symbol table. |
| `SetErrorMode(SEM_FAILCRITICALERRORS)`| `ggml-backend-dl.cpp:7-8, 18-19`       | Suppress Windows DLL-missing error dialogs during directory scan.      |
| `skip_permission_denied` dir iter     | `ggml-backend-reg.cpp:511`             | Don't abort the scan just because one directory is unreadable.         |
| `silent` flag in NDEBUG               | `ggml-backend-reg.cpp:567-571`         | Suppress "failed to load" noise in release builds.                     |
| `reg` back-pointer on device          | `ggml-backend-impl.h:206`              | `dev_backend_reg(dev)` is O(1); enables `unload_backend` filter.       |

### 12.2 Optimizations *not* present (worth noting)

* **No cross-device scoring in `init_best`.** First GPU wins, even if a
  more capable GPU exists further down the device list. See Finding F08.
* **No caching of `ggml_backend_score` calls.** Each `load_best` re-dlopens
  every candidate .so, calls `score_fn`, then closes it. The "best" .so
  is then re-dlopened a second time for actual loading. For a directory
  with 5 CPU variants this is 6 dlopens instead of 5.
* **No parallel `load_best` across backend names.** `ggml_backend_load_all`
  calls `load_best` for 15 names serially. On a system with many .so files
  this is slow startup.
* **No registry-level `supports_op` cache.** Every scheduler decision
  (ARTX22) calls `dev->iface.supports_op(dev, op)`, which for the CPU
  involves a `switch(op->op)` and a loop over extra buffer types. The
  result is not memoized.
* **No backend dependency graph.** A backend cannot declare "I require
  backend X to also be loaded" — the meta backend (ggml-backend-meta.cpp)
  works around this by wrapping multiple backends in a virtual device, but
  it is not part of the registry's contract.

---

## 13. Architectural Strengths

1. **Clean three-tier vtable hierarchy.** `reg_i` / `device_i` / `backend_i`
   cleanly separate "what backends exist" from "what devices exist" from
   "what streams exist". GwenLand should adopt this layering directly.

2. **Score-based multi-binary dispatch.** The `ggml_backend_score` +
   `load_best` pattern is the cleanest possible answer to "ship many ISA
   variants, pick the right one at runtime". It works for any backend, not
   just CPU. GwenLand should adopt it for `glproc` (x86 variants, ARM
   variants) and consider it for `glcuda` (CC-specific .so variants).

3. **`get_proc_address` extension hook.** String-keyed function lookup
   lets backends expose non-ABI'd functions to each other without bumping
   `GGML_BACKEND_API_VERSION`. This is how the CPU backend publishes
   `set_n_threads` and how CUDA publishes `comm_init`. Brilliantly simple.

4. **`offload_op` device hook.** Optional device method that says "even
   though this op's weights live in a CPU buffer, I want to run it on this
   GPU". Used by CUDA and Metal to opportunistically pull a `MUL_MAT` to
   GPU when batch size crosses a threshold (see Finding F09). Decouples
   weight placement from op placement.

5. **`api_version` integer check.** Minimal but sufficient for the common
   case ("this .so was compiled against an older ggml"). Catches the
   "stale .so in install dir" failure mode before any op runs.

6. **Device type taxonomy.** `CPU | GPU | IGPU | ACCEL | META` is the
   right granularity. `ACCEL` lets BLAS / AMX / ZenDNN register as a
   companion to the CPU rather than competing with it. `META` lets the
   tensor-parallel wrapper expose a single virtual device.

7. **Static-local singleton + flat vectors.** No clever data structures,
   no atomics, no locks. The simplest thing that could work, and it does.

---

## 14. Architectural Weaknesses

### W1 — Registry mutations are not thread-safe

**Evidence:** `register_backend` (`ggml-backend-reg.cpp:186-205`),
`register_device` (207-218), `load_backend` (220-264), `unload_backend`
(266-289) all mutate `backends` and `devices` vectors with no mutex. Only
the construction of the singleton itself is thread-safe (C++11
[stmt.dcl]/4).

**Impact:** Any caller that invokes `ggml_backend_register`,
`ggml_backend_load`, or `ggml_backend_unload` from a non-main thread
concurrent with another reader races on the vectors. In practice no caller
does this, but the public API does not forbid it.

**Why it's hard to fix:** Adding a mutex around every mutation is trivial;
the hard part is locking the *read* paths (`dev_count`, `dev_get`,
`dev_by_name`, `dev_by_type`) without imposing a lock on every scheduler
decision. A RwLock or hazard pointer would work; the team has chosen to
defer this entirely.

### W2 — Backend registration order is hardcoded

**Evidence:** `ggml-backend-reg.cpp:119-174` — 15 `#ifdef` blocks in a
fixed order, CPU last.

**Impact:** Adding a new backend requires editing this constructor and
adding a new `#ifdef`. There is no plugin manifest or auto-discovery of
statically-linked backends. The order also subtly affects `dev_by_type`,
which returns the first matching device.

### W3 — `ggml_backend_load_all_from_path` has a hardcoded 15-name list

**Evidence:** `ggml-backend-reg.cpp:566-593` calls `load_best` for `"blas"`,
`"zendnn"`, `"cann"`, `"cuda"`, `"hip"`, `"metal"`, `"rpc"`, `"sycl"`,
`"vulkan"`, `"virtgpu"`, `"opencl"`, `"hexagon"`, `"musa"`, `"openvino"`,
`"cpu"`.

**Impact:** Adding a new dynamically-loadable backend requires editing this
list. A new backend whose name is not on the list can still be loaded via
`GGML_BACKEND_PATH` env var (line 589) but not via the standard discovery
mechanism. See Finding F12.

### W4 — `init_best` has no cross-device scoring

**Evidence:** `ggml-backend-reg.cpp:382-390` — `dev = dev_by_type(GPU) ?
dev : dev_by_type(IGPU); dev = dev ? dev : dev_by_type(CPU);`.

**Impact:** On a system with a high-end discrete GPU and an integrated
GPU, `dev_by_type(GPU)` returns whichever was registered first. If the
IGPU was registered first (unlikely but possible), `init_best` returns
the IGPU. The user must explicitly call `init_by_name("CUDA0")` to get
the right device.

### W5 — `GGML_DISABLE_VULKAN` is a one-off

**Evidence:** `ggml-backend-reg.cpp:129-136` checks
`getenv("GGML_DISABLE_VULKAN")` inline in the constructor. No other
backend has an analogous env var.

**Impact:** A user who knows their CUDA driver is broken cannot tell the
registry to skip CUDA at runtime; they must rebuild without
`GGML_USE_CUDA`. The Vulkan special-case is inconsistent. See Finding F11.

### W6 — `unload_backend` cannot actually unload

**Evidence:** `ggml-backend-reg.cpp:176-184` — destructor calls
`entry.handle.release()` to *prevent* `dlclose` from running. Comment:
"backends cannot be safely unloaded without a function to destroy all the
backend resources, since backend threads may still be running."

**Impact:** Dynamic unloading is a dead feature. The `ggml_backend_unload`
API exists (line 397) but using it on a backend that has been initialized
leaks the .so handle. See Finding F10.

### W7 — `api_version` is just an integer

**Evidence:** `ggml-backend-impl.h:11` `#define GGML_BACKEND_API_VERSION 2`;
check at `ggml-backend-reg.cpp:246`.

**Impact:** No struct-layout fingerprint. A backend compiled against v2
but with a different `ggml_backend_device_i` struct layout (e.g., a
private fork added a method) will load and crash on the first call. There
is no `sizeof(struct)` check.

### W8 — `get_proc_address` strings are not namespaced

**Evidence:** CPU uses `"ggml_backend_set_n_threads"`,
`"ggml_backend_cpu_numa_init"`, `"ggml_threadpool_new"`. CUDA uses
`"ggml_backend_comm_init"`, `"ggml_backend_register_host_buffer"`. There
is no `"<backend>_<function>"` convention enforced.

**Impact:** A future backend that publishes a function with the same name
as another backend's function would shadow it. Not a bug today; latent.

### W9 — `dev_by_type` returns the first match

**Evidence:** `ggml-backend-reg.cpp:355-363` — linear scan, returns on
first match.

**Impact:** No way to enumerate *all* devices of a given type via this
function. Callers who want all GPUs must iterate `dev_count` and filter.
The `meta` backend (ggml-backend-meta.cpp) was added partly to address
this for tensor-parallel setups.

### W10 — `load_best` re-dlopens the winning .so

**Evidence:** `ggml-backend-reg.cpp:517` dlopens every candidate to call
`score_fn`; the unique_ptr deleter then closes it. The winning .so is
re-dlopened at line 261 (via `load_backend`).

**Impact:** Wasted startup time. On Linux `dlopen` of an already-mapped
.so is cached by the dynamic linker, so the second `dlopen` is cheap, but
the first `dlclose` may unmap the .so if no other reference holds it.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `GATE`          | **ADOPT** | Three-tier vtable (`reg_i` / `device_i` / `backend_i`) | Clean separation of concerns; same as ggml. |
| `GATE`          | **ADOPT** | `get_proc_address` string-keyed extension hook | Lets backends expose non-ABI'd functions to each other without version bumps. |
| `GATE`          | **ADOPT** | `offload_op` optional device hook | Decouples weight placement from op placement; essential for CPU+GPU hybrid. |
| `GATE`          | **ADOPT** | Score-based multi-binary dispatch | Picks best ISA variant at runtime; works for any backend. |
| `GATE`          | **ADAPT** | Meyers singleton registry | Keep the singleton, but add a RwLock so runtime registration is safe. |
| `GATE`          | **ADAPT** | `init_best` priority order | Keep GPU→IGPU→CPU, but add cross-device scoring (memory_total, compute capability). |
| `GATE`          | **ADAPT** | Hardcoded 15-name `load_all` list | Replace with a manifest file or directory glob; new backends should not require source edits. |
| `GATE`          | **REJECT**| `GGML_DISABLE_VULKAN` one-off | Every backend should support a `GGML_DISABLE_<NAME>` env var. |
| `GATE`          | **REJECT**| `unload_backend` that leaks the handle | Either implement a real `backend_shutdown` vtable method or remove the unload API. |
| `GATE`          | **MONITOR**| Integer `api_version` | Works for now; if GwenLand ever ships a stable ABI, replace with a struct-layout hash. |
| `glproc`        | **ADOPT** | `ggml_backend_cpu_score` pattern (ARTX01-F12) | Bitmask score; one .so per ISA target. |
| `glcuda` / `glmetal` / `glvulkan` | **ADOPT** | `GGML_BACKEND_DL_IMPL` macro | Same macro lets each backend emit the `ggml_backend_init` symbol. |

---

## 16. Recommendations

### R1 — ADOPT three-tier vtable hierarchy
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
GwenLand's `GATE` should define `gl_reg_i`, `gl_device_i`, `gl_backend_i` vtables with the same shape: `reg_i` exposes name + device count + device getter + `get_proc_address`; `device_i` exposes capabilities + `supports_op` + `offload_op` + buffer types; `backend_i` exposes `graph_compute` + async ops + events. Same ABI, same semantics.

### R2 — ADOPT `get_proc_address` extension hook
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
GwenLand backends will need to expose functions to each other (e.g., `glcuda` exposing `comm_init` for tensor parallelism, `glproc` exposing `set_n_threads`). The string-keyed lookup is the simplest mechanism that scales. Adopt it directly, but enforce a `"<backend>_<function>"` naming convention.

### R3 — ADOPT score-based multi-binary dispatch
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
GwenLand's `glproc` should ship as N .so files, one per ISA variant (Haswell, Skylake, IceLake, Zen4, Neoverse-V1, Neoverse-N2, …). Each .so exports `gl_backend_score()`. The loader picks the highest-scoring one. This is the only sane way to ship a CPU backend for diverse ISAs without runtime function-pointer dispatch.

### R4 — ADAPT Meyers singleton with RwLock
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
Keep the Meyers singleton for static-init thread-safety, but add a `std::shared_mutex` around `register_backend`, `register_device`, `load_backend`, `unload_backend`. Readers (`dev_count`, `dev_get`, `dev_by_name`, `dev_by_type`) take a shared lock; writers take an exclusive lock. This makes runtime registration safe.

### R5 — ADAPT `init_best` with cross-device scoring
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Replace the simple GPU→IGPU→CPU fallback with a scoring function that considers `memory_total`, compute capability, and device type. Among multiple GPUs, prefer the one with the most memory. Document the scoring function so users can predict which device will be selected.

### R6 — REJECT hardcoded `load_all` list; replace with manifest
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
Either glob `libgl-*.so` from the search path (and use the score function to filter) or read a manifest file listing backend names and search order. Adding a new backend should not require editing a source file.

### R7 — REJECT `GGML_DISABLE_VULKAN` one-off; generalize
**Priority:** Low
**Difficulty:** XS
**Dependencies:** R1
Every backend should respect `GL_DISABLE_<NAME>` env var. Implement this in the registry constructor, not per-backend.

### R8 — REJECT the unload-without-shutdown pattern
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R1
Either (a) add a `backend_shutdown` vtable method that backends must implement to join threads and free resources, and call it from `unload_backend` before `dlclose`; or (b) remove `unload_backend` from the public API. The current state (API exists, leaks the handle) is the worst of both worlds.

### R9 — MONITOR `api_version` integer
**Priority:** Low
**Difficulty:** S
**Dependencies:** R1
Keep the integer check for now, but if GwenLand ever ships a stable ABI, replace it with a struct-layout fingerprint (e.g., hash of the vtable struct definition). The integer alone cannot catch layout drift.

### R10 — ADOPT `offload_op` device hook
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
GwenLand's `glcuda` and `glmetal` should implement `offload_op` to claim `MUL_MAT` ops whose weights are CPU-resident when batch size is large enough. This decouples weight placement from op placement and is essential for hybrid CPU+GPU inference.

---

## 17. Findings

### Finding ARTX23-F01

```
Finding ID:           ARTX23-F01
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Backend registry
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_registry (ctor), register_backend, register_device, load_backend, unload_backend
Lines:                115-289
Summary:              The registry's mutation paths (register/unregister/load) are not
                      thread-safe; only the static-init of the singleton itself is.
Observation:          The registry is a Meyers singleton (get_reg() at line 292 returns
                      a function-local static). C++11 guarantees the constructor runs
                      exactly once, thread-safely. After construction, however, the
                      vectors `backends` and `devices` are mutated by register_backend,
                      register_device, load_backend, and unload_backend without any
                      mutex. Public API entries ggml_backend_register (line 298) and
                      ggml_backend_load (line 393) expose these mutations to any
                      caller. No documentation forbids calling them from non-main
                      threads. The codebase does not use ggml_critical_section
                      (audited in ggml-threading.cpp) anywhere in the registry.
Evidence:             ggml-backend-reg.cpp:115-289 (registry struct + mutators);
                      ggml-threading.cpp:4-11 (ggml_critical_section exists but is
                      unused in this file).
Architectural Impact: Runtime registration from worker threads would race on the
                      vectors. In practice no caller does this; the risk is latent.
Correctness Impact:   None today. Would become a UB bug if a future caller registers
                      a backend from a worker thread.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       ADAPT. Add a std::shared_mutex: writers (register/unload) take
                      exclusive lock; readers (dev_count/dev_get/dev_by_name/dev_by_type)
                      take shared lock. Keep the Meyers singleton for static-init safety.
Priority:             High
Difficulty:           S
Dependencies:         R1, R4
Confidence:           High
```

### Finding ARTX23-F02

```
Finding ID:           ARTX23-F02
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Backend registry constructor
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_registry::ggml_backend_registry
Lines:                119-174
Summary:              The registry constructor hardcodes the registration order of 15
                      backends via #ifdef GGML_USE_*; CPU is registered last by design.
Observation:          The constructor body is a sequence of `#ifdef GGML_USE_CUDA
                      register_backend(ggml_backend_cuda_reg()); #endif` blocks in
                      a fixed order: CUDA, Metal, SYCL, Vulkan, WebGPU, ZDNN, VirtGPU,
                      OpenCL, ZenDNN, Hexagon, CANN, BLAS, RPC, OpenVINO, ET, CPU.
                      CPU is last so that dev_by_type(GPU) returns a GPU before the
                      CPU device is in the list, which init_best relies on. Adding a
                      new backend requires editing this constructor and adding a new
                      #ifdef. There is no plugin manifest or auto-discovery.
Evidence:             ggml-backend-reg.cpp:119-174.
Architectural Impact: New backends require source edits to the registry. Order
                      subtly affects dev_by_type first-match semantics.
Correctness Impact:   None. Order is deterministic.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       ADAPT. Keep the #ifdef ladder for statically-linked backends,
                      but add a manifest-based or glob-based discovery for dynamically
                      loadable backends so adding one does not require source edits.
Priority:             Medium
Difficulty:           M
Dependencies:         R6
Confidence:           High
```

### Finding ARTX23-F03

```
Finding ID:           ARTX23-F03
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Dynamic-load symbol emission
Source File:          ggml/src/ggml-backend-impl.h
Function:             GGML_BACKEND_DL_IMPL, GGML_BACKEND_DL_SCORE_IMPL
Lines:                232-271
Summary:              The GGML_BACKEND_DL_IMPL macro emits the C-linkage
                      ggml_backend_init symbol the loader looks for; backends
                      use it to publish both a static reg_fn and a dynamic-load
                      entry point from the same source.
Observation:          When GGML_BACKEND_DL is defined, the macro expands to:
                        extern "C" GGML_BACKEND_API ggml_backend_reg_t
                        ggml_backend_init(void) { return reg_fn(); }
                      When GGML_BACKEND_DL is NOT defined, the macro expands to
                      nothing — the backend is statically linked and only the
                      reg_fn is called directly from the registry constructor.
                      GGML_BACKEND_DL_SCORE_IMPL does the same for
                      ggml_backend_score. This lets the same source file
                      (e.g., ggml-cpu.cpp:707) be built either as a static
                      lib linked into the main binary or as a standalone .so.
Evidence:             ggml-backend-impl.h:240-271; ggml-cpu.cpp:707;
                      ggml-cuda.cu:5425; ggml-metal.cpp:950.
Architectural Impact: One source file, two build modes. No code duplication
                      between static and dynamic backend builds.
Correctness Impact:   None.
Optimization Type:    Build-system simplification.
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Replicate the macro pattern in GwenLand: every
                      backend source file uses GL_BACKEND_DL_IMPL(reg_fn) so
                      the same source builds as static or dynamic.
Priority:             High
Difficulty:           XS
Dependencies:         R1
Confidence:           High
```

### Finding ARTX23-F04

```
Finding ID:           ARTX23-F04
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Backend selection (multi-binary ISA dispatch)
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_load_best
Lines:                480-560
Summary:              load_best enumerates libggml-<name>-*.so files, calls
                      ggml_backend_score() in each, and keeps the highest-scoring
                      variant. This is the cross-backend generalization of the
                      CPU multi-binary pattern (ARTX01-F12).
Observation:          load_best constructs a file prefix "libggml-<name>-" and
                      extension ".so" (or "ggml-<name>-" / ".dll" on Windows),
                      iterates every regular file in every search path that
                      matches the prefix+extension, dlopens it, looks up
                      ggml_backend_score, calls it, and remembers the path
                      with the highest score. The handle is then released
                      (unique_ptr deleter). If best_score == 0, falls back to
                      the bare name "libggml-<name>.so". The winning path is
                      re-dlopened via load_backend at line 261. This is the
                      only mechanism that lets multiple compiled variants of
                      the same backend coexist on disk and have the loader
                      pick the right one.
Evidence:             ggml-backend-reg.cpp:480-560; cpu-feats.cpp:263-325
                      (the score function itself, cited from ARTX01-F12).
Architectural Impact: Ship N .so files (one per ISA target), loader picks
                      the best. No runtime function-pointer dispatch inside
                      the .so.
Correctness Impact:   None.
Optimization Type:    Multi-binary ISA dispatch.
GwenLand Target:      GATE, glproc
Recommendation:       ADOPT. glproc should ship as N .so files (Haswell,
                      Skylake, IceLake, Zen4, Neoverse-V1, …) and the loader
                      should pick the best via the score function. Same
                      mechanism, no per-backend special-casing.
Priority:             High
Difficulty:           M
Dependencies:         R3
Confidence:           High
```

### Finding ARTX23-F05

```
Finding ID:           ARTX23-F05
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Backend ABI version check
Source File:          ggml/src/ggml-backend-impl.h, ggml/src/ggml-backend-reg.cpp
Function:             GGML_BACKEND_API_VERSION (define), load_backend (check)
Lines:                impl.h:11; reg.cpp:246-257
Summary:              ABI compatibility is verified by a single integer,
                      GGML_BACKEND_API_VERSION = 2. No struct-layout hash, no
                      symbol fingerprint.
Observation:          load_backend calls backend_init_fn() to obtain a
                      ggml_backend_reg_t, then checks reg->api_version !=
                      GGML_BACKEND_API_VERSION. On mismatch it logs and
                      returns nullptr. The integer is currently 2. There is
                      no sizeof check on the vtable structs, no symbol
                      fingerprint, no source-hash. A backend compiled against
                      v2 but with a privately-modified ggml_backend_device_i
                      struct (e.g., a fork added a method) will load and
                      crash on the first call to the new method.
Evidence:             ggml-backend-impl.h:11; ggml-backend-reg.cpp:246-257.
Architectural Impact: Minimal but sufficient for "stale .so in install dir".
                      Insufficient for "forked ABI drift".
Correctness Impact:   None today. Latent risk if the ABI ever forks.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       MONITOR. Keep the integer check; if GwenLand ever ships
                      a stable ABI, add a struct-layout hash (e.g., hash of
                      the vtable struct definition compiled into both the
                      loader and the backend).
Priority:             Low
Difficulty:           S
Dependencies:         R9
Confidence:           High
```

### Finding ARTX23-F06

```
Finding ID:           ARTX23-F06
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Cross-backend function exposure (get_proc_address)
Source File:          ggml/src/ggml-cpu/ggml-cpu.cpp, ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-metal/ggml-metal.cpp
Function:             ggml_backend_cpu_get_proc_address, ggml_backend_cuda_reg_get_proc_address, ggml_backend_metal_get_proc_address
Lines:                cpu.cpp:646-685; cuda.cu:5317-5338; metal.cpp:871-879
Summary:              Each backend implements get_proc_address as a string-keyed
                      lookup that returns function pointers for non-ABI'd
                      operations. CPU exposes 10 functions; CUDA exposes 6;
                      Metal exposes 1.
Observation:          The reg_i vtable includes an optional get_proc_address
                      method. When a caller (e.g., llama.cpp) wants a function
                      the standard ABI doesn't formalize, it calls
                      ggml_backend_reg_get_proc_address(reg, name) and gets
                      back a void*. The header (ggml-backend.h:203-223)
                      lists "common functions that may be obtained" and
                      typedefs the function pointer types, but each backend
                      picks which subset to publish. CPU publishes
                      set_n_threads, set_abort_callback, numa_init, set_use_ref,
                      threadpool_new/free/set_threadpool, get_features,
                      dev_get_extra_bufts. CUDA publishes comm_init,
                      comm_free, comm_allreduce_tensor (for tensor parallelism),
                      register/unregister_host_buffer, get_features. Metal
                      publishes only get_features. Vulkan publishes nothing.
                      The set is ad-hoc and grows by source edit.
Evidence:             cpu.cpp:646-685; cuda.cu:5317-5338; metal.cpp:871-879;
                      ggml-backend.h:203-223 (typedefs).
Architectural Impact: Backends can expose non-ABI'd functions to each other
                      without bumping GGML_BACKEND_API_VERSION. The cost is
                      that the set is not standardized — callers must know
                      which backends expose which strings.
Correctness Impact:   None. Lookup returns NULL if the name is not found.
Optimization Type:    String-keyed extension hook.
GwenLand Target:      GATE, glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Replicate the mechanism in GwenLand, but enforce
                      a "<backend>_<function>" naming convention so names
                      cannot collide across backends.
Priority:             High
Difficulty:           S
Dependencies:         R2
Confidence:           High
```

### Finding ARTX23-F07

```
Finding ID:           ARTX23-F07
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Device enumeration
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_dev_count, ggml_backend_dev_get, ggml_backend_dev_by_name, ggml_backend_dev_by_type
Lines:                336-363
Summary:              Devices are flattened across all backends into a single
                      vector; dev_by_name and dev_by_type do linear scans and
                      return the first match.
Observation:          The registry holds a single std::vector<ggml_backend_dev_t>
                      devices that is the union of every device of every
                      backend, in registration order. dev_count returns its
                      size; dev_get(i) indexes it; dev_by_name and dev_by_type
                      linear-scan and return the first match. There is no
                      per-backend sub-listing at the public API level; the
                      only way to enumerate a single backend's devices is
                      ggml_backend_reg_dev_count(reg) + dev_get(reg, i).
                      There is no way to enumerate *all* devices of a given
                      type — dev_by_type returns one. Callers who want all
                      GPUs must iterate dev_count and filter by dev_type(dev).
Evidence:             ggml-backend-reg.cpp:336-363.
Architectural Impact: O(N) lookup by name/type; N is total device count,
                      typically < 16. Not a bottleneck, but the API does
                      not expose "all GPUs" directly.
Correctness Impact:   None.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       ADAPT. Keep the flat device vector, but add
                      gl_dev_by_type_all(type, out, n_out) that returns
                      every matching device, so the scheduler can build a
                      multi-GPU plan without iterating the whole list.
Priority:             Medium
Difficulty:           XS
Dependencies:         R1
Confidence:           High
```

### Finding ARTX23-F08

```
Finding ID:           ARTX23-F08
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Backend auto-selection (init_best)
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_init_best
Lines:                382-390
Summary:              init_best picks the first GPU, else first IGPU, else
                      first CPU. No cross-device scoring; no ACCEL fallback.
Observation:          init_best is a 3-line cascade:
                        dev = dev_by_type(GPU) ? dev : dev_by_type(IGPU);
                        dev = dev ? dev : dev_by_type(CPU);
                        return dev ? dev_init(dev, nullptr) : nullptr;
                      On a system with multiple GPUs, dev_by_type(GPU)
                      returns whichever was registered first. If the IPU
                      was registered before the dGPU (unlikely but
                      possible), init_best returns the IGP. There is no
                      scoring function (memory_total, compute capability)
                      to pick the best GPU. There is also no ACCEL fallback:
                      if no GPU/IGPU/CPU is found (impossible in practice
                      but the API permits it), init_best returns nullptr
                      without trying ACCEL.
Evidence:             ggml-backend-reg.cpp:382-390.
Architectural Impact: Multi-GPU systems get the wrong default backend
                      unless the user calls init_by_name explicitly.
Correctness Impact:   None — the chosen backend is correct, just not optimal.
Optimization Type:    None (absence of optimization).
GwenLand Target:      GATE
Recommendation:       ADAPT. Replace the cascade with a scoring function:
                      among all GPU devices, pick the one with the most
                      memory_total; tiebreak by compute capability. Document
                      the scoring function so users can predict the result.
Priority:             Medium
Difficulty:           S
Dependencies:         R5
Confidence:           High
```

### Finding ARTX23-F09

```
Finding ID:           ARTX23-F09
Category:             ADOPT
Engine:               Shared
Component:            Backend offload hook (offload_op)
Source File:          ggml/src/ggml-backend-impl.h, ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-metal/ggml-metal.cpp
Function:             ggml_backend_device_i::offload_op, ggml_backend_cuda_device_offload_op, ggml_backend_metal_device_offload_op
Lines:                impl.h:194-196; cuda.cu:5184-5188, 5232; metal.cpp:757-763, 809
Summary:              offload_op is an optional device method that lets a GPU
                      backend claim a CPU-resident MUL_MAT when batch size
                      crosses a threshold. CPU, Vulkan, etc. set it to NULL.
Observation:          The device_i vtable includes an optional offload_op
                      method. The scheduler (ARTX22) consults it to decide
                      whether to assign an op to a backend even when the op's
                      weights are in an incompatible buffer (e.g., CPU host
                      buffer). CUDA implements it as:
                        return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
                      Metal implements it as:
                        return (op->op == MUL_MAT || op->op == MUL_MAT_ID) &&
                               get_op_batch_size(op) >= op_offload_min_batch_size;
                      CPU, Vulkan, BLAS, etc. set it to NULL, which the
                      dispatch wrapper (ggml-backend.cpp:633-640) treats as
                      false. This decouples weight placement from op
                      placement: a model's weights can stay in CPU memory
                      while the matmul runs on GPU when the batch is large
                      enough to amortize the H2D copy.
Evidence:             impl.h:194-196; cuda.cu:5184-5188, 5232; metal.cpp:757-763,
                      809; ggml-backend.cpp:633-640 (wrapper).
Architectural Impact: Hybrid CPU+GPU inference without forcing the user to
                      choose where every weight lives. The threshold is
                      per-device and configurable.
Correctness Impact:   None. The op produces the same result on either backend.
Optimization Type:    Cross-backend op stealing.
GwenLand Target:      glcuda, glmetal, GATE
Recommendation:       ADOPT. glcuda and glmetal should implement offload_op
                      with the same batch-size threshold pattern. GATE should
                      consult it during scheduling.
Priority:             High
Difficulty:           S
Dependencies:         R10
Confidence:           High
```

### Finding ARTX23-F10

```
Finding ID:           ARTX23-F10
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Backend unloading
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ~ggml_backend_registry, unload_backend
Lines:                176-184, 266-289
Summary:              The registry destructor explicitly leaks every dl handle
                      (entry.handle.release()) because backends cannot be
                      safely dlclose'd while their threads are still running.
                      unload_backend exists as a public API but is effectively
                      broken.
Observation:          The destructor comment at line 177 says: "FIXME: backends
                      cannot be safely unloaded without a function to destroy all
                      the backend resources, since backend threads may still be
                      running and accessing resources from the dynamic library."
                      The destructor therefore calls entry.handle.release() to
                      prevent the unique_ptr deleter from running dlclose.
                      unload_backend (lines 266-289) erases the reg entry from
                      the vectors but does not dlclose the handle either —
                      the unique_ptr deleter would run, but only if the
                      entry's handle was moved out, which it isn't. In effect,
                      unload_backend just removes the reg from enumeration
                      while leaving the .so mapped. The public API
                      ggml_backend_unload (line 397) calls unload_backend(reg,
                      true) and is silent about this limitation.
Evidence:             ggml-backend-reg.cpp:176-184, 266-289, 393-399.
Architectural Impact: Dynamic unloading is a dead feature. Any caller using
                      ggml_backend_unload will leak the .so handle and any
                      resources the backend allocated.
Correctness Impact:   None (correct, just leaks). Use after unload would be
                      a caller bug.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       REJECT this pattern. Either (a) add a backend_shutdown
                      vtable method that backends must implement to join
                      threads and free resources, and call it from unload
                      before dlclose; or (b) remove the unload API.
Priority:             Medium
Difficulty:           L
Dependencies:         R8
Confidence:           High
```

### Finding ARTX23-F11

```
Finding ID:           ARTX23-F11
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Runtime backend disable
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_registry (ctor)
Lines:                129-136
Summary:              GGML_DISABLE_VULKAN env var is the only runtime-disable
                      switch in the registry constructor. Every other backend
                      is gated only by compile-time #ifdef GGML_USE_*.
Observation:          The Vulkan branch of the constructor is:
                        #ifdef GGML_USE_VULKAN
                          if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
                            register_backend(ggml_backend_vk_reg());
                          } else {
                            GGML_LOG_DEBUG("Vulkan backend disabled ...");
                          }
                        #endif
                      No other backend has an analogous env var. A user who
                      knows their CUDA driver is broken cannot tell the
                      registry to skip CUDA at runtime; they must rebuild
                      without GGML_USE_CUDA. The Vulkan special-case is
                      inconsistent and apparently added in response to a
                      specific user complaint (no PR reference in the source).
Evidence:             ggml-backend-reg.cpp:129-136.
Architectural Impact: Inconsistent UX. Vulkan can be disabled at runtime;
                      CUDA/Metal/SYCL/etc. cannot.
Correctness Impact:   None.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       REJECT this pattern. Every backend should respect
                      GL_DISABLE_<NAME> env var. Implement this once in the
                      registry constructor, not per-backend.
Priority:             Low
Difficulty:           XS
Dependencies:         R7
Confidence:           High
```

### Finding ARTX23-F12

```
Finding ID:           ARTX23-F12
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Dynamic load_all enumeration
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_load_all_from_path
Lines:                566-593
Summary:              load_all hardcodes a 15-name list of backend names to
                      probe. Adding a new dynamically-loadable backend
                      requires editing this list.
Observation:          load_all_from_path calls ggml_backend_load_best for
                      each of: blas, zendnn, cann, cuda, hip, metal, rpc,
                      sycl, vulkan, virtgpu, opencl, hexagon, musa, openvino,
                      cpu. The list is in source, not in a manifest file.
                      A new backend whose name is not on the list can still
                      be loaded via the GGML_BACKEND_PATH env var (line 589),
                      but not via the standard discovery mechanism. The list
                      is also order-dependent: cpu is last, so if a system
                      has both a CPU .so and a CUDA .so, the CPU .so is
                      probed last (which is correct, but only by accident).
Evidence:             ggml-backend-reg.cpp:566-593.
Architectural Impact: New backends require source edits. No out-of-tree
                      backend can be auto-discovered without env-var help.
Correctness Impact:   None.
Optimization Type:    None.
GwenLand Target:      GATE
Recommendation:       ADAPT. Replace the hardcoded list with either (a) a
                      glob over libgl-*.so in the search path (using the
                      score function to filter), or (b) a manifest file
                      listing backend names and search order. Adding a new
                      backend should not require editing a source file.
Priority:             Medium
Difficulty:           M
Dependencies:         R6
Confidence:           High
```

### Finding ARTX23-F13

```
Finding ID:           ARTX23-F13
Category:             BACKEND_DESIGN
Engine:               Shared
Component:            Search path assembly
Source File:          ggml/src/ggml-backend-reg.cpp
Function:             ggml_backend_load_best, get_executable_path
Lines:                486-496, 401-462
Summary:              load_best searches GGML_BACKEND_DIR (compile-time),
                      then the executable's directory, then the current
                      working directory. The path resolution uses
                      /proc/self/exe on Linux, _NSGetExecutablePath on
                      macOS, GetModuleFileNameW on Windows.
Observation:          When user_search_path is nullptr, load_best assembles
                      a 3-entry search_paths vector: GGML_BACKEND_DIR
                      (compile-time #define, optional), get_executable_path()
                      (the directory containing the running binary), and
                      fs::current_path() (the CWD). get_executable_path is
                          Linux: readlink("/proc/self/exe")
                          FreeBSD: readlink("/proc/curproc/file")
                          macOS: _NSGetExecutablePath
                          Windows: GetModuleFileNameW(NULL, ...)
                          other: returns empty path
                      When user_search_path is provided, only that path is
                      searched. This is the mechanism ggml_backend_load_all_from_path
                      uses to load from a user-specified directory.
Evidence:             ggml-backend-reg.cpp:486-496, 401-462.
Architectural Impact: .so files are found relative to the binary, not
                      relative to a system library path. This makes ggml
                      "relocatable" — copy the binary and its .so dir
                      together and it works.
Correctness Impact:   None.
Optimization Type:    Relocatable install.
GwenLand Target:      GATE
Recommendation:       ADOPT. Same search-path strategy for GwenLand:
                      compile-time GL_BACKEND_DIR, then executable dir,
                      then CWD. Use the same per-OS executable-path
                      resolution.
Priority:             Medium
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether `ggml_backend_register` (the public API at line 298) is
  ever called from outside the registry constructor. Static analysis of
  the llama.cpp source tree would answer this; not in scope here. If no
  external caller uses it, the thread-safety concern (F01) is purely
  theoretical.

* **U2**. Whether any downstream user of ggml (KoboldCpp, Ollama, LM
  Studio, etc.) relies on `ggml_backend_unload` actually unloading. If
  they do, they are silently leaking the .so. Requires surveying
  downstream consumers.

* **U3**. Whether the `api_version = 2` bump from `1` involved a struct
  layout change. The git history would reveal this; the audited commit
  only shows the current state. If v1 and v2 have the same struct layout,
  the integer check is purely cosmetic; if they differ, the check is
  load-bearing.

* **U4**. Whether the meta backend (`ggml-backend-meta.cpp`) is intended
  to replace the flat device list with a hierarchical one. The meta
  backend wraps multiple physical devices in a single virtual device for
  tensor parallelism, but it is registered as a separate backend, not as
  a replacement for the flat list. Requires reading the meta backend's
  own design doc (not present in the audited tree).

* **U5**. Whether `ggml_backend_load_best`'s re-dlopen of the winning
  .so (W10) actually incurs a measurable cost on Linux. The dynamic
  linker may cache the .so after the first dlclose, making the second
  dlopen cheap. Requires runtime measurement.

* **U6**. Whether `op_offload_min_batch_size` (used by CUDA and Metal
  `offload_op`) is configurable per-model or hardcoded per-device. The
  value is set in `dev_ctx` initialization; tracing that requires
  reading the CUDA/Metal device init code, which is audited in ARTX11
  and ARTX17.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-backend-reg.cpp`                     | `ggml_backend_registry` (struct)               | 115-118       |
| R02       | `ggml/src/ggml-backend-reg.cpp`                     | `ggml_backend_registry` (ctor)                 | 119-174       |
| R03       | `ggml/src/ggml-backend-reg.cpp`                     | `~ggml_backend_registry`                       | 176-184       |
| R04       | `ggml/src/ggml-backend-reg.cpp`                     | `register_backend`, `register_device`          | 186-218       |
| R05       | `ggml/src/ggml-backend-reg.cpp`                     | `load_backend`                                 | 220-264       |
| R06       | `ggml/src/ggml-backend-reg.cpp`                     | `unload_backend`                               | 266-289       |
| R07       | `ggml/src/ggml-backend-reg.cpp`                     | `get_reg` (Meyers singleton)                   | 292-295       |
| R08       | `ggml/src/ggml-backend-reg.cpp`                     | `ggml_backend_register`, `ggml_backend_device_register` | 298-304 |
| R09       | `ggml/src/ggml-backend-reg.cpp`                     | `dev_count`, `dev_get`, `dev_by_name`, `dev_by_type` | 336-363 |
| R10       | `ggml/src/ggml-backend-reg.cpp`                     | `init_by_name`, `init_by_type`, `init_best`    | 366-390       |
| R11       | `ggml/src/ggml-backend-reg.cpp`                     | `ggml_backend_load_best`                       | 480-560       |
| R12       | `ggml/src/ggml-backend-reg.cpp`                     | `ggml_backend_load_all_from_path`              | 566-593       |
| R13       | `ggml/src/ggml-backend-reg.cpp`                     | `get_executable_path`                          | 401-462       |
| R14       | `ggml/src/ggml-backend-dl.cpp`                      | `dl_load_library`, `dl_get_sym`, `dl_error`    | 1-48          |
| R15       | `ggml/src/ggml-backend-dl.h`                        | `dl_handle_ptr`, deleter                       | 1-44          |
| R16       | `ggml/src/ggml-backend-impl.h`                      | `GGML_BACKEND_API_VERSION`                     | 11            |
| R17       | `ggml/src/ggml-backend-impl.h`                      | `ggml_backend_buffer_type_i`, `buffer_i`       | 17-70         |
| R18       | `ggml/src/ggml-backend-impl.h`                      | `ggml_backend_i`                               | 105-140       |
| R19       | `ggml/src/ggml-backend-impl.h`                      | `ggml_backend_device_i`                        | 160-202       |
| R20       | `ggml/src/ggml-backend-impl.h`                      | `ggml_backend_reg_i`, `ggml_backend_reg`       | 214-230       |
| R21       | `ggml/src/ggml-backend-impl.h`                      | `GGML_BACKEND_DL_IMPL`, `GGML_BACKEND_DL_SCORE_IMPL` | 232-271 |
| R22       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_reg_get_proc_address`            | 659-665       |
| R23       | `ggml/src/ggml-backend.cpp`                         | `ggml_backend_dev_offload_op` (wrapper)        | 633-640       |
| R24       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_get_proc_address`            | 646-685       |
| R25       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_device_get_props` (caps)     | 390-401       |
| R26       | `ggml/src/ggml-cpu/ggml-cpu.cpp`                    | `ggml_backend_cpu_reg`, `GGML_BACKEND_DL_IMPL` | 694-707       |
| R27       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `ggml_backend_cpu_x86_score`                   | 263-325       |
| R28       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_backend_cuda_device_offload_op`          | 5184-5188     |
| R29       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_backend_cuda_reg_get_proc_address`       | 5317-5338     |
| R30       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_device_offload_op`         | 757-763       |
| R31       | `ggml/src/ggml-metal/ggml-metal.cpp`                | `ggml_backend_metal_get_proc_address`          | 871-879       |
| R32       | `ggml/include/ggml-backend.h`                       | `enum ggml_backend_dev_type`                   | 134-145       |
| R33       | `ggml/include/ggml-backend.h`                       | `struct ggml_backend_dev_props`, `_dev_caps`   | 148-177       |
| R34       | `ggml/include/ggml-backend.h`                       | "common functions" typedefs                    | 203-223       |
| R35       | `ggml/src/ggml-threading.cpp`                       | `ggml_critical_section_*`                      | 4-11          |
