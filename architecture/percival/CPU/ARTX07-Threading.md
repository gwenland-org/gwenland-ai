# ARTX07 — CPU Threading Model

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glproc` (threadpool, barrier, affinity), `GATE` (graph kickoff, abort)

---

## 1. Executive Summary

The CPU threading model is a **custom pthread-based persistent
threadpool** with an OpenMP fallback, a central-counter barrier,
hybrid spin-then-block polling, NUMA-aware affinity, and per-thread
CPU pinning. It sits underneath the SPMD-with-barrier graph executor
described in ARTX01.

Six primitives compose the model: (1) `struct ggml_threadpool`
holding a mutex+condvar, three cache-aligned hot atomics
(`n_barrier`, `n_barrier_passed`, `current_chunk`), a packed
`n_graph` counter, and three control atomics (`stop`, `pause`,
`abort`); (2) `ggml_barrier` — a central-counter barrier using
`seq_cst` on entry and exit fences with a TSAN-safe dummy RMW
workaround; (3) `ggml_graph_compute_secondary_thread` — a worker
spin loop polling `1024*128*poll` rounds before blocking on
`pthread_cond_wait`; (4) `ggml_graph_compute_kickoff` — bumps
`n_graph` with `seq_cst` and `cond_broadcast`s; (5) NUMA — three
strategies applied via `pthread_setaffinity_np`; (6) CPU affinity +
priority — per-worker `bool[GGML_MAX_N_THREADS]` mask and a `prio`
field mapped to `SCHED_FIFO` / `THREAD_PRIORITY_*`.

For GwenLand, **ADOPT** the packed `n_graph` counter, cache-aligned
hot atomics, hybrid poll-then-block, and abort sentinel. **REJECT**
the `bool[GGML_MAX_N_THREADS]` cpumask (should be a bitmap), the
Windows ≤64-CPU affinity limit, and the `memory_order_relaxed` on
the abort atomic (relies on barrier sequencing — fragile).
**ADAPT** the central-counter barrier (consider sense-reversing to
avoid the per-generation reset write) and the NUMA chunking fallback
(make the policy pluggable).

ARTX01 touched threading at a high level (F02, F09). This audit
deepens those and adds lifecycle, polling, priority, affinity, abort,
and TSAN dimensions.

---

## 2. Purpose

Provide the CPU backend's concurrency substrate: own and recycle
worker pthreads across graph executions; publish new work via a
single atomic store; synchronize threads between nodes via a cheap
barrier; expose per-thread CPU affinity and scheduling priority;
detect and adapt to multi-socket NUMA topology; allow cooperative
abort of an in-flight graph; degrade to OpenMP when the custom
backend is unavailable; expose a portable
`ggml_critical_section_*` API for one-time initialization.

It is **not** responsible for per-op work partitioning (each op
splits work itself via `ith`/`nth`), graph scheduling across
backends, or kernel selection (see ARTX01-F03).

---

## 3. Source Files

| File                                       | Lines  | Role                                                                          |
| ------------------------------------------ | ------ | ----------------------------------------------------------------------------- |
| `ggml/src/ggml-cpu/ggml-cpu.c`             | 3896   | Threadpool struct, barrier, NUMA, affinity, priority, worker loop, kickoff    |
| `ggml/src/ggml-threading.cpp`              | 13     | `ggml_critical_section_start/end` — `std::mutex`-backed, one-time-init only   |
| `ggml/src/ggml-threading.h`                | 15     | Public API for the critical section                                           |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h`        | 540    | `ggml_compute_params`, `ggml_barrier` / `ggml_threadpool_chunk_set/add` decls |
| `ggml/include/ggml.h`                      | ~2930  | `ggml_threadpool_params`, `GGML_MAX_N_THREADS=512`, `ggml_sched_priority`     |
| `ggml/include/ggml-cpu.h`                  | ~153   | `ggml_numa_strategy` enum, `ggml_numa_init` / `ggml_is_numa`                  |
| `ggml/src/ggml.c`                          | ~8020  | `ggml_threadpool_params_default/init/match`                                   |

Key `ggml-cpu.c` ranges: 55–74 (cache-line/TSAN macros),
421–516 (threadpool/worker structs), 519–538 (cpu_relax),
540–722 (NUMA init), 575–619 (barrier + chunk ops),
2148–2218 (NUMA affinity), 2493–2709 (per-platform
affinity/priority/cpumask_next), 2711–2779 (free/pause/resume),
3060–3271 (worker loop, poll, kickoff), 3273–3425 (new_impl,
graph_compute), 3818–3895 (cpu_init, KMP_BLOCKTIME).

---

## 4. Architecture Overview

```
                ┌──────────────────────────────────────────────────────────┐
                │   ggml_graph_compute(cgraph, cplan)  [ggml-cpu.c:3350]   │
                │   ├─ reuse existing pool OR create disposable            │
                │   ├─ reset: current_chunk=0, abort=-1, ec=SUCCESS        │
                │   └─ branch on GGML_USE_OPENMP                            │
                └──────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴──────────────────┐
            ▼                                    ▼
   ┌──────────────────────┐           ┌────────────────────────────────┐
   │ OpenMP path          │           │ Custom pthread path            │
   │ #pragma omp parallel │           │ ggml_graph_compute_kickoff()   │
   │ each thread runs     │           │  ├─ bump n_graph (seq_cst)     │
   │  graph_compute_      │           │  ├─ cond_broadcast(&cond)      │
   │  thread(&workers[i]) │           │  └─ main thread runs           │
   │                      │           │     graph_compute_thread(w[0]) │
   │ KMP_BLOCKTIME=200    │           └────────────────────────────────┘
   └──────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────────┐
              │ ggml_graph_compute_thread [3060]         │
              │  for each node:                          │
              │    set_numa_thread_affinity(ith)         │
              │    try_fuse_ops / compute_forward(node)  │
              │    if ith==0: maybe abort = N+1          │
              │    if not last node: ggml_barrier(tp)    │
              │  ggml_barrier(tp)  // final              │
              └──────────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────────┐
              │ ggml_barrier [575]                       │
              │  n_threads==1: noop                      │
              │  OpenMP: #pragma omp barrier             │
              │  custom: central-counter (seq_cst)       │
              └──────────────────────────────────────────┘

   Worker lifetime [ggml-cpu.c:3201]
   ┌────────────────────────────────────────────────────────────────────┐
   │ apply_priority(prio); apply_affinity(cpumask)                      │
   │ while (true) {                                                     │
   │   while (pause) cond_wait(cond, mutex);                            │
   │   if (stop) break;                                                 │
   │   check_for_work: poll n_rounds=1024*128*poll relax rounds,        │
   │                   else mutex_lock_shared + cond_wait               │
   │   if (pending) { pending=false; graph_compute_thread(state) }      │
   │ }                                                                  │
   └────────────────────────────────────────────────────────────────────┘
```

Key design points: two mutually exclusive backends selected at
compile time via `GGML_USE_OPENMP`; one atomic (`n_graph`) publishes
work + active thread count; three hot atomics cache-aligned; barrier
is a no-op for `n_threads == 1`; affinity + priority applied
per-thread; NUMA detection is Linux-only via `/sys/devices/system/node`.

---

## 5. Execution Flow

### 5.1 Threadpool creation

`ggml_threadpool_new_impl` (`ggml-cpu.c:3273`) allocates the pool
via `ggml_aligned_malloc`, initializes fields (`n_graph=0`,
`n_barrier=0`, `n_barrier_passed=0`, `current_chunk=0`,
`stop=false`, `pause=tpp->paused`, `abort=-1`), allocates
`workers[n_threads]`, and under pthread spawns workers 1..n-1 via
`pthread_create(ggml_graph_compute_secondary_thread, &workers[j])`.
The main thread (worker 0) is the caller's thread — not spawned.
Each worker's cpumask is computed via `ggml_thread_cpumask_next`
before spawn; the main thread is placed last (higher-numbered CPUs)
per the comment at line 3321.

### 5.2 Graph compute entry

`ggml_graph_compute` (`ggml-cpu.c:3350`) calls `ggml_cpu_init()`
(idempotent), then either creates a disposable pool or resets the
existing pool's `cgraph`/`cplan`/`current_chunk`/`abort`/`ec`.
OpenMP path uses `#pragma omp parallel` + `#pragma omp single` to
update `n_graph`. Pthread path clamps `n_threads` to
`pool->n_threads`, calls `ggml_graph_compute_kickoff`, then the main
thread runs `ggml_graph_compute_thread(&workers[0])`. After all
workers finish: `clear_numa_thread_affinity()` on main thread,
return `pool->ec`.

### 5.3 Kickoff

`ggml_graph_compute_kickoff` (`ggml-cpu.c:3239`) locks the mutex,
reads `n_graph` (relaxed), extracts the generation counter (upper 16
bits), increments, re-packs with new active thread count (low 16),
stores with `memory_order_seq_cst`. The `seq_cst` is documented
(line 3252): "We need the full seq-cst fence here because of the
polling threads." If paused, re-applies main thread priority+affinity
and calls `resume_locked` (which broadcasts); else `cond_broadcast`.

### 5.4 Worker main loop

`ggml_graph_compute_secondary_thread` (`ggml-cpu.c:3201`) applies
priority and affinity, then enters `while (true)`: (a) inner
`while (pause)` cond_waits; (b) `if (stop) break`; (c)
`ggml_graph_compute_check_for_work` (poll then block); (d) if
pending, clear and call `ggml_graph_compute_thread`.

### 5.5 Per-graph worker

`ggml_graph_compute_thread` (`ggml-cpu.c:3060`) calls
`set_numa_thread_affinity(ith)`, builds `ggml_compute_params` with
`nth = n_graph & 0xffff`, then loops:
`for (node_n = 0; node_n < n_nodes && abort != node_n; node_n++)`:
skip empty/flagless nodes; try fusion; else `compute_forward`; if
`ith==0` and abort_callback fires, set `abort = node_n+1` (relaxed)
and `ec = ABORTED`; if not last node, `ggml_barrier(tp)`. Final
`ggml_barrier(tp)` after the loop.

### 5.6 Poll-then-block

`ggml_graph_compute_poll_for_work` (`ggml-cpu.c:3167`): spins
`n_rounds = 1024UL * 128 * poll` iterations of
`ggml_thread_cpu_relax()`, exiting early if `thread_ready` returns
true (compares `n_graph` to `state->last_graph`). If poll exhausts,
falls back to `mutex_lock_shared` + `cond_wait`.

### 5.7 Pause / resume / free

`pause` sets `pause=true` under mutex and broadcasts. `resume` sets
`pause=false` and broadcasts. `free` sets `stop=true`, `pause=false`,
broadcasts, joins workers 1..n-1, destroys mutex+cond, frees
pool+workers.

---

## 6. Data Layout

### 6.1 `struct ggml_threadpool` (`ggml-cpu.c:480`)

| Field                          | Type                  | Alignment     | Role                                              |
| ------------------------------ | --------------------- | ------------- | ------------------------------------------------- |
| `mutex`, `cond`                | mutex/cond            | default       | guards condvar + pause/stop state                 |
| `cgraph`, `cplan`              | pointers              | default       | current graph + plan                              |
| `n_graph`                      | `atomic_int`          | default       | packed: [gen : 16][active n_threads : 16]         |
| `n_barrier`                    | `atomic_int`          | **64B align** | central barrier counter                           |
| `n_barrier_passed`             | `atomic_int`          | **64B align** | central barrier sense                             |
| `current_chunk`                | `atomic_int`          | **64B align** | matmul chunk-stealing counter                     |
| `stop`, `pause`                | `atomic_bool`         | default       | pool lifecycle                                    |
| `abort`                        | `atomic_int`          | default       | abort sentinel (default `-1`)                     |
| `workers`                      | pointer               | default       | per-thread state array                            |
| `n_threads`, `prio`, `poll`    | int/int32/uint32      | default       | pool config                                       |
| `ec`                           | `enum ggml_status`    | default       | per-graph exit code                               |

The three cache-aligned atomics are the only explicit false-sharing
mitigation. Control atomics sit with `n_graph` and config fields but
are written rarely.

### 6.2 `struct ggml_compute_state` (`ggml-cpu.c:507`)

```c
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;       // pthread_t
    int  last_graph;          // last seen n_graph (polling)
    bool pending;             // work-is-pending flag
#endif
    bool cpumask[GGML_MAX_N_THREADS];  // 512 bytes (GGML_MAX_N_THREADS=512)
    struct ggml_threadpool * threadpool;
    int ith;
};
```

`cpumask` is 512 bytes per worker. With 64 workers, `workers[]` is
~33 KB, spanning ~512 cache lines (deepened in F13).

### 6.3 Mutex/cond macros

Windows: `SRWLOCK` + `CONDITION_VARIABLE` with
`SleepConditionVariableSRW(..., CONDITION_VARIABLE_LOCKMODE_SHARED)`.
Linux/BSD: `pthread_mutex_t` + `pthread_cond_t`, with
`ggml_mutex_lock_shared` aliased to plain `pthread_mutex_lock` (line
456) — **not** a read lock. The "shared" name is a portability
misnomer on pthreads.

### 6.4 NUMA structs (`ggml-cpu.c:544`)

`GGML_NUMA_MAX_NODES=8`, `GGML_NUMA_MAX_CPUS=512`. Static
`g_state.numa` is the only NUMA state — no per-pool NUMA context.
`cpuset` is `cpu_set_t` on Linux (captured via
`pthread_getaffinity_np`), `uint32_t` elsewhere.

### 6.5 `ggml_threadpool_params` (`ggml.h:2911`)

Fields: `bool cpumask[512]`, `int n_threads`, `enum ggml_sched_priority
prio` (LOW..REALTIME), `uint32_t poll` (0..100; default 50), `bool
strict_cpu`, `bool paused`. `ggml_threadpool_params_init`
(`ggml.c:8002`): `prio=0`, `poll=50`, `strict_cpu=false`,
`paused=false`, zeroed cpumask.

---

## 7. Memory Layout

### 7.1 Threadpool + worker allocation

Both allocated via `ggml_aligned_malloc` (`ggml-cpu.c:3279, 3299`).
The pool alignment ensures cache-aligned hot atomics start on a
cache-line boundary. Workers have no per-worker padding to prevent
false-sharing between adjacent `ggml_compute_state` structs — but
the 512-byte `cpumask` field guarantees each worker state spans ≥8
cache lines, so adjacent workers' `ith`/`pending` fields do not
alias (accidental mitigation via oversized struct).

### 7.2 Cache-line constants

Two distinct constants: `GGML_CACHE_LINE` (`ggml-cpu.c:60`,
hardcoded 64) used for `GGML_CACHE_ALIGN` on hot atomics; and
`CACHE_LINE_SIZE` (`ops.h:9-19`, uses
`std::hardware_destructive_interference_size` if available, else
64/128/256 by ISA) used for per-thread scratch offsets in
`params->wdata`. The two can disagree (e.g., POWER9:
`CACHE_LINE_SIZE=128` but `GGML_CACHE_LINE=64`).

### 7.3 Per-op scratch layout

`ggml_graph_plan` (`ggml-cpu.c:2781`) computes `work_size` by
walking every node and consulting `ggml_get_n_tasks`. For
`mul_mat_id`, reserves `CACHE_LINE_SIZE * n_as + CACHE_LINE_SIZE`
bytes for per-`cur_a` atomic chunk counters, each in its own cache
line via `incr_ptr_aligned` (line 1580).

---

## 8. Parallelism Strategy

This section is the meat of the audit. ARTX01 covered the SPMD model
at a high level; this section goes deep into the threading
primitives.

### 8.1 Two backends: OpenMP vs custom pthread

Selected at compile time via `GGML_USE_OPENMP`. OpenMP uses
`#pragma omp parallel` + `#pragma omp single`; the custom path
implements its own threadpool, polling, barrier, and lifecycle. The
custom path is the only one that supports `poll`, `prio`, paused
startup, persistent workers, and abort. OpenMP loses all of these.

To partially compensate, `ggml_cpu_init` sets `KMP_BLOCKTIME=200`
(line 3866) — the Intel OpenMP runtime env var keeping threads
spinning for 200ms before sleeping, approximating the custom path's
polling. The comment (line 3867): "less aggressive than setting the
wait policy to active, but should achieve similar results in most
cases." `pause`/`resume` are `UNUSED(threadpool)` under OpenMP
(line 2765, 2777).

### 8.2 Worker spin loop

`ggml_graph_compute_secondary_thread` (`ggml-cpu.c:3201`) is the
worker entry point. After applying priority and affinity, it enters
`while (true)` with three nested concerns: pause check
(`while (pause) cond_wait`), stop check (`if (stop) break`), and
work check (`ggml_graph_compute_check_for_work`).

The poll-then-block hybrid is the key latency design. Workers spin
for `n_rounds = 1024 * 128 * poll` rounds of `ggml_thread_cpu_relax`
before `pthread_cond_wait`. With default `poll=50`, that's ~6.5M
pause instructions (~300ms at 3GHz, since `_mm_pause` ≈ 140 cycles).
The comment (line 3170): "This seems to make 0...100 a decent range
for polling level across modern processors." The poll exits early if
work arrives. The block path always goes through mutex+cond_wait —
no "second short poll after wake."

### 8.3 Central-counter barrier

`ggml_barrier` (`ggml-cpu.c:575`):

```c
if (n_threads == 1) return;                          // fast path
int n_passed = atomic_load(&tp->n_barrier_passed);   // sample sense
int n_barrier = atomic_fetch_add(&tp->n_barrier, 1, seq_cst);  // enter
if (n_barrier == n_threads - 1) {                    // last arrival
    atomic_store(&tp->n_barrier, 0, relaxed);        // reset counter
    atomic_fetch_add(&tp->n_barrier_passed, 1, seq_cst);  // exit
    return;
}
while (atomic_load(&tp->n_barrier_passed) == n_passed) {  // spin
    ggml_thread_cpu_relax();
}
// exit fence (seq_cst or TSAN dummy RMW)
```

This is a **central-counter barrier with explicit reset**, not
sense-reversing. The last arrival resets `n_barrier` to 0 (relaxed)
and bumps `n_barrier_passed` (seq_cst). Other threads spin on
`n_barrier_passed` changing (relaxed). Because both atomics are
`GGML_CACHE_ALIGN` (64B), the reset of `n_barrier` does not perturb
the `n_barrier_passed` cache line that spinners read.

`seq_cst` on entry publishes prior writes; on exit, acquires the
published state. Under `GGML_TSAN_ENABLED` (detected via
`__has_feature(thread_sanitizer)` or `__SANITIZE_THREAD__`), the
exit fence is replaced by
`atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, seq_cst)` — a
dummy RMW, because TSAN does not support standalone fences (comment
at line 604). The same workaround appears in
`ggml_graph_compute_thread_sync` (line 3158-3163).

### 8.4 Packed `n_graph` counter

`n_graph` is a 32-bit `atomic_int` packing two values: low 16 bits
(`GGML_THREADPOOL_N_THREADS_MASK = 0xffff`) = active thread count
for the current graph; upper 16 bits = monotonic generation counter.
Kickoff reads relaxed, shifts right 16, increments, shifts back, ORs
in the new `n_threads`, stores with `seq_cst` (line 3246-3253).
Workers read `n_graph` relaxed: the barrier uses the low 16 bits as
the wait-for count (line 576); `compute_thread` uses them as `nth`
(line 3075); `thread_ready` compares the full 32-bit value to
`last_graph` to detect new work (line 3145-3150). One atomic store
publishes both "new work" and "how many threads participate."
Allows per-graph thread-count variation without pool recreation.

### 8.5 NUMA strategies

`ggml_numa_init` (`ggml-cpu.c:636`) reads
`/sys/devices/system/node/nodeN` and `/sys/devices/system/cpu/cpuN`
to enumerate topology. `ggml_is_numa()` returns `n_nodes > 1`.
`set_numa_thread_affinity` (`ggml-cpu.c:2148`) switches on
`numa_strategy`:

| Strategy                            | Behavior                                                    |
| ----------------------------------- | ----------------------------------------------------------- |
| `DISABLED` (0)                      | No-op (default before init)                                 |
| `DISTRIBUTE` (1)                    | `node_num = ith % n_nodes` — round-robin across nodes        |
| `ISOLATE` (2)                       | `node_num = current_node` — all on main thread's node        |
| `NUMACTL` (3)                       | Re-apply cpuset captured at init via `pthread_getaffinity_np`|
| `MIRROR` (4)                        | **Declared but unimplemented** — falls through to no-op      |

`MIRROR` is a dead enum: the switch has no case for it (line
2157-2175). Users passing `MIRROR` get no affinity silently — a
correctness gap.

### 8.6 NUMA-aware chunking

When `ggml_is_numa()` is true, `mul_mat` (line 1413) and
`mul_mat_id` (line 1667) collapse to one chunk per thread,
disabling dynamic stealing. The comment (line 1411) cites PR #6915:
"In theory, chunking should be just as useful on NUMA and non NUMA
systems, but testing disagreed with that." Deepens ARTX01-F09.

### 8.7 CPU affinity: `strict_cpu`

`ggml_thread_cpumask_next` (`ggml-cpu.c:2689`): if `!strict`,
`memcpy` the global mask to every worker (OS picks the actual core);
if `strict`, scan from `*iter` for the next set bit, set only that
bit, advance `*iter`. The modulo is "cheaper modulo" via subtraction
(line 2700). `strict_cpu` is the only way to get deterministic
per-core placement. Without it, the OS may bounce workers between
cores, defeating L1/L2 warmth.

### 8.8 Thread priority

Per-platform `ggml_thread_apply_priority`: Linux uses
`SCHED_BATCH` (LOW), `SCHED_OTHER` (NORMAL, no-op return),
`SCHED_FIFO` priorities 40/80/90 (MEDIUM/HIGH/REALTIME). macOS
similar but no `SCHED_BATCH`. Windows maps to
`THREAD_PRIORITY_BELOW_NORMAL`..`TIME_CRITICAL`; for `prio != LOW`,
calls `SetThreadInformation(ThreadPowerThrottling, StateMask=0)` to
disable Windows 11's aggressive core parking (line 2548-2559). The
comment: "Newer Windows 11 versions aggressively park CPU cores and
often place all our threads onto the first 4 cores which results in
terrible performance with n_threads > 4." `NORMAL` is a no-op on all
platforms. Failure (e.g., `EPERM` on Linux without `CAP_SYS_NICE`)
logs a warning; the caller does not check the return value.

### 8.9 Abort mechanism

`abort` is `atomic_int` initialized to `-1` (sentinel meaning "never
abort"). The worker loop condition:
`node_n < n_nodes && abort != node_n`. When thread 0's abort
callback fires (line 3109), it stores `abort = node_n + 1` (relaxed)
and sets `ec = ABORTED`. On the next iteration, the condition
`abort (N+1) != node_n (N+1)` is false → loop exits. All threads
exit on the next node boundary after the per-node barrier.

The `memory_order_relaxed` on both store and load is safe **only
because** the per-node barrier's `seq_cst` fence publishes thread 0's
store to all other threads before they read it. If the barrier were
ever removed (e.g., a "no-barrier mode" as suggested in ARTX01-R3),
the abort becomes racy. This dependency is undocumented.

### 8.10 `ggml_critical_section_*`

`ggml-threading.cpp` is 13 lines: a single `std::mutex` and two
functions — **not** the threadpool mutex, but a process-wide
critical section used only by `ggml_cpu_init` (line 3826) for
one-time init. Not on the hot path. The public API exposes only
`start` / `end`; the threadpool, barrier, and chunk ops are accessed
through opaque `ggml_threadpool*` pointers.

### 8.11 Per-op parallelism

Each op picks its own parallelism scheme inside `compute_forward_<op>`
(see ARTX01 §8.4). `ggml_get_n_tasks` (`ggml-cpu.c:2220`) is a
per-op table consulted only by `ggml_graph_plan` for work-buffer
sizing. A few ops deliberately set `n_tasks = 1` even when they
could parallelize: `GET_ROWS` / `SET_ROWS` (line 2336) has a FIXME:
"get_rows can use additional threads, but the cost of launching
additional threads decreases performance with GPU offloading."

---

## 9. SIMD / GPU Strategy

The only ISA-specific code is `ggml_thread_cpu_relax`
(`ggml-cpu.c:519`): `yield` on aarch64, `_mm_pause` on x86_64,
`pause` (or raw `0x100000F` encoding if `__riscv_zihintpause` is
unavailable) on riscv, no-op elsewhere. No GPU involvement in the
CPU threading layer.

---

## 10. Quantization Strategy

N/A. The threading layer is dtype-agnostic. The only interaction
with quantization is that `current_chunk` is shared across threads
during matmul chunk stealing — its semantics (chunk index) are
independent of quant format. See ARTX06.

---

## 11. Correctness Analysis

### 11.1 Memory ordering

Four patterns: (1) `seq_cst` for barrier entry/exit (strongest,
total happens-before edge); (2) `seq_cst` store for `n_graph` in
kickoff, `relaxed` loads in workers (the `seq_cst` store guarantees
relaxed loads eventually observe it — comment at line 3252); (3)
`relaxed` for `abort` store/load (safe only via barrier fence — see
8.9); (4) `relaxed` for `current_chunk` in matmul chunk stealing
(correct because `fetch_add` returns unique values, so chunks are
disjoint by construction).

### 11.2 Pause/resume race

Both called under the mutex. The worker's pause check at line 3212
reads `pause` without the mutex — a race-free fast-path check
(`pause` is `atomic_bool`, relaxed load well-defined). If false,
proceeds to work check; if true, enters the mutex-protected inner
loop. Double-check pattern avoids taking the mutex on every outer
iteration.

### 11.3 Abort ordering

As discussed in 8.9, the relaxed abort store/load pair is correct
only because of the barrier's `seq_cst` fence. If a future change
skips the barrier for any node, the abort becomes racy. The code
has no comment warning of this dependency.

### 11.4 NUMA strategy fallthrough

`set_numa_thread_affinity`'s switch has no `case` for
`GGML_NUMA_STRATEGY_MIRROR`. Callers passing `MIRROR` get no
affinity applied silently — no warning. Correctness gap.

### 11.5 Windows affinity >64 CPUs

`ggml_thread_apply_affinity` on Windows (line 2497) builds a 64-bit
bitmask from the first 64 `mask[]` entries. For bits set beyond
index 63, logs a warning and breaks. Workers requesting cores >63
are silently pinned to whatever bits *are* set in the first 64 —
possibly an empty mask, in which case `SetThreadAffinityMask` gets
`0`, returns `0`, and the function reports failure but the caller
does not act on it.

### 11.6 `pthread_mutex_lock_shared` is exclusive

On pthreads, `ggml_mutex_lock_shared` is `#define`d to
`pthread_mutex_lock` (line 456) — not a read lock. The "shared"
name is meaningful only on Windows (`AcquireSRWLockShared` +
`CONDITION_VARIABLE_LOCKMODE_SHARED`). On pthreads, multiple workers
blocking on `cond_wait` serialize on the mutex. Not a correctness
bug — a performance gap.

### 11.7 `set_numa_thread_affinity` called per graph

Called inside `ggml_graph_compute_thread` (line 3070) on every graph
invocation, not just at thread start. For `NUMACTL`, re-applies the
same mask every time — wasted `pthread_setaffinity_np` syscall
(~1-2µs). The affinity should be set once at thread start.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                          | Where                                        | Notes                                                              |
| ------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| Persistent threadpool                 | `ggml-cpu.c:3273`                            | Workers reused across graphs; avoids `pthread_create` per graph.   |
| Hybrid spin-then-block polling        | `ggml-cpu.c:3167-3199`                       | Tunable via `poll`; default 50 (~300ms spin).                      |
| Cache-aligned hot atomics             | `ggml-cpu.c:489-491`                         | `n_barrier`, `n_barrier_passed`, `current_chunk` each in own line. |
| Packed `n_graph` counter              | `ggml-cpu.c:205, 3246-3253`                  | One atomic store publishes work + thread-count.                    |
| Barrier no-op fast path               | `ggml-cpu.c:575-579`                         | `n_threads == 1` → immediate return.                               |
| NUMA-aware chunking fallback          | `ggml-cpu.c:1413, 1667`                      | One chunk per thread on NUMA; preserves locality.                  |
| Per-thread CPU affinity + priority    | `ggml-cpu.c:2614-2666`                       | `pthread_setaffinity_np` + `pthread_setschedparam` per worker.     |
| Strict CPU placement                  | `ggml-cpu.c:2689-2709`                       | Round-robin single-bit mask per worker.                            |
| Windows 11 throttling disable         | `ggml-cpu.c:2548-2559`                       | `SetThreadInformation(ThreadPowerThrottling)` for `prio != LOW`.   |
| KMP_BLOCKTIME tuning (OpenMP)         | `ggml-cpu.c:3866-3874`                       | 200ms block time to approximate custom polling.                    |
| Per-thread scratch cache-line padding | `ops.cpp:297, 618, 964`                      | `(ne0 + CACHE_LINE_SIZE_F32) * ith` offsets.                       |
| Abort sentinel with relaxed ordering  | `ggml-cpu.c:3088, 3111`                      | Cheap atomic read; correctness via barrier fence.                  |
| Mutex-free polling fast path          | `ggml-cpu.c:3174-3177`                       | Polls `n_graph` relaxed; no mutex contention during spin.          |

### 12.2 Optimizations *not* present

* **No sense-reversing barrier.** The reset-to-0 write causes a
  cache-line ownership transfer on every barrier.
* **No NUMA-aware chunk stealing.** The fallback is
  one-chunk-per-thread; no "steal only from same-node threads."
* **No threadpool resize.** `n_threads` is fixed at creation; the
  packed `n_graph` allows per-graph active-thread reduction but
  never growth.
* **No work-stealing across pools.** Each pool is independent.
* **No direct `futex`/`WaitOnAddress` use.** `pthread_cond_wait`
  goes through `futex`; a direct call would save mutex overhead.
* **No affinity inheritance for child ops.** No mechanism for an op
  to request a different mask (e.g., to prefer a specific NUMA node).

### 12.3 Polling cost model

`n_rounds = 131072 * poll`. With `_mm_pause` ≈ 140 cycles at 3GHz:
`poll=0` → 0 (immediate block); `poll=1` → ~6ms; `poll=50` →
~305ms; `poll=100` → ~610ms. The poll exits early if `thread_ready`
returns true; `n_rounds` is a worst-case bound. Actual `_mm_pause`
latency varies by microarchitecture.

### 12.4 Barrier cost model

Per-thread cost dominated by cache-line transfers: 16
`atomic_fetch_add` on `n_barrier` (~70ns each) + 1 reset + 1
`n_barrier_passed` increment + 15 spin re-fetches (~70ns each) ≈
~3-4µs per barrier on 16 threads, scaling linearly. For a 1000-node
graph, ~3-4ms of barrier overhead — non-trivial for low-latency
decode. Deepens ARTX01-U5.

---

## 13. Architectural Strengths

1. **Persistent threadpool with pause/resume.** Workers survive
   across graphs; the pool can be paused (workers sleep) and resumed
   without teardown. Essential for interactive inference.
2. **Packed `n_graph` counter.** Publishing work-availability and
   active-thread-count in one atomic store is elegant and efficient.
   Enables per-graph thread-count variation without pool recreation.
3. **Hybrid poll-then-block with tunable `poll`.** The 0..100 range
   covers power-vs-latency trade-offs. Default 50 is reasonable.
4. **Cache-aligned hot atomics.** The three highest-traffic atomics
   each in their own cache line. Correct false-sharing mitigation.
5. **NUMA topology detection + three strategies.** `DISTRIBUTE` /
   `ISOLATE` / `NUMACTL` cover common cases. `NUMACTL` defers to
   OS-scheduler pinning — nice for HPC environments.
6. **Measured NUMA chunking fallback.** Documented with PR #6915
   reference. Evidence-based engineering. Applies to both `mul_mat`
   and `mul_mat_id`.
7. **Abort sentinel mechanism.** `-1` sentinel is clean; node
   indices are ≥0 so `abort != node_n` is always true when abort is
   -1. Relaxed ordering is a calculated bet on the barrier fence.
8. **Per-thread priority + affinity.** Workers pinned to specific
   cores with specific priorities. Windows 11 throttling workaround
   shows attention to platform quirks.
9. **Strict CPU placement option.** Deterministic per-core placement
   for benchmarking and cache-sensitive workloads.
10. **`n_threads == 1` barrier fast path.** Eliminates all
    synchronization overhead for sequential workloads.

---

## 14. Architectural Weaknesses

### W1 — Central-counter barrier (deepens ARTX01-F02)

`ggml-cpu.c:575-611`. The barrier resets `n_barrier` to 0 on every
generation. A sense-reversing barrier (alternate between 0 and
`n_threads-1`) would avoid this write. Estimated ~3-4µs per barrier
on 16 threads, scaling linearly. Confirms ARTX01-U5 for ≥32-core
scaling concerns.

### W2 — `cpumask[GGML_MAX_N_THREADS]` bool array (deepens ARTX01-F10)

`ggml-cpu.c:513`. 512 bytes per worker; a bitmap would be 64 bytes.
`workers[]` array is ~33 KB for 64 workers, spanning ~512 cache
lines. `ggml_thread_cpumask_next` and `is_valid` scan linearly.

### W3 — NUMA strategy MIRROR declared but unimplemented

`ggml-cpu.h:33` declares `MIRROR = 4`; `ggml-cpu.c:2157-2175`
switch has no case for it. Silent correctness gap — users requesting
`MIRROR` get no affinity.

### W4 — Windows affinity limited to 64 CPUs

`ggml-cpu.c:2497-2528`. Single `uint64_t` bitmask; entries beyond
63 warned-and-ignored. Windows API supports `GROUP_AFFINITY` for
>64 CPUs but it is not used. Critical for dual-socket Xeon / 
Threadripper Pro.

### W5 — `memory_order_relaxed` on abort relies on barrier fence

`ggml-cpu.c:3088, 3111`. Correct only because the per-node barrier's
`seq_cst` fence publishes the store. If a "no-barrier mode" is ever
added (ARTX01-R3), the abort becomes racy. No code comment warns of
this dependency.

### W6 — `set_numa_thread_affinity` called per graph

`ggml-cpu.c:3070`. Called inside `ggml_graph_compute_thread` on
every graph. For `NUMACTL`, re-applies the same mask every time —
wasted ~1-2µs syscall. Should be set once at thread start.

### W7 — `clear_numa_thread_affinity` only on main thread

`ggml-cpu.c:3416`. Only worker 0's affinity is cleared after
compute. Workers 1..n-1 retain their pinning. Asymmetry worth noting.

### W8 — `ggml_mutex_lock_shared` is exclusive on pthreads

`ggml-cpu.c:456`. `#define ggml_mutex_lock_shared(m)
pthread_mutex_lock(m)`. Multiple workers blocking on `cond_wait`
serialize on the mutex. Windows gets real shared locks; pthreads
does not.

### W9 — OpenMP path loses polling, priority, pause, abort

`ggml-cpu.c:2765, 2777, 3378-3401`. `pause`/`resume` are
`UNUSED(threadpool)`; no polling or abort between nodes. Users
compiling with `GGML_USE_OPENMP` get a fundamentally weaker model.
Documentation does not make this explicit.

### W10 — Hardcoded `GGML_CACHE_LINE = 64`

`ggml-cpu.c:60`. On POWER9 (128-byte lines) or s390x VXE2 (256-byte
lines), `GGML_CACHE_ALIGN` on hot atomics is insufficient — adjacent
atomics can share a line. The separate `CACHE_LINE_SIZE` in `ops.h`
*does* account for these ISAs but is not used in `ggml-cpu.c`.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR | What | Reasoning |
| --------------- | -------------------------------- | ---- | --------- |
| `glproc`        | **ADOPT** | Persistent threadpool with pause/resume | Essential for interactive inference. |
| `glproc`        | **ADOPT** | Packed `n_graph` counter | Clean publish of work + thread-count. |
| `glproc`        | **ADOPT** | Cache-aligned hot atomics | Correct false-sharing mitigation. |
| `glproc`        | **ADOPT** | Hybrid poll-then-block with tunable `poll` | Covers power-vs-latency spectrum. |
| `glproc`        | **ADOPT** | Abort sentinel `-1` with `!= node_n` check | Clean, cheap, correct (caveat W5). |
| `glproc`        | **ADAPT** | Central-counter barrier | Keep, but consider sense-reversing. |
| `glproc`        | **ADAPT** | NUMA strategies | Keep DISTRIBUTE/ISOLATE/NUMACTL; implement MIRROR or remove. |
| `glproc`        | **ADAPT** | NUMA chunking fallback | Keep, but make policy pluggable. |
| `glproc`        | **ADAPT** | `strict_cpu` round-robin placement | Keep, but use a bitmap. |
| `glproc`        | **ADAPT** | Thread priority mapping | Keep, but check return value. |
| `glproc`        | **REJECT**| `bool[GGML_MAX_N_THREADS]` cpumask | Use a 64-byte bitmap. |
| `glproc`        | **REJECT**| Windows ≤64-CPU affinity limit | Use `GROUP_AFFINITY`. |
| `glproc`        | **REJECT**| Per-graph `set_numa_thread_affinity` | Set once at thread start. |
| `glproc`        | **REJECT**| `ggml_mutex_lock_shared` as exclusive | Use a real read-write lock on pthreads. |
| `glproc`        | **MONITOR**| OpenMP fallback | Keep for HPC; document trade-offs. |
| `glproc`        | **MONITOR**| TSAN dummy RMW workaround | Watch for TSAN upstream fence support. |
| `GATE`          | **ADOPT** | Per-graph thread-count variation via `n_graph` | Dynamic resource allocation per graph. |
| `GATE`          | **ADOPT** | Abort callback mechanism | Clean way to interrupt long graphs. |
| `GATE`          | **ADAPT** | Per-node barrier | Keep, but add no-barrier mode (with abort ordering comment). |
| `GATE`          | **REJECT**| OpenMP as primary backend | Persistent pthread pool is strictly more capable. |

---

## 16. Recommendations

### R1 — ADOPT persistent threadpool with pause/resume
**Priority:** Critical | **Difficulty:** M | **Dependencies:** none
Workers created once, reused across graphs, pausable/resumable. Add optional idle-timeout for memory-constrained environments.

### R2 — ADOPT packed `n_graph` counter
**Priority:** High | **Difficulty:** S | **Dependencies:** R1
Publish work + active-thread-count in one 32-bit atomic. Store `seq_cst` from kickoff; load `relaxed` in workers. Allows per-graph thread-count variation.

### R3 — ADAPT central-counter barrier to sense-reversing
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
Sense-reversing (alternate 0 / `n_threads-1`) avoids the per-generation reset write. Saves one cache-line transfer per barrier.

### R4 — ADOPT hybrid poll-then-block with tunable `poll`
**Priority:** High | **Difficulty:** S | **Dependencies:** R1
Spin `K * poll` rounds before `cond_wait`. Default `poll=50`. Consider adaptive mode based on inter-graph arrival time.

### R5 — ADOPT cache-aligned hot atomics
**Priority:** High | **Difficulty:** XS | **Dependencies:** none
Use `alignas(std::hardware_destructive_interference_size)`, not hardcoded 64.

### R6 — REJECT `bool[N]` cpumask; use bitmap
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
`uint64_t cpumask[8]` (512 bits / 64 bytes). Scan with `__builtin_ctzll`. Saves ~448 bytes per worker.

### R7 — ADAPT NUMA: implement MIRROR or remove the enum
**Priority:** Medium | **Difficulty:** XS | **Dependencies:** none
Do not leave a declared-but-unimplemented strategy.

### R8 — ADOPT abort sentinel mechanism
**Priority:** High | **Difficulty:** S | **Dependencies:** R1, R3
`atomic_int abort = -1`; workers check `abort != node_n`; thread 0 sets `abort = node_n+1`. **Document the barrier dependency.** If no-barrier mode is added, switch to `seq_cst`.

### R9 — REJECT per-graph `set_numa_thread_affinity`
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
Set NUMA affinity once at thread start. Provide a separate API for runtime strategy change.

### R10 — REJECT Windows ≤64-CPU affinity limit
**Priority:** Medium | **Difficulty:** M | **Dependencies:** R6
Use `GROUP_AFFINITY` + `SetThreadGroupAffinity` for >64 CPUs. Fall back to `SetThreadAffinityMask` for ≤64.

### R11 — ADOPT thread priority + Windows throttling workaround
**Priority:** Low | **Difficulty:** S | **Dependencies:** R1
Keep per-platform mapping. Adopt Windows 11 `ThreadPowerThrottling` disable for `prio != LOW`. Propagate failure to caller.

### R12 — MONITOR OpenMP fallback
**Priority:** Low | **Difficulty:** XS | **Dependencies:** R1
Keep as fallback for HPC. Document that it loses polling, priority, pause, abort. Watch for OpenMP 5.x features.

### R13 — ADOPT per-graph thread-count variation
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R2
Allow per-`ggml_graph_compute` `n_threads`, clamped to pool size. Enables dynamic resource allocation (4 threads for decode, 16 for prefill).

---

## 17. Findings

### Finding ARTX07-F01

```
Finding ID:           ARTX07-F01
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Threadpool lifecycle
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_threadpool_new_impl, ggml_threadpool_pause/_resume/_free
Lines:                2711-2779, 3273-3344
Summary:              Persistent pthread threadpool; pause/resume keeps workers alive; free is the only teardown.
Observation:          new_impl spawns n_threads-1 secondary threads via pthread_create. The main thread (worker 0) is the caller's thread. Pause sets pause=true under mutex and cond_broadcasts; workers enter an inner while(pause) loop that cond_waits. Resume sets pause=false and broadcasts. Free sets stop=true, pause=false, broadcasts, joins workers 1..n-1, destroys mutex+cond, frees pool. Workers are kept alive across graphs unconditionally — no idle-timeout.
Evidence:             ggml-cpu.c:2711-2740 (free), 2742-2779 (pause/resume locked), 3273-3344 (new_impl), 3210-3236 (worker loop with pause check at 3212).
Architectural Impact: Eliminates pthread_create overhead per graph (~10-30µs/thread). Essential for interactive inference. The pause mechanism lets a pool stay resident but sleep when idle, trading memory for latency.
Correctness Impact:   None. Lifecycle correctly synchronized via mutex.
Optimization Type:    persistent threads
GwenLand Target:      glproc
Recommendation:       ADOPT. Add an optional idle-timeout for memory-constrained environments.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX07-F02

```
Finding ID:           ARTX07-F02
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Worker spin loop
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_graph_compute_poll_for_work, ggml_graph_compute_check_for_work
Lines:                3167-3199
Summary:              Hybrid poll-then-block: workers spin 1024*128*poll rounds of cpu_relax before cond_wait.
Observation:          n_rounds = 1024UL * 128 * poll. With default poll=50 (ggml.c:8005), ~6.5M rounds of _mm_pause (~300ms at 3GHz). Poll exits early if thread_ready returns true. After exhausting poll, worker takes mutex shared and cond_waits. Comment at 3170: "This seems to make 0...100 a decent range for polling level across modern processors."
Evidence:             ggml-cpu.c:3170-3180 (poll_for_work), 3182-3199 (check_for_work), ggml.c:8005 (poll=50 default).
Architectural Impact: Tunable latency-vs-power trade-off. poll=0 = pure block (min power, max latency); poll=100 = aggressive spin. The 300ms default spin is long by most standards but fits inference where graphs arrive frequently.
Correctness Impact:   None. Poll is best-effort; cond_wait fallback ensures correctness.
Optimization Type:    asynchronous execution
GwenLand Target:      glproc
Recommendation:       ADOPT. Expose poll as per-pool parameter. Consider adaptive mode based on inter-graph arrival time.
Priority:             High
Difficulty:           S
Dependencies:         ARTX07-F01
Confidence:           High
```

### Finding ARTX07-F03

```
Finding ID:           ARTX07-F03
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Packed n_graph counter
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_graph_compute_kickoff, ggml_barrier, ggml_graph_compute_thread
Lines:                205-206, 576, 3075, 3145-3150, 3246-3253
Summary:              Single atomic_int n_graph packs active thread count (low 16) and generation counter (upper 16).
Observation:          GGML_THREADPOOL_N_THREADS_MASK = 0xffff. Kickoff reads n_graph relaxed, extracts generation (upper 16), increments, re-packs with new n_threads, stores seq_cst. Workers read n_graph relaxed: barrier uses low 16 as wait-for count; compute_thread uses low 16 as nth; thread_ready compares full 32 bits to last_graph. One atomic store publishes both "new work" and "how many threads participate."
Evidence:             ggml-cpu.c:205-206 (mask/bits), 576 (barrier reads low 16), 3075 (compute_thread reads low 16), 3145-3150 (thread_ready), 3246-3253 (kickoff packs and stores).
Architectural Impact: Allows per-graph thread-count variation without pool recreation. A 64-thread pool can run a graph with 8 active threads — workers ith >= 8 stay parked. Reduces publish cost from two atomics to one.
Correctness Impact:   None. 16-bit thread count limits pool to 65535 threads, well above GGML_MAX_N_THREADS=512.
Optimization Type:    asynchronous execution
GwenLand Target:      both
Recommendation:       ADOPT. Document the bit layout in the API contract.
Priority:             High
Difficulty:           S
Dependencies:         ARTX07-F01
Confidence:           High
```

### Finding ARTX07-F04

```
Finding ID:           ARTX07-F04
Category:             EXECUTION_GRAPH
Engine:               CPU
Component:            Central-counter barrier
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_barrier
Lines:                575-611
Summary:              Central-counter barrier with seq_cst entry/exit; last arrival resets counter to 0 and bumps n_barrier_passed.
Observation:          Deepens ARTX01-F02. Not sense-reversing — counter reset to 0 every generation. Last arrival does reset (relaxed) and bumps n_barrier_passed (seq_cst). Others spin on n_barrier_passed (relaxed loop with cpu_relax). After spin exits, a final seq_cst fence (or TSAN dummy RMW) ensures acquire. n_threads==1 is no-op fast path; OpenMP uses #pragma omp barrier. The two hot atomics are each GGML_CACHE_ALIGN, so reset of n_barrier does not perturb the n_barrier_passed line spinners read.
Evidence:             ggml-cpu.c:575-611 (barrier body), 489-491 (cache align), 205 (n_threads mask).
Architectural Impact: The reset write causes a cache-line ownership transfer every barrier — avoidable with sense-reversing. Estimated ~3-4µs per barrier on 16 threads, scaling linearly. Confirms ARTX01-U5.
Correctness Impact:   None. seq_cst fences provide correct happens-before.
Optimization Type:    None
GwenLand Target:      GATE
Recommendation:       ADAPT. Keep central-counter design, switch to sense-reversing. Document W5 caveat: relaxed abort depends on this barrier's seq_cst fence.
Priority:             High
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX07-F05

```
Finding ID:           ARTX07-F05
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            Cache-line alignment of hot atomics
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             struct ggml_threadpool
Lines:                60, 489-491
Summary:              n_barrier, n_barrier_passed, current_chunk are GGML_CACHE_ALIGN (64B); GGML_CACHE_LINE hardcoded to 64.
Observation:          Deepens ARTX01-F10 and W5. Three hot atomics each in own 64-byte cache line, preventing false sharing between barrier counter, barrier sense, and matmul chunk counter. But GGML_CACHE_LINE is hardcoded to 64 (line 60), with comment noting intent to switch to std::hardware_destructive_interference_size. The separate CACHE_LINE_SIZE in ops.h:9-19 already does the right thing per-ISA (128 on POWER9, 256 on VXE2).
Evidence:             ggml-cpu.c:60 (hardcoded 64), 62-63 (GGML_CACHE_ALIGN), 489-491 (aligned atomics), ops.h:9-19 (CACHE_LINE_SIZE per-ISA).
Architectural Impact: On x86/ARM, 64-byte alignment is correct. On POWER9 (128-byte lines), two adjacent 64-byte-aligned atomics could share a cache line — false-sharing mitigation fails silently.
Correctness Impact:   None. Alignment is a performance hint.
Optimization Type:    None
GwenLand Target:      glproc
Recommendation:       ADOPT in glproc, but use std::hardware_destructive_interference_size (C++17). If unavailable, replicate the per-ISA switch from ops.h.
Priority:             Medium
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX07-F06

```
Finding ID:           ARTX07-F06
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            NUMA strategies
Source File:          ggml/src/ggml-cpu/ggml-cpu.c, ggml/include/ggml-cpu.h
Function:             ggml_numa_init, set_numa_thread_affinity, clear_numa_thread_affinity
Lines:                ggml-cpu.h:28-35; ggml-cpu.c:636-722, 2148-2218
Summary:              Three implemented NUMA strategies (DISTRIBUTE, ISOLATE, NUMACTL) plus one declared-but-unimplemented (MIRROR).
Observation:          ggml_numa_init reads /sys/devices/system/node/nodeN and /sys/devices/system/cpu/cpuN to enumerate topology, captures current cpuset via pthread_getaffinity_np, records current node via getcpu(). set_numa_thread_affinity switches: DISTRIBUTE picks node = ith % n_nodes; ISOLATE picks current_node; NUMACTL re-applies captured cpuset. MIRROR (enum 4) has no case — falls through to default (no-op).
Evidence:             ggml-cpu.h:28-35 (enum), ggml-cpu.c:636-722 (init), 2157-2175 (strategy switch, no MIRROR case), 2193-2212 (clear).
Architectural Impact: DISTRIBUTE is default for multi-socket; ISOLATE keeps threads on main thread's node; NUMACTL defers to OS. MIRROR gap is a silent correctness issue.
Correctness Impact:   MIRROR is silently ignored. Other strategies correct.
Optimization Type:    None
GwenLand Target:      glproc
Recommendation:       ADAPT. Implement MIRROR or remove the enum value. Make strategy pluggable for NUMA-aware chunk stealing experiments.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX07-F07
Confidence:           High
```

### Finding ARTX07-F07

```
Finding ID:           ARTX07-F07
Category:             THREADING_MISMATCH
Engine:               CPU
Component:            NUMA-aware chunking fallback
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_compute_forward_mul_mat, ggml_compute_forward_mul_mat_id
Lines:                1410-1417, 1666-1675
Summary:              On NUMA, matmul and mul_mat_id collapse to one-chunk-per-thread, disabling dynamic stealing.
Observation:          Deepens ARTX01-F09. Both mul_mat (line 1413) and mul_mat_id (line 1667) check ggml_is_numa() and collapse to nchunk0 = nth, nchunk1 = 1 (or vice versa). Comment at 1411 cites PR #6915: "In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that." Trades load balancing for memory locality.
Evidence:             ggml-cpu.c:1410-1417 (mul_mat), 1666-1675 (mul_mat_id with disable_chunking = ggml_is_numa()).
Architectural Impact: Better locality on multi-socket. Worse balancing if chunks have variable cost. Binary fallback — no middle ground (e.g., "steal only from same-node threads").
Correctness Impact:   None. Chunks still disjoint; only assignment policy changes.
Optimization Type:    None
GwenLand Target:      glproc
Recommendation:       ADAPT. Keep as default, but make policy pluggable. A NUMA-aware chunk stealer that prefers same-node chunks could recover balancing.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX07-F06
Confidence:           High
```

### Finding ARTX07-F08

```
Finding ID:           ARTX07-F08
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            CPU affinity and strict_cpu placement
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_thread_cpumask_next, ggml_thread_apply_affinity, ggml_thread_cpumask_is_valid
Lines:                2493-2709
Summary:              Per-worker cpumask from global mask via strict_cpu round-robin or full copy; applied via pthread_setaffinity_np / SetThreadAffinityMask.
Observation:          cpumask_next: if !strict, memcpy global mask to every worker (OS picks core); if strict, scan from *iter for next set bit, set only that bit, advance *iter. Modulo via subtraction (line 2700, "cheaper modulo"). apply_affinity: Linux/Android uses sched_setaffinity or pthread_setaffinity_np; Windows uses SetThreadAffinityMask with 64-bit bitmask; macOS is no-op. Main thread (worker 0) placed last, towards higher-numbered CPUs (comment at 3321).
Evidence:             ggml-cpu.c:2493-2529 (Windows), 2614-2641 (Linux), 2579-2583 (macOS), 2682-2687 (is_valid), 2689-2709 (cpumask_next), 3320-3340 (placement order).
Architectural Impact: strict_cpu is the only way to get deterministic per-core placement. Without it, OS may bounce workers between cores, defeating L1/L2 warmth. "Main thread placed last" keeps orchestration on higher-numbered cores.
Correctness Impact:   None. Affinity is a hint.
Optimization Type:    None
GwenLand Target:      glproc
Recommendation:       ADAPT. Keep strict_cpu and round-robin, but use a bitmap (see F10). Check return value of pthread_setaffinity_np.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX07-F10
Confidence:           High
```

### Finding ARTX07-F09

```
Finding ID:           ARTX07-F09
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Thread priority
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_thread_apply_priority
Lines:                2531-2573 (Windows), 2585-2609 (macOS), 2643-2666 (Linux)
Summary:              Per-platform priority mapping; Windows 11 throttling disabled for prio != LOW; NORMAL is no-op.
Observation:          prio is int32_t holding ggml_sched_priority. Linux: LOW → SCHED_BATCH, NORMAL → SCHED_OTHER (no-op return), MEDIUM/HIGH/REALTIME → SCHED_FIFO priorities 40/80/90. macOS: same but no SCHED_BATCH. Windows: LOW..REALTIME map to THREAD_PRIORITY_BELOW_NORMAL..TIME_CRITICAL; for prio != LOW, SetThreadInformation(ThreadPowerThrottling, StateMask=0) disables Windows 11 core parking (comment 2543-2547: "Newer Windows 11 versions aggressively park CPU cores and often place all our threads onto the first 4 cores"). NORMAL is no-op on all platforms. Failure (EPERM without CAP_SYS_NICE) logs warning; caller does not check return.
Evidence:             ggml-cpu.c:2531-2573 (Windows), 2585-2609 (macOS), 2643-2666 (Linux), 3205 (apply at worker start), 3257 (apply on main thread resume).
Architectural Impact: SCHED_FIFO can dramatically improve latency by avoiding preemption, but requires elevated privileges. Windows throttling workaround is critical for n_threads > 4 on Windows 11.
Correctness Impact:   None. Priority is a hint; failure logged.
Optimization Type:    None
GwenLand Target:      glproc
Recommendation:       ADAPT. Keep per-platform mapping and Windows throttling workaround. Propagate failure to caller. Document privilege requirements for MEDIUM+.
Priority:             Low
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX07-F10

```
Finding ID:           ARTX07-F10
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            Per-worker cpumask storage
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             struct ggml_compute_state, ggml_thread_cpumask_next, ggml_thread_apply_affinity
Lines:                513, 2614-2641, 2682-2709, 3314, 3326, 3332
Summary:              cpumask is bool[GGML_MAX_N_THREADS] = 512 bytes per worker; should be 64-byte bitmap.
Observation:          Deepens ARTX01-F10. GGML_MAX_N_THREADS=512 (ggml.h:225). Each ggml_compute_state has 512-byte bool array. With 64 workers, workers[] is ~33 KB, spanning ~512 cache lines. is_valid scans all 512 linearly (2683-2686). cpumask_next scans linearly with wraparound (2696-2707). apply_affinity on Linux scans all 512 to build cpu_set_t (2620-2625). On Windows, packs first 64 entries into uint64_t (2503-2515) — effectively the bitmap already used internally. params_match memcmp's 512 bytes (ggml.c:8022).
Evidence:             ggml.h:225, ggml-cpu.c:513, 2503-2515 (Windows packs to u64), 2620-2625 (Linux scans), 2683-2686 (is_valid scans), 2696-2707 (cpumask_next scans), ggml.c:8022 (match memcmp).
Architectural Impact: 8x memory overhead vs a 64-byte bitmap. Linear scans where ctzll/popcnt would be O(1). Windows path already uses bitmap internally — bool[] converted to u64, throwing away >64-CPU entries.
Correctness Impact:   None.
Optimization Type:    None
GwenLand Target:      glproc
Recommendation:       REJECT the bool[] layout. Use uint64_t cpumask[8] in glproc. Scan with __builtin_ctzll. Saves 448 bytes per worker; O(1) per set bit.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX07-F08
Confidence:           High
```

### Finding ARTX07-F11

```
Finding ID:           ARTX07-F11
Category:             CORRECTNESS_SHORTCUT
Engine:               CPU
Component:            Abort mechanism
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_graph_compute_thread
Lines:                3088, 3109-3113, 3289, 3374
Summary:              atomic_int abort = -1 sentinel; thread 0 sets abort = node_n+1 (relaxed); workers check abort != node_n (relaxed).
Observation:          abort initialized to -1 (sentinel "never abort") at pool creation (3289) and graph reuse (3374). Worker loop: node_n < n_nodes && abort != node_n. When thread 0's abort_callback fires after node N, stores abort = N+1 (relaxed), sets ec = ABORTED. Next iteration: abort (N+1) != node_n (N+1) is false → loop exits. All threads exit on next node boundary after the per-node barrier. The relaxed ordering is safe ONLY because the barrier's seq_cst fence publishes thread 0's store to all other threads before they read it. If the barrier were ever removed, the abort becomes racy.
Evidence:             ggml-cpu.c:3088 (loop condition, relaxed load), 3109-3113 (thread 0 stores abort = node_n+1 relaxed), 3289 (init -1 in new_impl), 3374 (reset -1 in graph_compute reuse).
Architectural Impact: Clean, cheap abort. Relaxed load is ~1 cycle vs ~10+ for seq_cst. Sentinel -1 avoids a separate "abort_enabled" flag. The barrier dependency is implicit and undocumented — a maintainability hazard.
Correctness Impact:   Correct given the current barrier. Fragile: any change to the barrier (removal, reordering, weakening) breaks abort silently.
Optimization Type:    asynchronous execution
GwenLand Target:      both
Recommendation:       ADOPT the sentinel + relaxed ordering, but add a code comment at the abort store documenting the barrier dependency. In glproc's GATE, if a no-barrier mode is added, switch abort store/load to seq_cst.
Priority:             High
Difficulty:           S
Dependencies:         ARTX07-F04
Confidence:           High
```

### Finding ARTX07-F12

```
Finding ID:           ARTX07-F12
Category:             THREADING_MISMATCH
Engine:               CPU
Component:            Windows CPU affinity limit
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_thread_apply_affinity (Windows path)
Lines:                2493-2529
Summary:              Windows affinity uses single uint64_t bitmask; cores beyond index 63 are warned-and-ignored.
Observation:          Windows path builds uint64_t bitmask from first 64 mask[] entries (2503-2515). For entries 64..511, logs "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!" and breaks (2517-2522). SetThreadAffinityMask called with 64-bit mask. If user's mask has bits set only above 63, bitmask is 0, SetThreadAffinityMask returns 0, function returns false — caller does not check. Windows API for >64 CPUs is SetThreadGroupAffinity with GROUP_AFFINITY struct, not used. Comment at 2496: "TODO: support > 64 CPUs."
Evidence:             ggml-cpu.c:2496 (TODO), 2503-2515 (build 64-bit mask), 2517-2522 (warn-and-break), 2524-2528 (SetThreadAffinityMask).
Architectural Impact: On Windows servers with >64 logical CPUs (dual-socket Xeon, Threadripper Pro), workers cannot be pinned to cores >63. strict_cpu placement silently degrades to "first 64 cores only."
Correctness Impact:   None for ≤64-CPU systems. For >64-CPU systems, affinity is silently wrong.
Optimization Type:    None
GwenLand Target:      glproc
Recommendation:       REJECT. Use SetThreadGroupAffinity with GROUP_AFFINITY for >64 CPUs. Fall back to SetThreadAffinityMask for ≤64. Necessary for modern high-core-count Windows servers.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX07-F10
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Actual latency of the spin-then-block transition at `poll=50`. Static estimate ~300ms; actual `_mm_pause` latency varies by microarchitecture (Intel Alder Lake+ has longer `pause` than Skylake). Requires runtime measurement.
* **U2**. Whether the central-counter barrier scales acceptably beyond 32 threads. ARTX01-U5 flagged this; this audit estimates ~3-4µs per barrier on 16 threads but the scaling curve to 64+ threads is unknown. Requires profiling.
* **U3**. Whether `strict_cpu` round-robin produces measurably better performance than non-strict on modern Linux. The non-strict path defers to the OS, which may already do a good job. Requires A/B benchmarking.
* **U4**. Whether `SCHED_FIFO` priority actually improves inference latency given that it requires `CAP_SYS_NICE`. Most users run without elevated privileges, so the set fails silently. Requires benchmarking with/without root.
* **U5**. Whether `KMP_BLOCKTIME=200` approximates the custom path's polling. Custom path's `poll=50` spins ~300ms; KMP_BLOCKTIME=200 is 200ms with different semantics (per-parallel-region end, not per-graph-kickoff). Requires comparison.
* **U6**. Whether the abort mechanism's relaxed ordering is intentional or accidental. No code comment explains the barrier dependency. Requires git archaeology.
* **U7**. Whether `ggml_mutex_lock_shared` being exclusive on pthreads is a measurable bottleneck. Workers only take the mutex when falling through the poll, which is rare at `poll=50`. Requires mutex contention profiling.
* **U8**. Whether `MIRROR` NUMA strategy was ever implemented and removed, or never implemented. Requires git archaeology.

---

## 19. References

| Ref | File                                                | Function / Symbol                              | Lines         |
| --- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `GGML_CACHE_LINE`, `GGML_CACHE_ALIGN`, `GGML_TSAN_ENABLED` | 55-74 |
| R02 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `GGML_THREADPOOL_N_THREADS_MASK`, `_BITS`      | 205-206       |
| R03 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | mutex/cond macros (Win + pthread)              | 427-477       |
| R04 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `struct ggml_threadpool`                       | 480-504       |
| R05 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `struct ggml_compute_state`                    | 507-516       |
| R06 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_thread_cpu_relax`                        | 519-538       |
| R07 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | NUMA structs                                   | 544-563       |
| R08 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_barrier`, `ggml_threadpool_chunk_set/add` | 575-619     |
| R09 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_numa_init`, `ggml_is_numa`               | 636-726       |
| R10 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `set_numa_thread_affinity`, `clear_numa_...`   | 2148-2218     |
| R11 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_thread_apply_affinity/priority` (Win)    | 2493-2573     |
| R12 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_thread_apply_affinity/priority` (macOS)  | 2579-2609     |
| R13 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_thread_apply_affinity/priority` (Linux)  | 2614-2666     |
| R14 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_thread_cpumask_is_valid`, `_next`        | 2682-2709     |
| R15 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_threadpool_free`, `pause`, `resume`      | 2711-2779     |
| R16 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_plan`                              | 2781-3019     |
| R17 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_thread`                    | 3060-3133     |
| R18 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `thread_ready`, `_sync`, `poll_for_work`, `check_for_work` | 3139-3199 |
| R19 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_secondary_thread`          | 3201-3236     |
| R20 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute_kickoff`                   | 3239-3269     |
| R21 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_threadpool_new_impl`                     | 3273-3344     |
| R22 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_graph_compute`                           | 3350-3425     |
| R23 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_cpu_init` (KMP_BLOCKTIME)                | 3818-3895     |
| R24 | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `mul_mat` / `mul_mat_id` (NUMA chunking)       | 1410-1417, 1666-1675 |
| R25 | `ggml/src/ggml-threading.cpp`                       | `ggml_critical_section_start/end`              | 1-13          |
| R26 | `ggml/src/ggml-threading.h`                         | Public critical-section API                    | 1-15          |
| R27 | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `ggml_compute_params`, `ggml_barrier` decl     | 18-30, 531-535|
| R28 | `ggml/include/ggml.h`                               | `ggml_threadpool_params`, `ggml_sched_priority`| 225, 2900-2926|
| R29 | `ggml/include/ggml-cpu.h`                           | `ggml_numa_strategy` enum                      | 28-38         |
| R30 | `ggml/src/ggml.c`                                   | `ggml_threadpool_params_init/default/match`    | 8002-8023     |
| R31 | `ggml/src/ggml-cpu/ops.h`                           | `CACHE_LINE_SIZE` per-ISA constant             | 9-19          |
