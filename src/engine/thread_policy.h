// Compute-thread counts for decode and prefill.
//
// MEASURED (2026-08-21/22): this engine does not scale with threads and
// COLLAPSES when the process oversubscribes the cores. It runs ~9 threads of
// its own (IOCP completions, main, backend) and ggml's pool SPIN-waits, so once
// total threads exceed cores the spinners starve the workers: 22 requested on
// 22 cores measured 50-61 s/token against 2.3 at 4. Qwen3.6-35B-A3B fully
// resident: 4 threads 2.2 s/tok, 6 -> 2.3, 8 -> 2.3, 12 -> 3.4, 22 -> 62; the
// dense 27B has the same shape, so the fault is engine-wide.
//
// Decode is bandwidth-bound and flat from 4 threads upward; PREFILL is
// compute-bound over 512-token chunks and should use the machine. An explicit
// --threads sets both, because someone measuring wants one variable, not two.
#pragma once

namespace dray::engine {

struct ThreadChoice {
    int decode = 4;
    int prefill = 4;
    int ceiling = 1;       // cores minus the process's own threads
    int requested = 4;     // what was asked for (or the default)
    bool clamped = false;  // requested exceeded the ceiling
};

// requested: --threads (0 = default). hardware: std::thread::hardware_concurrency().
ThreadChoice choose_threads(int requested, int hardware);

// Threads this process runs besides ggml's pool; the oversubscription margin.
constexpr int kOwnThreads = 9;

}  // namespace dray::engine
