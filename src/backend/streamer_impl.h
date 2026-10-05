// Streamer::Impl -- the composition behind the public Streamer.
//
// PRIVATE to src/backend: included by stream_buffer.cpp (the coordinator) and
// stream_buffer_type.cpp (the ggml callbacks that reach it). Nothing outside
// the backend may include it; the engine sees only stream_buffer.h.
//
// It owns one of each component and decides, per graph node, which of them acts:
//
//   AccountedAlloc   the only allocator (Invariant 1)
//   PoisonBuffers    what unbacked and failed weights point at
//   IoScheduler      every disk read and the one registry of them
//   TensorRegistry   where every tensor lives on disk
//   ResidencyCache   what is in RAM, what may leave, static pinning
//   ExpertCompactor  MUL_MAT_ID compaction, GET_ROWS row slicing, Phases A and C
//   ExpertSlots      the frequency expert cache (with RoutingSkew)
//   UncondRing       Phase B, the unconditional stream read ahead
//   HitRates, PrivateIds

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "backend/accounted_alloc.h"
#include "backend/expert_compactor.h"
#include "backend/expert_slots.h"
#include "backend/hit_rates.h"
#include "backend/io_scheduler.h"
#include "backend/poison_buffers.h"
#include "backend/private_ids.h"
#include "backend/residency_cache.h"
#include "backend/routing_skew.h"
#include "backend/stream_buffer.h"
#include "backend/stream_flags.h"
#include "backend/tensor_registry.h"
#include "backend/uncond_ring.h"
#include "ggml-backend-impl.h"

namespace dray::backend {

// Members are declared in dependency order: each is constructed after, and
// destroyed before, everything it borrows. The allocator comes first so the
// arena outlives every block returned to it -- and, unlike the pre-split
// layout, the storage backend is now destroyed BEFORE the arena is unmapped,
// never after. The one back-pointer against this order (the scheduler's
// reclaimer, pointing at the later-declared cache) is cleared in ~Streamer.
struct Streamer::Impl {
    const plan::Plan& plan;
    Config            cfg;
    const StreamFlags flags;   // the env levers, read once at construction

    AccountedAlloc    mem;
    PoisonBuffers     poison;
    // Non-zero means at least one node computed against poison rather than real
    // weights. Atomic: read cross-thread by /health's engine_tainted while the
    // decode thread increments (swarm S4). Relaxed is enough: monotonic flag.
    std::atomic<uint64_t> failures{0};
    IoScheduler       io;
    TensorRegistry    tensors;
    ResidencyCache    cache;
    HitRates          hits;
    PrivateIds        ids;
    RoutingSkew       skew;
    ExpertSlots       slots;
    ExpertCompactor   compactor;
    UncondRing        ring;

    uint32_t pass_count = 0;   // forward passes seen (embedding lookups)
    uint64_t compacted = 0;    // MUL_MAT_ID/GET_ROWS nodes served compacted
    // Widest single batched union region (uniq(n_seq) x slot): the measured
    // admission floor. Certain death below 1x (B=38), proven clean at 2x (B=32).
    uint64_t batch_region_bound = 0;
    // Where does the streaming path spend its time? Accumulated nanoseconds
    // inside the two hot callbacks, reported under DRAY_IO_STATS. If these
    // dominate the wall clock, the gap is ours and fixable; if they do not, it
    // is elsewhere. (The I/O-side counters live in IoStats.)
    uint64_t ns_materialise = 0;
    // ggml's own compute of claimed nodes: from the end of materialise() to the start of
    // release() -- the node graph launched, run and joined by the CPU thread pool.
    uint64_t ns_node_compute = 0;
    std::chrono::steady_clock::time_point t_mat_end{};
    uint64_t ns_release = 0;
    uint64_t n_materialise = 0;
    // ns_materialise split by what the node was waiting for. Time in materialise is
    // time the compute thread is NOT computing, so this says which stream holds it.
    enum WaitClass { kWaitRouted, kWaitRows, kWaitWeights, kWaitOther, kWaitClasses };
    uint64_t ns_wait[kWaitClasses] = {};
    uint64_t n_wait[kWaitClasses] = {};
    // Where a whole weight comes from, timed (bring_in), plus the ring producer's
    // own CPU time, which runs inside both callbacks.
    enum Source_ { kFromCache, kFromRing, kFromDisk, kSources };
    uint64_t ns_from[kSources] = {};
    uint64_t n_from[kSources] = {};
    uint64_t bytes_from_disk = 0;
    uint64_t ns_produce = 0;

    // The ggml buffer type llama.cpp allocates weights from (stream_buffer_type.cpp).
    ggml_backend_buffer_type buft{};
    // What the buffer reports as its base: non-null and aligned for the
    // allocator's arithmetic, never dereferenced.
    void* fake_base = nullptr;
    // Tier 2: the ONE metal mapping over the pool arena. Metal refuses to
    // no-copy-wrap the same pages twice (nil buffer, and get_base crashes), so
    // the first alloc creates the master and every later alloc returns a view
    // sharing its context with free/clear neutered. Views outliving the master
    // is safe by construction: nothing executes after llama_model_free.
    ggml_backend_buffer_t metal_master = nullptr;

    Impl(mem::Accountant& a, const plan::Plan& p, Config c)
        : plan(p), cfg(c), flags(StreamFlags::from_env()),
          mem(a), poison(mem), io(mem, failures),
          cache(mem, poison, tensors),
          ids(flags.trace_compact),
          slots(mem, cache, skew, flags.no_eslots, cfg.admit_byte_fraction),
          compactor({mem, io, cache, tensors, ids, slots, skew, hits}, flags, cfg.n_seq, failures),
          ring({mem, io, cache, tensors, hits}, flags.no_retain) {
        if (!flags.no_pool) {
            // 2x cap: VA is free and exact-size reuse means fragmentation is
            // bounded, but 1x would make an unlucky size mix into a hard wall.
            mem.init_arena(cfg.cap * 2);
        }
        // A refused staging reservation reclaims cache, without the scheduler
        // knowing what a cache is.
        io.set_reclaimer(&cache);
    }

    // llama.cpp met tensor `t` in our buffer (at load, and for views on every
    // graph allocation): bind it to its disk source, read the floor, or point it
    // at the sentinel.
    ggml_status on_init_tensor(ggml_tensor* t);
    // llama.cpp reads a weight back.
    void on_get_tensor(const ggml_tensor* t, void* data, size_t offset, size_t size);

    // A whole tensor, from wherever it is cheapest: residency, then the ring,
    // then a direct read from disk. False = it could not be materialised; `t`
    // then points at poison, never at null.
    // `transient`: brought in only to be copied elsewhere (a GPU split's input):
    // never pinned static, and evictable with the routed experts, first.
    bool bring_in(ggml_tensor* t, uint64_t* streamed, bool transient = false);

    // The region a GPU split's copy reads, between copy_begin and copy_end
    // (ExpertCompactor::read_experts_for_copy): freed after the copy, and the
    // tensor points back where it pointed before.
    struct CopyLanding {
        ggml_tensor* w = nullptr;
        ggml_tensor* t = nullptr;
        uint8_t*     mem = nullptr;
        uint64_t     bytes = 0;
        void*        w_data = nullptr;
        void*        t_data = nullptr;
    };
    CopyLanding landing;
    // GPU split copies (copy_begin / copy_end): how many, the time spent making the
    // weight present (reads), and the time from then until the copy had completed --
    // the scheduler's copy plus its device drain. Printed only when any happened.
    uint64_t copies = 0;
    uint64_t ns_copy_ready = 0;
    uint64_t ns_copy_done = 0;
    std::chrono::steady_clock::time_point copy_t1{};
    // Every routed-expert tensor's bytes (the plan's tensor table). Whole GPU
    // copies stay cached only if ALL of them fit the cache budget; otherwise
    // a pass cycles them out before reuse and routed-only reads win.
    uint64_t routed_bytes = 0;
    uint64_t routed_max[ExpertCompactor::kCopyKinds] = {};   // largest routed tensor per kind
    // The GPU copy pool (set_gpu_copy_pool): where routed experts land for a GPU
    // split's copy -- host memory the GPU backend DMAs from directly, charged to the
    // cap by the engine. Empty: copies land in ordinary cache memory.
    void*    pinned_buf = nullptr;   // ggml_backend_buffer_t
    uint8_t* pinned = nullptr;
    uint64_t pinned_bytes = 0;
};

}  // namespace dray::backend
