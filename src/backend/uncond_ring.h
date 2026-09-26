// PHASE B: the unconditional-stream ring.
//
// 91% of bytes per token are the same addresses in the same order every token
// -- a zero-branch FIFO stream once read as a stop-and-wait gather. The ring is a
// fixed arena that those tensors stream through in graph order: produced ahead
// by produce() whenever a callback gives us the CPU, consumed by the materialiser
// as a lookup instead of a read. Consumption order equals production order, so
// a FIFO byte-ring has no fragmentation and cannot deadlock: full ring, producer
// waits; consumer (compute) drains it.
//
// The stream is CYCLIC -- after the last unconditional tensor of a pass, the next
// pass's first is already known -- so the producer wraps across token boundaries
// and the drive need never idle. A stale segment is byte-correct by definition
// (weights are immutable); desync costs duplicate reads, never wrong data.
//
// The ring's memory is an accounted, fixed line item (IoStaging), sized once by
// StreamBudget -- never carved from the cache at runtime, which is what killed
// the per-tensor prefetch it replaced.

#pragma once

#include <cstdint>
#include <deque>
#include <ostream>
#include <unordered_set>
#include <vector>

#include "backend/accounted_alloc.h"
#include "backend/hit_rates.h"
#include "backend/io_scheduler.h"
#include "backend/residency_cache.h"
#include "backend/tensor_registry.h"
#include "ggml.h"

namespace dray::backend {

class UncondRing {
public:
    // Everything the ring touches, borrowed from the Streamer that owns them.
    struct Parts {
        AccountedAlloc&       mem;
        IoScheduler&          io;
        ResidencyCache&       cache;
        const TensorRegistry& tensors;
        HitRates&             hits;
    };
    // `no_retain` is DRAY_NO_RETAIN.
    UncondRing(const Parts& parts, bool no_retain) : p_(parts), no_retain_(no_retain) {}
    UncondRing(const UncondRing&) = delete;
    UncondRing& operator=(const UncondRing&) = delete;

    // Allocates the arena; a refusal just means no ring (the sync path serves).
    void     allocate(uint64_t bytes, uint32_t align);
    uint64_t bytes() const { return bytes_; }

    // PASS 1: the graph declares its real working set by consuming it. The stream
    // list is built from THIS, not from creation order: the GGUF carries tensors
    // the graph never references (blk.0.attn_res_score.weight on K3), and one such
    // segment at the FIFO front wedged the entire ring -- 0 pops in 136k calls.
    void record_consumption(ggml_tensor* t);
    // A forward pass began. The ring activates at pass 2, once static pinning has
    // claimed its prefix in pass 1 and the pinned/streamed split is settled.
    void on_pass(uint32_t pass);

    // The producer: reclaim the consumed front (promoting into genuinely free
    // cache room), then submit ahead until the queue is deep or the ring full.
    // Called from the callbacks (end of materialise, release): between callbacks
    // the kernel keeps the already-submitted DMA flowing.
    void produce();
    // The consumer: points `t` at its landed segment. False = not in the ring (or
    // not landed correctly); the caller reads synchronously -- correct, slower.
    bool consume(ggml_tensor* t, uint64_t* streamed);
    // `t` is being served from residency: a pending segment for it would sit
    // unconsumed at the FIFO front forever and wedge reclaim (the attn_res_score
    // failure again, reached via LRU). Settles its DMA and retires it.
    void retire_pending(ggml_tensor* t);

    uint64_t hits() const { return hits_; }
    uint64_t promotions() const { return promotions_; }
    // "N ring-fed (front=..., segs, pops, calls; stops ...), " for the report.
    void append(std::ostream& o) const;

    // Teardown: outstanding chunk DMA lands in the arena, so every tag must
    // settle first; a dead backend forces the arena to leak, loudly.
    void release_all();

private:
    struct Segment {
        ggml_tensor*          t = nullptr;   // null = wrap padding
        uint64_t              pos = 0;       // arena offset
        uint64_t              span = 0;      // widened to device alignment
        uint32_t              head = 0;      // file-alignment lead-in
        std::vector<uint64_t> tags;          // outstanding chunk reads
        bool                  consumed = false;
        bool                  complete = false;  // every chunk submitted
        // SUBMITTED IS NOT LANDED. `complete` says every chunk was handed to the
        // backend; it says nothing about whether the bytes arrived. Retention once
        // promoted a segment on `complete` alone, so a short read or an EIO could
        // install partly-unwritten arena bytes as a resident weight and serve them
        // as a RAM hit every token, with no failure counted (2026-08-24 audit).
        bool                  reads_ok = true;
    };

    // The backend died with ring reads outstanding: the arena is still a DMA
    // target. Deactivating stops reuse -- the only reclaim site is gated on
    // `active_` -- and `died_` makes release_all() leak the arena. The death
    // must be REMEMBERED: settle() clears the dying segment's tags, so at
    // teardown that segment looks clean, and a ring whose only outstanding
    // reads were the dead ones used to be freed under DMA (review finding,
    // 2026-09-25).
    void retire_dead_ring();
    // Waits for a segment's chunks and releases them from the read-ahead count.
    bool settle_segment(Segment& sg, bool* dead);

    Parts          p_;
    const bool     no_retain_;
    uint8_t*       arena_ = nullptr;
    uint64_t       bytes_ = 0;
    uint64_t       head_off_ = 0;               // next placement offset
    uint64_t       used_ = 0;                   // reserved, incl. wrap padding
    std::deque<Segment>       queue_;           // FIFO, front = oldest
    std::vector<ggml_tensor*> stream_list_;     // streamed uncond tensors, graph order
    size_t         cursor_ = 0;                 // modular over stream_list_
    bool           active_ = false;
    bool           died_ = false;               // backend died under the ring
    std::vector<ggml_tensor*>        consumed_order_;
    std::unordered_set<ggml_tensor*> consumed_seen_;
    uint32_t       idle_backoff_ = 0;           // S22: sleeps after a fruitless lap
    size_t         live_chunks_ = 0;            // chunk reads enqueued, not yet settled
    // Why the producer stopped, per call site -- 64 ring-fed twice in a row with
    // two different theories is two theories too many. Indexed: 0=depth 1=lap
    // 2=full-at-wrap 3=full 4=submit-refused 5=produced-nothing-else.
    uint64_t       stop_[6] = {};
    uint64_t       segs_ = 0, pops_ = 0, calls_ = 0, hits_ = 0, promotions_ = 0;
};

}  // namespace dray::backend
