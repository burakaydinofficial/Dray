// Every disk read the streamer makes, and the ONE registry that tracks them.
//
// The storage backend (io/storage.h) is a submit/poll queue with no cancel. That
// contract shapes everything here:
//
//   ONE TAG SPACE. Every reader -- batched expert reads, sibling regions, the
//   ring, direct whole-tensor reads -- shares a single completion queue, so they
//   share a single registry keyed by a globally unique tag, and pump() is the
//   ONLY place completions are consumed. An earlier version had batch reads tag
//   requests 0,1,2... while another reader used a global counter: a completion
//   arriving during a batch poll was consumed as the wrong entry, and its waiter
//   blocked forever. Low CPU, low disk, no progress.
//
//   NO FREE UNDER DMA. A submitted read keeps writing into its buffer until the
//   backend reports it, success or failure. When the backend dies with reads
//   outstanding, the buffers are LEAKED deliberately and loudly (F2): freeing
//   under DMA trades a loud failure for silent corruption.
//
//   READS ONLY PROGRESS WHEN SOMEONE POLLS. There is no I/O thread: completions
//   are harvested on the compute thread, from inside the ggml callbacks. This
//   class is where a dedicated I/O thread would live, and nothing above it would
//   need to know.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/accounted_alloc.h"
#include "backend/tensor_source.h"
#include "io/storage.h"

namespace dray::backend {

// Frees cache memory when a read's staging buffer cannot be reserved. Staging is
// TRANSIENT AND MANDATORY -- without it no read can proceed -- so a refused
// reservation must reclaim rather than fail. Only called on that refusal, never
// on the per-read hot path.
class Reclaimer {
public:
    virtual bool reclaim(uint64_t bytes) = 0;
protected:
    ~Reclaimer() = default;
};

// One region of a shard to land at `dst`.
struct Slice {
    Source   src;
    uint8_t* dst = nullptr;
    uint64_t len = 0;
};

// I/O forensics, printed only under DRAY_IO_STATS. These counters located
// the bandwidth defect on 2026-08-24 and are worth keeping.
struct IoStats {
    // Queue depth sampled at submit, where the drive sees it. read_batch is NOT
    // the place to measure this: sibling regions are issued outside it,
    // unwaited, so counting slices per batch undercounts what is in flight.
    uint64_t depth_sum = 0, depth_n = 0, depth_max = 0, depth_min = ~0ull;
    uint64_t depth_low = 0;   // submits that found fewer than 4 in flight
    // TIME-WEIGHTED depth. The submit-sampled figure is biased high: it records
    // in_flight() at the one moment work is pushed in. This integrates depth over
    // elapsed time instead, which is what Little's law needs.
    uint64_t depth_area_ns = 0, depth_t_last = 0, depth_t0 = 0;
    size_t   depth_cur = 0;
    // Service time per request, microseconds: <200, <500, <1000, <2000, >=2000.
    // Four hypotheses about the missing bandwidth died for want of this number.
    uint64_t lat_bucket[5] = {0, 0, 0, 0, 0};
    uint64_t lat_sum_us = 0, lat_n = 0, lat_bytes = 0;
    // Harvesting cadence. If polls are rare and each returns a pile, the latency
    // is us not looking, not the drive being slow.
    uint64_t pump_calls = 0, pump_harvested = 0, pump_gap_us = 0, pump_last_ns = 0;
    uint64_t gap_bucket[5] = {0, 0, 0, 0, 0};   // <100us, <1ms, <5ms, <20ms, >=20ms
    uint64_t gap_max_us = 0;
    uint64_t gap_long_us = 0;   // total time spent in gaps >= 5ms
    uint64_t in_pump_ns = 0;    // time INSIDE pump, i.e. blocked in poll
    uint64_t memcpy_ns = 0, memcpy_bytes = 0;
    uint64_t stage_alloc_ns = 0, stage_alloc_n = 0;
    uint64_t io_batches = 0, io_slices = 0, io_batch_max = 0;

    // Everything from ", io ..." onward in the streamer report.
    void append(std::ostream& o) const;
};

class IoScheduler {
public:
    // `failures` only throttles diagnostics (the first few failures are logged).
    IoScheduler(AccountedAlloc& mem, const std::atomic<uint64_t>& failures)
        : mem_(mem), failures_(failures) {}
    ~IoScheduler();
    IoScheduler(const IoScheduler&) = delete;
    IoScheduler& operator=(const IoScheduler&) = delete;

    // Opens the backend and every shard uncached, and discovers the alignment.
    // False with `error` set on failure; whatever did open is closed at teardown.
    bool open(const std::vector<std::string>& shard_paths, uint32_t queue_depth,
              std::string* error);
    void set_reclaimer(Reclaimer* r) { reclaimer_ = r; }

    bool     is_open() const { return io_ != nullptr; }
    // The max alignment across ALL shards (Invariant 5: discovered, and the
    // conservative max is correct for every member).
    uint32_t align() const { return align_; }
    io::FileId file(int32_t shard) const { return shards_[static_cast<size_t>(shard)]; }
    size_t   in_flight() const { return io_->in_flight(); }
    size_t   max_in_flight() const { return io_->max_in_flight(); }
    std::string describe() const;

    // Harvests up to a batch of completions and routes each to its registry
    // entry, copying staged bytes to their destination. Returns how many were
    // harvested; 0 while reads are outstanding means the backend is dead.
    size_t pump(size_t min_complete);
    // Pumps until the backend will accept another request. False = backend dead.
    bool wait_for_slot();

    // Registers and submits one read through an accounted staging buffer (the
    // file range is widened to the shard's alignment). 0 = not started.
    uint64_t submit_staged(const Source& s, void* dst, uint64_t bytes);
    // Registers and submits one read straight into `dst`, which the caller has
    // already widened and aligned. The read succeeds when the device returns at
    // least head + bytes. Not sampled into the depth statistics. 0 = refused.
    uint64_t submit_unstaged(io::FileId f, uint64_t file_offset, uint32_t len,
                             void* dst, uint64_t head, uint64_t bytes);
    // Drops a completed tag, freeing its staging.
    void discard(uint64_t tag);
    // Waits for every tag, then discards them all. False if any read failed.
    // `backend_dead` (optional) distinguishes "a read landed with a bad status"
    // (memory is ours again) from "the backend died with reads OUTSTANDING"
    // (the buffers are still DMA targets and must be leaked, never reused).
    bool settle(std::vector<uint64_t>& tags, bool* backend_dead = nullptr);

    // Reads many slices as ONE batch: every request submitted before any is
    // waited on, so the drive sees a deep queue (the calibrated difference
    // between QD1 0.5-3.4 GB/s and QD16 6.6).
    bool read_batch(const std::vector<Slice>& slices);
    bool read_exact(const Source& s, void* dst, uint64_t bytes);
    // read_exact through a different handle to the same shard (self_check's
    // fresh handles), reusing the alignment widening verbatim.
    bool read_exact_via(io::FileId f, const Source& s, void* dst, uint64_t bytes);
    // A whole tensor with NO staging and NO copy: allocated at the widened span
    // and read into directly; the tensor then points at base + *out_head.
    void* read_whole(mem::Category cat, const Source& s, uint64_t bytes,
                     uint64_t* out_alloc, uint32_t* out_head);

    // Handles for callers that need independent access to the same files.
    io::FileId open_file(const std::string& path) { return io_->open(path); }
    void       close_file(io::FileId f) { io_->close(f); }

    const IoStats& stats() const { return stats_; }

private:
    struct InFlight {
        uint8_t* stage = nullptr;
        uint64_t head = 0;
        uint32_t span = 0;
        void*    mem = nullptr;    // destination the staged bytes are copied to
        uint64_t bytes = 0;
        bool     done = false;
        bool     ok = false;
        uint64_t got = 0;          // bytes the device actually returned
        uint64_t t_submit_ns = 0;  // 0 = not timed (unstaged reads)
        int      status = 0;
    };

    AccountedAlloc&                        mem_;
    const std::atomic<uint64_t>&           failures_;
    Reclaimer*                             reclaimer_ = nullptr;
    std::unique_ptr<io::Backend>           io_;
    std::vector<io::FileId>                shards_;
    uint32_t                               align_ = 4096;
    std::unordered_map<uint64_t, InFlight> pending_;   // by tag
    uint64_t                               next_tag_ = 1;
    IoStats                                stats_;
};

}  // namespace dray::backend
