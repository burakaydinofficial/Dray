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
//   A SOFTWARE QUEUE IN FRONT OF THE DEVICE. Callers enqueue; they are never
//   refused for queue depth and never wait for a slot. The scheduler keeps the
//   backend up to its depth, reads a node is waiting on (Urgent) before ring
//   read-ahead (Background). Order changes only WHEN a read lands, never which
//   bytes, so everything decided above this class stays deterministic.
//
//   ONE THREAD OWNS THE DEVICE. With the I/O thread running (the default once
//   loading is done), it alone calls backend submit/poll: it keeps the queue
//   full, harvests completions and copies staged bytes home WHILE the compute
//   thread computes. Before, reads only progressed between graph nodes -- in a
//   K3 run the device sat unharvested for ~91 of 121 s. Without the thread
//   (DRAY_IO_THREAD=0, and during load) the compute thread drives the same
//   service loop itself, from kick() and while waiting.
//
//   The compute thread keeps everything else: staging allocation and freeing,
//   the ledger, every decision. The thread changes WHEN a read lands, never
//   what is read, so bytes, nodes and text are identical with it on or off.
//   (A 2026-08-24 attempt put a second harvester inside the Windows backend,
//   next to its own reaping path, and raced; here there is only one consumer.)

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
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

// Which queue a read joins. Urgent: something is (or soon will be) waiting on it
// -- sync batches, whole tensors, expert regions. Background: speculative ring
// read-ahead, submitted only when no urgent read is queued.
enum class IoPriority : uint8_t { Urgent, Background };

// One region of a shard to land at `dst`.
struct Slice {
    Source   src;
    uint8_t* dst = nullptr;
    uint64_t len = 0;
};

// Merges slices that are adjacent both in their file and in memory, in (shard, offset)
// order, while a merged read stays within max_len (a slice is never split). A prompt
// chunk routes nearly every expert, and an expert tensor's experts sit next to each
// other on disk and in a full-size region: one request per expert (~640 KB on Qwen3.6
// 35B-A3B) read at ~3.6 GB/s on a drive that does 6 at 2 MiB. Callers pass the batch's
// bytes over the queue depth, so the queue stays full with the largest requests that
// fill it (one huge request per run left the drive idle: 2.8-3.0 GB/s).
void coalesce_slices(std::vector<Slice>& slices, uint64_t max_len);

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
    // submit_exact: reads served straight into place vs through a bounce buffer.
    uint64_t exact_direct = 0, exact_direct_bytes = 0, exact_staged = 0;
    // Who polled. Without the I/O thread a poll gap is the compute thread's
    // ABSENCE (reads finished, nobody collected them -- the 2026-08-24 defect).
    // With it, a gap is the I/O thread IDLE: nothing queued, nothing in flight,
    // so the engine had nothing to ask the drive for. Same numbers, opposite
    // meaning, so the report says which it is.
    bool     threaded = false;

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
    // The same, over a backend the caller supplies -- the seam the unit tests use
    // to drive this class with a scripted fake (out-of-order, short, failing,
    // dead). Takes ownership.
    bool open_with(std::unique_ptr<io::Backend> backend,
                   const std::vector<std::string>& shard_paths, std::string* error);
    void set_reclaimer(Reclaimer* r) { reclaimer_ = r; }

    // Moves the service loop onto the I/O thread; from then on only that thread
    // calls the backend. Call once loading is done (the file table is touched
    // single-threaded during load). No-op if already running; if the thread
    // cannot be created the scheduler simply stays inline.
    void start_thread();
    // Stops and joins the I/O thread. Every read must have been settled first;
    // the destructor calls it.
    void stop_thread();
    bool threaded() const { return threaded_; }

    bool     is_open() const { return io_ != nullptr; }
    // The max alignment across ALL shards (Invariant 5: discovered, and the
    // conservative max is correct for every member).
    uint32_t align() const { return align_; }
    io::FileId file(int32_t shard) const { return shards_[static_cast<size_t>(shard)]; }
    // Reads handed to the device and not yet harvested (diagnostics only).
    size_t   in_flight() const;
    size_t   max_in_flight() const { return io_->max_in_flight(); }
    std::string describe() const;

    // Makes progress without blocking: submits queued reads and harvests
    // whatever has already completed. For callers with CPU time to give; a
    // no-op when the I/O thread is doing it continuously.
    void kick();

    // Enqueues one read through an accounted staging buffer (the file range is
    // widened to the shard's alignment). 0 = not started: the staging could not
    // be allocated even after reclaiming, or the shard is unknown.
    uint64_t submit_staged(const Source& s, void* dst, uint64_t bytes);
    // Enqueues a read of exactly `bytes` of `s` into `dst`, WITHOUT a bounce buffer
    // where possible: the alignment-sized middle is read straight into `dst`, and
    // only the two sub-alignment edges go through staging. That needs the middle's
    // address aligned, i.e. dst congruent to s.offset modulo the alignment (expert
    // regions place their data at that offset); otherwise -- or when there is no
    // aligned middle -- it is one staged read, exactly submit_staged. Appends 1-3
    // tags; false = not started (tags already appended must still be settled).
    bool submit_exact(const Source& s, uint8_t* dst, uint64_t bytes, std::vector<uint64_t>* tags);
    // Enqueues one read straight into `dst`, which the caller has already
    // widened and aligned. The read succeeds when the device returns at least
    // head + bytes. Not sampled into the depth statistics. Never refused.
    uint64_t submit_unstaged(io::FileId f, uint64_t file_offset, uint32_t len,
                             void* dst, uint64_t head, uint64_t bytes,
                             IoPriority prio = IoPriority::Urgent);
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
    // The largest merged read that still keeps the queue full for `total` bytes
    // (coalesce_slices): total over the backend's queue depth.
    uint64_t merge_limit(uint64_t total) const;
    bool read_exact(const Source& s, void* dst, uint64_t bytes);
    // read_exact through a different handle to the same shard (self_check's
    // fresh handles), reusing the alignment widening verbatim.
    bool read_exact_via(io::FileId f, const Source& s, void* dst, uint64_t bytes);
    // A whole tensor with NO staging and NO copy: allocated at the widened span
    // and read into directly; the tensor then points at base + *out_head.
    // Issued in chunks of at most 1 GiB by default (a request length is 32 bits).
    void* read_whole(mem::Category cat, const Source& s, uint64_t bytes,
                     uint64_t* out_alloc, uint32_t* out_head);
    // The largest single request read_whole issues (1 GiB). Settable for a backend
    // with a smaller transfer limit, and so tests can exercise chunking on small files.
    void     set_whole_chunk(uint64_t bytes) { whole_chunk_ = bytes; }

    // Handles for callers that need independent access to the same files.
    // Load time only (before start_thread): the backend's file table is not
    // shared with the I/O thread.
    io::FileId open_file(const std::string& path) { return io_->open(path); }
    void       close_file(io::FileId f) { io_->close(f); }

    // A snapshot: the I/O thread keeps updating the live counters.
    IoStats stats() const;

private:
    enum class State : uint8_t { Queued, Submitted, Done };
    struct InFlight {
        io::ReadRequest req{};     // what goes to the backend
        uint8_t* stage = nullptr;  // staged reads: the bounce buffer (req.dst)
        uint64_t head = 0;
        uint32_t span = 0;
        void*    mem = nullptr;    // destination the staged bytes are copied to
        uint64_t bytes = 0;
        State    state = State::Queued;
        bool     ok = false;
        uint64_t got = 0;          // bytes the device actually returned
        uint64_t t_submit_ns = 0;  // 0 = not timed (unstaged reads)
        int      status = 0;
    };

    using Lock = std::unique_lock<std::mutex>;

    uint64_t enqueue(InFlight entry, IoPriority prio);
    // Every *_locked function and every function taking a Lock runs with m_ held.
    // Submits queued reads, urgent first, until the backend is full.
    void   submit_queued_locked();
    // Polls the backend (lock RELEASED while blocked), routes each completion to
    // its entry and copies staged bytes home (lock released while copying), then
    // marks the entries Done and wakes waiters. Returns how many were harvested.
    size_t harvest(Lock& lk, size_t min_complete);
    // Inline mode's service loop: submit, harvest, submit again. 0 from a
    // blocking call while reads are outstanding means the backend is dead.
    size_t service(Lock& lk, size_t min_complete);
    // Waits until every tag is Done. False: the backend died with some of them
    // outstanding.
    bool   wait_all(Lock& lk, const std::vector<uint64_t>& tags);
    bool   all_done(const std::vector<uint64_t>& tags) const;
    // Removes a Done entry, returning its staging (freed by the caller, off-lock).
    void   take_locked(uint64_t tag, uint8_t** stage, uint32_t* span, bool* ok);
    void   run();   // the I/O thread

    AccountedAlloc&                        mem_;
    const std::atomic<uint64_t>&           failures_;
    Reclaimer*                             reclaimer_ = nullptr;
    std::unique_ptr<io::Backend>           io_;
    std::vector<io::FileId>                shards_;
    uint32_t                               align_ = 4096;
    uint64_t                               whole_chunk_ = 1ull << 30;   // read_whole request size

    // Shared with the I/O thread, all under m_.
    mutable std::mutex                     m_;
    std::condition_variable                work_cv_;   // wakes the I/O thread
    std::condition_variable                done_cv_;   // wakes waiters
    std::unordered_map<uint64_t, InFlight> pending_;   // by tag; element addresses
                                                        // are stable across inserts
    std::deque<uint64_t>                   urgent_;     // queued tags, FIFO
    std::deque<uint64_t>                   background_;
    size_t                                 submitted_ = 0;   // entries in the device
    uint64_t                               next_tag_ = 1;
    bool                                   stop_ = false;
    bool                                   dead_ = false;    // I/O thread saw the device die
    IoStats                                stats_;

    std::thread                            worker_;
    bool                                   threaded_ = false;   // compute thread only
    bool                                   start_tried_ = false;
};

}  // namespace dray::backend
