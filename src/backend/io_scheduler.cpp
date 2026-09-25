#include "backend/io_scheduler.h"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace dray::backend {

namespace {

uint64_t steady_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// lifetime

IoScheduler::~IoScheduler() {
    if (io_) {
        for (io::FileId f : shards_) io_->close(f);
    }
}

bool IoScheduler::open(const std::vector<std::string>& shard_paths, uint32_t queue_depth,
                       std::string* error) {
    io_ = io::make_backend(queue_depth);
    if (!io_) { *error = "no storage backend"; return false; }

    for (const std::string& sp : shard_paths) {
        io::FileId f = io_->open(sp);
        if (f == io::kInvalidFile) { *error = "cannot open shard uncached: " + sp; return false; }
        shards_.push_back(f);
    }
    if (shards_.empty()) { *error = "plan carries no shards"; return false; }
    // Pass-4b: the max across ALL shards, not shard 0's -- shards of one model
    // share a directory in practice, but Invariant 5 says discovered, not
    // assumed, and the conservative max is correct for every member.
    align_ = 0;
    for (io::FileId sf : shards_) {
        const uint64_t a = io_->alignment(sf).max();
        if (a > align_) align_ = static_cast<uint32_t>(a);
    }
    if (align_ == 0) align_ = 4096;
    return true;
}

std::string IoScheduler::describe() const {
    return io_ ? io_->describe() : std::string("no backend");
}

// ---------------------------------------------------------------------------
// completions

size_t IoScheduler::pump(size_t min_complete) {
    if (io_->in_flight() == 0) return 0;
    {
        const uint64_t now = steady_ns();
        if (stats_.pump_last_ns) {
            const uint64_t g = (now - stats_.pump_last_ns) / 1000;
            stats_.pump_gap_us += g;
            stats_.gap_bucket[g < 100 ? 0 : g < 1000 ? 1 : g < 5000 ? 2 : g < 20000 ? 3 : 4]++;
            if (g > stats_.gap_max_us) stats_.gap_max_us = g;
            if (g >= 5000) stats_.gap_long_us += g;
        }
        stats_.pump_last_ns = now;
        ++stats_.pump_calls;
    }
    std::vector<io::Completion> comps(64);
    const auto pp0 = std::chrono::steady_clock::now();
    const size_t got = io_->poll(comps.data(), comps.size(), min_complete);
    stats_.in_pump_ns += ns_since(pp0);
    stats_.pump_harvested += got;
    for (size_t i = 0; i < got; ++i) {
        auto it = pending_.find(comps[i].tag);
        if (it == pending_.end()) continue;   // already settled
        InFlight& p = it->second;
        p.done = true;
        if (p.t_submit_ns) {
            const uint64_t us = (steady_ns() - p.t_submit_ns) / 1000;
            stats_.lat_sum_us += us; ++stats_.lat_n; stats_.lat_bytes += p.span;
            stats_.lat_bucket[us < 200 ? 0 : us < 500 ? 1 : us < 1000 ? 2 : us < 2000 ? 3 : 4]++;
        }
        // Short at EOF is legal: the span is widened to device alignment and the
        // last tensor in a shard runs past the end. The interior must be covered.
        p.ok = comps[i].status == 0 &&
               static_cast<uint64_t>(comps[i].bytes) >= p.head + p.bytes;
        // Keep what the device actually returned, so a failure can report the short
        // read rather than just asserting one happened.
        p.got    = static_cast<uint64_t>(comps[i].bytes);
        p.status = comps[i].status;
        if (p.ok && p.mem) {
            const auto c0 = std::chrono::steady_clock::now();
            std::memcpy(p.mem, p.stage + p.head, static_cast<size_t>(p.bytes));
            stats_.memcpy_ns += ns_since(c0);
            stats_.memcpy_bytes += p.bytes;
        }
    }
    return got;
}

bool IoScheduler::wait_for_slot() {
    // The op pool IS the queue-depth gate: beyond it submit() refuses, so drain.
    while (io_->in_flight() >= io_->max_in_flight()) {
        if (pump(1) == 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// submission

uint64_t IoScheduler::submit_staged(const Source& s, void* dst, uint64_t bytes) {
    if (s.shard < 0 || static_cast<size_t>(s.shard) >= shards_.size()) return 0;
    const io::FileId f = shards_[static_cast<size_t>(s.shard)];
    const io::Alignment al = io_->alignment(f);
    const uint64_t lo = (s.offset / al.offset) * al.offset;
    const uint64_t head = s.offset - lo;
    uint64_t span = head + bytes;
    span = ((span + al.length - 1) / al.length) * al.length;

    // Staging counts against the cap like anything else -- it was never accounted,
    // and with a deep queue it is not small. But unlike cache, staging is TRANSIENT
    // AND MANDATORY: without it no read can proceed, so a refused reservation must
    // reclaim rather than fail. Once accounting was switched on, reads began
    // failing outright at the cap with the cache still holding 9.89 of 10.21 GB.
    const auto sa0 = std::chrono::steady_clock::now();
    uint8_t* stage = static_cast<uint8_t*>(
        mem_.alloc(mem::Category::IoStaging, span, al.memory));
    stats_.stage_alloc_ns += ns_since(sa0);
    ++stats_.stage_alloc_n;
    if (!stage) {
        // T23: an earlier version pumped completions here hoping "completed
        // reads still holding staging" would free room. They cannot: pump()
        // only marks entries done -- staging is freed by each tag's OWNER
        // (settle/discard), none of which can run between iterations of a
        // loop on this single-threaded path. The ledger state was identical
        // on every retry, so each pass was a blocking poll for nothing.
        // Evicting cache is the one lever that actually moves the sum.
        if (reclaimer_ && reclaimer_->reclaim(span)) {
            stage = static_cast<uint8_t*>(
                mem_.alloc(mem::Category::IoStaging, span, al.memory));
        }
    }
    if (!stage) return 0;

    const uint64_t tag = next_tag_++;
    io::ReadRequest r{f, lo, static_cast<uint32_t>(span), stage, tag};
    if (io_->submit(&r, 1) != 1) {
        mem_.free(mem::Category::IoStaging, stage, span);
        return 0;
    }
    // TRUE queue depth, sampled where the drive sees it (see IoStats).
    {
        const size_t nf = io_->in_flight();
        {
            const uint64_t t = steady_ns();
            if (stats_.depth_t_last) stats_.depth_area_ns += stats_.depth_cur * (t - stats_.depth_t_last);
            else stats_.depth_t0 = t;
            stats_.depth_t_last = t;
            stats_.depth_cur = nf;
        }
        stats_.depth_sum += nf;
        ++stats_.depth_n;
        if (nf > stats_.depth_max) stats_.depth_max = nf;
        if (nf < stats_.depth_min) stats_.depth_min = nf;
        if (nf < 4) ++stats_.depth_low;
    }

    pending_[tag] = {stage, head, static_cast<uint32_t>(span), dst, bytes,
                     false, false, 0, steady_ns()};
    return tag;
}

uint64_t IoScheduler::submit_unstaged(io::FileId f, uint64_t file_offset, uint32_t len,
                                      void* dst, uint64_t head, uint64_t bytes) {
    const uint64_t tag = next_tag_++;
    io::ReadRequest r{f, file_offset, len, dst, tag};
    // stage and mem both null: pump has nothing to copy, discard nothing to free.
    pending_[tag] = {nullptr, head, len, nullptr, bytes, false, false};
    if (io_->submit(&r, 1) != 1) {
        pending_.erase(tag);
        return 0;
    }
    return tag;
}

void IoScheduler::discard(uint64_t tag) {
    auto it = pending_.find(tag);
    if (it == pending_.end()) return;
    mem_.free(mem::Category::IoStaging, it->second.stage, it->second.span);
    pending_.erase(it);
}

bool IoScheduler::settle(std::vector<uint64_t>& tags, bool* backend_dead) {
    if (backend_dead) *backend_dead = false;
    bool ok = true;
    for (;;) {
        bool all = true;
        for (uint64_t t : tags) {
            auto it = pending_.find(t);
            if (it != pending_.end() && !it->second.done) { all = false; break; }
        }
        if (all) break;
        if (pump(1) == 0) {
            // F2 (same shape as read_batch): a dead backend with reads
            // outstanding must LEAK the staging, never free under DMA.
            // I1: THE line H15 forgot -- without it the out-param stayed false,
            // both dtor consumers computed constants, and the teardown guard
            // was disarmed in the direction of always-free. A fix that adds a
            // flag and never sets it is worse than no fix: it retires the
            // FATAL message that would have said so.
            if (backend_dead) *backend_dead = true;
            std::fprintf(stderr,
                         "[dray] FATAL: backend dead with reads outstanding; "
                         "leaking %zu staging buffers (no-cancel contract)\n",
                         tags.size());
            for (uint64_t t : tags) pending_.erase(t);
            tags.clear();
            return false;
        }
    }
    for (uint64_t t : tags) {
        auto it = pending_.find(t);
        if (it == pending_.end() || !it->second.ok) ok = false;
        discard(t);
    }
    tags.clear();
    return ok;
}

// ---------------------------------------------------------------------------
// whole reads

bool IoScheduler::read_batch(const std::vector<Slice>& slices) {
    if (slices.empty()) return true;
    ++stats_.io_batches;
    stats_.io_slices += slices.size();
    if (slices.size() > stats_.io_batch_max) stats_.io_batch_max = slices.size();

    std::vector<uint64_t> tags;
    tags.reserve(slices.size());
    bool failed = false;

    for (const Slice& sl : slices) {
        // Respect the backend's depth: beyond it submit() refuses, so drain first.
        if (!wait_for_slot()) { failed = true; break; }
        const uint64_t tag = submit_staged(sl.src, sl.dst, sl.len);
        if (tag == 0) { failed = true; break; }
        tags.push_back(tag);
    }

    // F2: settle ALL submitted tags UNCONDITIONALLY before any discard --
    // success or failure, every submitted read has DMA outstanding into its
    // staging buffer, and discard() frees that buffer for immediate reuse.
    // The old shape skipped this wait when `failed` was set mid-submission,
    // freeing under DMA (the no-cancel contract, storage.h and D8; T23's
    // deletion of submit_one's drain loop made the path reachable). A failed
    // batch still waits for its own reads; only the RESULT is failed.
    for (;;) {
        bool outstanding = false;
        for (uint64_t t : tags) {
            auto it = pending_.find(t);
            if (it != pending_.end() && !it->second.done) { outstanding = true; break; }
        }
        if (!outstanding) break;
        if (pump(1) == 0) {
            // Backend dead with reads outstanding. Freeing under possible DMA
            // trades a loud failure for silent corruption, so the staging is
            // LEAKED deliberately: pending entries are dropped without a free,
            // the bytes stay charged to IoStaging forever, and the ledger
            // honestly shows the cost of a dead backend.
            std::fprintf(stderr,
                         "[dray] FATAL: backend dead with reads outstanding; "
                         "leaking %zu staging buffers (no-cancel contract)\n",
                         tags.size());
            for (uint64_t t : tags) pending_.erase(t);
            return false;
        }
    }

    for (uint64_t t : tags) {
        auto it = pending_.find(t);
        if (it == pending_.end() || !it->second.ok) failed = true;
        discard(t);
    }
    return !failed;
}

// A single-slice batch, so it shares the one tag space and the one completion
// dispatcher. It once submitted with tag 0 and polled the queue itself, which
// meant another reader's completion arriving here was consumed and lost, and
// its waiter blocked forever.
bool IoScheduler::read_exact(const Source& s, void* dst, uint64_t bytes) {
    const std::vector<Slice> one{{s, static_cast<uint8_t*>(dst), bytes}};
    return read_batch(one);
}

bool IoScheduler::read_exact_via(io::FileId f, const Source& s, void* dst, uint64_t bytes) {
    // Swap the shard table entry for the duration of one read, so the widening
    // and the tag handling are exactly read_exact's.
    const io::FileId saved = shards_[static_cast<size_t>(s.shard)];
    shards_[static_cast<size_t>(s.shard)] = f;
    const bool got = read_exact(s, dst, bytes);
    shards_[static_cast<size_t>(s.shard)] = saved;
    return got;
}

// NO staging buffer and NO copy. The destination is allocated at the
// ALIGNMENT-WIDENED span and read into directly; the padding costs at most two
// alignment units (~8 KB) against a tensor of up to 1.6 GB.
//
// Staging a whole tensor meant holding it twice -- output.weight is 1.14 GB, so
// a read needed 2.28 GB of transient space against a 0.85 GB churn reserve, and
// simply could not be satisfied at this cap. It also copied every byte read,
// which is pure memory bandwidth on the hot path for no benefit.
//
// Compaction still stages: its slices must land contiguously at an exact stride,
// and arbitrary file offsets cannot be made to line up with that. Those slices
// are megabytes, not gigabytes.
void* IoScheduler::read_whole(mem::Category cat, const Source& s, uint64_t bytes,
                              uint64_t* out_alloc, uint32_t* out_head) {
    auto bail = [&](const char* why) -> void* {
        if (failures_ < 8) {
            std::fprintf(stderr, "[dray] DIRECT %s: shard=%d bytes=%llu inflight=%zu/%zu\n",
                         why, s.shard, (unsigned long long)bytes,
                         io_->in_flight(), io_->max_in_flight());
            std::fflush(stderr);
        }
        return nullptr;
    };

    if (s.shard < 0 || static_cast<size_t>(s.shard) >= shards_.size()) return bail("BAD SHARD");
    const io::FileId f = shards_[static_cast<size_t>(s.shard)];
    const io::Alignment al = io_->alignment(f);

    const uint64_t lo = (s.offset / al.offset) * al.offset;
    const uint64_t head = s.offset - lo;
    uint64_t span = head + bytes;
    span = ((span + al.length - 1) / al.length) * al.length;

    uint8_t* base = static_cast<uint8_t*>(mem_.alloc(cat, span, al.memory));
    if (!base) return bail("ALLOC");

    // The op pool IS the queue-depth gate, so a full pool makes submit refuse. Drain
    // and retry rather than failing the read: a transient queue-full is not a
    // reason to abandon a weight the graph needs.
    uint64_t tag = 0;
    for (int attempt = 0; attempt < 64; ++attempt) {
        tag = submit_unstaged(f, lo, static_cast<uint32_t>(span), base, head, bytes);
        if (tag != 0) break;
        if (io_->in_flight() == 0) break;   // not queue pressure; genuinely refused
        if (pump(1) == 0) break;
    }
    if (tag == 0) {
        mem_.free(cat, base, span);
        return bail("SUBMIT");
    }
    bool backend_dead = false;
    for (;;) {
        auto it = pending_.find(tag);
        if (it == pending_.end() || it->second.done) break;
        // Pass-4b: a dead backend here means the read may STILL be writing into
        // `base` -- fabricating done/ok and freeing below was the third
        // free-under-DMA site of the F2 class (the first two were fixed, this
        // one was not enumerated). Same remedy: leak deliberately and loudly.
        if (pump(1) == 0) { backend_dead = true; break; }
    }
    if (backend_dead) {
        std::fprintf(stderr,
                     "[dray] FATAL: backend dead with a direct read outstanding; "
                     "leaking its staging (no-cancel contract)\n");
        pending_.erase(tag);
        return nullptr;
    }
    auto it = pending_.find(tag);
    const bool ok = (it != pending_.end()) && it->second.ok;
    if (!ok && failures_ < 8) {
        // Name the reason. "read failed" without the numbers has cost hours on this
        // codebase every time; status and the short-read arithmetic identify it at
        // a glance.
        std::fprintf(stderr,
            "[dray] DIRECT READ FAIL shard=%d off=%llu lo=%llu head=%llu bytes=%llu "
            "span=%u got=%lld status=%d\n",
            s.shard, (unsigned long long)s.offset, (unsigned long long)lo,
            (unsigned long long)head, (unsigned long long)bytes, (unsigned)span,
            it == pending_.end() ? -1LL : (long long)it->second.got,
            it == pending_.end() ? -1 : it->second.status);
        std::fflush(stderr);
    }
    pending_.erase(tag);
    if (!ok) { mem_.free(cat, base, span); return nullptr; }

    *out_alloc = span;
    *out_head  = static_cast<uint32_t>(head);
    return base;
}

// ---------------------------------------------------------------------------

void IoStats::append(std::ostream& o) const {
    o << ", io " << io_batches << " batches/" << io_slices
      << " slices (max " << io_batch_max << "), qdepth mean "
      << (depth_n ? (depth_sum / depth_n) : 0) << " max " << depth_max
      << " min " << (depth_min == ~0ull ? 0 : depth_min)
      << " (" << depth_low << " submits found <4 in flight)";
    if (depth_t_last > depth_t0) {
        o << ", time-weighted depth "
          << (depth_area_ns / (depth_t_last - depth_t0));
    }
    if (lat_n) {
        o << ", read latency mean " << (lat_sum_us / lat_n) << "us over "
          << lat_n << " reads (" << (lat_bytes / lat_n / 1024) << " KiB mean): <200us "
          << lat_bucket[0] << ", <500 " << lat_bucket[1] << ", <1ms " << lat_bucket[2]
          << ", <2ms " << lat_bucket[3] << ", slower " << lat_bucket[4]
          << "; polls " << pump_calls << " mean gap "
          << (pump_calls > 1 ? pump_gap_us / (pump_calls - 1) : 0) << "us, "
          << (pump_calls ? pump_harvested / pump_calls : 0) << " harvested each"
          << "; gap buckets <100us " << gap_bucket[0] << " <1ms " << gap_bucket[1]
          << " <5ms " << gap_bucket[2] << " <20ms " << gap_bucket[3]
          << " longer " << gap_bucket[4] << ", max " << gap_max_us << "us, "
          << (gap_long_us / 1000) << "ms total spent in gaps over 5ms, "
          << (in_pump_ns / 1000000) << "ms BLOCKED INSIDE poll"
          << "; staging memcpy " << (memcpy_ns / 1000000) << "ms for "
          << (memcpy_bytes / 1000000) << " MB on the poll thread"
          << ", staging alloc " << (stage_alloc_ns / 1000000) << "ms over "
          << stage_alloc_n << " reads";
    }
}

}  // namespace dray::backend
