#include "backend/io_scheduler.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <system_error>

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

void log_leak(size_t n) {
    std::fprintf(stderr,
                 "[dray] FATAL: backend dead with reads outstanding; "
                 "leaking %zu staging buffers (no-cancel contract)\n", n);
}

}  // namespace

// ---------------------------------------------------------------------------
// lifetime

IoScheduler::~IoScheduler() {
    stop_thread();
    if (io_) {
        for (io::FileId f : shards_) io_->close(f);
    }
}

bool IoScheduler::open(const std::vector<std::string>& shard_paths, uint32_t queue_depth,
                       std::string* error) {
    return open_with(io::make_backend(queue_depth), shard_paths, error);
}

bool IoScheduler::open_with(std::unique_ptr<io::Backend> backend,
                            const std::vector<std::string>& shard_paths, std::string* error) {
    io_ = std::move(backend);
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

void IoScheduler::start_thread() {
    if (threaded_ || start_tried_ || !io_) return;
    start_tried_ = true;   // once: a failed attempt is not retried on every node
    try {
        worker_ = std::thread([this] { run(); });
        threaded_ = true;
    } catch (const std::system_error&) {
        // No thread: the compute thread keeps servicing the queue, as before.
    }
}

void IoScheduler::stop_thread() {
    if (!threaded_) return;
    {
        Lock lk(m_);
        stop_ = true;
    }
    work_cv_.notify_all();
    worker_.join();
    threaded_ = false;
}

std::string IoScheduler::describe() const {
    return io_ ? io_->describe() : std::string("no backend");
}

size_t IoScheduler::in_flight() const {
    Lock lk(m_);
    return submitted_;
}

IoStats IoScheduler::stats() const {
    Lock lk(m_);
    IoStats s = stats_;
    s.threaded = threaded_;
    return s;
}

// ---------------------------------------------------------------------------
// the service loop: the only code that talks to the backend

void IoScheduler::submit_queued_locked() {
    auto drain = [this](std::deque<uint64_t>& q) -> bool {   // false: backend full
        while (!q.empty()) {
            auto it = pending_.find(q.front());
            if (it == pending_.end() || it->second.state != State::Queued) {
                q.pop_front();   // settled or abandoned while queued
                continue;
            }
            InFlight& p = it->second;
            if (io_->submit(&p.req, 1) != 1) {
                // The op pool IS the queue-depth gate, so a full pool refuses.
                // A refusal with nothing in flight is not queue pressure: the
                // read can never be submitted, so it completes as failed (it
                // holds no DMA, so its memory is safe to free).
                if (io_->in_flight() != 0) return false;
                p.state = State::Done;
                p.ok = false;
                p.status = -1;
                q.pop_front();
                done_cv_.notify_all();
                continue;
            }
            q.pop_front();
            p.state = State::Submitted;
            ++submitted_;
            if (p.stage) {
                // TRUE queue depth, sampled where the drive sees it (see IoStats).
                const size_t nf = io_->in_flight();
                const uint64_t t = steady_ns();
                if (stats_.depth_t_last) stats_.depth_area_ns += stats_.depth_cur * (t - stats_.depth_t_last);
                else stats_.depth_t0 = t;
                stats_.depth_t_last = t;
                stats_.depth_cur = nf;
                stats_.depth_sum += nf;
                ++stats_.depth_n;
                if (nf > stats_.depth_max) stats_.depth_max = nf;
                if (nf < stats_.depth_min) stats_.depth_min = nf;
                if (nf < 4) ++stats_.depth_low;
                p.t_submit_ns = steady_ns();
            }
        }
        return true;
    };
    if (drain(urgent_)) drain(background_);
}

size_t IoScheduler::harvest(Lock& lk, size_t min_complete) {
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
    io::Completion comps[64];
    const auto pp0 = std::chrono::steady_clock::now();
    lk.unlock();   // never hold the lock while blocked on the device
    const size_t got = io_->poll(comps, 64, min_complete);
    lk.lock();
    stats_.in_pump_ns += ns_since(pp0);
    stats_.pump_harvested += got;

    InFlight* copies[64];
    size_t n_copy = 0;
    for (size_t i = 0; i < got; ++i) {
        auto it = pending_.find(comps[i].tag);
        if (it == pending_.end()) continue;   // already settled
        InFlight& p = it->second;
        --submitted_;
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
        if (p.ok && p.mem) copies[n_copy++] = &p;   // Done only once the bytes are home
        else               p.state = State::Done;
    }
    if (n_copy) {
        // Off-lock: the entries are not Done, so no waiter touches them, and
        // their addresses survive concurrent inserts (unordered_map).
        lk.unlock();
        uint64_t copy_ns = 0, copy_bytes = 0;
        for (size_t i = 0; i < n_copy; ++i) {
            InFlight& p = *copies[i];
            const auto c0 = std::chrono::steady_clock::now();
            std::memcpy(p.mem, p.stage + p.head, static_cast<size_t>(p.bytes));
            copy_ns += ns_since(c0);
            copy_bytes += p.bytes;
        }
        lk.lock();
        stats_.memcpy_ns += copy_ns;
        stats_.memcpy_bytes += copy_bytes;
        for (size_t i = 0; i < n_copy; ++i) copies[i]->state = State::Done;
    }
    if (got) done_cv_.notify_all();
    return got;
}

size_t IoScheduler::service(Lock& lk, size_t min_complete) {
    submit_queued_locked();
    if (submitted_ == 0) return 0;
    const size_t got = harvest(lk, min_complete);
    // Refill the slots just freed, so the drive is not idle until the next call.
    if (got) submit_queued_locked();
    return got;
}

void IoScheduler::run() {
    Lock lk(m_);
    for (;;) {
        work_cv_.wait(lk, [this] {
            return stop_ || (!dead_ && (!urgent_.empty() || !background_.empty() || submitted_ > 0));
        });
        if (dead_ || (stop_ && urgent_.empty() && background_.empty() && submitted_ == 0)) {
            if (stop_) return;
            continue;   // dead: nothing more will ever complete; wait for stop
        }
        submit_queued_locked();
        if (submitted_ == 0) continue;   // everything queued was refused outright
        if (harvest(lk, 1) == 0) {
            // The device stopped answering with reads outstanding. Waiters see
            // dead_ and take the leak paths (F2): nothing is freed under DMA.
            dead_ = true;
            done_cv_.notify_all();
            continue;
        }
        submit_queued_locked();
    }
}

void IoScheduler::kick() {
    if (threaded_) return;   // the I/O thread is already doing this, continuously
    Lock lk(m_);
    service(lk, 0);
}

bool IoScheduler::all_done(const std::vector<uint64_t>& tags) const {
    for (uint64_t t : tags) {
        auto it = pending_.find(t);
        if (it != pending_.end() && it->second.state != State::Done) return false;
    }
    return true;
}

bool IoScheduler::wait_all(Lock& lk, const std::vector<uint64_t>& tags) {
    while (!all_done(tags)) {
        if (threaded_) {
            if (dead_) return false;
            done_cv_.wait(lk);
        } else if (service(lk, 1) == 0) {
            return false;
        }
    }
    return true;
}

void IoScheduler::take_locked(uint64_t tag, uint8_t** stage, uint32_t* span, bool* ok) {
    *stage = nullptr;
    *span = 0;
    auto it = pending_.find(tag);
    if (it == pending_.end()) { *ok = false; return; }
    *ok = it->second.ok;
    *stage = it->second.stage;
    *span = it->second.span;
    pending_.erase(it);
}

// ---------------------------------------------------------------------------
// submission

uint64_t IoScheduler::enqueue(InFlight entry, IoPriority prio) {
    Lock lk(m_);
    const uint64_t tag = next_tag_++;
    entry.req.tag = tag;
    entry.state = State::Queued;
    pending_[tag] = entry;
    (prio == IoPriority::Urgent ? urgent_ : background_).push_back(tag);
    if (threaded_) {
        lk.unlock();
        work_cv_.notify_one();
    } else {
        // Straight to the device when it has room, exactly as a direct submit did.
        submit_queued_locked();
    }
    return tag;
}

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
    //
    // Allocated here, at enqueue, on the caller's thread. Staging is only ever
    // freed by its owner (settle/discard), never on completion, so allocating a
    // whole batch before its reads reach the device performs exactly the ledger
    // operations, in exactly the order, that allocating each at submit did.
    const auto sa0 = std::chrono::steady_clock::now();
    uint8_t* stage = static_cast<uint8_t*>(
        mem_.alloc(mem::Category::IoStaging, span, al.memory));
    const uint64_t alloc_ns = ns_since(sa0);
    if (!stage) {
        // T23: an earlier version pumped completions here hoping "completed
        // reads still holding staging" would free room. They cannot: a
        // completion only marks its entry done -- staging is freed by each
        // tag's OWNER (settle/discard). Evicting cache is the one lever that
        // actually moves the sum.
        if (reclaimer_ && reclaimer_->reclaim(span)) {
            stage = static_cast<uint8_t*>(
                mem_.alloc(mem::Category::IoStaging, span, al.memory));
        }
    }
    {
        Lock lk(m_);
        stats_.stage_alloc_ns += alloc_ns;
        ++stats_.stage_alloc_n;
    }
    if (!stage) return 0;

    InFlight e;
    e.req = io::ReadRequest{f, lo, static_cast<uint32_t>(span), stage, 0};
    e.stage = stage;
    e.head = head;
    e.span = static_cast<uint32_t>(span);
    e.mem = dst;
    e.bytes = bytes;
    return enqueue(e, IoPriority::Urgent);
}

bool IoScheduler::submit_exact(const Source& s, uint8_t* dst, uint64_t bytes,
                               std::vector<uint64_t>* tags) {
    if (s.shard < 0 || static_cast<size_t>(s.shard) >= shards_.size()) return false;
    const io::FileId f = shards_[static_cast<size_t>(s.shard)];
    const uint64_t a = io_->alignment(f).max();
    const uint64_t end = s.offset + bytes;
    const uint64_t lo = (s.offset + a - 1) / a * a;   // first aligned byte inside
    const uint64_t hi = end / a * a;                   // aligned end inside
    uint8_t* mid = dst + (lo - s.offset);
    const bool direct = hi > lo && hi - lo <= UINT32_MAX &&
                        reinterpret_cast<uintptr_t>(mid) % a == 0;
    if (!direct) {
        const uint64_t t = submit_staged(s, dst, bytes);
        if (t == 0) return false;
        tags->push_back(t);
        Lock lk(m_);
        ++stats_.exact_staged;
        return true;
    }
    // The middle, straight into place: no staging, no copy.
    tags->push_back(submit_unstaged(f, lo, static_cast<uint32_t>(hi - lo), mid, 0, hi - lo));
    // The edges (each under one alignment unit), through a bounce buffer.
    if (lo > s.offset) {
        const uint64_t t = submit_staged(s, dst, lo - s.offset);
        if (t == 0) return false;
        tags->push_back(t);
    }
    if (hi < end) {
        Source tail = s;
        tail.offset = hi;
        const uint64_t t = submit_staged(tail, dst + (hi - s.offset), end - hi);
        if (t == 0) return false;
        tags->push_back(t);
    }
    Lock lk(m_);
    ++stats_.exact_direct;
    stats_.exact_direct_bytes += hi - lo;
    return true;
}

uint64_t IoScheduler::submit_unstaged(io::FileId f, uint64_t file_offset, uint32_t len,
                                      void* dst, uint64_t head, uint64_t bytes,
                                      IoPriority prio) {
    // stage and mem both null: nothing to copy on completion, nothing to free.
    InFlight e;
    e.req = io::ReadRequest{f, file_offset, len, dst, 0};
    e.head = head;
    e.span = len;
    e.bytes = bytes;
    return enqueue(e, prio);
}

void IoScheduler::discard(uint64_t tag) {
    uint8_t* stage = nullptr;
    uint32_t span = 0;
    bool ok = false;
    {
        Lock lk(m_);
        take_locked(tag, &stage, &span, &ok);
    }
    mem_.free(mem::Category::IoStaging, stage, span);
}

bool IoScheduler::settle(std::vector<uint64_t>& tags, bool* backend_dead) {
    if (backend_dead) *backend_dead = false;
    std::vector<std::pair<uint8_t*, uint32_t>> frees;
    bool ok = true;
    {
        Lock lk(m_);
        if (!wait_all(lk, tags)) {
            // F2 (same shape as read_batch): a dead backend with reads
            // outstanding must LEAK the staging, never free under DMA.
            // I1: THE line H15 forgot -- without it the out-param stayed false,
            // both dtor consumers computed constants, and the teardown guard
            // was disarmed in the direction of always-free. A fix that adds a
            // flag and never sets it is worse than no fix: it retires the
            // FATAL message that would have said so.
            if (backend_dead) *backend_dead = true;
            log_leak(tags.size());
            for (uint64_t t : tags) pending_.erase(t);
            tags.clear();
            return false;
        }
        for (uint64_t t : tags) {
            uint8_t* stage; uint32_t span; bool tok;
            take_locked(t, &stage, &span, &tok);
            if (!tok) ok = false;
            frees.emplace_back(stage, span);
        }
    }
    for (const auto& f : frees) mem_.free(mem::Category::IoStaging, f.first, f.second);
    tags.clear();
    return ok;
}

// ---------------------------------------------------------------------------
// whole reads

bool IoScheduler::read_batch(const std::vector<Slice>& slices) {
    if (slices.empty()) return true;
    {
        Lock lk(m_);
        ++stats_.io_batches;
        stats_.io_slices += slices.size();
        if (slices.size() > stats_.io_batch_max) stats_.io_batch_max = slices.size();
    }

    std::vector<uint64_t> tags;
    tags.reserve(slices.size());
    bool failed = false;

    // Every read enqueued before any is waited on, so the drive sees the whole
    // batch as a deep queue.
    for (const Slice& sl : slices) {
        if (!submit_exact(sl.src, sl.dst, sl.len, &tags)) { failed = true; break; }
    }

    // F2: settle ALL enqueued tags UNCONDITIONALLY before any discard --
    // success or failure, every submitted read has DMA outstanding into its
    // staging buffer, and discard() frees that buffer for immediate reuse.
    // The old shape skipped this wait when `failed` was set mid-submission,
    // freeing under DMA (the no-cancel contract, storage.h and D8; T23's
    // deletion of submit_one's drain loop made the path reachable). A failed
    // batch still waits for its own reads; only the RESULT is failed.
    //
    // A dead backend with reads outstanding: freeing under possible DMA trades a
    // loud failure for silent corruption, so settle() LEAKS the staging
    // deliberately -- the bytes stay charged to IoStaging forever, and the
    // ledger honestly shows the cost of a dead backend.
    if (!settle(tags)) failed = true;
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
    // and the tag handling are exactly read_exact's. The I/O thread never reads
    // the shard table (requests carry their FileId), so this is safe either way.
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
                         in_flight(), io_->max_in_flight());
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

    // Queued like any other urgent read: a full device queue delays it, never
    // refuses it. A read the backend genuinely refuses completes as failed.
    const uint64_t tag = submit_unstaged(f, lo, static_cast<uint32_t>(span), base, head, bytes);
    const std::vector<uint64_t> one{tag};
    bool ok = false;
    uint64_t got = 0;
    int status = 0;
    bool found = false;
    {
        Lock lk(m_);
        if (!wait_all(lk, one)) {
            // Pass-4b: a dead backend here means the read may STILL be writing into
            // `base` -- fabricating done/ok and freeing below was the third
            // free-under-DMA site of the F2 class (the first two were fixed, this
            // one was not enumerated). Same remedy: leak deliberately and loudly.
            std::fprintf(stderr,
                         "[dray] FATAL: backend dead with a direct read outstanding; "
                         "leaking its staging (no-cancel contract)\n");
            pending_.erase(tag);
            return nullptr;
        }
        auto it = pending_.find(tag);
        found = it != pending_.end();
        if (found) { ok = it->second.ok; got = it->second.got; status = it->second.status; }
        pending_.erase(tag);
    }
    if (!ok && failures_ < 8) {
        // Name the reason. "read failed" without the numbers has cost hours on this
        // codebase every time; status and the short-read arithmetic identify it at
        // a glance.
        std::fprintf(stderr,
            "[dray] DIRECT READ FAIL shard=%d off=%llu lo=%llu head=%llu bytes=%llu "
            "span=%u got=%lld status=%d\n",
            s.shard, (unsigned long long)s.offset, (unsigned long long)lo,
            (unsigned long long)head, (unsigned long long)bytes, (unsigned)span,
            found ? (long long)got : -1LL, found ? status : -1);
        std::fflush(stderr);
    }
    if (!ok) { mem_.free(cat, base, span); return nullptr; }

    *out_alloc = span;
    *out_head  = static_cast<uint32_t>(head);
    return base;
}

// ---------------------------------------------------------------------------

void IoStats::append(std::ostream& o) const {
    o << ", io " << io_batches << " batches/" << io_slices
      << " slices (max " << io_batch_max << "), " << exact_direct << " direct reads ("
      << exact_direct_bytes / 1000000 << " MB in place), " << exact_staged << " staged, qdepth mean "
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
          << "; " << (threaded ? "I/O thread" : "compute thread") << " polls " << pump_calls
          << " mean gap " << (pump_calls > 1 ? pump_gap_us / (pump_calls - 1) : 0) << "us, "
          << (pump_calls ? pump_harvested / pump_calls : 0) << " harvested each"
          << "; gap buckets <100us " << gap_bucket[0] << " <1ms " << gap_bucket[1]
          << " <5ms " << gap_bucket[2] << " <20ms " << gap_bucket[3]
          << " longer " << gap_bucket[4] << ", max " << gap_max_us << "us, "
          << (gap_long_us / 1000)
          << (threaded ? "ms IDLE in gaps over 5ms (nothing queued: the engine asked for nothing), "
                       : "ms total spent in gaps over 5ms (completions uncollected), ")
          << (in_pump_ns / 1000000) << "ms BLOCKED INSIDE poll"
          << "; staging memcpy " << (memcpy_ns / 1000000) << "ms for "
          << (memcpy_bytes / 1000000) << " MB on the "
          << (threaded ? "I/O thread" : "compute thread")
          << ", staging alloc " << (stage_alloc_ns / 1000000) << "ms over "
          << stage_alloc_n << " reads";
    }
}

}  // namespace dray::backend
