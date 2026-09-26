#include "backend/uncond_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace dray::backend {

void UncondRing::allocate(uint64_t bytes, uint32_t align) {
    arena_ = static_cast<uint8_t*>(p_.mem.alloc(mem::Category::IoStaging, bytes, align));
    if (arena_) bytes_ = bytes;
}

void UncondRing::record_consumption(ggml_tensor* t) {
    if (!arena_ || active_) return;
    const Source* s = p_.tensors.source_of(t);
    if (s && !s->pinned && !s->sliceable && t->ne[2] <= 1 && consumed_seen_.insert(t).second) {
        consumed_order_.push_back(t);
    }
}

// Everything the ring will stream, in graph order: bound, not floor, not
// row-sliced, not an expert tensor. Built at the SECOND forward pass, once
// pinning has claimed its prefix -- the split between pinned and streamed is
// settled by then and never changes.
void UncondRing::on_pass(uint32_t pass) {
    if (!arena_ || active_ || pass != 2) return;
    stream_list_.clear();
    for (ggml_tensor* t : consumed_order_) {
        const Source* s = p_.tensors.source_of(t);
        if (!s) continue;
        if (s->pinned || s->sliceable) continue;
        if (t->ne[2] > 1) continue;              // routed experts: compacted path
        // NO resident filter. The producer skips resident tensors per lap, which
        // adapts every cycle; filtering at build time forced dynamic admission to
        // exist, and admission APPENDED tensors out of consumption order -- the
        // producer then tripped its lap check chronically and the drive idled.
        // Measured: 19.9 -> 28.9 s/tok at n=16 from that one commit, bytes
        // identical. Order is the ring's load-bearing invariant; nothing may
        // insert into the list except this build, in consumed order.
        stream_list_.push_back(t);
    }
    cursor_ = 0;
    active_ = !stream_list_.empty();
    std::fprintf(stderr, "[dray] RING stream_list=%zu of consumed=%zu (pass %u)\n",
                 stream_list_.size(), consumed_order_.size(), pass);
    std::fflush(stderr);
}

void UncondRing::produce() {
    if (!arena_ || !active_) return;
    // S22: once the cache has absorbed the whole stream, every call walked the
    // entire stream list to discover nothing streamable -- thousands of no-op
    // scans per token. A fruitless FULL lap arms a 256-call backoff; after an
    // eviction the producer wakes at most one backoff late and streams the
    // same bytes it always would (difftest-gated: byte-identical).
    if (idle_backoff_ > 0) { --idle_backoff_; return; }
    ++calls_;

    // Free the front -- but NEVER a segment the node being computed right now is
    // using: its compute runs after this materialise returns. One-node batching
    // guarantees the next materialise call means the previous compute finished.
    //
    // RETENTION (consume != free): before a consumed segment's bytes are recycled,
    // promote them into the cache IF the budget has genuinely free room -- never
    // by evicting, and never into the churn reserve. Above the knee this converts
    // the ring's one pass over the churn set into residency, and the producer's
    // resident-check then stops re-streaming those tensors: the ring self-empties
    // as the cache absorbs its stream. At tight caps the room test fails on every
    // call and this is a no-op. One memcpy per tensor per RUN, not per token.
    while (!queue_.empty() && queue_.front().consumed && queue_.front().tags.empty() &&
           !p_.cache.is_protected(queue_.front().t)) {
        Segment& fr = queue_.front();
        // fr.reads_ok, not just fr.complete: promoting a segment whose reads
        // failed installs partly-unwritten arena bytes as a permanent cache hit.
        if (!no_retain_ && fr.t && fr.complete && fr.reads_ok && !p_.cache.contains(fr.t)) {
            const uint64_t nb = ggml_nbytes(fr.t);
            if (p_.cache.fits_beside_churn(nb)) {
                void* mem = p_.mem.alloc(mem::Category::ExpertCache, nb, kHostAlign);
                if (mem) {
                    std::memcpy(mem, arena_ + fr.pos + fr.head, static_cast<size_t>(nb));
                    Resident nr;
                    nr.mem = mem; nr.bytes = nb; nr.pinned = false;
                    nr.prio = Prio::Unconditional;
                    nr.cat = mem::Category::ExpertCache;
                    // Promoted bytes come from the ring, which holds the file
                    // layout. Stale repack traits here would tell the kernels to
                    // read it as interleaved.
                    fr.t->extra = nullptr;
                    p_.cache.add(fr.t, nr);
                    ++promotions_;
                }
            }
        }
        used_ -= fr.span;
        queue_.pop_front();
        ++pops_;
    }
    if (stream_list_.empty()) return;

    // Bound the read-ahead in chunks, counted HERE: chunks enqueued and not yet
    // settled by the consumer. It used to be the backend's in-flight count,
    // which moves with harvest timing and so made production -- and through
    // retention, residency and bytes -- depend on when completions happened to
    // be polled. This count moves only at production and consumption, so the
    // schedule is a function of the graph alone. (The old check never fired in
    // any measured run; the arena fills first. Expert batches no longer need
    // depth left for them: the scheduler submits urgent reads before ring reads.)
    const size_t cap_depth = p_.io.max_in_flight();
    const size_t depth_target = cap_depth > 16 ? cap_depth - 16 : cap_depth;

    size_t guard = 0;
    bool did_work = false;
    while (guard++ < stream_list_.size()) {
        // Give the scheduler the CPU first: it refills the device and routes any
        // completions, so the ring's window keeps moving between callbacks.
        p_.io.kick();
        if (live_chunks_ >= depth_target) { ++stop_[0]; break; }
        ggml_tensor* t = stream_list_[cursor_ % stream_list_.size()];

        // A full lap: the next tensor is already queued and unconsumed. Deep enough.
        bool queued = false;
        for (const auto& sg : queue_) {
            if (sg.t == t && !sg.consumed) { queued = true; break; }
        }
        if (queued) { ++stop_[1]; break; }

        const Source* src = p_.tensors.source_of(t);
        if (!src || p_.cache.contains(t)) { ++cursor_; continue; }

        // Widened by the GLOBAL max alignment, not the shard's own: the arena is
        // one address space with one placement granularity.
        const uint64_t nb    = ggml_nbytes(t);
        const uint32_t align = p_.io.align();
        const uint64_t lo    = (src->offset / align) * align;
        const uint32_t head  = static_cast<uint32_t>(src->offset - lo);
        const uint64_t span  = ((head + nb + align - 1) / align) * align;
        // Too big to pipeline (would monopolise the ring): the sync path handles it.
        if (span > bytes_ / 2) { ++cursor_; continue; }

        // Wrap: segments must be contiguous, so pad the tail and place at 0.
        if (head_off_ + span > bytes_) {
            const uint64_t pad = bytes_ - head_off_;
            if (used_ + pad + span > bytes_) { ++stop_[2]; break; }
            Segment ps;
            ps.pos = head_off_; ps.span = pad; ps.consumed = true;
            queue_.push_back(std::move(ps));
            used_ += pad;
            head_off_ = 0;
        }
        if (used_ + span > bytes_) { ++stop_[3]; break; }

        // Submit in <=8 MiB chunks: per calibration, moderate requests at depth
        // beat one giant request, and no single request occupies the drive long
        // enough to delay an expert batch.
        Segment sg;
        sg.t = t; sg.pos = head_off_; sg.span = span; sg.head = head;
        sg.complete = true;
        const io::FileId f = p_.io.file(src->shard);
        const uint64_t interior = head + nb;
        bool fail = false;
        for (uint64_t off = 0; off < span && !fail; off += (8ull << 20)) {
            const uint32_t clen =
                static_cast<uint32_t>(std::min<uint64_t>(8ull << 20, span - off));
            // What this chunk must actually deliver; the widened tail past EOF may
            // legally come up short.
            const uint64_t need =
                off >= interior ? 0 : std::min<uint64_t>(clen, interior - off);
            const uint64_t tag = p_.io.submit_unstaged(f, lo + off, clen, arena_ + sg.pos + off,
                                                       0, need, IoPriority::Background);
            if (tag == 0) { fail = true; sg.complete = false; }
            else sg.tags.push_back(tag);
        }
        live_chunks_ += sg.tags.size();
        if (fail && sg.tags.empty()) { ++stop_[4]; break; }
        used_ += span;
        head_off_ = (head_off_ + span) % bytes_;
        queue_.push_back(std::move(sg));
        did_work = true;
        ++cursor_;
        ++segs_;
        if (fail) { ++stop_[4]; break; }
    }
    // A complete lap that submitted nothing (everything resident or oversized):
    // sleep. Depth/lap/full breaks are NOT fruitless -- work is in flight.
    if (!did_work && guard > stream_list_.size()) {
        idle_backoff_ = 256;
    }
}

bool UncondRing::consume(ggml_tensor* t, uint64_t* streamed) {
    if (!arena_ || !active_) return false;
    for (auto& sg : queue_) {
        if (sg.t != t || sg.consumed) continue;
        // A partially submitted segment holds valid bytes only where chunks were
        // actually issued: serving it would be silent corruption. Settle what is
        // outstanding, mark consumed, and let the sync path read properly.
        // Three-arg settle: false can mean "a read landed with a bad status"
        // (safe, memory is ours again) or "the backend died with reads
        // OUTSTANDING" (the arena bytes are still a DMA target). Only the
        // out-param distinguishes them, and this site used to pass nullptr --
        // letting the producer recycle a segment the kernel could still write
        // (2026-08-24 audit).
        bool dead = false;
        const bool settled_ok = settle_segment(sg, &dead);
        if (dead) retire_dead_ring();
        const bool ok = settled_ok && sg.complete;
        sg.consumed = true;
        if (!settled_ok) sg.reads_ok = false;   // retention must not promote this
        if (!ok) return false;
        // Raw file bytes, NOT the interleaved layout a repack may have left
        // traits for. extra is set once at init_tensor and was never cleared, so
        // the repacked-matmul kernel could dispatch over non-interleaved data
        // (2026-08-24 audit). Clearing it here is free when repack is off.
        t->extra = nullptr;
        t->data = arena_ + sg.pos + sg.head;
        *streamed += ggml_nbytes(t);
        p_.hits.add_uncond(ggml_nbytes(t), ggml_nbytes(t));
        ++hits_;
        return true;
    }
    return false;
}

void UncondRing::retire_pending(ggml_tensor* t) {
    if (!active_) return;
    for (auto& sg : queue_) {
        if (sg.t == t && !sg.consumed) {
            // The bool was once discarded here, which is how a failed read
            // reached retention looking clean. And the dead-backend case must
            // retire the ring, not just this segment: outstanding DMA does not
            // care which segment the caller was touching.
            bool dead = false;
            if (!settle_segment(sg, &dead)) sg.reads_ok = false;
            if (dead) retire_dead_ring();
            sg.consumed = true;
            break;
        }
    }
}

bool UncondRing::settle_segment(Segment& sg, bool* dead) {
    live_chunks_ -= sg.tags.size();   // settle() clears the tags either way
    return p_.io.settle(sg.tags, dead);
}

void UncondRing::retire_dead_ring() {
    std::fprintf(stderr,
                 "[dray] FATAL: backend dead with ring reads outstanding; "
                 "ring disabled and its memory retired (no-cancel contract)\n");
    active_ = false;
    died_ = true;
}

void UncondRing::append(std::ostream& o) const {
    o << hits_ << " ring-fed (front="
      << (queue_.empty() ? "none" : (queue_.front().t ? queue_.front().t->name : "pad"))
      << (queue_.empty() ? "" : (queue_.front().consumed ? "/consumed" : "/UNCONSUMED"))
      << ", " << segs_ << " segs, " << pops_
      << " pops, " << calls_ << " calls; stops d=" << stop_[0]
      << " lap=" << stop_[1] << " wrapfull=" << stop_[2]
      << " full=" << stop_[3] << " refuse=" << stop_[4] << "), ";
}

void UncondRing::release_all() {
    bool dead = false;
    for (auto& sg : queue_) {
        bool d1 = false;
        settle_segment(sg, &d1);
        dead = dead || d1;
    }
    queue_.clear();
    // H15: leak only on a genuinely dead backend -- seen now, or earlier by a
    // consume/retire that already settled (and so cleared) the dying reads.
    if (arena_ && !dead && !died_) p_.mem.free_uncharged(arena_, bytes_);
    else if (arena_) std::fprintf(stderr, "[dray] FATAL: teardown leaking the ring "
                                          "(backend died with reads outstanding)\n");
}

}  // namespace dray::backend
