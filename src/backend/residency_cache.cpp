#include "backend/residency_cache.h"

#include <algorithm>
#include <chrono>
#include <iterator>

namespace dray::backend {

uint64_t ResidencyCache::static_allowance() const {
    const uint64_t b = mem_.cache_budget();
    return b > churn_reserve_ ? b - churn_reserve_ : 0;
}

bool ResidencyCache::claim_static(uint64_t bytes) {
    if (static_used_ + bytes > static_allowance()) return false;
    static_used_ += bytes;
    return true;
}

Resident* ResidencyCache::find(const ggml_tensor* t) {
    auto it = entries_.find(t);
    return it == entries_.end() ? nullptr : &it->second;
}

void ResidencyCache::add(const ggml_tensor* t, const Resident& r) {
    entries_[t] = r;
    lru_.push_front(t);
}

void ResidencyCache::drop(const ggml_tensor* t) {
    auto it = entries_.find(t);
    if (it == entries_.end()) return;
    mem_.free(it->second.cat, it->second.mem, it->second.bytes);
    entries_.erase(it);
    lru_.remove(t);
}

bool ResidencyCache::is_protected(const ggml_tensor* t) const {
    return std::find(current_.begin(), current_.end(), t) != current_.end();
}

void* ResidencyCache::take_region(uint64_t bytes, uint32_t align) {
    for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
        auto r = entries_.find(*it);
        if (r == entries_.end()) continue;
        const Resident& e = r->second;
        if (e.pinned || e.prio != Prio::RoutedExpert || e.bytes != bytes ||
            e.cat != mem::Category::ExpertCache || !e.mem) continue;
        if (align && reinterpret_cast<uintptr_t>(e.mem) % align != 0) continue;
        if (is_protected(*it)) continue;   // needed by the node being computed
        void* m = e.mem;
        const ggml_tensor* victim = *it;
        entries_.erase(r);
        lru_.erase(std::next(it).base());
        // As in make_room: back to the sentinel, never null.
        const_cast<ggml_tensor*>(victim)->data = poison_.sentinel();
        ++regions_reused_;
        return m;
    }
    return nullptr;
}

bool ResidencyCache::make_room(uint64_t need) {
    for (int pass = 0; pass < 2 && mem_.cache_used() + need > mem_.cache_budget(); ++pass) {
        const Prio evictable = (pass == 0) ? Prio::RoutedExpert : Prio::Unconditional;
        while (mem_.cache_used() + need > mem_.cache_budget() && !lru_.empty()) {
            bool evicted = false;
            for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
                auto r = entries_.find(*it);
                if (r == entries_.end()) continue;
                if (r->second.pinned) continue;
                if (r->second.prio != evictable) continue;   // lower-value class first
                if (is_protected(*it)) continue;  // needed by the node being computed
                const auto f0 = std::chrono::steady_clock::now();
                mem_.free(r->second.cat, r->second.mem, r->second.bytes);
                ns_evict_free_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - f0).count());
                ++evictions_;
                evicted_bytes_ += r->second.bytes;
                const ggml_tensor* victim = *it;
                entries_.erase(r);
                lru_.erase(std::next(it).base());
                // Back to the sentinel, not null: ggml asserts on a null data pointer
                // in several places, and an evicted tensor may still be inspected
                // before it is next materialised.
                const_cast<ggml_tensor*>(victim)->data = poison_.sentinel();
                evicted = true;
                break;
            }
            if (!evicted) break;   // nothing left in this class; try the next one
        }
    }
    return mem_.cache_used() + need <= mem_.cache_budget();
}

void ResidencyCache::demote_static_to_allowance() {
    while (static_used_ > static_allowance()) {
        bool demoted = false;
        for (auto& kv : entries_) {
            if (!kv.second.pinned) continue;
            const Source* s = tensors_.source_of(kv.first);
            if (!s) continue;           // NO-SOURCE entry: see below
            if (s->pinned) continue;    // floor: never
            // The two `continue`s above are load-bearing in different ways. A
            // floor tensor must never be demoted. A NO-SOURCE entry (created by
            // init_tensor for llama-owned tensors) is marked pinned but never
            // added to static_used and never entered the LRU -- demoting it used
            // to subtract bytes that were never added, clamping static_used to 0
            // after a few iterations while the REAL static pins survived
            // untouched. The loop then reported success with the over-claim
            // intact, which on a tight-RSS machine sends output.weight to the
            // poison path: the exact failure this function exists to prevent
            // (2026-08-24 audit).
            kv.second.pinned = false;
            static_used_ -= std::min(static_used_, kv.second.bytes);
            demoted = true;
            break;
        }
        if (!demoted) break;   // only floor remains; nothing more to give back
    }
}

void ResidencyCache::shrink_to_budget() {
    while (mem_.cache_used() > mem_.cache_budget() && !lru_.empty()) {
        if (!make_room(0)) break;
    }
}

void ResidencyCache::release_all() {
    for (auto& kv : entries_) {
        if (kv.second.mem) mem_.free_uncharged(kv.second.mem, kv.second.bytes);
    }
}

}  // namespace dray::backend
