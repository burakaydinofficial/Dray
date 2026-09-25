#include "backend/expert_slots.h"

#include <cstdint>
#include <cstring>

namespace dray::backend {

const void* ExpertSlots::find(const ggml_tensor* w, int32_t e) const {
    auto t = slots_.find(w);
    if (t == slots_.end()) return nullptr;
    auto s = t->second.find(e);
    return s == t->second.end() ? nullptr : s->second.mem;
}

const void* ExpertSlots::hit(const ggml_tensor* w, int32_t e) {
    const void* p = find(w, e);
    if (p) ++hits_;
    return p;
}

// Admission is the retention rule again: only genuinely free room, never the
// churn reserve. History in git: free-room-only found zero by decode; UNBOUNDED
// eviction-assisted admission ratcheted in_use past budget. The pool bound is
// what makes make_room here compose: slots stay below the evictable mass.
void ExpertSlots::admit(const ggml_tensor* w, int32_t e, const void* src, uint64_t stride) {
    if (disabled_ || find(w, e)) return;
    const std::vector<uint32_t>* hist = skew_.counts(w);

    // Pool full: replace the coldest slot OF THIS TENSOR if the newcomer is
    // measurably hotter. Admit-only filled the pool with a cross-section and
    // measured h_routed=4% at 3.4% coverage -- rank-blind, exactly as predicted.
    // Per-tensor confinement keeps the victim scan tiny; the 2x hysteresis stops
    // equal-count churn. Eviction frees the SAME bytes the newcomer needs, so
    // the pool bound is preserved by construction.
    if (bytes_ + stride > pool_) {
        if (!hist) return;
        auto tv = slots_.find(w);
        if (tv == slots_.end() || tv->second.empty()) return;
        auto count_of = [hist](int32_t id) -> uint32_t {
            return (id >= 0 && static_cast<size_t>(id) < hist->size())
                       ? (*hist)[static_cast<size_t>(id)] : 0;
        };
        int32_t victim = -1;
        uint32_t vcount = UINT32_MAX;
        for (const auto& kv : tv->second) {
            const uint32_t c = count_of(kv.first);
            if (c < vcount) { vcount = c; victim = kv.first; }
        }
        if (victim < 0 || count_of(e) < vcount * 2) return;
        auto vs = tv->second.find(victim);
        mem_.free(mem::Category::ExpertCache, vs->second.mem, vs->second.bytes);
        bytes_ -= vs->second.bytes;
        tv->second.erase(vs);
    }
    // STORM GUARD: filling a 10 GB pool in one token cost 105 s (900 admissions,
    // each a make_room plus a 10 MB memcpy, on the decode path). Amortise: at
    // most 32 admissions per pass, and only experts the histogram has seen
    // before (the doorkeeper) -- under 75-88% skew the hot ones return within a
    // few tokens, the cold ones never earn a slot.
    if (admits_pass_ >= 32) return;
    if (!hist) return;
    if (e < 0 || static_cast<size_t>(e) >= hist->size()) return;
    if ((*hist)[static_cast<size_t>(e)] < 2) return;

    if (!cache_.fits_beside_churn(stride)) {
        if (!cache_.make_room(stride + cache_.churn_reserve())) return;
        if (!cache_.fits_beside_churn(stride)) return;
    }
    void* m = mem_.alloc(mem::Category::ExpertCache, stride, kHostAlign);
    if (!m) return;
    std::memcpy(m, src, static_cast<size_t>(stride));
    slots_[w][e] = Slot{m, stride};
    bytes_ += stride;
    ++admits_pass_;
}

void ExpertSlots::release_all() {
    for (auto& tw : slots_) {
        for (auto& kv : tw.second) {
            if (kv.second.mem) mem_.free_uncharged(kv.second.mem, kv.second.bytes);
        }
    }
    slots_.clear();
}

}  // namespace dray::backend
