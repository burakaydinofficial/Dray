#include "backend/expert_slots.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace dray::backend {

ExpertSlots::ExpertSlots(AccountedAlloc& mem, ResidencyCache& cache, const RoutingSkew& skew,
                         bool disabled, double admit_byte_fraction)
    : mem_(mem), cache_(cache), skew_(skew), disabled_(disabled),
      // Integer from here on: the budget is compared exactly, never as a sum
      // of rounded fractions (ten reads at 10% must pay for exactly one).
      admit_ppm_(admit_byte_fraction <= 0 ? 0
                 : static_cast<uint64_t>(admit_byte_fraction * 1e6 + 0.5)) {}

void ExpertSlots::on_pass() {
    read_pass_ = 0;
    spent_pass_ = 0;
}

void ExpertSlots::append(std::ostream& o) const {
    o << ", eslot admissions " << admissions_ << " (" << admitted_bytes_ / 1000000000.0
      << " GB, " << admit_ns_ / 1000000 << "ms), " << declined_for_budget_
      << " declined by the byte budget";
    o << ", " << released_ << " slots released under pressure (" << released_bytes_ / 1000000000.0
      << " GB)";
}

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
    // Every offer is an expert that was just read from disk, so it earns its
    // share of admission budget whether or not it is kept.
    read_pass_ += stride;
    if (disabled_ || find(w, e)) return;
    const std::vector<uint32_t>* hist = skew_.counts(w);

    // Only experts the histogram has seen before (the doorkeeper): under 75-88%
    // skew the hot ones return within a few tokens, the cold ones never earn a
    // slot.
    if (!hist) return;
    if (e < 0 || static_cast<size_t>(e) >= hist->size()) return;
    if ((*hist)[static_cast<size_t>(e)] < 2) return;

    // STORM GUARD, as a byte budget: filling a 10 GB pool in one token once
    // cost 105 s (900 admissions, each a make_room plus a 10 MB memcpy, on the
    // decode path). The cost is copy work, proportional to bytes admitted, so
    // admission may copy at most admit_ppm_ per million of the bytes this pass
    // has read. It replaces a fixed 32 admissions per pass, a count that only
    // meant something at K3's ~10 MB slots -- at Qwen3.8-Flash-Next's ~1 MB it
    // admitted ~32 MB per token and a 31 GB pool never filled. Bytes, not time:
    // a clock-driven budget made bytes read differ between identical runs.
    if ((spent_pass_ + stride) * 1000000ull > read_pass_ * admit_ppm_) {
        ++declined_for_budget_;
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed {    // REPORT ONLY: the time never feeds a decision
        uint64_t* ns; std::chrono::steady_clock::time_point t0;
        ~Timed() {
            *ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
        }
    } timed{&admit_ns_, t0};

    // Pool full: replace the coldest slot OF THIS TENSOR if the newcomer is
    // measurably hotter. Admit-only filled the pool with a cross-section and
    // measured h_routed=4% at 3.4% coverage -- rank-blind, exactly as predicted.
    // Per-tensor confinement keeps the victim scan tiny; the 2x hysteresis stops
    // equal-count churn. Eviction frees the SAME bytes the newcomer needs, so
    // the pool bound is preserved by construction.
    if (bytes_ + stride > pool_) {
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

    if (!cache_.fits_beside_churn(stride)) {
        admitting_ = true;   // make_room must not free slots to make room for a slot
        const bool room = cache_.make_room(stride + cache_.churn_reserve());
        admitting_ = false;
        if (!room || !cache_.fits_beside_churn(stride)) return;
    }
    void* m = mem_.alloc(mem::Category::ExpertCache, stride, kHostAlign);
    if (!m) return;
    std::memcpy(m, src, static_cast<size_t>(stride));
    slots_[w][e] = Slot{m, stride};
    bytes_ += stride;
    spent_pass_ += stride;
    admitted_bytes_ += stride;
    ++admissions_;
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

namespace dray::backend {

uint64_t ExpertSlots::release(uint64_t bytes) {
    if (admitting_ || bytes == 0 || slots_.empty()) return 0;
    // Every slot with its pick count; the coldest go first. A pass over the
    // pool is cheap next to what pressure costs, and pressure is rare.
    struct Victim { uint32_t count; const ggml_tensor* w; int32_t e; };
    std::vector<Victim> all;
    for (const auto& tw : slots_) {
        const std::vector<uint32_t>* hist = skew_.counts(tw.first);
        for (const auto& s : tw.second) {
            const uint32_t c = (hist && s.first >= 0 && static_cast<size_t>(s.first) < hist->size())
                                   ? (*hist)[static_cast<size_t>(s.first)] : 0;
            all.push_back({ c, tw.first, s.first });
        }
    }
    std::sort(all.begin(), all.end(), [](const Victim& a, const Victim& b) { return a.count < b.count; });
    uint64_t freed = 0;
    for (const Victim& v : all) {
        if (freed >= bytes) break;
        auto tv = slots_.find(v.w);
        auto sv = tv->second.find(v.e);
        mem_.free(mem::Category::ExpertCache, sv->second.mem, sv->second.bytes);
        freed += sv->second.bytes;
        bytes_ -= sv->second.bytes;
        tv->second.erase(sv);
        if (tv->second.empty()) slots_.erase(tv);
        ++released_;
    }
    released_bytes_ += freed;
    return freed;
}

}  // namespace dray::backend
