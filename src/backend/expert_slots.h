// The frequency expert cache: single experts kept in RAM because the router
// keeps picking them (owner design; evidence: the top 10% of experts take
// 36%/75%/88% of picks on testbed/K3/GLM, see RoutingSkew).
//
// Per-expert slots, separate from compacted regions and from the LRU. A hit is a
// memcpy into the region being built instead of a disk read.
//
// THE PLAN-TIME LINE ITEM (the law all three speculative features derive from):
// slots may hold at most the past-knee surplus -- budget minus the unconditional
// stream, churn and ring -- sized once at construction (StreamBudget). A pinned
// unconditional byte saves exactly one read per token, so pinning outranks slots
// per byte until the whole stream fits; below the knee the pool is zero and the
// cache is silent. Bounded this way, make_room during admission is SAFE: the
// evictable mass always covers the request, and slots can never ratchet in_use
// past budget.

#pragma once

#include <cstdint>
#include <ostream>
#include <unordered_map>

#include "backend/accounted_alloc.h"
#include "backend/residency_cache.h"
#include "backend/routing_skew.h"
#include "ggml.h"

namespace dray::backend {

class ExpertSlots final : public SpillSource {
public:
    // `disabled` is DRAY_NO_ESLOTS. `admit_byte_fraction`: see Config.
    ExpertSlots(AccountedAlloc& mem, ResidencyCache& cache, const RoutingSkew& skew,
                bool disabled, double admit_byte_fraction);
    ExpertSlots(const ExpertSlots&) = delete;
    ExpertSlots& operator=(const ExpertSlots&) = delete;

    void set_pool(uint64_t bytes) { pool_ = bytes; }

    // Expert `e` of fused tensor `w`, counted as a hit; null when not cached.
    const void* hit(const ggml_tensor* w, int32_t e);
    // Offers an expert that was JUST READ from disk (every caller passes a
    // miss). The read earns admit_byte_fraction of its size in admission
    // budget for this pass, whether or not this expert is kept; an admission
    // spends its size. Silently declines when the expert does not earn a slot
    // or the budget does not cover it; never an error.
    void admit(const ggml_tensor* w, int32_t e, const void* src, uint64_t stride);
    // A new forward pass: the admission budget starts over.
    void on_pass();

    uint64_t hits() const { return hits_; }
    uint64_t bytes() const { return bytes_; }
    uint64_t pool() const { return pool_; }
    uint64_t admissions() const { return admissions_; }
    uint64_t declined_for_budget() const { return declined_for_budget_; }
    uint64_t released() const { return released_; }
    uint64_t released_bytes() const { return released_bytes_; }
    // Under memory pressure (ResidencyCache::make_room): frees the coldest slots
    // -- fewest router picks first -- until `bytes` are freed. While admitting,
    // nothing: admission never evicts slots to make room for a slot (its own
    // replacement rule decides that).
    uint64_t release(uint64_t bytes) override;
    // ", eslot admissions N (X GB, Y ms), M declined by the byte budget"
    void append(std::ostream& o) const;

    // Teardown: returns every slot's memory without crediting the ledger.
    void release_all();

private:
    // Self-sized: at teardown the ggml tensors are already gone, so nb[2] must
    // not be consulted.
    struct Slot { void* mem = nullptr; uint64_t bytes = 0; };

    const void* find(const ggml_tensor* w, int32_t e) const;

    AccountedAlloc&       mem_;
    ResidencyCache&       cache_;
    const RoutingSkew&    skew_;
    const bool            disabled_;
    std::unordered_map<const ggml_tensor*, std::unordered_map<int32_t, Slot>> slots_;
    uint64_t pool_ = 0;
    uint64_t bytes_ = 0;
    uint64_t hits_ = 0;

    // The admission budget (the storm guard): within a pass, slots may take at
    // most admit_ppm_ per million of the bytes the pass has read. Bytes, not
    // time: the decision must not depend on the clock, or bytes read stop
    // being reproducible run to run.
    const uint64_t admit_ppm_;          // admit_byte_fraction, in parts per million
    uint64_t     read_pass_ = 0;        // bytes offered (just read) this pass
    uint64_t     spent_pass_ = 0;       // bytes admitted this pass
    uint64_t     admissions_ = 0;
    uint64_t     admitted_bytes_ = 0;
    uint64_t     declined_for_budget_ = 0;
    uint64_t     admit_ns_ = 0;         // REPORT ONLY: never feeds a decision
    uint64_t     released_ = 0;
    uint64_t     released_bytes_ = 0;
    bool         admitting_ = false;
};

}  // namespace dray::backend
