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
#include <unordered_map>

#include "backend/accounted_alloc.h"
#include "backend/residency_cache.h"
#include "backend/routing_skew.h"
#include "ggml.h"

namespace dray::backend {

class ExpertSlots {
public:
    // `disabled` is DRAY_NO_ESLOTS.
    ExpertSlots(AccountedAlloc& mem, ResidencyCache& cache, const RoutingSkew& skew,
                bool disabled)
        : mem_(mem), cache_(cache), skew_(skew), disabled_(disabled) {}
    ExpertSlots(const ExpertSlots&) = delete;
    ExpertSlots& operator=(const ExpertSlots&) = delete;

    void set_pool(uint64_t bytes) { pool_ = bytes; }

    // Expert `e` of fused tensor `w`, counted as a hit; null when not cached.
    const void* hit(const ggml_tensor* w, int32_t e);
    // Offers a freshly read expert for caching. Silently declines when it does
    // not earn a slot; never an error.
    void admit(const ggml_tensor* w, int32_t e, const void* src, uint64_t stride);
    // A new forward pass: the per-pass admission budget refills.
    void on_pass() { admits_pass_ = 0; }

    uint64_t hits() const { return hits_; }
    uint64_t bytes() const { return bytes_; }
    uint64_t pool() const { return pool_; }

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
    uint32_t admits_pass_ = 0;   // per-pass admission budget (the storm guard)
};

}  // namespace dray::backend
