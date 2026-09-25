// Routing-skew histogram: how often each expert of each fused expert tensor is
// picked at decode.
//
// Whether a frequency-kept expert cache can beat plain recency is a property of
// THIS distribution -- measured, never assumed (the model was trained for
// balance; Invariant 8). Measured shares for the top 10% of experts: 36% on
// testbed, 75% on K3, 88% on GLM. ExpertSlots reads it to decide what to keep.

#pragma once

#include <cstdint>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "ggml.h"

namespace dray::backend {

class RoutingSkew {
public:
    // One decode pick of the experts in `uniq` for fused tensor `w`.
    void record(const ggml_tensor* w, const std::vector<int32_t>& uniq, int64_t n_expert);
    // Per-expert pick counts for `w`; null before `w`'s first recorded pick.
    const std::vector<uint32_t>* counts(const ggml_tensor* w) const;

    // ", skew: top-10% experts take N% of picks (...)" -- nothing before any pick.
    void append(std::ostream& o) const;

private:
    std::unordered_map<const ggml_tensor*, std::vector<uint32_t>> hist_;
};

}  // namespace dray::backend
