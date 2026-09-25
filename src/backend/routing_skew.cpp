#include "backend/routing_skew.h"

#include <algorithm>

namespace dray::backend {

void RoutingSkew::record(const ggml_tensor* w, const std::vector<int32_t>& uniq,
                         int64_t n_expert) {
    auto& h = hist_[w];
    if (h.empty()) h.resize(static_cast<size_t>(n_expert), 0);
    for (int32_t e : uniq) {
        if (e >= 0 && e < n_expert) ++h[static_cast<size_t>(e)];
    }
}

const std::vector<uint32_t>* RoutingSkew::counts(const ggml_tensor* w) const {
    auto it = hist_.find(w);
    return it == hist_.end() ? nullptr : &it->second;
}

// The share of decode picks landing on each tensor's top-10% most-picked
// experts, averaged. Uniform routing scores ~0.10 plus sparsity noise; a
// frequency cache only earns its complexity well above that.
void RoutingSkew::append(std::ostream& o) const {
    if (hist_.empty()) return;
    double share_sum = 0.0;
    uint64_t picks_sum = 0;
    size_t counted = 0;
    for (const auto& kv : hist_) {
        const auto& h = kv.second;
        uint64_t tot = 0;
        for (uint32_t c : h) tot += c;
        if (tot == 0) continue;
        std::vector<uint32_t> s(h.begin(), h.end());
        std::sort(s.begin(), s.end(), [](uint32_t a, uint32_t b) { return a > b; });
        const size_t top = s.size() >= 10 ? s.size() / 10 : 1;
        uint64_t t10 = 0;
        for (size_t i = 0; i < top; ++i) t10 += s[i];
        share_sum += static_cast<double>(t10) / static_cast<double>(tot);
        picks_sum += tot;
        ++counted;
    }
    if (counted) {
        o << ", skew: top-10% experts take "
          << static_cast<int>(100.0 * share_sum / counted + 0.5)
          << "% of picks (" << (picks_sum / counted) << " picks/tensor, "
          << counted << " tensors)";
    }
}

}  // namespace dray::backend
