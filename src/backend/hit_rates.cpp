#include "backend/hit_rates.h"

namespace dray::backend {

void HitRates::append(std::ostream& o) const {
    o << ", h_routed=";
    if (routed_needed) {
        o << static_cast<int>(100.0 * (1.0 - static_cast<double>(routed_read) /
                                             static_cast<double>(routed_needed)) + 0.5) << "%";
    } else {
        o << "unknown";
    }
    o << " h_bytes=";
    if (routed_needed + uncond_needed) {
        const double read_t = static_cast<double>(routed_read + uncond_read);
        const double need_t = static_cast<double>(routed_needed + uncond_needed);
        o << static_cast<int>(100.0 * (1.0 - read_t / need_t) + 0.5) << "%";
    } else {
        o << "unknown";
    }
}

}  // namespace dray::backend
