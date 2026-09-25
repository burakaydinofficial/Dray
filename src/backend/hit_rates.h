// The two hit rates, Invariant 6. NEVER one symbol.
//
//   h_routed = routed bytes served from RAM / routed bytes needed. Feeds the cost
//              model; a whole-file denominator would flatter thrashing configs.
//   h_bytes  = ALL active bytes served from RAM / all needed. What the
//              RAM-vs-disk regime crossover requires.
//
// They diverge sharply: at the knee h_routed is 0 while h_bytes is ~87-90%. A
// zero denominator prints "unknown", never a default.

#pragma once

#include <cstdint>
#include <ostream>

#include "ggml.h"

namespace dray::backend {

struct HitRates {
    uint64_t routed_needed = 0, routed_read = 0;
    uint64_t uncond_needed = 0, uncond_read = 0;

    void add_routed(uint64_t needed, uint64_t read) { routed_needed += needed; routed_read += read; }
    void add_uncond(uint64_t needed, uint64_t read) { uncond_needed += needed; uncond_read += read; }
    // By the tensor's SHAPE: expert-fused (ne[2] > 1) is routed, anything else
    // unconditional. H8: a routed tensor's hits and misses must land in ONE
    // class, whichever path served them, or h_routed lies.
    void add_for(const ggml_tensor* t, uint64_t needed, uint64_t read) {
        if (t->ne[2] > 1) add_routed(needed, read);
        else              add_uncond(needed, read);
    }

    // ", h_routed=N% h_bytes=M%" (or "unknown").
    void append(std::ostream& o) const;
};

}  // namespace dray::backend
