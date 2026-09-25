// Where a tensor's bytes actually live on disk. Filled from the residency Plan
// at init_tensor time, never from the bytes llama.cpp offers us.

#pragma once

#include <cstdint>

namespace dray::backend {

struct Source {
    int32_t  shard = -1;
    uint64_t offset = 0;   // within its shard
    uint64_t bytes = 0;
    bool     pinned = false;  // part of the mandatory floor: resident for the run
    // Kept so the floor can be charged to the right ledger category (router gates
    // are 1.5-2.4 GB on these models -- a user who cannot see that in the startup
    // report will think the cache is broken).
    bool     is_router_gate = false;
    uint64_t disk_stride = 0;  // repacked companion: per-expert stride on DISK
    // True when this tensor is never materialised WHOLE: routed experts go through
    // MUL_MAT_ID compaction, the token embedding through GET_ROWS row slicing. Only
    // the rest set the floor under a workable cache size.
    bool     sliceable = false;
    bool     routed = false;   // I2: expert-fused; whole only under NO_COMPACT
};

}  // namespace dray::backend
