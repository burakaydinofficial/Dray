#include "backend/stream_flags.h"

#include <cstdlib>

namespace dray::backend {

namespace {

bool on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}

}  // namespace

StreamFlags StreamFlags::from_env() {
    StreamFlags f;
    f.no_compact    = on("DRAY_NO_COMPACT");
    f.no_rowslice   = on("DRAY_NO_ROWSLICE");
    f.no_fuse       = on("DRAY_NO_FUSE");
    f.no_early      = on("DRAY_NO_EARLY");
    f.no_reuse      = on("DRAY_NO_REUSE");
    f.no_retain     = on("DRAY_NO_RETAIN");
    f.no_eslots     = on("DRAY_NO_ESLOTS");
    f.no_pool       = on("DRAY_NO_POOL");
    f.compact_all   = on("DRAY_COMPACT_ALL");
    f.fast_nodes    = on("DRAY_FAST_NODES");
    f.metal         = on("DRAY_METAL");
    f.trace         = on("DRAY_TRACE");
    f.trace_compact = on("DRAY_TRACE_COMPACT");
    f.io_stats      = on("DRAY_IO_STATS");
    if (const char* v = std::getenv("DRAY_IO_THREAD")) f.io_thread = v[0] != '0';
    if (const char* v = std::getenv("DRAY_RING_MB")) {
        f.ring_mb = std::strtoull(v, nullptr, 10);
    }
    return f;
}

}  // namespace dray::backend
