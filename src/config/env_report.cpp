// See env_report.h.

#include "config/env_report.h"

#include <cstdlib>

namespace dray::config {

namespace {
// Every lever the code reads. tests/test_env_census.cpp enforces shape and
// output format (it cannot grep the tree; keeping this list complete is a
// same-commit discipline, stated in the header). Order is the report order.
const char* const kDrayEnvVars[] = {
    "DRAY_CKPT_EVERY",
    "DRAY_CKPT_SECONDS",
    "DRAY_COMPACT_ALL",
    "DRAY_METAL",
    "DRAY_NO_COMPACT",
    "DRAY_NO_EARLY",
    "DRAY_NO_ESLOTS",
    "DRAY_NO_FUSE",
    "DRAY_ROTATE_UNSAFE",
    "DRAY_FAST_NODES",
    "DRAY_NO_POOL",
    "DRAY_REPACK",
    "DRAY_NO_RETAIN",
    "DRAY_NO_REUSE",
    "DRAY_NO_ROWSLICE",
    "DRAY_RING_MB",
    "DRAY_SLOW_LOAD",
    "DRAY_FORCE_STREAM",
    "DRAY_IO_STATS",
    "DRAY_RESIDENT",
    "DRAY_TRACE",
    "DRAY_VULKAN",
    "DRAY_TRACE_COMPACT",
    "DRAY_URING",
    "DRAY_ALLOW_DEGRADED",
    "DRAY_ALLOW_TRUNCATED",
    "DRAY_RESUME_ATTEMPTS",
    "GGML_METAL_NO_RESIDENCY",
};
}  // namespace

std::string env_report() {
    std::string out;
    for (const char* name : kDrayEnvVars) {
        const char* v = std::getenv(name);
        if (v == nullptr || v[0] == '\0') continue;
        out += out.empty() ? "env: " : " ";
        out += name;
        out += "=";
        out += v;
    }
    return out;
}

const char* const* known_env_vars(size_t* count) {
    if (count != nullptr) {
        *count = sizeof(kDrayEnvVars) / sizeof(kDrayEnvVars[0]);
    }
    return kDrayEnvVars;
}

}  // namespace dray::config
