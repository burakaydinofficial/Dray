#include "engine/thread_policy.h"

namespace dray::engine {

ThreadChoice choose_threads(int decode, int prefill, int reserved, int hardware) {
    ThreadChoice c;
    c.ceiling = hardware > reserved ? hardware - reserved : 1;
    c.requested = decode;
    c.decode = decode;
    if (c.decode > c.ceiling) {
        c.clamped = true;
        c.decode = c.ceiling;
    }
    c.prefill = prefill > 0 ? prefill : c.ceiling;
    if (c.prefill > c.ceiling) {
        c.clamped = true;
        c.prefill = c.ceiling;
    }
    return c;
}

}  // namespace dray::engine
