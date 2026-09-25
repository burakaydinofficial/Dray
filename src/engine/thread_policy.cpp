#include "engine/thread_policy.h"

namespace dray::engine {

ThreadChoice choose_threads(int requested, int hardware) {
    ThreadChoice c;
    c.ceiling = hardware > kOwnThreads ? hardware - kOwnThreads : 1;
    // The default is the MEASURED optimum, not the core count.
    c.requested = requested > 0 ? requested : 4;
    int want = c.requested;
    if (want > c.ceiling) {
        c.clamped = true;
        want = c.ceiling;
    }
    c.decode = want;
    c.prefill = requested > 0 ? want : c.ceiling;
    return c;
}

}  // namespace dray::engine
