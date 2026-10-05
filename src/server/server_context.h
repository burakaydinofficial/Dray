// What every route and the job worker share. Shared ownership, so a handler
// that outlives serve_main's locals (an SSE provider mid-response) can never
// reach freed state.
#pragma once

#include <memory>

namespace dray::engine { class Engine; class Scheduler; class RamKvStore; }

namespace dray::server {

class JobRegistry;
class JobStore;

struct ServerContext {
    // Owned by serve_main; outlives the listener. Read-only from routes except
    // through the scheduler: generation, and anything that reads the streamer's
    // live state, happens on the scheduler's thread (Scheduler::post).
    engine::Engine* engine = nullptr;
    // Every generation -- chat, streaming, background jobs -- is a request here.
    std::shared_ptr<engine::Scheduler> sched;
    // Kept and parked conversations; read only on the scheduler's thread
    // (Scheduler::post). Owned by serve_main.
    const engine::RamKvStore* kv = nullptr;
    std::shared_ptr<JobRegistry> jobs;
    std::shared_ptr<JobStore> store;   // null without --jobs-dir
    // Requests (running + waiting) past which a foreground request is told to
    // retry (503) instead of holding an HTTP thread: the pool minus headroom
    // for /health and the job routes.
    int foreground_limit = 1;
};

}  // namespace dray::server
