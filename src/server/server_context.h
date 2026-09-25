// What every route and the job worker share. Shared ownership, so a handler
// that outlives serve_main's locals (an SSE provider mid-response) can never
// reach freed state.
#pragma once

#include <memory>
#include <mutex>

namespace dray::engine { class Engine; }

namespace dray::server {

class JobRegistry;
class JobStore;

struct ServerContext {
    engine::Engine* engine = nullptr;          // owned by serve_main; outlives the listener
    std::shared_ptr<std::mutex> gate;          // one generation at a time (Invariant 7)
    std::shared_ptr<JobRegistry> jobs;
    std::shared_ptr<JobStore> store;           // null without --jobs-dir
};

}  // namespace dray::server
