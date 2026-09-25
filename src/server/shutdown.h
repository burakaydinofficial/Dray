// Graceful shutdown (S28). Signal handlers and POST /admin/shutdown only set a
// flag (async-signal-safe); the StopWatcher thread notices it, cancels every
// job and calls srv.stop(), which makes listen() return so the drain in
// serve_main runs: jobs stop, the worker joins, the engine closes.
#pragma once

#include <memory>
#include <thread>

#include "httplib.h"

namespace dray::server {

class JobRegistry;

// The process-wide stop flag.
bool stop_requested();
void request_stop();
void install_stop_handlers();   // SIGINT/SIGTERM, or the console handler on Windows

class StopWatcher {
public:
    StopWatcher(httplib::Server& srv, std::shared_ptr<JobRegistry> jobs);
    void join();

private:
    void run();
    httplib::Server& srv_;
    std::shared_ptr<JobRegistry> jobs_;
    std::thread thread_;
};

}  // namespace dray::server
