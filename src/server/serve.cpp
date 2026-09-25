// serve_main: the OpenAI-compatible server, assembled from its parts.
//
//   RequestGuard   pre-routing admission policy (loopback, auth, body bounds)
//   Engine         the model, one generation at a time behind `gate`
//   JobRegistry    background jobs in memory
//   JobStore       their sidecars and checkpoints under --jobs-dir
//   JobWorker      drains the registry
//   http_routes    the protocol surface
//   StopWatcher    turns a stop request into the drain below
//
// Admission is SERIALIZED (Invariant 7): one generation at a time, a mutex held
// for the request's whole lifetime. Batching past ~8 would dissolve the
// streaming premise, and v1 does not batch at all.

#include "server/serve.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

#include "httplib.h"
#include "json.hpp"

#include "engine/engine.h"
#include "server/http_routes.h"
#include "server/job_registry.h"
#include "server/job_store.h"
#include "server/job_worker.h"
#include "server/request_guard.h"
#include "server/server_context.h"
#include "server/shutdown.h"

namespace dray::server {

using nlohmann::json;

namespace {

// Swarm S12: bind FIRST. Discovering a taken port after minutes of model load,
// behind an already-printed success banner, was the worst order.
bool bind_loopback(httplib::Server& srv, int port) {
    // Windows makes SO_REUSEADDR (which httplib sets) permit DOUBLE binds, so
    // bind_to_port succeeds on a taken port and two servers share it silently.
    // A connect probe is the portable truth: if anything answers, refuse.
    {
        httplib::Client probe("127.0.0.1", port);
        probe.set_connection_timeout(0, 200000);
        auto pr = probe.Get("/health");
        if (pr || (pr.error() != httplib::Error::Connection &&
                   pr.error() != httplib::Error::ConnectionTimeout)) {
            std::fprintf(stderr, "serve: 127.0.0.1:%d already has a listener; nothing was loaded\n", port);
            return false;
        }
    }
    if (!srv.bind_to_port("127.0.0.1", port)) {
        std::fprintf(stderr, "serve: cannot bind 127.0.0.1:%d (in use, or no permission); nothing was loaded\n", port);
        return false;
    }
    return true;
}

}  // namespace

int serve_main(const engine::EngineConfig& cfg, int port, const std::string& jobs_dir,
               const std::string& api_key) {
    // Redirected stderr is FULLY buffered on Windows CRT; a wedged drain then
    // shows an empty log. Logs must be honest under redirection.
    setvbuf(stderr, nullptr, _IONBF, 0);

    httplib::Server srv;
    // Swarm S15: bound request bodies; httplib otherwise buffers arbitrarily
    // large payloads in RAM on a machine whose premise is a small footprint.
    srv.set_payload_max_length(32ull << 20);
    if (!bind_loopback(srv, port)) return 1;

    const RequestGuard guard(api_key);
    srv.set_pre_routing_handler([guard](const httplib::Request& req, httplib::Response& res) {
        return guard.check(req, res);
    });

    std::string err;
    std::unique_ptr<engine::Engine> engine = engine::Engine::open(cfg, &err);
    if (!engine) {
        std::fprintf(stderr, "serve: %s\n", err.c_str());
        return 1;
    }

    ServerContext ctx;
    ctx.engine = engine.get();
    ctx.gate = std::make_shared<std::mutex>();
    ctx.jobs = std::make_shared<JobRegistry>();
    if (!jobs_dir.empty()) {
        std::unique_ptr<JobStore> store = JobStore::open(jobs_dir, cfg.model_path, *engine, &err);
        if (!store) {
            std::fprintf(stderr, "serve: %s\n", err.c_str());
            return 1;
        }
        store->scan(*ctx.jobs);
        ctx.store = std::move(store);
    }

    JobWorker worker(ctx);

    // Swarm S8 backstop: anything still escaping a handler becomes a JSON 500
    // with a message, never a bodiless one.
    srv.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "unknown exception";
        try { if (ep) std::rethrow_exception(ep); } catch (const std::exception& ex) { what = ex.what(); } catch (...) {}
        res.status = 500;
        res.set_content(json{ { "error", { { "message", what } } } }.dump(), "application/json");
    });
    install_status_routes(srv, ctx);
    install_chat_completions(srv, ctx);
    install_job_routes(srv, ctx);
    install_admin_routes(srv);

    std::fprintf(stderr, "dray serving on http://127.0.0.1:%d (OpenAI-compatible; "
                         "admission serialized)\n", port);
    install_stop_handlers();
    StopWatcher watcher(srv, ctx.jobs);

    const bool ok = srv.listen_after_bind();
    std::fprintf(stderr, "serve: drain 1/4 listener returned\n");
    ctx.jobs->stop();
    worker.join();
    std::fprintf(stderr, "serve: drain 2/4 worker joined\n");
    watcher.join();
    std::fprintf(stderr, "serve: drain 3/4 watcher joined\n");
    engine.reset();
    std::fprintf(stderr, "serve: drain 4/4 engine closed; exiting\n");
    return ok ? 0 : 1;
}

}  // namespace dray::server
