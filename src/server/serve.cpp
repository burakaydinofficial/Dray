// serve_main: the OpenAI-compatible server, assembled from its parts.
//
//   RequestGuard   pre-routing admission policy (loopback, auth, body bounds)
//   Engine         the model
//   Scheduler      every generation (chat, SSE, jobs) as a request; serve.max_parallel
//                  slots decode together, sharing each step's weight reads
//   JobRegistry    background jobs in memory
//   JobStore       their sidecars and checkpoints under --jobs-dir
//   JobWorker      drains the registry
//   http_routes    the protocol surface
//   StopWatcher    turns a stop request into the drain below
//
// Parallel by configuration (serve.max_parallel, --parallel): the planner funds
// that many slots inside --cap, and the scheduler runs them in one decode per
// step -- the measured throughput case for this engine (Qwen3.8-Flash-Next: 64
// sequences ~4 tok/s against ~1 for one). The default is 1.

#include "server/serve.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

#include "httplib.h"
#include "json.hpp"

#include "engine/engine.h"
#include "engine/llama_stepper.h"
#include "engine/ram_kv_store.h"
#include "engine/scheduler.h"
#include "mem/accountant.h"
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

// Parked conversations charged to the cap's ledger, but never beyond the
// memory the streamer can do without (the expert slot pool, which gives way:
// its slots are evictable). A refusal makes the pool drop its oldest entries.
class LedgerCharge final : public engine::MemoryCharge {
public:
    LedgerCharge(mem::Accountant& acct, uint64_t ceiling) : acct_(acct), ceiling_(ceiling) {}
    bool charge(uint64_t bytes) override {
        if (used_ + bytes > ceiling_) return false;
        if (!acct_.reserve(mem::Category::Misc, static_cast<size_t>(bytes))) return false;
        used_ += bytes;
        return true;
    }
    void refund(uint64_t bytes) override {
        acct_.release(mem::Category::Misc, static_cast<size_t>(bytes));
        used_ -= bytes;
    }

private:
    mem::Accountant& acct_;
    const uint64_t   ceiling_;
    uint64_t         used_ = 0;   // the scheduler's thread only
};


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

int serve_main(const engine::EngineConfig& cfg, const ServeOptions& opt) {
    const int port = opt.port;
    const std::string& jobs_dir = opt.jobs_dir;
    const std::string& api_key = opt.api_key;
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

    // Declared before ctx so they outlive the scheduler that ctx owns.
    engine::LlamaStepper stepper(*engine);

    // Conversation reuse. A kept conversation's attention KV sits in its slot's
    // own funded KV: free. Recurrent checkpoints are extra memory, so all of
    // them -- slots x checkpoints_per_slot x one sequence's state, plus framing
    // -- are reserved on the ledger now, or the server refuses with the price.
    engine::RamKvStore::Config kcfg;
    kcfg.idle_timeout = std::chrono::seconds(opt.kv_idle_timeout_s);
    kcfg.checkpoint_offset = opt.checkpoint_offset;
    kcfg.checkpoints_per_slot = stepper.needs_checkpoints() ? opt.checkpoints_per_slot : 0;
    uint64_t ckpt_reserved = 0;
    if (kcfg.checkpoints_per_slot > 0) {
        const uint64_t state = engine->recurrent_state_per_sequence();
        kcfg.max_checkpoint_bytes = state + (1ull << 20);   // + serializer framing
        ckpt_reserved = kcfg.max_checkpoint_bytes * static_cast<uint64_t>(kcfg.checkpoints_per_slot) *
                        static_cast<uint64_t>(stepper.slots());
        if (!engine->accountant()->reserve(mem::Category::Misc, ckpt_reserved)) {
            std::fprintf(stderr,
                         "serve: REFUSED: %d slot(s) x %d checkpoint(s) of %.1f MiB = %.1f MiB of "
                         "recurrent checkpoints do not fit the cap; raise --cap, lower "
                         "--parallel, or set serve.checkpoints_per_slot (0 = no reuse)\n",
                         stepper.slots(), kcfg.checkpoints_per_slot,
                         kcfg.max_checkpoint_bytes / 1048576.0, ckpt_reserved / 1048576.0);
            return 1;
        }
    }
    // Parked conversations: -1 = the slots' whole context, as the owner framed
    // the default total KV limit.
    const int64_t pool_tokens = opt.kv_pool_tokens < 0
        ? static_cast<int64_t>(stepper.context_per_slot()) * stepper.slots()
        : opt.kv_pool_tokens;
    LedgerCharge pool_memory(*engine->accountant(), engine->spare_cache_bytes());
    kcfg.pool_max_tokens = pool_tokens;
    kcfg.memory = &pool_memory;
    // Torn down in the drain after the scheduler stops and before the engine
    // closes: it refunds parked memory to the engine's ledger.
    auto kv_store = std::make_unique<engine::RamKvStore>(stepper, kcfg);
    std::fprintf(stderr, "serve: parked conversations: up to %lld tokens, %.1f GiB (the expert slot pool's "
                         "share; parked entries push cold experts out)\n",
                 static_cast<long long>(pool_tokens), engine->spare_cache_bytes() / 1073741824.0);
    const std::string reuse_note =
        !stepper.needs_checkpoints()
            ? "attention KV cut back to the shared prefix (no checkpoints needed)"
        : kcfg.checkpoints_per_slot > 0
            ? "recurrent checkpoints " + std::to_string(kcfg.checkpoints_per_slot) + " per slot, " +
                  std::to_string(ckpt_reserved >> 20) + " MiB reserved"
            : "recurrent model with checkpoints off: only exact continuations are reused";
    std::fprintf(stderr, "serve: conversation reuse on; %s\n", reuse_note.c_str());

    ServerContext ctx;
    ctx.engine = engine.get();
    ctx.jobs = std::make_shared<JobRegistry>();
    if (!jobs_dir.empty()) {
        std::unique_ptr<JobStore> job_store = JobStore::open(jobs_dir, cfg.model_path, *engine, &err);
        if (!job_store) {
            std::fprintf(stderr, "serve: %s\n", err.c_str());
            return 1;
        }
        job_store->scan(*ctx.jobs);
        ctx.store = std::move(job_store);
    }

    // From here on only the scheduler's thread touches the model.
    engine::Scheduler::Config scfg;
    scfg.prefill_tokens_per_step = opt.prefill_tokens_per_step;
    ctx.sched = std::make_shared<engine::Scheduler>(stepper, scfg, kv_store.get());
    ctx.kv = kv_store.get();
    ctx.sched->start();

    // Every running or waiting foreground request holds an HTTP thread, so the
    // pool is sized past the parallel width, with headroom that is never
    // handed to foreground requests (/health and the job routes stay live).
    const unsigned pool = std::max<unsigned>(CPPHTTPLIB_THREAD_POOL_COUNT,
                                             static_cast<unsigned>(opt.max_parallel) + 8u);
    srv.new_task_queue = [pool] { return new httplib::ThreadPool(pool); };
    ctx.foreground_limit = static_cast<int>(pool) - 4;

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
                         "%d request(s) in parallel)\n", port, opt.max_parallel);
    install_stop_handlers();
    StopWatcher watcher(srv, ctx.jobs);

    const bool ok = srv.listen_after_bind();
    std::fprintf(stderr, "serve: drain 1/5 listener returned\n");
    ctx.jobs->stop();
    worker.join();
    std::fprintf(stderr, "serve: drain 2/5 worker joined\n");
    watcher.join();
    std::fprintf(stderr, "serve: drain 3/5 watcher joined\n");
    // In-flight generations see the stop at their next step boundary; jobs
    // take their final checkpoint there (T1). Wait for every one to report.
    if (!ctx.sched->wait_idle(std::chrono::minutes(30))) {
        std::fprintf(stderr, "serve: drain: requests still running after 30 min; stopping anyway\n");
    }
    ctx.sched->stop();
    std::fprintf(stderr, "serve: drain 4/5 scheduler stopped\n");
    kv_store.reset();   // refunds parked memory while the engine's ledger still exists
    engine.reset();
    std::fprintf(stderr, "serve: drain 5/5 engine closed; exiting\n");
    return ok ? 0 : 1;
}

}  // namespace dray::server
