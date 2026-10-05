// Status, job and admin routes. Chat completions live in chat_completions.cpp.

#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <future>
#include <mutex>
#include <string>
#include <utility>

#include "json.hpp"
#include "llama.h"

#include "cache/snapshot_cache.h"
#include "engine/engine.h"
#include "engine/scheduler.h"
#include "engine/ram_kv_store.h"
#include "server/http_routes.h"
#include "server/job_registry.h"
#include "server/job_store.h"
#include "server/shutdown.h"

namespace dray::server {

using nlohmann::json;
using engine::Engine;
using engine::EngineCounters;

namespace {

void not_found(httplib::Response& res) {
    res.status = 404;
    res.set_content(json{ { "error", { { "message", "unknown id" } } } }.dump(), "application/json");
}

}  // namespace

void install_status_routes(httplib::Server& srv, const ServerContext& ctx) {
    Engine* eng = ctx.engine;
    auto sched = ctx.sched;
    const engine::RamKvStore* kv = ctx.kv;

    // Swarm S4: engine_report() walks the streamer's live containers, which the
    // decode thread mutates; calling it concurrently is a use-after-free waiting
    // on a deque block boundary. /health therefore reads the streamer only ON
    // the scheduler's thread, between two steps (Scheduler::post). A step can
    // take seconds on a flagship, so /health waits briefly and otherwise serves
    // the last snapshot and says so. Taint and identity are safe always (atomic
    // counter, immutable string).
    struct Snapshot {
        std::string report;
        EngineCounters counters;
        engine::RamKvStore::Stats kv;
    };
    auto health_cache = std::make_shared<std::pair<std::mutex, Snapshot>>();
    srv.Get("/health", [eng, sched, kv, health_cache](const httplib::Request&, httplib::Response& res) {
        auto fresh = std::make_shared<Snapshot>();
        std::future<void> f = sched->post([eng, kv, fresh, health_cache] {
            // T12: report and counters read together, at the same boundary.
            fresh->report = eng->streamer_report();
            fresh->counters = eng->counters();
            if (kv) fresh->kv = kv->stats();
            std::lock_guard<std::mutex> ck(health_cache->first);
            health_cache->second = *fresh;
        });
        const bool live = f.wait_for(std::chrono::milliseconds(250)) == std::future_status::ready;
        Snapshot shown;
        if (live) {
            shown = *fresh;
        } else {
            std::lock_guard<std::mutex> ck(health_cache->first);
            shown = health_cache->second;
            if (shown.report.empty()) shown.report = "a step is in progress; no snapshot yet";
        }
        const engine::Scheduler::Stats st = sched->stats();
        const std::string& report = shown.report;
        const EngineCounters& hc = shown.counters;
        json j = {
            { "status", eng->tainted() ? "tainted" : "ok" },
            { "busy", st.active > 0 },
            { "requests", { { "active", st.active }, { "waiting", st.waiting } } },
            // Conversation reuse: prompts that continued a kept conversation and
            // the prompt tokens they did not prefill again.
            { "kv_reuse", { { "requests", st.reused_requests }, { "tokens", st.reused_tokens } } },
            // Conversations parked outside the slots (read with the report, at a
            // step boundary; the last snapshot while a step runs).
            { "kv_pool", { { "entries", shown.kv.pool_entries }, { "tokens", shown.kv.pool_tokens },
                           { "bytes", shown.kv.pool_bytes }, { "parked", shown.kv.parked },
                           { "restored", shown.kv.restored }, { "evicted", shown.kv.pool_evicted } } },
            // T13: lock-free live fields, valid even mid-generation -- a cap
            // breach hours into a job is visible NOW, not at its end.
            { "generation", { { "tokens", eng->live_tokens() },
                              { "over_cap", eng->live_over_cap() } } },
            { "model", eng->model_id() },
            { "report", report },
            { "report_is_live", live },
        };
        if (live) {
            // R7/S37: structured counters, read with the report at one step boundary.
            j["counters"] = {
                { "bytes_streamed", hc.bytes_streamed },
                { "bytes_gather", hc.bytes_gather },
                { "nodes", hc.nodes },
                { "failures", hc.failures },
                { "resident_bytes", hc.resident_bytes },
                { "over_cap", hc.over_cap },
            };
        }
        res.set_content(j.dump(2), "application/json");
    });

    srv.Get("/v1/models", [eng](const httplib::Request&, httplib::Response& res) {
        json j = {
            { "object", "list" },
            { "data", { { { "id", eng->model_id() },
                          { "object", "model" },
                          { "owned_by", "dray" },
                          { "created", static_cast<int64_t>(std::time(nullptr)) },
                          // Per request: each slot has its own context (n_ctx is the sum).
                          { "context_length", llama_n_ctx_seq(eng->context()) } } } },
        };
        res.set_content(j.dump(2), "application/json");
    });
}

void install_job_routes(httplib::Server& srv, const ServerContext& ctx) {
    Engine* eng = ctx.engine;
    auto jobs = ctx.jobs;
    auto store = ctx.store;

    srv.Get(R"(/v1/responses/([A-Za-z0-9-]+))",
            [jobs, eng](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        auto lk = jobs->lock();
        const std::shared_ptr<Job> job = jobs->find_locked(id);
        if (!job) { not_found(res); return; }
        json j = {
            { "id", id }, { "object", "response" },
            { "status", job->status },
            { "model", eng->model_id() },
            { "output_text", job->text },
            { "usage", {
                { "prompt_tokens", job->tokens_in < 0 ? json(nullptr) : json(job->tokens_in) },
                { "completion_tokens", job->tokens_out },
                { "total_tokens", job->tokens_in < 0 ? json(nullptr)
                    : json(job->tokens_in + job->tokens_out) } } },
            { "error", job->error.empty() ? json(nullptr) : json(job->error) },
            { "seed", job->gp.seed },
            { "checkpoint_note", job->ckpt_note.empty() ? json(nullptr) : json(job->ckpt_note) },
            { "tainted", eng->tainted() },
        };
        res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // Swarm S5: cooperative cancel. Queued jobs cancel instantly; a running job
    // stops at its next between-decodes check. H18: no Content-Type gate --
    // OpenAI SDK cancel calls carry no body; the pre-routing Origin refusal is
    // the actual browser control.
    srv.Post(R"(/v1/responses/([A-Za-z0-9-]+)/cancel)",
             [jobs, store](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        auto lk = jobs->lock();
        const std::shared_ptr<Job> job = jobs->find_locked(id);
        if (!job) { not_found(res); return; }
        job->cancel.store(true, std::memory_order_relaxed);
        if (job->status == "queued") {
            jobs->dequeue_locked(id);
            job->status = "cancelled";
            job->finished_at = std::chrono::steady_clock::now();
            // T7: cancel must be DURABLE, or a queued resumed job resurrects on
            // the next start from its surviving sidecar and snapshot.
            if (store && job->resume) {
                store->remove_sidecar(id);
                if (job->resume_hash) store->checkpoints().discard(job->resume_hash);
            }
        }
        res.set_content(json{ { "id", id }, { "status", job->status },
                              { "cancel_requested", true } }.dump(),
                        "application/json");
    });

    // T7/F15: terminal jobs are DELETEable immediately (files included, so the
    // id cannot resurrect) and reaped from memory after an hour. A reaped
    // failed job's surviving sidecar retries at next start BY DESIGN, bounded
    // by the attempts quarantine.
    srv.Delete(R"(/v1/responses/([A-Za-z0-9-]+))",
               [jobs, store](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        auto lk = jobs->lock();
        const std::shared_ptr<Job> job = jobs->find_locked(id);
        if (!job) { not_found(res); return; }
        const std::string& st = job->status;
        if (st == "queued" || st == "in_progress") {
            res.status = 409;
            res.set_content(json{ { "error", { { "message",
                "job is " + st + "; cancel it first" } } } }.dump(),
                            "application/json");
            return;
        }
        if (store && job->resume_hash) store->checkpoints().discard(job->resume_hash);
        if (store) store->remove_all_files(id);
        jobs->erase_locked(id);
        res.set_content(json{ { "id", id }, { "deleted", true } }.dump(), "application/json");
    });
}

// Loopback-only by construction: the same graceful path as Ctrl-C, from the
// flag onward. Exists for operators (systemd ExecStop, scripts), and because
// signal delivery is untestable from some harnesses while this is untestable
// from none.
void install_admin_routes(httplib::Server& srv) {
    srv.Post("/admin/shutdown", [](const httplib::Request& req, httplib::Response& res) {
        // F11: a bodyless cross-origin POST is a CORS simple request; requiring
        // a JSON content type makes the browser preflight it, which loopback
        // servers never answer for foreign origins.
        if (req.get_header_value("Content-Type").rfind("application/json", 0) != 0) {
            res.status = 415;
            res.set_content("{\"error\":{\"message\":\"send Content-Type: application/json\"}}",
                            "application/json");
            return;
        }
        request_stop();
        res.set_content("{\"stopping\":true}", "application/json");
    });
}

}  // namespace dray::server
