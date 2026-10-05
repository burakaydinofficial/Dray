// POST /v1/chat/completions: the route. It reads the request (chat_request),
// admits it, and dispatches to one of three responses -- a background job id
// (202), a blocking completion, or an SSE stream -- whose wire format lives in
// chat_response and whose model format lives in chat_format. What is left here
// is transport: the scheduler hand-off, the stream channel, keep-alives, and
// cancellation when the client goes away.
//
// Every generation is a request to the scheduler: foreground requests run in
// parallel up to serve.max_parallel (each in its own slot, sharing every step's
// weight reads); past the foreground limit they get 503 + Retry-After, and
// background jobs queue instead.

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>

#include "json.hpp"

#include "engine/engine.h"
#include "engine/scheduler.h"
#include "server/chat_format.h"
#include "server/chat_request.h"
#include "server/chat_response.h"
#include "server/http_routes.h"
#include "server/job_registry.h"
#include "server/shutdown.h"

namespace dray::server {

using nlohmann::json;
using engine::GenResult;

namespace {

void send_json(httplib::Response& res, int status, const json& j) {
    res.status = status;
    res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
}

// Background: queue a job, answer 202 with its id.
void respond_background(const ServerContext& ctx, ChatRequest& rq, const std::string& id,
                        const std::string& model, httplib::Response& res) {
    engine::GenParams& gp = rq.params;
    // R10: a checkpointed sampled job needs a CONCRETE seed to be
    // resumable-reproducible; an absent seed gets one derived from the id,
    // echoed below so the client can reproduce.
    if (gp.temperature > 0.0f && gp.seed == 0) {
        gp.seed = static_cast<uint32_t>(job_ckpt_hash(id, 0) & 0x7fffffffu) | 1u;
    }
    auto job = std::make_shared<Job>();
    job->gp = gp;
    if (!ctx.jobs->submit(id, job)) {   // bounded: see JobRegistry::kMaxPending
        send_json(res, 429, error_body({ 429, "job queue full (256 pending); retry after some complete", "" }));
        return;
    }
    send_json(res, 202, json{ { "id", id }, { "object", "response" }, { "status", "queued" },
                              { "model", model }, { "seed", gp.seed } });
}

// Blocking: wait for the scheduler's result, answer once.
void respond_blocking(const ServerContext& ctx, ChatRequest& rq, const ResponseId& rid,
                      httplib::Response& res) {
    rq.params.should_continue = [] { return !stop_requested(); };   // T2
    auto result = std::make_shared<std::promise<GenResult>>();
    std::future<GenResult> done = result->get_future();
    engine::Request q;
    q.params = rq.params;
    q.done = [result](GenResult r) { result->set_value(std::move(r)); };
    ctx.sched->submit(std::move(q));
    const GenResult r = done.get();
    const bool tainted = ctx.engine->tainted();
    if (!r.error.empty()) {
        send_json(res, r.bad_request ? 400 : 500, error_body({ 0, r.error, "" }));
        return;
    }
    if (untrustworthy(r, tainted)) {
        send_json(res, 500, untrustworthy_body(r, tainted));
        return;
    }
    ChatOutput out(rq.rendered);
    out.feed(r.text);
    out.finish();
    send_json(res, 200, completion_body(rid, r, out, tainted));
}

// Pieces cross from the scheduler's thread to the HTTP thread writing the stream.
struct StreamChannel {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> pieces;
    bool done = false;
    GenResult result;
};

// Streaming: SSE chunks as the model's format completes them.
void respond_streaming(const ServerContext& ctx, const ChatRequest& rq, const ResponseId& rid,
                       httplib::Response& res) {
    auto sched = ctx.sched;
    engine::Engine* eng = ctx.engine;
    auto finished = std::make_shared<bool>(false);
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [sched, eng, rq, rid, finished](size_t, httplib::DataSink& sink) {
            if (*finished) return false;
            // Swarm S2: the provider runs outside httplib's routing try/catch;
            // anything escaping here would terminate the process.
            try {
                auto emit = [&sink](const json& j) {
                    const std::string s =
                        "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                    sink.write(s.c_str(), s.size());
                };
                // T11: something reaches the wire before the first sampled token,
                // or SDK read timeouts fire mid-prefill and retry the whole request.
                emit(chunk_body(rid, json{ { "role", "assistant" } }, nullptr));

                ChatOutput out(rq.rendered);
                auto ch = std::make_shared<StreamChannel>();
                engine::Request q;
                q.params = rq.params;
                q.params.should_continue = [] { return !stop_requested(); };
                q.sink = [ch](const std::string& piece) {
                    std::lock_guard<std::mutex> lk(ch->mu);
                    ch->pieces.push_back(piece);
                    ch->cv.notify_one();
                };
                q.done = [ch](GenResult r) {
                    std::lock_guard<std::mutex> lk(ch->mu);
                    ch->result = std::move(r);
                    ch->done = true;
                    ch->cv.notify_one();
                };
                const uint64_t request_id = sched->submit(std::move(q));

                bool gone = false;
                for (;;) {
                    std::deque<std::string> batch;
                    bool done = false;
                    {
                        std::unique_lock<std::mutex> lk(ch->mu);
                        ch->cv.wait_for(lk, std::chrono::seconds(2),
                                        [&] { return !ch->pieces.empty() || ch->done; });
                        batch.swap(ch->pieces);
                        done = ch->done;
                    }
                    for (const std::string& piece : batch) {
                        for (const std::string& d : out.feed(piece)) {
                            emit(chunk_body(rid, json::parse(d), nullptr));
                        }
                    }
                    if (done) break;
                    // F13: an SSE comment keeps SDK read timeouts alive while the
                    // request waits for a slot or prefills; it costs nothing.
                    if (batch.empty()) {
                        static const char kAlive[] = ": waiting\n\n";
                        sink.write(kAlive, sizeof(kAlive) - 1);
                    }
                    // Swarm S5: a vanished client stops the burn at the next step.
                    if (!gone && !sink.is_writable()) {
                        gone = true;
                        sched->cancel(request_id);
                    }
                }
                const GenResult r = ch->result;
                for (const std::string& d : out.finish()) emit(chunk_body(rid, json::parse(d), nullptr));
                // Swarm S1: failure and taint MUST reach the stream; only a clean
                // run may claim stop, length or tool_calls.
                if (!r.error.empty() || r.aborted || eng->tainted()) {
                    emit(error_chunk_body(rid, !r.error.empty() ? r.error
                        : "generation aborted: a weight failed to materialise; output untrustworthy"));
                } else {
                    emit(chunk_body(rid, json::object(), finish_reason(r, out.has_tool_calls())));
                }
                static const char kDone[] = "data: [DONE]\n\n";
                sink.write(kDone, sizeof(kDone) - 1);
                sink.done();
                *finished = true;
                return false;
            } catch (...) {
                *finished = true;
                return false;   // the connection drops; better than the process
            }
        });
}

}  // namespace

void install_chat_completions(httplib::Server& srv, const ServerContext& ctx) {
    // The model's own chat template and output parser (tools, reasoning).
    auto format = std::make_shared<ChatFormat>(ctx.engine->model());
    srv.Post("/v1/chat/completions", [ctx, format](const httplib::Request& req, httplib::Response& res) {
        ChatRequest rq;
        ApiError err;
        if (!parse_chat_request(req.body, *format, &rq, &err)) {
            send_json(res, err.status, error_body(err));
            return;
        }
        const ResponseId rid{ ctx.jobs->next_id(), ctx.engine->model_id(),
                              static_cast<int64_t>(std::time(nullptr)) };
        if (rq.background) {
            respond_background(ctx, rq, rid.id, rid.model, res);
            return;
        }
        // Swarm S9, restated for the scheduler: a foreground request holds an
        // HTTP thread while it runs or waits. Past the limit it is told to retry
        // rather than eat the pool until even /health cannot get a thread; the
        // job API is the tool for deep queues.
        const engine::Scheduler::Stats st = ctx.sched->stats();
        if (st.active + st.waiting >= ctx.foreground_limit) {
            res.set_header("Retry-After", "30");
            send_json(res, 503, error_body({ 503,
                "the server is at its parallel limit; retry, or submit with background:true", "" }));
            return;
        }
        if (rq.stream) {
            respond_streaming(ctx, rq, rid, res);
        } else {
            respond_blocking(ctx, rq, rid, res);
        }
    });
}

}  // namespace dray::server
