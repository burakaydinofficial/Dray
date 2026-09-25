// POST /v1/chat/completions: request parsing, then one of three responses --
// a background job id (202), a blocking completion, or an SSE stream.
//
// Admission is SERIALIZED (Invariant 7): a foreground request try-locks the
// gate and answers 503 + Retry-After when busy; background jobs queue instead.

#include <cstdint>
#include <ctime>
#include <memory>
#include <string>

#include "json.hpp"
#include "llama.h"

#include "engine/engine.h"
#include "server/chat_prompt.h"
#include "server/http_routes.h"
#include "server/job_registry.h"
#include "server/shutdown.h"

namespace dray::server {

using nlohmann::json;
using engine::Engine;
using engine::GenParams;
using engine::GenResult;

void install_chat_completions(httplib::Server& srv, const ServerContext& ctx) {
    Engine* eng = ctx.engine;
    auto gate = ctx.gate;
    auto jobs = ctx.jobs;
    srv.Post("/v1/chat/completions",
             [eng, gate, jobs](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& ex) {
            res.status = 400;
            res.set_content(json{ { "error", { { "message", ex.what() } } } }.dump(),
                            "application/json");
            return;
        }
        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(json{ { "error", { { "message", "messages[] required" } } } }.dump(),
                            "application/json");
            return;
        }

        GenParams gp;
        bool used_template = false;
        // Swarm S8: value() throws on a present null ("max_tokens": null is an
        // ordinary OpenAI client payload) -- a malformed request is a 400 with a
        // message, never an opaque 500.
        bool stream = false, background = false;
        try {
            gp.prompt = build_prompt(eng->model(), body["messages"], &used_template);
            auto num = [&body](const char* k, double def) {
                auto it = body.find(k);
                return (it == body.end() || it->is_null()) ? def : it->get<double>();
            };
            // Clamp BEFORE casting: converting an out-of-range double to an
            // integer is undefined behaviour, and {"max_tokens": 1e20} is
            // well-formed JSON that used to reach static_cast<int32_t> raw --
            // on x86-64 it produced INT32_MIN and a 200 with empty content
            // instead of a 400 (2026-08-24 audit). NaN maps to the default.
            auto num_i32 = [&num](const char* k, int32_t def, int32_t lo, int32_t hi) {
                const double d = num(k, static_cast<double>(def));
                if (d != d) return def;
                if (d < static_cast<double>(lo)) return lo;
                if (d > static_cast<double>(hi)) return hi;
                return static_cast<int32_t>(d);
            };
            auto num_u32 = [&num](const char* k, uint32_t def) {
                const double d = num(k, static_cast<double>(def));
                if (d != d || d < 0.0) return def;
                if (d > 4294967295.0) return static_cast<uint32_t>(4294967295u);
                return static_cast<uint32_t>(d);
            };
            auto flag = [&body](const char* k) {
                auto it = body.find(k);
                return it != body.end() && it->is_boolean() && it->get<bool>();
            };
            // T10: max_completion_tokens is what current SDKs send; honour it,
            // preferring it over the deprecated max_tokens when both appear.
            gp.max_tokens  = num_i32("max_completion_tokens",
                                     num_i32("max_tokens", 512, 1, 1 << 24),
                                     1, 1 << 24);
            // T10: stop may be a string or an array of strings.
            {
                auto sit = body.find("stop");
                if (sit != body.end() && !sit->is_null()) {
                    if (sit->is_string()) gp.stop.push_back(sit->get<std::string>());
                    else if (sit->is_array()) {
                        for (const auto& s2 : *sit) {
                            if (s2.is_string()) gp.stop.push_back(s2.get<std::string>());
                        }
                    }
                }
            }
            // T10: refuse what we silently discarded -- an ignored parameter is
            // indistinguishable from a honoured one, which Invariant 6 forbids.
            for (const char* unsup : { "tools", "response_format", "logit_bias", "logprobs" }) {
                auto uit = body.find(unsup);
                if (uit != body.end() && !uit->is_null()) {
                    res.status = 400;
                    res.set_content(json{ { "error", { { "message",
                        std::string(unsup) + " is not supported by this server" },
                        { "code", "unsupported_parameter" } } } }.dump(), "application/json");
                    return;
                }
            }
            {
                auto nit = body.find("n");
                if (nit != body.end() && nit->is_number() && nit->get<double>() > 1) {
                    res.status = 400;
                    res.set_content(json{ { "error", { { "message",
                        "n>1 is not supported (admission is serialized)" },
                        { "code", "unsupported_parameter" } } } }.dump(), "application/json");
                    return;
                }
            }
            gp.temperature = static_cast<float>(num("temperature", 0.8));
            gp.seed        = num_u32("seed", 0);
            stream     = flag("stream");
            background = flag("background");
        } catch (const std::exception& ex) {
            res.status = 400;
            res.set_content(json{ { "error", { { "message",
                std::string("invalid request field: ") + ex.what() } } } }.dump(),
                            "application/json");
            return;
        }
        const std::string id = jobs->next_id();
        const std::string model_id = eng->model_id();

        if (background) {
            // R10: a checkpointed sampled job needs a CONCRETE seed to be
            // resumable-reproducible; absent seed gets one derived from the id,
            // echoed below so the client can reproduce.
            if (gp.temperature > 0.0f && gp.seed == 0) {
                gp.seed = static_cast<uint32_t>(job_ckpt_hash(id, 0) & 0x7fffffffu) | 1u;
            }
            auto job = std::make_shared<Job>();
            job->gp = gp;
            // Bounded: see JobRegistry::kMaxPending.
            if (!jobs->submit(id, job)) {
                res.status = 429;
                res.set_content(
                    "{\"error\":{\"message\":\"job queue full (256 pending); "
                    "retry after some complete\"}}",
                    "application/json");
                return;
            }
            res.status = 202;
            res.set_content(json{ { "id", id }, { "object", "response" },
                                  { "status", "queued" }, { "model", model_id },
                                  { "seed", gp.seed } }.dump(),
                            "application/json");
            return;
        }

        if (!stream) {
            // Swarm S9: a blocking wait here eats an httplib pool thread per
            // queued client until even /health cannot get one. Busy = 503 +
            // Retry-After; the job API is the right tool for queueing.
            std::unique_lock<std::mutex> lk(*gate, std::try_to_lock);
            if (!lk.owns_lock()) {
                res.status = 503;
                res.set_header("Retry-After", "30");
                res.set_content(json{ { "error", { { "message",
                    "a generation is in progress; retry, or submit with background:true" } } } }.dump(),
                                "application/json");
                return;
            }
            // T2: the shutdown flag must reach every generation loop.
            gp.should_continue = [] { return !stop_requested(); };
            GenResult r = eng->generate(gp, nullptr);
            if (!r.error.empty()) {
                res.status = r.bad_request ? 400 : 500;
                res.set_content(json{ { "error", { { "message", r.error } } } }.dump(),
                                "application/json");
                return;
            }
            // A run that computed against poison, or one the engine has marked
            // tainted, is NOT a completion. The streaming path has always emitted
            // an error frame for exactly this (see the `bad` test below); the
            // non-streaming path tested only r.error and returned 200 with normal
            // content, demoting the truth to a vendor extension no client reads.
            //
            // This is the failure mode the whole project exists to prevent, and
            // it is the one shape where it reached a user: a materialise failure
            // that still decodes cleanly leaves r.error EMPTY and only r.aborted
            // set, so the check above cannot see it (2026-08-24 audit).
            if (r.aborted || r.cancelled || eng->tainted()) {
                res.status = 500;
                res.set_content(
                    json{ { "error",
                            { { "message",
                                r.aborted
                                    ? "generation aborted: a weight failed to materialise, so "
                                      "the output was computed against poison and is not "
                                      "trustworthy"
                                    : (r.cancelled
                                           ? "generation cancelled before completion"
                                           : "engine is tainted; earlier failures make this "
                                             "output untrustworthy") },
                              { "type", "dray_untrustworthy_output" } } },
                          { "dray", { { "aborted", r.aborted },
                                         { "cancelled", r.cancelled },
                                         { "tainted", eng->tainted() },
                                         { "partial_tokens", r.tokens_out } } } }
                        .dump(-1, ' ', false, json::error_handler_t::replace),
                    "application/json");
                return;
            }
            json j = {
                { "id", id },
                { "object", "chat.completion" },
                { "created", static_cast<int64_t>(std::time(nullptr)) },   // T9
                { "model", model_id },
                { "choices", { {
                    { "index", 0 },
                    { "message", { { "role", "assistant" }, { "content", r.text } } },
                    { "logprobs", nullptr },                               // T9
                    { "finish_reason", r.truncated_by_eog ? "stop" : "length" },
                } } },
                // T9: prompt_tokens and total_tokens are non-optional in every
                // real client's schema; the partial object crashed openai-python
                // one line after the truthy check passed.
                { "usage", { { "prompt_tokens", r.tokens_in },
                             { "completion_tokens", r.tokens_out },
                             { "total_tokens", r.tokens_in + r.tokens_out } } },
                { "dray", { { "chat_template", used_template ? "model" : "fallback" },
                               { "aborted", r.aborted },
                               { "tainted", eng->tainted() } } },
            };
            res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            return;
        }

        // STREAMING. The generation runs inside the content provider so tokens
        // flow as they are produced; the admission lock is held by a shared_ptr
        // that lives exactly as long as the response does.
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(*gate, std::try_to_lock);
        if (!lock->owns_lock()) {
            res.status = 503;
            res.set_header("Retry-After", "30");
            res.set_content(json{ { "error", { { "message",
                "a generation is in progress; retry, or submit with background:true" } } } }.dump(),
                            "application/json");
            return;
        }
        auto done = std::make_shared<bool>(false);
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [eng, gp, id, model_id, lock, done](size_t, httplib::DataSink& sink) {
                if (*done) return false;
                // Swarm S2: the provider runs outside httplib's routing
                // try/catch; anything escaping here terminates the process.
                try {
                auto emit = [&](const json& j) {
                    const std::string s =
                        "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                    sink.write(s.c_str(), s.size());
                };
                GenParams gpl = gp;
                // Swarm S5: a vanished client stops the burn within one token.
                gpl.should_continue = [&sink] {
                    return sink.is_writable() && !stop_requested();
                };
                const int64_t created = static_cast<int64_t>(std::time(nullptr));   // F14: one per response
                // T11: something reaches the wire before the first sampled token
                // or SDK read-timeouts fire mid-prefill and retry the whole
                // generation. The role chunk is conformant; the prefill progress
                // frames are SSE comments, which cost nothing.
                emit(json{
                    { "id", id },
                    { "object", "chat.completion.chunk" },
                    { "created", created },
                    { "model", model_id },
                    { "choices", { { { "index", 0 },
                                     { "delta", { { "role", "assistant" } } },
                                     { "finish_reason", nullptr } } } },
                });
                gpl.on_prefill = [&sink](int32_t np) {
                    const std::string c = ": prefill " + std::to_string(np) + " tokens\n\n";
                    sink.write(c.c_str(), c.size());
                };
                // F13: one comment frame per prefill chunk keeps SDK read
                // timeouts alive through minutes-long prefills -- the half of
                // T11 that delivers its stated benefit.
                gpl.on_prefill_progress = [&sink](int32_t done_toks) {
                    const std::string c =
                        ": prefill progress " + std::to_string(done_toks) + "\n\n";
                    sink.write(c.c_str(), c.size());
                };
                GenResult r = eng->generate(gpl,
                    [&](const std::string& piece) {
                        emit(json{
                            { "id", id },
                            { "object", "chat.completion.chunk" },
                            { "created", created },
                            { "model", model_id },
                            { "choices", { { { "index", 0 },
                                             { "delta", { { "content", piece } } },
                                             { "finish_reason", nullptr } } } },
                        });
                    });
                // Swarm S1: failure and taint MUST reach the stream. A hard error
                // or aborted/tainted run gets an explicit error frame; only a
                // clean run may claim stop/length.
                const bool bad = !r.error.empty() || r.aborted || eng->tainted();
                if (bad) {
                    emit(json{
                        { "id", id },
                        { "object", "chat.completion.chunk" },
                        { "created", created },
                        { "model", model_id },
                        { "error", { { "message", !r.error.empty() ? r.error :
                            "generation aborted: a weight failed to materialise; output untrustworthy" } } },
                        { "choices", { { { "index", 0 },
                                         { "delta", json::object() },
                                         { "finish_reason", "error" } } } },
                    });
                } else {
                    emit(json{
                        { "id", id },
                        { "object", "chat.completion.chunk" },
                        { "created", created },
                        { "model", model_id },
                        { "choices", { { { "index", 0 },
                                         { "delta", json::object() },
                                         { "finish_reason",
                                           r.cancelled ? "cancelled"
                                         : r.truncated_by_eog ? "stop" : "length" } } } },
                    });
                }
                static const char kDone[] = "data: [DONE]\n\n";
                sink.write(kDone, sizeof(kDone) - 1);
                sink.done();
                *done = true;
                return false;
                } catch (...) {
                    *done = true;
                    return false;   // connection drops; better than the process
                }
            });
    });
}

}  // namespace dray::server
