// The public protocol: an OpenAI-compatible API only (owner, 2026-08-14), in the
// spirit of llama-server. No interface, no dashboard, ever.
//
//   POST /v1/chat/completions   stream:true -> SSE deltas, else full JSON
//   GET  /v1/models
//   GET  /health                includes the streamer's honest report line
//
// Admission is SERIALIZED (Invariant 7): one generation at a time, a mutex held
// for the request's whole lifetime. Concurrent requests queue on the lock --
// batching past ~8 would dissolve the streaming premise, and v1 does not batch
// at all. Every response carries an id (ldg-N); reconnectable ids arrive with
// the job layer, and nothing in this surface will have to change for it.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_map>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif
#include <chrono>
#include <csignal>
#include <ctime>

#include "httplib.h"
#include "json.hpp"
#include "llama.h"

#include "server/engine.h"
#include "cache/snapshot_cache.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using nlohmann::json;

namespace dray::server {

namespace {

std::atomic<uint64_t> g_id_counter{0};

// S28: graceful shutdown. The handler only sets a flag (async-signal-safe);
// a watcher thread does the actual srv.stop(), which makes listen() return
// and the drain sequence after it -- previously dead code -- run: jobs stop,
// the worker joins, the engine closes.
std::atomic<bool> g_stop_requested{false};
#if defined(_WIN32)
BOOL WINAPI dray_ctrl_handler(DWORD) { g_stop_requested.store(true); return TRUE; }
#else
void dray_sig_handler(int) { g_stop_requested.store(true); }
#endif
std::string now_id() {
    return "ldg-" + std::to_string(++g_id_counter);
}

// Checkpoint key: unique per (job, position). Semantic prefix reuse is not the
// goal here -- surviving a crash is -- so the key only has to never collide.
uint64_t job_ckpt_hash(const std::string& id, int32_t tokens_done) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : id) { h ^= (uint8_t)c; h *= 0x100000001b3ull; }
    return h ^ static_cast<uint64_t>(tokens_done);
}

// messages[] -> one prompt string. The model's own chat template when it has
// one; otherwise an explicit, boring fallback -- and the response says which
// was used, because silently improvising a template changes model behaviour.
std::string build_prompt(const llama_model* model, const json& messages,
                         bool* used_template) {
    *used_template = false;
    std::vector<llama_chat_message> msgs;
    std::vector<std::string> keep;  // llama_chat_message holds char*: keep alive
    keep.reserve(messages.size() * 2);
    for (const auto& m : messages) {
        keep.push_back(m.is_object() ? m.value("role", "user") : "user");
        // Swarm S8: content may be a string, null, or the spec's content-part
        // array; value() throws on a present null and on arrays.
        std::string content;
        const auto ci = m.is_object() ? m.find("content") : m.end();
        if (ci != m.end()) {
            if (ci->is_string()) content = ci->get<std::string>();
            else if (ci->is_array()) {
                for (const auto& part : *ci) {
                    if (part.is_object() && part.value("type", "") == "text")
                        content += part.value("text", "");
                }
            }
        }
        keep.push_back(content);
        msgs.push_back({ keep[keep.size() - 2].c_str(), keep.back().c_str() });
    }

    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl) {
        std::vector<char> buf(1 << 16);
        int32_t r = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                              true, buf.data(),
                                              static_cast<int32_t>(buf.size()));
        if (r > static_cast<int32_t>(buf.size())) {
            buf.resize(static_cast<size_t>(r));
            r = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true,
                                          buf.data(), static_cast<int32_t>(buf.size()));
        }
        if (r > 0) {
            *used_template = true;
            return std::string(buf.data(), static_cast<size_t>(r));
        }
    }
    // Fallback template, built from the already-sanitized pairs above -- the
    // raw json re-read this replaced was the same S8 throw in other clothes.
    std::string p;
    for (size_t i = 0; i + 1 < keep.size(); i += 2) {
        p += keep[i];
        p += ": ";
        p += keep[i + 1];
        p += "\n";
    }
    p += "assistant:";
    return p;
}

}  // namespace

int serve_main(const EngineConfig& cfg, int port, const std::string& jobs_dir,
               const std::string& api_key) {
    // Redirected stderr is FULLY buffered on Windows CRT; a wedged drain then
    // shows an empty log, which is how the drain hang below was nearly
    // undiagnosable. Logs must be honest under redirection.
    setvbuf(stderr, nullptr, _IONBF, 0);
    // Swarm S12: bind FIRST. Discovering a taken port after minutes of model
    // load, behind an already-printed success banner, was the worst order.
    httplib::Server srv;
    // Swarm S15: bound request bodies; httplib otherwise buffers arbitrarily
    // large payloads in RAM on a machine whose premise is a small footprint.
    srv.set_payload_max_length(32ull << 20);
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
            return 1;
        }
    }
    if (!srv.bind_to_port("127.0.0.1", port)) {
        std::fprintf(stderr, "serve: cannot bind 127.0.0.1:%d (in use, or no permission); nothing was loaded\n", port);
        return 1;
    }

    // S23: optional bearer auth (absent key = open on localhost; /health stays
    // open for monitors). T14 mitigation, always on: the vendored httplib's
    // payload cap does not bound CHUNKED bodies, so chunked transfer on /v1/*
    // draws 411 instead of buffering unbounded RAM -- the honest alternative
    // to a risky wholesale httplib bump before release.
    {
        const std::string want = api_key.empty() ? std::string() : "Bearer " + api_key;
        srv.set_pre_routing_handler(
            [want](const httplib::Request& req, httplib::Response& res) {
                // F11: /health is the ONLY exempt path (monitors). Everything
                // else -- /v1/* AND /admin/* -- passes the gates below; the
                // old /v1-prefix test left /admin/shutdown as the one
                // unauthenticated state change after the operator paid for auth.
                if (req.path == "/health") {
                    return httplib::Server::HandlerResponse::Unhandled;
                }
                // T15/G5: EQUALITY against the loopback names, after stripping an
                // optional :port -- the old prefix test tolerated the port by
                // tolerating ANY suffix, so localhost.attacker.com passed and a
                // DNS-rebound page became same-origin (no preflight, readable
                // responses). An absent Host is refused too, and any request
                // carrying an Origin header is refused outright: no legitimate
                // CLI or SDK client sends one, and every browser does.
                if (!req.get_header_value("Origin").empty()) {
                    res.status = 403;
                    res.set_content("{\"error\":{\"message\":\"browser-origin requests are not accepted\"}}",
                                    "application/json");
                    return httplib::Server::HandlerResponse::Handled;
                }
                std::string host = req.get_header_value("Host");
                for (char& hcv : host) hcv = static_cast<char>(std::tolower(static_cast<unsigned char>(hcv)));   // H17
                {
                    const size_t rb = host.rfind(']');   // [::1]:port keeps its brackets
                    const size_t cp = host.rfind(':');
                    if (cp != std::string::npos && (rb == std::string::npos || cp > rb)) {
                        host.erase(cp);
                    }
                }
                if (host != "127.0.0.1" && host != "localhost" &&
                    host != "::1" && host != "[::1]") {
                    res.status = 403;
                    res.set_content("{\"error\":{\"message\":\"host not allowed\"}}",
                                    "application/json");
                    return httplib::Server::HandlerResponse::Handled;
                }
                std::string te = req.get_header_value("Transfer-Encoding");
                for (char& tc : te) tc = static_cast<char>(std::tolower(static_cast<unsigned char>(tc)));
                // Pass-4b: httplib decides chunked-ness case-insensitively; a
                // case-sensitive gate let "Chunked" bypass the 411 AND the
                // payload cap behind it.
                if (te.find("chunked") != std::string::npos) {
                    res.status = 411;
                    res.set_content("{\"error\":{\"message\":\"chunked transfer not supported; send Content-Length\"}}",
                                    "application/json");
                    return httplib::Server::HandlerResponse::Handled;
                }
                // Pass-4c: the cap has THREE holes, not one. set_payload_max_length
                // is consulted only on the Content-Length branch; chunked was
                // closed above, and a request carrying NEITHER header falls
                // through to httplib's read-until-the-peer-closes path with no
                // bound at all. On a process whose whole premise is a RAM cap,
                // one connection could buffer gigabytes outside the accountant
                // (2026-08-24 audit).
                // Only methods that carry a body: a GET or DELETE legitimately has
                // no Content-Length, and 411ing those would break /v1/models,
                // /v1/responses/{id} and every poll.
                if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
                    !req.has_header("Content-Length")) {
                    res.status = 411;
                    res.set_content("{\"error\":{\"message\":\"Content-Length required\"}}",
                                    "application/json");
                    return httplib::Server::HandlerResponse::Handled;
                }
                // Constant-time comparison: operator== short-circuits at the
                // first differing byte, which leaks match length over loopback
                // timing. The length check leaks only the length, which the
                // "Bearer " prefix already makes guessable (2026-08-24 audit).
                auto ct_equal = [](const std::string& a, const std::string& b) {
                    if (a.size() != b.size()) return false;
                    unsigned char acc = 0;
                    for (size_t i = 0; i < a.size(); ++i) {
                        acc = static_cast<unsigned char>(
                            acc | (static_cast<unsigned char>(a[i]) ^
                                   static_cast<unsigned char>(b[i])));
                    }
                    return acc == 0;
                };
                if (want.empty() || ct_equal(req.get_header_value("Authorization"), want)) {
                    return httplib::Server::HandlerResponse::Unhandled;
                }
                res.status = 401;
                res.set_header("WWW-Authenticate", "Bearer");
                res.set_content("{\"error\":{\"message\":\"invalid or missing API key\"}}",
                                "application/json");
                return httplib::Server::HandlerResponse::Handled;
            });
    }

    std::string err;
    Engine* eng = engine_open(cfg, &err);
    if (!eng) {
        std::fprintf(stderr, "serve: %s\n", err.c_str());
        return 1;
    }

    // One generation at a time -- the whole request holds this.
    auto gate = std::make_shared<std::mutex>();

    // THE JOB REGISTRY (v1): background requests return an id immediately and a
    // single worker drains them FIFO under the same admission gate as
    // foreground requests -- Invariant 7 does not care who asked. Poll with
    // GET /v1/responses/{id}; text accumulates as tokens land, so a poll
    // mid-generation shows honest partial output. In-memory only: periodic
    // checkpoints and cross-restart resume ship TOGETHER later, because a
    // checkpoint nobody can resume is diagnostics wearing a feature's name.
    struct Job {
        std::string status = "queued";   // queued -> in_progress -> completed/failed/cancelled
        std::string text;
        std::string error;
        std::string ckpt_note;   // F4: last checkpoint failure, surfaced on the poll
        int32_t     tokens_in = -1;   // F14: prompt tokens; -1 = unknown (renders null)
        int32_t     attempts = 0;       // F15/G6: consecutive NO-PROGRESS resumes
        int32_t     loaded_tokens = 0;  // G6: sidecar position at load, the progress bar
        std::atomic<bool> cancel{false};
        std::chrono::steady_clock::time_point finished_at{};   // T7: TTL reaping
        int32_t     tokens_out = 0;
        GenParams   gp;
        // Crash-resume state, populated by the startup scan from a sidecar.
        bool        resume = false;
        int32_t     resume_pending = 0;
        uint64_t    resume_hash = 0;
        int32_t     orig_max = 0;
    };
    struct Jobs {
        std::mutex mu;
        std::condition_variable cv;
        std::unordered_map<std::string, std::shared_ptr<Job>> by_id;
        std::deque<std::string> fifo;
        std::atomic<bool> stopping{false};   // read by the worker predicate without the mutex
    };
    auto jobs = std::make_shared<Jobs>();

    // CHECKPOINT + CRASS-RESUME, shipped together. With --jobs-dir set, every
    // background job checkpoints at safepoints (state BEFORE the pending token,
    // the token in the sidecar -- snaptest's proven shape) and the startup scan
    // re-enqueues interrupted jobs under their ORIGINAL ids. A killed server
    // resumes where the last checkpoint left it. GREEDY jobs resume with text
    // identical to an uninterrupted run (proven by hand-run kill/restart, and
    // claimed only for greedy). SAMPLED jobs resume PLAUSIBLY, not identically
    // (R10: llama's sampler API cannot replay an RNG stream mid-flight); they
    // get a fresh per-resume seed derived from job seed and position, recorded
    // on the job so a given resume is itself reproducible.
    std::shared_ptr<dray::cache::SnapshotCache> snap;
    if (!jobs_dir.empty()) {
        dray::cache::CompatStamp stamp;
        std::string serr;
        if (!dray::cache::make_compat_stamp(cfg.model_path, engine_ctx(eng),
                                               "dray-jobs-v1", &stamp, &serr)) {
            std::fprintf(stderr, "serve: jobs stamp: %s\n", serr.c_str());
            engine_close(eng);
            return 1;
        }
        dray::cache::Config cc;
        cc.dir = jobs_dir;
        // T6/F4: checkpoint blobs live inside the cap via the standing floor
        // allowance engine_open reserved (acct_prereserved) -- refusal happens
        // once at admission, never silently per safepoint at steady state.
        // If reservation fails at peak pressure the checkpoint is REFUSED and
        // the store-failure line says so -- honest degradation, never a silent
        // multi-GB allocation outside the cap (Invariant 1).
        cc.acct = engine_accountant(eng);
        // G1: pass the allowance SIZE -- coverage is per-blob, never blanket.
        cc.prereserved_bytes = engine_checkpoint_allowance(eng);
        // Pinned entries are not charged against the Reuse budget, but open()
        // demands a nonzero one -- its contract, honored rather than argued with.
        cc.budget_bytes = 1ull << 30;
        snap = std::make_shared<dray::cache::SnapshotCache>(cc, stamp);
        if (!snap->open(&serr)) {
            std::fprintf(stderr, "serve: jobs dir: %s\n", serr.c_str());
            engine_close(eng);
            return 1;
        }
        // Startup scan: every sidecar is an interrupted job. Re-register under
        // the original id, seed the id counter past it, re-enqueue.
        uint64_t max_id = 0;
        for (const auto& de : std::filesystem::directory_iterator(jobs_dir)) {
            // Battery residual (leg B): a quarantined job's poll record died
            // with the process that quarantined it, 404ing an id whose durable
            // evidence sits right here. Register .job.failed files so the
            // operator's poll works across restarts too.
            if (de.path().extension() == ".failed") {
                try {
                    json qj;
                    std::ifstream qf(de.path());
                    qf >> qj;
                    const std::string qid = qj.value("id", "");
                    if (!qid.empty() &&
                        qid.find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-") == std::string::npos) {
                        auto qjob = std::make_shared<Job>();
                        qjob->status = "quarantined";
                        qjob->error = "exceeded resume attempts; sidecar quarantined to .job.failed";
                        qjob->text = qj.value("text", "");
                        qjob->tokens_out = qj.value("tokens_done", 0);
                        qjob->resume_hash = qj.value("hash", 0ull);   // I4: DELETE needs it
                        qjob->finished_at = std::chrono::steady_clock::now();
                        uint64_t qn = 0;
                        if (qid.rfind("ldg-", 0) == 0) qn = std::strtoull(qid.c_str() + 4, nullptr, 10);
                        max_id = std::max(max_id, qn);
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        jobs->by_id[qid] = qjob;
                    }
                } catch (...) { /* unreadable quarantine file: the rename already logged */ }
                continue;
            }
            if (de.path().extension() != ".job") continue;
            try {
                json sj;
                { std::ifstream f(de.path()); f >> sj; }
                auto job = std::make_shared<Job>();
                job->status = "queued";
                job->text = sj.value("text", "");
                job->tokens_out = sj.value("tokens_done", 0);
                job->gp.prompt = sj.value("prompt", "");
                job->gp.max_tokens = sj.value("max_tokens", 512);
                job->gp.temperature = sj.value("temperature", 0.0f);
                job->gp.seed = sj.value("seed", 0u);
                job->gp.stop = sj.value("stop", std::vector<std::string>{});   // F7
                // H6: id parse, validation and counter seeding HOIST above the
                // quarantine branch -- continuing over them let a quarantined
                // ordinal be REISSUED by now_id(), silently replacing the
                // quarantined record, and skipped the S27 whitelist on the one
                // id that reaches the filesystem twice.
                const std::string id = sj.value("id", "");
                if (id.empty()) continue;
                if (id.find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-") != std::string::npos) {
                    std::fprintf(stderr, "serve: rejecting sidecar with suspicious id\n");
                    continue;
                }
                uint64_t num = 0;
                if (id.rfind("ldg-", 0) == 0) num = std::strtoull(id.c_str() + 4, nullptr, 10);
                max_id = std::max(max_id, num);
                job->resume_hash = sj.value("hash", 0ull);
                // F15: a job that keeps failing must not resurrect forever. Past
                // the bound its sidecar is quarantined, snapshot left for the
                // operator, and the scan moves on.
                // G6: the count must reach DISK at scan time or early failures
                // (redecode, prefill, store) freeze it forever and the loop the
                // bound exists for is unbounded. And it counts CONSECUTIVE
                // resumes WITHOUT PROGRESS: any publish that advances past the
                // loaded position resets it to zero, so graceful restarts of a
                // healthy job never accumulate toward the ceiling.
                job->attempts = sj.value("attempts", 0) + 1;
                job->loaded_tokens = sj.value("tokens_done", 0);
                static const int kAttemptCeiling = [] {
                    // H16: atoi yields 0 on garbage, and a 0 ceiling quarantines
                    // every job on its first resume -- the S11 lesson, again.
                    const char* v = std::getenv("DRAY_RESUME_ATTEMPTS");
                    if (!v || !v[0]) return 3;
                    char* end = nullptr;
                    const long n2 = std::strtol(v, &end, 10);
                    if (end == v || *end != '\0' || n2 < 1 || n2 > 1000) {
                        std::fprintf(stderr, "serve: DRAY_RESUME_ATTEMPTS=%s is not a "
                                             "sane ceiling; using 3\n", v);
                        return 3;
                    }
                    return static_cast<int>(n2);
                }();
                if (job->attempts > kAttemptCeiling) {
                    std::error_code qec;
                    std::filesystem::rename(de.path(),
                        std::filesystem::path(de.path().string() + ".failed"), qec);
                    std::fprintf(stderr, "serve: job %s exceeded resume attempts; quarantined\n",
                                 sj.value("id", "?").c_str());
                    // G6: visible on the poll instead of a 404 that reads as
                    // "no such job". Terminal; DELETE clears the record.
                    job->status = "quarantined";
                    job->error = "exceeded resume attempts; sidecar quarantined to .job.failed";
                    job->finished_at = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lk(jobs->mu);
                    jobs->by_id[id] = job;   // H6: validated id, counter already seeded
                    continue;
                }
                {
                    // G6/H5: persist the incremented count through the SAME
                    // durable pattern the publisher uses. The truncate-in-place
                    // this replaces destroyed the only copy at open -- an ENOSPC
                    // write (no crash needed) left a torn .job that the next
                    // scan drops forever, orphaning an eviction-invisible
                    // pinned blob. The comment that justified it described an
                    // outcome the code could not produce.
                    sj["attempts"] = job->attempts;
                    const std::filesystem::path tmp2 = de.path().string() + ".new";
                    bool rewrote = false;
#if defined(_WIN32)
                    std::FILE* rf = nullptr;
                    fopen_s(&rf, tmp2.string().c_str(), "wb");
#else
                    std::FILE* rf = std::fopen(tmp2.string().c_str(), "wb");
#endif
                    if (rf) {
                        const std::string body2 =
                            sj.dump(2, ' ', false, json::error_handler_t::replace);
                        const bool w2 =
                            std::fwrite(body2.data(), 1, body2.size(), rf) == body2.size() &&
                            std::fflush(rf) == 0;
#if defined(_WIN32)
                        const bool s2 = w2 && _commit(_fileno(rf)) == 0;
#else
                        const bool s2 = w2 && fsync(fileno(rf)) == 0;
#endif
                        std::fclose(rf);
                        if (s2) {
                            std::error_code rec;
                            std::filesystem::rename(tmp2, de.path(), rec);
                            rewrote = !rec;
                        }
                    }
                    if (!rewrote) {
                        std::fprintf(stderr, "serve: could not persist attempts for %s; "
                                             "prior sidecar remains valid\n",
                                     sj.value("id", "?").c_str());
                    }
                }
                job->resume = true;
                job->resume_pending = sj.value("pending", 0);
                job->orig_max = job->gp.max_tokens;
                // H6: id was parsed, validated and counter-seeded ABOVE the
                // quarantine branch (resume_hash too); register the runnable
                // job here.
                std::lock_guard<std::mutex> lk(jobs->mu);
                jobs->by_id[id] = job;
                jobs->fifo.push_back(id);
                std::fprintf(stderr, "serve: resuming job %s at %d tokens\n",
                             id.c_str(), job->tokens_out);
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "serve: bad sidecar %s: %s\n",
                             de.path().string().c_str(), ex.what());
            }
        }
        uint64_t cur = g_id_counter.load();
        while (cur < max_id && !g_id_counter.compare_exchange_weak(cur, max_id)) {}
    }

    std::thread worker([eng, gate, jobs, snap, jobs_dir] {
        // Swarm S11: atoi returns 0 on garbage and 0 faulted the modulo below;
        // a nonsense cadence is refused loudly at startup, not divided by.
        const int32_t ckpt_every = [] {
            const char* v = std::getenv("DRAY_CKPT_EVERY");
            if (!v) return 256;
            char* end = nullptr;
            const long n = std::strtol(v, &end, 10);
            // M1: bound BEFORE narrowing -- on LP64 a 2^32 value narrowed to
            // int32 as 0 and reached the modulo (SIGFPE at token two of the
            // first job); LONG_MAX narrowed to -1 (a full pinned store per
            // token). Windows was immune by the accident of 32-bit long.
            if (end == v || *end || n <= 0 || n > 1000000) {
                std::fprintf(stderr, "serve: DRAY_CKPT_EVERY=%s is not a sane cadence; using 256\n", v);
                return 256;
            }
            return static_cast<int32_t>(n);
        }();
        for (;;) {
            std::string id;
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(jobs->mu);
                jobs->cv.wait(lk, [&] { return jobs->stopping || !jobs->fifo.empty(); });
                if (jobs->stopping) return;
                id = jobs->fifo.front();
                jobs->fifo.pop_front();
                // find(), not operator[]: a fifo id with no registry entry used
                // to default-construct a null shared_ptr and deref it on the
                // next line. Unreachable through the normal transitions, but
                // the startup scan's duplicate-registration path perturbs
                // exactly that invariant, and the guard is free (2026-08-24).
                auto jit = jobs->by_id.find(id);
                if (jit == jobs->by_id.end()) continue;
                job = jit->second;
                job->status = "in_progress";
            }

            GenParams gp = job->gp;
            // Shutdown must also stop the worker's generation. The stop watcher
            // cancels queued jobs, but if listen() returns for any reason other
            // than the watcher (a socket error), jobs->stopping is set with the
            // watcher already gone -- and this predicate only checked
            // job->cancel, so worker.join() could block on a generation with
            // days left in it (2026-08-24 audit).
            gp.should_continue = [job, jobs] {
                return !job->cancel.load(std::memory_order_relaxed) && !jobs->stopping;
            };
            // F14: prompt tokens for usage; jobs never wired this hook before.
            gp.on_prefill = [job, jobs](int32_t np) {
                std::lock_guard<std::mutex> lk(jobs->mu);
                job->tokens_in = np;
            };
            // F8: the resumed prefix; the final registry text is this + the
            // session's authoritative r.text.
            std::string text_base;
            {
                std::lock_guard<std::mutex> lk(jobs->mu);
                text_base = job->text;
            }
            uint64_t prev_hash = 0;
            // S28: seconds are this project's honest unit; 256 tokens spans 25
            // minutes to hours across the model set, so time also triggers.
            const long ckpt_secs = [] {
                const char* v = std::getenv("DRAY_CKPT_SECONDS");
                if (!v) return 600L;
                char* end = nullptr;
                const long n = std::strtol(v, &end, 10);
                // M2: silent 600 on garbage disabled the ONLY checkpoint
                // mechanism for slow jobs without a word. Loud, like every
                // other knob.
                if (end == v || *end || n <= 0 || n > 604800L) {
                    std::fprintf(stderr, "serve: DRAY_CKPT_SECONDS=%s is not a sane interval; using 600\n", v);
                    return 600L;
                }
                return n;
            }();
            auto last_ckpt = std::chrono::steady_clock::now();
            const std::filesystem::path sidecar =
                jobs_dir.empty() ? std::filesystem::path{}
                                 : std::filesystem::path(jobs_dir) / (id + ".job");

            std::lock_guard<std::mutex> gen(*gate);

            if (job->resume && snap) {
                auto entry = snap->lookup(job->resume_hash);
                std::string lerr;
                if (entry && snap->load(*entry, engine_ctx(eng), 0, &lerr)) {
                    gp.resume = true;
                    gp.resume_pending = job->resume_pending;
                    // F5: the engine reports session-local indices; this base
                    // makes every safepoint count absolute across restarts.
                    gp.resume_tokens_done = job->tokens_out;
                    gp.max_tokens = job->orig_max - job->tokens_out;
                    prev_hash = job->resume_hash;
                    // Seed the stop matcher with the prefix tail so a stop
                    // straddling the resume boundary still fires. Longest stop
                    // minus one byte is the widest straddle possible; the seed
                    // cannot hold a full stop because the pre-crash session
                    // would have finished on it (2026-08-24 audit).
                    if (!gp.stop.empty() && !text_base.empty()) {
                        size_t longest = 0;
                        for (const std::string& st : gp.stop) {
                            longest = std::max(longest, st.size());
                        }
                        if (longest > 1) {
                            const size_t tail = std::min(text_base.size(), longest - 1);
                            gp.stop_seed = text_base.substr(text_base.size() - tail);
                        }
                    }
                    if (gp.temperature > 0.0f) {
                        // R10: the RNG stream cannot be replayed; a fresh seed
                        // derived from (job seed, position) keeps THIS resume
                        // reproducible and is disclosed on the job record.
                        gp.seed = static_cast<uint32_t>(
                            (gp.seed * 2654435761u) ^ static_cast<uint32_t>(job->tokens_out)) | 1u;
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        job->gp.seed = gp.seed;   // disclosed via the poll response
                    }
                } else if (lerr.rfind("TRANSIENT:", 0) == 0) {
                    // H1: a TRANSIENT refusal (cap has no room right now) keeps
                    // EVERYTHING -- checkpoint, index, text, counts -- and
                    // fails the job for this session so the next start retries
                    // with the state intact. Restarting from the prompt here
                    // destroyed a perfectly good checkpoint.
                    std::fprintf(stderr, "serve: job %s restore deferred (%s)\n",
                                 id.c_str(), lerr.c_str());
                    // I5: a deferral is NOT a resume attempt -- the engine never
                    // ran. The scan already durably persisted +1; hand it back
                    // the same durable way, or deterministic cap pressure
                    // quarantines a healthy job in kAttemptCeiling boots.
                    {
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        if (job->attempts > 0) --job->attempts;
                        job->status = "deferred";   // non-terminal word: next start retries
                        job->error = lerr;
                        job->ckpt_note = lerr;
                        job->finished_at = std::chrono::steady_clock::now();
                    }
                    {
                        const std::filesystem::path sp =
                            std::filesystem::path(jobs_dir) / (id + ".job");
                        try {
                            json dj;
                            std::ifstream df(sp);
                            df >> dj;
                            dj["attempts"] = job->attempts;
                            const std::filesystem::path dt = sp.string() + ".new";
#if defined(_WIN32)
                            std::FILE* xf = nullptr;
                            fopen_s(&xf, dt.string().c_str(), "wb");
#else
                            std::FILE* xf = std::fopen(dt.string().c_str(), "wb");
#endif
                            if (xf) {
                                const std::string xb =
                                    dj.dump(2, ' ', false, json::error_handler_t::replace);
                                const bool xw =
                                    std::fwrite(xb.data(), 1, xb.size(), xf) == xb.size() &&
                                    std::fflush(xf) == 0;
#if defined(_WIN32)
                                const bool xs = xw && _commit(_fileno(xf)) == 0;
#else
                                const bool xs = xw && fsync(fileno(xf)) == 0;
#endif
                                std::fclose(xf);
                                if (xs) {
                                    std::error_code xec;
                                    std::filesystem::rename(dt, sp, xec);
                                }
                            }
                        } catch (...) { /* prior sidecar remains; one extra counted attempt */ }
                    }
                    continue;
                } else {
                    // Checkpoint PERMANENTLY unusable: restart from the prompt
                    // rather than fail -- the sidecar kept everything needed,
                    // and honesty means a fresh run, not a silent partial.
                    // H1/H2: discard the dead blob NOW instead of parking its
                    // hash in prev_hash -- parked, the restarted run re-derived
                    // the same (id, position) hash at the old position, the
                    // dedupe guard swallowed a REAL checkpoint there, and the
                    // time trigger was consumed doing it.
                    if (lerr.empty()) lerr = "no indexed checkpoint for this hash";
                    // I6: discard-by-index is a guaranteed no-op here -- load's
                    // permanent failure already DE-INDEXED the entry while
                    // deliberately leaving the file. Unlink by the PATH the
                    // lookup handed us, or the .lzs re-indexes as Pinned at
                    // every open() forever.
                    if (entry) {
                        std::error_code uec;
                        std::filesystem::remove(entry->path, uec);
                    } else if (job->resume_hash) {
                        snap->discard(job->resume_hash);   // never loaded: still indexed
                    }
                    std::fprintf(stderr, "serve: job %s checkpoint unusable (%s); "
                                         "restarting from prompt\n",
                                 id.c_str(), lerr.c_str());
                    std::lock_guard<std::mutex> lk(jobs->mu);
                    job->text.clear();
                    job->tokens_out = 0;
                    job->loaded_tokens = 0;     // H4: the fifth counter, reset with its four siblings
                    job->resume_hash = 0;
                    text_base.clear();          // F8: no seeded prefix on a fresh restart
                    gp.resume_tokens_done = 0;  // F5: counts restart absolute-from-zero
                }
            }

            if (snap && !jobs_dir.empty()) {
                gp.on_safepoint = [&](int32_t next_tok, int32_t true_done) {
                    // T7: the engine's second argument is the TRUE decoded-token
                    // count. job->tokens_out counts token_cb invocations, which
                    // UTF-8 hold-back merges and the final flush inflates -- a
                    // resumed job budgeted from it overshoots on non-ASCII text.
                    const int32_t done = true_done;
                    std::string text_now;
                    {
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        text_now = job->text;
                    }
                    if (done == 0) return;
                    // T1: a draining shutdown (or an engine-offered final
                    // safepoint after cancel) bypasses the cadence so zero
                    // progress is lost; ordinary tokens keep the cadence.
                    bool draining;
                    {
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        draining = jobs->stopping || job->cancel.load(std::memory_order_relaxed);
                    }
                    const auto nowt = std::chrono::steady_clock::now();
                    const bool time_due = std::chrono::duration_cast<std::chrono::seconds>(nowt - last_ckpt).count() >= ckpt_secs;
                    if (!draining && done % ckpt_every != 0 && !time_due) return;
                    const uint64_t h = job_ckpt_hash(id, done);
                    if (h == prev_hash) return;   // already checkpointed here
                    // H2: consume the time-trigger slot only when a checkpoint
                    // is actually attempted -- a dedupe return above this line
                    // used to burn the whole ckpt_secs interval writing nothing.
                    last_ckpt = nowt;
                    std::string serr2;
                    if (!snap->store(h, static_cast<uint64_t>(done),
                                     engine_ctx(eng), 0,
                                     dray::cache::Retention::Pinned, &serr2)) {
                        std::fprintf(stderr, "serve: ckpt store failed: %s\n", serr2.c_str());
                        std::lock_guard<std::mutex> lk2(jobs->mu);
                        job->ckpt_note = serr2;   // F4: operators see it on the poll
                        return;
                    }
                    // H3: the attempts value must be decided BEFORE it is made
                    // durable -- resetting after the publish persisted the
                    // pre-reset count, and a one-publish session (every graceful
                    // drain) then counted a healthy restart toward quarantine.
                    int32_t attempts_now;
                    {
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        attempts_now = (done > job->loaded_tokens) ? 0 : job->attempts;
                    }
                    json sj = {
                        { "v", 1 },   // F7: schema version, so the next field is detectable
                        { "id", id }, { "prompt", gp.prompt },
                        { "max_tokens", job->orig_max ? job->orig_max : job->gp.max_tokens },
                        { "temperature", gp.temperature }, { "seed", gp.seed },
                        { "stop", gp.stop },   // F7: the one client field the sidecar lost
                        { "attempts", attempts_now },   // F15/H3
                        { "text", text_now }, { "tokens_done", done },
                        { "pending", next_tok }, { "hash", h },
                    };
                    // Swarm S6/R11: atomic publish, checked. Truncate-in-place
                    // left a zero-length sidecar reachable by a plain kill, and
                    // the old checkpoint was discarded regardless of success.
                    // T4 (release sweep): durable publish, to the standard the
                    // project already states in snapshot_cache.cpp -- fsync
                    // BEFORE rename (ofstream::flush only reaches the OS, and
                    // a power loss inside the writeback window published a
                    // truncated sidecar); no pre-remove (rename replaces, and
                    // the remove opened the one window with NO sidecar while
                    // making the failure message a lie); and NOT a ".tmp"
                    // suffix, which snapshot_cache::open() sweeps at boot --
                    // destroying the flushed record exactly when recovery
                    // would need it.
                    const std::filesystem::path tmp = sidecar.string() + ".new";
                    bool published = false;
                    {
                        const std::string body =
                            sj.dump(-1, ' ', false, json::error_handler_t::replace);
#if defined(_WIN32)
                        std::FILE* f = nullptr;
                        fopen_s(&f, tmp.string().c_str(), "wb");
#else
                        std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
#endif
                        if (f) {
                            const bool wrote =
                                std::fwrite(body.data(), 1, body.size(), f) == body.size() &&
                                std::fflush(f) == 0;
#if defined(_WIN32)
                            const bool synced = wrote && _commit(_fileno(f)) == 0;
#else
                            const bool synced = wrote && fsync(fileno(f)) == 0;
#endif
                            std::fclose(f);
                            if (synced) {
                                std::error_code ec;
                                std::filesystem::rename(tmp, sidecar, ec);
                                published = !ec;
                            }
                        }
                    }
                    if (!published) {
                        std::fprintf(stderr,
                                     "serve: checkpoint sidecar publish FAILED for %s; "
                                     "previous checkpoint remains valid, unpublished temp kept\n",
                                     id.c_str());
                        snap->discard(h);   // the new snapshot has no sidecar; drop it
                        return;
                    }
                    if (prev_hash) snap->discard(prev_hash);
                    prev_hash = h;
                    {
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        // G7: resume_hash had exactly one writer (the startup
                        // scan), so DELETE discarded a stale or zero hash while
                        // the LIVE pinned blob survived unreferenced. Track it.
                        job->resume_hash = h;
                        // G6/H3: progress past the loaded position proves the
                        // job is healthy; the value was DECIDED before the
                        // publish so the sidecar carries it, and memory now
                        // agrees with disk.
                        job->attempts = attempts_now;
                    }
                };
            }
            if (!job->orig_max) job->orig_max = gp.max_tokens;

            // Swarm S2: nothing thrown below may escape this thread -- an
            // uncaught exception here is std::terminate for the whole server.
            GenResult r;
            try {
                r = engine_generate(eng, gp,
                    [&](const std::string& piece) {
                        std::lock_guard<std::mutex> lk(jobs->mu);
                        job->text += piece;
                        ++job->tokens_out;
                    });
            } catch (const std::exception& ex) {
                r.error = std::string("internal: ") + ex.what();
            } catch (...) {
                r.error = "internal: unknown exception";
            }

            // T1 (release sweep): this cleanup ran UNCONDITIONALLY, so the S28
            // shutdown watcher's cancel-everything made Ctrl+C DELETE the
            // checkpoint a SIGKILL would have preserved -- four individually
            // correct fixes jointly inverting the safety property. Checkpoint
            // state is destroyed only on genuine completion or an EXPLICIT
            // user cancel; shutdown-cancel and failures keep it for resume.
            bool user_cancelled;
            {
                std::lock_guard<std::mutex> lk(jobs->mu);
                // F3: jobs->stopping alone races -- the main thread sets it
                // only after listen() returns, while the watcher cancels jobs
                // BEFORE calling stop(). g_stop_requested is written strictly
                // before the watcher touches any job and never by user cancel,
                // so it closes the window in which a shutdown-cancel could be
                // misread as a user cancel and delete the checkpoint.
                user_cancelled = r.cancelled && !jobs->stopping &&
                                 !g_stop_requested.load(std::memory_order_relaxed);
            }
            const bool done_clean = r.error.empty() && !r.aborted && !r.cancelled;
            if (snap && !sidecar.empty() && (done_clean || user_cancelled)) {
                std::error_code ec;
                std::filesystem::remove(sidecar, ec);
                if (prev_hash) snap->discard(prev_hash);
            }
            std::lock_guard<std::mutex> lk(jobs->mu);
            // T7/F5: the engine's true count, made absolute across restarts.
            job->tokens_out = gp.resume_tokens_done + r.tokens_out;
            // F8: r.text is the authority for THIS session (stop trimming
            // happens there); the registry keeps seed + session. base_trim is
            // the resume-straddle case: the stop began inside text_base, so the
            // overhang comes off the persisted prefix too.
            if (r.base_trim > 0 && r.base_trim <= text_base.size()) {
                text_base.erase(text_base.size() - r.base_trim);
            }
            job->text = text_base + r.text;
            // Swarm S1+S7: aborted counts as failed, and the error must not
            // destroy the partial output the client may still want.
            // A shutdown-cancel is NOT terminal: the sidecar and checkpoint
            // survive by design and the next boot resumes the job. Reporting it
            // "cancelled" told the client a lie in both directions -- the poll
            // said terminal, the reaper deleted the record an hour later, and
            // the job then silently ran again at startup. "interrupted" is what
            // it actually is (2026-08-24 audit).
            job->status = r.cancelled ? (user_cancelled ? "cancelled" : "interrupted")
                        : (r.error.empty() && !r.aborted) ? "completed" : "failed";
            job->finished_at = std::chrono::steady_clock::now();
            if (!r.error.empty()) job->error = r.error;
            else if (r.aborted) job->error = "aborted: a weight failed to materialise; output untrustworthy";
        }
    });

    // Swarm S8 backstop: anything still escaping a handler becomes a JSON 500
    // with a message, never a bodiless one.
    srv.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "unknown exception";
        try { if (ep) std::rethrow_exception(ep); } catch (const std::exception& ex) { what = ex.what(); } catch (...) {}
        res.status = 500;
        res.set_content(json{ { "error", { { "message", what } } } }.dump(), "application/json");
    });

    // Swarm S4: engine_report() walks the streamer's live containers, which the
    // decode thread mutates; calling it concurrently is a use-after-free waiting
    // on a deque block boundary. /health therefore only reads the streamer when
    // it can take the admission gate (nothing generating); otherwise it serves
    // the last idle snapshot and says so. Taint and identity are safe always
    // (atomic counter, immutable string).
    auto health_cache = std::make_shared<std::pair<std::mutex, std::string>>();
    srv.Get("/health", [eng, gate, health_cache](const httplib::Request&, httplib::Response& res) {
        std::string report;
        bool live = false;
        EngineCounters hc;
        {
            std::unique_lock<std::mutex> g(*gate, std::try_to_lock);
            if (g.owns_lock()) {
                report = engine_report(eng);
                live = true;
                // T12: counters must be read HERE, while the gate is genuinely
                // held -- the old code read them after this scope closed, under
                // a comment claiming otherwise.
                hc = engine_counters(eng);
                std::lock_guard<std::mutex> ck(health_cache->first);
                health_cache->second = report;
            }
        }
        if (!live) {
            std::lock_guard<std::mutex> ck(health_cache->first);
            report = health_cache->second.empty()
                         ? "generation in progress; no idle snapshot yet"
                         : health_cache->second;
        }
        json j = {
            { "status", engine_tainted(eng) ? "tainted" : "ok" },
            { "busy", !live },
            // T13: lock-free live fields, valid even mid-generation -- a cap
            // breach hours into a job is visible NOW, not at its end.
            { "generation", { { "tokens", engine_live_tokens(eng) },
                              { "over_cap", engine_live_over_cap(eng) } } },
            { "model", engine_model_id(eng) },
            { "report", report },
            { "report_is_live", live },
        };
        if (live) {
            // R7/S37: structured counters, captured above under the gate.
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
            { "data", { { { "id", engine_model_id(eng) },
                          { "object", "model" },
                          { "owned_by", "dray" },
                          { "created", static_cast<int64_t>(std::time(nullptr)) },
                          { "context_length", llama_n_ctx(engine_ctx(eng)) } } } },
        };
        res.set_content(j.dump(2), "application/json");
    });

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
            gp.prompt = build_prompt(engine_model(eng), body["messages"], &used_template);
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
        const std::string id = now_id();
        const std::string model_id = engine_model_id(eng);

        if (background) {
            // R10: a checkpointed sampled job needs a CONCRETE seed to be
            // resumable-reproducible; absent seed gets one derived from the id,
            // echoed below so the client can reproduce.
            if (gp.temperature > 0.0f && gp.seed == 0) {
                gp.seed = static_cast<uint32_t>(job_ckpt_hash(id, 0) & 0x7fffffffu) | 1u;
            }
            auto job = std::make_shared<Job>();
            job->gp = gp;
            {
                std::lock_guard<std::mutex> lk(jobs->mu);
                // Bounded, on a process whose premise is a byte cap: each job
                // holds its prompt and accumulating text outside the accountant,
                // and the queue had no limit at all. 256 pending jobs on an
                // engine that serializes admission is already days of work
                // (2026-08-24 audit).
                if (jobs->fifo.size() >= 256) {
                    res.status = 429;
                    res.set_content(
                        "{\"error\":{\"message\":\"job queue full (256 pending); "
                        "retry after some complete\"}}",
                        "application/json");
                    return;
                }
                jobs->by_id[id] = job;
                jobs->fifo.push_back(id);
            }
            jobs->cv.notify_one();
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
            gp.should_continue = [] { return !g_stop_requested.load(std::memory_order_relaxed); };
            GenResult r = engine_generate(eng, gp, nullptr);
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
            if (r.aborted || r.cancelled || engine_tainted(eng)) {
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
                                         { "tainted", engine_tainted(eng) },
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
                               { "tainted", engine_tainted(eng) } } },
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
                    return sink.is_writable() && !g_stop_requested.load(std::memory_order_relaxed);
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
                GenResult r = engine_generate(eng, gpl,
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
                const bool bad = !r.error.empty() || r.aborted || engine_tainted(eng);
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

    srv.Get(R"(/v1/responses/([A-Za-z0-9-]+))",
            [jobs, eng](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(jobs->mu);
        auto it = jobs->by_id.find(id);
        if (it == jobs->by_id.end()) {
            res.status = 404;
            res.set_content(json{ { "error", { { "message", "unknown id" } } } }.dump(),
                            "application/json");
            return;
        }
        json j = {
            { "id", id }, { "object", "response" },
            { "status", it->second->status },
            { "model", engine_model_id(eng) },
            { "output_text", it->second->text },
            { "usage", {
                { "prompt_tokens", it->second->tokens_in < 0 ? json(nullptr)
                                                             : json(it->second->tokens_in) },
                { "completion_tokens", it->second->tokens_out },
                { "total_tokens", it->second->tokens_in < 0 ? json(nullptr)
                    : json(it->second->tokens_in + it->second->tokens_out) } } },
            { "error", it->second->error.empty() ? json(nullptr) : json(it->second->error) },
            { "seed", it->second->gp.seed },
            { "checkpoint_note", it->second->ckpt_note.empty() ? json(nullptr)
                                                              : json(it->second->ckpt_note) },
            { "tainted", engine_tainted(eng) },
        };
        res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // Swarm S5: cooperative cancel for background jobs. Queued jobs cancel
    // instantly; a running job stops at its next between-decodes check.
    srv.Post(R"(/v1/responses/([A-Za-z0-9-]+)/cancel)",
             [jobs, snap, jobs_dir](const httplib::Request& req, httplib::Response& res) {
        // H18: no Content-Type gate here -- OpenAI SDK cancel calls carry no
        // body, so a 415 broke protocol-legal requests, and the pre-routing
        // Origin refusal is the actual browser control.
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(jobs->mu);
        auto it = jobs->by_id.find(id);
        if (it == jobs->by_id.end()) {
            res.status = 404;
            res.set_content(json{ { "error", { { "message", "unknown id" } } } }.dump(),
                            "application/json");
            return;
        }
        it->second->cancel.store(true, std::memory_order_relaxed);
        if (it->second->status == "queued") {
            for (auto q = jobs->fifo.begin(); q != jobs->fifo.end(); ++q) {
                if (*q == id) { jobs->fifo.erase(q); break; }
            }
            it->second->status = "cancelled";
            it->second->finished_at = std::chrono::steady_clock::now();
            // T7: cancel must be DURABLE. A queued resumed job cancelled only
            // in memory returned 200 "cancelled" and then resurrected on the
            // next start, because its sidecar and snapshot survived.
            if (snap && it->second->resume) {
                std::error_code ec;
                std::filesystem::remove(std::filesystem::path(jobs_dir) / (id + ".job"), ec);
                if (it->second->resume_hash) snap->discard(it->second->resume_hash);
            }
        }
        res.set_content(json{ { "id", id }, { "status", it->second->status },
                              { "cancel_requested", true } }.dump(),
                        "application/json");
    });

    // Loopback-only by construction (the server binds 127.0.0.1): the same
    // graceful path as Ctrl-C, from the flag onward -- cancel-all, final
    // checkpoints, drain. Exists for operators (systemd ExecStop, scripts)
    // and because signal delivery is untestable from some harnesses while
    // this is untestable from none.
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
        g_stop_requested.store(true);
        res.set_content("{\"stopping\":true}", "application/json");
    });

    // T7/F15: terminal jobs are DELETEable immediately (files included, so the
    // id cannot resurrect) and reaped from MEMORY after an hour. A reaped
    // failed job's surviving sidecar retries at next start BY DESIGN --
    // crash-resume semantics -- bounded by the attempts quarantine above.
    srv.Delete(R"(/v1/responses/([A-Za-z0-9-]+))",
               [jobs, snap, jobs_dir](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(jobs->mu);
        auto it = jobs->by_id.find(id);
        if (it == jobs->by_id.end()) {
            res.status = 404;
            res.set_content(json{ { "error", { { "message", "unknown id" } } } }.dump(),
                            "application/json");
            return;
        }
        const std::string& st = it->second->status;
        if (st == "queued" || st == "in_progress") {
            res.status = 409;
            res.set_content(json{ { "error", { { "message",
                "job is " + st + "; cancel it first" } } } }.dump(),
                            "application/json");
            return;
        }
        // F15: durable for preserved jobs -- the T1 guard keeps sidecars for
        // failed/shutdown jobs, and the startup scan re-enqueues every
        // survivor; DELETE must take the files or the id resurrects.
        if (snap && it->second->resume_hash) snap->discard(it->second->resume_hash);
        if (!jobs_dir.empty()) {
            std::error_code ec;
            std::filesystem::remove(std::filesystem::path(jobs_dir) / (id + ".job"), ec);
            // I4: quarantine renamed the sidecar; DELETE must reach that name
            // too or the record resurrects at every start.
            std::filesystem::remove(std::filesystem::path(jobs_dir) / (id + ".job.failed"), ec);
        }
        jobs->by_id.erase(it);
        res.set_content(json{ { "id", id }, { "deleted", true } }.dump(),
                        "application/json");
    });

    std::fprintf(stderr, "dray serving on http://127.0.0.1:%d (OpenAI-compatible; "
                         "admission serialized)\n", port);
    // S28: handlers set only a flag; the watcher calls srv.stop(), which makes
    // listen() return and the drain below -- previously dead code -- run.
#if defined(_WIN32)
    SetConsoleCtrlHandler(dray_ctrl_handler, TRUE);
#else
    std::signal(SIGINT, dray_sig_handler);
    std::signal(SIGTERM, dray_sig_handler);
#endif
    std::thread stop_watcher([&srv, jobs] {
        int reap_tick = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            {
                std::lock_guard<std::mutex> lk(jobs->mu);
                if (jobs->stopping) return;   // normal shutdown already draining
                // T7: reap terminal jobs after an hour, every ~30 s.
                if (++reap_tick >= 150) {
                    reap_tick = 0;
                    const auto now = std::chrono::steady_clock::now();
                    for (auto it = jobs->by_id.begin(); it != jobs->by_id.end();) {
                        const std::string& st = it->second->status;
                        const bool terminal = st == "completed" || st == "failed" ||
                                              st == "cancelled" || st == "quarantined" ||
                                              st == "deferred";   // I4/I5
                        if (terminal &&
                            std::chrono::duration_cast<std::chrono::seconds>(
                                now - it->second->finished_at).count() > 3600) {
                            it = jobs->by_id.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
            if (g_stop_requested.load()) break;
        }
        std::fprintf(stderr, "serve: stop requested; cancelling jobs and draining\n");
        {
            std::lock_guard<std::mutex> lk(jobs->mu);
            for (auto& kv : jobs->by_id) kv.second->cancel.store(true);
        }
        srv.stop();
    });

    const bool ok = srv.listen_after_bind();
    std::fprintf(stderr, "serve: drain 1/4 listener returned\n");
    {
        std::lock_guard<std::mutex> lk(jobs->mu);
        jobs->stopping = true;
    }
    jobs->cv.notify_all();
    worker.join();
    std::fprintf(stderr, "serve: drain 2/4 worker joined\n");
    stop_watcher.join();
    std::fprintf(stderr, "serve: drain 3/4 watcher joined\n");
    engine_close(eng);
    std::fprintf(stderr, "serve: drain 4/4 engine closed; exiting\n");
    return ok ? 0 : 1;
}

}  // namespace dray::server
