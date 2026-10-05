#include "server/job_worker.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "json.hpp"

#include "cache/snapshot_cache.h"
#include "engine/engine.h"
#include "engine/scheduler.h"
#include "server/job_registry.h"
#include "server/job_store.h"
#include "server/shutdown.h"

namespace dray::server {

using nlohmann::json;
using engine::GenParams;
using engine::GenResult;

namespace {

// Swarm S11 / M1: a nonsense cadence is refused loudly, never divided by. Bound
// BEFORE narrowing -- on LP64 2^32 narrowed to 0 and reached the modulo.
int32_t checkpoint_every() {
    const char* v = std::getenv("DRAY_CKPT_EVERY");
    if (!v) return 256;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end || n <= 0 || n > 1000000) {
        std::fprintf(stderr, "serve: DRAY_CKPT_EVERY=%s is not a sane cadence; using 256\n", v);
        return 256;
    }
    return static_cast<int32_t>(n);
}

// S28: seconds are this project's honest unit; 256 tokens spans 25 minutes to
// hours across the model set, so time also triggers. M2: loud on garbage.
long checkpoint_seconds() {
    const char* v = std::getenv("DRAY_CKPT_SECONDS");
    if (!v) return 600L;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end || n <= 0 || n > 604800L) {
        std::fprintf(stderr, "serve: DRAY_CKPT_SECONDS=%s is not a sane interval; using 600\n", v);
        return 600L;
    }
    return n;
}

// Checkpoints one job at safepoints: state BEFORE the pending token's decode,
// the token recorded in the sidecar (snaptest's proven resume shape).
class Checkpointer {
public:
    Checkpointer(ServerContext& ctx, const std::string& id, const std::shared_ptr<Job>& job,
                 const GenParams& gp, int32_t every)
        : ctx_(ctx), id_(id), job_(job), gp_(gp), every_(every),
          secs_(checkpoint_seconds()), last_(std::chrono::steady_clock::now()) {}

    uint64_t prev_hash = 0;   // the live checkpoint; discarded on completion

    // `seq`: the scheduler slot (llama sequence) holding this job.
    void on_safepoint(int32_t seq, int32_t next_tok, int32_t true_done);

private:
    ServerContext& ctx_;
    const std::string& id_;
    const std::shared_ptr<Job>& job_;
    const GenParams& gp_;
    const int32_t every_;
    const long secs_;
    std::chrono::steady_clock::time_point last_;
};

void Checkpointer::on_safepoint(int32_t seq, int32_t next_tok, int32_t true_done) {
    JobRegistry& jobs = *ctx_.jobs;
    // T7: the engine's second argument is the TRUE decoded-token count.
    // job->tokens_out counts token_cb invocations, which UTF-8 hold-back merges
    // and the final flush inflates.
    const int32_t done = true_done;
    std::string text_now;
    {
        auto lk = jobs.lock();
        text_now = job_->text;
    }
    if (done == 0) return;
    // T1: a draining shutdown (or an engine-offered final safepoint after
    // cancel) bypasses the cadence so zero progress is lost.
    bool draining;
    {
        auto lk = jobs.lock();
        draining = jobs.stopping() || job_->cancel.load(std::memory_order_relaxed);
    }
    const auto nowt = std::chrono::steady_clock::now();
    const bool time_due =
        std::chrono::duration_cast<std::chrono::seconds>(nowt - last_).count() >= secs_;
    if (!draining && done % every_ != 0 && !time_due) return;
    const uint64_t h = job_ckpt_hash(id_, done);
    if (h == prev_hash) return;   // already checkpointed here
    // H2: consume the time-trigger slot only when a checkpoint is attempted.
    last_ = nowt;
    std::string serr2;
    cache::SnapshotCache& snap = ctx_.store->checkpoints();
    if (!snap.store(h, static_cast<uint64_t>(done), ctx_.engine->context(), seq,
                    cache::Retention::Pinned, &serr2)) {
        std::fprintf(stderr, "serve: ckpt store failed: %s\n", serr2.c_str());
        auto lk2 = jobs.lock();
        job_->ckpt_note = serr2;   // F4: operators see it on the poll
        return;
    }
    // H3: the attempts value is decided BEFORE it is made durable.
    int32_t attempts_now;
    {
        auto lk = jobs.lock();
        attempts_now = (done > job_->loaded_tokens) ? 0 : job_->attempts;
    }
    const json sj = {
        { "v", 1 },   // F7: schema version, so the next field is detectable
        { "id", id_ }, { "prompt", gp_.prompt },
        { "max_tokens", job_->orig_max ? job_->orig_max : job_->gp.max_tokens },
        { "temperature", gp_.temperature }, { "seed", gp_.seed },
        { "stop", gp_.stop },   // F7: the one client field the sidecar lost
        { "attempts", attempts_now },   // F15/H3
        { "text", text_now }, { "tokens_done", done },
        { "pending", next_tok }, { "hash", h },
    };
    // Swarm S6/R11 + T4: atomic, durable publish; no pre-remove (rename
    // replaces, and removing first opened a window with NO sidecar).
    if (!ctx_.store->publish(id_, sj)) {
        std::fprintf(stderr,
                     "serve: checkpoint sidecar publish FAILED for %s; "
                     "previous checkpoint remains valid, unpublished temp kept\n",
                     id_.c_str());
        snap.discard(h);   // the new snapshot has no sidecar; drop it
        return;
    }
    if (prev_hash) snap.discard(prev_hash);
    prev_hash = h;
    {
        auto lk = jobs.lock();
        // G7: track the LIVE pinned blob so DELETE discards the right one.
        job_->resume_hash = h;
        // G6/H3: progress past the loaded position proves the job healthy.
        job_->attempts = attempts_now;
    }
}

enum class Restore { Resumed, Deferred, RestartFromPrompt };

// Restores a resumed job's checkpoint into sequence `seq` of the live context,
// or decides why not. Runs on the scheduler's thread (Request::restore).
Restore restore_checkpoint(ServerContext& ctx, const std::string& id, const std::shared_ptr<Job>& job,
                           GenParams& gp, std::string& text_base, uint64_t& prev_hash,
                           int32_t seq) {
    JobRegistry& jobs = *ctx.jobs;
    cache::SnapshotCache& snap = ctx.store->checkpoints();
    auto entry = snap.lookup(job->resume_hash);
    std::string lerr;
    if (entry && snap.load(*entry, ctx.engine->context(), seq, &lerr)) {
        gp.resume = true;
        gp.resume_pending = job->resume_pending;
        // F5: the engine reports session-local indices; this base makes every
        // safepoint count absolute across restarts.
        gp.resume_tokens_done = job->tokens_out;
        gp.max_tokens = job->orig_max - job->tokens_out;
        prev_hash = job->resume_hash;
        // Seed the stop matcher with the prefix tail so a stop straddling the
        // resume boundary still fires. Longest stop minus one byte is the widest
        // straddle possible.
        if (!gp.stop.empty() && !text_base.empty()) {
            size_t longest = 0;
            for (const std::string& st : gp.stop) longest = std::max(longest, st.size());
            if (longest > 1) {
                const size_t tail = std::min(text_base.size(), longest - 1);
                gp.stop_seed = text_base.substr(text_base.size() - tail);
            }
        }
        if (gp.temperature > 0.0f) {
            // R10: the RNG stream cannot be replayed; a fresh seed derived from
            // (job seed, position) keeps THIS resume reproducible, disclosed.
            gp.seed = static_cast<uint32_t>(
                (gp.seed * 2654435761u) ^ static_cast<uint32_t>(job->tokens_out)) | 1u;
            auto lk = jobs.lock();
            job->gp.seed = gp.seed;   // disclosed via the poll response
        }
        return Restore::Resumed;
    }
    if (lerr.rfind("TRANSIENT:", 0) == 0) {
        // H1: a TRANSIENT refusal (cap has no room right now) keeps EVERYTHING
        // and fails the job for this session so the next start retries with
        // the state intact.
        std::fprintf(stderr, "serve: job %s restore deferred (%s)\n", id.c_str(), lerr.c_str());
        // I5: a deferral is NOT a resume attempt -- the engine never ran.
        {
            auto lk = jobs.lock();
            if (job->attempts > 0) --job->attempts;
            job->status = "deferred";   // non-terminal word: next start retries
            job->error = lerr;
            job->ckpt_note = lerr;
            job->finished_at = std::chrono::steady_clock::now();
        }
        ctx.store->persist_attempts(id, job->attempts);
        return Restore::Deferred;
    }
    // Checkpoint PERMANENTLY unusable: restart from the prompt rather than fail.
    // H1/H2: discard the dead blob NOW instead of parking its hash in prev_hash.
    if (lerr.empty()) lerr = "no indexed checkpoint for this hash";
    // I6: load's permanent failure already DE-INDEXED the entry while leaving
    // the file, so unlink by the PATH the lookup handed us.
    if (entry) {
        std::error_code uec;
        std::filesystem::remove(entry->path, uec);
    } else if (job->resume_hash) {
        snap.discard(job->resume_hash);   // never loaded: still indexed
    }
    std::fprintf(stderr, "serve: job %s checkpoint unusable (%s); restarting from prompt\n",
                 id.c_str(), lerr.c_str());
    auto lk = jobs.lock();
    job->text.clear();
    job->tokens_out = 0;
    job->loaded_tokens = 0;     // H4
    job->resume_hash = 0;
    text_base.clear();          // F8: no seeded prefix on a fresh restart
    gp.resume_tokens_done = 0;  // F5: counts restart absolute-from-zero
    return Restore::RestartFromPrompt;
}

}  // namespace

JobWorker::JobWorker(ServerContext ctx) : ctx_(std::move(ctx)) {
    thread_ = std::thread([this] { run(); });
}

void JobWorker::join() { thread_.join(); }

void JobWorker::run() {
    ckpt_every_ = checkpoint_every();
    for (;;) {
        std::string id;
        std::shared_ptr<Job> job = ctx_.jobs->take_next(&id);
        if (!job) return;   // stopping
        process(id, job);
    }
}

namespace {

// One job's session. Its Request callbacks run on the scheduler's thread after
// process() has returned, so everything they touch lives here, shared.
struct JobRun {
    ServerContext ctx;
    std::string   id;
    std::shared_ptr<Job> job;
    GenParams     gp;          // what the session runs with; restore adjusts it
    std::string   text_base;   // F8: the resumed prefix
    std::unique_ptr<Checkpointer> ckpt;   // refers to gp, id, job above
    bool          deferred = false;       // restore deferred: state already recorded
};

// A job's terminal state, from the session's result.
void finalize(JobRun& run, GenResult r) {
    JobRegistry& jobs = *run.ctx.jobs;
    const std::shared_ptr<Job>& job = run.job;
    // T1: checkpoint state is destroyed only on genuine completion or an
    // EXPLICIT user cancel; shutdown-cancel and failures keep it for resume.
    bool user_cancelled;
    {
        auto lk = jobs.lock();
        // F3: stopping alone races -- the watcher cancels jobs BEFORE calling
        // stop(). The stop flag is written strictly before the watcher touches
        // any job and never by user cancel.
        user_cancelled = r.cancelled && !jobs.stopping() && !stop_requested();
    }
    const bool done_clean = r.error.empty() && !r.aborted && !r.cancelled;
    if (run.ctx.store && (done_clean || user_cancelled)) {
        run.ctx.store->remove_sidecar(run.id);
        if (run.ckpt->prev_hash) run.ctx.store->checkpoints().discard(run.ckpt->prev_hash);
    }
    auto lk = jobs.lock();
    // F14: prompt tokens for usage (0 on a resume: unknown there, T9).
    if (r.tokens_in > 0) job->tokens_in = r.tokens_in;
    // T7/F5: the engine's true count, made absolute across restarts.
    job->tokens_out = run.gp.resume_tokens_done + r.tokens_out;
    // F8: r.text is the authority for THIS session; base_trim is the resume
    // straddle, where the stop began inside text_base.
    std::string base = run.text_base;
    if (r.base_trim > 0 && r.base_trim <= base.size()) base.erase(base.size() - r.base_trim);
    job->text = base + r.text;
    // Swarm S1+S7: aborted counts as failed, and the error must not destroy the
    // partial output. A shutdown-cancel is NOT terminal: "interrupted", and the
    // next boot resumes it.
    job->status = r.cancelled ? (user_cancelled ? "cancelled" : "interrupted")
                : (r.error.empty() && !r.aborted) ? "completed" : "failed";
    job->finished_at = std::chrono::steady_clock::now();
    if (!r.error.empty()) job->error = r.error;
    else if (r.aborted) job->error = "aborted: a weight failed to materialise; output untrustworthy";
}

void fail_job(JobRun& run, const std::string& why) {
    auto lk = run.ctx.jobs->lock();
    run.job->status = "failed";
    run.job->error = why;
    run.job->finished_at = std::chrono::steady_clock::now();
}

}  // namespace

// Submits the job and returns: the scheduler runs it in a slot of its own,
// alongside foreground requests and other jobs. Swarm S2: nothing thrown in a
// callback may escape onto the scheduler's thread -- it serves everyone.
void JobWorker::process(const std::string& id, const std::shared_ptr<Job>& job) {
    auto run = std::make_shared<JobRun>();
    run->ctx = ctx_;
    run->id = id;
    run->job = job;
    run->gp = job->gp;
    auto jobs_sp = ctx_.jobs;
    // Shutdown must also stop the job's generation, even when listen()
    // returned for a reason other than the watcher (2026-08-24 audit).
    run->gp.should_continue = [job, jobs_sp] {
        return !job->cancel.load(std::memory_order_relaxed) && !jobs_sp->stopping();
    };
    {
        auto lk = ctx_.jobs->lock();
        run->text_base = job->text;
    }
    run->ckpt = std::make_unique<Checkpointer>(run->ctx, run->id, run->job, run->gp, ckpt_every_);
    if (!job->orig_max) job->orig_max = run->gp.max_tokens;

    engine::Request q;
    q.params = run->gp;
    q.sink = [run](const std::string& piece) {
        auto lk = run->ctx.jobs->lock();
        run->job->text += piece;
        ++run->job->tokens_out;
    };
    if (job->resume && ctx_.store) {
        q.restore = [run](int32_t seq, GenParams& p) {
            try {
                const Restore how = restore_checkpoint(run->ctx, run->id, run->job, p,
                                                       run->text_base, run->ckpt->prev_hash, seq);
                run->gp = p;   // the checkpointer and finalize see the resumed session
                if (how == Restore::Deferred) {
                    run->deferred = true;
                    return engine::Restored::Abandon;
                }
                return how == Restore::Resumed ? engine::Restored::Resumed : engine::Restored::Prefill;
            } catch (const std::exception& ex) {
                fail_job(*run, std::string("internal: restore: ") + ex.what());
            } catch (...) {
                fail_job(*run, "internal: restore: unknown exception");
            }
            run->deferred = true;   // terminal state already recorded
            return engine::Restored::Abandon;
        };
    }
    if (ctx_.store) {
        q.safepoint = [run](int32_t seq, int32_t next_tok, int32_t done) {
            try {
                run->ckpt->on_safepoint(seq, next_tok, done);
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "serve: job %s checkpoint threw: %s\n", run->id.c_str(), ex.what());
            } catch (...) {
                std::fprintf(stderr, "serve: job %s checkpoint threw\n", run->id.c_str());
            }
        };
    }
    q.done = [run](GenResult r) {
        if (run->deferred) return;
        try {
            finalize(*run, std::move(r));
        } catch (const std::exception& ex) {
            fail_job(*run, std::string("internal: ") + ex.what());
        } catch (...) {
            fail_job(*run, "internal: unknown exception");
        }
    };
    ctx_.sched->submit(std::move(q));
}

}  // namespace dray::server
