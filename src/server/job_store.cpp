#include "server/job_store.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "cache/snapshot_cache.h"
#include "engine/engine.h"
#include "server/job_registry.h"

namespace dray::server {

using nlohmann::json;

namespace {

// H16: atoi yields 0 on garbage, and a 0 ceiling quarantines every job on its
// first resume -- the S11 lesson, again.
int attempt_ceiling() {
    static const int ceiling = [] {
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
    return ceiling;
}

uint64_t id_ordinal(const std::string& id) {
    return id.rfind("ldg-", 0) == 0 ? std::strtoull(id.c_str() + 4, nullptr, 10) : 0;
}

}  // namespace

JobStore::~JobStore() = default;

std::unique_ptr<JobStore> JobStore::open(const std::string& dir, const std::string& model_path,
                                         engine::Engine& eng, std::string* err) {
    cache::CompatStamp stamp;
    std::string serr;
    if (!cache::make_compat_stamp(model_path, eng.context(), "dray-jobs-v1", &stamp, &serr)) {
        *err = "jobs stamp: " + serr;
        return nullptr;
    }
    cache::Config cc;
    cc.dir = dir;
    // T6/F4: checkpoint blobs live inside the cap via the standing floor
    // allowance the engine reserved at open -- refusal happens once at
    // admission, never silently per safepoint at steady state. If reservation
    // fails at peak pressure the checkpoint is REFUSED and the store-failure
    // line says so (Invariant 1).
    cc.acct = eng.accountant();
    // G1: pass the allowance SIZE -- coverage is per-blob, never blanket.
    cc.prereserved_bytes = eng.checkpoint_allowance();
    // Pinned entries are not charged against the Reuse budget, but open()
    // demands a nonzero one -- its contract, honored rather than argued with.
    cc.budget_bytes = 1ull << 30;
    std::unique_ptr<JobStore> s(new JobStore());
    s->dir_ = dir;
    s->snap_ = std::make_unique<cache::SnapshotCache>(cc, stamp);
    if (!s->snap_->open(&serr)) {
        *err = "jobs dir: " + serr;
        return nullptr;
    }
    return s;
}

std::filesystem::path JobStore::sidecar_path(const std::string& id) const {
    return std::filesystem::path(dir_) / (id + ".job");
}

bool JobStore::write_json_durably(const std::filesystem::path& target, const json& j, int indent) {
    const std::filesystem::path tmp = target.string() + ".new";
    const std::string body = j.dump(indent, ' ', false, json::error_handler_t::replace);
#if defined(_WIN32)
    std::FILE* f = nullptr;
    fopen_s(&f, tmp.string().c_str(), "wb");
#else
    std::FILE* f = std::fopen(tmp.string().c_str(), "wb");
#endif
    if (!f) return false;
    const bool wrote = std::fwrite(body.data(), 1, body.size(), f) == body.size() &&
                       std::fflush(f) == 0;
#if defined(_WIN32)
    const bool synced = wrote && _commit(_fileno(f)) == 0;
#else
    const bool synced = wrote && fsync(fileno(f)) == 0;
#endif
    std::fclose(f);
    if (!synced) return false;
    std::error_code ec;
    std::filesystem::rename(tmp, target, ec);
    return !ec;
}

bool JobStore::publish(const std::string& id, const json& sidecar) {
    return write_json_durably(sidecar_path(id), sidecar, -1);
}

void JobStore::persist_attempts(const std::string& id, int32_t attempts) {
    const std::filesystem::path sp = sidecar_path(id);
    try {
        json dj;
        std::ifstream df(sp);
        df >> dj;
        dj["attempts"] = attempts;
        write_json_durably(sp, dj, 2);
    } catch (...) { /* prior sidecar remains; one extra counted attempt */ }
}

void JobStore::remove_sidecar(const std::string& id) {
    std::error_code ec;
    std::filesystem::remove(sidecar_path(id), ec);
}

void JobStore::remove_all_files(const std::string& id) {
    std::error_code ec;
    std::filesystem::remove(sidecar_path(id), ec);
    // I4: quarantine renamed the sidecar; DELETE must reach that name too or
    // the record resurrects at every start.
    std::filesystem::remove(std::filesystem::path(dir_) / (id + ".job.failed"), ec);
}

void JobStore::scan(JobRegistry& registry) {
    uint64_t max_id = 0;
    for (const auto& de : std::filesystem::directory_iterator(dir_)) {
        // Battery residual (leg B): a quarantined job's poll record died with
        // the process that quarantined it, 404ing an id whose durable evidence
        // sits right here. Register .job.failed files so the operator's poll
        // works across restarts too.
        if (de.path().extension() == ".failed") {
            try {
                json qj;
                std::ifstream qf(de.path());
                qf >> qj;
                const std::string qid = qj.value("id", "");
                if (!qid.empty() && is_safe_job_id(qid)) {
                    auto qjob = std::make_shared<Job>();
                    qjob->status = "quarantined";
                    qjob->error = "exceeded resume attempts; sidecar quarantined to .job.failed";
                    qjob->text = qj.value("text", "");
                    qjob->tokens_out = qj.value("tokens_done", 0);
                    qjob->resume_hash = qj.value("hash", 0ull);   // I4: DELETE needs it
                    qjob->finished_at = std::chrono::steady_clock::now();
                    max_id = std::max(max_id, id_ordinal(qid));
                    auto lk = registry.lock();
                    registry.register_locked(qid, qjob, false);
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
            // H6: id parse, validation and counter seeding come BEFORE the
            // quarantine branch -- a quarantined ordinal must never be reissued.
            const std::string id = sj.value("id", "");
            if (id.empty()) continue;
            if (!is_safe_job_id(id)) {
                std::fprintf(stderr, "serve: rejecting sidecar with suspicious id\n");
                continue;
            }
            max_id = std::max(max_id, id_ordinal(id));
            job->resume_hash = sj.value("hash", 0ull);
            // F15/G6: a job that keeps failing must not resurrect forever. The
            // count reaches DISK at scan time and counts CONSECUTIVE resumes
            // WITHOUT PROGRESS: any publish past the loaded position resets it.
            job->attempts = sj.value("attempts", 0) + 1;
            job->loaded_tokens = sj.value("tokens_done", 0);
            if (job->attempts > attempt_ceiling()) {
                std::error_code qec;
                std::filesystem::rename(de.path(),
                    std::filesystem::path(de.path().string() + ".failed"), qec);
                std::fprintf(stderr, "serve: job %s exceeded resume attempts; quarantined\n",
                             sj.value("id", "?").c_str());
                // G6: visible on the poll instead of a 404 that reads as "no
                // such job". Terminal; DELETE clears the record.
                job->status = "quarantined";
                job->error = "exceeded resume attempts; sidecar quarantined to .job.failed";
                job->finished_at = std::chrono::steady_clock::now();
                auto lk = registry.lock();
                registry.register_locked(id, job, false);
                continue;
            }
            // G6/H5: persist the incremented count through the SAME durable
            // pattern the publisher uses; truncate-in-place destroyed the only
            // copy on ENOSPC.
            sj["attempts"] = job->attempts;
            if (!write_json_durably(de.path(), sj, 2)) {
                std::fprintf(stderr, "serve: could not persist attempts for %s; "
                                     "prior sidecar remains valid\n",
                             sj.value("id", "?").c_str());
            }
            job->resume = true;
            job->resume_pending = sj.value("pending", 0);
            job->orig_max = job->gp.max_tokens;
            auto lk = registry.lock();
            registry.register_locked(id, job, true);
            std::fprintf(stderr, "serve: resuming job %s at %d tokens\n",
                         id.c_str(), job->tokens_out);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "serve: bad sidecar %s: %s\n",
                         de.path().string().c_str(), ex.what());
        }
    }
    registry.seed_ids_past(max_id);
}

}  // namespace dray::server
