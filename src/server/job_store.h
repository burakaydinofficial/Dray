// Durable job state under --jobs-dir: one JSON sidecar per interrupted job
// (<id>.job; <id>.job.failed once quarantined) plus its checkpoint blob in the
// SnapshotCache.
//
// CHECKPOINT + CRASH-RESUME, shipped together. Every background job checkpoints
// at safepoints (state BEFORE the pending token, the token in the sidecar --
// snaptest's proven shape) and the startup scan re-enqueues interrupted jobs
// under their ORIGINAL ids. GREEDY jobs resume with text identical to an
// uninterrupted run; SAMPLED jobs resume PLAUSIBLY (R10: llama's sampler API
// cannot replay an RNG stream mid-flight) with a disclosed derived seed.
#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "json.hpp"

namespace dray::cache { class SnapshotCache; }
namespace dray::engine { class Engine; }

namespace dray::server {

class JobRegistry;

class JobStore {
public:
    // Opens the snapshot cache in dir, stamped to this model and engine. On
    // failure returns null and sets *error ("jobs stamp: ..." / "jobs dir: ...").
    static std::unique_ptr<JobStore> open(const std::string& dir, const std::string& model_path,
                                          engine::Engine& engine, std::string* error);
    ~JobStore();

    // Startup scan: every sidecar is an interrupted job, re-registered under
    // its original id and re-queued -- unless it exceeded the resume-attempt
    // ceiling, in which case it is quarantined and registered as terminal.
    void scan(JobRegistry& registry);

    std::filesystem::path sidecar_path(const std::string& id) const;
    // Publishes a sidecar durably (fsync before rename). False = not published;
    // the previous sidecar remains valid.
    bool publish(const std::string& id, const nlohmann::json& sidecar);
    // I5: a deferred restore is not a resume attempt; hand the count back.
    void persist_attempts(const std::string& id, int32_t attempts);
    void remove_sidecar(const std::string& id);
    // DELETE: the sidecar AND its quarantined name, or the id resurrects.
    void remove_all_files(const std::string& id);

    cache::SnapshotCache& checkpoints() { return *snap_; }

    // fsync-before-rename, to the standard snapshot_cache.cpp states. Writes
    // "<target>.new" -- NOT ".tmp", which SnapshotCache::open() sweeps at boot,
    // destroying the flushed record exactly when recovery would need it.
    static bool write_json_durably(const std::filesystem::path& target,
                                   const nlohmann::json& j, int indent);

private:
    JobStore() = default;
    std::string dir_;
    std::unique_ptr<cache::SnapshotCache> snap_;
};

}  // namespace dray::server
