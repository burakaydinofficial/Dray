// The in-memory job registry: background requests return an id immediately and
// the worker drains them FIFO into the scheduler, where they run alongside
// foreground requests -- a slot does not care who asked. Text accumulates as tokens
// land, so a poll mid-generation shows honest partial output.
//
// LOCKING. One mutex guards the registry AND every Job's fields. Methods named
// *_locked require the caller to hold lock(); the others take it themselves.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "engine/engine_types.h"

namespace dray::server {

struct Job {
    std::string status = "queued";   // queued -> in_progress -> completed/failed/cancelled
                                     // (also interrupted, deferred, quarantined)
    std::string text;
    std::string error;
    std::string ckpt_note;           // F4: last checkpoint failure, surfaced on the poll
    int32_t     tokens_in = -1;      // F14: prompt tokens; -1 = unknown (renders null)
    int32_t     attempts = 0;        // F15/G6: consecutive NO-PROGRESS resumes
    int32_t     loaded_tokens = 0;   // G6: sidecar position at load, the progress bar
    std::atomic<bool> cancel{false};
    std::chrono::steady_clock::time_point finished_at{};   // T7: TTL reaping
    int32_t     tokens_out = 0;
    engine::GenParams gp;
    // Crash-resume state, populated by the startup scan from a sidecar.
    bool        resume = false;
    int32_t     resume_pending = 0;
    uint64_t    resume_hash = 0;
    int32_t     orig_max = 0;
};

class JobRegistry {
public:
    // Bounded, on a process whose premise is a byte cap: each job holds its
    // prompt and accumulating text outside the accountant. 256 pending jobs on
    // an engine that serializes admission is already days of work.
    static constexpr size_t kMaxPending = 256;

    std::unique_lock<std::mutex> lock() const { return std::unique_lock<std::mutex>(mu_); }

    // New response ids: "ldg-N", monotonic for the process. seed_ids_past makes
    // resumed jobs' original ids unreachable by new requests.
    std::string next_id();
    void seed_ids_past(uint64_t n);

    // Queues a new job and wakes the worker. False when kMaxPending are queued.
    bool submit(const std::string& id, std::shared_ptr<Job> job);

    // Blocks until a job is queued or the registry stops. Marks the job
    // in_progress. Returns null once stopping.
    std::shared_ptr<Job> take_next(std::string* id);

    std::shared_ptr<Job> find_locked(const std::string& id) const;
    // Registers a job from the startup scan: runnable ones are also queued.
    void register_locked(const std::string& id, std::shared_ptr<Job> job, bool runnable);
    void dequeue_locked(const std::string& id);
    void erase_locked(const std::string& id);
    void cancel_all_locked();
    // T7: forget terminal jobs finished more than max_age ago.
    void reap_terminal_locked(std::chrono::seconds max_age);

    // Ends take_next for good (the drain).
    void stop();
    // Readable without the lock (the worker's predicates).
    bool stopping() const { return stopping_.load(); }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<std::string, std::shared_ptr<Job>> by_id_;
    std::deque<std::string> fifo_;
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> id_counter_{0};
};

// Checkpoint key: unique per (job, position). Semantic prefix reuse is not the
// goal -- surviving a crash is -- so the key only has to never collide.
uint64_t job_ckpt_hash(const std::string& id, int32_t tokens_done);

// Ids that may reach the filesystem: [A-Za-z0-9-] only (S27).
bool is_safe_job_id(const std::string& id);

}  // namespace dray::server
