#include "server/job_registry.h"

namespace dray::server {

std::string JobRegistry::next_id() {
    return "ldg-" + std::to_string(++id_counter_);
}

void JobRegistry::seed_ids_past(uint64_t n) {
    uint64_t cur = id_counter_.load();
    while (cur < n && !id_counter_.compare_exchange_weak(cur, n)) {}
}

bool JobRegistry::submit(const std::string& id, std::shared_ptr<Job> job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (fifo_.size() >= kMaxPending) return false;
        by_id_[id] = std::move(job);
        fifo_.push_back(id);
    }
    cv_.notify_one();
    return true;
}

std::shared_ptr<Job> JobRegistry::take_next(std::string* id) {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return stopping_ || !fifo_.empty(); });
        if (stopping_) return nullptr;
        *id = fifo_.front();
        fifo_.pop_front();
        // find(), not operator[]: a fifo id with no registry entry used to
        // default-construct a null shared_ptr and deref it. Unreachable through
        // the normal transitions, but the startup scan's duplicate-registration
        // path perturbs exactly that invariant, and the guard is free.
        auto it = by_id_.find(*id);
        if (it == by_id_.end()) continue;
        it->second->status = "in_progress";
        return it->second;
    }
}

std::shared_ptr<Job> JobRegistry::find_locked(const std::string& id) const {
    auto it = by_id_.find(id);
    return it == by_id_.end() ? nullptr : it->second;
}

void JobRegistry::register_locked(const std::string& id, std::shared_ptr<Job> job, bool runnable) {
    by_id_[id] = std::move(job);
    if (runnable) fifo_.push_back(id);
}

void JobRegistry::dequeue_locked(const std::string& id) {
    for (auto q = fifo_.begin(); q != fifo_.end(); ++q) {
        if (*q == id) { fifo_.erase(q); break; }
    }
}

void JobRegistry::erase_locked(const std::string& id) { by_id_.erase(id); }

void JobRegistry::cancel_all_locked() {
    for (auto& kv : by_id_) kv.second->cancel.store(true);
}

void JobRegistry::reap_terminal_locked(std::chrono::seconds max_age) {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = by_id_.begin(); it != by_id_.end();) {
        const std::string& st = it->second->status;
        const bool terminal = st == "completed" || st == "failed" ||
                              st == "cancelled" || st == "quarantined" ||
                              st == "deferred";   // I4/I5
        if (terminal &&
            std::chrono::duration_cast<std::chrono::seconds>(now - it->second->finished_at) > max_age) {
            it = by_id_.erase(it);
        } else {
            ++it;
        }
    }
}

void JobRegistry::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopping_ = true;
    }
    cv_.notify_all();
}

uint64_t job_ckpt_hash(const std::string& id, int32_t tokens_done) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : id) { h ^= (uint8_t)c; h *= 0x100000001b3ull; }
    return h ^ static_cast<uint64_t>(tokens_done);
}

bool is_safe_job_id(const std::string& id) {
    return id.find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-") == std::string::npos;
}

}  // namespace dray::server
