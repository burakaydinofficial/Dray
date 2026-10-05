// The background worker: drains the job registry FIFO into the scheduler, where
// each job runs in a slot of its own alongside foreground requests. Resumed jobs
// restore their checkpoint into that slot, every job checkpoints at safepoints,
// and each records its terminal state -- all on the scheduler's thread.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "server/server_context.h"

namespace dray::server {

struct Job;

class JobWorker {
public:
    explicit JobWorker(ServerContext ctx);
    void join();

private:
    void run();
    void process(const std::string& id, const std::shared_ptr<Job>& job);

    ServerContext ctx_;
    int32_t ckpt_every_ = 256;   // DRAY_CKPT_EVERY, read once at start
    std::thread thread_;
};

}  // namespace dray::server
