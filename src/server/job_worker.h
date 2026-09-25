// The single background worker: drains the job registry FIFO under the same
// admission gate as foreground requests, restores checkpoints for resumed jobs,
// checkpoints at safepoints, and records each job's terminal state.
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
