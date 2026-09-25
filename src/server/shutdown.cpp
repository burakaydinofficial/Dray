#include "server/shutdown.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "server/job_registry.h"

namespace dray::server {

namespace {

std::atomic<bool> g_stop_requested{false};

#if defined(_WIN32)
BOOL WINAPI dray_ctrl_handler(DWORD) { g_stop_requested.store(true); return TRUE; }
#else
void dray_sig_handler(int) { g_stop_requested.store(true); }
#endif

}  // namespace

bool stop_requested() { return g_stop_requested.load(std::memory_order_relaxed); }
void request_stop()   { g_stop_requested.store(true); }

void install_stop_handlers() {
#if defined(_WIN32)
    SetConsoleCtrlHandler(dray_ctrl_handler, TRUE);
#else
    std::signal(SIGINT, dray_sig_handler);
    std::signal(SIGTERM, dray_sig_handler);
#endif
}

StopWatcher::StopWatcher(httplib::Server& srv, std::shared_ptr<JobRegistry> jobs)
    : srv_(srv), jobs_(std::move(jobs)), thread_([this] { run(); }) {}

void StopWatcher::join() { thread_.join(); }

void StopWatcher::run() {
    int reap_tick = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        {
            auto lk = jobs_->lock();
            if (jobs_->stopping()) return;   // normal shutdown already draining
            // T7: reap terminal jobs after an hour, every ~30 s.
            if (++reap_tick >= 150) {
                reap_tick = 0;
                jobs_->reap_terminal_locked(std::chrono::seconds(3600));
            }
        }
        if (g_stop_requested.load()) break;
    }
    std::fprintf(stderr, "serve: stop requested; cancelling jobs and draining\n");
    {
        auto lk = jobs_->lock();
        jobs_->cancel_all_locked();
    }
    srv_.stop();
}

}  // namespace dray::server
