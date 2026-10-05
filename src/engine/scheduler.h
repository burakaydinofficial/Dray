// Continuous batching: many requests share one engine, each in its own slot
// (its own KV), all advancing together.
//
// Every step is ONE decode carrying a token for every decoding slot plus up to
// `prefill_tokens_per_step` prompt tokens of slots still prefilling -- so a new
// request joins at any step without stalling the others, and every step's
// weight reads (the unconditional stream, the union of routed experts) are
// shared by all of them. That sharing is this engine's whole case for serving
// many clients at once: measured on Qwen3.8-Flash-Next, 64 sequences decode at
// ~4 tokens/s in total against ~1 for one.
//
// Semantics match BatchGenerator exactly -- a token's text is delivered after
// the step that decodes it, sampling uses the output row of that same step --
// so a set of requests admitted together produces what `dray batch` does.
//
// Threading: submit/cancel from any thread; everything else on the scheduler's
// own loop thread, including every Request callback. step() is public so tests
// (and a synchronous caller) can drive the loop without the thread.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "engine/kv_store.h"
#include "engine/slot.h"
#include "engine/stepper.h"

namespace dray::engine {

class Scheduler {
public:
    struct Config {
        // Prompt tokens a step may carry besides the decode tokens: the latency
        // knob. Larger admits new requests faster; smaller keeps the running
        // streams' steps short. Capped by the step's size (n_batch).
        int32_t prefill_tokens_per_step = 256;
    };

    // `store` (optional, not owned; must outlive the scheduler): keeps finished
    // conversations and chooses slots so a new prompt reuses a kept prefix.
    Scheduler(Stepper& stepper, Config config, KvStore* store = nullptr);
    ~Scheduler();   // stops the loop; every unfinished request completes as cancelled
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Thread-safe. Queued until a slot is free; returns the request id.
    uint64_t submit(Request request);
    // Thread-safe. A queued request completes as cancelled without running; an
    // active one ends at the next step boundary.
    void     cancel(uint64_t id);

    void start();   // runs step() on a loop thread until stop()
    void stop();

    // Thread-safe. Runs `fn` on the loop thread between two steps (at once when
    // idle) -- the only safe place to read engine internals while requests run
    // (the streamer's containers are the decode thread's). Without a running
    // loop, runs at the next step().
    std::future<void> post(std::function<void()> fn);
    // Thread-safe. Waits until no request is active or queued; false on timeout.
    bool wait_idle(std::chrono::milliseconds timeout);

    // One iteration: admit, cancel, one decode step, deliver. Returns false
    // when there is nothing to do (no active slot, empty queue).
    bool step();

    struct Stats {
        int64_t steps = 0;
        int64_t decode_tokens = 0;
        int64_t prefill_tokens = 0;
        int32_t active = 0;
        int32_t prefilling = 0;   // active slots still feeding their prompt
        int64_t reused_requests = 0;   // prompts that started from a kept conversation
        int64_t reused_tokens = 0;     // prompt tokens those did not prefill again
        int32_t waiting = 0;
    };
    Stats stats() const;

private:
    struct Queued { uint64_t id; Request request; };

    void admit();
    void finish(Slot& slot);
    void retire();   // a request got its done: one fewer in flight

    Stepper&  stepper_;
    KvStore*  store_ = nullptr;
    const Config cfg_;
    std::vector<std::unique_ptr<Slot>> slots_;
    int32_t   next_prefill_ = 0;   // round-robin start among prefilling slots

    mutable std::mutex      mu_;       // guards queue_, cancels_, running_, stats_
    std::condition_variable wake_;
    std::deque<Queued>      queue_;
    std::vector<uint64_t>   cancels_;
    std::vector<std::packaged_task<void()>> posted_;
    std::condition_variable idle_;     // signalled when in_flight_ drops to 0
    int64_t                 in_flight_ = 0;   // submitted, done not yet called
    uint64_t                next_id_ = 1;
    bool                    running_ = false;
    Stats                   stats_;
    std::thread             loop_;
};

}  // namespace dray::engine
