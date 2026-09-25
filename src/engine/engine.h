// The engine: one loaded model and its streaming context, reused across
// requests. `run`, `batch`, `snaptest` and `serve` all go through it; only the
// --no-stream reference path in main.cpp keeps its own independent setup, BY
// DESIGN -- it is the differential test's reference half and must not share the
// code it checks.
//
// Serialized admission is the caller's job (Invariant 7): generate() assumes one
// generation at a time, and the server holds a mutex across each request.
//
// OWNERSHIP. Engine owns, in construction order: the Plan, the Accountant (the
// cap ledger), the Streamer (the weight streaming machinery), the merged GGUF
// metadata, and llama's model and context. Nothing outside this class touches
// the Streamer; generators and consumers see only the narrow surface below.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "engine/engine_types.h"
#include "engine/streamed_text.h"

struct llama_model;
struct llama_context;
struct llama_vocab;
namespace dray::mem { class Accountant; }
namespace dray::plan { struct Plan; }
namespace dray::backend { class Streamer; struct MergedMetadata; }

namespace dray::engine {

struct LoadState;   // engine_internal.h

class Engine {
public:
    // Opens the full stack. Prints the plan and integrity report exactly like a
    // run always has (the operator of a server deserves the same honesty as the
    // operator of a run). On failure returns null and sets *error; an error
    // beginning "REFUSED" is an admission refusal, not a fault.
    static std::unique_ptr<Engine> open(const EngineConfig& config, std::string* error);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // One generation. on_piece (may be empty) receives each detokenized piece
    // as it is produced -- the streaming hook. Clears KV/recurrent state first
    // unless resuming: requests are independent in v1.
    GenResult generate(const GenParams& params, const TokenSink& on_piece);

    // --- llama handles, for persistence (capture/restore between decodes),
    //     chat templates and diagnostics.
    llama_context*     context() const { return lctx_; }
    const llama_vocab* vocab() const   { return vocab_; }
    const llama_model* model() const   { return model_; }

    // --- readouts. Read counters only when no generation is running, or from
    //     the generating thread: the underlying streamer counters are plain
    //     integers by design. live_* are atomics, safe from any thread.
    EngineCounters counters() const;
    std::string plan_report() const;
    std::string streamer_report() const;
    std::string accountant_report() const;
    const std::string& model_id() const { return model_id_; }
    uint64_t failures() const;
    bool tainted() const { return failures() > 0; }   // any weight ever failed
    int32_t live_tokens() const { return live_tokens_.load(std::memory_order_relaxed); }
    bool live_over_cap() const  { return live_over_cap_.load(std::memory_order_relaxed); }

    // --- the cap ledger, so checkpoint blobs cannot bypass it (T6), and the
    //     standing checkpoint allowance reserved at open (G1; 0 = none).
    mem::Accountant* accountant() const { return acct_.get(); }
    uint64_t checkpoint_allowance() const { return ckpt_allowance_; }

    // --- for the generators in this module (BatchGenerator, CohortRotator).
    uint32_t funded_sequences() const;   // sequences the admission funded
    int32_t prefill_chunk() const { return prefill_batch_; }
    bool weights_failed() const;         // a weight failed to materialise
    void note_progress(int32_t tokens) { live_tokens_.store(tokens, std::memory_order_relaxed); }
    void latch_over_cap();               // records a cap breach for live readers
    // The multi-day discipline, per token: bind the cap on the worse of RSS and
    // commit as the run proceeds.
    void bind_cap_to_process();

private:
    Engine();

    // open(), in order. Each returns false with *error set; the destructor then
    // releases whatever was acquired (the old engine_close semantics).
    bool reserve_floor(std::string* error);
    bool start_streamer(const EngineConfig& config, bool gpu_consent, std::string* error);
    bool load_model(const EngineConfig& config, bool gpu_consent, std::string* error);
    bool verify_floor(std::string* error);
    bool check_storage_backend(std::string* error);
    bool create_context(const EngineConfig& config, bool gpu_consent, std::string* error);
    bool reserve_checkpoint_allowance(const EngineConfig& config, std::string* error);
    bool admit(const EngineConfig& config, std::string* error);

    // T13: written per token by the generating thread, readable lock-free by
    // /health while busy -- a cap breach hours into a generation must not wait
    // for the generation to end to be reportable.
    std::atomic<int32_t> live_tokens_{0};
    std::atomic<bool>    live_over_cap_{false};

    std::unique_ptr<plan::Plan>              plan_;
    std::unique_ptr<mem::Accountant>         acct_;
    std::unique_ptr<backend::Streamer>       streamer_;
    std::unique_ptr<backend::MergedMetadata> meta_;
    std::unique_ptr<LoadState>               load_;   // see engine_internal.h
    llama_model*       model_ = nullptr;
    llama_context*     lctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    std::string model_id_;
    uint64_t    cap_ = 0;
    uint64_t    ckpt_allowance_ = 0;   // G1: standing Misc reservation
    bool        resident_mode_ = false;
    // One prefill chunk. Client input must never reach
    // GGML_ASSERT(n_tokens <= n_batch) -- an abort, not an error return -- so
    // prompts are decoded in chunks of exactly this many tokens.
    int32_t     prefill_batch_ = 512;
};

}  // namespace dray::engine
