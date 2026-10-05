// The scheduler's boundary with the model: everything that touches llama.cpp
// (tokens, one mixed decode step, per-slot sampling and KV) and nothing else.
// The real implementation wraps an Engine (LlamaStepper); tests script one.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dray::engine {

class Stepper {
public:
    virtual ~Stepper() = default;

    // Slots the admission funded (llama sequences, each with its own KV).
    virtual int32_t slots() const = 0;
    // Tokens one slot's context holds (its private KV ring).
    virtual int32_t context_per_slot() const = 0;
    // Tokens one decode may carry (llama n_batch): decode + prefill together.
    virtual int32_t max_step_tokens() const = 0;

    virtual bool        tokenize(const std::string& text, std::vector<int32_t>* out) = 0;
    virtual std::string piece(int32_t token) = 0;
    virtual bool        is_end(int32_t token) = 0;

    // Empties a slot's KV (and recurrent state): every new request starts here.
    virtual void    clear_kv(int32_t slot) = 0;
    // Tokens currently in a slot's KV (after a checkpoint restore).
    virtual int32_t kv_tokens(int32_t slot) = 0;
    // Keeps the first `n` tokens of a slot's KV and drops the rest. False when
    // the memory cannot be cut back there: recurrent state holds only its
    // latest position (the KV is then left unchanged).
    virtual bool    truncate_kv(int32_t slot, int32_t n) = 0;
    // Recurrent and hybrid models: the part of a slot's state that cannot be
    // truncated, saved and restored whole (llama's partial state) -- a
    // checkpoint to come back to. needs_checkpoints() false = attention only,
    // where truncate_kv always works and checkpoints are never needed.
    virtual bool    needs_checkpoints() const = 0;
    virtual bool    save_checkpoint(int32_t slot, std::vector<uint8_t>* out) = 0;
    virtual bool    load_checkpoint(int32_t slot, const std::vector<uint8_t>& in) = 0;
    // A slot's WHOLE state (attention KV and recurrent state), to park a
    // conversation outside the slots and bring it back into any slot later.
    virtual bool    save_state(int32_t slot, std::vector<uint8_t>* out) = 0;
    virtual bool    load_state(int32_t slot, const std::vector<uint8_t>& in) = 0;
    // A slot starts a request: a sampler is made for it (temperature <= 0 =
    // greedy; seed 0 = llama's default seed). The KV is left as it is.
    virtual void open_slot(int32_t slot, float temperature, uint32_t seed) = 0;
    virtual void close_slot(int32_t slot) = 0;

    // One entry of a step: `token` at `pos` in `slot`'s sequence; `logits`
    // requests an output row (the last prompt token, every decode token).
    struct Entry {
        int32_t slot = 0;
        int32_t token = 0;
        int32_t pos = 0;
        bool    logits = false;
    };
    // One llama_decode over `batch`. False = the step failed.
    virtual bool decode(const std::vector<Entry>& batch) = 0;
    // Samples `slot`'s next token from output row `row` of the last decode
    // (`row` = the entry's index in the batch).
    virtual int32_t sample(int32_t slot, int32_t row) = 0;
    // A weight failed to materialise: the step's output is not trustworthy.
    virtual bool tainted() const = 0;
    // After every successful step (progress readout, cap latching).
    virtual void after_step(int64_t /*steps*/) {}
};

}  // namespace dray::engine
