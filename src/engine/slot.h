// One sequence's life in the scheduler: a request's prompt, its position in its
// slot's private KV, the token waiting to be decoded, and the text assembled so
// far. Plain state and small steps; the Scheduler decides when each happens.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine/engine_types.h"
#include "engine/streamed_text.h"

namespace dray::engine {

// How a request's restore hook left its slot.
enum class Restored : uint8_t {
    Resumed,   // saved state is in the slot; decode params.resume_pending next
    Prefill,   // nothing to resume after all: start from the prompt
    Abandon,   // do not run now (done is still called, as cancelled)
};

// A request as the scheduler receives it. Every callback runs on the
// scheduler's thread -- the only thread that touches the model.
struct Request {
    GenParams params;
    TokenSink sink;                        // each delivered piece
    std::function<void(GenResult)> done;   // exactly once, at the end
    // Optional. At admission, after the slot's KV was emptied: load saved state
    // into `seq` and adjust `params` for the resumed session (remaining budget,
    // resume_tokens_done, stop seed, seed).
    std::function<Restored(int32_t seq, GenParams& params)> restore;
    // Optional. After a token is sampled and BEFORE it is decoded -- the state a
    // checkpoint can resume from, the token being the one to re-decode --
    // with the absolute count of tokens done (resume_tokens_done included).
    std::function<void(int32_t seq, int32_t next_token, int32_t tokens_done)> safepoint;
};

class Slot {
public:
    enum class State : uint8_t { Free, Prefilling, Decoding };

    explicit Slot(int32_t index) : index_(index) {}
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

    int32_t index() const { return index_; }
    State   state() const { return state_; }
    bool    free() const { return state_ == State::Free; }
    uint64_t request_id() const { return id_; }

    // Takes the request; the prompt is already tokenized and fits the slot.
    // `reuse`: prompt tokens already in the slot's KV (a kept conversation's
    // prefix) -- prefill starts after them. `checkpoint_at`: the prompt position
    // where prefill must pause for a checkpoint (-1 = none).
    void begin(uint64_t id, Request request, std::vector<int32_t> prompt,
               int32_t reuse = 0, int32_t checkpoint_at = -1);
    // Prefill has reached the checkpoint position and waits for it.
    bool checkpoint_due() const {
        return state_ == State::Prefilling && checkpoint_at_ >= 0 && !checkpointed_ &&
               fed_ == checkpoint_at_;
    }
    void mark_checkpointed() { checkpointed_ = true; }
    // Exactly the tokens in this slot's KV, when known (not after a job resume).
    bool known_history() const { return known_history_; }
    std::vector<int32_t> kv_tokens() const;
    // Takes a request whose saved state is already in the slot's KV (`n_past`
    // tokens); it starts decoding at params.resume_pending.
    void begin_resumed(uint64_t id, Request request, int32_t n_past);
    const Request& request() const { return request_; }
    // Absolute tokens done: earlier sessions' plus this one's.
    int32_t tokens_done() const { return request_.params.resume_tokens_done + result_.tokens_out; }
    // Delivers held-back bytes now (before a checkpoint reads the text).
    void    flush_text() { if (text_) text_->flush(); }

    // --- prefill -------------------------------------------------------------
    int32_t prefill_remaining() const { return static_cast<int32_t>(prompt_.size()) - fed_; }
    // The next prompt token and its position; true when it is the LAST one
    // (the step must request its logits).
    bool    next_prompt_token(int32_t* token, int32_t* pos);

    // --- decode --------------------------------------------------------------
    int32_t pending() const { return pending_; }
    int32_t position() const { return n_past_; }   // where the pending token goes
    // The first token, sampled from the last prompt token's logits.
    void    start_decoding(int32_t first);
    // The pending token was decoded: deliver its text. Returns false when the
    // request is complete (stop sequence, or max_tokens emitted).
    bool    emit(const std::string& piece);
    void    set_pending(int32_t token) { pending_ = token; }

    // --- end -----------------------------------------------------------------
    GenResult& result() { return result_; }
    bool       cancel_requested() const;
    void       request_cancel() { cancel_ = true; }
    // Flushes held-back text, hands the result to the request's `done` and
    // returns the slot to Free.
    void       finish();

private:
    const int32_t index_;
    State    state_ = State::Free;
    uint64_t id_ = 0;
    bool     cancel_ = false;
    Request  request_;
    std::vector<int32_t> prompt_;
    std::vector<int32_t> history_;   // prompt, then every emitted (decoded) token
    bool     known_history_ = true;
    int32_t  checkpoint_at_ = -1;
    bool     checkpointed_ = false;
    int32_t  fed_ = 0;        // prompt tokens already in a step
    int32_t  n_past_ = 0;     // tokens in this slot's KV
    int32_t  pending_ = 0;    // sampled, not yet decoded
    GenResult result_;
    std::unique_ptr<StreamedText> text_;   // refers into request_: never moves
};

}  // namespace dray::engine
