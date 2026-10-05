#include "engine/slot.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace dray::engine {

void Slot::begin(uint64_t id, Request request, std::vector<int32_t> prompt,
                 int32_t reuse, int32_t checkpoint_at) {
    id_ = id;
    cancel_ = false;
    request_ = std::move(request);
    prompt_ = std::move(prompt);
    history_ = prompt_;
    known_history_ = true;
    checkpoint_at_ = checkpoint_at;
    checkpointed_ = false;
    fed_ = reuse;      // already in the KV
    n_past_ = reuse;
    pending_ = 0;
    result_ = GenResult{};
    result_.tokens_in = static_cast<int32_t>(prompt_.size());
    // StreamedText keeps references to the stops, the seed and the sink: all
    // live in request_, which stays put for the whole request.
    text_ = std::make_unique<StreamedText>(&result_, request_.params.stop,
                                           request_.params.stop_seed, request_.sink);
    state_ = State::Prefilling;
}

void Slot::begin_resumed(uint64_t id, Request request, int32_t n_past) {
    begin(id, std::move(request), {});
    known_history_ = false;   // the restored state's tokens are not known here
    result_.tokens_in = 0;   // unknown on a resume (T9)
    n_past_ = n_past;
    pending_ = request_.params.resume_pending;
    state_ = State::Decoding;
}

std::vector<int32_t> Slot::kv_tokens() const {
    const size_t n = std::min(history_.size(), static_cast<size_t>(n_past_));
    return std::vector<int32_t>(history_.begin(), history_.begin() + static_cast<std::ptrdiff_t>(n));
}

bool Slot::next_prompt_token(int32_t* token, int32_t* pos) {
    *token = prompt_[static_cast<size_t>(fed_)];
    *pos = fed_;
    ++fed_;
    ++n_past_;
    return fed_ == static_cast<int32_t>(prompt_.size());
}

void Slot::start_decoding(int32_t first) {
    pending_ = first;
    state_ = State::Decoding;
}

bool Slot::emit(const std::string& piece) {
    ++n_past_;   // the pending token is now in the KV
    history_.push_back(pending_);
    text_->feed(piece.data(), piece.size());
    ++result_.tokens_out;
    if (result_.stop_hit) {
        result_.truncated_by_eog = true;
        return false;
    }
    return result_.tokens_out < request_.params.max_tokens;
}

bool Slot::cancel_requested() const {
    if (cancel_) return true;
    const auto& keep_going = request_.params.should_continue;
    return keep_going && !keep_going();
}

void Slot::finish() {
    if (text_) text_->flush();
    GenResult out = std::move(result_);
    auto done = std::move(request_.done);
    text_.reset();
    request_ = Request{};
    prompt_.clear();
    state_ = State::Free;
    if (done) done(std::move(out));
}

}  // namespace dray::engine
