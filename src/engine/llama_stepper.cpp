#include "engine/llama_stepper.h"

#include "llama.h"

#include "engine/engine.h"

namespace dray::engine {

LlamaStepper::LlamaStepper(Engine& engine) : engine_(engine) {
    samplers_.assign(static_cast<size_t>(slots()), nullptr);
}

LlamaStepper::~LlamaStepper() {
    for (llama_sampler* s : samplers_) {
        if (s) llama_sampler_free(s);
    }
}

int32_t LlamaStepper::slots() const {
    return static_cast<int32_t>(engine_.funded_sequences());
}

int32_t LlamaStepper::context_per_slot() const {
    // J3: under n_seq > 1 llama_n_ctx() is the aggregate; each sequence owns
    // llama_n_ctx_seq cells of its private ring.
    return static_cast<int32_t>(llama_n_ctx_seq(engine_.context()));
}

int32_t LlamaStepper::max_step_tokens() const {
    return static_cast<int32_t>(llama_n_batch(engine_.context()));
}

bool LlamaStepper::tokenize(const std::string& text, std::vector<int32_t>* out) {
    out->assign(text.size() + 8, 0);
    int32_t n = llama_tokenize(engine_.vocab(), text.c_str(), static_cast<int32_t>(text.size()),
                               out->data(), static_cast<int32_t>(out->size()), true, true);
    if (n < 0) {
        out->assign(static_cast<size_t>(-n), 0);
        n = llama_tokenize(engine_.vocab(), text.c_str(), static_cast<int32_t>(text.size()),
                           out->data(), static_cast<int32_t>(out->size()), true, true);
    }
    if (n < 0) return false;
    out->resize(static_cast<size_t>(n));
    return true;
}

std::string LlamaStepper::piece(int32_t token) {
    char buf[256];
    const int32_t n = llama_token_to_piece(engine_.vocab(), token, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

bool LlamaStepper::is_end(int32_t token) {
    return llama_vocab_is_eog(engine_.vocab(), token);
}

void LlamaStepper::clear_kv(int32_t slot) {
    // This sequence's KV and recurrent state, nothing else's.
    llama_memory_seq_rm(llama_get_memory(engine_.context()), slot, -1, -1);
}

int32_t LlamaStepper::kv_tokens(int32_t slot) {
    return static_cast<int32_t>(llama_memory_seq_pos_max(llama_get_memory(engine_.context()), slot)) + 1;
}

void LlamaStepper::open_slot(int32_t slot, float temperature, uint32_t seed) {
    close_slot(slot);
    llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature > 0.0f) {
        llama_sampler_chain_add(s, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist(seed == 0 ? LLAMA_DEFAULT_SEED : seed));
    } else {
        llama_sampler_chain_add(s, llama_sampler_init_greedy());
    }
    samplers_[static_cast<size_t>(slot)] = s;
}

void LlamaStepper::close_slot(int32_t slot) {
    llama_sampler*& s = samplers_[static_cast<size_t>(slot)];
    if (s) llama_sampler_free(s);
    s = nullptr;
}

bool LlamaStepper::decode(const std::vector<Entry>& batch) {
    llama_batch b = llama_batch_init(static_cast<int32_t>(batch.size()), 0, 1);
    for (size_t i = 0; i < batch.size(); ++i) {
        b.token[i] = batch[i].token;
        b.pos[i] = batch[i].pos;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = batch[i].slot;
        b.logits[i] = static_cast<int8_t>(batch[i].logits);
    }
    b.n_tokens = static_cast<int32_t>(batch.size());
    const bool ok = llama_decode(engine_.context(), b) == 0;
    llama_batch_free(b);
    return ok;
}

int32_t LlamaStepper::sample(int32_t slot, int32_t row) {
    return llama_sampler_sample(samplers_[static_cast<size_t>(slot)], engine_.context(), row);
}

bool LlamaStepper::tainted() const { return engine_.weights_failed(); }

void LlamaStepper::after_step(int64_t steps) {
    engine_.note_progress(static_cast<int32_t>(steps));
    engine_.latch_over_cap();
    engine_.bind_cap_to_process();
}

}  // namespace dray::engine

namespace dray::engine {

bool LlamaStepper::truncate_kv(int32_t slot, int32_t n) {
    // False when the memory cannot drop [n, end): recurrent state keeps only
    // its latest position. The KV is then left as it was.
    return llama_memory_seq_rm(llama_get_memory(engine_.context()), slot, n, -1);
}

bool LlamaStepper::needs_checkpoints() const {
    return llama_model_is_recurrent(engine_.model()) || llama_model_is_hybrid(engine_.model());
}

bool LlamaStepper::save_checkpoint(int32_t slot, std::vector<uint8_t>* out) {
    // The part of the state that cannot be truncated (recurrent), alone.
    const size_t need = llama_state_seq_get_size_ext(engine_.context(), slot,
                                                     LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (need == 0) return false;
    out->resize(need);
    const size_t got = llama_state_seq_get_data_ext(engine_.context(), out->data(), need, slot,
                                                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (got == 0) return false;
    out->resize(got);
    return true;
}

bool LlamaStepper::load_checkpoint(int32_t slot, const std::vector<uint8_t>& in) {
    return llama_state_seq_set_data_ext(engine_.context(), in.data(), in.size(), slot,
                                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == in.size();
}

}  // namespace dray::engine

namespace dray::engine {

bool LlamaStepper::save_state(int32_t slot, std::vector<uint8_t>* out) {
    const size_t need = llama_state_seq_get_size(engine_.context(), slot);
    if (need == 0) return false;
    out->resize(need);
    const size_t got = llama_state_seq_get_data(engine_.context(), out->data(), need, slot);
    if (got == 0) return false;
    out->resize(got);
    return true;
}

bool LlamaStepper::load_state(int32_t slot, const std::vector<uint8_t>& in) {
    return llama_state_seq_set_data(engine_.context(), in.data(), in.size(), slot) == in.size();
}

}  // namespace dray::engine
