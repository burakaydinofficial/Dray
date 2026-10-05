// The Stepper over a real Engine: llama_decode for a mixed step, one sampler
// per slot, per-slot KV (the slots are llama sequences, each with its private
// KV ring), and the engine's per-step duties (progress, cap latch and rebind).
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "engine/stepper.h"

struct llama_sampler;

namespace dray::engine {

class Engine;

class LlamaStepper final : public Stepper {
public:
    explicit LlamaStepper(Engine& engine);
    ~LlamaStepper() override;
    LlamaStepper(const LlamaStepper&) = delete;
    LlamaStepper& operator=(const LlamaStepper&) = delete;

    int32_t slots() const override;
    int32_t context_per_slot() const override;
    int32_t max_step_tokens() const override;

    bool        tokenize(const std::string& text, std::vector<int32_t>* out) override;
    std::string piece(int32_t token) override;
    bool        is_end(int32_t token) override;

    void    clear_kv(int32_t slot) override;
    int32_t kv_tokens(int32_t slot) override;
    bool    truncate_kv(int32_t slot, int32_t n) override;
    bool    needs_checkpoints() const override;
    bool    save_checkpoint(int32_t slot, std::vector<uint8_t>* out) override;
    bool    load_checkpoint(int32_t slot, const std::vector<uint8_t>& in) override;
    bool    save_state(int32_t slot, std::vector<uint8_t>* out) override;
    bool    load_state(int32_t slot, const std::vector<uint8_t>& in) override;
    void open_slot(int32_t slot, float temperature, uint32_t seed) override;
    void close_slot(int32_t slot) override;

    bool    decode(const std::vector<Entry>& batch) override;
    int32_t sample(int32_t slot, int32_t row) override;
    bool    tainted() const override;
    void    after_step(int64_t steps) override;

private:
    Engine& engine_;
    std::vector<llama_sampler*> samplers_;   // one per slot; null when closed
};

}  // namespace dray::engine
