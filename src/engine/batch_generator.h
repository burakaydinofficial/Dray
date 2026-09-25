// Batched lockstep decode: N prompts through one streamer, one llama_decode per
// step carrying every live sequence. The unconditional stream is read once per
// step for all of them and routed experts once per step for the UNION of their
// selections -- the whole economic case for batching on this engine.
//
// Correctness before speed: B copies of one greedy prompt MUST produce B
// byte-identical outputs matching the single-stream control (scripts/batchdiff).
#pragma once

#include "engine/engine_types.h"

namespace dray::engine {

class Engine;

class BatchGenerator {
public:
    explicit BatchGenerator(Engine& engine) : engine_(engine) {}
    BatchResult run(const BatchParams& params);

private:
    Engine& engine_;
};

}  // namespace dray::engine
