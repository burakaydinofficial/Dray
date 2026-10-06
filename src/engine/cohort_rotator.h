// Cohort rotation: more prompts than the admission funds, served in cohorts of
// the funded width. A cohort decodes for `span` tokens, parks its per-sequence
// KV/recurrent state to state_dir, and the next cohort loads. Opt-in only: the
// engine's one sustained-write feature (writes are reported like reads).
//
// span >= max_tokens: each cohort runs to completion, nothing is parked.
// span < max_tokens: sequences are parked mid-generation and resumed on the
// cohort's next turn, token-for-token identical to running to completion
// (scripts/rotategate.ps1).
#pragma once

#include "engine/engine_types.h"

namespace dray::engine {

class Engine;

class CohortRotator {
public:
    explicit CohortRotator(Engine& engine) : engine_(engine) {}
    RotateResult run(const RotateParams& params);

private:
    Engine& engine_;
};

}  // namespace dray::engine
