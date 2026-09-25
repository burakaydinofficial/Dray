// Cohort rotation: more prompts than the admission funds, served in cohorts of
// the funded width. A cohort decodes for `span` tokens, parks its per-sequence
// KV/recurrent state to state_dir, and the next cohort loads. Opt-in only: the
// engine's one sustained-write feature (writes are reported like reads).
//
// Mid-generation parking (span < max_tokens) is refused unless
// DRAY_ROTATE_UNSAFE=1: the fork's llama_state_seq restore is measurably
// nondeterministic (DECISIONS), so cohorts run to completion and then rotate.
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
