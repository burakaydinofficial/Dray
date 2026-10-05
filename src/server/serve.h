// The OpenAI-compatible server (see serve.cpp for its parts).
#pragma once

#include <string>

#include "engine/engine_types.h"

namespace dray::server {

struct ServeOptions {
    int         port = 8080;
    std::string jobs_dir;                 // background jobs with crash-resume; empty = off
    std::string api_key;                  // required as a Bearer token on /v1/* when set
    int         max_parallel = 1;         // settings: serve.max_parallel (= the funded n_seq)
    int         prefill_tokens_per_step = 256;   // settings: serve.prefill_tokens_per_step
    int         checkpoints_per_slot = 2;        // settings: serve.checkpoints_per_slot
    int         checkpoint_offset = 4;           // settings: serve.checkpoint_offset
    int         kv_idle_timeout_s = 600;         // settings: serve.kv_idle_timeout_s
    int         kv_pool_tokens = -1;             // settings: serve.kv_pool_tokens (-1 = ctx x slots)
};

int serve_main(const engine::EngineConfig& cfg, const ServeOptions& opt);

}  // namespace dray::server
