// The OpenAI-compatible server (see serve.cpp for the protocol surface).
#pragma once

#include <string>

#include "engine/engine_types.h"

namespace dray::server {

// Blocking server loop. Returns the process exit code.
int serve_main(const engine::EngineConfig& cfg, int port, const std::string& jobs_dir,
               const std::string& api_key = std::string());

}  // namespace dray::server
