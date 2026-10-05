// Settings for a model file: reads general.architecture from the GGUF header
// (nothing else) and resolves every layer (config/settings.h). The one entry
// point for callers that have a model path -- `dray config`, serve (which
// needs serve.max_parallel before the engine opens: the planner funds that
// many sequences) and Engine::open -- so they can never disagree.
#pragma once

#include <string>

#include "config/settings.h"

namespace dray::config {

// `config_dir` is --config-dir ("" = DRAY_CONFIG_DIR / the platform default).
// On success also returns the architecture in *arch (may be null).
bool resolve_for_model(const std::string& model_path, const std::string& config_dir,
                       const CliSettings& cli, Settings* out, std::string* arch,
                       std::string* err);

}  // namespace dray::config
