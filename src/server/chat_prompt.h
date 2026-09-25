// messages[] -> one prompt string. The model's own chat template when it has
// one; otherwise an explicit, boring fallback -- and the caller is told which
// was used, because silently improvising a template changes model behaviour.
#pragma once

#include <string>

#include "json.hpp"

struct llama_model;

namespace dray::server {

std::string build_prompt(const llama_model* model, const nlohmann::json& messages,
                         bool* used_template);

}  // namespace dray::server
