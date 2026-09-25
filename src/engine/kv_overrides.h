// GGUF metadata overrides: CLI --override-kv strings first, then the model
// profile's kv_defaults for keys the CLI did not name.
//
// Released conversions sometimes LACK keys the loader requires (MiniMax M3's
// five indexer hyperparameters were the first case). Dedupe is ours so
// precedence never depends on llama's override-application order. A parse
// failure is a load failure: a silently dropped override would load a model
// with wrong hyperparameters and generate fluent nonsense.
#pragma once

#include <string>
#include <vector>

#include "llama.h"

namespace dray::models { struct Profile; }

namespace dray::engine {

struct KvOverrides {
    // Parsed entries, CLI first. Empty when nothing was requested; otherwise
    // pass terminated() to llama_model_params::kv_overrides.
    std::vector<llama_model_kv_override> entries;

    // Null-key terminated copy for llama. Valid while this object lives.
    const llama_model_kv_override* terminated();

private:
    std::vector<llama_model_kv_override> terminated_;
};

// Parses "key=int|float|bool|str:value" strings. The earliest entry for a key
// wins. On failure returns false and names the bad string in *error, in the
// exact wording the engine has always refused with.
bool build_kv_overrides(const std::vector<std::string>& cli,
                        const models::Profile* profile,
                        KvOverrides* out, std::string* error);

}  // namespace dray::engine
