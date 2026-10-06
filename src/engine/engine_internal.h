// Private to src/engine/*.cpp. Load-time parameters llama is handed POINTERS to:
// llama keeps a copy of llama_model_params, so the arrays behind those pointers
// must live as long as the model does.
#pragma once

#include <vector>

#include "llama.h"
#include "engine/kv_overrides.h"

namespace dray::engine {

struct LoadState {
    llama_model_tensor_buft_override buft_overrides[2] = {};
    KvOverrides kv_overrides;
    std::vector<ggml_backend_dev_t> offload_devices;   // what llama may offload to; empty without --gpu
};

}  // namespace dray::engine
