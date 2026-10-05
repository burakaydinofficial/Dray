#include "cli/engine_config.h"

namespace dray::cli {

engine::EngineConfig engine_config_from(const Args& a) {
    engine::EngineConfig ec;
    ec.model_path    = a.model;
    ec.cap           = a.cap;
    ec.n_ctx         = a.n_ctx;
    ec.repack_dir    = a.repack_dir;
    ec.kv_overrides  = a.kv_overrides;
    ec.gpu           = a.gpu;
    ec.kv_quant      = a.kv;
    ec.prefill_chunk = a.prefill_chunk;
    ec.n_threads     = a.n_threads;
    ec.force_stream  = a.force_stream;
    ec.resident      = a.resident;
    ec.config_dir    = a.config_dir;
    ec.vram_cap      = a.vram_cap;
    return ec;
}

}  // namespace dray::cli
