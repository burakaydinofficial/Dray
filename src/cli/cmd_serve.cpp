#include <cstdio>
#include <string>

#include "cli/commands.h"
#include "cli/engine_config.h"
#include "config/model_settings.h"
#include "engine/engine_types.h"
#include "server/serve.h"

namespace dray::cli {

int cmd_serve(const Args& a) {
    // serve.max_parallel is needed BEFORE the engine opens: the planner funds
    // exactly that many sequences (each a slot with its own KV), inside --cap.
    config::CliSettings cli;
    cli.threads = a.n_threads;
    cli.prefill_chunk = a.prefill_chunk;
    cli.parallel = a.parallel;
    cli.vram_cap = a.vram_cap;
    config::Settings s;
    std::string err;
    if (!config::resolve_for_model(a.model, a.config_dir, cli, &s, nullptr, &err)) {
        std::fprintf(stderr, "serve: settings: %s\n", err.c_str());
        return 1;
    }
    dray::engine::EngineConfig ec = engine_config_from(a);
    ec.reserve_checkpoint = !a.jobs_dir.empty();   // F4
    ec.n_seq = static_cast<uint32_t>(s.serve_max_parallel.value);
    dray::server::ServeOptions opt;
    opt.port = a.port;
    opt.jobs_dir = a.jobs_dir;
    opt.api_key = a.api_key;
    opt.max_parallel = s.serve_max_parallel.value;
    opt.prefill_tokens_per_step = s.serve_prefill_tokens_per_step.value;
    opt.checkpoints_per_slot = s.serve_checkpoints_per_slot.value;
    opt.checkpoint_offset = s.serve_checkpoint_offset.value;
    opt.kv_idle_timeout_s = s.serve_kv_idle_timeout_s.value;
    opt.kv_pool_tokens = s.serve_kv_pool_tokens.value;
    return dray::server::serve_main(ec, opt);
}

}  // namespace dray::cli
