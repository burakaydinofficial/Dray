#include "cli/commands.h"
#include "cli/engine_config.h"

#include "engine/engine_types.h"
#include "server/serve.h"

namespace dray::cli {

int cmd_serve(const Args& a) {
    dray::engine::EngineConfig ec = engine_config_from(a);
    ec.reserve_checkpoint = !a.jobs_dir.empty();   // F4
    return dray::server::serve_main(ec, a.port, a.jobs_dir, a.api_key);
}

}  // namespace dray::cli
