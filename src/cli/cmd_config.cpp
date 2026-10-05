// `dray config -m <model.gguf>`: every effective setting for this model and
// the layer it came from (built-in, shipped/user system, shipped/user model,
// command line). Reads only the GGUF header, for general.architecture.

#include <cstdio>
#include <iostream>
#include <string>

#include "cli/args.h"
#include "cli/commands.h"
#include "config/model_settings.h"
#include "config/paths.h"

namespace dray::cli {

int cmd_config(const Args& a) {
    config::CliSettings cli;
    cli.threads = a.n_threads;
    cli.prefill_chunk = a.prefill_chunk;
    cli.parallel = a.parallel;
    cli.vram_cap = a.vram_cap;
    config::Settings s;
    std::string arch, err;
    if (!config::resolve_for_model(a.model, a.config_dir, cli, &s, &arch, &err)) {
        std::fprintf(stderr, "settings: %s\n", err.c_str());
        return 1;
    }
    const config::Locations where = config::default_locations(a.config_dir);
    std::cout << "model architecture: " << arch << "\n"
              << "shipped config dir: " << (where.shipped.empty() ? "(unknown)" : where.shipped.string()) << "\n"
              << "user config dir:    " << (where.user.empty() ? "(none)" : where.user.string()) << "\n\n";
    config::describe(s, std::cout);
    return 0;
}

}  // namespace dray::cli
