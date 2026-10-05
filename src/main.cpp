// dray - run frontier-scale MoE models from SSD on a polite slice of RAM.
//
// Entry point: parse the command line, dispatch to one command. Every command
// lives in src/cli/cmd_<name>.cpp.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "cli/args.h"
#include "cli/commands.h"

namespace {

struct Command {
    const char* name;
    int (*run)(const dray::cli::Args&);
};

constexpr Command kCommands[] = {
    { "plan",      dray::cli::cmd_plan },
    { "config",    dray::cli::cmd_config },
    { "calibrate", dray::cli::cmd_calibrate },
    { "verify",    dray::cli::cmd_verify },
    { "repack",    dray::cli::cmd_repack },
    { "snaptest",  dray::cli::cmd_snaptest },
    { "batch",     dray::cli::cmd_batch },
    { "run",       dray::cli::cmd_run },
    { "serve",     dray::cli::cmd_serve },
};

}  // namespace

// ggml initialises every compiled-in GPU backend the first time its backend
// registry is touched, used or not -- and a merely initialised Vulkan cost ~25%
// of decode on the reference laptop (Flash Next: 1.75-1.96 s/token CPU-only
// build, 2.23-2.38 with Vulkan compiled in and unused; same bytes, same buffers,
// the extra is CPU compute time; with GGML_DISABLE_VULKAN 1.90-1.92). Without
// GPU consent (--gpu, or DRAY_VULKAN=1) the process switches Vulkan off for
// itself, before any ggml call. A GGML_DISABLE_VULKAN the user set is left alone.
static void keep_unused_gpu_backends_asleep(const dray::cli::Args& a) {
    const char* consent = std::getenv("DRAY_VULKAN");
    if (a.gpu || (consent && std::string(consent) == "1")) return;
    if (std::getenv("GGML_DISABLE_VULKAN")) return;
#if defined(_WIN32)
    _putenv_s("GGML_DISABLE_VULKAN", "1");
#else
    setenv("GGML_DISABLE_VULKAN", "1", 0);
#endif
}

int main(int argc, char** argv) {
    dray::cli::Args a;
    if (!dray::cli::parse_args(argc, argv, &a)) return 1;
    if (a.help || a.cmd.empty()) { dray::cli::usage(); return a.help ? 0 : 1; }
    if (a.model.empty()) { std::fprintf(stderr, "-m <model.gguf> is required\n"); return 1; }
    keep_unused_gpu_backends_asleep(a);

    for (const Command& c : kCommands) {
        if (a.cmd == c.name) return c.run(a);
    }
    std::fprintf(stderr, "unknown subcommand: %s\n", a.cmd.c_str());
    dray::cli::usage();
    return 1;
}
