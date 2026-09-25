// dray - run frontier-scale MoE models from SSD on a polite slice of RAM.
//
// Entry point: parse the command line, dispatch to one command. Every command
// lives in src/cli/cmd_<name>.cpp.

#include <cstdio>
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
    { "calibrate", dray::cli::cmd_calibrate },
    { "verify",    dray::cli::cmd_verify },
    { "stream",    dray::cli::cmd_stream },
    { "repack",    dray::cli::cmd_repack },
    { "snaptest",  dray::cli::cmd_snaptest },
    { "batch",     dray::cli::cmd_batch },
    { "run",       dray::cli::cmd_run },
    { "serve",     dray::cli::cmd_serve },
};

}  // namespace

int main(int argc, char** argv) {
    dray::cli::Args a;
    if (!dray::cli::parse_args(argc, argv, &a)) return 1;
    if (a.help || a.cmd.empty()) { dray::cli::usage(); return a.help ? 0 : 1; }
    if (a.model.empty()) { std::fprintf(stderr, "-m <model.gguf> is required\n"); return 1; }

    for (const Command& c : kCommands) {
        if (a.cmd == c.name) return c.run(a);
    }
    std::fprintf(stderr, "unknown subcommand: %s\n", a.cmd.c_str());
    dray::cli::usage();
    return 1;
}
