#include "cli/commands.h"

#include <cstdio>

#include "tools/repack.h"

namespace dray::cli {

int cmd_repack(const Args& a) {
    if (a.out_dir.empty()) { std::fprintf(stderr, "repack: --out DIR required\n"); return 1; }
    return dray::tools::repack_write(a.model, a.out_dir);
}

}  // namespace dray::cli
