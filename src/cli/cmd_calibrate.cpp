#include "cli/commands.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "calib/calibrate.h"
#include "io/storage.h"

namespace dray::cli {

int cmd_calibrate(const Args& a) {
    auto backend = dray::io::make_backend(64);
    if (!backend) { std::fprintf(stderr, "no storage backend\n"); return 1; }
    dray::io::FileId f = backend->open(a.model);
    if (f == dray::io::kInvalidFile) {
        std::fprintf(stderr, "cannot open %s in uncached mode\n", a.model.c_str());
        return 1;
    }
    std::printf("backend: %s\n", backend->describe().c_str());
    dray::calib::Options opt;
    dray::calib::Calibration c = dray::calib::calibrate(*backend, f, backend->size(f), opt);
    std::printf("%s\n", c.report().c_str());
    // T5: persist the summary so run/serve can quote real ceilings (with age).
    const std::string calp = a.model + ".lzcal.json";
    std::string serr;
    if (dray::calib::save_calibration(c, calp, &serr)) {
        std::printf("calibration saved: %s\n", calp.c_str());
    } else {
        std::fprintf(stderr, "calibration NOT saved: %s\n", serr.c_str());
    }
    backend->close(f);
    return 0;
}


}  // namespace dray::cli
