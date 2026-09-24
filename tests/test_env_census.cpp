// S30's enforcement: every getenv("DRAY_*") in the source tree must appear
// in the canonical list env_report() prints from. A lever the report does not
// know about is a measurement that cannot be reproduced from its own log.
// The census here is compiled-in (tests cannot grep), so adding a lever means
// touching BOTH the code and kDrayEnvVars -- this test makes forgetting
// loud in the one direction it can check: every canonical name must be
// non-empty, unique, and DRAY_-or-GGML_-prefixed, and env_report must
// format exactly the set that is set.

#include "harness.h"

#include <cstdlib>
#include <set>
#include <string>

#include "config/env_report.h"

using dray::config::env_report;
using dray::config::known_env_vars;

LZ_TEST(census_is_wellformed_and_unique) {
    size_t n = 0;
    const char* const* v = known_env_vars(&n);
    LZ_CHECK_GE(n, 17u);
    std::set<std::string> seen;
    for (size_t i = 0; i < n; ++i) {
        std::string s = v[i];
        LZ_CHECK(!s.empty());
        LZ_CHECK(s.rfind("DRAY_", 0) == 0 || s.rfind("GGML_", 0) == 0);
        LZ_CHECK(seen.insert(s).second);   // unique
    }
}

LZ_TEST(report_names_exactly_the_set_levers) {
#if defined(_WIN32)
    _putenv_s("DRAY_NO_FUSE", "1");
    _putenv_s("DRAY_RING_MB", "512");
#else
    setenv("DRAY_NO_FUSE", "1", 1);
    setenv("DRAY_RING_MB", "512", 1);
#endif
    const std::string r = env_report();
    LZ_CHECK(r.find("DRAY_NO_FUSE=1") != std::string::npos);
    LZ_CHECK(r.find("DRAY_RING_MB=512") != std::string::npos);
    LZ_CHECK(r.rfind("env: ", 0) == 0);
#if defined(_WIN32)
    _putenv_s("DRAY_NO_FUSE", "");
    _putenv_s("DRAY_RING_MB", "");
#else
    unsetenv("DRAY_NO_FUSE");
    unsetenv("DRAY_RING_MB");
#endif
    LZ_CHECK(env_report().find("NO_FUSE") == std::string::npos);
}
