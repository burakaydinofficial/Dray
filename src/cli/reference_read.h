// Independent uncached reference reads for the diagnostics (verify, stream).
#pragma once

#include <cstdint>
#include <string>

namespace dray::cli {

bool read_reference_uncached(const std::string& path, uint64_t offset,
                             uint64_t len, uint8_t* dst);

}  // namespace dray::cli
