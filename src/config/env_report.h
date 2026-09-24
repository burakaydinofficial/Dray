// The experimental control surface, named in one place (swarm S30).
//
// The environment variables below gate measured behaviour -- bisect switches,
// backend opt-ins, cadence knobs. A measurement whose log does not record
// which levers were pulled is not reproducible from its own output, which
// quietly undermines the readout's whole purpose. env_report() returns the
// NON-DEFAULT set as one line ("env: DRAY_NO_FUSE=1 DRAY_RING_MB=512"),
// or an empty string when everything is default -- callers print it into the
// plan report and the JSONL context so every run carries its own provenance.
//
// Adding a lever? Add it to kDrayEnvVars in env_report.cpp in the same
// commit, or the census test fails.

#pragma once

#include <string>

namespace dray::config {

// One line naming every set DRAY_*/GGML_METAL_NO_RESIDENCY variable and its
// value; empty string if none are set.
std::string env_report();

// The canonical list, exposed for the census test.
const char* const* known_env_vars(size_t* count);

}  // namespace dray::config
