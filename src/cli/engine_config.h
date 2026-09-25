// The ONE mapping from command-line options to an engine configuration.
//
// It used to be four hand-copied blocks (run, batch, snaptest, serve), and a
// field added to three of them and not the fourth is a flag that is parsed and
// then silently ignored: serve dropped --repack once (S29), and snaptest still
// did until this existed. Commands set only what is genuinely theirs on top.

#pragma once

#include "cli/args.h"
#include "engine/engine_types.h"

namespace dray::cli {

engine::EngineConfig engine_config_from(const Args& a);

}  // namespace dray::cli
