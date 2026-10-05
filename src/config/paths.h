// Where the settings files live.
//
//   shipped: config/ next to the binary -- the defaults this build ships with
//            (system.json, models/<arch>.json), installed by the build.
//   user:    the user's own copies, overriding the shipped ones per key:
//            --config-dir DIR, else DRAY_CONFIG_DIR, else the platform's
//            per-user config directory:
//              Windows  %APPDATA%\dray
//              macOS    ~/Library/Application Support/dray
//              other    $XDG_CONFIG_HOME/dray, else ~/.config/dray
#pragma once

#include <string>

#include "config/settings.h"

namespace dray::config {

// `config_dir` is --config-dir ("" when not given).
Locations default_locations(const std::string& config_dir);

}  // namespace dray::config
