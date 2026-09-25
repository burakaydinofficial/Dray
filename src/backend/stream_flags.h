// The streamer's environment levers, read ONCE when a Streamer is built.
//
// Every one of these is a bisect or diagnostic switch -- the defaults are the
// engine. They used to be function-local statics scattered through
// stream_buffer.cpp, three of them read in two places each, which is how a
// config-forced mode and an env-forced mode drifted apart (swarm R4). One
// struct, filled once, read everywhere.
//
// Every name here must also be listed in config/env_report.cpp: the census test
// enforces it, and a lever the startup report cannot name is a measurement that
// cannot be reproduced from its own log.

#pragma once

#include <cstdint>
#include <optional>

namespace dray::backend {

struct StreamFlags {
    // Mechanism bisects: each turns one mechanism off so a fault can be pinned.
    bool no_compact  = false;   // DRAY_NO_COMPACT   whole expert tensors, no compaction
    bool no_rowslice = false;   // DRAY_NO_ROWSLICE  whole embedding table, no row slicing
    bool no_fuse     = false;   // DRAY_NO_FUSE      no sibling (gate/up/down) region reads
    bool no_early    = false;   // DRAY_NO_EARLY     no early unlock at the router's release
    bool no_reuse    = false;   // DRAY_NO_REUSE     never reuse a region for equal routing
    bool no_retain   = false;   // DRAY_NO_RETAIN    ring never promotes into the cache
    bool no_eslots   = false;   // DRAY_NO_ESLOTS    no frequency expert cache
    bool no_pool     = false;   // DRAY_NO_POOL      plain host allocations, no VA arena
    bool compact_all = false;   // DRAY_COMPACT_ALL  identity compaction (every expert)

    // Opt-ins.
    bool fast_nodes  = false;   // DRAY_FAST_NODES   skip nodes with no disk-backed source
    bool repack      = false;   // DRAY_REPACK       repack whole 2D weights (measured -20%)
    bool metal       = false;   // DRAY_METAL        Apple: map the pool arena for Metal

    // Diagnostics.
    bool trace         = false; // DRAY_TRACE          every materialised node, unbuffered
    bool trace_compact = false; // DRAY_TRACE_COMPACT  compaction and private-ids checks
    bool io_stats      = false; // DRAY_IO_STATS       I/O forensics in the report

    // DRAY_RING_MB: ring size override. Set-but-empty is a value (0 MiB, i.e.
    // no ring), not "unset" -- that is how the variable has always read.
    std::optional<uint64_t> ring_mb;

    static StreamFlags from_env();
};

}  // namespace dray::backend
