// Startup calibration: what this drive actually delivers, at this engine's block
// size and queue depth (CLAUDE.md, cost model). The vendor sequential rating is
// never a denominator, and neither is a single measured number.
//
// WHY TWO CLASSES AND NOT ONE CEILING
//
// A single "fraction of ceiling" is not just imprecise, it manufactures a fake
// regression. The traffic the engine issues has two structurally different
// classes with different achievable bandwidth on the same drive:
//
//   STREAM  unconditional weights. Prefetchable with unlimited lookahead, large
//           ordered extents, the queue can always be kept full.
//   GATHER  routed experts. Short horizon (the router only just told us which
//           experts), scattered offsets, 2.5-24 MB extents, several in flight.
//
// The mix is not constant across the operating range: at a small cap traffic is
// ~90% stream, and at the knee it is 100% gather. If one ceiling is used, moving
// along that range changes the achieved fraction even when the engine is doing
// everything perfectly, and the readout reports a regression that is not there.
// So every ceiling here is per class, and fraction_of_ceiling() blends the two
// ceilings BY THE CALLER'S ACTUAL MIX before dividing.
//
// WHY THE CEILING IS THE CHOSEN POINT, NOT THE PEAK
//
// ClassResult::bw is the bandwidth at the point the engine will actually run at,
// not the best point in the sweep. Using the peak as the denominator while the
// engine deliberately runs somewhere else would report <100% forever for a reason
// that has nothing to do with the engine. The peak is reported separately
// (peak_bw / peak_block_bytes / peak_depth) so the gap stays visible.
//
// WHY THIS IS MEASURED ON DEMAND AND QUOTED WITH ITS AGE (T5)
//
// An earlier version of this header claimed calibration runs at every startup.
// It never did, and it should not: the sweep's cost is real, and demanding it
// before every multi-hour job means nobody runs it at all -- which is how
// set_ceilings sat with zero callers while the README promised limiter
// verdicts. The actual contract: `dray calibrate` measures and persists a
// summary; run/serve load it and every quoted fraction carries "calibrated
// <when>" in its provenance, because thermal and SLC state carry between runs
// and a cold-drive number quoted on a hot drive must at least say when it was
// taken. No calibration on disk means the verdict is UNKNOWN, stated, with
// the command that would fix it. The sweep's cost stays bounded and stated:
// see upper_bound_bytes().
//
// Uses only io::Backend -- no platform APIs, so calibration measures exactly the
// path the engine uses, including its submission and completion overhead.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "io/storage.h"

namespace dray::mem {
class Accountant;
}

namespace dray::calib {

enum class TrafficClass : uint8_t {
    Stream = 0,   // sequential ordered extents
    Gather = 1,   // 2.5-24 MB extents at pseudo-random aligned offsets
    _Count = 2,
};

const char* traffic_class_name(TrafficClass);

// The default sweep axes. Block sizes 64 KiB .. 2 MiB and depths 1..64 bracket
// every plausible NVMe/SATA/USB operating point; anything outside is either
// per-request overhead bound (below 64 KiB) or pure latency cost (above QD 64,
// where the staging RAM alone is 128 MiB at a 2 MiB block).
std::vector<uint32_t> default_block_sizes();
std::vector<uint32_t> default_depths();

// One measured cell of the sweep. Kept even when skipped or failed: a hole in the
// grid is information, and silently dropping it would flatter the result.
struct SweepPoint {
    uint32_t block_bytes = 0;       // effective block, after alignment rounding
    uint32_t requested_depth = 0;
    uint32_t effective_depth = 0;   // clipped by staging bytes and max_in_flight()

    uint64_t bytes = 0;             // bytes the drive actually delivered
    double   seconds = 0.0;         // includes the queue-fill ramp, which is real
    double   bw = 0.0;              // bytes / seconds, in B/s

    uint32_t requests = 0;
    uint32_t errors = 0;            // Completion::status != 0
    uint32_t short_reads = 0;       // status 0 but bytes < requested

    bool     skipped = false;
    bool     clipped = false;       // effective_depth < requested_depth
    std::string note;               // why it was skipped, when it was
};

struct ClassResult {
    TrafficClass cls = TrafficClass::Stream;
    bool ok = false;

    // The operating point the engine should use for this class. Not the peak:
    // chosen as the cheapest point within knee_tolerance of the peak, because
    // depth costs staging RAM (block * depth) and, for gather, latency on a
    // horizon that is only one router step long.
    uint32_t block_bytes = 0;
    uint32_t depth = 0;
    double   bw = 0.0;              // B/s at the chosen point -- THE ceiling

    // The peak, for the report only. Never a denominator.
    uint32_t peak_block_bytes = 0;
    uint32_t peak_depth = 0;
    double   peak_bw = 0.0;

    // Full sweep, row-major: sweep[bi * depth_axis.size() + di].
    std::vector<uint32_t>   block_axis;
    std::vector<uint32_t>   depth_axis;
    std::vector<SweepPoint> sweep;

    uint64_t bytes_read = 0;
    double   seconds = 0.0;
    uint32_t errors = 0;
    uint32_t short_reads = 0;
};

// Result of blending the two ceilings by an observed traffic mix.
struct CeilingFraction {
    bool     valid = false;         // false means unknown -- never treat as 0 or 1

    uint64_t bytes_stream = 0;
    uint64_t bytes_gather = 0;
    double   elapsed_seconds = 0.0;

    double   ideal_seconds = 0.0;   // bs/stream_bw + bg/gather_bw
    double   achieved_bw = 0.0;     // (bs + bg) / elapsed
    double   ceiling_bw = 0.0;      // (bs + bg) / ideal_seconds, i.e. mix-blended
    double   fraction = 0.0;        // achieved_bw / ceiling_bw

    // How much of ideal_seconds each class accounts for. This is the mix, and it
    // is why the blended ceiling moves with the operating range instead of
    // pretending one number covers both.
    double   stream_share = 0.0;
    double   gather_share = 0.0;
};

// True per-class fractions, for callers that can time the two classes separately.
// Prefer this when available: with a single elapsed there is no way to split the
// two, and any per-class number derived from one elapsed would be invented.
struct ClassFraction {
    bool   valid = false;
    double achieved_bw = 0.0;
    double ceiling_bw = 0.0;
    double fraction = 0.0;
};

struct CeilingSplit {
    ClassFraction stream;
    ClassFraction gather;
};

struct Options {
    std::vector<uint32_t> block_sizes = default_block_sizes();
    std::vector<uint32_t> depths      = default_depths();

    bool measure_stream = true;
    bool measure_gather = true;

    // Per-point work. A point stops at whichever of these comes first. The byte
    // target scales with block * depth because a fixed target would never fill a
    // deep queue: a 2 MiB x QD 64 queue is 128 MiB, so a 48 MiB point would never
    // reach steady state and would report a depth it never actually ran at.
    uint32_t min_batches_per_point = 3;          // full queues before stopping
    uint64_t min_bytes_per_point   = 12u << 20;  // 12 MiB
    uint64_t max_bytes_per_point   = 192u << 20; // 192 MiB
    uint32_t max_ms_per_point      = 50;         // caps a slow drive's total cost

    // With the default grid this bounds the whole run at ~2.6 GiB and ~3 s.
    // Call upper_bound_bytes() for the exact figure and print it.

    // Spins up the backend's completion threads and lifts the drive out of a
    // low-power state, so point one is not charged for both.
    uint64_t warmup_bytes = 32u << 20;

    // Transient staging. Bounded because it is charged against the RAM cap
    // (Invariant 1) at startup, before the slab exists. 128 MiB admits the whole
    // default grid; lowering it skips the deepest points rather than silently
    // running them at a smaller depth and mislabelling the result.
    uint64_t max_staging_bytes = 128u << 20;

    // Gather shape. Routed experts are 2.5-24 MB contiguous extents; 8 MiB is the
    // midpoint and is a PLACEHOLDER, not a measurement. The caller should pass the
    // plan's mean expert slot bytes and set gather_extent_from_model, so the
    // report can say which it was.
    uint32_t gather_extent_bytes = 8u << 20;
    bool     gather_extent_from_model = false;

    // Decode needs k * n_moe_layers experts per token, and the scheduler has
    // several in flight at once, so gather traffic interleaves across extents
    // rather than draining one before starting the next. This is what makes the
    // gather class structurally different from the stream class.
    uint32_t gather_concurrent_extents = 8;

    // Fixed seed: offsets are a deterministic function of (seed, class, block,
    // depth), so two runs on the same drive are comparable, while no two points
    // read the same region -- which would let the drive's own cache answer the
    // later ones.
    uint64_t seed = 0x9E3779B97F4A7C15ull;

    // A point within this much of the peak is treated as equal to it, and the
    // cheapest such point wins. 3% is below the run-to-run spread we can resolve
    // in a 50 ms sample, so calling it a tie is honest rather than generous.
    double knee_tolerance = 0.03;

    // Optional: charge the staging buffer to the cap. Strongly recommended --
    // 128 MiB is 3% of a 4 GB cap and Invariant 1 says the cap means total.
    mem::Accountant* accountant = nullptr;
};

struct Calibration {
    bool        ok = false;
    std::string error;    // non-empty when !ok; never silently substitute defaults

    // THE two ceilings. B/s at each class's chosen operating point. Zero means
    // "not measured" -- callers must treat that as unknown, not as a number.
    double stream_bw = 0.0;
    double gather_bw = 0.0;

    // The engine's operating point. Taken from the gather class: decode is the hot
    // path, it is the class with the short horizon, and its staging footprint sets
    // the RAM charge. The stream optimum is in stream.block_bytes / stream.depth
    // and prefill should use that instead.
    uint32_t block_bytes = 0;
    uint32_t queue_depth = 0;

    ClassResult stream;
    ClassResult gather;

    io::Alignment alignment;        // discovered, never assumed (Invariant 5)
    uint64_t staging_bytes = 0;
    uint64_t file_bytes = 0;        // region actually swept

    uint32_t gather_extent_bytes = 0;
    bool     gather_extent_from_model = false;
    uint32_t gather_concurrent_extents = 0;

    uint64_t total_bytes_read = 0;  // what this calibration cost the drive
    double   total_seconds = 0.0;
    uint64_t seed = 0;

    double   bandwidth(TrafficClass) const;
    uint32_t block_for(TrafficClass) const;
    uint32_t depth_for(TrafficClass) const;

    // Blend the two ceilings by this mix, then divide. A fraction above 1.0 is
    // reported as-is and never clamped: it means calibration was pessimistic
    // (colder drive, better coalescing) and hiding it would be a lie in the
    // flattering direction.
    CeilingFraction fraction_of_ceiling(uint64_t bytes_stream,
                                        uint64_t bytes_gather,
                                        double elapsed_seconds) const;

    // Genuine per-class fractions. Requires per-class elapsed time.
    CeilingSplit fraction_of_ceiling(uint64_t bytes_stream, double elapsed_stream,
                                     uint64_t bytes_gather, double elapsed_gather) const;

    std::string report() const;
};

// Bytes this configuration will read, before running it. Deterministic; print it.
// align_unit is the alignment the blocks will be rounded up to (0 or 1 = none);
// pass alignment(file).max() for an exact figure.
uint64_t upper_bound_bytes(const Options&, uint32_t align_unit = 1);

// Sweeps both classes and returns the result. Never throws; on failure returns a
// Calibration with ok=false and a populated error, because a fabricated default
// bandwidth would silently poison every ETA downstream.
// T5: persistence. Saves/loads the SUMMARY (ceilings, operating points, cost,
// timestamp) -- never the sweeps. A loaded calibration is quoted WITH its age
// in every readout line (Ceilings.provenance), because thermal and SLC state
// carry between runs: a labeled stale ceiling is honest, an unlabeled one is
// not, and "unknown forever unless you re-sweep before every multi-hour job"
// serves nobody.
bool save_calibration(const Calibration& c, const std::string& path, std::string* err);
bool load_calibration(Calibration* c, std::string* when, const std::string& path,
                      std::string* err);

Calibration calibrate(io::Backend& backend, io::FileId file, uint64_t file_size,
                      const Options& options = Options{});

}  // namespace dray::calib
