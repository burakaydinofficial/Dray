#include "calib/calibrate.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <sstream>

#include "mem/accountant.h"

namespace dray::calib {

namespace {

using Clock = std::chrono::steady_clock;

// Only a hung or broken backend reaches this. There is no cancel() in the
// io::Backend interface (see storage.h), so a stall is not recoverable by
// abandoning the reads -- it is only survivable by never freeing the buffers the
// kernel may still write into. calibrate() leaks the staging buffer in that case.
constexpr double kHardDeadlineSeconds = 30.0;

constexpr size_t kSubmitBatch  = 16;
constexpr size_t kHarvestBatch = 64;

inline double secs_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

inline uint64_t round_up_u64(uint64_t v, uint64_t unit) {
    if (unit <= 1) return v;
    return ((v + unit - 1) / unit) * unit;
}

// splitmix64. Written out rather than pulled from <random> because
// std::uniform_int_distribution is not specified to produce the same sequence
// across implementations, and the whole point of the fixed seed is that two runs
// -- and two machines comparing notes -- read the same offsets.
inline uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Distinct per point, deterministic across runs. Distinctness matters: if every
// point read the same region, the drive's own DRAM/SLC cache would answer the
// later points and the sweep would slope upward for a reason that has nothing to
// do with block size or depth.
inline uint64_t point_seed(uint64_t base, uint32_t cls, uint32_t block, uint32_t depth) {
    uint64_t s = base;
    s ^= static_cast<uint64_t>(cls)   * 0xD6E8FEB86659FD93ull;
    s ^= static_cast<uint64_t>(block) * 0xA0761D6478BD642Full;
    s ^= static_cast<uint64_t>(depth) * 0xE7037ED1A0B428DBull;
    return splitmix64(s);
}

// Produces the block indices a class reads. Indices, not offsets: the caller
// multiplies by the effective block size, which is itself a multiple of the
// runtime alignment, so every offset and length satisfies io::Alignment without
// anything here hardcoding 4096 (Invariant 5).
class ExtentGen {
public:
    ExtentGen(bool gather, uint64_t seed, uint64_t n_blocks,
              uint64_t blocks_per_extent, uint32_t n_open)
        : gather_(gather), state_(seed), n_blocks_(n_blocks ? n_blocks : 1),
          bpe_(blocks_per_extent ? blocks_per_extent : 1) {
        if (bpe_ > n_blocks_) bpe_ = n_blocks_;
        n_ext_ = n_blocks_ / bpe_;
        if (n_ext_ == 0) n_ext_ = 1;

        if (gather_) {
            uint64_t open = n_open ? n_open : 1;
            if (open > n_ext_) open = n_ext_;
            open_.resize(static_cast<size_t>(open));
            for (Ext& e : open_) { e.start = draw_start(); e.used = 0; }
        } else {
            cursor_ = draw() % n_blocks_;
        }
    }

    uint64_t next() {
        if (!pending_.empty()) {
            const uint64_t v = pending_.back();
            pending_.pop_back();
            return v;
        }
        if (!gather_) {
            const uint64_t v = cursor_;
            cursor_ = (cursor_ + 1) % n_blocks_;
            return v;
        }
        Ext& e = open_[rr_];
        rr_ = (rr_ + 1) % open_.size();
        if (e.used >= bpe_) { e.start = draw_start(); e.used = 0; }
        const uint64_t v = e.start + e.used;
        ++e.used;
        return v < n_blocks_ ? v : (v % n_blocks_);
    }

    // Returns an index the backend refused, so the next call re-issues it. Without
    // this a partially accepted submit would punch holes in the stream class and
    // it would stop being a sequential scan.
    void unget(uint64_t v) { pending_.push_back(v); }

private:
    struct Ext { uint64_t start = 0; uint64_t used = 0; };

    uint64_t draw() { return splitmix64(state_); }
    uint64_t draw_start() { return (draw() % n_ext_) * bpe_; }

    bool     gather_;
    uint64_t state_;
    uint64_t n_blocks_;
    uint64_t bpe_;
    uint64_t n_ext_ = 1;
    uint64_t cursor_ = 0;
    size_t   rr_ = 0;
    std::vector<Ext>      open_;
    std::vector<uint64_t> pending_;
};

// Staging is charged to the cap like every other resident byte (Invariant 1).
// 128 MiB is 3% of a 4 GB cap, and it is live at startup before the slab exists,
// so leaving it unaccounted would be exactly the kind of "on top of the cap" term
// the invariant forbids.
class Staging {
public:
    Staging(mem::Accountant* acct, size_t bytes, uint32_t align)
        : acct_(acct), bytes_(bytes) {
        if (bytes_ == 0) return;
        if (acct_ && !acct_->reserve(mem::Category::IoStaging, bytes_)) {
            bytes_ = 0;
            return;
        }
        base_ = static_cast<uint8_t*>(mem::aligned_alloc_host(bytes_, align ? align : 1));
        if (!base_) {
            if (acct_) acct_->release(mem::Category::IoStaging, bytes_);
            bytes_ = 0;
        }
    }

    ~Staging() {
        if (!armed_) return;  // deliberately leaked; see disarm()
        if (base_) mem::aligned_free_host(base_, bytes_);
        if (acct_ && bytes_) acct_->release(mem::Category::IoStaging, bytes_);
    }

    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;

    // Abandons the buffer instead of freeing it. Called only when reads are still
    // outstanding and cannot be cancelled: freeing memory the kernel still owns a
    // DMA target in is a use-after-free that would corrupt whatever gets the
    // address next. The accounting is abandoned with it, which is correct -- those
    // bytes really are still held.
    void disarm() { armed_ = false; }

    bool     ok() const { return base_ != nullptr; }
    uint8_t* base() const { return base_; }
    size_t   bytes() const { return bytes_; }

private:
    mem::Accountant* acct_ = nullptr;
    uint8_t*         base_ = nullptr;
    size_t           bytes_ = 0;
    bool             armed_ = true;
};

struct PointCtx {
    io::Backend*   be = nullptr;
    io::FileId     file = io::kInvalidFile;
    uint8_t*       staging = nullptr;
    uint64_t       staging_bytes = 0;
    uint64_t       usable = 0;
    const Options* opt = nullptr;
};

// Bytes one sweep cell reads. Scales with block * depth because a fixed target
// would never fill a deep queue: 2 MiB x QD 64 is 128 MiB in flight, so a 48 MiB
// cell would never reach steady state yet would still be labelled QD 64.
uint64_t point_target_bytes(const Options& o, uint32_t block, uint64_t depth) {
    uint64_t t = static_cast<uint64_t>(o.min_batches_per_point) * block * depth;
    if (t < o.min_bytes_per_point) t = o.min_bytes_per_point;
    if (t > o.max_bytes_per_point) t = o.max_bytes_per_point;
    if (t < block) t = block;
    return t;
}

SweepPoint run_point(const PointCtx& ctx, TrafficClass cls, uint32_t block,
                     uint32_t requested_depth, bool* stalled) {
    const Options& o = *ctx.opt;

    SweepPoint p;
    p.block_bytes = block;
    p.requested_depth = requested_depth;

    uint64_t depth = requested_depth;
    const uint64_t slots_that_fit = ctx.staging_bytes / block;
    if (depth > slots_that_fit) depth = slots_that_fit;
    const uint64_t backend_max = ctx.be->max_in_flight() ? ctx.be->max_in_flight() : 1;
    if (depth > backend_max) depth = backend_max;
    if (depth == 0) {
        p.skipped = true;
        p.note = "staging buffer holds less than one block";
        return p;
    }
    p.effective_depth = static_cast<uint32_t>(depth);
    p.clipped = p.effective_depth < requested_depth;

    const uint64_t n_blocks = ctx.usable / block;
    if (n_blocks == 0) {
        p.skipped = true;
        p.note = "file shorter than one block";
        return p;
    }

    // Gather reads whole expert-sized extents placed at pseudo-random aligned
    // offsets, and interleaves across several of them at once, because decode
    // needs k * n_moe_layers experts per token and the scheduler has more than one
    // in flight. Draining one extent before starting the next would measure
    // something much closer to the stream class than what decode actually does.
    uint64_t blocks_per_extent = n_blocks;
    uint32_t open_extents = 1;
    if (cls == TrafficClass::Gather) {
        uint64_t extent = o.gather_extent_bytes;
        if (extent < block) extent = block;
        blocks_per_extent = extent / block;
        if (blocks_per_extent == 0) blocks_per_extent = 1;
        open_extents = o.gather_concurrent_extents ? o.gather_concurrent_extents : 1;
    }

    ExtentGen gen(cls == TrafficClass::Gather,
                  point_seed(o.seed, static_cast<uint32_t>(cls), block, requested_depth),
                  n_blocks, blocks_per_extent, open_extents);

    const uint64_t target = point_target_bytes(o, block, depth);
    const double   max_s  = static_cast<double>(o.max_ms_per_point) / 1000.0;

    std::vector<uint32_t> free_slots;
    free_slots.reserve(static_cast<size_t>(depth));
    for (uint32_t i = static_cast<uint32_t>(depth); i-- > 0;) free_slots.push_back(i);

    io::ReadRequest batch[kSubmitBatch];
    io::Completion  comps[kHarvestBatch];

    size_t   outstanding = 0;
    uint64_t issued_bytes = 0;
    uint64_t good_bytes = 0;
    uint32_t requests = 0;
    uint32_t errors = 0;
    uint32_t shorts = 0;
    bool     stop_issue = false;
    bool     stuck = false;

    // The clock starts at the first submission, so the queue-fill ramp is inside
    // the measurement. That is deliberate: the engine pays that ramp too, and
    // excluding it would report a bandwidth the engine can never realize.
    const Clock::time_point t0 = Clock::now();
    Clock::time_point t1 = t0;

    for (;;) {
        t1 = Clock::now();
        const double elapsed = secs_between(t0, t1);
        if (!stop_issue && (issued_bytes >= target || elapsed >= max_s ||
                            elapsed >= kHardDeadlineSeconds)) {
            stop_issue = true;
        }
        if (stop_issue && outstanding == 0) break;

        if (!stop_issue) {
            size_t nb = 0;
            while (nb < kSubmitBatch && !free_slots.empty() &&
                   (outstanding + nb) < static_cast<size_t>(depth) &&
                   issued_bytes + static_cast<uint64_t>(nb) * block < target) {
                const uint32_t slot = free_slots.back();
                free_slots.pop_back();
                io::ReadRequest& r = batch[nb];
                r.file   = ctx.file;
                r.offset = gen.next() * block;
                r.length = block;
                r.dst    = ctx.staging + static_cast<size_t>(slot) * static_cast<size_t>(block);
                r.tag    = slot;
                ++nb;
            }

            size_t accepted = 0;
            if (nb > 0) {
                accepted = ctx.be->submit(batch, nb);
                if (accepted > nb) accepted = nb;  // a backend must not over-report
                for (size_t i = nb; i-- > accepted;) {
                    free_slots.push_back(static_cast<uint32_t>(batch[i].tag));
                    gen.unget(batch[i].offset / block);
                }
                outstanding  += accepted;
                issued_bytes += static_cast<uint64_t>(accepted) * block;
                requests     += static_cast<uint32_t>(accepted);
            }

            if (outstanding == 0) {
                // Nothing in flight and the backend took nothing: no progress is
                // possible, so stop rather than spin.
                if (p.note.empty()) p.note = "backend refused submission with an empty queue";
                break;
            }
        }

        if (outstanding == 0) continue;

        size_t got = ctx.be->poll(comps, kHarvestBatch, 1);
        if (got == 0) {
            if (secs_between(t0, Clock::now()) >= kHardDeadlineSeconds) {
                stuck = true;
                p.note = "backend stopped completing";
                break;
            }
            continue;
        }
        if (got > kHarvestBatch) got = kHarvestBatch;

        for (size_t i = 0; i < got; ++i) {
            const io::Completion& c = comps[i];
            if (c.tag < depth) free_slots.push_back(static_cast<uint32_t>(c.tag));
            if (outstanding > 0) --outstanding;

            if (c.status != 0) {
                // Bytes moved under a failed read are not counted. storage.h warns
                // that short reads and EIO are expected at this volume; counting
                // their partial bytes would inflate the ceiling, and the ceiling is
                // a denominator.
                ++errors;
            } else {
                good_bytes += c.bytes;
                if (c.bytes < block) ++shorts;
            }
        }
    }

    t1 = Clock::now();

    if (stuck && stalled) *stalled = true;

    p.seconds     = secs_between(t0, t1);
    p.bytes       = good_bytes;
    p.requests    = requests;
    p.errors      = errors;
    p.short_reads = shorts;
    p.bw          = p.seconds > 0.0 ? static_cast<double>(good_bytes) / p.seconds : 0.0;
    return p;
}

// Picks the operating point: the cheapest point within knee_tolerance of the peak,
// not the peak itself. Depth costs staging RAM (block * depth, charged to the cap)
// and, for gather, latency on a horizon one router step long; buying 2% more
// bandwidth with 4x the queue is a bad trade at a 4 GB cap.
void choose_operating_point(ClassResult& r, double tolerance) {
    const SweepPoint* peak = nullptr;
    for (const SweepPoint& p : r.sweep) {
        if (p.skipped || p.bw <= 0.0) continue;
        if (!peak || p.bw > peak->bw) peak = &p;
    }
    if (!peak) {
        r.ok = false;
        return;
    }
    r.peak_bw          = peak->bw;
    r.peak_block_bytes = peak->block_bytes;
    r.peak_depth       = peak->effective_depth;

    const double threshold = peak->bw * (1.0 - tolerance);
    const SweepPoint* best = peak;
    uint64_t best_cost = static_cast<uint64_t>(peak->block_bytes) * peak->effective_depth;

    for (const SweepPoint& p : r.sweep) {
        if (p.skipped || p.bw < threshold) continue;
        const uint64_t cost = static_cast<uint64_t>(p.block_bytes) * p.effective_depth;
        const bool better =
            cost < best_cost ||
            (cost == best_cost && p.effective_depth < best->effective_depth) ||
            (cost == best_cost && p.effective_depth == best->effective_depth && p.bw > best->bw);
        if (better) {
            best = &p;
            best_cost = cost;
        }
    }

    r.block_bytes = best->block_bytes;
    r.depth       = best->effective_depth;
    r.bw          = best->bw;
    r.ok          = r.bw > 0.0;
}

ClassResult run_class(const PointCtx& ctx, TrafficClass cls,
                      const std::vector<uint32_t>& blocks,
                      const std::vector<uint32_t>& depths, bool* stalled) {
    ClassResult r;
    r.cls = cls;
    r.block_axis = blocks;
    r.depth_axis = depths;
    r.sweep.reserve(blocks.size() * depths.size());

    for (uint32_t b : blocks) {
        for (uint32_t d : depths) {
            if (*stalled) {
                SweepPoint p;
                p.block_bytes = b;
                p.requested_depth = d;
                p.skipped = true;
                p.note = "abandoned after a backend stall";
                r.sweep.push_back(p);
                continue;
            }
            SweepPoint p = run_point(ctx, cls, b, d, stalled);
            r.bytes_read  += p.bytes;
            r.seconds     += p.seconds;
            r.errors      += p.errors;
            r.short_reads += p.short_reads;
            r.sweep.push_back(p);
        }
    }

    choose_operating_point(r, ctx.opt->knee_tolerance);
    return r;
}

std::string fmt_f(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return std::string(buf);
}

std::string fmt_size(uint64_t bytes) {
    char buf[64];
    if (bytes >= (1ull << 30)) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB",
                      static_cast<double>(bytes) / static_cast<double>(1ull << 30));
    } else if (bytes >= (1ull << 20)) {
        std::snprintf(buf, sizeof(buf), "%.2f MiB",
                      static_cast<double>(bytes) / static_cast<double>(1ull << 20));
    } else if (bytes >= (1ull << 10)) {
        std::snprintf(buf, sizeof(buf), "%.2f KiB",
                      static_cast<double>(bytes) / static_cast<double>(1ull << 10));
    } else {
        std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
    }
    return std::string(buf);
}

// Short label for an axis value: "64 KiB", "2 MiB".
std::string fmt_block(uint32_t bytes) {
    char buf[32];
    if (bytes >= (1u << 20) && (bytes % (1u << 20)) == 0) {
        std::snprintf(buf, sizeof(buf), "%u MiB", bytes >> 20);
    } else if (bytes >= (1u << 10) && (bytes % (1u << 10)) == 0) {
        std::snprintf(buf, sizeof(buf), "%u KiB", bytes >> 10);
    } else {
        std::snprintf(buf, sizeof(buf), "%u B", bytes);
    }
    return std::string(buf);
}

std::string rjust(const std::string& s, size_t width) {
    if (s.size() >= width) return s;
    return std::string(width - s.size(), ' ') + s;
}

std::string ljust(const std::string& s, size_t width) {
    if (s.size() >= width) return s;
    return s + std::string(width - s.size(), ' ');
}

// GB/s in decimal 1e9 bytes, because that is the unit every drive is rated in and
// quoting a binary number against a decimal rating is how a 7% discrepancy gets
// mistaken for an engine problem.
std::string fmt_bw(double bytes_per_second) {
    return fmt_f(bytes_per_second / 1.0e9, 2);
}

void render_grid(std::ostringstream& os, const ClassResult& r, const char* title) {
    constexpr size_t kLabelWidth = 22;
    constexpr size_t kCellWidth = 9;

    os << "  " << ljust(title, kLabelWidth);
    for (uint32_t d : r.depth_axis) os << rjust("QD " + std::to_string(d), kCellWidth);
    os << "\n";

    const size_t nd = r.depth_axis.size();
    for (size_t bi = 0; bi < r.block_axis.size(); ++bi) {
        os << "  " << ljust("  " + fmt_block(r.block_axis[bi]), kLabelWidth);
        for (size_t di = 0; di < nd; ++di) {
            const size_t idx = bi * nd + di;
            if (idx >= r.sweep.size()) { os << rjust("--", kCellWidth); continue; }
            const SweepPoint& p = r.sweep[idx];
            std::string cell;
            if (p.skipped || p.bw <= 0.0) {
                cell = "--";
            } else {
                cell = fmt_bw(p.bw);
                if (p.block_bytes == r.block_bytes && p.effective_depth == r.depth) cell += "*";
                else if (p.clipped) cell += "c";
            }
            os << rjust(cell, kCellWidth);
        }
        os << "\n";
    }
}

}  // namespace

const char* traffic_class_name(TrafficClass c) {
    switch (c) {
        case TrafficClass::Stream: return "stream";
        case TrafficClass::Gather: return "gather";
        default: return "unknown";
    }
}

std::vector<uint32_t> default_block_sizes() {
    return {64u << 10, 128u << 10, 256u << 10, 512u << 10, 1u << 20, 2u << 20};
}

std::vector<uint32_t> default_depths() {
    return {1, 4, 16, 32, 64};
}

uint64_t upper_bound_bytes(const Options& o, uint32_t align_unit) {
    const uint64_t unit = align_unit ? align_unit : 1;

    std::vector<uint32_t> blocks;
    for (uint32_t b : o.block_sizes) {
        if (b == 0) continue;
        const uint64_t e = round_up_u64(b, unit);
        if (e == 0 || e > 0xFFFFFFFFull) continue;
        const uint32_t eb = static_cast<uint32_t>(e);
        if (std::find(blocks.begin(), blocks.end(), eb) == blocks.end()) blocks.push_back(eb);
    }
    std::vector<uint32_t> depths;
    for (uint32_t d : o.depths) {
        if (d == 0) continue;
        if (std::find(depths.begin(), depths.end(), d) == depths.end()) depths.push_back(d);
    }

    uint64_t per_class = 0;
    for (uint32_t b : blocks) {
        for (uint32_t d : depths) per_class += point_target_bytes(o, b, d);
    }

    const uint64_t classes = (o.measure_stream ? 1u : 0u) + (o.measure_gather ? 1u : 0u);
    return per_class * classes + o.warmup_bytes;
}

double Calibration::bandwidth(TrafficClass c) const {
    switch (c) {
        case TrafficClass::Stream: return stream_bw;
        case TrafficClass::Gather: return gather_bw;
        default: return 0.0;
    }
}

uint32_t Calibration::block_for(TrafficClass c) const {
    switch (c) {
        case TrafficClass::Stream: return stream.block_bytes;
        case TrafficClass::Gather: return gather.block_bytes;
        default: return 0;
    }
}

uint32_t Calibration::depth_for(TrafficClass c) const {
    switch (c) {
        case TrafficClass::Stream: return stream.depth;
        case TrafficClass::Gather: return gather.depth;
        default: return 0;
    }
}

CeilingFraction Calibration::fraction_of_ceiling(uint64_t bytes_stream_in,
                                                 uint64_t bytes_gather_in,
                                                 double elapsed_seconds) const {
    CeilingFraction f;
    f.bytes_stream = bytes_stream_in;
    f.bytes_gather = bytes_gather_in;
    f.elapsed_seconds = elapsed_seconds;

    if (elapsed_seconds <= 0.0) return f;
    // An unmeasured ceiling is unknown, not 1.0 and not 0.0 (Invariant 6). Refusing
    // here is what keeps a missing calibration from silently becoming a number.
    if (bytes_stream_in > 0 && stream_bw <= 0.0) return f;
    if (bytes_gather_in > 0 && gather_bw <= 0.0) return f;

    const double ideal_stream = stream_bw > 0.0
        ? static_cast<double>(bytes_stream_in) / stream_bw : 0.0;
    const double ideal_gather = gather_bw > 0.0
        ? static_cast<double>(bytes_gather_in) / gather_bw : 0.0;
    const double ideal = ideal_stream + ideal_gather;
    if (ideal <= 0.0) return f;

    const double total_bytes = static_cast<double>(bytes_stream_in) +
                               static_cast<double>(bytes_gather_in);

    f.ideal_seconds = ideal;
    f.stream_share  = ideal_stream / ideal;
    f.gather_share  = ideal_gather / ideal;

    // The blended ceiling moves with the mix. That is the whole point: at a small
    // cap the mix is ~90% stream and this number sits near stream_bw; at the knee
    // it is 100% gather and it sits at gather_bw. A single fixed ceiling would
    // show the fraction collapsing across that range and report a regression the
    // engine did not cause.
    f.ceiling_bw  = total_bytes / ideal;
    f.achieved_bw = total_bytes / elapsed_seconds;
    f.fraction    = ideal / elapsed_seconds;  // == achieved_bw / ceiling_bw

    // Not clamped to 1.0. Above 1.0 means calibration was pessimistic -- a colder
    // drive, or better coalescing than the sweep's shape -- and clamping it would
    // hide a measurement error in the flattering direction.
    f.valid = true;
    return f;
}

CeilingSplit Calibration::fraction_of_ceiling(uint64_t bytes_stream_in, double elapsed_stream,
                                              uint64_t bytes_gather_in,
                                              double elapsed_gather) const {
    CeilingSplit s;

    if (stream_bw > 0.0 && elapsed_stream > 0.0) {
        s.stream.achieved_bw = static_cast<double>(bytes_stream_in) / elapsed_stream;
        s.stream.ceiling_bw  = stream_bw;
        s.stream.fraction    = s.stream.achieved_bw / stream_bw;
        s.stream.valid       = true;
    }
    if (gather_bw > 0.0 && elapsed_gather > 0.0) {
        s.gather.achieved_bw = static_cast<double>(bytes_gather_in) / elapsed_gather;
        s.gather.ceiling_bw  = gather_bw;
        s.gather.fraction    = s.gather.achieved_bw / gather_bw;
        s.gather.valid       = true;
    }
    return s;
}

Calibration calibrate(io::Backend& backend, io::FileId file, uint64_t file_size,
                      const Options& options) {
    Calibration c;
    c.seed = options.seed;
    c.gather_extent_from_model = options.gather_extent_from_model;
    c.gather_concurrent_extents =
        options.gather_concurrent_extents ? options.gather_concurrent_extents : 1;

    if (file == io::kInvalidFile) {
        c.error = "calibration: invalid file id";
        return c;
    }

    // Clamp DOWN to what the backend says the file is, never up: a read past EOF
    // comes back as a short read and would be scored as a slow one.
    uint64_t usable = file_size;
    const uint64_t backend_size = backend.size(file);
    if (backend_size > 0 && backend_size < usable) usable = backend_size;
    if (usable == 0) {
        c.error = "calibration: file size is zero";
        return c;
    }
    c.file_bytes = usable;

    c.alignment = backend.alignment(file);
    const uint32_t unit = c.alignment.max() ? c.alignment.max() : 1;

    // Round every block up to the runtime alignment, then dedupe. Because each
    // effective block is a whole multiple of the alignment unit, every offset
    // (block_index * block) and every length (block) satisfies io::Alignment with
    // no constant anywhere in this file (Invariant 5).
    std::vector<uint32_t> blocks;
    for (uint32_t b : options.block_sizes) {
        if (b == 0) continue;
        const uint64_t e = round_up_u64(b, unit);
        if (e == 0 || e > 0xFFFFFFFFull || e > usable) continue;
        const uint32_t eb = static_cast<uint32_t>(e);
        if (std::find(blocks.begin(), blocks.end(), eb) == blocks.end()) blocks.push_back(eb);
    }
    std::sort(blocks.begin(), blocks.end());
    if (blocks.empty()) {
        c.error = "calibration: no block size survives this file's size and alignment";
        return c;
    }

    std::vector<uint32_t> depths;
    for (uint32_t d : options.depths) {
        if (d == 0) continue;
        if (std::find(depths.begin(), depths.end(), d) == depths.end()) depths.push_back(d);
    }
    std::sort(depths.begin(), depths.end());
    if (depths.empty()) {
        c.error = "calibration: no queue depth to sweep";
        return c;
    }

    c.gather_extent_bytes =
        options.gather_extent_bytes ? options.gather_extent_bytes : blocks.back();

    // Staging: enough for the deepest point, bounded by the option, and halved
    // until it can actually be reserved. Halving rather than failing means a tight
    // cap loses the deepest points -- which are then reported as skipped -- instead
    // of losing the whole calibration.
    uint64_t want = static_cast<uint64_t>(blocks.back()) * depths.back();
    if (want > options.max_staging_bytes) want = options.max_staging_bytes;
    // Floor is the SMALLEST block, not the largest: a staging budget too small for
    // the 2 MiB row should cost that row, not silently overrun the budget the
    // caller set against the cap.
    if (want < blocks.front()) want = blocks.front();
    want = round_up_u64(want, unit);

    const uint32_t mem_align = c.alignment.memory ? c.alignment.memory : 1;
    std::unique_ptr<Staging> staging;
    for (uint64_t size = want;;) {
        auto attempt = std::make_unique<Staging>(options.accountant,
                                                 static_cast<size_t>(size), mem_align);
        if (attempt->ok()) {
            staging = std::move(attempt);
            break;
        }
        if (size <= blocks.front()) break;
        uint64_t half = round_up_u64(size / 2, unit);
        if (half < blocks.front()) half = blocks.front();
        if (half >= size) break;
        size = half;
    }
    if (!staging) {
        c.error = "calibration: could not reserve a staging buffer of " +
                  fmt_size(blocks.front()) + " within the cap";
        return c;
    }
    c.staging_bytes = staging->bytes();

    PointCtx ctx;
    ctx.be = &backend;
    ctx.file = file;
    ctx.staging = staging->base();
    ctx.staging_bytes = staging->bytes();
    ctx.usable = usable;
    ctx.opt = &options;

    const Clock::time_point run0 = Clock::now();
    bool stalled = false;

    // One warm-up pass, discarded. Without it the first cell is charged for the
    // backend's completion threads starting and for the drive leaving a low-power
    // state, and the sweep's first row slopes for a reason that is not block size.
    if (options.warmup_bytes > 0) {
        Options warm = options;
        warm.min_batches_per_point = 1;
        warm.min_bytes_per_point = options.warmup_bytes;
        warm.max_bytes_per_point = options.warmup_bytes;
        warm.max_ms_per_point = 2000;

        PointCtx wctx = ctx;
        wctx.opt = &warm;
        const uint32_t warm_depth = depths.back() < 4u ? depths.back() : 4u;
        const SweepPoint wp = run_point(wctx, TrafficClass::Stream, blocks.back(),
                                        warm_depth, &stalled);
        c.total_bytes_read += wp.bytes;
    }

    if (options.measure_stream) {
        c.stream = run_class(ctx, TrafficClass::Stream, blocks, depths, &stalled);
        c.total_bytes_read += c.stream.bytes_read;
    }
    if (options.measure_gather) {
        c.gather = run_class(ctx, TrafficClass::Gather, blocks, depths, &stalled);
        c.total_bytes_read += c.gather.bytes_read;
    }

    c.total_seconds = secs_between(run0, Clock::now());

    if (stalled) {
        staging->disarm();
        c.error = "calibration: the storage backend stopped completing reads; the "
                  "staging buffer was abandoned rather than freed because the "
                  "interface has no cancel()";
        return c;
    }

    c.stream_bw = c.stream.ok ? c.stream.bw : 0.0;
    c.gather_bw = c.gather.ok ? c.gather.bw : 0.0;

    if (options.measure_stream && !c.stream.ok) {
        c.error = "calibration: the stream sweep produced no usable point";
        return c;
    }
    if (options.measure_gather && !c.gather.ok) {
        c.error = "calibration: the gather sweep produced no usable point";
        return c;
    }

    // The engine's single operating point comes from gather: decode is the hot
    // path, it is the class with the one-router-step horizon, and its block*depth
    // sets the staging charge against the cap. Prefill is a single ordered scan and
    // should use stream.block_bytes / stream.depth instead.
    if (c.gather.ok) {
        c.block_bytes = c.gather.block_bytes;
        c.queue_depth = c.gather.depth;
    } else if (c.stream.ok) {
        c.block_bytes = c.stream.block_bytes;
        c.queue_depth = c.stream.depth;
    }

    c.ok = c.block_bytes != 0;
    if (!c.ok) c.error = "calibration: no traffic class was measured";
    return c;
}

std::string Calibration::report() const {
    std::ostringstream os;

    os << "storage calibration\n";
    if (!ok) {
        os << "  UNAVAILABLE: " << (error.empty() ? "unknown reason" : error) << "\n";
        os << "  bandwidth is reported as unknown; no default is substituted\n";
        return os.str();
    }

    char seedbuf[32];
    std::snprintf(seedbuf, sizeof(seedbuf), "0x%016llX", static_cast<unsigned long long>(seed));

    os << "  cost: " << fmt_size(total_bytes_read) << " read in "
       << fmt_f(total_seconds, 2) << " s (seed " << seedbuf << ")\n";
    os << "  alignment: memory " << alignment.memory << " B / offset " << alignment.offset
       << " B / length " << alignment.length << " B; swept " << fmt_size(file_bytes)
       << "; staging " << fmt_size(staging_bytes) << "\n";
    os << "  gather shape: " << fmt_size(gather_extent_bytes) << " extents, "
       << gather_concurrent_extents << " concurrent"
       << (gather_extent_from_model ? " (extent size from the model's expert slots)"
                                    : " (extent size ASSUMED, not from the model)")
       << "\n";

    os << "\n";
    os << "  stream ceiling: " << fmt_bw(stream_bw) << " GB/s at "
       << fmt_block(stream.block_bytes) << " x QD " << stream.depth;
    if (stream.peak_bw > 0.0) {
        os << "   [peak " << fmt_bw(stream.peak_bw) << " at "
           << fmt_block(stream.peak_block_bytes) << " x QD " << stream.peak_depth << "]";
    }
    os << "\n";
    os << "  gather ceiling: " << fmt_bw(gather_bw) << " GB/s at "
       << fmt_block(gather.block_bytes) << " x QD " << gather.depth;
    if (gather.peak_bw > 0.0) {
        os << "   [peak " << fmt_bw(gather.peak_bw) << " at "
           << fmt_block(gather.peak_block_bytes) << " x QD " << gather.peak_depth << "]";
    }
    os << "\n";
    os << "  engine operating point: " << fmt_block(block_bytes) << " x QD " << queue_depth
       << " (gather; decode is the hot path)\n";
    os << "  the two ceilings are never merged into one number: the traffic mix moves\n"
          "  from ~90% stream at a small cap to 100% gather at the knee\n";

    os << "\n";
    if (!stream.sweep.empty()) {
        render_grid(os, stream, "stream GB/s");
        os << "\n";
    }
    if (!gather.sweep.empty()) {
        render_grid(os, gather, "gather GB/s");
        os << "\n";
    }
    os << "  * = chosen operating point, c = depth clipped by staging or backend, "
          "-- = not measured\n";

    const uint32_t errs = stream.errors + gather.errors;
    const uint32_t shorts = stream.short_reads + gather.short_reads;
    if (errs || shorts) {
        os << "\n  WARNING: " << errs << " failed and " << shorts
           << " short reads during calibration. Failed reads contributed no bytes, so\n"
              "  these ceilings are not inflated by them -- but a drive returning errors\n"
              "  under a 50 ms sweep will return more of them under a multi-day run.\n";
    }

    return os.str();
}

}  // namespace dray::calib

// T5: summary persistence. See the header note: loaded ceilings travel with
// their age, never as fresh truth.
#include "json.hpp"
#include <ctime>
#include <fstream>

namespace dray::calib {

bool save_calibration(const Calibration& c, const std::string& path, std::string* err) {
    if (!c.ok) {
        if (err) *err = "refusing to persist a failed calibration";
        return false;
    }
    char when[32];
    const std::time_t now = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    gmtime_s(&tmv, &now);
#else
    gmtime_r(&now, &tmv);
#endif
    std::strftime(when, sizeof(when), "%Y-%m-%dT%H:%M:%SZ", &tmv);

    nlohmann::json j = {
        { "version", 1 },
        { "when", when },
        { "stream_bw", c.stream_bw },
        { "gather_bw", c.gather_bw },
        { "block_bytes", c.block_bytes },
        { "queue_depth", c.queue_depth },
        { "stream", { { "block_bytes", c.stream.block_bytes },
                      { "depth", c.stream.depth },
                      { "bw", c.stream.bw },
                      { "peak_bw", c.stream.peak_bw } } },
        { "gather", { { "block_bytes", c.gather.block_bytes },
                      { "depth", c.gather.depth },
                      { "bw", c.gather.bw },
                      { "peak_bw", c.gather.peak_bw } } },
        { "alignment", { { "memory", c.alignment.memory },
                         { "offset", c.alignment.offset },
                         { "length", c.alignment.length } } },
        { "total_bytes_read", c.total_bytes_read },
        { "total_seconds", c.total_seconds },
        { "seed", c.seed },
    };
    std::ofstream f(path, std::ios::trunc);
    f << j.dump(2);
    if (!f.good()) {
        if (err) *err = "could not write " + path;
        return false;
    }
    return true;
}

bool load_calibration(Calibration* c, std::string* when, const std::string& path,
                      std::string* err) {
    std::ifstream f(path);
    if (!f.good()) {
        if (err) *err = "no calibration at " + path;
        return false;
    }
    nlohmann::json j;
    try {
        f >> j;
        if (j.value("version", 0) != 1) {
            if (err) *err = "unknown calibration version";
            return false;
        }
        c->stream_bw = j.value("stream_bw", 0.0);
        c->gather_bw = j.value("gather_bw", 0.0);
        c->block_bytes = j.value("block_bytes", 0u);
        c->queue_depth = j.value("queue_depth", 0u);
        if (when) *when = j.value("when", "unknown time");
        c->ok = c->stream_bw > 0.0 && c->gather_bw > 0.0;
        if (!c->ok && err) *err = "calibration has no usable ceilings";
        return c->ok;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("unparseable calibration: ") + ex.what();
        return false;
    }
}

}  // namespace dray::calib
