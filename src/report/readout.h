// The live readout: tokens, rate, ETA, bandwidth, and the one question this
// project exists to answer -- is the engine or the drive the limit?
//
// This is a product feature, not logging (CLAUDE.md, Invariant 6). Four rules
// shape every decision below:
//
//   * SECONDS PER TOKEN IS THE DEFAULT UNIT, flipping to tok/s only above
//     1 tok/s. "3.4 s/tok" is a figure a user can plan a day around; the same
//     run printed as "0.29 tok/s" hides the magnitude they are planning around.
//     ETAs are rendered in hours and days.
//
//   * NO float32 ANYWHERE. Counters are int64_t, derived quantities double. A
//     float32 counter is exact only to 2^24 = 16,777,216: a token counter would
//     survive a multi-day run, but at ~1 GB/s the byte counters cross 2^24 in
//     under 17 ms and would then round silently -- and a silently rounded byte
//     total is exactly the kind of flattering number this project forbids.
//
//   * AN ETA CARRIES ITS WINDOW. An ETA is a division and the divisor is a
//     choice, so every ETA here reports the span and token count it came from.
//     When the rate has moved more than Options::drift_threshold within the last
//     hour the point estimate is WITHDRAWN and replaced by a range: an ETA taken
//     from an instantaneous rate on a run whose rate is falling is an optimistic
//     figure, and this engine does not emit optimistic figures.
//
//   * UNOBTAINABLE IS "unknown". Never omitted, never defaulted to zero. In the
//     JSONL stream the key is always present with a null value, so a consumer
//     can distinguish "not measured" from "measured zero".
//
// Two sinks, because a multi-day run outlives the terminal it started in:
//   (a) one terminal line that refreshes in place, and
//   (b) a JSONL status stream appended to a file, size-bounded by rotation, so a
//       detached run can be inspected later with an ordinary `tail`.
// The terminal line is a viewport and may be clamped to the terminal width; the
// JSONL record and the status block never drop a field.
//
// Not thread-safe: it is ticked from the decode loop. A mutex would buy nothing
// and would sit on the hot path.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "plan/residency.h"

namespace dray::report {

// The two read classes are measured and reported SEPARATELY because they have
// different ceilings and different profiles (CLAUDE.md, cost model): the stream
// class is a single ordered scan, for which `fio --rw=read` is the right model;
// the gather class is large-block random, for which it is the wrong one. One
// blended "disk bandwidth" number would let a stalled gather hide behind a
// healthy stream.
enum class ByteClass : uint8_t {
    Stream = 0,   // unconditional bulk: read every token when not resident
    Gather = 1,   // routed experts: read only when the router selects them
    _Count = 2,
};
inline constexpr size_t kNumClasses = static_cast<size_t>(ByteClass::_Count);

const char* byte_class_name(ByteClass);

enum class Phase : uint8_t { Idle, Prefill, Decode, Done };
const char* phase_name(Phase);

// The headline verdict. Derived only from the per-class ceiling fractions, and
// only when a ceiling was actually measured -- never guessed from feel.
enum class Limiter : uint8_t {
    Unknown,  // no calibrated ceiling, or nothing read yet
    Drive,    // at or near the measured ceiling: this is as fast as the disk goes
    Engine,   // far below the ceiling: the disk is idle waiting on us
    Mixed,
};
const char* limiter_name(Limiter);

// Everything the engine knows, as raw cumulative facts. The readout derives; the
// caller never pre-divides, because a caller that divides has already chosen a
// window and the whole point is that the window is disclosed.
//
// Values that genuinely cannot be obtained stay empty and are rendered as
// "unknown". Do NOT pass 0 to mean "not measured".
struct Counters {
    Phase   phase = Phase::Idle;

    int64_t tokens_done = 0;
    std::optional<int64_t> tokens_requested;  // empty = open-ended: no ETA is possible
    std::optional<int64_t> context_len;
    std::optional<int64_t> context_cap;

    // Cumulative bytes actually transferred from disk, per class. Bytes that hit
    // in the cache are NOT counted here -- this is disk traffic, not demand.
    int64_t bytes_read[kNumClasses] = {};

    // Two distinct hit rates, never one symbol (CLAUDE.md): at the knee h_routed
    // is 0 while h_bytes is ~87-90%, so a single "cache hit rate" would be a
    // different number depending on which one the reader assumed.
    int64_t routed_bytes_demanded = 0;  // h_routed denominator: routed-expert bytes only
    int64_t routed_bytes_hit = 0;
    int64_t active_bytes_demanded = 0;  // h_bytes denominator: ALL active bytes per token
    int64_t active_bytes_hit = 0;

    std::optional<int64_t> resident_bytes;  // mem::Accountant::used()
    std::optional<int64_t> resident_cap;    // mem::Accountant::cap()
    std::optional<int64_t> rss_bytes;       // mem::Accountant::process_rss(), 0 -> pass empty

    std::optional<int64_t> reads_issued;
    std::optional<int64_t> short_reads;   // io::Completion::bytes < requested
    std::optional<int64_t> read_errors;   // io::Completion::status != 0

    // fed[] distinguishes "measured zero" (everything from RAM: honest 0 rate)
    // from "nothing ever wired here" (unknown, per the never-defaulted rule).
    bool    bytes_fed[kNumClasses] = {};
    int64_t bytes_of(ByteClass c) const { return bytes_read[static_cast<size_t>(c)]; }
    void    add_bytes(ByteClass c, int64_t n) {
        bytes_read[static_cast<size_t>(c)] += n;
        bytes_fed[static_cast<size_t>(c)] = true;
    }
};

// Measured read ceilings, per class, at THIS engine's own block size and queue
// depth -- never a vendor sequential rating (CLAUDE.md, cost model). Empty means
// uncalibrated, which prints as "unknown" and suppresses the limiter verdict
// rather than substituting a plausible-looking default.
struct Ceilings {
    std::optional<double> bytes_per_s[kNumClasses];
    std::string provenance;  // e.g. "calibrated 128 KiB blocks QD 32, 2026-08-12"

    std::optional<double> of(ByteClass c) const { return bytes_per_s[static_cast<size_t>(c)]; }
    void set(ByteClass c, double v) { bytes_per_s[static_cast<size_t>(c)] = v; }
};

// A rate and an ETA are never bare numbers here: they travel with the window
// they were computed from and with the evidence that the window is still
// representative.
struct RateEstimate {
    std::optional<double> tokens_per_s;  // empty when there is not enough data yet

    int64_t window_ns = 0;      // the span actually used, disclosed to the user
    int64_t window_tokens = 0;
    bool    whole_run_fallback = false;  // the recent window was too thin to divide by

    // Rate movement over (up to) the last hour, measured as the second half of
    // the span against the first half. Raw fractional change over drift_span_ns
    // -- deliberately NOT extrapolated to a full hour, since extrapolating a
    // trend is inventing a measurement.
    std::optional<double> drift_fraction;
    std::optional<double> early_tokens_per_s;
    std::optional<double> late_tokens_per_s;
    int64_t drift_span_ns = 0;
    bool    falling = false;
    bool    rising = false;

    // Falling run: eta_seconds is ABSENT and eta_high_seconds is ABSENT too --
    // eta_low_seconds is a lower bound and nothing observed bounds the other end
    // while the rate keeps dropping. Stable or rising: all three are populated,
    // and the point estimate never comes from the faster of the observed rates.
    std::optional<double> eta_seconds;
    std::optional<double> eta_low_seconds;
    std::optional<double> eta_high_seconds;
    bool    eta_is_range = false;
    bool    eta_upper_open = false;  // rate still falling: there is no upper bound

    std::string window_label;  // e.g. "5m00s/88tok"
    std::string note;          // why the ETA looks the way it does; may be empty
};

struct Snapshot {
    Counters counters;

    int64_t now_ns = 0;
    int64_t run_start_ns = 0;
    int64_t wall_elapsed_ns = 0;
    int64_t phase_elapsed_ns = 0;

    // Mean over the current phase. Prefill and decode differ by two orders of
    // magnitude, so blending them would produce a figure describing neither.
    std::optional<double> seconds_per_token;

    RateEstimate rate;

    std::optional<double> h_routed;  // cache bytes / routed-expert bytes
    std::optional<double> h_bytes;   // active bytes served from RAM

    // Wall-clock sustained throughput over the disclosed window -- NOT bytes
    // divided by time-spent-inside-reads. In-read time would exclude the gaps
    // and report a bandwidth the run never achieved.
    std::optional<double> sustained_bytes_per_s[kNumClasses];
    std::optional<double> run_bytes_per_s[kNumClasses];
    std::optional<double> ceiling_fraction[kNumClasses];
    int64_t bandwidth_window_ns = 0;

    Limiter limiter = Limiter::Unknown;
};

class Readout {
public:
    struct Options {
        std::string run_id;   // echoed into every JSONL record so a detached run is identifiable
        std::string model;

        // JSONL sink. Bounded by rotation: at one record per 5 s a record is
        // ~800 B, i.e. ~14 MB/day, so 32 MiB + 1 rotation retains roughly the
        // last 4.5 days of a run at the default interval.
        std::string jsonl_path;
        int64_t     jsonl_max_bytes = 32ll << 20;
        int32_t     jsonl_rotations = 1;
        int64_t     jsonl_min_interval_ns = 5'000'000'000;

        // 4 Hz. At seconds-per-token nothing but the elapsed clock changes
        // between frames, and a faster refresh only makes the line unreadable.
        int64_t terminal_min_interval_ns = 250'000'000;
        int32_t terminal_width = 120;  // <= 0 disables clamping

        int64_t rate_window_ns = 300'000'000'000;     // 5 min
        int64_t rate_min_window_ns = 30'000'000'000;  // below this it is not a rate
        int64_t rate_min_tokens = 2;
        int64_t drift_span_ns = 3'600'000'000'000;    // "within the last hour"
        double  drift_threshold = 0.15;

        // Sampling granularity of the rate history. 5 s over a 1 h span is 720
        // samples of 32 B: bounded, and fine enough for a 5 min window.
        int64_t history_gap_ns = 5'000'000'000;

        // The only two unmeasured numbers in this file. They classify, they do
        // not compute: the underlying fractions are always printed alongside the
        // verdict so a user can disagree with the thresholds.
        double drive_limited_fraction = 0.85;
        double engine_limited_fraction = 0.50;

        // Prefill runs ~100x faster than decode; carrying prefill samples into
        // the decode window would produce an ETA that is wrong by two orders of
        // magnitude, in the optimistic direction.
        bool clear_history_on_phase_change = true;
    };

    // The default argument lives OUT of class: GCC requires the enclosing class
    // complete before a nested type's default member initializers can be used in
    // a default argument; MSVC does not. Same API either way.
    explicit Readout(Options opt);
    Readout();
    ~Readout();

    Readout(const Readout&) = delete;
    Readout& operator=(const Readout&) = delete;

    const Options& options() const { return opt_; }

    void set_ceilings(const Ceilings&);
    const Ceilings& ceilings() const { return ceil_; }

    // Free-form identification printed in the status block and the final
    // summary: backend describe() string, plan summary, GGUF path.
    void set_context(std::string backend, std::string plan_summary);

    // now_ns must come from a steady clock -- monotonic_ns() below is the one.
    // A wall clock would let an NTP step produce a negative elapsed time.
    void update(const Counters&, int64_t now_ns);

    // Drops the rate history without touching the run-elapsed clock. Called
    // automatically on a phase change unless disabled.
    void reset_rate_history(int64_t now_ns);

    Snapshot snapshot() const;

    std::string terminal_line() const;  // one line, clamped to terminal_width
    std::string status_block() const;   // every field, nothing clamped
    std::string final_summary() const;

    // Refreshes in place: CR, line, pad over the previous line's tail. Leaves no
    // newline, so the next refresh overwrites this one. Give it a DIFFERENT
    // stream from the one generated tokens go to (stderr vs stdout), or the CR
    // will chew through the output the user actually asked for.
    void render_terminal(std::ostream&);
    void end_terminal(std::ostream&);

    // Appends one record and flushes, so an external reader sees it immediately
    // and a kill -9 loses at most the record in flight. Rotates past the bound.
    bool        write_jsonl();
    std::string jsonl_record() const;
    bool        jsonl_healthy() const { return jsonl_error_.empty(); }
    const std::string& jsonl_error() const { return jsonl_error_; }

    // update() + rate-limited terminal refresh + rate-limited JSONL append.
    // term may be null for a headless run.
    void tick(const Counters&, int64_t now_ns, std::ostream* term);

private:
    struct Sample {
        int64_t ns = 0;
        int64_t tokens = 0;
        int64_t bytes[kNumClasses] = {};
    };

    const Sample* anchor_at(int64_t target_ns) const;
    RateEstimate  compute_rate() const;
    bool          ensure_jsonl_open();
    void          rotate_jsonl();

    Options  opt_;
    Ceilings ceil_;
    std::string backend_;
    std::string plan_summary_;

    Counters cur_;
    std::deque<Sample> history_;

    bool    started_ = false;
    int64_t run_start_ns_ = 0;
    int64_t now_ns_ = 0;
    Phase   phase_ = Phase::Idle;
    int64_t phase_start_ns_ = 0;
    int64_t phase_start_tokens_ = 0;

    int64_t last_terminal_ns_ = 0;
    int64_t last_jsonl_ns_ = 0;
    size_t  last_line_len_ = 0;

    std::ofstream jsonl_;
    int64_t       jsonl_bytes_ = 0;
    std::string   jsonl_error_;
};

// Steady clock in nanoseconds. The only clock the readout should be fed.
int64_t monotonic_ns();

// ---------------------------------------------------------------------------
// Formatters. Free functions so the startup report and the refusal path can use
// the same rendering the live line uses. Every one of them renders an empty
// optional as "unknown" -- that is the whole reason they take optionals.
//
// ASCII only: the Windows console codepage is not reliably UTF-8, and a readout
// that renders as mojibake is a readout the user stops trusting.
// ---------------------------------------------------------------------------

// SECONDS PER TOKEN by default; tok/s only above 1 tok/s (Invariant 6).
std::string format_token_rate(std::optional<double> tokens_per_s);

std::string format_duration(std::optional<double> seconds);   // hours and days
std::string format_bytes(std::optional<int64_t> bytes);       // binary: memory is binary
std::string format_bandwidth(std::optional<double> bytes_per_s);  // decimal GB/s: matches
                                                                  // drive datasheets and fio
std::string format_percent(std::optional<double> fraction);
std::string format_count(std::optional<int64_t> n);

// The RAM curve, from plan::Plan::ram_curve(). Rendered as PROJECTIONS and
// labelled as such: the byte figures are exact plan arithmetic, but converting
// them to s/tok assumes a bandwidth and assumes no compute overlap, so they are
// an upper bound on what a larger cap buys, not a promise.
//
// bytes_per_s should be the measured GATHER ceiling: near the design centre the
// marginal byte a bigger cap saves is a routed-expert gather, not a stream read.
// Empty renders the projection column as "unknown" rather than inventing one.
std::string ram_curve_table(const std::vector<plan::Plan::CurvePoint>& curve,
                            uint64_t current_cap,
                            std::optional<double> bytes_per_s = std::nullopt,
                            const std::string& bandwidth_label = std::string());

}  // namespace dray::report
