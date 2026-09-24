#include "report/readout.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <system_error>
#include <utility>

namespace dray::report {

namespace {

constexpr double kNsPerSec = 1e9;

double ns_to_s(int64_t ns) { return static_cast<double>(ns) / kNsPerSec; }

// MSVC annotates snprintf with _Printf_format_string_, so a forwarded format
// parameter trips C4774 at /W4. The format strings all originate as literals at
// the call sites; the indirection is only here.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4774)
#endif
template <typename... Args>
std::string sprintf_str(const char* format, Args... args) {
    char buf[256];
    const int n = std::snprintf(buf, sizeof(buf), format, args...);
    if (n < 0) {
        return std::string();
    }
    const size_t need = static_cast<size_t>(n);
    if (need < sizeof(buf)) {
        return std::string(buf, need);
    }
    std::string big(need + 1, '\0');
    std::snprintf(&big[0], big.size(), format, args...);
    big.resize(need);
    return big;
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

const char* const kUnknown = "unknown";

bool usable(const std::optional<double>& v) {
    return v.has_value() && std::isfinite(*v);
}

// Thousands separators: a raw 12004113 is unreadable in a line the user scans
// four times a second.
std::string grouped(int64_t v) {
    // Magnitude taken through uint64_t: negating INT64_MIN is UB, and a counter
    // that has gone negative is exactly when this gets called.
    const uint64_t mag = v < 0 ? (~static_cast<uint64_t>(v) + 1u) : static_cast<uint64_t>(v);
    const std::string digits = std::to_string(mag);
    std::string out;
    out.reserve(digits.size() + digits.size() / 3 + 1);
    const size_t lead = digits.size() % 3 == 0 ? 3 : digits.size() % 3;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i == lead || (i > lead && (i - lead) % 3 == 0)) {
            out.push_back(',');
        }
        out.push_back(digits[i]);
    }
    return v < 0 ? ("-" + out) : out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char ch : s) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out += sprintf_str("\\u%04x", static_cast<unsigned int>(static_cast<unsigned char>(ch)));
                } else {
                    out.push_back(ch);
                }
        }
    }
    return out;
}

std::string jstr(const std::string& s) { return "\"" + json_escape(s) + "\""; }

// Unobtainable is null, never 0 and never absent: a consumer must be able to
// tell "not measured" from "measured zero" (Invariant 6).
std::string jnum(const std::optional<double>& v) {
    if (!usable(v)) {
        return "null";
    }
    return sprintf_str("%.10g", *v);
}

std::string jint(const std::optional<int64_t>& v) {
    if (!v.has_value()) {
        return "null";
    }
    return std::to_string(*v);
}

std::string jbool(bool b) { return b ? "true" : "false"; }

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()).count() % 1000;
    std::tm tmv{};
#if defined(_WIN32)
    if (gmtime_s(&tmv, &t) != 0) {
        return std::string(kUnknown);
    }
#else
    if (gmtime_r(&t, &tmv) == nullptr) {
        return std::string(kUnknown);
    }
#endif
    return sprintf_str("%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                       tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                       tmv.tm_hour, tmv.tm_min, tmv.tm_sec, static_cast<int>(ms));
}

std::string eta_display(const RateEstimate& r) {
    if (r.eta_seconds.has_value() && !r.eta_is_range) {
        return format_duration(r.eta_seconds);
    }
    if (r.eta_low_seconds.has_value() && r.eta_high_seconds.has_value()) {
        std::string s = format_duration(r.eta_low_seconds) + "-" + format_duration(r.eta_high_seconds);
        if (r.eta_upper_open) {
            s += "+";
        }
        return s;
    }
    if (r.eta_low_seconds.has_value()) {
        // Lower bound only: the rate is still falling, so "not less than this" is
        // the entire defensible statement.
        return ">=" + format_duration(r.eta_low_seconds);
    }
    if (r.eta_seconds.has_value()) {
        return format_duration(r.eta_seconds);
    }
    return std::string(kUnknown);
}

// The window is never optional decoration: it is what makes the ETA above it
// checkable.
std::string window_display(const RateEstimate& r) {
    if (r.window_label.empty()) {
        return std::string("win ") + kUnknown;
    }
    return "win " + r.window_label + (r.whole_run_fallback ? "*" : "");
}

std::string drift_display(const RateEstimate& r) {
    if (!r.drift_fraction.has_value()) {
        return std::string();
    }
    const std::string pct = sprintf_str("%+.1f%%", *r.drift_fraction * 100.0);
    return "rate " + pct + " over " + format_duration(ns_to_s(r.drift_span_ns));
}

std::string bytes_or_unknown(const std::optional<int64_t>& v) { return format_bytes(v); }

}  // namespace

// ---------------------------------------------------------------------------
// names
// ---------------------------------------------------------------------------

const char* byte_class_name(ByteClass c) {
    switch (c) {
        case ByteClass::Stream: return "stream";
        case ByteClass::Gather: return "gather";
        case ByteClass::_Count: break;
    }
    return kUnknown;
}

const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Idle:    return "idle";
        case Phase::Prefill: return "prefill";
        case Phase::Decode:  return "decode";
        case Phase::Done:    return "done";
    }
    return kUnknown;
}

const char* limiter_name(Limiter l) {
    switch (l) {
        case Limiter::Unknown: return "unknown";
        case Limiter::Drive:   return "drive-limited";
        case Limiter::Engine:  return "engine-limited";
        case Limiter::Mixed:   return "mixed";
    }
    return kUnknown;
}

// ---------------------------------------------------------------------------
// formatters
// ---------------------------------------------------------------------------

std::string format_token_rate(std::optional<double> tokens_per_s) {
    if (!usable(tokens_per_s) || *tokens_per_s <= 0.0) {
        return std::string(kUnknown);
    }
    const double r = *tokens_per_s;
    // Invariant 6: seconds per token is the default; tok/s only above 1 tok/s.
    if (r > 1.0) {
        return r >= 100.0 ? sprintf_str("%.0f tok/s", r) : sprintf_str("%.1f tok/s", r);
    }
    const double spt = 1.0 / r;
    if (spt >= 1000.0) {
        return sprintf_str("%.0f s/tok", spt);
    }
    return sprintf_str("%.1f s/tok", spt);
}

std::string format_duration(std::optional<double> seconds) {
    if (!usable(seconds) || *seconds < 0.0) {
        return std::string(kUnknown);
    }
    const double v = *seconds;
    if (v < 1.0) {
        return "<1s";
    }
    const long long t = static_cast<long long>(v);
    if (t < 60) {
        return sprintf_str("%llds", t);
    }
    if (t < 3600) {
        return sprintf_str("%lldm%02llds", t / 60, t % 60);
    }
    // Hours and days past this point: a multi-day run reported in minutes is a
    // number nobody can hold in their head.
    if (t < 172800) {
        return sprintf_str("%lldh%02lldm", t / 3600, (t % 3600) / 60);
    }
    return sprintf_str("%lldd%02lldh", t / 86400, (t % 86400) / 3600);
}

std::string format_bytes(std::optional<int64_t> bytes) {
    if (!bytes.has_value() || *bytes < 0) {
        return std::string(kUnknown);
    }
    const int64_t b = *bytes;
    const double d = static_cast<double>(b);
    // Binary units: memory is what these describe, and every OS reports it this way.
    if (b < 1024) {
        return sprintf_str("%lldB", static_cast<long long>(b));
    }
    if (b < 1024LL * 1024) {
        return sprintf_str("%.1fKiB", d / 1024.0);
    }
    if (b < 1024LL * 1024 * 1024) {
        return sprintf_str("%.1fMiB", d / (1024.0 * 1024.0));
    }
    if (b < 1024LL * 1024 * 1024 * 1024) {
        return sprintf_str("%.2fGiB", d / (1024.0 * 1024.0 * 1024.0));
    }
    return sprintf_str("%.2fTiB", d / (1024.0 * 1024.0 * 1024.0 * 1024.0));
}

std::string format_bandwidth(std::optional<double> bytes_per_s) {
    if (!usable(bytes_per_s) || *bytes_per_s < 0.0) {
        return std::string(kUnknown);
    }
    // Decimal units, deliberately: drive datasheets and fio both report decimal,
    // and the user's whole question is how this compares to those.
    const double v = *bytes_per_s;
    if (v < 1e6) {
        return sprintf_str("%.0f kB/s", v / 1e3);
    }
    if (v < 1e9) {
        return sprintf_str("%.0f MB/s", v / 1e6);
    }
    return sprintf_str("%.2f GB/s", v / 1e9);
}

std::string format_percent(std::optional<double> fraction) {
    if (!usable(fraction)) {
        return std::string(kUnknown);
    }
    return sprintf_str("%.1f%%", *fraction * 100.0);
}

std::string format_count(std::optional<int64_t> n) {
    if (!n.has_value()) {
        return std::string(kUnknown);
    }
    return grouped(*n);
}

int64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Readout
// ---------------------------------------------------------------------------

Readout::Readout(Options opt) : opt_(std::move(opt)) {}
Readout::Readout() : Readout(Options{}) {}

Readout::~Readout() {
    if (jsonl_.is_open()) {
        jsonl_.flush();
        jsonl_.close();
    }
}

void Readout::set_ceilings(const Ceilings& c) { ceil_ = c; }

void Readout::set_context(std::string backend, std::string plan_summary) {
    backend_ = std::move(backend);
    plan_summary_ = std::move(plan_summary);
}

void Readout::reset_rate_history(int64_t now_ns) {
    history_.clear();
    Sample s;
    s.ns = now_ns;
    s.tokens = cur_.tokens_done;
    for (size_t i = 0; i < kNumClasses; ++i) {
        s.bytes[i] = cur_.bytes_read[i];
    }
    history_.push_back(s);
}

void Readout::update(const Counters& c, int64_t now_ns) {
    // A steady clock cannot go backwards, but a caller passing the wrong clock
    // can. Clamping here keeps every span non-negative rather than producing a
    // negative ETA that looks like a bug in the engine.
    if (started_ && now_ns < now_ns_) {
        now_ns = now_ns_;
    }

    const bool first = !started_;
    const bool restart = started_ && c.tokens_done < cur_.tokens_done;
    const bool phase_change = started_ && !restart && c.phase != phase_;

    cur_ = c;
    now_ns_ = now_ns;
    started_ = true;

    if (first || restart) {
        run_start_ns_ = now_ns;
        phase_ = c.phase;
        phase_start_ns_ = now_ns;
        phase_start_tokens_ = c.tokens_done;
        last_terminal_ns_ = 0;
        last_jsonl_ns_ = 0;
        reset_rate_history(now_ns);
    } else if (phase_change) {
        phase_ = c.phase;
        phase_start_ns_ = now_ns;
        phase_start_tokens_ = c.tokens_done;
        // Prefill runs roughly two orders of magnitude faster than decode.
        // Carrying its samples into the decode window would produce an ETA that
        // is wrong by that factor, in the optimistic direction.
        if (opt_.clear_history_on_phase_change) {
            reset_rate_history(now_ns);
        }
    }

    if (history_.empty() || now_ns - history_.back().ns >= opt_.history_gap_ns) {
        Sample s;
        s.ns = now_ns;
        s.tokens = c.tokens_done;
        for (size_t i = 0; i < kNumClasses; ++i) {
            s.bytes[i] = c.bytes_read[i];
        }
        history_.push_back(s);
    }

    // Bounded: the drift span is the oldest thing anything here asks about.
    while (history_.size() > 2 && now_ns - history_.front().ns > opt_.drift_span_ns) {
        history_.pop_front();
    }
}

const Readout::Sample* Readout::anchor_at(int64_t target_ns) const {
    const Sample* best = nullptr;
    for (const Sample& s : history_) {
        if (s.ns <= target_ns) {
            best = &s;
        } else {
            break;  // history_ is ascending in time
        }
    }
    if (best == nullptr && !history_.empty()) {
        best = &history_.front();
    }
    return best;
}

RateEstimate Readout::compute_rate() const {
    RateEstimate r;
    if (!started_ || history_.empty()) {
        r.note = "no samples yet: rate and ETA unknown";
        return r;
    }

    int64_t span = 0;
    int64_t toks = 0;
    if (const Sample* a = anchor_at(now_ns_ - opt_.rate_window_ns)) {
        span = now_ns_ - a->ns;
        toks = cur_.tokens_done - a->tokens;
    }

    // A rate taken from a 200 ms window is not a rate, it is the last token's
    // duration wearing a rate's clothes -- and on this engine one token is
    // seconds long, so the "instantaneous" window is almost always the thin one.
    if (span < opt_.rate_min_window_ns || toks < opt_.rate_min_tokens) {
        span = now_ns_ - phase_start_ns_;
        toks = cur_.tokens_done - phase_start_tokens_;
        r.whole_run_fallback = true;
    }

    r.window_ns = span;
    r.window_tokens = toks;
    if (span > 0 && toks >= 1) {
        r.tokens_per_s = static_cast<double>(toks) / ns_to_s(span);
    }
    r.window_label = format_duration(ns_to_s(span)) + "/" + std::to_string(toks) + "tok";

    // Drift: second half of the covered span against the first half. Capped at
    // drift_span_ns ("within the last hour") and reported raw, never
    // extrapolated to a full hour -- extrapolating a trend is inventing a
    // measurement.
    const int64_t avail = now_ns_ - history_.front().ns;
    const int64_t dspan = std::min(avail, opt_.drift_span_ns);
    if (dspan >= 2 * opt_.rate_min_window_ns) {
        const int64_t t0 = now_ns_ - dspan;
        const int64_t tm = t0 + dspan / 2;
        const Sample* s0 = anchor_at(t0);
        const Sample* sm = anchor_at(tm);
        if (s0 != nullptr && sm != nullptr && sm->ns > s0->ns && now_ns_ > sm->ns) {
            const double early = static_cast<double>(sm->tokens - s0->tokens) / ns_to_s(sm->ns - s0->ns);
            const double late = static_cast<double>(cur_.tokens_done - sm->tokens) / ns_to_s(now_ns_ - sm->ns);
            r.drift_span_ns = now_ns_ - s0->ns;
            r.early_tokens_per_s = early;
            r.late_tokens_per_s = late;
            if (early > 0.0) {
                r.drift_fraction = (late - early) / early;
                r.falling = *r.drift_fraction < -opt_.drift_threshold;
                r.rising = *r.drift_fraction > opt_.drift_threshold;
            } else if (late > 0.0) {
                r.rising = true;  // was stalled, now moving: no finite ratio exists
            }
        }
    }

    if (!cur_.tokens_requested.has_value()) {
        r.note = "no token budget: ETA unknown";
        return r;
    }

    const int64_t remaining = *cur_.tokens_requested - cur_.tokens_done;
    if (remaining <= 0) {
        r.eta_seconds = 0.0;
        r.eta_low_seconds = 0.0;
        r.eta_high_seconds = 0.0;
        return r;
    }
    if (!usable(r.tokens_per_s) || *r.tokens_per_s <= 0.0) {
        r.note = "no tokens observed in the window: ETA unknown";
        return r;
    }

    double fast = *r.tokens_per_s;
    double slow = *r.tokens_per_s;
    if (usable(r.early_tokens_per_s) && *r.early_tokens_per_s > 0.0) {
        fast = std::max(fast, *r.early_tokens_per_s);
        slow = std::min(slow, *r.early_tokens_per_s);
    }
    if (usable(r.late_tokens_per_s) && *r.late_tokens_per_s > 0.0) {
        fast = std::max(fast, *r.late_tokens_per_s);
        slow = std::min(slow, *r.late_tokens_per_s);
    }
    const double rem = static_cast<double>(remaining);

    if (r.falling) {
        // Forbidden by Invariant 6: a point ETA from the current rate on a run
        // whose rate is falling is an optimistic figure. A range would not fix
        // it either -- its low end would come from the pre-decline rate, which is
        // an optimistic number wearing a bound's clothes. So: the slowest recent
        // rate becomes a LOWER bound, and there is no upper bound at all while
        // the trend continues, because nothing observed bounds it.
        r.eta_low_seconds = rem / slow;
        r.eta_high_seconds = std::nullopt;
        r.eta_is_range = true;
        r.eta_upper_open = true;
        r.note = "rate falling (" + sprintf_str("%.1f%%", *r.drift_fraction * 100.0) + " over " +
                 format_duration(ns_to_s(r.drift_span_ns)) +
                 "): point ETA withheld; this is a lower bound, and while the rate keeps "
                 "falling there is no upper one";
    } else if (r.rising) {
        // A pessimistic point estimate is allowed; an optimistic one is not, so
        // the point comes from the slower (earlier) rate.
        r.eta_low_seconds = rem / fast;
        r.eta_high_seconds = rem / slow;
        r.eta_is_range = true;
        r.eta_seconds = r.eta_high_seconds;
        r.note = "rate rising: point ETA uses the slower earlier rate";
    } else {
        r.eta_low_seconds = rem / fast;
        r.eta_high_seconds = rem / slow;
        r.eta_seconds = rem / *r.tokens_per_s;
        if (r.whole_run_fallback) {
            r.note = "window too thin for a recent rate; used the whole phase";
        }
    }
    return r;
}

Snapshot Readout::snapshot() const {
    Snapshot s;
    s.counters = cur_;
    s.now_ns = now_ns_;
    s.run_start_ns = run_start_ns_;
    s.wall_elapsed_ns = started_ ? (now_ns_ - run_start_ns_) : 0;
    s.phase_elapsed_ns = started_ ? (now_ns_ - phase_start_ns_) : 0;

    const int64_t phase_tokens = cur_.tokens_done - phase_start_tokens_;
    if (phase_tokens > 0 && s.phase_elapsed_ns > 0) {
        s.seconds_per_token = ns_to_s(s.phase_elapsed_ns) / static_cast<double>(phase_tokens);
    }

    s.rate = compute_rate();

    if (cur_.routed_bytes_demanded > 0) {
        s.h_routed = static_cast<double>(cur_.routed_bytes_hit) /
                     static_cast<double>(cur_.routed_bytes_demanded);
    }
    if (cur_.active_bytes_demanded > 0) {
        s.h_bytes = static_cast<double>(cur_.active_bytes_hit) /
                    static_cast<double>(cur_.active_bytes_demanded);
    }

    // Sustained throughput over the same window the rate was taken from, so the
    // two lines of the readout describe the same stretch of time.
    if (s.rate.window_ns > 0) {
        if (const Sample* a = anchor_at(now_ns_ - s.rate.window_ns)) {
            const int64_t span = now_ns_ - a->ns;
            s.bandwidth_window_ns = span;
            if (span > 0) {
                for (size_t i = 0; i < kNumClasses; ++i) {
                    if (!cur_.bytes_fed[i]) continue;   // unfed stays unknown (S37)
                    const int64_t delta = cur_.bytes_read[i] - a->bytes[i];
                    if (delta >= 0) {
                        s.sustained_bytes_per_s[i] = static_cast<double>(delta) / ns_to_s(span);
                    }
                }
            }
        }
    }
    if (s.wall_elapsed_ns > 0) {
        for (size_t i = 0; i < kNumClasses; ++i) {
            if (!cur_.bytes_fed[i]) continue;           // unfed stays unknown (S37)
            s.run_bytes_per_s[i] = static_cast<double>(cur_.bytes_read[i]) / ns_to_s(s.wall_elapsed_ns);
        }
    }

    // A class that has never read a byte has no fraction: printing 0% would read
    // as "the drive is idle" when in truth nothing was ever asked of it.
    for (size_t i = 0; i < kNumClasses; ++i) {
        const std::optional<double> ceiling = ceil_.bytes_per_s[i];
        if (usable(ceiling) && *ceiling > 0.0 && cur_.bytes_read[i] > 0 &&
            usable(s.sustained_bytes_per_s[i])) {
            s.ceiling_fraction[i] = *s.sustained_bytes_per_s[i] / *ceiling;
        }
    }

    double best = -1.0;
    for (size_t i = 0; i < kNumClasses; ++i) {
        if (usable(s.ceiling_fraction[i])) {
            best = std::max(best, *s.ceiling_fraction[i]);
        }
    }
    if (best < 0.0) {
        s.limiter = Limiter::Unknown;
    } else if (best >= opt_.drive_limited_fraction) {
        s.limiter = Limiter::Drive;
    } else if (best <= opt_.engine_limited_fraction) {
        s.limiter = Limiter::Engine;
    } else {
        s.limiter = Limiter::Mixed;
    }
    return s;
}

std::string Readout::terminal_line() const {
    const Snapshot s = snapshot();

    // Highest value first: the line is a viewport and gets truncated from the
    // right on a narrow terminal. Nothing is lost -- status_block() and the JSONL
    // record always carry every field.
    std::vector<std::string> segs;
    segs.push_back(opt_.model.empty() ? std::string("dray") : opt_.model);
    segs.push_back(phase_name(cur_.phase));
    segs.push_back("tok " + grouped(cur_.tokens_done) + "/" +
                   (cur_.tokens_requested.has_value() ? grouped(*cur_.tokens_requested)
                                                      : std::string(kUnknown)));
    segs.push_back(format_token_rate(s.rate.tokens_per_s));
    segs.push_back("eta " + eta_display(s.rate) + " " + window_display(s.rate));
    if (s.rate.falling || s.rate.rising) {
        segs.push_back("[" + drift_display(s.rate) + "]");
    }
    segs.push_back("up " + format_duration(ns_to_s(s.wall_elapsed_ns)));
    segs.push_back(limiter_name(s.limiter));
    for (size_t i = 0; i < kNumClasses; ++i) {
        const ByteClass c = static_cast<ByteClass>(i);
        segs.push_back(std::string(byte_class_name(c)) + " " +
                       format_bandwidth(s.sustained_bytes_per_s[i]) + " (" +
                       format_percent(s.ceiling_fraction[i]) + " of ceiling)");
    }
    segs.push_back("hit r" + format_percent(s.h_routed) + " b" + format_percent(s.h_bytes));
    segs.push_back("res " + bytes_or_unknown(cur_.resident_bytes) + "/" +
                   bytes_or_unknown(cur_.resident_cap));
    segs.push_back("ctx " + format_count(cur_.context_len) + "/" + format_count(cur_.context_cap));
    if (cur_.read_errors.has_value() && *cur_.read_errors > 0) {
        segs.push_back("ERRORS " + grouped(*cur_.read_errors));
    }

    std::string full;
    for (const std::string& seg : segs) {
        if (!full.empty()) {
            full += "  ";
        }
        full += seg;
    }
    const size_t width = opt_.terminal_width > 0 ? static_cast<size_t>(opt_.terminal_width) : 0;
    if (width == 0 || full.size() <= width) {
        return full;
    }

    // Rebuild within the width, reserving room for the "..." that says something
    // was cut. Silently dropping fields would be exactly the dishonesty this
    // component exists to prevent.
    const std::string tail = " ...";
    std::string out;
    for (const std::string& seg : segs) {
        const std::string cand = out.empty() ? seg : out + "  " + seg;
        if (cand.size() + tail.size() > width) {
            break;
        }
        out = cand;
    }
    if (out.empty()) {
        out = full.substr(0, width > tail.size() ? width - tail.size() : width);
    }
    return out + tail;
}

std::string Readout::status_block() const {
    const Snapshot s = snapshot();
    std::ostringstream o;

    o << "dray"
      << "  model=" << (opt_.model.empty() ? std::string(kUnknown) : opt_.model)
      << "  run=" << (opt_.run_id.empty() ? std::string(kUnknown) : opt_.run_id)
      << "  phase=" << phase_name(cur_.phase) << "\n";

    o << sprintf_str("  %-16s %-24s", "tokens",
                     (grouped(cur_.tokens_done) + " / " +
                      (cur_.tokens_requested.has_value() ? grouped(*cur_.tokens_requested)
                                                         : std::string(kUnknown))).c_str())
      << sprintf_str("%-12s %s", "context",
                     (format_count(cur_.context_len) + " / " + format_count(cur_.context_cap)).c_str())
      << "\n";

    o << sprintf_str("  %-16s %-24s", "wall elapsed", format_duration(ns_to_s(s.wall_elapsed_ns)).c_str())
      << sprintf_str("%-12s %s", "this phase", format_duration(ns_to_s(s.phase_elapsed_ns)).c_str())
      << "\n";

    o << sprintf_str("  %-16s %-24s", "rate (window)", format_token_rate(s.rate.tokens_per_s).c_str())
      << sprintf_str("%-12s %s", "phase mean",
                     format_token_rate(usable(s.seconds_per_token) && *s.seconds_per_token > 0.0
                                           ? std::optional<double>(1.0 / *s.seconds_per_token)
                                           : std::nullopt).c_str())
      << "\n";

    o << sprintf_str("  %-16s %-24s", "eta", eta_display(s.rate).c_str())
      << sprintf_str("%-12s %s", "from",
                     (s.rate.window_label.empty() ? std::string(kUnknown)
                                                  : s.rate.window_label +
                                                        (s.rate.whole_run_fallback ? " (whole phase)" : ""))
                         .c_str())
      << "\n";

    const std::string drift = drift_display(s.rate);
    o << sprintf_str("  %-16s %s\n", "rate movement",
                     (drift.empty() ? std::string(kUnknown) + " (not enough history yet)" : drift).c_str());

    o << sprintf_str("  %-16s h_routed %s   h_bytes %s\n", "cache hit",
                     format_percent(s.h_routed).c_str(), format_percent(s.h_bytes).c_str());

    for (size_t i = 0; i < kNumClasses; ++i) {
        o << sprintf_str("  %-16s %-12s of ceiling %-12s = %s\n", byte_class_name(static_cast<ByteClass>(i)),
                         format_bandwidth(s.sustained_bytes_per_s[i]).c_str(),
                         format_bandwidth(ceil_.bytes_per_s[i]).c_str(),
                         format_percent(s.ceiling_fraction[i]).c_str());
    }
    o << sprintf_str("  %-16s %s\n", "verdict", limiter_name(s.limiter));
    if (!ceil_.provenance.empty()) {
        o << sprintf_str("  %-16s %s\n", "ceiling from", ceil_.provenance.c_str());
    }

    o << sprintf_str("  %-16s %s / %s   rss %s\n", "resident",
                     bytes_or_unknown(cur_.resident_bytes).c_str(),
                     bytes_or_unknown(cur_.resident_cap).c_str(),
                     bytes_or_unknown(cur_.rss_bytes).c_str());

    o << sprintf_str("  %-16s stream %s   gather %s\n", "bytes read",
                     format_bytes(cur_.bytes_of(ByteClass::Stream)).c_str(),
                     format_bytes(cur_.bytes_of(ByteClass::Gather)).c_str());

    o << sprintf_str("  %-16s issued %s   short %s   errors %s\n", "reads",
                     format_count(cur_.reads_issued).c_str(),
                     format_count(cur_.short_reads).c_str(),
                     format_count(cur_.read_errors).c_str());

    if (!s.rate.note.empty()) {
        o << sprintf_str("  %-16s %s\n", "note", s.rate.note.c_str());
    }
    if (!jsonl_error_.empty()) {
        o << sprintf_str("  %-16s %s\n", "status log", jsonl_error_.c_str());
    }
    return o.str();
}

std::string Readout::final_summary() const {
    const Snapshot s = snapshot();
    std::ostringstream o;
    o << "run finished\n";
    o << sprintf_str("  %-16s %s\n", "tokens", grouped(cur_.tokens_done).c_str());
    o << sprintf_str("  %-16s %s\n", "wall elapsed", format_duration(ns_to_s(s.wall_elapsed_ns)).c_str());

    std::optional<double> mean_rate;
    if (s.wall_elapsed_ns > 0 && cur_.tokens_done > 0) {
        mean_rate = static_cast<double>(cur_.tokens_done) / ns_to_s(s.wall_elapsed_ns);
    }
    o << sprintf_str("  %-16s %s (whole run, including prefill)\n", "mean rate",
                     format_token_rate(mean_rate).c_str());

    o << sprintf_str("  %-16s h_routed %s   h_bytes %s\n", "cache hit",
                     format_percent(s.h_routed).c_str(), format_percent(s.h_bytes).c_str());

    for (size_t i = 0; i < kNumClasses; ++i) {
        std::optional<double> frac;
        const std::optional<double> ceiling = ceil_.bytes_per_s[i];
        if (usable(ceiling) && *ceiling > 0.0 && cur_.bytes_read[i] > 0 && usable(s.run_bytes_per_s[i])) {
            frac = *s.run_bytes_per_s[i] / *ceiling;
        }
        o << sprintf_str("  %-16s %s over the run, %s of ceiling %s; total %s\n",
                         byte_class_name(static_cast<ByteClass>(i)),
                         format_bandwidth(s.run_bytes_per_s[i]).c_str(),
                         format_percent(frac).c_str(),
                         format_bandwidth(ceiling).c_str(),
                         format_bytes(cur_.bytes_read[i]).c_str());
    }
    o << sprintf_str("  %-16s issued %s   short %s   errors %s\n", "reads",
                     format_count(cur_.reads_issued).c_str(),
                     format_count(cur_.short_reads).c_str(),
                     format_count(cur_.read_errors).c_str());
    if (!backend_.empty()) {
        o << sprintf_str("  %-16s %s\n", "backend", backend_.c_str());
    }
    if (!plan_summary_.empty()) {
        o << sprintf_str("  %-16s %s\n", "plan", plan_summary_.c_str());
    }
    if (!opt_.jsonl_path.empty()) {
        o << sprintf_str("  %-16s %s\n", "status log", opt_.jsonl_path.c_str());
    }
    return o.str();
}

void Readout::render_terminal(std::ostream& out) {
    const std::string line = terminal_line();
    out << '\r' << line;
    if (last_line_len_ > line.size()) {
        out << std::string(last_line_len_ - line.size(), ' ');
    }
    out.flush();
    last_line_len_ = line.size();
    last_terminal_ns_ = now_ns_;
}

void Readout::end_terminal(std::ostream& out) {
    out << '\n';
    out.flush();
    last_line_len_ = 0;
}

std::string Readout::jsonl_record() const {
    const Snapshot s = snapshot();
    std::ostringstream o;

    const auto secs = [](int64_t ns) { return std::optional<double>(ns_to_s(ns)); };

    o << "{";
    o << "\"ts\":" << jstr(utc_timestamp());
    o << ",\"mono_ns\":" << std::to_string(s.now_ns);
    o << ",\"run\":" << (opt_.run_id.empty() ? std::string("null") : jstr(opt_.run_id));
    o << ",\"model\":" << (opt_.model.empty() ? std::string("null") : jstr(opt_.model));
    o << ",\"phase\":" << jstr(phase_name(cur_.phase));
    o << ",\"tokens_done\":" << std::to_string(cur_.tokens_done);
    o << ",\"tokens_requested\":" << jint(cur_.tokens_requested);
    o << ",\"elapsed_s\":" << jnum(secs(s.wall_elapsed_ns));
    o << ",\"phase_elapsed_s\":" << jnum(secs(s.phase_elapsed_ns));
    o << ",\"s_per_tok_phase_mean\":" << jnum(s.seconds_per_token);

    o << ",\"rate\":{"
      << "\"tok_per_s\":" << jnum(s.rate.tokens_per_s)
      << ",\"s_per_tok\":"
      << jnum(usable(s.rate.tokens_per_s) && *s.rate.tokens_per_s > 0.0
                  ? std::optional<double>(1.0 / *s.rate.tokens_per_s)
                  : std::nullopt)
      << ",\"window_s\":" << jnum(secs(s.rate.window_ns))
      << ",\"window_tokens\":" << std::to_string(s.rate.window_tokens)
      << ",\"whole_run_fallback\":" << jbool(s.rate.whole_run_fallback)
      << ",\"drift_fraction\":" << jnum(s.rate.drift_fraction)
      << ",\"drift_span_s\":"
      << jnum(s.rate.drift_span_ns > 0 ? secs(s.rate.drift_span_ns) : std::nullopt)
      << ",\"early_tok_per_s\":" << jnum(s.rate.early_tokens_per_s)
      << ",\"late_tok_per_s\":" << jnum(s.rate.late_tokens_per_s)
      << ",\"falling\":" << jbool(s.rate.falling)
      << ",\"rising\":" << jbool(s.rate.rising)
      << "}";

    o << ",\"eta\":{"
      << "\"seconds\":" << jnum(s.rate.eta_seconds)
      << ",\"low_seconds\":" << jnum(s.rate.eta_low_seconds)
      << ",\"high_seconds\":" << jnum(s.rate.eta_high_seconds)
      << ",\"is_range\":" << jbool(s.rate.eta_is_range)
      << ",\"upper_open\":" << jbool(s.rate.eta_upper_open)
      << ",\"note\":" << (s.rate.note.empty() ? std::string("null") : jstr(s.rate.note))
      << "}";

    o << ",\"hit\":{"
      << "\"h_routed\":" << jnum(s.h_routed)
      << ",\"h_bytes\":" << jnum(s.h_bytes)
      << ",\"routed_bytes_hit\":" << std::to_string(cur_.routed_bytes_hit)
      << ",\"routed_bytes_demanded\":" << std::to_string(cur_.routed_bytes_demanded)
      << ",\"active_bytes_hit\":" << std::to_string(cur_.active_bytes_hit)
      << ",\"active_bytes_demanded\":" << std::to_string(cur_.active_bytes_demanded)
      << "}";

    o << ",\"bw\":{";
    for (size_t i = 0; i < kNumClasses; ++i) {
        if (i != 0) {
            o << ",";
        }
        o << jstr(byte_class_name(static_cast<ByteClass>(i))) << ":{"
          << "\"bytes\":" << std::to_string(cur_.bytes_read[i])
          << ",\"sustained_bytes_per_s\":" << jnum(s.sustained_bytes_per_s[i])
          << ",\"run_bytes_per_s\":" << jnum(s.run_bytes_per_s[i])
          << ",\"ceiling_bytes_per_s\":" << jnum(ceil_.bytes_per_s[i])
          << ",\"ceiling_fraction\":" << jnum(s.ceiling_fraction[i])
          << "}";
    }
    o << "}";
    o << ",\"bw_window_s\":" << jnum(secs(s.bandwidth_window_ns));
    o << ",\"ceiling_provenance\":"
      << (ceil_.provenance.empty() ? std::string("null") : jstr(ceil_.provenance));
    o << ",\"limiter\":" << jstr(limiter_name(s.limiter));

    o << ",\"mem\":{"
      << "\"resident_bytes\":" << jint(cur_.resident_bytes)
      << ",\"cap_bytes\":" << jint(cur_.resident_cap)
      << ",\"rss_bytes\":" << jint(cur_.rss_bytes)
      << "}";

    o << ",\"ctx\":{"
      << "\"len\":" << jint(cur_.context_len)
      << ",\"cap\":" << jint(cur_.context_cap)
      << "}";

    o << ",\"io\":{"
      << "\"reads_issued\":" << jint(cur_.reads_issued)
      << ",\"short_reads\":" << jint(cur_.short_reads)
      << ",\"errors\":" << jint(cur_.read_errors)
      << "}";

    o << "}";
    return o.str();
}

bool Readout::ensure_jsonl_open() {
    if (opt_.jsonl_path.empty()) {
        return false;
    }
    if (jsonl_.is_open()) {
        return true;
    }
    namespace fs = std::filesystem;
    const fs::path p(opt_.jsonl_path);
    std::error_code ec;
    if (p.has_parent_path() && !p.parent_path().empty()) {
        fs::create_directories(p.parent_path(), ec);
        ec.clear();
    }
    const std::uintmax_t existing = fs::file_size(p, ec);
    jsonl_bytes_ = ec ? 0 : static_cast<int64_t>(existing);

    jsonl_.open(p, std::ios::out | std::ios::app | std::ios::binary);
    if (!jsonl_.is_open()) {
        jsonl_error_ = "cannot open status log: " + opt_.jsonl_path;
        return false;
    }
    jsonl_error_.clear();
    return true;
}

void Readout::rotate_jsonl() {
    namespace fs = std::filesystem;
    if (jsonl_.is_open()) {
        jsonl_.flush();
        jsonl_.close();
    }
    std::error_code ec;
    const fs::path base(opt_.jsonl_path);
    if (opt_.jsonl_rotations <= 0) {
        fs::remove(base, ec);
    } else {
        // Shift .N-1 -> .N and drop the oldest, so the whole log is bounded at
        // (rotations + 1) * jsonl_max_bytes no matter how many days the run lasts.
        for (int32_t i = opt_.jsonl_rotations; i >= 1; --i) {
            const fs::path dst(opt_.jsonl_path + "." + std::to_string(i));
            const fs::path src = (i == 1) ? base : fs::path(opt_.jsonl_path + "." + std::to_string(i - 1));
            fs::remove(dst, ec);
            ec.clear();
            if (fs::exists(src, ec)) {
                fs::rename(src, dst, ec);
                ec.clear();
            }
        }
    }
    jsonl_bytes_ = 0;
}

bool Readout::write_jsonl() {
    if (opt_.jsonl_path.empty()) {
        return false;
    }
    const std::string rec = jsonl_record();
    if (!ensure_jsonl_open()) {
        return false;
    }
    const int64_t need = static_cast<int64_t>(rec.size()) + 1;
    if (opt_.jsonl_max_bytes > 0 && jsonl_bytes_ + need > opt_.jsonl_max_bytes) {
        rotate_jsonl();
        if (!ensure_jsonl_open()) {
            return false;
        }
    }
    jsonl_ << rec << '\n';
    // Flushed per record: a detached multi-day run is inspected by tailing this
    // file while it runs, and a kill -9 must lose at most the record in flight.
    jsonl_.flush();
    if (!jsonl_.good()) {
        jsonl_error_ = "status log write failed: " + opt_.jsonl_path;
        return false;
    }
    jsonl_bytes_ += need;
    last_jsonl_ns_ = now_ns_;
    return true;
}

void Readout::tick(const Counters& c, int64_t now_ns, std::ostream* term) {
    update(c, now_ns);
    const bool last_frame = (cur_.phase == Phase::Done);

    if (term != nullptr &&
        (last_terminal_ns_ == 0 || last_frame ||
         now_ns_ - last_terminal_ns_ >= opt_.terminal_min_interval_ns)) {
        render_terminal(*term);
    }
    if (!opt_.jsonl_path.empty() &&
        (last_jsonl_ns_ == 0 || last_frame ||
         now_ns_ - last_jsonl_ns_ >= opt_.jsonl_min_interval_ns)) {
        write_jsonl();
    }
}

// ---------------------------------------------------------------------------
// RAM curve
// ---------------------------------------------------------------------------

std::string ram_curve_table(const std::vector<plan::Plan::CurvePoint>& curve,
                            uint64_t current_cap,
                            std::optional<double> bytes_per_s,
                            const std::string& bandwidth_label) {
    std::ostringstream o;
    o << "RAM curve -- PROJECTIONS, not measurements.\n";
    o << "  Byte figures are exact plan arithmetic. The s/tok column additionally assumes\n";
    o << "  " << (usable(bytes_per_s) ? format_bandwidth(bytes_per_s) : std::string("an unknown bandwidth"))
      << (bandwidth_label.empty() ? std::string() : " (" + bandwidth_label + ")")
      << " and no compute overlap, so it is a floor on time, not a promise.\n";
    o << "  Policy results do not generalize across models: this curve is for this one.\n\n";

    if (curve.empty()) {
        o << "  (no curve available)\n";
        return o.str();
    }

    // Baseline for the "vs current" column: the exact current cap if it is on
    // the curve, otherwise the largest point at or below it.
    const plan::Plan::CurvePoint* base = nullptr;
    for (const plan::Plan::CurvePoint& p : curve) {
        if (p.cap == current_cap) {
            base = &p;
            break;
        }
        if (p.cap <= current_cap && (base == nullptr || p.cap > base->cap)) {
            base = &p;
        }
    }

    o << sprintf_str("  %9s  %13s  %13s  %11s\n", "cap", "bytes/token", "proj s/tok", "vs current");
    o << sprintf_str("  %9s  %13s  %13s  %11s\n", "---------", "-------------", "-------------",
                     "-----------");

    for (const plan::Plan::CurvePoint& p : curve) {
        std::string proj;
        if (p.bytes_per_token == 0) {
            proj = "no disk";
        } else if (usable(bytes_per_s) && *bytes_per_s > 0.0) {
            const double tok_per_s = *bytes_per_s / static_cast<double>(p.bytes_per_token);
            proj = format_token_rate(tok_per_s);
        } else {
            proj = kUnknown;
        }

        std::string vs;
        if (base == nullptr) {
            vs = kUnknown;
        } else if (p.cap == base->cap) {
            vs = "(current)";
        } else if (base->bytes_per_token == 0) {
            vs = kUnknown;
        } else {
            const double d = (static_cast<double>(p.bytes_per_token) -
                              static_cast<double>(base->bytes_per_token)) /
                             static_cast<double>(base->bytes_per_token);
            vs = sprintf_str("%+.1f%%", d * 100.0);
        }

        o << sprintf_str("  %9s  %13s  %13s  %11s%s\n",
                         format_bytes(static_cast<int64_t>(p.cap)).c_str(),
                         format_bytes(static_cast<int64_t>(p.bytes_per_token)).c_str(),
                         proj.c_str(), vs.c_str(), p.is_knee ? "   <- knee" : "");
    }
    return o.str();
}

}  // namespace dray::report
