// The readout's honesty contract, unit-tested with an injected clock
// (swarm S37 residual): unknown is never rendered as zero, fed classes report
// real rates including an honest zero, and the snapshot's byte totals track
// exactly what was added.

#include "harness.h"

#include "report/readout.h"

using dray::report::ByteClass;
using dray::report::Counters;
using dray::report::Phase;
using dray::report::Readout;
using dray::report::Snapshot;

namespace {

constexpr int64_t kNs = 1000000000ll;  // 1 s

Readout make_ro() {
    Readout::Options o;
    o.model = "synthetic";
    return Readout(o);
}

}  // namespace

LZ_TEST(unfed_classes_render_unknown_not_zero) {
    Readout ro = make_ro();
    Counters c;
    c.tokens_requested = 8;
    c.phase = Phase::Decode;
    // Feed several updates with time passing and NO byte feeds at all.
    for (int t = 1; t <= 4; ++t) {
        ++c.tokens_done;
        ro.update(c, t * kNs);
    }
    const Snapshot s = ro.snapshot();
    for (size_t i = 0; i < dray::report::kNumClasses; ++i) {
        LZ_CHECK(!s.sustained_bytes_per_s[i].has_value());
        LZ_CHECK(!s.run_bytes_per_s[i].has_value());
    }
}

LZ_TEST(fed_zero_is_an_honest_zero_and_fed_bytes_are_real_rates) {
    Readout ro = make_ro();
    Counters c;
    c.tokens_requested = 8;
    c.phase = Phase::Decode;

    // Gather is fed real bytes; Stream is fed an explicit zero every token
    // (everything from RAM) -- the two must render differently from unfed.
    for (int t = 1; t <= 4; ++t) {
        ++c.tokens_done;
        c.add_bytes(ByteClass::Gather, 1000);
        c.add_bytes(ByteClass::Stream, 0);
        ro.update(c, t * kNs);
    }
    const Snapshot s = ro.snapshot();
    const auto g = s.run_bytes_per_s[static_cast<size_t>(ByteClass::Gather)];
    const auto st = s.run_bytes_per_s[static_cast<size_t>(ByteClass::Stream)];
    LZ_CHECK(g.has_value());
    LZ_CHECK(st.has_value());
    LZ_CHECK_GT(*g, 0.0);
    LZ_CHECK_EQ(*st, 0.0);   // measured zero, not fabricated zero
    LZ_CHECK_EQ(c.bytes_of(ByteClass::Gather), 4000);
}

// T26: the falling-rate contract, the sole documented ETA-honesty rule with
// no prior enforcement: on a falling rate the point ETA and the upper bound
// are WITHHELD (eta_upper_open), never extrapolated. A steady run first, as
// the control; readout.cpp's drift gate needs a real window, hence 120 ticks.
LZ_TEST(falling_rate_withholds_point_eta_and_upper_bound) {
    Readout ro = make_ro();
    Counters c;
    c.tokens_requested = 4000;
    c.phase = Phase::Decode;

    // Control: steady 1 tok/s for 120 s. A point-or-range ETA must exist and
    // the upper end must be bounded.
    int64_t now = 0;
    for (int t = 1; t <= 120; ++t) {
        ++c.tokens_done;
        now = t * kNs;
        ro.update(c, now);
    }
    {
        const Snapshot s = ro.snapshot();
        LZ_CHECK(s.rate.eta_seconds.has_value() || s.rate.eta_low_seconds.has_value());
        LZ_CHECK(!s.rate.eta_upper_open);
        LZ_CHECK(!s.rate.window_label.empty());   // the window is always disclosed
    }

    // The rate collapses 10x: same token count, ten seconds a token. The point
    // ETA and the upper bound must vanish; the lower bound may remain.
    for (int t = 0; t < 60; ++t) {
        ++c.tokens_done;
        now += 10 * kNs;
        ro.update(c, now);
    }
    {
        const Snapshot s = ro.snapshot();
        LZ_CHECK(s.rate.eta_upper_open);
        LZ_CHECK(!s.rate.eta_seconds.has_value());
        LZ_CHECK(!s.rate.eta_high_seconds.has_value());
        LZ_CHECK(s.rate.eta_low_seconds.has_value());
    }
}
