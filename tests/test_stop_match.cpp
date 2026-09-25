// Stop sequences: where a generation is cut, on every path.
//
// find_stop() is the one rule the single-stream, batch and rotation paths
// share; these cases pin its two semantics (earliest match across all stops,
// first occurrence within one) and StreamedText's delivery around it: what a
// streaming client receives must be exactly the text the result keeps, and a
// stop that began before a crash (in the resume seed) must be reported as
// overhang to trim, not silently kept.

#include "harness.h"

#include <string>
#include <vector>

#include "engine/streamed_text.h"

using dray::engine::find_stop;
using dray::engine::GenResult;
using dray::engine::StopMatch;
using dray::engine::StreamedText;
using dray::engine::TokenSink;

namespace {

// Feeds `pieces` through a StreamedText and returns the result, with
// everything the sink received concatenated into *streamed.
GenResult run(const std::vector<std::string>& pieces, const std::vector<std::string>& stops,
              const std::string& seed = "", std::string* streamed = nullptr) {
    GenResult r;
    std::string got;
    const TokenSink sink = [&got](const std::string& s) { got += s; };
    StreamedText st(&r, stops, seed, sink);
    for (const std::string& p : pieces) {
        st.feed(p.data(), p.size());
        if (r.stop_hit) break;
    }
    st.flush();
    if (streamed) *streamed = got;
    return r;
}

}  // namespace

LZ_TEST(no_stop_no_match) {
    const StopMatch m = find_stop("hello world", 5, {"Paris"});
    LZ_CHECK(!m.found());
    LZ_CHECK(!find_stop("hello", 5, {}).found());
    LZ_CHECK(!find_stop("hello", 5, {""}).found());   // an empty stop never matches
}

LZ_TEST(first_occurrence_within_one_stop) {
    // "\n\n" completed by a piece "\n\n\n": cut before the FIRST newline. The
    // old rfind cut before the second and left a stray "\n".
    const StopMatch m = find_stop("abc\n\n\n", 3, {"\n\n"});
    LZ_REQUIRE(m.found());
    LZ_CHECK_EQ(m.pos, 3u);
    LZ_CHECK_EQ(m.len, 2u);
}

LZ_TEST(earliest_match_across_stops) {
    // The first-LISTED stop that matched used to win; the earliest must.
    const StopMatch m = find_stop("It is \nParis", 6, {"Paris", "\n"});
    LZ_REQUIRE(m.found());
    LZ_CHECK_EQ(m.pos, 6u);
    LZ_CHECK_EQ(m.len, 1u);
}

LZ_TEST(match_straddling_earlier_pieces) {
    // "Hel" arrived earlier; "lo world" completes "Hello".
    const StopMatch m = find_stop("Hello world", 8, {"Hello"});
    LZ_REQUIRE(m.found());
    LZ_CHECK_EQ(m.pos, 0u);
}

LZ_TEST(streamed_text_trims_and_streams_the_same_bytes) {
    std::string streamed;
    const GenResult r = run({"The capital", " is Paris.", "\n\n\nNext"}, {"\n\n"}, "", &streamed);
    LZ_CHECK(r.stop_hit);
    LZ_CHECK_EQ(r.text, std::string("The capital is Paris."));
    LZ_CHECK_EQ(streamed, r.text);   // the wire carries exactly what is kept
}

LZ_TEST(streamed_text_earliest_stop_wins) {
    std::string streamed;
    const GenResult r = run({"Answer:", " \nParis"}, {"Paris", "\n"}, "", &streamed);
    LZ_CHECK(r.stop_hit);
    LZ_CHECK_EQ(r.text, std::string("Answer: "));
    LZ_CHECK_EQ(streamed, r.text);
}

LZ_TEST(streamed_text_resume_seed_straddle_reports_overhang) {
    // Before a crash the session produced "...The end"; after resume the next
    // piece is "!!". With stop "end!!", the match begins 3 bytes inside the
    // persisted prefix: nothing of this session survives, and the caller must
    // drop 3 bytes from its stored text.
    const GenResult r = run({"!!"}, {"end!!"}, "The end");
    LZ_CHECK(r.stop_hit);
    LZ_CHECK_EQ(r.text, std::string(""));
    LZ_CHECK_EQ(r.base_trim, 3u);
}

LZ_TEST(streamed_text_no_stop_streams_everything) {
    std::string streamed;
    const GenResult r = run({"a", "b", "c"}, {"zzz"}, "", &streamed);
    LZ_CHECK(!r.stop_hit);
    LZ_CHECK_EQ(r.text, std::string("abc"));
    LZ_CHECK_EQ(streamed, std::string("abc"));
}
