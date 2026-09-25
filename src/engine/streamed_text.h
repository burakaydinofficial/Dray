// Assembles one generation's text from detokenized pieces: stop-sequence
// matching and UTF-8-safe delivery to a streaming consumer.
//
// Two jobs, one owner, because they interact at every piece:
//
//   * STOP SEQUENCES (T10/F8) match against the growing text. On a hit the text
//     is trimmed at the match and the not-yet-delivered span up to the trim
//     point is emitted, so SSE and the job registry carry the same bytes the
//     result keeps. A resumed job also passes the TAIL of its earlier text as
//     stop_seed: a stop whose prefix was generated before a crash and whose
//     suffix arrives after resume lives across that boundary. A match that
//     begins inside the seed reports the overhang in GenResult::base_trim.
//
//   * UTF-8 (swarm S2): a llama token can be a bare byte, so a codepoint can
//     straddle two pieces. Incomplete tails are held back until they complete;
//     otherwise a JSON dump of the fragment throws through the SSE and worker
//     threads.
//
// A partial stop that straddles earlier pieces can still leak its prefix to the
// wire; full holdback is a post-tag item, noted in DECISIONS.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "engine/engine_types.h"

namespace dray::engine {

using TokenSink = std::function<void(const std::string&)>;

// Where to cut `text` for a stop sequence, if its newest `new_bytes` completed
// one. THE stop rule, shared by the single-stream, batch and rotation paths:
//
//   * the EARLIEST match wins, across every stop -- OpenAI's semantics. With
//     stops ["Paris", "\n"] and a piece "\nParis", the cut is at the "\n";
//   * within one stop, the FIRST occurrence -- with "\n\n" and a piece
//     "\n\n\n", the cut is before the first newline, not the second.
//
// Only matches that end inside the new bytes are candidates: anything earlier
// would already have stopped generation on an earlier piece. Three copies of
// this loop once disagreed on both points (two searched with rfind and left a
// stray "\n"; all three took the first-LISTED stop that matched).
struct StopMatch {
    size_t pos = std::string::npos;   // npos = no stop completed
    size_t len = 0;
    bool   found() const { return pos != std::string::npos; }
};
StopMatch find_stop(const std::string& text, size_t new_bytes,
                    const std::vector<std::string>& stops);

class StreamedText {
public:
    // Writes text, stop_hit and base_trim into *result. sink may be empty
    // (non-streaming callers).
    StreamedText(GenResult* result, const std::vector<std::string>& stops,
                 const std::string& stop_seed, const TokenSink& sink);

    // Appends one detokenized piece and delivers whatever is complete.
    void feed(const char* bytes, size_t n);

    // Delivers any held-back bytes now (before a checkpoint reads the text, or
    // at the end of a generation). No-op without a sink.
    void flush();

private:
    GenResult* result_;
    const std::vector<std::string>& stops_;
    const std::string& stop_seed_;
    const TokenSink& sink_;
    std::string pending_;   // bytes held back until their codepoint completes
};

// Length of the longest prefix of s that ends on a UTF-8 codepoint boundary.
size_t utf8_complete_prefix(const std::string& s);

}  // namespace dray::engine
