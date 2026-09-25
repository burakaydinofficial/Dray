#include "engine/streamed_text.h"

namespace dray::engine {

size_t utf8_complete_prefix(const std::string& s) {
    size_t i = s.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0x80) == 0) return i;                // ASCII tail: complete
        if ((c & 0xC0) == 0xC0) {                     // lead byte at i-1
            const size_t need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3
                              : (c & 0xF8) == 0xF0 ? 4 : 1;
            return (s.size() - (i - 1)) >= need ? s.size() : i - 1;
        }
        --i; ++back;                                  // continuation: keep scanning
    }
    return s.size();  // malformed anyway; the dump-side replace backstop holds
}

StopMatch find_stop(const std::string& text, size_t new_bytes,
                    const std::vector<std::string>& stops) {
    StopMatch best;
    const size_t old_end = text.size() - new_bytes;
    for (const std::string& st : stops) {
        if (st.empty() || text.size() < st.size()) continue;
        // First occurrence that reaches at least the start of the new bytes.
        const size_t from = old_end >= st.size() ? old_end - st.size() : 0;
        const size_t at = text.find(st, from);
        if (at != std::string::npos && at < best.pos) {
            best.pos = at;
            best.len = st.size();
        }
    }
    return best;
}

StreamedText::StreamedText(GenResult* result, const std::vector<std::string>& stops,
                           const std::string& stop_seed, const TokenSink& sink)
    : result_(result), stops_(stops), stop_seed_(stop_seed), sink_(sink) {}

void StreamedText::feed(const char* b, size_t np) {
    GenResult& r = *result_;
    const size_t base = r.text.size();   // delivered + pending, before this piece
    r.text.append(b, np);
    // Match against seed+text, not text alone: see the header. The seed cannot
    // contain a full stop -- the pre-crash session would have trimmed and
    // finished on it.
    const size_t seedn = stop_seed_.size();
    const StopMatch m = seedn == 0 ? find_stop(r.text, np, stops_)
                                   : find_stop(stop_seed_ + r.text, np, stops_);
    if (m.found()) {
        if (m.pos < seedn) {
            // Straddle: the stop begins inside the persisted prefix. Nothing of
            // this session survives, and the caller must drop the overhang from
            // its stored text.
            r.base_trim = seedn - m.pos;
            r.text.clear();
            r.stop_hit = true;
            pending_.clear();
            return;
        }
        const size_t at = m.pos - seedn;
        r.text.erase(at);
        r.stop_hit = true;
        if (sink_) {
            const size_t delivered = base - pending_.size();
            if (at > delivered) sink_(r.text.substr(delivered, at - delivered));
        }
        pending_.clear();
        return;
    }
    if (!sink_) return;
    pending_.append(b, np);
    const size_t ok = utf8_complete_prefix(pending_);
    if (ok) { sink_(pending_.substr(0, ok)); pending_.erase(0, ok); }
}

void StreamedText::flush() {
    if (sink_ && !pending_.empty()) { sink_(pending_); pending_.clear(); }
}

}  // namespace dray::engine
