#include "sched/prefetch.h"

#include <algorithm>

namespace dray::sched {

std::vector<MergedRead> coalesce(const io::ReadRequest* reqs, size_t n,
                                 const CoalesceOptions& opt) {
    std::vector<MergedRead> out;
    if (!reqs || n == 0) return out;

    // Sort indices by (file, offset). Ascending offset order is free to produce
    // and helps the drive's internal scheduling.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [reqs](size_t a, size_t b) {
        if (reqs[a].file != reqs[b].file) return reqs[a].file < reqs[b].file;
        return reqs[a].offset < reqs[b].offset;
    });

    for (size_t k = 0; k < n; ++k) {
        const io::ReadRequest& r = reqs[order[k]];
        if (r.length == 0) continue;

        if (!out.empty()) {
            MergedRead& cur = out.back();
            const uint64_t cur_end = cur.offset + cur.length;
            const bool same_file = cur.file == r.file;
            // Overlap or a small enough gap, and the result still fits the cap.
            if (same_file && r.offset >= cur.offset && r.offset <= cur_end + opt.max_gap_bytes) {
                const uint64_t new_end = std::max<uint64_t>(cur_end, r.offset + r.length);
                const uint64_t new_len = new_end - cur.offset;
                if (new_len <= opt.max_merge_bytes) {
                    cur.length = static_cast<uint32_t>(new_len);
                    cur.parts.push_back(order[k]);
                    continue;
                }
            }
        }

        MergedRead m;
        m.file = r.file;
        m.offset = r.offset;
        m.length = r.length;
        m.parts.push_back(order[k]);
        out.push_back(std::move(m));
    }

    return out;
}

uint64_t merged_bytes(const std::vector<MergedRead>& v) {
    uint64_t total = 0;
    for (const MergedRead& m : v) total += m.length;
    return total;
}

}  // namespace dray::sched
