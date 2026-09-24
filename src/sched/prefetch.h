// Read coalescing and ordering for the expert gather.
//
// Decode traffic is a large-block GATHER, not a sequential scan: a layer's experts
// live in three fused tensors and one token touches k*3 non-adjacent slices per MoE
// layer. Two things make that cheaper without changing what is read:
//
//   * MERGE adjacent or near-adjacent slices into one request. Router-selected
//     experts are frequently consecutive ids, and consecutive ids ARE contiguous
//     within a fused tensor, so merging is often possible and turns k reads into
//     far fewer. A small gap is worth reading through rather than splitting, since
//     one extra request costs more than a few KB of wasted transfer.
//
//   * ORDER by file offset. Even on NVMe, issuing a batch in ascending offset
//     order helps the drive's internal scheduling and costs nothing to do.
//
// Merging is bounded by max_merge_bytes so one giant request cannot exceed what the
// backend can service efficiently (blocks above max_hw_sectors_kb trigger io_uring
// async-worker fallback on Linux, reintroducing the overhead we are avoiding).

#pragma once

#include <cstdint>
#include <vector>

#include "io/storage.h"

namespace dray::sched {

struct CoalesceOptions {
    // Read through a gap rather than splitting, when the gap is no larger than
    // this. One saved request is worth a few KB of wasted transfer.
    uint32_t max_gap_bytes = 64u * 1024u;
    // Never build a request larger than this. Calibration supplies the real value.
    uint32_t max_merge_bytes = 2u * 1024u * 1024u;
};

// A merged request, plus the mapping back to the slices that asked for it.
struct MergedRead {
    uint64_t offset = 0;
    uint32_t length = 0;
    io::FileId file = io::kInvalidFile;
    // Indices into the original request array that this covers.
    std::vector<size_t> parts;
};

// Sorts by offset and merges. Input order is preserved through `parts`, so the
// caller can scatter results back to the right destinations.
//
// NOTE: merged reads land in a staging buffer and are then copied out to each
// slice's destination, because the destinations are separate slab slots and are
// not contiguous. The caller decides whether the copy is worth it; for widely
// separated slices it is not, which is what max_gap_bytes controls.
std::vector<MergedRead> coalesce(const io::ReadRequest* reqs, size_t n,
                                 const CoalesceOptions& opt);

// Total bytes that would be transferred for a merged plan, including bytes read
// through gaps. Compare against the sum of the original lengths to decide whether
// merging is paying for itself -- this ratio belongs in the readout, because
// "bytes read per token" is the metric this project is denominated in.
uint64_t merged_bytes(const std::vector<MergedRead>&);

}  // namespace dray::sched
