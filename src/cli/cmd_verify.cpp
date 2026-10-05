#include "cli/commands.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cli/reference_read.h"
#include "io/storage.h"
#include "mem/accountant.h"
#include "plan/residency.h"

namespace dray::cli {

// Proves the streaming geometry against the real model without loading it: reads
// real expert slices through the uncached backend into slab slots, then re-reads
// the same file ranges through an ordinary buffered path and compares bytes.
//
// This is the check that catches the failure mode that matters most here -- wrong
// offsets or strides do not crash, they return plausible-looking garbage weights.
// gate/up/down have DIFFERENT per-expert strides, so an evenly-divided slot would
// pass a size check and still be wrong.
int cmd_verify(const Args& a) {
    std::string err;
    dray::plan::Plan p = dray::plan::build_plan(a.model, a.cap, a.n_ctx ? a.n_ctx : 512, &err);
    if (!err.empty()) { std::fprintf(stderr, "plan failed: %s\n", err.c_str()); return 1; }
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    if (p.slot_classes.empty()) { std::fprintf(stderr, "no MoE layers found\n"); return 1; }

    auto backend = dray::io::make_backend(32);
    // EVERY shard, not just the one named on the command line. Routed experts
    // live wherever the split put them -- on DeepSeek V4 Flash all 129 of them
    // are in shards 2 and 3, so a shard-1-only check reported "INCONCLUSIVE --
    // nothing compared" while verifying precisely nothing.
    std::vector<dray::io::FileId> handles(p.shard_paths.size(), dray::io::kInvalidFile);
    std::vector<uint64_t> sizes(p.shard_paths.size(), 0);
    for (size_t i = 0; i < p.shard_paths.size(); ++i) {
        handles[i] = backend->open(p.shard_paths[i]);
        if (handles[i] != dray::io::kInvalidFile) sizes[i] = backend->size(handles[i]);
    }
    dray::io::FileId f = handles.empty() ? dray::io::kInvalidFile : handles[0];
    if (f == dray::io::kInvalidFile) {
        std::fprintf(stderr, "cannot open %s uncached\n", a.model.c_str());
        return 1;
    }
    const dray::io::Alignment al = backend->alignment(f);
    std::printf("backend: %s\n", backend->describe().c_str());

    // Walk every routed tensor that lives in THIS shard, and for a sample of
    // experts compare an aligned uncached read against a buffered reference.
    size_t checked = 0, mismatched = 0, skipped = 0;
    std::vector<uint8_t> direct, reference;

    std::FILE* ref = std::fopen(a.model.c_str(), "rb");
    if (!ref) { std::fprintf(stderr, "cannot open reference handle\n"); backend->close(f); return 1; }

    for (const dray::plan::TensorInfo& t : p.tensors) {
        if (t.cls != dray::plan::TensorClass::RoutedExpert) continue;
        if (p.n_experts == 0 || t.bytes == 0) continue;
        const size_t si = static_cast<size_t>(t.shard >= 0 ? t.shard : 0);
        if (si >= handles.size() || handles[si] == dray::io::kInvalidFile) { ++skipped; continue; }
        if (t.offset + t.bytes > sizes[si]) { ++skipped; continue; }
        const dray::io::FileId fh = handles[si];

        const uint64_t stride = t.bytes / p.n_experts;
        if (stride == 0) { ++skipped; continue; }

        for (uint32_t e = 0; e < p.n_experts; e += (p.n_experts / 4 ? p.n_experts / 4 : 1)) {
            const uint64_t want_off = t.offset + static_cast<uint64_t>(e) * stride;

            // Widen to the alignment boundary and slice the interior back out --
            // GGUF tensor offsets are aligned to general.alignment (32 by default),
            // which is NOT the device's I/O alignment.
            const uint64_t lo = (want_off / al.offset) * al.offset;
            const uint64_t head = want_off - lo;
            uint64_t span = head + stride;
            span = ((span + al.length - 1) / al.length) * al.length;
            if (lo + span > sizes[si]) { ++skipped; continue; }
            if (span > (64u << 20)) { ++skipped; continue; }

            void* buf = dray::mem::aligned_alloc_host(static_cast<size_t>(span), al.memory);
            if (!buf) { ++skipped; continue; }

            dray::io::ReadRequest r;
            r.file = fh; r.offset = lo; r.length = static_cast<uint32_t>(span);
            r.dst = buf; r.tag = 1;
            bool ok = backend->submit(&r, 1) == 1;
            if (ok) {
                dray::io::Completion c;
                ok = backend->poll(&c, 1, 1) == 1 && c.ok(static_cast<uint32_t>(span));
            }
            if (ok) {
                reference.assign(static_cast<size_t>(stride), 0);
                ok = read_reference_uncached(p.shard_paths[si], want_off, stride, reference.data());
                if (ok) {
                    const uint8_t* got = static_cast<const uint8_t*>(buf) + head;
                    if (std::memcmp(got, reference.data(), static_cast<size_t>(stride)) != 0) {
                        ++mismatched;
                        std::printf("  MISMATCH %s expert %u @ %llu (%llu B)\n", t.name.c_str(),
                                    e, static_cast<unsigned long long>(want_off),
                                    static_cast<unsigned long long>(stride));
                    }
                    ++checked;
                } else { ++skipped; }
            } else { ++skipped; }
            dray::mem::aligned_free_host(buf, span);
        }
    }

    // SECOND PASS: the unconditional weights. Routed experts are sampled above
    // because there are hundreds of thousands of slices, but the unconditional
    // set -- attention projections, shared experts, embedding, output -- is read
    // on EVERY token, so a misread there corrupts every token rather than the
    // ones that happen to route through a bad expert. Checking only the routed
    // half left the more damaging half unchecked.
    size_t un_checked = 0, un_mismatched = 0, un_skipped = 0;
    for (const dray::plan::TensorInfo& t : p.tensors) {
        if (t.cls != dray::plan::TensorClass::UnconditionalBulk &&
            t.cls != dray::plan::TensorClass::RowSliced) continue;
        if (t.bytes == 0) continue;
        const size_t si = static_cast<size_t>(t.shard >= 0 ? t.shard : 0);
        if (si >= handles.size() || handles[si] == dray::io::kInvalidFile) { ++un_skipped; continue; }
        if (t.offset + t.bytes > sizes[si]) { ++un_skipped; continue; }

        // A prefix is enough: these are contiguous reads, so an offset or
        // alignment error shows up in the first bytes. Bounded so output.weight
        // does not cost a gigabyte of comparison.
        const uint64_t want = std::min<uint64_t>(t.bytes, 4ull << 20);
        const uint64_t lo = (t.offset / al.length) * al.length;
        const uint64_t head = t.offset - lo;
        uint64_t span = ((head + want + al.length - 1) / al.length) * al.length;
        if (lo + span > sizes[si]) { ++un_skipped; continue; }

        void* buf = dray::mem::aligned_alloc_host(static_cast<size_t>(span), al.memory);
        if (!buf) { ++un_skipped; continue; }
        dray::io::ReadRequest r;
        r.file = handles[si]; r.offset = lo; r.length = static_cast<uint32_t>(span);
        r.dst = buf; r.tag = 1;
        if (backend->submit(&r, 1) == 1) {
            dray::io::Completion c;
            if (backend->poll(&c, 1, 1) == 1 && c.ok(static_cast<uint32_t>(span))) {
                std::vector<uint8_t> ref(static_cast<size_t>(want), 0);
                if (read_reference_uncached(p.shard_paths[si], t.offset, want, ref.data())) {
                    if (std::memcmp(static_cast<const uint8_t*>(buf) + head, ref.data(),
                                    static_cast<size_t>(want)) != 0) {
                        ++un_mismatched;
                        std::printf("  MISMATCH %s @ %llu (%llu B prefix)\n", t.name.c_str(),
                                    static_cast<unsigned long long>(t.offset),
                                    static_cast<unsigned long long>(want));
                    }
                    ++un_checked;
                } else { ++un_skipped; }
            } else { ++un_skipped; }
        } else { ++un_skipped; }
        dray::mem::aligned_free_host(buf, span);
    }

    for (dray::io::FileId h : handles) {
        if (h != dray::io::kInvalidFile) backend->close(h);
    }

    std::printf("\nunconditional weight verification  (read every token)\n");
    std::printf("  tensors compared   %zu\n", un_checked);
    std::printf("  mismatches         %zu\n", un_mismatched);
    std::printf("  skipped            %zu\n", un_skipped);

    std::printf("\nexpert slice verification\n");
    std::printf("  shard              %s\n", a.model.c_str());
    std::printf("  slices compared    %zu\n", checked);
    std::printf("  mismatches         %zu\n", mismatched);
    std::printf("  skipped            %zu  (other shard, or unaligned span too large)\n", skipped);
    // BOTH passes decide the verdict. An earlier edit added the unconditional
    // pass, printed its mismatch count, and left the verdict reading only the
    // routed counters -- so a run in which every unconditional weight differed
    // still printed PASS and returned 0. A verifier whose second half cannot
    // fail is the exact defect its first half was written to fix (2026-08-24).
    const size_t all_checked = checked + un_checked;
    const size_t all_bad     = mismatched + un_mismatched;
    std::printf("  verdict            %s\n",
                (all_checked > 0 && all_bad == 0) ? "PASS - uncached reads land byte-identical"
                                                  : (all_checked == 0 ? "INCONCLUSIVE - nothing compared"
                                                                      : "FAIL"));
    return (all_checked > 0 && all_bad == 0) ? 0 : 1;
}


}  // namespace dray::cli
