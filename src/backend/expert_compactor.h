// EXPERT COMPACTION -- the streamer's central trick -- and row slicing, its twin.
//
// MUL_MAT_ID reads only n_expert_used of n_expert experts, but the fused tensor
// covers all of them: materialising it whole reads ~4.7 GiB per MoE layer to use
// ~98 MB (on Qwen3.8, ~450 GB vs ~9 GB per token). And mul_mat_id indexes every
// expert by one uniform stride nb[2], so scattered memory cannot be expressed by
// repointing tensor->data. So: read ONLY the router-selected experts into ONE
// compact region at the tensor's own stride, and give the node PRIVATE ids
// holding compact indices 0..k-1 (PrivateIds). The router ran earlier in the same
// graph, so the ids are readable by the time the node runs.
//
// GET_ROWS on the token embedding is the same idea one node type over: rows are
// uniformly strided by nb[1], and a 1.09 GiB table is read for a handful of
// ~7.5 KB rows. That single tensor is what made a small cap impossible.
//
// PHASE A (fused sibling reads): gate/up/down of a layer share one ids tensor, so
// the moment ONE of them derives the routing, the other two regions are submitted
// unwaited -- the whole layer's reads in flight together, up/down streaming
// while gate computes. PHASE C (early unlock): the router's ids tensor is itself
// a graph node; the instant its release callback fires, all three regions are
// submitted, several small ops before the first MUL_MAT_ID asks for them.
// Which ids unlock which experts is LEARNED at the first compaction (the graph
// declares it by consuming them together), which keeps this model-agnostic.

#pragma once

#include <atomic>
#include <cstdint>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "backend/accounted_alloc.h"
#include "backend/expert_slots.h"
#include "backend/hit_rates.h"
#include "backend/io_scheduler.h"
#include "backend/private_ids.h"
#include "backend/residency_cache.h"
#include "backend/routing_skew.h"
#include "backend/stream_flags.h"
#include "backend/tensor_registry.h"
#include "ggml.h"

namespace dray::backend {

class ExpertCompactor {
public:
    // Everything a compaction touches, borrowed from the Streamer that owns them.
    struct Parts {
        AccountedAlloc&       mem;
        IoScheduler&          io;
        ResidencyCache&       cache;
        const TensorRegistry& tensors;
        PrivateIds&           ids;
        ExpertSlots&          slots;
        RoutingSkew&          skew;
        HitRates&             hits;
    };
    // `n_seq` is the admitted batch width: a node whose ids carry at most that
    // many rows is a decode step, anything wider is prefill.
    ExpertCompactor(const Parts& parts, const StreamFlags& flags, uint32_t n_seq,
                    const std::atomic<uint64_t>& failures)
        : p_(parts), flags_(flags), decode_width_(n_seq ? n_seq : 1), failures_(failures) {}
    ExpertCompactor(const ExpertCompactor&) = delete;
    ExpertCompactor& operator=(const ExpertCompactor&) = delete;

    // MUL_MAT_ID: materialises only the experts `ids` selects from `w` and
    // repoints `node` at private compact ids. False = declined or failed; the
    // caller falls back to the whole tensor (after restoring the original ids).
    bool compact_experts(ggml_tensor* node, ggml_tensor* w, ggml_tensor* ids,
                         uint64_t* streamed);
    // GET_ROWS: the same for rows of a plain 2-D table. Declines when the slice
    // would be most of the table: the whole tensor is cacheable across tokens.
    bool compact_rows(ggml_tensor* node, ggml_tensor* w, ggml_tensor* ids,
                      uint64_t* streamed);
    // Phase C, from the release callback: if `node` is a known router ids
    // tensor, submit its layer's three regions now.
    void unlock_early(ggml_tensor* node);

    uint64_t early_unlocks() const { return early_unlocks_; }
    // IO_STATS: time inside Phase C and in region room/allocation.
    void append_timing(std::ostream& o) const;
    // The expert union the churn reserve was sized for (StreamBudget); 0 = only
    // pure decode steps take the fast paths.
    void set_funded_union(uint64_t n) { funded_union_ = n; }

    // Teardown: every pending region's DMA must complete before its memory may
    // be freed; a region whose backend died is leaked, loudly.
    void release_all();

    // For a GPU split's copy of an expert-fused weight: a full-size region with
    // ONLY `experts` present, each at its own index (the copy reads exactly
    // those), read in one deep batch, frequency-cache hits copied from RAM.
    // Never cached -- the rest of the region is garbage. Returns the data
    // pointer, or null; *mem and *bytes are what to free.
    // With copy slots set (set_copy_slots), it lands in the slot of w's expert
    // kind instead -- memory the GPU backend DMAs from directly -- and *bytes is 0
    // (nothing to free); a matching prefetch there is waited for, not re-read.
    uint8_t* read_experts_for_copy(ggml_tensor* w, const int32_t* experts, int64_t n,
                                   uint8_t** mem, uint64_t* bytes, uint64_t* streamed);

    // GPU copy landing slots, one per expert kind (gate / up / down): pinned memory
    // owned by the caller, which must call settle_copy_slots() before freeing it.
    // A layer's three expert tensors use the SAME expert ids and are copied in
    // consecutive splits, so when one is copied the other two kinds' experts are
    // read ahead into their slots (prefetch_copy_siblings), overlapping the disk
    // with the copy and the GPU work in between.
    static constexpr int kCopyKinds = 3;
    static int copy_kind(const ggml_tensor* w);   // 0 gate (also fused/other), 1 up, 2 down
    static int copy_kind_of(const char* tensor_name);
    void set_copy_slots(uint8_t* const base[kCopyKinds], const uint64_t bytes[kCopyKinds]);
    void prefetch_copy_siblings(ggml_tensor* w, const int32_t* experts, int64_t n, uint64_t* streamed);
    void settle_copy_slots();
    uint64_t copy_prefetch_hits() const { return copy_prefetch_hits_; }
    uint64_t copy_prefetch_misses() const { return copy_prefetch_misses_; }

private:
    // A sibling region from submission until its own node adopts it. NOT in the
    // LRU, so unreclaimable while pending -- bounded by two sibling regions
    // (~100 MB on K3), which the churn reserve covers.
    struct PendingRegion {
        uint8_t*              mem = nullptr;
        uint64_t              bytes = 0;
        std::vector<int32_t>  uniq;    // the routing this layout was built for
        std::vector<uint64_t> tags;    // outstanding reads
        std::vector<size_t>   miss;    // slots that came from DISK (bytes accounting
                                       // and eslot admission)
        uint32_t              head = 0;   // data starts at mem + head (alloc_region)
    };
    // The expert tensors one ids tensor unlocks, learned at first compaction.
    struct ExpertTrio {
        ggml_tensor* w[3] = {nullptr, nullptr, nullptr};
        int64_t      n_expert = 0;
    };

    // The decode paths (sibling prefetch, early unlock, routing statistics) for
    // a step: a pure decode step (at most the funded width of tokens), or one
    // whose union of experts is no larger than the union the churn reserve
    // funded (StreamBudget: expected union at the funded width, k for one
    // stream). Per layer the slot size is fixed, so that IS a byte bound: a
    // prompt chunk riding along with decode tokens (continuous batching) keeps
    // the fast paths while its region fits the reserve, and falls back to the
    // safe path the moment it would not.
    // The frequency cache admits only on DECODE-width steps (at most the funded
    // width of tokens). Prefill -- and a prompt chunk riding along with decode
    // tokens -- never admits: its allocations would compete with the static
    // pinning claim that has not settled (GLM 16 GiB: slots admitted at prefill
    // packed the budget, the region had no room). One rule for both paths.
    bool admits(const ggml_tensor* ids) const {
        return ids->ne[1] <= static_cast<int64_t>(decode_width_);
    }
    bool fast_path(const ggml_tensor* ids, size_t n_uniq) const {
        if (ids->ne[1] <= static_cast<int64_t>(decode_width_)) return true;
        return funded_union_ > 0 && n_uniq <= funded_union_;
    }
    void submit_sibling_region(ggml_tensor* w, const std::vector<int32_t>& uniq);
    void submit_layer_siblings(ggml_tensor* w, const std::vector<int32_t>& uniq);
    // Room for a region of `need` bytes of expert slots, placed so each slot can
    // take its expert's aligned middle as a DIRECT read (IoScheduler::submit_exact):
    // the data starts at mem + head, head = the tensor's file offset modulo the
    // alignment, which every expert shares when the on-disk stride is a multiple
    // of it (K3, GLM: all of them). Otherwise head is 0 and reads are staged, as
    // before. Evicts through make_room. False = no room.
    // `spare`: the tensor's own previous region, reclaimed for this replacement
    // (ResidencyCache::reclaim_own with region_geom's bytes and align) -- used
    // as is, with no eviction and no allocation.
    bool alloc_region(const Source& src, uint64_t stride, uint64_t need,
                      uint8_t** mem, uint64_t* bytes, uint32_t* head, void* spare = nullptr);
    // Size and placement a region of `need` bytes gets for `src`: the one rule
    // alloc_region and reclaim_own must agree on.
    struct RegionGeom { uint64_t bytes = 0; uint32_t head = 0; uint32_t align = 0; };
    RegionGeom region_geom(const Source& src, uint64_t stride, uint64_t need) const;

    Parts                        p_;
    const StreamFlags&           flags_;
    const uint32_t               decode_width_;
    uint64_t                     funded_union_ = 0;   // set_funded_union
    const std::atomic<uint64_t>& failures_;
    std::unordered_map<const ggml_tensor*, PendingRegion> pending_;
    struct CopySlot {
        uint8_t*              mem = nullptr;   // the pinned slot (caller-owned)
        uint64_t              bytes = 0;
        bool                  pending = false; // reads in flight into it (tags)
        const ggml_tensor*    w = nullptr;     // what they are for
        std::vector<int32_t>  experts;
        uint8_t*              data = nullptr;
        std::vector<uint64_t> tags;
    };
    CopySlot copy_slots_[kCopyKinds];
    uint64_t copy_prefetch_hits_ = 0, copy_prefetch_misses_ = 0;
    // Waits for a slot's reads; false if any failed. A dead backend leaves the slot
    // unusable (its memory is still a DMA target), loudly.
    bool settle_slot(CopySlot& s);
    std::unordered_map<const ggml_tensor*, ExpertTrio>    trio_by_ids_;
    uint64_t early_unlocks_ = 0;
    uint64_t ns_unlock_ = 0, ns_room_ = 0, ns_alloc_ = 0;
    int      traced_ = 0;   // DRAY_TRACE_COMPACT lines printed so far
};

}  // namespace dray::backend
