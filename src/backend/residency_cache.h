// What is in RAM right now, and what may leave.
//
// Every materialised weight -- a pinned floor tensor, a whole unconditional
// tensor, a compacted expert region -- is one Resident entry here, charged to
// the ledger through AccountedAlloc. Eviction, static pinning and the protected
// working set of the node being computed all live in this one place, so "is
// this byte evictable?" has exactly one answer.
//
// BELOW THE KNEE, POLICY IS IRRELEVANT AND SIZE IS EVERYTHING. The graph sweeps
// every layer once per token, so each weight is touched exactly once per cycle.
// When the working set exceeds the cache, LRU evicts every entry before its
// reuse: hit rate 0%, and reordering WHICH entry is evicted cannot change that
// (measured: priority eviction left bytes-streamed byte-identical at 594.007 GB).
// Pinning a fixed subset instead converts that 0% into cache/working-set -- on
// Qwen3.8 at 12 GiB, ~8.9 GB pinned of a 36.9 GB stream, ~24% no longer re-read
// every token. This is the static-placement-beats-dynamic-caching result from
// the prior art (~11% vs 100% on a balanced model), from the other direction.

#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "backend/accounted_alloc.h"
#include "backend/io_scheduler.h"
#include "backend/poison_buffers.h"
#include "backend/tensor_registry.h"
#include "ggml.h"

namespace dray::backend {

// Eviction priority. Falls straight out of reads-per-token: an unconditional
// weight is read EVERY token, a routed expert only when selected (k/n_experts,
// so 10/512 on Qwen3.8). Evicting an unconditional weight to house an expert
// trades a 1.0 for a 0.02 -- which is what pure LRU does, and why one cap was
// doing ~50 GB/token against the ~31.7 the RAM curve predicts.
enum class Prio : uint8_t {
    RoutedExpert  = 0,   // evict these first
    Unconditional = 1,   // ~51x more valuable per byte on Qwen3.8
};

struct Resident {
    void*    mem = nullptr;
    uint64_t bytes = 0;
    uint32_t refs = 0;
    bool     pinned = false;
    Prio     prio = Prio::Unconditional;
    // Which ledger line this memory is charged to, so releasing it credits the
    // same category that reserving it debited. Getting this wrong would drift the
    // Accountant silently, which is the failure the single-ledger change exists to
    // prevent.
    mem::Category cat = mem::Category::ExpertCache;
    // Offset from the allocation base to the tensor's first byte. Non-zero when the
    // read was widened to device alignment and landed directly in this buffer, with
    // no staging copy. `mem`+`bytes` describe the ALLOCATION (what to free and what
    // is charged); the tensor points at mem+head.
    uint32_t head = 0;
    // For compacted regions: the exact expert (or row) ordering this layout holds.
    // A compacted region is ONLY valid for the routing decision that produced it,
    // so reuse is gated on the ordering matching, not on recency. Empty means a
    // whole tensor, valid for any routing.
    std::vector<int32_t> uniq;
};

// Memory held outside the LRU that the cache may ask back under pressure: the
// expert slot pool. Asked after dead expert regions and before unconditional
// weights -- a pinned unconditional byte saves a read on EVERY token, a slot
// only when its expert is picked again.
class SpillSource {
public:
    virtual ~SpillSource() = default;
    // Frees at least `bytes` if it can (its least valuable first); returns the
    // bytes freed.
    virtual uint64_t release(uint64_t bytes) = 0;
};

class ResidencyCache final : public Reclaimer {
public:
    using Entries = std::unordered_map<const ggml_tensor*, Resident>;

    ResidencyCache(AccountedAlloc& mem, const PoisonBuffers& poison,
                   const TensorRegistry& tensors)
        : mem_(mem), poison_(poison), tensors_(tensors) {}

    // Room the node being computed is guaranteed: static pinning never takes it,
    // and speculative admissions (ring retention, eslots) never dip into it.
    void     set_churn_reserve(uint64_t bytes) { churn_reserve_ = bytes; }
    uint64_t churn_reserve() const { return churn_reserve_; }
    // True when `bytes` fits in GENUINELY free budget, leaving the churn reserve
    // untouched -- the admission rule for everything that is optional.
    bool fits_beside_churn(uint64_t bytes) const {
        return mem_.cache_used() + bytes + churn_reserve_ <= mem_.cache_budget();
    }

    // How much of the cache budget may be held permanently. Derived from the live
    // ledger, never stored: the floor is charged during init_tensor, AFTER
    // construction, so a value computed once would size pinning against a
    // ceiling that no longer exists. Router gates alone are 1.5-2.4 GB.
    uint64_t static_allowance() const;
    uint64_t static_used() const { return static_used_; }
    // First-come static residency while the allowance has room. First-come is
    // deliberate: it matches the access order, so the pinned set is a contiguous
    // prefix of the layer sweep -- and since every weight is read once per cycle,
    // WHICH ones are pinned does not change the hit rate, only how many.
    bool claim_static(uint64_t bytes);

    Resident*      find(const ggml_tensor* t);
    bool           contains(const ggml_tensor* t) const { return entries_.count(t) != 0; }
    const Entries& entries() const { return entries_; }
    size_t         size() const { return entries_.size(); }
    size_t         lru_size() const { return lru_.size(); }
    // Evictions and the time spent freeing their memory (IO_STATS).
    uint64_t       evictions() const { return evictions_; }
    uint64_t       evicted_bytes() const { return evicted_bytes_; }
    uint64_t       ns_evict_free() const { return ns_evict_free_; }

    // An entry that never enters the LRU: the mandatory floor, and tensors
    // llama.cpp owns. Replaces any existing entry for `t`.
    void add_permanent(const ggml_tensor* t, const Resident& r) { entries_[t] = r; }
    // An evictable entry, newest in the LRU. Replaces any existing entry for `t`.
    void add(const ggml_tensor* t, const Resident& r);
    // Frees `t`'s memory (crediting its category) and forgets it. No-op if absent.
    void drop(const ggml_tensor* t);

    // Evicts by PRIORITY first, recency second -- every routed region goes before
    // any unconditional weight -- skipping pinned entries and the protected
    // working set, until `need` more bytes fit the budget. Evicted tensors point
    // back at the sentinel. True when the bytes now fit.
    bool make_room(uint64_t need);
    bool reclaim(uint64_t bytes) override { return make_room(bytes); }
    // The slot pool, consulted by make_room between its two passes (see SpillSource).
    void set_spill(SpillSource* s) { spill_ = s; }

    // Hands over the memory of a dead routed-expert region of EXACTLY `bytes`
    // (least recent first, never pinned or protected), removing it from the
    // cache without freeing it. Its ledger charge carries over unchanged to the
    // caller's new region -- same category, same bytes -- so the cap accounting
    // is untouched, and the pages are neither decommitted nor recommitted: a
    // free+alloc per region cost K3 30 s per 16-token run. Null when none fits.
    void* take_region(uint64_t bytes, uint32_t align);
    uint64_t regions_reused() const { return regions_reused_; }
    // `t`'s OWN region, handed back for its replacement instead of freed: the
    // caller is about to give `t` a new region of exactly `bytes` at `align`
    // (a new routing for the same expert tensor). The pages are committed and
    // touched already, so the replacement skips the free, the fresh commit and
    // the OS zero-fill of every page; the ledger charge carries over. Protection
    // is not consulted: replacing its own region is exactly what the current
    // node is doing.
    //
    // On a MISMATCH (size, alignment, class, pinned) nullptr is returned and the
    // entry is either freed (`Mismatch::Drop`: the caller replaces it now, as the
    // node's own compaction always did) or left exactly where it is
    // (`Mismatch::Keep`: a sibling submitted ahead of its node, which never
    // touched the old entry -- the node decides later. Dropping there discarded
    // whole resident tensors under COMPACT_ALL: h_routed 94% -> 13%.)
    enum class Mismatch { Drop, Keep };
    void* reclaim_own(const ggml_tensor* t, uint64_t bytes, uint32_t align, Mismatch on_mismatch);
    uint64_t regions_recycled() const { return regions_recycled_; }

    // Rebudget support. When the budget contracts under them, static pins become
    // an over-claim nothing can evict (measured on Linux: 3.27 GB pinned of 2.39
    // allowed, output.weight homeless, the poison path reached). Demotes static
    // pins -- never the floor -- back to evictable until the claim fits.
    void demote_static_to_allowance();
    // Evicts until the cache is within budget or nothing evictable is left.
    void shrink_to_budget();

    // The weights the node being computed depends on: never evicted while set.
    // Replaces refcounting, which leaked whenever a materialise failed.
    void protect_none() { current_.clear(); }
    void protect(ggml_tensor* t) { current_.push_back(t); }
    bool is_protected(const ggml_tensor* t) const;
    const std::vector<ggml_tensor*>& protected_tensors() const { return current_; }

    // Teardown: returns every entry's memory without crediting the ledger.
    void release_all();

private:
    uint64_t evictions_ = 0, evicted_bytes_ = 0, ns_evict_free_ = 0;
    uint64_t regions_reused_ = 0;
    uint64_t regions_recycled_ = 0;
    AccountedAlloc&               mem_;
    const PoisonBuffers&          poison_;
    const TensorRegistry&         tensors_;
    Entries                       entries_;
    std::list<const ggml_tensor*> lru_;       // front = newest
    uint64_t                      static_used_ = 0;
    uint64_t                      churn_reserve_ = 0;
    SpillSource*                  spill_ = nullptr;
    std::vector<ggml_tensor*>     current_;   // mutable: materialise rewrites ->data
};

}  // namespace dray::backend
