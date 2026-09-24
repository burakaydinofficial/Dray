// Expert cache: per-layer slot arenas holding routed experts streamed from SSD.
//
// GEOMETRY IS FORCED, NOT STYLISTIC. ggml_mul_mat_id takes ONE tensor covering all
// experts for a layer and indexes it by a uniform stride nb[2] from one base
// pointer. Scattered slots therefore cannot be expressed by repointing
// tensor->data. So slots for a layer live in one contiguous arena of uniform slot
// size, the layer's expert tensor is built over that arena with
//   ne[2] = n_slots, nb[2] = slot_bytes
// and each token the scheduler writes a remap tensor translating router-chosen
// expert ids into slot indices. This is the same shape llama.cpp PR #25294 uses.
//
// Slot size is per-layer, read from the GGUF tensor table (Invariant 3): MiniMax
// M3 UD-Q2_K_XL has three distinct per-layer expert sizes, and one global slot
// sized to the largest would waste ~26% of the arena on 54 of 57 layers.
//
// EVICTION MUST NEVER PICK A SLOT WITH A READ IN FLIGHT. The storage interface
// deliberately has no cancel() (the portable macOS floor cannot cancel), so an
// in-flight slot is pinned by refcount until its completion is harvested.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "io/storage.h"
#include "mem/accountant.h"
#include "plan/residency.h"

namespace dray::cache {

static constexpr int32_t kNoSlot = -1;

struct SlotState {
    int32_t  expert_id = -1;    // which expert currently occupies this slot
    uint32_t refcount = 0;      // >0 while a read is in flight or compute is using it
    uint64_t last_used = 0;     // logical clock for LRU
    bool     valid = false;     // false until a read completes successfully
};

// One layer's arena. Owns a Slab and the slot bookkeeping.
class LayerCache {
public:
    LayerCache(mem::Accountant&, const plan::LayerSlotClass&, size_t n_slots,
               uint32_t align);

    bool     valid() const { return slab_ && slab_->valid(); }
    int32_t  layer() const { return spec_.layer; }
    size_t   n_slots() const { return slab_ ? slab_->n_slots() : 0; }
    uint64_t slot_bytes() const { return spec_.slot_bytes; }
    void*    base() { return slab_ ? slab_->base() : nullptr; }

    // Slot currently holding expert_id, or kNoSlot.
    int32_t find(int32_t expert_id) const;

    // Chooses a slot to (re)use for expert_id. Never returns a slot with
    // refcount > 0. Returns kNoSlot when every slot is pinned.
    int32_t acquire(int32_t expert_id, uint64_t clock);

    void mark_valid(int32_t slot);
    void mark_failed(int32_t slot);
    void pin(int32_t slot);
    void unpin(int32_t slot);

    void*    slot_ptr(int32_t slot);
    const SlotState& state(int32_t slot) const { return slots_[slot]; }

    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }

private:
    plan::LayerSlotClass    spec_;
    std::unique_ptr<mem::Slab> slab_;
    std::vector<SlotState>  slots_;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
};

// Per-token remap handed to ggml_mul_mat_id: for each routed expert selected by
// the router, the slot index holding it. Built fresh each token.
struct Remap {
    std::vector<int32_t> slot_of_selected;  // size = n_expert_used
    bool complete = false;                  // false if any expert failed to load
};

class ExpertCache {
public:
    // Opens one handle per shard from plan.shard_paths. A tensor's offset is only
    // meaningful against its own shard, so a single FileId cannot address these
    // models -- every target ships 4-14 shards.
    ExpertCache(mem::Accountant&, io::Backend&, const plan::Plan&, uint32_t align);
    ~ExpertCache();

    bool valid() const;
    // Empty on success; otherwise why the cache could not be built.
    const std::string& error() const { return error_; }

    // Ensures the given experts for a layer are resident, issuing reads for misses
    // and blocking until all complete. Returns a Remap. On a read failure the
    // corresponding entry is kNoSlot and complete==false: callers MUST refuse to
    // compute rather than feed a partially filled slot to mul_mat_id, which would
    // produce plausible text from garbage weights.
    Remap ensure(int32_t layer, const int32_t* expert_ids, size_t n);

    void release(int32_t layer, const Remap&);

    LayerCache* layer_cache(int32_t layer);

    uint64_t bytes_read() const { return bytes_read_; }
    uint64_t hits() const;
    uint64_t misses() const;
    double   hit_rate() const;   // routed hit rate: hits / (hits + misses)

    std::string report() const;

private:
    mem::Accountant& acct_;
    io::Backend&     backend_;
    std::vector<io::FileId> shard_files_;   // one per plan.shard_paths entry
    std::string      error_;
    uint32_t         align_ = 4096;
    uint64_t         clock_ = 0;
    uint64_t         bytes_read_ = 0;

    std::vector<std::unique_ptr<LayerCache>> layers_;   // indexed by layer id
    // Flat, layer-major, 3 entries per layer in {gate, up, down} order. Strides
    // differ per tensor -- on M3 gate/up are IQ2_XS and down is IQ3_XXS -- so a
    // single per-expert size would be wrong.
    std::vector<uint64_t> expert_offsets_;
    std::vector<uint64_t> expert_strides_;
    std::vector<int32_t>  expert_shards_;   // which shard each fused tensor lives in
    uint32_t n_experts_ = 0;

    uint64_t   expert_offset(int32_t layer, int32_t which) const;
    uint64_t   expert_stride(int32_t layer, int32_t which) const;
    io::FileId expert_file(int32_t layer, int32_t which) const;
};

}  // namespace dray::cache
