// One contiguous VA-reserved arena for every hot-path allocation.
//
// Why this exists (2026-08-16): the streamer's per-token region churn was
// thousands of individual OS allocations -- correct under the cap, but
// scattered across the address space. Metal's zero-copy path wants host
// pointers inside ranges registered ONCE, and a pool of scattered mallocs
// cannot be registered. Reserving one span the size of the cap (address space
// costs nothing) and committing/decommitting pages inside it gives:
//   * one range to register with any future GPU backend, ever;
//   * commit charge == live bytes, so the RSS/commit reconcile that binds the
//     cap keeps meaning what it says (decommit on free is NOT optional);
//   * exact-size free lists, because slot classes make sizes repeat per layer.
//
// The accountant still gates every byte BEFORE the pool is asked (Invariant 1
// lives in alloc_acct, not here). This class manages address space, not budget.
//
// Exhaustion (VA fragmentation beyond the 2x headroom) falls back to plain
// aligned_alloc_host at the call site: correct, merely unregistered. The
// fallback count is exposed so a Metal-day audit can see whether it ever fired.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace dray::mem {

class PoolArena {
public:
    PoolArena() = default;
    ~PoolArena();
    PoolArena(const PoolArena&) = delete;
    PoolArena& operator=(const PoolArena&) = delete;

    // Reserve (not commit) `reserve_bytes` of address space. Returns false and
    // leaves the pool disabled if the OS refuses; callers then use their
    // fallback path for everything.
    bool init(uint64_t reserve_bytes);
    bool enabled() const { return base_ != nullptr; }

    // Both return nullptr when the pool cannot serve the request (disabled, or
    // VA exhausted); the caller falls back. align must be a power of two and
    // is honoured up to the page size implicitly (blocks are page-granular).
    void* alloc(uint64_t bytes, uint32_t align);
    void  free(void* p, uint64_t bytes);

    bool contains(const void* p) const {
        return base_ && p >= base_ && p < base_ + reserved_;
    }

    // For future GPU-backend registration and for the report line.
    void*    base() const { return base_; }
    uint64_t span() const { return reserved_; }
    uint64_t committed() const { return committed_; }
    uint64_t page() const { return page_; }
    uint64_t fallbacks() const { return fallbacks_; }
    void     note_fallback() { ++fallbacks_; }

private:
    uint8_t* base_ = nullptr;
    uint64_t reserved_ = 0;
    uint64_t bump_ = 0;         // high-water mark of ever-carved space
    uint64_t committed_ = 0;    // live committed bytes (block-granular)
    uint64_t fallbacks_ = 0;
    uint64_t page_ = 4096;
    bool always_rw_ = false;   // POSIX GPU-mapped mode; see init()
    // Exact padded-size free lists: offsets of decommitted blocks available for
    // recommit. Sizes repeat (uniform slot classes per layer), so exact-size
    // reuse wins and coalescing complexity is not paid.
    std::map<uint64_t, std::vector<uint64_t>> free_;
};

}  // namespace dray::mem
