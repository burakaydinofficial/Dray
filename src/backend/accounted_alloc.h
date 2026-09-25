// THE ONLY WAY THE STREAMER ALLOCATES.
//
// Invariant 1 says every allocation goes through one accounted allocator, and it
// was not being honoured: the streamer once tracked its own `in_use` against its
// own `budget` while the Accountant -- which owns the cap -- was told about the
// sentinel and nothing else. Slots and per-read staging were invisible to it, so
// the cap bound our estimate of memory rather than memory. Measured on Qwen3.8 at
// a 12 GiB cap: 13.57 GB private bytes.
//
// Reserving BEFORE allocating is the point. A refused reservation means the
// caller must evict or wait, never proceed.
//
// Memory comes from one contiguous VA arena (mem::PoolArena) when it is enabled:
// block sizes repeat (uniform slot classes), commit tracks live bytes, and a GPU
// backend registers ONE range instead of chasing thousands of mallocs. Plain host
// memory is the fallback, and every free routes by origin -- freeing an arena
// block with the host deallocator is heap corruption.

#pragma once

#include <cstdint>

#include "mem/accountant.h"
#include "mem/pool_arena.h"

namespace dray::backend {

// ggml's host tensor alignment: every block a tensor may point into.
inline constexpr uint32_t kHostAlign = 64;

class AccountedAlloc {
public:
    explicit AccountedAlloc(mem::Accountant& ledger) : ledger_(ledger) {}
    AccountedAlloc(const AccountedAlloc&) = delete;
    AccountedAlloc& operator=(const AccountedAlloc&) = delete;

    // Reserves `reserve_bytes` of VA for the arena. Not called under
    // DRAY_NO_POOL, which leaves every allocation on the host fallback.
    void init_arena(uint64_t reserve_bytes) { arena_.init(reserve_bytes); }

    // Charged to `cat` before the memory exists; null when the ledger refuses or
    // the host is out of memory (the charge is then undone).
    void* alloc(mem::Category cat, uint64_t bytes, uint32_t align);
    // Credits the same category the allocation debited.
    void  free(mem::Category cat, void* p, uint64_t bytes);

    // Memory the ledger does not see through this class: the poison sentinel
    // (charged once by its owner), the emergency poison buffer (deliberately
    // outside the cap -- the run is dying), and teardown, where charged blocks
    // are returned without crediting a ledger that outlives nothing that reads
    // it. Arena first, host fallback; free routes by origin.
    void* alloc_uncharged(uint64_t bytes, uint32_t align);
    void  free_uncharged(void* p, uint64_t bytes);

    // The cache's live share, derived from the ledger at the moment of asking.
    // NEVER mirrored: a copy is updated at a different instant than the ledger,
    // and the two disagree exactly at the cap -- where make_room once reported
    // room while every reservation was being refused.
    uint64_t cache_used() const;
    // Whatever the cap has left after every non-cache category: floor, KV,
    // compute scratch, and staging currently in flight.
    uint64_t cache_budget() const;

    mem::Accountant&       ledger()       { return ledger_; }
    const mem::Accountant& ledger() const { return ledger_; }
    const mem::PoolArena&  arena()  const { return arena_; }

private:
    mem::Accountant& ledger_;
    mem::PoolArena   arena_;
};

}  // namespace dray::backend
