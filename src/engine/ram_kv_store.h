// KvStore in RAM: each slot keeps its last conversation's tokens (its KV stays
// where it is) and, for recurrent models, up to `checkpoints_per_slot`
// checkpoints of the state that cannot be truncated. The checkpoint memory is
// reserved by the caller at load (slots x checkpoints_per_slot x one state),
// inside the cap; this class never grows past that count.
//
// Beyond the slots, a POOL of parked conversations: when a slot is needed for
// something else, its conversation (whole state + checkpoints) is saved here
// instead of lost, and a later prompt that continues it is restored into any
// free slot. So kept conversations are not limited to the number of slots, and
// a client sharing only a prefix (a system prompt) no longer destroys another
// client's conversation: a full copy is parked before a partial match cuts it.
// Bounded by a token budget and by memory charged through MemoryCharge; least
// recently used first out; idle ones expire like slot-kept ones.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <list>
#include <vector>

#include "engine/kv_store.h"
#include "engine/stepper.h"

namespace dray::engine {

// Memory for parked conversations, charged to the caller's ledger. A refusal
// makes the pool give up its oldest entries, or not park.
class MemoryCharge {
public:
    virtual ~MemoryCharge() = default;
    virtual bool charge(uint64_t bytes) = 0;
    virtual void refund(uint64_t bytes) = 0;
};

class RamKvStore final : public KvStore {
public:
    struct Config {
        // Tokens the pool of parked conversations may hold in all; 0 = no pool
        // (a slot's conversation is lost when the slot is reused).
        int64_t pool_max_tokens = 0;
        // Where parked memory is charged (not owned). Null = uncharged (tests).
        MemoryCharge* memory = nullptr;
        // Recurrent models only; 0 = no checkpoints (reuse then needs a KV that
        // can be truncated, i.e. attention-only models).
        int32_t checkpoints_per_slot = 1;
        // The checkpoint sits this many tokens before a prompt's end: a chat's
        // next turn re-tokenizes the end of this one, and that tail is re-fed.
        int32_t checkpoint_offset = 4;
        // The largest checkpoint the caller reserved room for (bytes; 0 = no
        // bound). A bigger one is not kept: the reservation is the hard limit.
        uint64_t max_checkpoint_bytes = 0;
        // A kept conversation idle longer than this is no longer preferred.
        std::chrono::seconds idle_timeout{600};
        // Milliseconds, monotonic. Injectable for tests; steady_clock if empty.
        std::function<int64_t()> clock;
    };

    RamKvStore(Stepper& stepper, Config config);
    ~RamKvStore() override;   // refunds whatever the pool still holds

    Placement place(const std::vector<int32_t>& prompt,
                    const std::vector<int32_t>& free_slots) override;
    int32_t   checkpoint_position(int32_t prompt_len, int32_t reuse) const override;
    void      checkpoint(int32_t slot, const std::vector<int32_t>& tokens) override;
    void      retain(int32_t slot, std::vector<int32_t> tokens) override;
    void      forget(int32_t slot) override;

    struct Stats {
        int64_t placed = 0;
        int64_t reused = 0;            // placements that reused a prefix
        int64_t reused_tokens = 0;     // prompt tokens not prefilled again
        int64_t checkpoints_taken = 0;
        int64_t checkpoints_restored = 0;
        int64_t checkpoints_refused = 0;   // larger than max_checkpoint_bytes
        int64_t parked = 0;                // conversations saved into the pool
        int64_t restored = 0;              // ... and brought back into a slot
        int64_t pool_evicted = 0;          // dropped: budget, memory or idle
        int64_t pool_entries = 0;
        int64_t pool_tokens = 0;
        uint64_t pool_bytes = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    struct Checkpoint {
        int32_t n_tokens = 0;
        std::vector<uint8_t> data;
    };
    struct Kept {
        std::vector<int32_t> tokens;   // exactly what the slot's KV holds
        std::vector<Checkpoint> checkpoints;   // oldest first; each a prefix of tokens
        int64_t last_used = 0;
        bool    valid = false;
    };

    struct Parked {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> state;              // the slot's whole state
        std::vector<Checkpoint> checkpoints;
        int64_t last_used = 0;
        uint64_t bytes = 0;                      // what was charged
    };

    Kept&   kept(int32_t slot);
    int64_t now() const;
    // Leaves `slot` holding its first `n` kept tokens (truncating, or restoring
    // the newest checkpoint at or before n). Returns the tokens it now holds.
    int32_t cut_to(int32_t slot, int32_t n);
    // Saves `slot`'s kept conversation into the pool (a copy: the slot keeps
    // it). False when there is no pool, nothing kept, or no room for it.
    bool    park(int32_t slot);
    // Takes an entry out of the pool (its memory stays charged until it is
    // restored or refunded).
    Parked  take(std::list<Parked>::iterator it);
    // Replaces `slot`'s state with a taken conversation; refunds its memory.
    bool    restore(Parked p, int32_t slot);
    void    refund(const Parked& p);
    void    drop(std::list<Parked>::iterator it);
    void    expire_pool(int64_t t, int64_t timeout);

    Stepper& stepper_;
    const Config cfg_;
    std::vector<Kept> kept_;
    std::list<Parked> pool_;   // most recently used first
    Stats stats_;
};

}  // namespace dray::engine
