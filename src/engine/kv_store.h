// What the scheduler keeps of finished conversations, and how it reuses them.
//
// Every slot has its own KV, funded at load, so keeping a finished
// conversation in its slot costs no extra memory: it stays until the slot is
// needed for something else (least recently used first) or its idle timeout.
// A new request goes to the free slot whose kept tokens share the longest
// prefix with its prompt, and only the rest of the prompt is prefilled.
//
// Attention KV is cut back to the shared prefix directly. Recurrent state
// cannot be cut back -- it holds only its latest position -- so the store
// checkpoints it a few tokens before each prompt's end (a chat's next turn
// shares the whole previous prompt; the re-tokenized tail is re-prefilled) and
// restores the newest checkpoint at or before the divergence point.
//
// The scheduler talks only to this interface: a disk-backed store later is a
// different implementation chosen by configuration, and nothing else changes.
#pragma once

#include <cstdint>
#include <vector>

namespace dray::engine {

class KvStore {
public:
    virtual ~KvStore() = default;

    struct Placement {
        int32_t slot = -1;    // the slot to use (one of the free slots offered)
        int32_t reuse = 0;    // prompt tokens already in its KV; prefill starts here
    };
    // Chooses one of `free_slots` for `prompt` and leaves its KV holding
    // exactly the first `reuse` prompt tokens (truncated, or a checkpoint
    // restored, or emptied). Always leaves at least one prompt token to feed:
    // the first sampled token needs that token's logits.
    virtual Placement place(const std::vector<int32_t>& prompt,
                            const std::vector<int32_t>& free_slots) = 0;

    // The prompt position at which the scheduler must end a step so the store
    // can checkpoint there (-1 = none): `prompt_len` tokens, the first `reuse`
    // already in the KV.
    virtual int32_t checkpoint_position(int32_t prompt_len, int32_t reuse) const = 0;
    // `slot`'s KV now holds exactly `tokens` (a step boundary at the position
    // checkpoint_position named): take the checkpoint.
    virtual void    checkpoint(int32_t slot, const std::vector<int32_t>& tokens) = 0;

    // A request ended in `slot` and its KV holds `tokens`: keep them.
    virtual void    retain(int32_t slot, std::vector<int32_t> tokens) = 0;
    // `slot`'s KV is no longer trustworthy or is being used otherwise: forget it.
    virtual void    forget(int32_t slot) = 0;
};

}  // namespace dray::engine
