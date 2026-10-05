#include "engine/ram_kv_store.h"

#include <algorithm>
#include <utility>

namespace dray::engine {

namespace {

int32_t shared_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return static_cast<int32_t>(i);
}

}  // namespace

RamKvStore::RamKvStore(Stepper& stepper, Config config)
    : stepper_(stepper), cfg_(std::move(config)) {
    kept_.resize(static_cast<size_t>(std::max(1, stepper_.slots())));
}

RamKvStore::~RamKvStore() {
    while (!pool_.empty()) drop(pool_.begin());
}

RamKvStore::Kept& RamKvStore::kept(int32_t slot) {
    return kept_[static_cast<size_t>(slot)];
}

int64_t RamKvStore::now() const {
    if (cfg_.clock) return cfg_.clock();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

int32_t RamKvStore::cut_to(int32_t slot, int32_t n) {
    Kept& k = kept(slot);
    if (stepper_.truncate_kv(slot, n)) {
        k.tokens.resize(static_cast<size_t>(n));
    } else {
        // Recurrent state: the newest checkpoint at or before n, then the
        // attention part cut back to the same point.
        const Checkpoint* best = nullptr;
        for (const Checkpoint& c : k.checkpoints) {
            if (c.n_tokens <= n && (!best || c.n_tokens > best->n_tokens)) best = &c;
        }
        if (!best || !stepper_.load_checkpoint(slot, best->data) ||
            !stepper_.truncate_kv(slot, best->n_tokens)) {
            return 0;
        }
        ++stats_.checkpoints_restored;
        k.tokens.resize(static_cast<size_t>(best->n_tokens));
    }
    const int32_t held = static_cast<int32_t>(k.tokens.size());
    // Checkpoints past the cut describe tokens that are gone.
    k.checkpoints.erase(std::remove_if(k.checkpoints.begin(), k.checkpoints.end(),
                                       [held](const Checkpoint& c) { return c.n_tokens > held; }),
                        k.checkpoints.end());
    return held;
}

// --- the pool ----------------------------------------------------------------

RamKvStore::Parked RamKvStore::take(std::list<Parked>::iterator it) {
    stats_.pool_tokens -= static_cast<int64_t>(it->tokens.size());
    stats_.pool_bytes -= it->bytes;
    --stats_.pool_entries;
    Parked p = std::move(*it);
    pool_.erase(it);
    return p;
}

void RamKvStore::refund(const Parked& p) {
    if (cfg_.memory && p.bytes) cfg_.memory->refund(p.bytes);
}

void RamKvStore::drop(std::list<Parked>::iterator it) {
    refund(take(it));
}

void RamKvStore::expire_pool(int64_t t, int64_t timeout) {
    for (auto it = pool_.begin(); it != pool_.end();) {
        if (t - it->last_used > timeout) {
            auto dead = it++;
            drop(dead);
            ++stats_.pool_evicted;
        } else {
            ++it;
        }
    }
}

bool RamKvStore::park(int32_t slot) {
    const Kept& k = kept(slot);
    if (cfg_.pool_max_tokens <= 0 || !k.valid || k.tokens.empty()) return false;
    const int64_t n = static_cast<int64_t>(k.tokens.size());
    if (n > cfg_.pool_max_tokens) return false;   // could never fit

    Parked p;
    if (!stepper_.save_state(slot, &p.state)) return false;
    p.tokens = k.tokens;
    p.checkpoints = k.checkpoints;
    p.bytes = p.state.size();
    for (const Checkpoint& c : p.checkpoints) p.bytes += c.data.size();

    // Room: the oldest parked conversations go first, for the token budget
    // and for memory the ledger will not grant.
    while (!pool_.empty() && stats_.pool_tokens + n > cfg_.pool_max_tokens) {
        drop(std::prev(pool_.end()));
        ++stats_.pool_evicted;
    }
    if (cfg_.memory) {
        while (!cfg_.memory->charge(p.bytes)) {
            if (pool_.empty()) return false;   // no room at all: not parked
            drop(std::prev(pool_.end()));
            ++stats_.pool_evicted;
        }
    }
    p.last_used = k.last_used;
    stats_.pool_tokens += n;
    stats_.pool_bytes += p.bytes;
    ++stats_.pool_entries;
    ++stats_.parked;
    pool_.push_front(std::move(p));
    return true;
}

bool RamKvStore::restore(Parked p, int32_t slot) {
    refund(p);   // from here it lives in the slot's funded KV, or nowhere
    stepper_.clear_kv(slot);
    forget(slot);
    if (!stepper_.load_state(slot, p.state)) {
        stepper_.clear_kv(slot);   // a state that does not load back is worth nothing
        return false;
    }
    Kept& k = kept(slot);
    k.tokens = std::move(p.tokens);
    k.checkpoints = std::move(p.checkpoints);
    k.valid = true;
    // A slot never holds more checkpoints than it was reserved for.
    while (static_cast<int32_t>(k.checkpoints.size()) > std::max(0, cfg_.checkpoints_per_slot)) {
        k.checkpoints.erase(k.checkpoints.begin());
    }
    ++stats_.restored;
    return true;
}

// --- placement -----------------------------------------------------------------

KvStore::Placement RamKvStore::place(const std::vector<int32_t>& prompt,
                                     const std::vector<int32_t>& free_slots) {
    ++stats_.placed;
    const int64_t t = now();
    const int64_t timeout = std::chrono::duration_cast<std::chrono::milliseconds>(cfg_.idle_timeout).count();
    expire_pool(t, timeout);
    // At least one prompt token must be fed: its logits give the first token.
    const int32_t limit = static_cast<int32_t>(prompt.size()) - 1;

    // The longest shared prefix among the free slots' conversations ...
    int32_t best_slot = -1, slot_len = 0;
    for (int32_t s : free_slots) {
        const Kept& k = kept(s);
        if (!k.valid || t - k.last_used > timeout) continue;
        const int32_t c = shared_prefix(k.tokens, prompt);
        if (c > slot_len) { best_slot = s; slot_len = c; }
    }
    // ... and among the parked ones.
    auto best_parked = pool_.end();
    int32_t parked_len = 0;
    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
        const int32_t c = shared_prefix(it->tokens, prompt);
        if (c > parked_len) { best_parked = it; parked_len = c; }
    }

    // A slot to receive a conversation from elsewhere: an empty one, else the
    // least recently used -- whose own conversation is parked first.
    auto vacate = [&]() -> int32_t {
        int32_t pick = -1;
        for (int32_t s : free_slots) {
            if (!kept(s).valid) { pick = s; break; }
        }
        if (pick < 0) {
            for (int32_t s : free_slots) {
                if (pick < 0 || kept(s).last_used < kept(pick).last_used) pick = s;
            }
        }
        if (pick >= 0 && kept(pick).valid) park(pick);
        return pick;
    };
    auto reused = [&](int32_t slot, int32_t held) -> Placement {
        kept(slot).last_used = t;
        ++stats_.reused;
        stats_.reused_tokens += held;
        return {slot, held};
    };

    if (best_parked != pool_.end() && parked_len > slot_len && std::min(parked_len, limit) > 0) {
        // Out of the pool FIRST: vacating a slot parks its conversation, which
        // may evict the oldest entries -- this one included, were it still there.
        Parked p = take(best_parked);
        const int32_t slot = vacate();
        if (slot < 0) {
            refund(p);
        } else if (restore(std::move(p), slot)) {
            const int32_t held = cut_to(slot, std::min(parked_len, limit));
            if (held > 0) return reused(slot, held);
        }
    } else if (best_slot >= 0 && std::min(slot_len, limit) > 0) {
        const int32_t target = std::min(slot_len, limit);
        // A partial match cuts this conversation back; keep a full copy so the
        // client it belongs to can still continue it.
        if (target < static_cast<int32_t>(kept(best_slot).tokens.size())) park(best_slot);
        const int32_t held = cut_to(best_slot, target);
        if (held > 0) return reused(best_slot, held);
    }

    // Nothing to reuse: start empty, parking what the slot held.
    const int32_t pick = vacate();
    if (pick < 0) return {};
    forget(pick);
    stepper_.clear_kv(pick);
    kept(pick).last_used = t;
    return {pick, 0};
}

int32_t RamKvStore::checkpoint_position(int32_t prompt_len, int32_t reuse) const {
    if (cfg_.checkpoints_per_slot <= 0 || !stepper_.needs_checkpoints()) return -1;
    const int32_t p = prompt_len - cfg_.checkpoint_offset;
    return p > reuse ? p : -1;
}

void RamKvStore::checkpoint(int32_t slot, const std::vector<int32_t>& tokens) {
    if (cfg_.checkpoints_per_slot <= 0) return;
    // Room first: the reservation covers checkpoints_per_slot of them per slot,
    // and the new one exists as soon as it is saved -- never one more.
    Kept& k = kept(slot);
    while (static_cast<int32_t>(k.checkpoints.size()) >= cfg_.checkpoints_per_slot) {
        k.checkpoints.erase(k.checkpoints.begin());
    }
    Checkpoint c;
    c.n_tokens = static_cast<int32_t>(tokens.size());
    if (!stepper_.save_checkpoint(slot, &c.data)) return;
    if (cfg_.max_checkpoint_bytes && c.data.size() > cfg_.max_checkpoint_bytes) {
        ++stats_.checkpoints_refused;   // never past what the cap reserved
        return;
    }
    k.checkpoints.push_back(std::move(c));
    ++stats_.checkpoints_taken;
}

void RamKvStore::retain(int32_t slot, std::vector<int32_t> tokens) {
    Kept& k = kept(slot);
    k.tokens = std::move(tokens);
    k.valid = !k.tokens.empty();
    k.last_used = now();
}

void RamKvStore::forget(int32_t slot) {
    Kept& k = kept(slot);
    k.tokens.clear();
    k.checkpoints.clear();
    k.valid = false;
}

}  // namespace dray::engine
