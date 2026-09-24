// The authority on resident bytes (Invariant 1).
//
// The cap means TOTAL resident bytes. The mandatory floor and KV reservation are
// computed at load and SUBTRACTED to yield the expert-cache budget -- never added
// on top, because a cap that terms sit on top of promises nothing.
//
// Accounting categories exist so the startup report can show where a user's 4 GB
// actually went; on the big models the router-gate table alone is 1.5-2.4 GB, and
// a user who can't see that will think the cache is broken.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dray::mem {

// Every resident byte belongs to exactly one of these. Gates/norms are FLOOR, not
// CACHE -- they are reserved before the cache budget exists, so the two are never
// double-counted (this was a real bug in an earlier draft of the design).
enum class Category : uint8_t {
    RouterGates,      // ffn_gate_inp etc. Resident at any cap: a non-resident gate
                      // must be read before you know which experts to read, adding
                      // a serialized round trip per layer per token.
    NormsAndBiases,   // tens of MB; per-request overhead dominates their byte cost
    KvCache,          // attention KV, sized by n_ctx
    RecurrentState,   // KDA / Gated DeltaNet per-sequence state (fixed size)
    PrefillActivation,// wave-partitioned prefill buffer: hidden * n_ctx * 2, x2
    ComputeScratch,   // ggml graph buffers
    IoStaging,        // aligned bounce buffers for the storage backend
    ExpertCache,      // the slab -- whatever is left after everything above
    Misc,             // tokenizer, vocab, sampler, cache index
    _Count
};

const char* category_name(Category);

struct Breakdown {
    size_t bytes[static_cast<size_t>(Category::_Count)] = {};
    size_t total() const;
};

// Not a general-purpose allocator: it hands out nothing. It is a ledger that the
// real allocators (Slab, ggml buffers) consult before reserving, and it is the
// single number reconciled against OS-reported RSS on a timer.
class Accountant {
public:
    explicit Accountant(size_t cap_bytes);

    // Returns false if the reservation would exceed the cap. Callers must handle
    // this by refusing admission or shrinking a budget -- never by proceeding.
    bool reserve(Category, size_t bytes);
    void release(Category, size_t bytes);

    // Records bytes the process is holding that no caller reserved: llama.cpp's own
    // KV and compute buffers, vocab, allocator overhead and fragmentation.
    //
    // UNCONDITIONAL, unlike reserve(). A ledger that refuses to record an overrun
    // cannot be used to enforce the cap -- it would report health while the process
    // sat over the line. Recording it is what makes the cache budget shrink to
    // compensate, since the budget is derived as cap minus everything non-cache.
    void charge_unreserved(size_t bytes);
    size_t unreserved() const;

    size_t cap() const { return cap_; }
    // One category's current total. Exists so callers never keep their own mirror
    // of a ledger line: a mirror updated at a different moment than the ledger
    // disagrees with it exactly when the difference matters -- at the cap.
    size_t used_in(Category) const;
    size_t used() const;
    size_t available() const;
    Breakdown breakdown() const;

    // Sum of everything that is not ExpertCache: the mandatory floor plus scratch.
    size_t non_cache() const;

    // Human-readable startup report, one line per non-zero category.
    std::string report() const;

    // Reconciliation against the OS. A widening gap between used() and RSS is a
    // bug, not noise -- over 10^6 seconds fragmentation produces RSS growth with
    // zero leaked bytes. Returns RSS in bytes, or 0 if unavailable.
    static size_t process_rss();

    // Private COMMITTED bytes: what the process has taken from the system and not
    // given back, whether or not it is currently in the working set.
    //
    // Both figures are needed and they diverge. Measured on Qwen3.8 at a 12 GiB cap:
    // working set 11.97 GB, committed 13.59 GB. The 1.6 GB gap is memory the
    // allocator retains across the malloc/free churn of streaming -- not leaked, but
    // not available to anything else either, which is what "leaves the system usable"
    // is about. Binding the cap on the working set alone declared success while the
    // process held 0.7 GB more than the user allowed.
    //
    // Returns 0 where it cannot be obtained (Invariant 6: reported unknown, never
    // defaulted). Only implemented on Windows so far -- the other backends are not
    // built or tested yet, and a plausible-looking wrong number is worse than none.
    static size_t process_committed();

private:
    size_t cap_;
    mutable std::atomic<size_t> counts_[static_cast<size_t>(Category::_Count)];
    // Bytes the OS says we hold that no category claims. Set wholesale by
    // charge_unreserved rather than accumulated, so repeated reconciliation does
    // not double-count the same overhead.
    std::atomic<size_t> unreserved_{0};
};

// Fixed-size slot allocator. One instance per (layer size class), because expert
// tensor sizes are NOT uniform across layers -- MiniMax M3 UD-Q2_K_XL has three
// distinct per-layer sizes (18,137,088 / 20,938,752 / 24,477,696 bytes) and a
// single global slot sized to the largest wastes ~26% of the arena on 54 of 57
// layers. Slot size is read from the GGUF tensor table (Invariant 3), never
// computed from an assumed bits-per-weight.
//
// Slots for one layer are CONTIGUOUS and uniformly strided so the layer's
// ffn_*_exps is a valid ggml tensor with nb[2] = slot_bytes and ne[2] = n_slots;
// ggml_mul_mat_id indexes all experts from one base pointer by one stride, so
// scattered slots cannot be expressed by repointing tensor->data.
class Slab {
public:
    // Reserves n_slots * slot_bytes through the accountant. Check valid().
    Slab(Accountant&, Category, size_t slot_bytes, size_t n_slots, uint32_t align);
    ~Slab();

    Slab(const Slab&) = delete;
    Slab& operator=(const Slab&) = delete;

    bool     valid() const { return base_ != nullptr; }
    void*    slot(size_t i);
    const void* slot(size_t i) const;
    void*    base() { return base_; }
    size_t   slot_bytes() const { return slot_bytes_; }
    size_t   n_slots() const { return n_slots_; }
    size_t   bytes() const { return slot_bytes_ * n_slots_; }

private:
    Accountant& acct_;
    Category    cat_;
    void*       base_ = nullptr;
    size_t      slot_bytes_ = 0;
    size_t      n_slots_ = 0;
};

// Aligned host allocation honouring the storage backend's memory alignment.
void* aligned_alloc_host(size_t bytes, uint32_t align);
// Size is REQUIRED: it selects the same allocator the matching alloc used. Probing
// the pointer to guess (VirtualQuery) would work today and is precisely the kind of
// inferred-semantics assumption that has caused most of the bugs in this engine.
// Every caller knows the size it asked for.
void  aligned_free_host(void*, size_t bytes);

}  // namespace dray::mem
