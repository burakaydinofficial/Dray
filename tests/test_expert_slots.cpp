// ExpertSlots admission budget: within a pass, slots may take at most
// admit_byte_fraction of the bytes the pass has read (every offer is an expert
// just read from disk). It replaced a fixed 32 admissions per pass -- a count
// that only meant something at one model's slot size -- and, unlike a time
// budget, keeps the decisions (and so bytes read) deterministic.

#include "harness.h"

#include <cstdint>
#include <vector>

#include "backend/accounted_alloc.h"
#include "backend/expert_slots.h"
#include "backend/poison_buffers.h"
#include "backend/residency_cache.h"
#include "backend/routing_skew.h"
#include "backend/tensor_registry.h"
#include "ggml.h"
#include "mem/accountant.h"

using dray::backend::AccountedAlloc;
using dray::backend::ExpertSlots;
using dray::backend::PoisonBuffers;
using dray::backend::ResidencyCache;
using dray::backend::RoutingSkew;
using dray::backend::TensorRegistry;
namespace mem = dray::mem;

namespace {

constexpr uint64_t kStride = 64 * 1024;
constexpr int32_t kExperts = 256;

struct Rig {
    mem::Accountant acct{64ull << 20};
    AccountedAlloc  alloc{acct};
    PoisonBuffers   poison{alloc};
    TensorRegistry  tensors;
    ResidencyCache  cache{alloc, poison, tensors};
    RoutingSkew     skew;
    ExpertSlots     slots;
    ggml_tensor     w{};                       // only its address is used
    std::vector<uint8_t> src = std::vector<uint8_t>(kStride, 0x5a);

    explicit Rig(double fraction) : slots(alloc, cache, skew, false, fraction) {
        slots.set_pool(32ull << 20);
        std::vector<int32_t> all;
        for (int32_t e = 0; e < kExperts; ++e) all.push_back(e);
        skew.record(&w, all, kExperts);        // every expert seen twice:
        skew.record(&w, all, kExperts);        // past the doorkeeper
    }
    ~Rig() { slots.release_all(); }
    // Offers experts [first, first + n); returns how many were admitted.
    uint64_t offer(int32_t first, int32_t n) {
        const uint64_t before = slots.admissions();
        for (int32_t e = first; e < first + n; ++e) slots.admit(&w, e, src.data(), kStride);
        return slots.admissions() - before;
    }
};

}  // namespace

LZ_TEST(a_pass_admits_its_share_of_what_it_read) {
    // 40 experts read at 10%: 4 experts' worth of budget, earned one tenth at
    // a time -- so exactly 4 admissions, and every other offer is declined.
    Rig r(0.10);
    r.slots.on_pass();
    LZ_CHECK_EQ(r.offer(0, 40), 4u);
    LZ_CHECK_EQ(r.slots.declined_for_budget(), 36u);
    LZ_CHECK_EQ(r.slots.bytes(), 4 * kStride);
}

LZ_TEST(a_new_pass_starts_a_fresh_budget) {
    Rig r(0.10);
    r.slots.on_pass();
    LZ_CHECK_EQ(r.offer(0, 40), 4u);
    r.slots.on_pass();
    LZ_CHECK_EQ(r.offer(0, 9), 0u);            // 0.9 of one expert: not yet
    LZ_CHECK_EQ(r.offer(100, 1), 1u);          // the tenth read pays for one
    LZ_CHECK_EQ(r.slots.admissions(), 5u);
}

LZ_TEST(the_budget_does_not_carry_across_passes) {
    // Unspent budget is not banked: a pass that read a lot and admitted little
    // must not license a storm in the next one.
    Rig r(0.10);
    r.slots.on_pass();
    for (int i = 0; i < 100; ++i) r.slots.admit(&r.w, -1, r.src.data(), kStride);   // reads, none admissible
    r.slots.on_pass();
    LZ_CHECK_EQ(r.offer(0, 5), 0u);
}

LZ_TEST(the_fraction_is_the_policy) {
    Rig half(0.50);
    half.slots.on_pass();
    LZ_CHECK_EQ(half.offer(0, 40), 20u);

    Rig none(0.0);
    none.slots.on_pass();
    LZ_CHECK_EQ(none.offer(0, 10), 0u);
    LZ_CHECK_EQ(none.slots.declined_for_budget(), 10u);
}

LZ_TEST(identical_offers_make_identical_decisions) {
    // Determinism, the reason this is a byte budget and not a time budget.
    Rig a(0.10), b(0.10);
    for (int pass = 0; pass < 3; ++pass) {
        a.slots.on_pass();
        b.slots.on_pass();
        a.offer(pass * 30, 30);
        b.offer(pass * 30, 30);
    }
    LZ_CHECK_EQ(a.slots.admissions(), b.slots.admissions());
    LZ_CHECK_EQ(a.slots.bytes(), b.slots.bytes());
    for (int32_t e = 0; e < 90; ++e) {
        LZ_CHECK_EQ(a.slots.hit(&a.w, e) != nullptr, b.slots.hit(&b.w, e) != nullptr);
    }
}

// --- evictable slots: the pool gives memory back under pressure ------------

namespace {

using dray::backend::Prio;
using dray::backend::Resident;

// A Rig with the pool wired into the cache as its spill source and an
// admission fraction of 1 (admit whatever was read).
struct SpillRig : Rig {
    SpillRig() : Rig(1.0) { cache.set_spill(&slots); }
    // Offers expert e `picks` more times in the histogram, making it hotter.
    void heat(int32_t e, int picks) {
        for (int i = 0; i < picks; ++i) skew.record(&w, std::vector<int32_t>{ e }, kExperts);
    }
    void* add_entry(ggml_tensor* t, uint64_t bytes, Prio prio) {
        void* m = alloc.alloc(mem::Category::ExpertCache, bytes, 4096);
        Resident r;
        r.mem = m; r.bytes = bytes; r.prio = prio;
        cache.add(t, r);
        t->data = m;
        return m;
    }
};

}  // namespace

LZ_TEST(release_frees_the_coldest_slots_first) {
    SpillRig r;
    r.slots.on_pass();
    r.offer(0, 8);                             // experts 0..7 admitted
    LZ_REQUIRE_EQ(r.slots.admissions(), 8u);
    for (int32_t e = 4; e < 8; ++e) r.heat(e, 5);   // 4..7 are hot
    const uint64_t freed = r.slots.release(3 * kStride);
    LZ_CHECK_EQ(freed, 3 * kStride);
    LZ_CHECK_EQ(r.slots.released(), 3u);
    int cold_left = 0, hot_left = 0;
    for (int32_t e = 0; e < 4; ++e) cold_left += r.slots.hit(&r.w, e) ? 1 : 0;
    for (int32_t e = 4; e < 8; ++e) hot_left += r.slots.hit(&r.w, e) ? 1 : 0;
    LZ_CHECK_EQ(cold_left, 1);                 // three of the cold four went
    LZ_CHECK_EQ(hot_left, 4);                  // every hot one stayed
    LZ_CHECK_EQ(r.slots.bytes(), 5 * kStride);
}

LZ_TEST(make_room_asks_the_pool_before_unconditional_weights) {
    SpillRig r;
    r.slots.on_pass();
    r.offer(0, 32);                            // 2 MiB of slots
    ggml_tensor uncond{};
    r.add_entry(&uncond, 8 * kStride, Prio::Unconditional);
    // Fill the rest of the budget so a new region cannot fit without eviction.
    const uint64_t budget = r.alloc.cache_budget(), used = r.alloc.cache_used();
    ggml_tensor filler{};
    r.add_entry(&filler, budget - used, Prio::Unconditional);
    filler.data = nullptr;
    r.cache.protect(&filler);                  // in use by the node being computed
    LZ_REQUIRE(r.cache.make_room(4 * kStride));
    LZ_CHECK_GE(r.slots.released_bytes(), 4 * kStride);   // the pool gave it back
    LZ_CHECK(r.cache.find(&uncond) != nullptr);             // the unconditional weight stayed
}

LZ_TEST(admission_never_frees_slots_to_make_room_for_a_slot) {
    SpillRig r;
    r.slots.on_pass();
    r.offer(0, 16);
    const uint64_t before = r.slots.admissions();
    // Budget full: the next admission needs room, and must not get it from slots.
    ggml_tensor filler{};
    r.add_entry(&filler, r.alloc.cache_budget() - r.alloc.cache_used(), Prio::Unconditional);
    r.cache.protect(&filler);
    r.offer(16, 4);
    LZ_CHECK_EQ(r.slots.released(), 0u);
    LZ_CHECK_EQ(r.slots.admissions(), before);   // declined instead
}
