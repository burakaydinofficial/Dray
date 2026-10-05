// ResidencyCache::take_region: a dead routed-expert region handed over for reuse
// instead of freed and reallocated. It must behave as an eviction in every way
// that matters -- the victim is gone from the cache and points at the sentinel,
// the ledger is unchanged (the charge carries over to the new owner) -- and it
// must never hand over anything an eviction of a routed region would not take.

#include "harness.h"

#include <cstdint>
#include <string>
#include <vector>

#include "backend/accounted_alloc.h"
#include "backend/poison_buffers.h"
#include "backend/residency_cache.h"
#include "backend/tensor_registry.h"
#include "ggml.h"
#include "mem/accountant.h"

using dray::backend::AccountedAlloc;
using dray::backend::PoisonBuffers;
using dray::backend::Prio;
using dray::backend::Resident;
using dray::backend::ResidencyCache;
using dray::backend::TensorRegistry;
namespace mem = dray::mem;

namespace {

constexpr uint64_t kRegion = 3 * 4096;

struct Rig {
    mem::Accountant acct{64ull << 20};
    AccountedAlloc  alloc{acct};
    PoisonBuffers   poison{alloc};
    TensorRegistry  tensors;
    ResidencyCache  cache{alloc, poison, tensors};
    ggml_context*   ctx = nullptr;
    std::vector<ggml_tensor*> ts;
    bool ok = false;

    Rig() {
        std::string err;
        ok = poison.create_sentinel(4096, &err);
        ggml_init_params p{};
        p.mem_size = ggml_tensor_overhead() * 16;
        p.no_alloc = true;
        ctx = ggml_init(p);
        ok = ok && ctx != nullptr;
        for (int i = 0; ok && i < 8; ++i) {
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
            ggml_set_name(t, ("t" + std::to_string(i)).c_str());
            ts.push_back(t);
        }
    }
    ~Rig() {
        cache.release_all();
        if (ctx) ggml_free(ctx);
    }
    // A cached region for ts[i], charged to the ledger like a real one.
    void* add(int i, uint64_t bytes, Prio prio, bool pinned = false) {
        void* m = alloc.alloc(mem::Category::ExpertCache, bytes, 4096);
        Resident r;
        r.mem = m; r.bytes = bytes; r.prio = prio; r.pinned = pinned;
        r.cat = mem::Category::ExpertCache;
        cache.add(ts[static_cast<size_t>(i)], r);
        ts[static_cast<size_t>(i)]->data = m;
        return m;
    }
};

}  // namespace

LZ_TEST(takes_the_least_recent_same_size_routed_region_as_an_eviction) {
    Rig r;
    LZ_REQUIRE(r.ok);
    void* older = r.add(0, kRegion, Prio::RoutedExpert);
    r.add(1, kRegion, Prio::RoutedExpert);
    const size_t used = r.acct.used();

    void* got = r.cache.take_region(kRegion, 4096);
    LZ_CHECK(got == older);                              // least recent first
    LZ_CHECK(r.cache.find(r.ts[0]) == nullptr);          // gone from the cache
    LZ_CHECK(r.ts[0]->data == r.poison.sentinel());      // never left dangling
    LZ_CHECK(r.cache.find(r.ts[1]) != nullptr);          // the other one stays
    LZ_CHECK_EQ(r.acct.used(), used);                    // the charge carried over
    LZ_CHECK_EQ(r.cache.regions_reused(), 1u);
    // The new owner frees it like any region it allocated.
    r.alloc.free(mem::Category::ExpertCache, got, kRegion);
    LZ_CHECK_EQ(r.acct.used(), used - kRegion);
}

LZ_TEST(never_takes_what_a_routed_eviction_would_not) {
    Rig r;
    LZ_REQUIRE(r.ok);
    r.add(0, kRegion, Prio::Unconditional);              // an unconditional weight
    r.add(1, kRegion, Prio::RoutedExpert, true);         // pinned
    r.add(2, kRegion + 4096, Prio::RoutedExpert);        // wrong size
    r.add(3, kRegion, Prio::RoutedExpert);
    r.cache.protect(r.ts[3]);                            // needed by the current node
    LZ_CHECK(r.cache.take_region(kRegion, 4096) == nullptr);
    LZ_CHECK_EQ(r.cache.regions_reused(), 0u);
    for (int i = 0; i < 4; ++i) LZ_CHECK(r.cache.find(r.ts[static_cast<size_t>(i)]) != nullptr);

    r.cache.protect_none();                              // the node is done
    LZ_CHECK(r.cache.take_region(kRegion, 4096) != nullptr);
    LZ_CHECK(r.cache.find(r.ts[3]) == nullptr);
    LZ_CHECK(r.cache.find(r.ts[0]) != nullptr);          // still never the others
    LZ_CHECK(r.cache.find(r.ts[1]) != nullptr);
    LZ_CHECK(r.cache.find(r.ts[2]) != nullptr);
}

LZ_TEST(refuses_a_region_not_aligned_as_asked) {
    Rig r;
    LZ_REQUIRE(r.ok);
    r.add(0, kRegion, Prio::RoutedExpert);
    // Asking for a stricter alignment than the region can be sure to have: a
    // region is only reusable where its placement satisfies the new owner.
    const uintptr_t at = reinterpret_cast<uintptr_t>(r.cache.find(r.ts[0])->mem);
    uint32_t strict = 4096;
    while (strict && at % (strict * 2) == 0) strict *= 2;   // the next power it FAILS
    LZ_CHECK(r.cache.take_region(kRegion, strict * 2) == nullptr);
    LZ_CHECK(r.cache.take_region(kRegion, 4096) != nullptr);
}

LZ_TEST(a_tensor_recycles_its_own_region_for_its_next_routing) {
    // reclaim_own: the replacement of a tensor's region takes the old memory
    // when size and alignment match -- no free, no fresh commit, charge carried.
    Rig r;
    LZ_REQUIRE(r.ok);
    void* old = r.add(0, kRegion, Prio::RoutedExpert);
    r.add(1, kRegion, Prio::RoutedExpert);
    r.cache.protect(r.ts[0]);                            // the current node's own tensor
    const size_t used = r.acct.used();

    void* got = r.cache.reclaim_own(r.ts[0], kRegion, 4096, ResidencyCache::Mismatch::Drop);
    LZ_CHECK(got == old);                                // its own, even while protected
    LZ_CHECK(r.cache.find(r.ts[0]) == nullptr);
    LZ_CHECK(r.ts[0]->data == r.poison.sentinel());
    LZ_CHECK(r.cache.find(r.ts[1]) != nullptr);          // never another tensor's
    LZ_CHECK_EQ(r.acct.used(), used);                    // the charge carried over
    LZ_CHECK_EQ(r.cache.regions_recycled(), 1u);
    r.alloc.free(mem::Category::ExpertCache, got, kRegion);
}

LZ_TEST(a_region_that_does_not_fit_is_freed_as_before) {
    Rig r;
    LZ_REQUIRE(r.ok);
    r.add(0, kRegion, Prio::RoutedExpert);
    const size_t used = r.acct.used();
    // A different size, Keep (a sibling ahead of its node): untouched -- still
    // cached, still charged, nothing handed back.
    LZ_CHECK(r.cache.reclaim_own(r.ts[0], kRegion + 4096, 4096,
                                 ResidencyCache::Mismatch::Keep) == nullptr);
    LZ_CHECK(r.cache.find(r.ts[0]) != nullptr);
    LZ_CHECK_EQ(r.acct.used(), used);
    // Drop (the node's own replacement): freed, exactly what the replacement
    // did before reclaim_own existed.
    LZ_CHECK(r.cache.reclaim_own(r.ts[0], kRegion + 4096, 4096,
                                 ResidencyCache::Mismatch::Drop) == nullptr);
    LZ_CHECK(r.cache.find(r.ts[0]) == nullptr);
    LZ_CHECK_EQ(r.acct.used(), used - kRegion);
    LZ_CHECK_EQ(r.cache.regions_recycled(), 0u);
    // No region at all: nothing to do.
    LZ_CHECK(r.cache.reclaim_own(r.ts[5], kRegion, 4096, ResidencyCache::Mismatch::Drop) == nullptr);
}
