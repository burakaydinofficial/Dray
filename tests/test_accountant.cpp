// The accountant is the single authority on resident bytes (Invariant 1), so a
// defect here is not "a memory bug" -- it is the engine quietly breaking the one
// promise the design makes to a user with 16 GB of RAM. That is why these cases
// assert exact byte figures rather than ranges, and why the concurrency case
// asserts that reserve() grants every byte that fits as well as never granting
// one that does not: a ledger that spuriously refuses would refuse a model that
// fits, and a ledger that transiently overshoots would publish a figure the RSS
// reconciler is required to treat as a breach.

#include "harness.h"

#include "mem/accountant.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using dray::mem::Accountant;
using dray::mem::Breakdown;
using dray::mem::Category;
using dray::mem::Slab;

namespace {

constexpr size_t kCatCount = static_cast<size_t>(Category::_Count);

size_t breakdown_of(const Accountant& a, Category c) {
    return a.breakdown().bytes[static_cast<size_t>(c)];
}

}  // namespace

LZ_TEST(cap_is_reported_verbatim) {
    Accountant a(4ull * 1024 * 1024 * 1024);
    LZ_CHECK_EQ(a.cap(), 4ull * 1024 * 1024 * 1024);
    LZ_CHECK_EQ(a.used(), 0u);
    LZ_CHECK_EQ(a.available(), a.cap());
    LZ_CHECK_EQ(a.non_cache(), 0u);
}

LZ_TEST(reserve_refuses_the_byte_that_would_overshoot) {
    Accountant a(1000);

    LZ_CHECK(a.reserve(Category::RouterGates, 600));
    LZ_CHECK_EQ(a.used(), 600u);
    LZ_CHECK_EQ(a.available(), 400u);

    // Must refuse, and refusing must not partially consume: a caller that
    // handles false by shrinking a budget needs available() to still be true.
    LZ_CHECK(!a.reserve(Category::KvCache, 401));
    LZ_CHECK_EQ(a.used(), 600u);
    LZ_CHECK_EQ(a.available(), 400u);

    LZ_CHECK(a.reserve(Category::KvCache, 400));
    LZ_CHECK_EQ(a.used(), 1000u);
    LZ_CHECK_EQ(a.available(), 0u);

    LZ_CHECK(!a.reserve(Category::Misc, 1));
    LZ_CHECK_EQ(a.used(), 1000u);

    // Exactly at the cap is inside the cap.
    LZ_CHECK(a.reserve(Category::Misc, 0));
    LZ_CHECK_EQ(a.used(), 1000u);
}

LZ_TEST(release_returns_bytes_to_the_budget) {
    Accountant a(1000);
    LZ_REQUIRE(a.reserve(Category::ComputeScratch, 750));
    LZ_CHECK_EQ(a.used(), 750u);

    a.release(Category::ComputeScratch, 250);
    LZ_CHECK_EQ(a.used(), 500u);
    LZ_CHECK_EQ(a.available(), 500u);
    LZ_CHECK_EQ(breakdown_of(a, Category::ComputeScratch), 500u);

    // Released bytes are usable again by a different category.
    LZ_CHECK(a.reserve(Category::IoStaging, 500));
    LZ_CHECK_EQ(a.used(), 1000u);

    a.release(Category::ComputeScratch, 500);
    a.release(Category::IoStaging, 500);
    LZ_CHECK_EQ(a.used(), 0u);
    for (size_t i = 0; i < kCatCount; ++i) {
        LZ_CHECK_EQ(a.breakdown().bytes[i], 0u);
    }
}

LZ_TEST(breakdown_totals_match_used_and_non_cache_excludes_the_slab) {
    Accountant a(1ull << 30);
    LZ_REQUIRE(a.reserve(Category::RouterGates, 1'500'000));
    LZ_REQUIRE(a.reserve(Category::NormsAndBiases, 7'000));
    LZ_REQUIRE(a.reserve(Category::KvCache, 900'000));
    LZ_REQUIRE(a.reserve(Category::ExpertCache, 40'000'000));

    const Breakdown b = a.breakdown();
    LZ_CHECK_EQ(b.bytes[static_cast<size_t>(Category::RouterGates)], 1'500'000u);
    LZ_CHECK_EQ(b.bytes[static_cast<size_t>(Category::NormsAndBiases)], 7'000u);
    LZ_CHECK_EQ(b.bytes[static_cast<size_t>(Category::KvCache)], 900'000u);
    LZ_CHECK_EQ(b.bytes[static_cast<size_t>(Category::ExpertCache)], 40'000'000u);
    LZ_CHECK_EQ(b.total(), a.used());

    // Gates and norms are FLOOR, not CACHE -- the two must never be
    // double-counted (this was a real bug in an earlier draft of the design).
    LZ_CHECK_EQ(a.non_cache(), a.used() - 40'000'000u);
    LZ_CHECK_EQ(a.non_cache(), 2'407'000u);
}

LZ_TEST(every_category_has_a_name_and_shows_up_in_the_report) {
    for (size_t i = 0; i < kCatCount; ++i) {
        const char* n = dray::mem::category_name(static_cast<Category>(i));
        LZ_REQUIRE(n != nullptr);
        LZ_CHECK(std::strlen(n) > 0);
    }

    // A user whose 4 GB went somewhere unexpected has to be able to see where;
    // on the big models the router-gate table alone is 1.5-2.4 GB.
    Accountant a(1ull << 30);
    LZ_REQUIRE(a.reserve(Category::RouterGates, 1'500'000));
    const std::string r = a.report();
    LZ_CHECK(!r.empty());
    LZ_CHECK_CONTAINS(r, dray::mem::category_name(Category::RouterGates));
}

LZ_TEST(concurrent_reserve_never_overshoots_the_cap) {
    constexpr size_t kCap = 100'000;
    constexpr size_t kChunk = 97;  // coprime with the cap: the last grant is partial-fit
    constexpr int kThreads = 8;
    constexpr int kAttempts = 400;  // 8 * 400 * 97 = 310,400 >> cap

    Accountant a(kCap);

    std::atomic<size_t> granted{0};
    std::atomic<size_t> peak_seen{0};
    std::atomic<bool> done{false};

    // used() is the figure reconciled against OS RSS on a timer, so a value
    // above the cap is observable and would be reported as a breach even if it
    // is rolled back a microsecond later.
    std::thread watchdog([&]() {
        while (!done.load(std::memory_order_relaxed)) {
            const size_t u = a.used();
            size_t prev = peak_seen.load(std::memory_order_relaxed);
            while (u > prev && !peak_seen.compare_exchange_weak(prev, u)) {
            }
        }
    });

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&]() {
            size_t local = 0;
            for (int i = 0; i < kAttempts; ++i) {
                if (a.reserve(Category::ExpertCache, kChunk)) {
                    local += kChunk;
                }
            }
            granted.fetch_add(local, std::memory_order_relaxed);
        });
    }
    for (std::thread& w : workers) {
        w.join();
    }
    done.store(true, std::memory_order_relaxed);
    watchdog.join();

    LZ_CHECK_LE(a.used(), kCap);
    LZ_CHECK_LE(peak_seen.load(), kCap);
    LZ_CHECK_EQ(a.used(), granted.load());
    LZ_CHECK_EQ(breakdown_of(a, Category::ExpertCache), granted.load());
}

LZ_TEST(concurrent_reserve_grants_every_byte_that_fits) {
    // The complement of the case above. A fetch_add-then-roll-back
    // implementation refuses reservations that DO fit whenever two threads race,
    // and a spurious refusal at load time means refusing a model that fits.
    // A compare-exchange loop grants exactly floor(cap / chunk) reservations.
    constexpr size_t kCap = 100'000;
    constexpr size_t kChunk = 97;
    constexpr int kThreads = 8;
    constexpr int kAttempts = 400;

    Accountant a(kCap);
    std::atomic<int> successes{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&]() {
            int local = 0;
            for (int i = 0; i < kAttempts; ++i) {
                if (a.reserve(Category::ExpertCache, kChunk)) {
                    ++local;
                }
            }
            successes.fetch_add(local, std::memory_order_relaxed);
        });
    }
    for (std::thread& w : workers) {
        w.join();
    }

    const int expected = static_cast<int>(kCap / kChunk);  // 1030
    LZ_CHECK_EQ(successes.load(), expected);
    LZ_CHECK_EQ(a.used(), static_cast<size_t>(expected) * kChunk);
    LZ_CHECK_LT(a.available(), kChunk);
}

LZ_TEST(concurrent_reserve_and_release_balances_to_zero) {
    constexpr int kThreads = 8;
    constexpr int kRounds = 500;
    Accountant a(64 * 1024);

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&]() {
            for (int i = 0; i < kRounds; ++i) {
                if (a.reserve(Category::IoStaging, 512)) {
                    a.release(Category::IoStaging, 512);
                }
            }
        });
    }
    for (std::thread& w : workers) {
        w.join();
    }

    LZ_CHECK_EQ(a.used(), 0u);
    LZ_CHECK_EQ(a.available(), a.cap());
}

LZ_TEST(slab_slots_are_contiguous_and_strided_by_slot_bytes) {
    // 110,592 B is the smallest of the fixture's three per-layer expert sizes.
    // The stride assertion is the load-bearing one: the layer's ffn_*_exps is a
    // ggml tensor with nb[2] = slot_bytes and ne[2] = n_slots, so slots must be
    // packed at exactly slot_bytes -- NOT individually rounded up to `align`.
    constexpr size_t kSlot = 110'592;
    constexpr size_t kSlots = 6;

    Accountant a(16ull << 20);
    Slab s(a, Category::ExpertCache, kSlot, kSlots, 4096);
    LZ_REQUIRE(s.valid());

    LZ_CHECK_EQ(s.slot_bytes(), kSlot);
    LZ_CHECK_EQ(s.n_slots(), kSlots);
    LZ_CHECK_EQ(s.bytes(), kSlot * kSlots);
    LZ_REQUIRE(s.base() != nullptr);

    char* base = static_cast<char*>(s.base());
    for (size_t i = 0; i < kSlots; ++i) {
        char* p = static_cast<char*>(s.slot(i));
        LZ_CHECK_EQ(p - base, static_cast<ptrdiff_t>(i * kSlot));
    }

    // Whole arena is writable: a slot that overlaps the next one shows up here.
    std::memset(base, 0xAB, s.bytes());
    LZ_CHECK_EQ(static_cast<unsigned char>(base[s.bytes() - 1]), 0xABu);

    const Slab& cs = s;
    LZ_CHECK_EQ(cs.slot(3), static_cast<const void*>(base + 3 * kSlot));
}

LZ_TEST(slab_stride_is_not_rounded_up_to_alignment) {
    // Slot sizes come from the GGUF tensor table (Invariant 3) and are not
    // multiples of anything in particular. Rounding the stride up to `align`
    // would silently break nb[2].
    constexpr size_t kSlot = 1000;
    constexpr size_t kSlots = 7;

    Accountant a(1 << 20);
    Slab s(a, Category::ExpertCache, kSlot, kSlots, 4096);
    LZ_REQUIRE(s.valid());
    LZ_CHECK_EQ(s.bytes(), 7000u);

    char* base = static_cast<char*>(s.base());
    LZ_CHECK_EQ(static_cast<char*>(s.slot(1)) - base, static_cast<ptrdiff_t>(kSlot));
    LZ_CHECK_EQ(static_cast<char*>(s.slot(6)) - base, static_cast<ptrdiff_t>(6 * kSlot));
}

LZ_TEST(slab_base_honours_the_requested_alignment) {
    // The base is the destination of uncached reads, so it must satisfy the
    // storage backend's memory alignment -- a runtime value (Invariant 5).
    Accountant a(4 << 20);
    for (uint32_t align : {64u, 512u, 4096u}) {
        Slab s(a, Category::ExpertCache, 4096, 4, align);
        LZ_REQUIRE(s.valid());
        const uintptr_t addr = reinterpret_cast<uintptr_t>(s.base());
        LZ_CHECK_EQ(addr % align, 0u);
    }
}

LZ_TEST(slab_accounts_its_bytes_and_gives_them_back) {
    Accountant a(16ull << 20);
    const size_t before = a.used();
    {
        Slab s(a, Category::ExpertCache, 65'536, 16, 4096);
        LZ_REQUIRE(s.valid());
        LZ_CHECK_GE(a.used(), before + s.bytes());
        // Alignment padding is allowed; anything beyond that is unaccounted.
        LZ_CHECK_LE(a.used(), before + s.bytes() + 4096);
        LZ_CHECK_GE(breakdown_of(a, Category::ExpertCache), s.bytes());
    }
    LZ_CHECK_EQ(a.used(), before);
    LZ_CHECK_EQ(breakdown_of(a, Category::ExpertCache), 0u);
}

LZ_TEST(slab_fails_cleanly_when_the_cap_is_exhausted) {
    constexpr size_t kCap = 1 << 20;
    Accountant a(kCap);
    LZ_REQUIRE(a.reserve(Category::KvCache, kCap - 4096));
    const size_t before = a.used();

    // Refuse, do not shrink, do not allocate, do not crash: the caller's only
    // correct response to a cap it cannot meet is refusing admission.
    Slab s(a, Category::ExpertCache, 8192, 16, 4096);
    LZ_CHECK(!s.valid());
    LZ_CHECK(s.base() == nullptr);
    LZ_CHECK_EQ(a.used(), before);
    LZ_CHECK_EQ(breakdown_of(a, Category::ExpertCache), 0u);

    // And the accountant is still usable afterwards.
    LZ_CHECK(a.reserve(Category::Misc, 4096));
    LZ_CHECK_EQ(a.used(), kCap);
}

LZ_TEST(destroying_a_failed_slab_does_not_release_bytes_it_never_took) {
    Accountant a(4096);
    LZ_REQUIRE(a.reserve(Category::Misc, 4096));
    {
        Slab s(a, Category::ExpertCache, 1024, 8, 64);
        LZ_CHECK(!s.valid());
    }
    // A destructor that released unconditionally would underflow the ledger
    // here, and an underflowed size_t reads as ~18 exabytes resident.
    LZ_CHECK_EQ(a.used(), 4096u);
    LZ_CHECK_LE(a.used(), a.cap());
}

LZ_TEST(aligned_alloc_host_honours_alignment_and_frees) {
    for (uint32_t align : {64u, 512u, 4096u}) {
        void* p = dray::mem::aligned_alloc_host(3u * align + 17u, align);
        LZ_REQUIRE(p != nullptr);
        LZ_CHECK_EQ(reinterpret_cast<uintptr_t>(p) % align, 0u);
        std::memset(p, 0x5A, 3u * align + 17u);
        dray::mem::aligned_free_host(p, 3u * align + 17u);
    }
    // Freeing nothing must be safe: error paths reach this with a null pointer.
    dray::mem::aligned_free_host(nullptr, 0);
}

LZ_TEST(process_rss_is_either_unavailable_or_plausible) {
    const size_t rss = dray::mem::Accountant::process_rss();
    // Invariant 6: anything unobtainable is reported unknown, never defaulted.
    // 0 means unknown; any other value must be at least a plausible process.
    if (rss != 0) {
        LZ_CHECK_GT(rss, 256u * 1024u);
        LZ_CHECK_LT(rss, 1ull << 42);
    }
    LZ_CHECK_EQ(dray::mem::Accountant::process_rss() == 0, rss == 0);
}

// S18: the reserve gate must bind on TOTAL resident truth, including the
// unreserved charge the RSS reconcile maintains -- not on category counters
// alone, which approved reservations past the cap at peak pressure.
LZ_TEST(reserve_counts_the_unreserved_charge) {
    dray::mem::Accountant a(1000);
    LZ_CHECK(a.reserve(dray::mem::Category::IoStaging, 300));
    a.charge_unreserved(600);
    LZ_CHECK_EQ(a.used(), 900u);
    LZ_CHECK(!a.reserve(dray::mem::Category::IoStaging, 200));  // 1100 > cap
    LZ_CHECK(a.reserve(dray::mem::Category::IoStaging, 100));   // exactly cap
    LZ_CHECK_EQ(a.used(), 1000u);
    a.charge_unreserved(0);
    LZ_CHECK_EQ(a.used(), 400u);
    LZ_CHECK(a.reserve(dray::mem::Category::IoStaging, 600));
}
