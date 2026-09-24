// PoolArena: the properties the streamer depends on, weight-free.
//
// What matters: blocks are writable and page-aligned; exact-size reuse returns
// the same offsets instead of growing the bump; commit tracks live bytes so
// the RSS reconcile keeps meaning what it says; VA exhaustion returns nullptr
// (the caller's fallback) rather than corrupting; contains() discriminates
// pool pointers from host pointers, because teardown routes frees by origin.

#include "harness.h"

#include <cstdint>
#include <cstring>

#include "mem/pool_arena.h"

using dray::mem::PoolArena;

LZ_TEST(disabled_pool_declines_and_init_is_once) {
    PoolArena p;
    LZ_CHECK(!p.enabled());
    LZ_CHECK(p.alloc(4096, 64) == nullptr);
    LZ_CHECK(p.init(1ull << 24));
    LZ_CHECK(p.enabled());
    LZ_CHECK(!p.init(1ull << 20));
}

LZ_TEST(blocks_are_writable_page_aligned_and_owned) {
    PoolArena p;
    LZ_CHECK(p.init(1ull << 24));
    void* a = p.alloc(10000, 64);
    LZ_CHECK(a != nullptr);
    LZ_CHECK_EQ(reinterpret_cast<uintptr_t>(a) & (p.page() - 1), 0u);   // T27: page-derived
    LZ_CHECK(p.contains(a));
    int on_stack = 0;
    LZ_CHECK(!p.contains(&on_stack));
    std::memset(a, 0xAB, 10000);   // writable end to end or we crash here
    // Page-derived, not hardcoded 4096: Apple silicon uses 16K pages (swarm S19).
    const uint64_t span = (10000 + p.page() - 1) / p.page() * p.page();
    LZ_CHECK_EQ(p.committed(), span);
    p.free(a, 10000);
    LZ_CHECK_EQ(p.committed(), 0u);
}

LZ_TEST(exact_size_reuse_and_commit_tracks_live_bytes) {
    PoolArena p;
    LZ_CHECK(p.init(1ull << 24));
    void* a = p.alloc(10000, 64);
    void* b = p.alloc(10000, 64);
    LZ_CHECK(a != nullptr);
    LZ_CHECK(b != nullptr);
    LZ_CHECK(b != a);
    p.free(a, 10000);
    const uint64_t span2 = (10000 + p.page() - 1) / p.page() * p.page();
    LZ_CHECK_EQ(p.committed(), span2);   // only b remains committed
    void* c = p.alloc(10000, 64);
    LZ_CHECK(c == a);                     // freed block reused, bump untouched
    std::memset(c, 0xCD, 10000);          // recommitted memory writable again
    p.free(b, 10000);
    p.free(c, 10000);
    LZ_CHECK_EQ(p.committed(), 0u);
}

LZ_TEST(va_exhaustion_returns_null_then_reuses) {
    PoolArena p;
    LZ_CHECK(p.init(1ull << 16));   // 64 KiB on purpose
    void* a = p.alloc(40000, 64);
    LZ_CHECK(a != nullptr);
    LZ_CHECK(p.alloc(40000, 64) == nullptr);   // exhausted: caller falls back
    p.free(a, 40000);
    void* c = p.alloc(40000, 64);
    LZ_CHECK(c == a);               // exhausted pool still serves freed blocks
}
