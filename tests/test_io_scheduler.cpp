// IoScheduler against a scripted fake backend: every read path, every failure.
//
// These are the cases no model run can produce on demand -- completions out of
// order, short reads inside the data, EIO, a backend that dies with reads still
// outstanding -- and they are where the F2/H15 rules live: never free memory a
// read may still be writing (leak it, loudly), and free everything else. Each
// case checks the bytes that arrived AND the ledger afterwards, because a path
// that returns the right answer while leaking or double-freeing staging is
// still broken.
//
// Written before the I/O thread exists, against today's single-threaded
// scheduler, so the suite is known to pass on correct code; the threaded
// scheduler must then pass it unchanged.

#include "harness.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "backend/accounted_alloc.h"
#include "backend/io_scheduler.h"
#include "fixtures/fake_backend.h"
#include "mem/accountant.h"

using dray::backend::AccountedAlloc;
using dray::backend::IoScheduler;
using dray::backend::Slice;
using dray::backend::Source;
using dray::test::FakeBackend;
namespace mem = dray::mem;

namespace {

constexpr uint64_t kFileBytes = 8ull << 20;   // 8 MiB per fake shard

struct Rig {
    mem::Accountant acct{1ull << 30};
    AccountedAlloc alloc{acct};
    std::atomic<uint64_t> failures{0};
    IoScheduler io{alloc, failures};
    FakeBackend* fake = nullptr;

    explicit Rig(FakeBackend::Script s, size_t shards = 2, bool threaded = false,
                 uint64_t file_bytes = kFileBytes) {
        std::vector<std::pair<std::string, uint64_t>> files;
        std::vector<std::string> paths;
        for (size_t i = 0; i < shards; ++i) {
            files.emplace_back("shard" + std::to_string(i), file_bytes);
            paths.push_back("shard" + std::to_string(i));
        }
        auto b = std::make_unique<FakeBackend>(files, s);
        fake = b.get();
        std::string err;
        ok = io.open_with(std::move(b), paths, &err);
        if (ok && threaded) {
            io.start_thread();
            ok = io.threaded();
        }
    }
    size_t staging() const { return acct.used_in(mem::Category::IoStaging); }
    bool ok = false;
};

Source src(int32_t shard, uint64_t offset, uint64_t bytes) {
    Source s;
    s.shard = shard;
    s.offset = offset;
    s.bytes = bytes;
    return s;
}

// True when buf holds exactly [offset, offset+n) of shard `shard`.
bool matches(const uint8_t* buf, int32_t shard, uint64_t offset, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        if (buf[i] != FakeBackend::byte_at(shard, offset + i)) return false;
    }
    return true;
}

}  // namespace

LZ_TEST(read_exact_at_unaligned_offsets_and_lengths) {
    Rig r({});
    LZ_REQUIRE(r.ok);
    std::mt19937_64 rng(7);
    for (int i = 0; i < 200; ++i) {
        const uint64_t len = 1 + rng() % (256 * 1024);
        const uint64_t off = rng() % (kFileBytes - len);
        const int32_t sh = static_cast<int32_t>(rng() % 2);
        std::vector<uint8_t> buf(len);
        LZ_REQUIRE(r.io.read_exact(src(sh, off, len), buf.data(), len));
        LZ_CHECK(matches(buf.data(), sh, off, len));
    }
    LZ_CHECK_EQ(r.fake->violations(), 0u);   // every request widened to alignment
    LZ_CHECK_EQ(r.staging(), 0u);            // every staging buffer returned
}

LZ_TEST(batch_deeper_than_the_queue_completes_out_of_order) {
    FakeBackend::Script s;
    s.queue_depth = 4;
    Rig r(s);
    LZ_REQUIRE(r.ok);
    std::vector<std::vector<uint8_t>> bufs(64);
    std::vector<Slice> slices;
    for (size_t i = 0; i < bufs.size(); ++i) {
        const uint64_t len = 5000 + i * 97;
        const uint64_t off = i * 100003;
        bufs[i].resize(len);
        slices.push_back({src(static_cast<int32_t>(i % 2), off, len), bufs[i].data(), len});
    }
    LZ_REQUIRE(r.io.read_batch(slices));
    for (size_t i = 0; i < bufs.size(); ++i) {
        LZ_CHECK(matches(bufs[i].data(), static_cast<int32_t>(i % 2), i * 100003, bufs[i].size()));
    }
    LZ_CHECK_EQ(r.staging(), 0u);
}

LZ_TEST(short_at_end_of_file_is_legal) {
    // The widened span runs past EOF; the device returns less than the span but
    // all of the data asked for. That is a success, not a short read.
    Rig r({});
    LZ_REQUIRE(r.ok);
    const uint64_t len = 3000;
    const uint64_t off = kFileBytes - len;
    std::vector<uint8_t> buf(len);
    LZ_REQUIRE(r.io.read_exact(src(1, off, len), buf.data(), len));
    LZ_CHECK(matches(buf.data(), 1, off, len));
}

LZ_TEST(short_inside_the_data_fails_and_frees) {
    FakeBackend::Script s;
    s.short_rate = 1.0;
    Rig r(s);
    LZ_REQUIRE(r.ok);
    std::vector<uint8_t> buf(100000);
    LZ_CHECK(!r.io.read_exact(src(0, 12345, buf.size()), buf.data(), buf.size()));
    LZ_CHECK_EQ(r.staging(), 0u);   // the read completed: its staging is ours again
}

LZ_TEST(short_read_is_judged_against_lead_in_plus_data) {
    // The request is widened backwards to alignment, so the data starts `head`
    // bytes into the transfer. A device that returns more than the data length
    // but less than head + data has NOT delivered the data. Offset 2*4096+3000
    // gives head 3000; 2000 bytes wanted; the device returns 4000.
    FakeBackend::Script s;
    s.cap_bytes = 4000;
    Rig r(s);
    LZ_REQUIRE(r.ok);
    std::vector<uint8_t> buf(2000);
    LZ_CHECK(!r.io.read_exact(src(0, 2 * 4096 + 3000, 2000), buf.data(), 2000));
    LZ_CHECK_EQ(r.staging(), 0u);
}

LZ_TEST(an_error_fails_the_batch_and_frees_everything) {
    FakeBackend::Script s;
    s.eio_rate = 0.3;
    s.queue_depth = 6;
    Rig r(s);
    LZ_REQUIRE(r.ok);
    std::vector<std::vector<uint8_t>> bufs(40, std::vector<uint8_t>(20000));
    std::vector<Slice> slices;
    for (size_t i = 0; i < bufs.size(); ++i) {
        slices.push_back({src(0, i * 50000, 20000), bufs[i].data(), 20000});
    }
    LZ_CHECK(!r.io.read_batch(slices));
    LZ_CHECK_EQ(r.staging(), 0u);   // failed reads LANDED, so nothing is leaked
}

LZ_TEST(dead_backend_leaks_staging_rather_than_freeing_under_dma) {
    FakeBackend::Script s;
    s.queue_depth = 8;
    s.die_after = 3;     // three reads come back, the rest never do
    Rig r(s);
    LZ_REQUIRE(r.ok);
    std::vector<std::vector<uint8_t>> bufs(10, std::vector<uint8_t>(10000));
    std::vector<Slice> slices;
    for (size_t i = 0; i < bufs.size(); ++i) {
        slices.push_back({src(0, i * 40000, 10000), bufs[i].data(), 10000});
    }
    LZ_CHECK(!r.io.read_batch(slices));
    // The outstanding reads' staging is still a DMA target: it must stay charged
    // (leaked, visibly), never returned for reuse.
    LZ_CHECK_GT(r.staging(), 0u);
}

LZ_TEST(settle_distinguishes_a_dead_backend_from_a_failed_read) {
    {   // reads that land with an error: failed, NOT dead, memory ours again
        FakeBackend::Script s;
        s.eio_rate = 1.0;
        Rig r(s);
        LZ_REQUIRE(r.ok);
        std::vector<uint8_t> buf(8192);
        std::vector<uint64_t> tags{r.io.submit_staged(src(0, 0, 8192), buf.data(), 8192)};
        bool dead = true;
        LZ_CHECK(!r.io.settle(tags, &dead));
        LZ_CHECK(!dead);
        LZ_CHECK(tags.empty());
        LZ_CHECK_EQ(r.staging(), 0u);
    }
    {   // reads that never come back: failed AND dead, staging leaked
        FakeBackend::Script s;
        s.die_after = 0;
        Rig r(s);
        LZ_REQUIRE(r.ok);
        std::vector<uint8_t> buf(8192);
        std::vector<uint64_t> tags{r.io.submit_staged(src(0, 0, 8192), buf.data(), 8192)};
        bool dead = false;
        LZ_CHECK(!r.io.settle(tags, &dead));
        LZ_CHECK(dead);
        LZ_CHECK(tags.empty());
        LZ_CHECK_GT(r.staging(), 0u);
    }
}

LZ_TEST(unstaged_read_is_judged_by_the_bytes_it_must_cover) {
    // The ring's shape: a chunk read straight into an aligned arena, required to
    // deliver only `bytes` (its tail may legally fall past EOF).
    Rig r({});
    LZ_REQUIRE(r.ok);
    const uint32_t len = 16384;
    std::vector<uint8_t> raw(len + 4096);
    uint8_t* dst = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(raw.data()) + 4095) & ~uintptr_t(4095));
    const dray::io::FileId f = r.io.file(0);
    std::vector<uint64_t> tags{r.io.submit_unstaged(f, 8192, len, dst, 0, len)};
    LZ_REQUIRE(tags[0] != 0);
    LZ_CHECK(r.io.settle(tags));
    LZ_CHECK(matches(dst, 0, 8192, len));
}

LZ_TEST(urgent_reads_reach_the_device_before_queued_read_ahead) {
    // Two slots. Four ring reads (Background) take both and queue two more; then
    // a read something is waiting on (Urgent) arrives. When slots free up, the
    // urgent read must go first -- read-ahead never delays a node's own read.
    FakeBackend::Script s;
    s.queue_depth = 2;
    s.shuffle = false;
    Rig r(s);
    LZ_REQUIRE(r.ok);
    const uint32_t len = 4096;
    std::vector<uint8_t> raw(8 * len + 4096);
    uint8_t* arena = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(raw.data()) + 4095) & ~uintptr_t(4095));
    const dray::io::FileId f = r.io.file(0);
    std::vector<uint64_t> bg;
    for (int i = 0; i < 4; ++i) {
        bg.push_back(r.io.submit_unstaged(f, uint64_t(i) * len, len, arena + i * len, 0, len,
                                          dray::backend::IoPriority::Background));
    }
    std::vector<uint8_t> want(1000);
    std::vector<uint64_t> urgent{r.io.submit_staged(src(1, 5000, 1000), want.data(), 1000)};
    LZ_REQUIRE(urgent[0] != 0);
    const std::vector<uint64_t> bg_tags = bg;          // settle() consumes its vector
    const uint64_t urgent_tag = urgent[0];
    LZ_CHECK(r.io.settle(urgent));
    LZ_CHECK(matches(want.data(), 1, 5000, 1000));
    LZ_CHECK(r.io.settle(bg));
    for (int i = 0; i < 4; ++i) LZ_CHECK(matches(arena + i * len, 0, uint64_t(i) * len, len));
    const auto& log = r.fake->submit_log();
    LZ_REQUIRE_EQ(log.size(), 5u);
    LZ_CHECK_EQ(log[0], bg_tags[0]);   // took the free slots at enqueue
    LZ_CHECK_EQ(log[1], bg_tags[1]);
    LZ_CHECK_EQ(log[2], urgent_tag);   // jumped the two queued read-aheads
    LZ_CHECK_EQ(log[3], bg_tags[2]);
    LZ_CHECK_EQ(log[4], bg_tags[3]);
    LZ_CHECK_EQ(r.staging(), 0u);
}

LZ_TEST(exact_reads_land_in_place_when_the_destination_is_congruent) {
    // An expert region: slots of a 4096-multiple stride, data placed at the file
    // offset's remainder. Each slot's aligned middle must be read straight in
    // (no staging), the edges staged, and every byte land where it belongs.
    Rig r({});
    LZ_REQUIRE(r.ok);
    const uint64_t stride = 3 * 4096, base = 5 * 4096 + 1234;   // head 1234
    const uint32_t head = base % 4096;
    std::vector<uint8_t> raw(4 * stride + 2 * 4096);
    uint8_t* mem = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(raw.data()) + 4095) & ~uintptr_t(4095));
    uint8_t* data = mem + head;
    const int experts[4] = {7, 0, 12, 3};
    std::vector<uint64_t> tags;
    for (int k = 0; k < 4; ++k) {
        const uint64_t off = base + uint64_t(experts[k]) * stride;
        LZ_REQUIRE(r.io.submit_exact(src(1, off, stride), data + k * stride, stride, &tags));
    }
    LZ_CHECK_EQ(tags.size(), 12u);                  // middle + two edges, each
    LZ_CHECK(r.io.settle(tags));
    for (int k = 0; k < 4; ++k) {
        LZ_CHECK(matches(data + k * stride, 1, base + uint64_t(experts[k]) * stride, stride));
    }
    LZ_CHECK_EQ(r.io.stats().exact_direct, 4u);
    LZ_CHECK_EQ(r.io.stats().exact_direct_bytes, 4u * (stride - 4096));
    LZ_CHECK_EQ(r.fake->violations(), 0u);
    LZ_CHECK_EQ(r.staging(), 0u);
}

LZ_TEST(exact_reads_fall_back_to_staging_when_they_cannot_go_in_place) {
    Rig r({});
    LZ_REQUIRE(r.ok);
    // Misaligned destination: one staged read, correct bytes.
    std::vector<uint8_t> raw(3 * 4096 + 8192);
    uint8_t* odd = reinterpret_cast<uint8_t*>(
        ((reinterpret_cast<uintptr_t>(raw.data()) + 4095) & ~uintptr_t(4095)) + 7);
    std::vector<uint64_t> tags;
    LZ_REQUIRE(r.io.submit_exact(src(0, 4096 + 100, 3 * 4096), odd, 3 * 4096, &tags));
    LZ_CHECK_EQ(tags.size(), 1u);
    LZ_CHECK(r.io.settle(tags));
    LZ_CHECK(matches(odd, 0, 4096 + 100, 3 * 4096));
    // No aligned middle at all (a small slice): one staged read.
    std::vector<uint8_t> small(700);
    LZ_REQUIRE(r.io.submit_exact(src(0, 50, 700), small.data(), 700, &tags));
    LZ_CHECK_EQ(tags.size(), 1u);
    LZ_CHECK(r.io.settle(tags));
    LZ_CHECK(matches(small.data(), 0, 50, 700));
    LZ_CHECK_EQ(r.io.stats().exact_direct, 0u);
    LZ_CHECK_EQ(r.io.stats().exact_staged, 2u);
    LZ_CHECK_EQ(r.fake->violations(), 0u);
    LZ_CHECK_EQ(r.staging(), 0u);
}

LZ_TEST(read_whole_lands_at_base_plus_head_and_frees_on_failure) {
    Rig r({});
    LZ_REQUIRE(r.ok);
    const uint64_t off = 4096 * 3 + 777, bytes = 300000;
    uint64_t alloc = 0;
    uint32_t head = 0;
    auto* base = static_cast<uint8_t*>(
        r.io.read_whole(mem::Category::ExpertCache, src(1, off, bytes), bytes, &alloc, &head));
    LZ_REQUIRE(base != nullptr);
    LZ_CHECK_EQ(head, 777u);
    LZ_CHECK(matches(base + head, 1, off, bytes));
    LZ_CHECK_EQ(r.acct.used_in(mem::Category::ExpertCache), alloc);   // charged at the span
    r.alloc.free(mem::Category::ExpertCache, base, alloc);
    LZ_CHECK_EQ(r.acct.used_in(mem::Category::ExpertCache), 0u);

    FakeBackend::Script s;
    s.eio_rate = 1.0;
    Rig bad(s);
    LZ_REQUIRE(bad.ok);
    LZ_CHECK(bad.io.read_whole(mem::Category::ExpertCache, src(0, 0, 50000), 50000,
                               &alloc, &head) == nullptr);
    LZ_CHECK_EQ(bad.acct.used_in(mem::Category::ExpertCache), 0u);   // freed, not leaked
}

LZ_TEST(whole_reads_are_chunked_and_every_chunk_is_judged) {
    // A whole read goes out as requests of at most whole_chunk() bytes: one
    // request per tensor used to truncate a 28.8 GB table to 3.03 GB (the
    // length is 32 bits). Scaled down: 64 KiB chunks over a ~3 MiB tensor at an
    // unaligned offset -- the bytes must land contiguously at base + head.
    Rig r({});
    LZ_REQUIRE(r.ok);
    r.io.set_whole_chunk(64 * 1024);
    const uint64_t off = 4096 * 9 + 123, bytes = 3 * 1024 * 1024 + 5000;
    uint64_t alloc = 0;
    uint32_t head = 0;
    auto* base = static_cast<uint8_t*>(
        r.io.read_whole(mem::Category::ExpertCache, src(1, off, bytes), bytes, &alloc, &head));
    LZ_REQUIRE(base != nullptr);
    LZ_CHECK_EQ(head, 123u);
    LZ_CHECK(matches(base + head, 1, off, bytes));
    LZ_CHECK_GE(r.fake->submit_log().size(), bytes / (64 * 1024));   // really chunked
    r.alloc.free(mem::Category::ExpertCache, base, alloc);


    // The tensor ends at a file end that is NOT aligned: the widened last chunk
    // runs past EOF and the device returns less than it asked for -- legal, as
    // it was for the single request, because all of the data arrived.
    const uint64_t odd_size = kFileBytes - 1000;
    Rig e({}, 2, false, odd_size);
    LZ_REQUIRE(e.ok);
    e.io.set_whole_chunk(64 * 1024);
    const uint64_t tail_bytes = 200000, tail_off = odd_size - tail_bytes;
    base = static_cast<uint8_t*>(e.io.read_whole(mem::Category::ExpertCache,
                                                 src(0, tail_off, tail_bytes), tail_bytes,
                                                 &alloc, &head));
    LZ_REQUIRE(base != nullptr);
    LZ_CHECK(matches(base + head, 0, tail_off, tail_bytes));
    e.alloc.free(mem::Category::ExpertCache, base, alloc);
    LZ_CHECK_EQ(e.staging(), 0u);

    // A chunk cut short inside the data fails the whole read, and the memory is
    // freed (the reads have all landed; nothing is left in flight).
    FakeBackend::Script s;
    s.cap_bytes = 40000;   // every request returns at most 40 KB of its 64 KiB
    Rig bad(s);
    LZ_REQUIRE(bad.ok);
    bad.io.set_whole_chunk(64 * 1024);
    LZ_CHECK(bad.io.read_whole(mem::Category::ExpertCache, src(0, 4096, 500000), 500000,
                               &alloc, &head) == nullptr);
    LZ_CHECK_EQ(bad.acct.used_in(mem::Category::ExpertCache), 0u);
}

LZ_TEST(randomised_mix_never_corrupts_or_leaks) {
    // Many small rigs with random scripts: whatever goes wrong, a batch that
    // reports success delivered exactly the right bytes, and unless the backend
    // died, every staging byte came back.
    std::mt19937_64 rng(2026);
    for (int round = 0; round < 300; ++round) {
        FakeBackend::Script s;
        s.seed = rng();
        s.queue_depth = 1 + rng() % 12;
        s.align = (rng() % 2) ? 4096 : 512;
        s.eio_rate = (rng() % 4 == 0) ? 0.05 : 0.0;
        s.short_rate = (rng() % 4 == 0) ? 0.05 : 0.0;
        const bool dies = rng() % 8 == 0;
        if (dies) s.die_after = rng() % 20;
        const size_t shards = 1 + rng() % 3;
        Rig r(s, shards);
        LZ_REQUIRE(r.ok);
        const size_t n = 1 + rng() % 30;
        std::vector<std::vector<uint8_t>> bufs(n);
        std::vector<Slice> slices;
        std::vector<std::pair<int32_t, uint64_t>> where;
        for (size_t i = 0; i < n; ++i) {
            const uint64_t len = 1 + rng() % 70000;
            const uint64_t off = rng() % (kFileBytes - len);
            const int32_t sh = static_cast<int32_t>(rng() % shards);
            bufs[i].resize(len);
            slices.push_back({src(sh, off, len), bufs[i].data(), len});
            where.emplace_back(slices.back().src.shard, off);
        }
        const bool ok = r.io.read_batch(slices);
        if (ok) {
            for (size_t i = 0; i < n; ++i) {
                LZ_CHECK(matches(bufs[i].data(), where[i].first, where[i].second, bufs[i].size()));
            }
        }
        if (!dies) LZ_CHECK_EQ(r.staging(), 0u);
        LZ_CHECK_EQ(r.fake->violations(), 0u);
    }
}

// ---------------------------------------------------------------------------
// The same contracts with the I/O thread owning the device. The fake is only
// ever called from that thread (R1), so these runs are also what
// ThreadSanitizer checks on the Linux build.

LZ_TEST(threaded_deep_batch_out_of_order) {
    FakeBackend::Script s;
    s.queue_depth = 4;
    Rig r(s, 2, true);
    LZ_REQUIRE(r.ok);
    std::vector<std::vector<uint8_t>> bufs(64);
    std::vector<Slice> slices;
    for (size_t i = 0; i < bufs.size(); ++i) {
        const uint64_t len = 5000 + i * 97;
        bufs[i].resize(len);
        slices.push_back({src(static_cast<int32_t>(i % 2), i * 100003, len), bufs[i].data(), len});
    }
    LZ_REQUIRE(r.io.read_batch(slices));
    for (size_t i = 0; i < bufs.size(); ++i) {
        LZ_CHECK(matches(bufs[i].data(), static_cast<int32_t>(i % 2), i * 100003, bufs[i].size()));
    }
    LZ_CHECK_EQ(r.staging(), 0u);
}

LZ_TEST(threaded_dead_backend_leaks_and_reports_death) {
    FakeBackend::Script s;
    s.queue_depth = 8;
    s.die_after = 3;
    Rig r(s, 2, true);
    LZ_REQUIRE(r.ok);
    std::vector<uint8_t> a(10000), b(10000);
    std::vector<uint64_t> tags;
    for (int i = 0; i < 10; ++i) {
        tags.push_back(r.io.submit_staged(src(0, uint64_t(i) * 40000, 10000),
                                          i % 2 ? a.data() : b.data(), 10000));
    }
    bool dead = false;
    LZ_CHECK(!r.io.settle(tags, &dead));
    LZ_CHECK(dead);
    LZ_CHECK_GT(r.staging(), 0u);   // outstanding reads' staging stays charged
}

LZ_TEST(threaded_whole_read_and_failed_read) {
    Rig r({}, 2, true);
    LZ_REQUIRE(r.ok);
    uint64_t alloc = 0;
    uint32_t head = 0;
    auto* base = static_cast<uint8_t*>(
        r.io.read_whole(mem::Category::ExpertCache, src(1, 4096 + 5, 200000), 200000,
                        &alloc, &head));
    LZ_REQUIRE(base != nullptr);
    LZ_CHECK(matches(base + head, 1, 4096 + 5, 200000));
    r.alloc.free(mem::Category::ExpertCache, base, alloc);

    FakeBackend::Script s;
    s.eio_rate = 1.0;
    Rig bad(s, 2, true);
    LZ_REQUIRE(bad.ok);
    std::vector<uint8_t> buf(8192);
    std::vector<uint64_t> tags{bad.io.submit_staged(src(0, 0, 8192), buf.data(), 8192)};
    bool dead = true;
    LZ_CHECK(!bad.io.settle(tags, &dead));
    LZ_CHECK(!dead);                  // failed and landed: not a dead backend
    LZ_CHECK_EQ(bad.staging(), 0u);
}

LZ_TEST(threaded_randomised_mix_never_corrupts_or_leaks) {
    // As the single-threaded mix, plus interleaved read-ahead (Background,
    // unstaged) enqueued between batches and settled after -- the ring's shape.
    std::mt19937_64 rng(4052);
    for (int round = 0; round < 200; ++round) {
        FakeBackend::Script s;
        s.seed = rng();
        s.queue_depth = 1 + rng() % 12;
        s.eio_rate = (rng() % 4 == 0) ? 0.05 : 0.0;
        s.short_rate = (rng() % 4 == 0) ? 0.05 : 0.0;
        const bool dies = rng() % 8 == 0;
        if (dies) s.die_after = rng() % 20;
        const size_t shards = 1 + rng() % 3;
        Rig r(s, shards, true);
        LZ_REQUIRE(r.ok);

        std::vector<uint8_t> raw(6 * 16384 + 4096);
        uint8_t* arena = reinterpret_cast<uint8_t*>(
            (reinterpret_cast<uintptr_t>(raw.data()) + 4095) & ~uintptr_t(4095));
        std::vector<uint64_t> ahead;
        for (int i = 0; i < 6; ++i) {
            ahead.push_back(r.io.submit_unstaged(r.io.file(0), uint64_t(i) * 16384, 16384,
                                                 arena + i * 16384, 0, 16384,
                                                 dray::backend::IoPriority::Background));
        }
        const size_t n = 1 + rng() % 30;
        std::vector<std::vector<uint8_t>> bufs(n);
        std::vector<Slice> slices;
        for (size_t i = 0; i < n; ++i) {
            const uint64_t len = 1 + rng() % 70000;
            const uint64_t off = rng() % (kFileBytes - len);
            const int32_t sh = static_cast<int32_t>(rng() % shards);
            bufs[i].resize(len);
            slices.push_back({src(sh, off, len), bufs[i].data(), len});
        }
        const bool ok = r.io.read_batch(slices);
        if (ok) {
            for (size_t i = 0; i < n; ++i) {
                LZ_CHECK(matches(bufs[i].data(), slices[i].src.shard, slices[i].src.offset,
                                 bufs[i].size()));
            }
        }
        const bool ahead_ok = r.io.settle(ahead);
        if (ahead_ok) {
            for (int i = 0; i < 6; ++i) LZ_CHECK(matches(arena + i * 16384, 0, uint64_t(i) * 16384, 16384));
        }
        if (!dies) LZ_CHECK_EQ(r.staging(), 0u);
    }
}

LZ_TEST(threaded_large_reads_land_before_they_are_reported) {
    // A read must not be reported Done until its staged bytes are home: the
    // waiter frees the staging and reads the destination the moment it sees
    // Done. Large reads in a deep queue make the I/O thread's copy window
    // milliseconds long, so a completion reported early is actually hit --
    // ThreadSanitizer flags the overlap, and the byte check catches it too.
    //
    // A waiter only re-checks when woken, and the wake-up is sent after the copy,
    // so an early "Done" is visible only to a waiter that ARRIVES mid-copy. The
    // waiter's arrival is therefore staggered across the I/O thread's work.
    FakeBackend::Script s;
    s.queue_depth = 64;
    for (int round = 0; round < 40; ++round) {
        s.seed = 900 + round;
        Rig r(s, 2, true);
        LZ_REQUIRE(r.ok);
        const uint64_t len = 1ull << 20;
        std::vector<std::vector<uint8_t>> bufs(24, std::vector<uint8_t>(len));
        std::vector<uint64_t> tags;
        std::vector<std::pair<int32_t, uint64_t>> where;
        for (size_t i = 0; i < bufs.size(); ++i) {
            const uint64_t off = (i * 1234567u) % (kFileBytes - len);
            const int32_t sh = static_cast<int32_t>(i % 2);
            tags.push_back(r.io.submit_staged(src(sh, off, len), bufs[i].data(), len));
            where.emplace_back(sh, off);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(250 * (round % 16)));
        LZ_REQUIRE(r.io.settle(tags));
        for (size_t i = 0; i < bufs.size(); ++i) {
            LZ_CHECK(matches(bufs[i].data(), where[i].first, where[i].second, len));
        }
        LZ_CHECK_EQ(r.staging(), 0u);
    }
}
