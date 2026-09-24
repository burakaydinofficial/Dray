// The storage backend's submission and completion shape sets the threading
// model for everything above it, so its contract has to hold before anything
// else can be trusted. These cases run against a real file on a real filesystem
// because the alignment constraints under test are runtime values discovered
// from the mount (Invariant 5) -- a stub would answer every alignment question
// with the number the test wanted to hear.
//
// The short-read case is the one that matters most. At design-center caps this
// engine reads ~1 PB/day, possibly from an external drive; short reads and EIO
// are expected, not exceptional. A partially filled slot handed to
// ggml_mul_mat_id yields plausible text from garbage weights, which is the worst
// failure mode this project has: wrong output that looks right.

#include "harness.h"

#include "fixtures/make_gguf.h"
#include "fixtures/temp_dir.h"
#include "io/storage.h"
#include "mem/accountant.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using dray::io::Alignment;
using dray::io::Backend;
using dray::io::Completion;
using dray::io::FileId;
using dray::io::kInvalidFile;
using dray::io::ReadRequest;

namespace {

// Odd size on purpose: file_size % alignment is then non-zero for every power of
// two, which is what makes the EOF short read reachable at all.
constexpr uint64_t kFileBytes = 3ull * 1024 * 1024 + 777;
constexpr uint32_t kSeed = 0x51A5;

struct Env {
    dray::testfix::TempDir dir{"storage"};
    bool written = false;
    Env() {
        written = dray::testfix::write_pattern_file(dir.file("blob.bin"), kFileBytes, kSeed);
    }
};

const Env& env() {
    static Env e;
    return e;
}

std::filesystem::path blob() { return env().dir.file("blob.bin"); }

bool is_pow2(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Owns the aligned destination so every case gets a buffer the backend will
// actually accept.
class Buffer {
public:
    Buffer(size_t bytes, uint32_t align) : bytes_(bytes) {
        p_ = dray::mem::aligned_alloc_host(bytes, align);
    }
    ~Buffer() { dray::mem::aligned_free_host(p_, bytes_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* data() const { return p_; }
    size_t size() const { return bytes_; }

private:
    void* p_ = nullptr;
    size_t bytes_ = 0;
};

// Drains until `want` completions have been harvested or the deadline passes.
size_t drain(Backend& b, std::vector<Completion>& out, size_t want, int timeout_ms = 10000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    size_t got = 0;
    while (got < want && std::chrono::steady_clock::now() < deadline) {
        Completion buf[64];
        const size_t n = b.poll(buf, 64, 1);
        for (size_t i = 0; i < n; ++i) {
            out.push_back(buf[i]);
        }
        got += n;
        if (n == 0) {
            // poll(min_complete=1) came back empty; not fatal on its own, the
            // deadline decides. Yield rather than spin the core the build needs.
            std::this_thread::yield();
        }
    }
    return got;
}

}  // namespace

LZ_TEST(fixture_file_exists) {
    LZ_REQUIRE(env().written);
    LZ_CHECK_EQ(std::filesystem::file_size(blob()), kFileBytes);
}

LZ_TEST(make_backend_never_returns_null_and_says_what_it_is) {
    std::unique_ptr<Backend> b = dray::io::make_backend(32);
    // The blocking-pread thread pool is always available as a floor, so there is
    // no platform on which this may fail.
    LZ_REQUIRE(b != nullptr);

    // The startup report has to identify the actual mechanism; "unknown backend"
    // is how a user ends up unable to tell whether the engine or the drive is
    // the limit.
    const std::string d = b->describe();
    LZ_CHECK_GT(d.size(), 8u);
    LZ_CHECK_GE(b->max_in_flight(), 1u);
    LZ_CHECK_EQ(b->in_flight(), 0u);
}

LZ_TEST(alignment_is_queried_and_is_a_power_of_two) {
    std::unique_ptr<Backend> b = dray::io::make_backend(32);
    LZ_REQUIRE(b != nullptr);
    const FileId f = b->open(blob());
    LZ_REQUIRE_NE(f, kInvalidFile);

    const Alignment a = b->alignment(f);
    LZ_CHECK(is_pow2(a.memory));
    LZ_CHECK(is_pow2(a.offset));
    LZ_CHECK(is_pow2(a.length));
    // Never hardcode 4096 -- but a discovered value outside this range means the
    // query failed and something defaulted badly.
    LZ_CHECK_LE(a.memory, 1u << 20);
    LZ_CHECK_LE(a.offset, 1u << 20);
    LZ_CHECK_LE(a.length, 1u << 20);
    LZ_CHECK_EQ(a.max(), std::max(a.memory, std::max(a.offset, a.length)));

    LZ_CHECK_EQ(b->size(f), kFileBytes);
    b->close(f);
}

LZ_TEST(an_aligned_read_returns_exactly_the_bytes_on_disk) {
    std::unique_ptr<Backend> b = dray::io::make_backend(32);
    LZ_REQUIRE(b != nullptr);
    const FileId f = b->open(blob());
    LZ_REQUIRE_NE(f, kInvalidFile);

    const Alignment a = b->alignment(f);
    const uint32_t stride = std::max(a.max(), 1u);
    const uint32_t length = stride * 4;
    const uint64_t offset = stride * 3;

    Buffer buf(length, a.memory);
    LZ_REQUIRE(buf.data() != nullptr);
    std::memset(buf.data(), 0, length);

    ReadRequest r;
    r.file = f;
    r.offset = offset;
    r.length = length;
    r.dst = buf.data();
    r.tag = 0xC0FFEEull;

    LZ_REQUIRE_EQ(b->submit(&r, 1), 1u);

    std::vector<Completion> comps;
    LZ_REQUIRE_EQ(drain(*b, comps, 1), 1u);
    LZ_REQUIRE_EQ(comps.size(), 1u);

    const Completion& c = comps[0];
    LZ_CHECK_EQ(c.tag, 0xC0FFEEull);   // tags are opaque and returned verbatim
    LZ_CHECK_EQ(c.status, 0);
    LZ_CHECK_EQ(c.bytes, length);
    LZ_CHECK(c.ok(length));
    LZ_CHECK(dray::testfix::verify_buffer(kSeed, buf.data(), length, offset));

    LZ_CHECK_EQ(b->in_flight(), 0u);
    b->close(f);
}

LZ_TEST(a_short_read_at_eof_is_a_short_read_not_an_error) {
    std::unique_ptr<Backend> b = dray::io::make_backend(32);
    LZ_REQUIRE(b != nullptr);
    const FileId f = b->open(blob());
    LZ_REQUIRE_NE(f, kInvalidFile);

    const Alignment a = b->alignment(f);
    const uint64_t size = b->size(f);
    const uint32_t stride = std::max(a.max(), 1u);

    uint64_t offset = (size / stride) * stride;
    if (offset >= size && offset >= stride) {
        offset -= stride;
    }
    uint32_t length = stride;
    while (size - offset >= length) {
        length += stride;  // stays a multiple of every alignment
    }
    const uint32_t expected = static_cast<uint32_t>(size - offset);
    LZ_REQUIRE_LT(expected, length);
    LZ_REQUIRE_GT(expected, 0u);

    Buffer buf(length, a.memory);
    LZ_REQUIRE(buf.data() != nullptr);
    std::memset(buf.data(), 0, length);

    ReadRequest r;
    r.file = f;
    r.offset = offset;
    r.length = length;
    r.dst = buf.data();
    r.tag = 7;

    LZ_REQUIRE_EQ(b->submit(&r, 1), 1u);
    std::vector<Completion> comps;
    LZ_REQUIRE_EQ(drain(*b, comps, 1), 1u);

    const Completion& c = comps[0];
    // Status and bytes are separate fields because callers MUST check both: a
    // status of 0 here does not mean the slot is filled.
    LZ_CHECK_EQ(c.status, 0);
    LZ_CHECK_EQ(c.bytes, expected);
    LZ_CHECK_LT(c.bytes, length);
    LZ_CHECK(!c.ok(length));
    LZ_CHECK(c.ok(expected));
    LZ_CHECK(dray::testfix::verify_buffer(kSeed, buf.data(), expected, offset));

    b->close(f);
}

LZ_TEST(submit_accepts_fewer_than_asked_when_the_queue_is_full) {
    // Deliberately shallow: the caller's contract is to poll() and retry, and it
    // can only be exercised by overflowing the queue.
    std::unique_ptr<Backend> b = dray::io::make_backend(4);
    LZ_REQUIRE(b != nullptr);
    const FileId f = b->open(blob());
    LZ_REQUIRE_NE(f, kInvalidFile);

    const Alignment a = b->alignment(f);
    const uint32_t stride = std::max(a.max(), 1u);
    const size_t mif = b->max_in_flight();
    LZ_REQUIRE_GE(mif, 1u);
    // A queue depth of 4 that reports thousands in flight is not a queue depth.
    LZ_REQUIRE_LE(mif, 1024u);

    const size_t n = mif + 8;
    const uint64_t blocks = std::max<uint64_t>(b->size(f) / stride, 1u);

    Buffer buf(n * stride, a.memory);
    LZ_REQUIRE(buf.data() != nullptr);

    std::vector<ReadRequest> reqs(n);
    for (size_t i = 0; i < n; ++i) {
        reqs[i].file = f;
        reqs[i].offset = (static_cast<uint64_t>(i) % blocks) * stride;
        reqs[i].length = stride;
        reqs[i].dst = static_cast<char*>(buf.data()) + i * stride;
        reqs[i].tag = i;
    }

    const size_t accepted = b->submit(reqs.data(), n);
    LZ_CHECK_GE(accepted, 1u);
    LZ_CHECK_LT(accepted, n);      // never blocks, so it must refuse the excess
    LZ_CHECK_LE(accepted, mif);

    std::vector<Completion> comps;
    LZ_CHECK_EQ(drain(*b, comps, accepted), accepted);

    // Now that the queue has drained, the rest must be accepted.
    size_t remaining = n - accepted;
    size_t offset_in = accepted;
    int rounds = 0;
    while (remaining > 0 && rounds < 100) {
        const size_t took = b->submit(reqs.data() + offset_in, remaining);
        offset_in += took;
        remaining -= took;
        if (took == 0) {
            LZ_FAIL("submit accepted nothing with an empty queue");
            break;
        }
        drain(*b, comps, took);
        ++rounds;
    }
    LZ_CHECK_EQ(remaining, 0u);
    LZ_CHECK_EQ(comps.size(), n);

    // Every tag comes back exactly once, and every read is whole: these offsets
    // are all well inside the file.
    std::vector<int> seen(n, 0);
    for (const Completion& c : comps) {
        LZ_REQUIRE_LT(c.tag, n);
        ++seen[static_cast<size_t>(c.tag)];
        LZ_CHECK(c.ok(stride));
    }
    for (size_t i = 0; i < n; ++i) {
        LZ_CHECK_EQ(seen[i], 1);
    }

    LZ_CHECK_EQ(b->in_flight(), 0u);
    b->close(f);
}

LZ_TEST(poll_with_min_complete_zero_does_not_block) {
    std::unique_ptr<Backend> b = dray::io::make_backend(32);
    LZ_REQUIRE(b != nullptr);

    Completion out[8];

    // Nothing has ever been submitted: a backend that waits here deadlocks the
    // scheduler on its first idle tick.
    auto t0 = std::chrono::steady_clock::now();
    const size_t n0 = b->poll(out, 8, 0);
    auto t1 = std::chrono::steady_clock::now();
    LZ_CHECK_EQ(n0, 0u);
    LZ_CHECK_LT(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(), 1000);

    const FileId f = b->open(blob());
    LZ_REQUIRE_NE(f, kInvalidFile);
    const Alignment a = b->alignment(f);
    const uint32_t stride = std::max(a.max(), 1u);

    Buffer buf(stride, a.memory);
    LZ_REQUIRE(buf.data() != nullptr);

    ReadRequest r;
    r.file = f;
    r.offset = 0;
    r.length = stride;
    r.dst = buf.data();
    r.tag = 99;
    LZ_REQUIRE_EQ(b->submit(&r, 1), 1u);

    // With one read outstanding, a non-blocking poll may return 0 or 1, but it
    // must return.
    t0 = std::chrono::steady_clock::now();
    const size_t n1 = b->poll(out, 8, 0);
    t1 = std::chrono::steady_clock::now();
    LZ_CHECK_LE(n1, 1u);
    LZ_CHECK_LT(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(), 1000);

    std::vector<Completion> comps;
    for (size_t i = 0; i < n1; ++i) {
        comps.push_back(out[i]);
    }
    drain(*b, comps, n1 == 0 ? 1u : 0u);
    LZ_CHECK_EQ(comps.size(), 1u);
    b->close(f);
}

LZ_TEST(opening_a_file_that_is_not_there_fails_without_throwing) {
    std::unique_ptr<Backend> b = dray::io::make_backend(8);
    LZ_REQUIRE(b != nullptr);
    const FileId f = b->open(env().dir.file("no_such_blob.bin"));
    LZ_CHECK_EQ(f, kInvalidFile);
}

LZ_TEST(a_file_can_be_reopened_after_close) {
    // The cache index reopens the model file across a resume; a backend that
    // leaks or invalidates ids here breaks multi-day operation.
    std::unique_ptr<Backend> b = dray::io::make_backend(8);
    LZ_REQUIRE(b != nullptr);
    for (int i = 0; i < 4; ++i) {
        const FileId f = b->open(blob());
        LZ_REQUIRE_NE(f, kInvalidFile);
        LZ_CHECK_EQ(b->size(f), kFileBytes);
        b->close(f);
    }
}
