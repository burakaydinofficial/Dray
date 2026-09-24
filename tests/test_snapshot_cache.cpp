// SnapshotCache unit coverage, reconciled against the REAL header. The previous
// version of this file predates src/cache/snapshot_cache.h and guessed a
// contract (snapshot_hash, u64 stamps, a 3-arg constructor) that never existed.
// The header is the designed artifact -- 249 documented lines whose store/load
// semantics are bound to a live llama_context by design -- so the test moved to
// the API, not the reverse. The old CMake note argued the opposite; it was
// written before the header existed and is superseded by this reconciliation.
//
// Covered WITHOUT a context: prefix-hash properties, directory lifecycle
// (open, .tmp sweep, foreign files), and index/stats/unpin/discard/eviction
// edges. NOT covered here, on purpose: store()/load() round-trip, compat-stamp
// refusal, blob-hash verification, Pinned-vs-Reuse eviction under pressure --
// all capture or restore live context state, so they belong to an integration
// test with the testbed control model, alongside the persistence wiring.
// Registered as coverage debt in DECISIONS.md rather than silently absent.

#include "harness.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "cache/snapshot_cache.h"

namespace fs = std::filesystem;
using dray::cache::CompatStamp;
using dray::cache::Config;
using dray::cache::kPrefixHashSeed;
using dray::cache::prefix_hash;
using dray::cache::prefix_hash_extend;
using dray::cache::SnapshotCache;

namespace {

struct TempDir {
    fs::path p;
    explicit TempDir(const char* tag) {
        p = fs::temp_directory_path() / (std::string("dray_snap_") + tag);
        fs::remove_all(p);
        fs::create_directories(p);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(p, ec); }
};

CompatStamp stamp_a() {
    CompatStamp s;
    s.model_file_size = 12345;
    s.model_mtime = 777;
    s.engine_build_id = 0xA11CE;
    s.n_ctx = 2048;
    std::snprintf(s.model_id, sizeof(s.model_id), "test-arch quant-tier-A");
    return s;
}

}  // namespace

LZ_TEST(hash_is_deterministic_and_incremental) {
    const int32_t t[] = { 1, 2, 3, 4, 5 };
    LZ_CHECK_EQ(prefix_hash(t, 5), prefix_hash(t, 5));
    // Extending token by token equals hashing the whole prefix: the property
    // that lets a running decode maintain the hash incrementally.
    uint64_t h = kPrefixHashSeed;
    for (int32_t tok : t) h = prefix_hash_extend(h, tok);
    LZ_CHECK_EQ(h, prefix_hash(t, 5));
}

LZ_TEST(hash_is_order_and_length_sensitive) {
    const int32_t a[] = { 1, 2, 3 };
    const int32_t b[] = { 3, 2, 1 };
    LZ_CHECK_NE(prefix_hash(a, 3), prefix_hash(b, 3));
    LZ_CHECK_NE(prefix_hash(a, 2), prefix_hash(a, 3));
    LZ_CHECK_EQ(prefix_hash(nullptr, 0), kPrefixHashSeed);
    LZ_CHECK_NE(prefix_hash(nullptr, 0), prefix_hash(a, 1));
}

LZ_TEST(open_on_empty_dir_reports_empty_stats) {
    TempDir d("open");
    Config cfg;
    cfg.dir = d.p;
    cfg.budget_bytes = 4ull << 20;
    SnapshotCache c(cfg, stamp_a());
    std::string err;
    LZ_CHECK(c.open(&err));
    LZ_CHECK(err.empty());

    const auto st = c.stats();
    LZ_CHECK_EQ(st.n_reuse, size_t{0});
    LZ_CHECK_EQ(st.n_pinned, size_t{0});
    LZ_CHECK_EQ(st.reuse_bytes, uint64_t{0});
    LZ_CHECK_EQ(st.budget_bytes, uint64_t{4ull << 20});
}

LZ_TEST(open_sweeps_torn_tmp_files) {
    TempDir d("tmp");
    // A torn write from a crashed instance: open() must delete it, because a
    // snapshot exists iff a fully-written, correctly-stamped file exists.
    const fs::path torn = d.p / "partial.tmp";
    { std::ofstream f(torn, std::ios::binary); f << "torn write"; }
    LZ_CHECK(fs::exists(torn));

    Config cfg;
    cfg.dir = d.p;
    cfg.budget_bytes = 1ull << 20;
    SnapshotCache c(cfg, stamp_a());
    std::string err;
    LZ_CHECK(c.open(&err));
    LZ_CHECK(!fs::exists(torn));
}

LZ_TEST(open_ignores_but_preserves_foreign_files) {
    TempDir d("foreign");
    const fs::path alien = d.p / "notes.txt";
    { std::ofstream f(alien); f << "not a snapshot"; }

    Config cfg;
    cfg.dir = d.p;
    cfg.budget_bytes = 1ull << 20;
    SnapshotCache c(cfg, stamp_a());
    std::string err;
    LZ_CHECK(c.open(&err));
    LZ_CHECK_EQ(c.stats().n_reuse, size_t{0});
    LZ_CHECK(fs::exists(alien));
}

LZ_TEST(empty_index_edges_are_safe) {
    TempDir d("edges");
    Config cfg;
    cfg.dir = d.p;
    cfg.budget_bytes = 1ull << 20;
    SnapshotCache c(cfg, stamp_a());
    std::string err;
    LZ_CHECK(c.open(&err));

    const int32_t t[] = { 9, 9, 9 };
    const uint64_t h = prefix_hash(t, 3);

    LZ_CHECK(!c.lookup(h).has_value());
    LZ_CHECK(!c.unpin(h));      // nothing pinned under this hash
    LZ_CHECK(!c.discard(h));    // nothing to discard
    c.evict_to_budget();        // no-op on empty; must not crash or spin
    c.flush();                  // empty queue; must return immediately
    LZ_CHECK_EQ(c.stats().n_reuse, size_t{0});
}
