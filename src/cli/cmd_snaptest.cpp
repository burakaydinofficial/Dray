#include "cli/commands.h"
#include "cli/engine_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

#include "cache/snapshot_cache.h"
#include "engine/engine.h"

namespace dray::cli {

// SNAPSHOT ROUND-TRIP PROOF -- the integration test the unit suite registered
// as debt (store/load need a live context; here is one, through the same
// engine_open every consumer uses).
//
// Shape: generate k tokens greedily, capture the state BEFORE decoding the
// pending token and remember that token -- the exact resume shape the job
// layer uses, because a snapshot taken after a decode has no logits and the
// pending-token re-decode is what regenerates them. Continue to the end for
// the direct text; then clear, restore, re-decode the pending token, continue
// again. Greedy determinism makes the verdict binary: the two continuations
// must match token for token, or persistence is broken.

int cmd_snaptest(const Args& a) {
    namespace cache = dray::cache;
    namespace srv = dray::engine;

    srv::EngineConfig ec = engine_config_from(a);
    if (ec.n_ctx == 0) ec.n_ctx = 512;   // a round-trip test needs no long context
    std::string err;
    std::unique_ptr<srv::Engine> eng = srv::Engine::open(ec, &err);
    if (!eng) { std::fprintf(stderr, "snaptest: %s\n", err.c_str()); return 1; }
    llama_context* lctx = eng->context();
    const llama_vocab* vocab = eng->vocab();

    auto tokenize = [&](const std::string& s) {
        std::vector<llama_token> t(s.size() + 8);
        int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                                   t.data(), (int32_t)t.size(), true, true);
        t.resize(n > 0 ? (size_t)n : 0);
        return t;
    };
    auto greedy_next = [&]() {
        llama_sampler* g = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(g, llama_sampler_init_greedy());
        llama_token t = llama_sampler_sample(g, lctx, -1);
        llama_sampler_free(g);
        return t;
    };
    auto decode1 = [&](llama_token t) {
        llama_batch b = llama_batch_get_one(&t, 1);
        return llama_decode(lctx, b) == 0;
    };

    std::vector<llama_token> toks = tokenize(a.prompt);
    if (toks.empty()) { eng.reset(); return 1; }
    llama_batch pre = llama_batch_get_one(toks.data(), (int32_t)toks.size());
    if (llama_decode(lctx, pre) != 0) { eng.reset(); return 1; }

    uint64_t h = cache::prefix_hash(toks.data(), toks.size());
    const int32_t k = 8, tail = 8;

    // k-1 tokens decoded; the k-th becomes the pending token at the capture point.
    for (int32_t i = 0; i < k - 1; ++i) {
        llama_token t = greedy_next();
        h = cache::prefix_hash_extend(h, t);
        if (!decode1(t)) { eng.reset(); return 1; }
    }
    const llama_token pending = greedy_next();

    cache::CompatStamp stamp;
    if (!cache::make_compat_stamp(a.model, lctx, "snaptest-build", &stamp, &err)) {
        std::fprintf(stderr, "snaptest: stamp: %s\n", err.c_str());
        eng.reset();
        return 1;
    }
    cache::Config cc;
    cc.dir = std::filesystem::temp_directory_path() / "dray_snaptest";
    std::filesystem::remove_all(cc.dir);
    cc.budget_bytes = 1ull << 30;
    cc.acct = eng->accountant();   // F4: gated path = tested path
    cache::SnapshotCache snap(cc, stamp);
    if (!snap.open(&err)) { std::fprintf(stderr, "snaptest: %s\n", err.c_str()); eng.reset(); return 1; }
    if (!snap.store(h, toks.size() + k - 1, lctx, 0, cache::Retention::Pinned, &err)) {
        std::fprintf(stderr, "snaptest: store: %s\n", err.c_str());
        eng.reset();
        return 1;
    }
    std::printf("stored: %.1f MB pinned\n", snap.stats().pinned_bytes / 1e6);

    // Direct continuation: pending + tail more.
    std::vector<llama_token> direct;
    direct.push_back(pending);
    if (!decode1(pending)) { eng.reset(); return 1; }
    for (int32_t i = 0; i < tail; ++i) {
        llama_token t = greedy_next();
        direct.push_back(t);
        if (!decode1(t)) { eng.reset(); return 1; }
    }

    // Nuke and restore.
    llama_memory_clear(llama_get_memory(lctx), true);
    auto entry = snap.lookup(h);
    if (!entry) { std::fprintf(stderr, "snaptest: lookup miss after store\n"); eng.reset(); return 1; }
    if (!snap.load(*entry, lctx, 0, &err)) {
        std::fprintf(stderr, "snaptest: load: %s\n", err.c_str());
        eng.reset();
        return 1;
    }

    std::vector<llama_token> resumed;
    resumed.push_back(pending);
    if (!decode1(pending)) { eng.reset(); return 1; }
    for (int32_t i = 0; i < tail; ++i) {
        llama_token t = greedy_next();
        resumed.push_back(t);
        if (!decode1(t)) { eng.reset(); return 1; }
    }

    const bool same = direct == resumed;
    std::printf("SNAPSHOT ROUND-TRIP: %s (%zu tokens compared)\n",
                same ? "IDENTICAL" : "MISMATCH", direct.size());

    // Stamp refusal: a cache opened under a different build id must not see
    // this snapshot -- serving a checkpoint across an incompatible state format
    // is fluent garbage with extra steps.
    cache::CompatStamp other = stamp;
    other.engine_build_id ^= 0xDEAD;
    cache::SnapshotCache snap2(cc, other);
    std::string e2;
    const bool refused = snap2.open(&e2) && !snap2.lookup(h).has_value();
    std::printf("STAMP REFUSAL: %s\n", refused ? "correct (foreign stamp invisible)"
                                               : "BROKEN (foreign snapshot visible)");

    std::filesystem::remove_all(cc.dir);
    eng.reset();
    return (same && refused) ? 0 : 1;
}


}  // namespace dray::cli
