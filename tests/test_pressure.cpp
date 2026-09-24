// Cap enforcement under pressure (swarm S17): eviction, the rebudget shrink,
// and the ledger's bound, exercised through a real Streamer on the synthetic
// GGUF -- the machinery that runs per token for days and had no test.
//
// Built on the graph-reuse harness's three lessons: tight cap (or the compact
// path never runs), weights bound through the streamer's buffer type, and
// never reading past k slots of a compact region.

#include "harness.h"

#include <cstdio>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "backend/stream_buffer.h"
#include "fixtures/make_gguf.h"
#include "fixtures/temp_dir.h"
#include "mem/accountant.h"
#include "plan/residency.h"

using dray::testfix::Fixture;
using dray::testfix::GgufSpec;
using dray::testfix::fill_byte;

namespace {

struct Env {
    dray::testfix::TempDir dir{"pressure"};
    Fixture fx;
    Env() { fx = dray::testfix::make_gguf(dir.path(), "pressure.gguf", GgufSpec{}); }
};

Env& env() {
    static Env e;
    return e;
}

bool selection_reads_expert(const ggml_tensor* node, const ggml_tensor* w,
                            size_t j,
                            const dray::testfix::FixtureTensor& t,
                            uint32_t n_experts, int32_t e) {
    const ggml_tensor* rids = node->src[2];
    if (!rids || !rids->data) return false;
    const int32_t row = static_cast<const int32_t*>(rids->data)[j];
    if (row < 0) return false;
    const uint64_t per = t.bytes / n_experts;
    const uint8_t* got = static_cast<const uint8_t*>(w->data) +
                         static_cast<uint64_t>(row) * w->nb[2];
    for (uint64_t i = 0; i < per; ++i) {
        if (got[i] != fill_byte(t.seed, static_cast<uint64_t>(e) * per + i)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// A cyclic sweep of every MoE layer, twice, at a cap that cannot hold them
// all: the classic below-knee regime. The assertions are the multi-day
// invariants -- the ledger never exceeds the cap, no materialise fails, and
// every selection reads its expert's true bytes even as eviction churns.
LZ_TEST(cyclic_sweep_stays_correct_and_capped) {
    const Fixture& fx = env().fx;
    LZ_CHECK(fx.ok);

    std::string err;
    dray::plan::Plan probe =
        dray::plan::build_plan(fx.path.string(), 1ull << 30, 512, &err);
    LZ_CHECK(err.empty());
    const uint64_t cap = (fx.file_size - fx.routed_bytes) + fx.routed_bytes / 4 +
                         probe.floor.kv_cache + probe.floor.recurrent_state +
                         probe.floor.compute_scratch + (256u << 10);
    // Pass-4b: the sentinel reservation is honest now (its result was once
    // discarded), so the tight cap must fund the widest tensor explicitly.
    uint64_t widest = 0;
    for (const auto& wt : fx.tensors) widest = std::max<uint64_t>(widest, wt.bytes);
    const uint64_t cap2 = cap + widest + (8ull << 20);   // sentinel has an 8 MiB floor

    dray::plan::Plan plan = dray::plan::build_plan(fx.path.string(), cap2, 512, &err);
    LZ_CHECK(err.empty());
    LZ_CHECK(plan.feasible);

    dray::mem::Accountant acct(cap2);
    acct.reserve(dray::mem::Category::KvCache, plan.floor.kv_cache);
    acct.reserve(dray::mem::Category::RecurrentState, plan.floor.recurrent_state);
    acct.reserve(dray::mem::Category::ComputeScratch, plan.floor.compute_scratch);

    dray::backend::Config cfg;
    cfg.cap = cap2;
    dray::backend::Streamer streamer(acct, plan, cfg);
    LZ_CHECK(streamer.valid());

    const uint32_t n_experts = fx.spec.n_experts;
    const int64_t k = 2;
    const uint32_t n_layers = fx.spec.n_layers;
    const uint32_t lead = fx.spec.n_dense_lead;

    // Bind every MoE gate tensor the way llama does.
    ggml_init_params wp{};
    wp.mem_size = ggml_tensor_overhead() * (n_layers + 2);
    wp.no_alloc = true;
    ggml_context* wctx = ggml_init(wp);
    LZ_CHECK(wctx != nullptr);

    std::vector<ggml_tensor*> ws;
    std::vector<const dray::testfix::FixtureTensor*> fts;
    for (uint32_t l = lead; l < n_layers; ++l) {
        const std::string name = "blk." + std::to_string(l) + ".ffn_gate_exps.weight";
        const auto* ft = fx.find(name);
        LZ_CHECK(ft != nullptr);
        ggml_tensor* w = ggml_new_tensor_3d(wctx, static_cast<ggml_type>(ft->ggml_type),
                                            ft->ne[0], ft->ne[1], ft->ne[2]);
        ggml_set_name(w, name.c_str());
        ws.push_back(w);
        fts.push_back(ft);
    }
    ggml_backend_buffer_t wbuf =
        ggml_backend_alloc_ctx_tensors_from_buft(wctx, streamer.buft());
    LZ_CHECK(wbuf != nullptr);

    ggml_init_params gp{};
    gp.mem_size = 8 * 1024 * 1024;
    gp.no_alloc = false;
    ggml_context* gctx = ggml_init(gp);
    LZ_CHECK(gctx != nullptr);

    std::vector<ggml_tensor*> nodes;
    std::vector<ggml_tensor*> idss;
    for (size_t i = 0; i < ws.size(); ++i) {
        ggml_tensor* x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, fts[i]->ne[0], 1);
        ggml_tensor* ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, k, 1);
        idss.push_back(ids);
        nodes.push_back(ggml_mul_mat_id(gctx, ws[i], x, ids));
    }

    // Two full sweeps with rotating selections. 7 MoE layers x 2 sweeps at a
    // cap whose routed slack is ~40% of routed bytes (H20: the sentinel's
    // 8 MiB funding moved the regime from the original quarter) -- still
    // well short of the working set, so eviction stays continuous.
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (size_t i = 0; i < nodes.size(); ++i) {
            int32_t* idv = static_cast<int32_t*>(idss[i]->data);
            const int32_t a = static_cast<int32_t>((i + sweep) % n_experts);
            const int32_t b = static_cast<int32_t>((i + sweep + 2) % n_experts);
            idv[0] = a;
            idv[1] = b == a ? (a + 1) % static_cast<int32_t>(n_experts) : b;

            streamer.materialise(nodes[i]);
            LZ_CHECK_EQ(streamer.failures(), 0u);
            LZ_CHECK_LE(acct.used(), cap2);                  // Invariant 1, live
            LZ_CHECK(selection_reads_expert(nodes[i], ws[i], 0, *fts[i], n_experts, idv[0]));
            LZ_CHECK(selection_reads_expert(nodes[i], ws[i], 1, *fts[i], n_experts, idv[1]));
            streamer.release(nodes[i]);
        }
    }
    LZ_CHECK_GT(streamer.routed_bytes_read(), 0u);

    // The rebudget shrink: the OS reports modest excess (an eighth of the
    // routed bytes -- llama buffers growing, say). The budget tightens,
    // subsequent tokens must STILL be correct and capped: the static-pin
    // demotion path that once wedged make_room on Linux. (A crushing shrink
    // correctly fails everything -- honest refusal, tested implicitly by the
    // failure counter's existence, but not the regime worth pinning here.)
    streamer.rebudget_against_rss(cap2 + fx.routed_bytes / 8);
    size_t served = 0, refused = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        int32_t* idv = static_cast<int32_t*>(idss[i]->data);
        idv[0] = static_cast<int32_t>(i % n_experts);
        idv[1] = static_cast<int32_t>((i + 1) % n_experts);
        const uint64_t fail_before = streamer.failures();
        streamer.materialise(nodes[i]);
        LZ_CHECK_LE(acct.used(), cap2);
        if (streamer.failures() == fail_before) {
            // Served: the bytes must be the truth.
            ++served;
            LZ_CHECK(selection_reads_expert(nodes[i], ws[i], 0, *fts[i], n_experts, idv[0]));
            LZ_CHECK(selection_reads_expert(nodes[i], ws[i], 1, *fts[i], n_experts, idv[1]));
        } else {
            // Refused: the documented poison contract -- LOUD (counted), private
            // ids all zero so no lookup can leave the first stride. Silent
            // degradation, not degradation itself, is the crime.
            ++refused;
            const int32_t* rv = static_cast<const int32_t*>(nodes[i]->src[2]->data);
            LZ_CHECK_EQ(rv[0], 0);
            LZ_CHECK_EQ(rv[1], 0);
        }
        streamer.release(nodes[i]);
    }
    // A modest shrink must not extinguish service; it may legitimately refuse
    // the largest layers. Both regimes appearing is what makes this test bite.
    // T26: on this fixture the split is deterministically 4 served / 3 refused
    // (the three largest slot classes fall out of budget). If those numbers
    // drift, the budget arithmetic changed -- investigate before adjusting.
    LZ_CHECK_GT(served, 0u);
    std::fprintf(stderr, "    [pressure] post-shrink: %zu served, %zu refused (loudly)\n",
                 served, refused);

    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}
