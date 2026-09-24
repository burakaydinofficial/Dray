// The graph-reuse differential (swarm S16): the axis every historical
// silent-corruption bug lived on, exercised through the PRODUCT's own mapping.
//
// llama.cpp reuses graphs across decode steps: the same MUL_MAT_ID node object
// is evaluated again with new router ids written into the same ids tensor. The
// 2026-08-13 bug derived the second token's mapping from its OWN already-
// remapped ids and corrupted every token after the first, while every counter
// read clean. test_compaction cannot regress this: it re-implements the
// mapping locally and never links a product symbol. This test constructs a
// real Streamer over the synthetic GGUF, materialises one MUL_MAT_ID node
// TWICE with different selections written in place between calls, and checks
// the bytes the streamer placed against the fixture's deterministic fill
// pattern -- a pure-function reference that needs no file I/O and no compute.

#include "harness.h"

#include <cstdio>
#include <cstring>
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
    dray::testfix::TempDir dir{"graph_reuse"};
    Fixture fx;
    Env() { fx = dray::testfix::make_gguf(dir.path(), "reuse.gguf", GgufSpec{}); }
};

Env& env() {
    static Env e;
    return e;
}

struct Ctx {
    ggml_context* ctx = nullptr;
    ~Ctx() { if (ctx) ggml_free(ctx); }
};

ggml_context* make_ctx(size_t mb) {
    ggml_init_params p{};
    p.mem_size   = mb * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    return ggml_init(p);
}

// What MUL_MAT_ID actually consumes: for logical selection j, the node's
// CURRENT src[2] (the streamer may have installed its private remap) names the
// physical row; that row of w must hold expert e's bytes per the fixture's
// fill function. Layout-agnostic on purpose: fresh compact regions serve in
// selection order, expert-slot cache hits serve wherever the slot lives, and
// both are correct exactly when this predicate holds.
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
            // Diagnostic: whose bytes ARE these?
            for (int32_t ce = 0; ce < static_cast<int32_t>(n_experts); ++ce) {
                bool all = true;
                for (uint64_t q = 0; q < per && all; ++q) {
                    all = got[q] == fill_byte(t.seed, static_cast<uint64_t>(ce) * per + q);
                }
                if (all) {
                    std::fprintf(stderr, "    [diag] j=%zu row=%d wanted expert %d, holds expert %d\n",
                                 j, row, e, ce);
                    return false;
                }
            }
            std::fprintf(stderr, "    [diag] j=%zu row=%d wanted expert %d, holds NO known expert (mismatch at byte %llu)\n",
                         j, row, e, static_cast<unsigned long long>(i));
            return false;
        }
    }
    return true;
}

}  // namespace

LZ_TEST(materialise_remaps_fresh_ids_on_graph_reuse) {
    const Fixture& fx = env().fx;
    LZ_CHECK(fx.ok);

    // The cap must be TIGHT: at a generous cap the whole fixture is statically
    // pinned, materialise serves the resident whole tensor, and the compact
    // path -- the thing under test -- never runs (slot j would be expert j).
    // Everything except the routed experts, plus a quarter of them, forces the
    // gather path while keeping the plan feasible.
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

    // The first MoE block. The fixture leads with one dense block.
    const auto* ft = fx.find("blk.1.ffn_gate_exps.weight");
    LZ_CHECK(ft != nullptr);
    const uint32_t n_experts = env().fx.spec.n_experts;   // 4
    const int64_t k = 2;

    // The weight must be BOUND the way llama binds it: allocated through the
    // streamer's buffer type, whose init_tensor hook registers it (im.bound
    // gates the compact path). A plain ctx tensor is invisible to the
    // streamer, and materialise correctly leaves foreign tensors alone --
    // measured: untouched zeros, src[2] unswapped.
    ggml_init_params wp{};
    wp.mem_size = ggml_tensor_overhead() * 4;
    wp.no_alloc = true;
    ggml_context* wctx = ggml_init(wp);
    LZ_CHECK(wctx != nullptr);
    ggml_tensor* w = ggml_new_tensor_3d(wctx, static_cast<ggml_type>(ft->ggml_type),
                                        ft->ne[0], ft->ne[1], ft->ne[2]);
    ggml_set_name(w, "blk.1.ffn_gate_exps.weight");
    ggml_backend_buffer_t wbuf =
        ggml_backend_alloc_ctx_tensors_from_buft(wctx, streamer.buft());
    LZ_CHECK(wbuf != nullptr);

    Ctx c;
    c.ctx = make_ctx(64);
    LZ_CHECK(c.ctx != nullptr);

    ggml_tensor* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, ft->ne[0], 1);
    ggml_tensor* ids = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, k, 1);
    int32_t* idv = static_cast<int32_t*>(ids->data);

    // Token 1: experts {2, 0}.
    idv[0] = 2; idv[1] = 0;
    ggml_tensor* node = ggml_mul_mat_id(c.ctx, w, x, ids);

    streamer.materialise(node);
    LZ_CHECK_EQ(streamer.failures(), 0u);
    // Precondition: the gather path actually ran (whole-resident service would
    // make every check below vacuously test the wrong mechanism).
    LZ_CHECK_GT(streamer.routed_bytes_read(), 0u);
    LZ_CHECK(selection_reads_expert(node, w, 0, *ft, n_experts, 2));
    LZ_CHECK(selection_reads_expert(node, w, 1, *ft, n_experts, 0));
    streamer.release(node);

    // Token 2: llama reuses the graph -- SAME node, SAME ids tensor, new
    // values written in place. The 2026-08-13 bug read its own remap here.
    idv[0] = 3; idv[1] = 1;

    streamer.materialise(node);
    LZ_CHECK_EQ(streamer.failures(), 0u);
    LZ_CHECK(selection_reads_expert(node, w, 0, *ft, n_experts, 3));
    LZ_CHECK(selection_reads_expert(node, w, 1, *ft, n_experts, 1));
    streamer.release(node);

    // Token 3: repeat an earlier expert in a different slot, plus a fresh one.
    // Catches mappings that "stick" per expert id rather than per selection.
    idv[0] = 1; idv[1] = 2;

    streamer.materialise(node);
    LZ_CHECK_EQ(streamer.failures(), 0u);
    LZ_CHECK(selection_reads_expert(node, w, 0, *ft, n_experts, 1));
    LZ_CHECK(selection_reads_expert(node, w, 1, *ft, n_experts, 2));
    streamer.release(node);

    // Token 4 (T26): STRIDED ids, the layout ggml_argsort_top_k actually
    // produces -- an [n_expert, 1] backing viewed as [k, 1], with the tail
    // poisoned so a contiguous walk reads obviously wrong values. This is the
    // one ids shape the 2026-08-13 bug class lived on, and it was
    // ctest-unreachable until now.
    {
        ggml_tensor* backing = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32,
                                                  static_cast<int64_t>(n_experts), 1);
        int32_t* bv = static_cast<int32_t*>(backing->data);
        for (uint32_t q = 0; q < n_experts; ++q) bv[q] = -12345;   // poison
        bv[0] = 3; bv[1] = 0;
        ggml_tensor* sids = ggml_view_2d(c.ctx, backing, k, 1, backing->nb[1], 0);
        ggml_tensor* node2 = ggml_mul_mat_id(c.ctx, w, x, sids);
        streamer.materialise(node2);
        LZ_CHECK_EQ(streamer.failures(), 0u);
        LZ_CHECK(selection_reads_expert(node2, w, 0, *ft, n_experts, 3));
        LZ_CHECK(selection_reads_expert(node2, w, 1, *ft, n_experts, 0));
        streamer.release(node2);
    }

    // T27: both siblings free these; leaking them left a ggml buffer
    // outliving the Streamer its callbacks point at.
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}
