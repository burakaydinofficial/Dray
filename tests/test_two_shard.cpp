// Multi-shard GGUF coverage (swarm S20): "right offset, wrong file" has
// shipped once before, and nothing guarded it. The split fixture writes two
// real GGUF shards (full KV block, split.no/count keys, llama's naming), and
// this test proves the plan spans both files and that the streamer, asked for
// an expert whose tensor lives in the SECOND shard, returns bytes that can
// only have come from that file -- the fixture's per-tensor fill seeds make a
// same-offset read from the wrong shard a guaranteed mismatch.

#include "harness.h"

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
    dray::testfix::TempDir dir{"two_shard"};
    Fixture fx;
    Env() { fx = dray::testfix::make_gguf_split(dir.path(), "split", GgufSpec{}); }
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

LZ_TEST(plan_spans_both_shards) {
    const Fixture& fx = env().fx;
    LZ_CHECK(fx.ok);
    LZ_CHECK_EQ(fx.shard_paths.size(), 2u);

    std::string err;
    dray::plan::Plan plan =
        dray::plan::build_plan(fx.shard_paths[0].string(), 1ull << 30, 512, &err);
    LZ_CHECK(err.empty());
    LZ_CHECK(plan.feasible);
    LZ_CHECK_EQ(plan.shard_paths.size(), 2u);

    // Every fixture tensor appears in the plan, on the shard the fixture put
    // it on, at the offset the shard's own table states.
    size_t on_second = 0;
    for (const auto& ft : fx.tensors) {
        bool found = false;
        for (const auto& pt : plan.tensors) {
            if (pt.name != ft.name) continue;
            found = true;
            LZ_CHECK_EQ(static_cast<int32_t>(pt.shard), ft.shard_index);
            LZ_CHECK_EQ(pt.offset, ft.offset);
            LZ_CHECK_EQ(pt.bytes, ft.bytes);
            if (ft.shard_index == 1) ++on_second;
            break;
        }
        LZ_CHECK(found);
    }
    LZ_CHECK_GT(on_second, 0u);
}

LZ_TEST(streamer_reads_second_shard_bytes_not_first) {
    const Fixture& fx = env().fx;
    LZ_CHECK(fx.ok);

    std::string err;
    dray::plan::Plan probe =
        dray::plan::build_plan(fx.shard_paths[0].string(), 1ull << 30, 512, &err);
    LZ_CHECK(err.empty());
    const uint64_t cap = (fx.file_size - fx.routed_bytes) + fx.routed_bytes / 4 +
                         probe.floor.kv_cache + probe.floor.recurrent_state +
                         probe.floor.compute_scratch + (256u << 10);
    // Pass-4b: the sentinel reservation is honest now (its result was once
    // discarded), so the tight cap must fund the widest tensor explicitly.
    uint64_t widest = 0;
    for (const auto& wt : fx.tensors) widest = std::max<uint64_t>(widest, wt.bytes);
    const uint64_t cap2 = cap + widest + (8ull << 20);   // sentinel has an 8 MiB floor
    dray::plan::Plan plan =
        dray::plan::build_plan(fx.shard_paths[0].string(), cap2, 512, &err);
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

    // A routed gate tensor that lives in the SECOND shard.
    const dray::testfix::FixtureTensor* ft = nullptr;
    for (const auto& t : fx.tensors) {
        if (t.shard_index == 1 && t.name.find("ffn_gate_exps") != std::string::npos) {
            ft = &t;
            break;
        }
    }
    LZ_CHECK(ft != nullptr);

    const uint32_t n_experts = fx.spec.n_experts;
    ggml_init_params wp{};
    wp.mem_size = ggml_tensor_overhead() * 4;
    wp.no_alloc = true;
    ggml_context* wctx = ggml_init(wp);
    LZ_CHECK(wctx != nullptr);
    ggml_tensor* w = ggml_new_tensor_3d(wctx, static_cast<ggml_type>(ft->ggml_type),
                                        ft->ne[0], ft->ne[1], ft->ne[2]);
    ggml_set_name(w, ft->name.c_str());
    ggml_backend_buffer_t wbuf =
        ggml_backend_alloc_ctx_tensors_from_buft(wctx, streamer.buft());
    LZ_CHECK(wbuf != nullptr);

    ggml_init_params gp{};
    gp.mem_size = 4 * 1024 * 1024;
    gp.no_alloc = false;
    ggml_context* gctx = ggml_init(gp);
    ggml_tensor* x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, ft->ne[0], 1);
    ggml_tensor* ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, 2, 1);
    int32_t* idv = static_cast<int32_t*>(ids->data);
    idv[0] = 1;
    idv[1] = 3;
    ggml_tensor* node = ggml_mul_mat_id(gctx, w, x, ids);

    streamer.materialise(node);
    LZ_CHECK_EQ(streamer.failures(), 0u);
    LZ_CHECK_GT(streamer.routed_bytes_read(), 0u);
    // The fill seed is per-tensor: bytes matching THIS tensor's pattern can
    // only have come from shard 2 at this tensor's offset.
    LZ_CHECK(selection_reads_expert(node, w, 0, *ft, n_experts, 1));
    LZ_CHECK(selection_reads_expert(node, w, 1, *ft, n_experts, 3));
    streamer.release(node);

    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}
