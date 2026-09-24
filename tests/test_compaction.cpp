// Differential test for expert compaction.
//
// This is the check that makes the streamer's central trick ASSERTABLE rather
// than inferred from output that reads plausibly. The trick: instead of
// materialising a whole [cols, rows, n_expert] expert tensor, materialise only the
// experts a token selects into a compact region with the same stride, and give the
// mul_mat_id node a private ids tensor holding compact indices.
//
// If that is wrong it does not crash. It produces fluent text from the wrong
// weights. Six bugs in this component were found only after the output "looked
// fine", so the bar here is bit-identical results, not plausibility.
//
// Every case computes twice and compares:
//    reference: mul_mat_id(full_experts,    x, original_ids)
//    compacted: mul_mat_id(compact_experts, x, private_remapped_ids)
//
// THE IDS ARE A STRIDED VIEW ON PURPOSE. In a real graph selected_experts comes
// from ggml_argsort_top_k, which views the first k entries of each row of an
// [n_expert, n_tokens] argsort result -- so nb[1] is the FULL row, not k*4. A
// fixture with contiguous ids passes while the real thing reads out of bounds;
// that exact difference is why a 1-token run once "worked" and 5 tokens faulted.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"

#include "harness.h"

namespace {

constexpr int64_t kCols     = 8;    // input width  (as->ne[0])
constexpr int64_t kRows     = 4;    // output width (as->ne[1])
constexpr int64_t kNExpert  = 16;

// Expert e is filled with a value that depends on e, so indexing the wrong expert
// changes the result rather than happening to agree.
float expert_value(int64_t e, int64_t r, int64_t c) {
    return static_cast<float>(e + 1) * 100.0f + static_cast<float>(r) * 10.0f +
           static_cast<float>(c) * 0.5f;
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

// Fills a [kCols, kRows, n_expert] F32 tensor, taking experts from `which` in
// order. Passing all of 0..n_expert-1 builds the full tensor; passing a subset
// builds the compact region the streamer would produce.
void fill_experts(ggml_tensor* t, const std::vector<int32_t>& which) {
    float* d = static_cast<float*>(t->data);
    for (size_t k = 0; k < which.size(); ++k) {
        for (int64_t r = 0; r < kRows; ++r) {
            for (int64_t c = 0; c < kCols; ++c) {
                d[k * kRows * kCols + r * kCols + c] = expert_value(which[k], r, c);
            }
        }
    }
}

// Builds ids the way ggml_argsort_top_k does: an [n_expert, n_tokens] backing
// tensor, viewed as [k, n_tokens]. The view's nb[1] is the full backing row.
ggml_tensor* make_strided_ids(ggml_context* ctx, const std::vector<int32_t>& sel,
                              int64_t k, int64_t n_tokens) {
    ggml_tensor* backing = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kNExpert, n_tokens);
    int32_t* b = static_cast<int32_t*>(backing->data);
    // Poison the tail so a contiguous walk reads obviously wrong values instead of
    // accidentally-valid ones.
    for (int64_t i = 0; i < kNExpert * n_tokens; ++i) b[i] = -12345;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            b[t * kNExpert + j] = sel[static_cast<size_t>(t * k + j)];
        }
    }
    ggml_tensor* view = ggml_view_2d(ctx, backing, k, n_tokens,
                                     backing->nb[1], /*offset*/ 0);
    return view;
}

std::vector<float> run_mul_mat_id(ggml_context* ctx, ggml_tensor* experts,
                                  ggml_tensor* x, ggml_tensor* ids) {
    ggml_tensor* out = ggml_mul_mat_id(ctx, experts, x, ids);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    if (st != GGML_STATUS_SUCCESS) return {};
    const int64_t n = ggml_nelements(out);
    std::vector<float> r(static_cast<size_t>(n));
    std::memcpy(r.data(), out->data, static_cast<size_t>(n) * sizeof(float));
    return r;
}

// The streamer's algorithm, extracted so the test exercises the same logic:
// unique experts in first-seen order, and each id replaced by its position.
void derive(const std::vector<int32_t>& sel, std::vector<int32_t>* uniq,
            std::vector<int32_t>* remapped) {
    uniq->clear();
    remapped->assign(sel.size(), 0);
    for (size_t i = 0; i < sel.size(); ++i) {
        auto f = std::find(uniq->begin(), uniq->end(), sel[i]);
        if (f == uniq->end()) { uniq->push_back(sel[i]); f = uniq->end() - 1; }
        (*remapped)[i] = static_cast<int32_t>(f - uniq->begin());
    }
}

// How the compact region is ordered. Sorted is a NEGATIVE CONTROL: it is the
// plausible-looking bug where the region is built in ascending expert order while
// the ids are remapped by first-seen position. If the harness cannot tell that
// apart from the correct mapping, it cannot detect the real bugs either.
enum class Order { FirstSeen, SortedButIdsNotFixed };

// One full differential case.
bool differential(const std::vector<int32_t>& sel, int64_t k, int64_t n_tokens,
                  std::string* why, Order order = Order::FirstSeen) {
    Ctx a, b;
    a.ctx = make_ctx(64);
    b.ctx = make_ctx(64);
    if (!a.ctx || !b.ctx) { *why = "ggml_init failed"; return false; }

    std::vector<int32_t> all(kNExpert);
    for (int32_t i = 0; i < kNExpert; ++i) all[static_cast<size_t>(i)] = i;

    std::vector<int32_t> uniq, remapped;
    derive(sel, &uniq, &remapped);
    if (order == Order::SortedButIdsNotFixed) {
        std::sort(uniq.begin(), uniq.end());   // remapped deliberately left stale
    }

    // Identical input for both runs.
    std::vector<float> xbuf(static_cast<size_t>(kCols * k * n_tokens));
    for (size_t i = 0; i < xbuf.size(); ++i) {
        xbuf[i] = 0.25f + static_cast<float>(i % 7) * 0.125f;
    }

    // --- reference: whole tensor, original ids -----------------------------
    ggml_tensor* full = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, kCols, kRows, kNExpert);
    fill_experts(full, all);
    ggml_tensor* xa = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, kCols, k, n_tokens);
    std::memcpy(xa->data, xbuf.data(), xbuf.size() * sizeof(float));
    ggml_tensor* ids_a = make_strided_ids(a.ctx, sel, k, n_tokens);
    const std::vector<float> ref = run_mul_mat_id(a.ctx, full, xa, ids_a);

    // --- compacted: only selected experts, private contiguous remapped ids --
    // ne[2] stays n_expert-shaped only in the real streamer (where the tensor is
    // pre-existing); here the compact tensor is genuinely uniq.size() deep, which
    // is the stricter shape and still must agree.
    ggml_tensor* comp = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, kCols, kRows,
                                           static_cast<int64_t>(uniq.size()));
    fill_experts(comp, uniq);
    ggml_tensor* xb = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, kCols, k, n_tokens);
    std::memcpy(xb->data, xbuf.data(), xbuf.size() * sizeof(float));
    ggml_tensor* ids_b = ggml_new_tensor_2d(b.ctx, GGML_TYPE_I32, k, n_tokens);
    std::memcpy(ids_b->data, remapped.data(), remapped.size() * sizeof(int32_t));
    const std::vector<float> got = run_mul_mat_id(b.ctx, comp, xb, ids_b);

    if (ref.empty() || got.empty()) { *why = "graph compute failed"; return false; }
    if (ref.size() != got.size()) { *why = "output size mismatch"; return false; }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            *why = "element " + std::to_string(i) + ": reference " +
                   std::to_string(ref[i]) + " vs compacted " + std::to_string(got[i]);
            return false;
        }
    }
    return true;
}

// harness.h offers a boolean check and a message-only failure, not both at once.
// Compose them so a mismatch reports WHICH element diverged, which is the only
// part of a failure here that is actually diagnostic.
void expect_identical(const std::vector<int32_t>& sel, int64_t k, int64_t n_tokens) {
    std::string why;
    const bool ok = differential(sel, k, n_tokens, &why);
    if (!ok) LZ_FAIL(why);
    LZ_CHECK(ok);
}

}  // namespace

LZ_TEST(compaction_matches_whole_tensor_single_token) {
    // k=4 distinct experts, one token.
    expect_identical({3, 7, 1, 12}, 4, 1);
}

LZ_TEST(compaction_matches_whole_tensor_multi_token) {
    // The case a contiguous-ids fixture would pass and the real strided view
    // fails: with more than one token, nb[1] is a full backing row, not k*4.
    expect_identical({3, 7, 1, 12, 0, 5, 9, 2, 15, 4, 6, 11}, 4, 3);
}

LZ_TEST(compaction_handles_experts_repeated_within_a_token) {
    // uniq is shorter than k: the compact region must be indexed by position, not
    // by a per-slot assumption.
    expect_identical({5, 5, 2, 5}, 4, 1);
}

LZ_TEST(compaction_handles_experts_shared_across_tokens) {
    // Token 2 reuses token 1's experts in a different order -- the mapping spans
    // the whole ids tensor, not one token.
    expect_identical({8, 1, 3, 4, 3, 1, 8, 4}, 4, 2);
}

LZ_TEST(compaction_handles_unsorted_and_descending_selection) {
    // First-seen order is NOT sorted order. If the compact region were built in
    // ascending expert order while ids were remapped by first-seen position, this
    // is the case that catches it.
    expect_identical({15, 14, 13, 12, 0, 1, 2, 3}, 4, 2);
}

LZ_TEST(compaction_handles_every_expert_selected) {
    // uniq == n_expert: compaction degenerates to the whole tensor and must still
    // agree, including the identity remapping.
    std::vector<int32_t> sel;
    for (int32_t e = 0; e < kNExpert; ++e) sel.push_back(e);
    expect_identical(sel, kNExpert, 1);
}

LZ_TEST(compaction_handles_a_single_expert_everywhere) {
    // Degenerate but real: a router that collapses. uniq.size() == 1.
    expect_identical({6, 6, 6, 6, 6, 6}, 3, 2);
}

LZ_TEST(negative_control_the_harness_detects_a_wrong_mapping) {
    // A test that cannot fail proves nothing. Every bug this file exists to catch
    // produces a WRONG RESULT, never an error, so the harness must be shown to
    // distinguish right from wrong rather than merely to run.
    //
    // The injected bug is the realistic one: build the compact region in ascending
    // expert order while the ids still say first-seen position. The selection below
    // is descending, so the two orderings genuinely disagree.
    std::string why;
    const bool identical = differential({15, 14, 13, 12, 0, 1, 2, 3}, 4, 2, &why,
                                        Order::SortedButIdsNotFixed);
    if (identical) {
        LZ_FAIL("harness did not detect a deliberately wrong expert ordering -- "
                "the differential check has no teeth and every PASS above is void");
    }
    LZ_CHECK(!identical);
}
