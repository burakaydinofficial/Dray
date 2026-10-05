// The residency planner decides almost all achieved performance at small caps,
// and it is the only component that can refuse admission. Both of those are
// arithmetic over a tensor table, so both are fully testable against a synthetic
// GGUF -- which is the only way they get tested at all, since the smallest real
// target is 143 GB.
//
// Two things here are deliberately hostile to a plausible-looking implementation:
//
//   * the fixture's per-layer expert sizes are non-uniform, so a planner that
//     sizes slots from one global number produces wrong slot classes;
//   * the fixture's <arch>.expert_feed_forward_length KV reports only the
//     SMALLEST of the three widths, so a planner that trusts metadata over the
//     tensor table (Invariant 3) gets the big layers wrong and only the big
//     layers wrong.

#include "harness.h"

#include "fixtures/make_gguf.h"
#include "fixtures/temp_dir.h"
#include "plan/residency.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using dray::plan::build_plan;
using dray::plan::Plan;
using dray::plan::TensorClass;
using dray::testfix::Fixture;
using dray::testfix::FixtureClass;
using dray::testfix::FixtureLayer;
using dray::testfix::FixtureTensor;
using dray::testfix::GgufSpec;

namespace {

// One fixture per test binary: the file is ~6 MB and every case wants the same
// one. Built on first use so a construction failure is reported by a case rather
// than by a static initialiser.
struct Env {
    dray::testfix::TempDir dir{"residency"};
    Fixture fx;
    Env() { fx = dray::testfix::make_gguf(dir.path(), "tiny.gguf", GgufSpec{}); }
};

const Fixture& fixture() {
    static Env e;
    return e.fx;
}

std::string path() { return fixture().path.string(); }

constexpr uint64_t kGenerousCap = 512ull << 20;

TensorClass expected_class(FixtureClass c) {
    switch (c) {
        case FixtureClass::RouterGate:        return TensorClass::RouterGate;
        case FixtureClass::NormOrBias:        return TensorClass::NormOrBias;
        case FixtureClass::RoutedExpert:      return TensorClass::RoutedExpert;
        case FixtureClass::UnconditionalBulk: return TensorClass::UnconditionalBulk;
        case FixtureClass::RowSliced:         return TensorClass::RowSliced;
    }
    return TensorClass::UnconditionalBulk;
}

uint64_t sum_floor_fields(const dray::plan::Floor& f) {
    return f.router_gates + f.norms_biases + f.kv_cache + f.recurrent_state +
           f.prefill_activation + f.compute_scratch + f.io_staging;
}

Plan plan_at(uint64_t cap, uint32_t n_ctx) {
    std::string err;
    return build_plan(path(), cap, n_ctx, &err);
}

}  // namespace

LZ_TEST(fixture_is_usable) {
    const Fixture& fx = fixture();
    LZ_REQUIRE(fx.ok);
    LZ_CHECK(fx.error.empty());
    LZ_CHECK_GT(fx.tensors.size(), 0u);
    LZ_CHECK_GE(fx.distinct_slot_bytes().size(), 3u);
}

LZ_TEST(every_tensor_is_classified_with_the_table_s_own_bytes) {
    const Fixture& fx = fixture();
    std::string err;
    const Plan p = build_plan(path(), kGenerousCap, 512, &err);

    LZ_REQUIRE(p.feasible);
    LZ_CHECK(err.empty());
    LZ_CHECK_EQ(p.tensors.size(), fx.tensors.size());

    for (const FixtureTensor& want : fx.tensors) {
        const dray::plan::TensorInfo* got = nullptr;
        for (const dray::plan::TensorInfo& t : p.tensors) {
            if (t.name == want.name) {
                got = &t;
                break;
            }
        }
        if (got == nullptr) {
            LZ_FAIL("missing tensor in plan: " + want.name);
            continue;
        }
        // Byte figures come from the tensor table, never from an assumed
        // bits-per-weight (Invariant 3). The fixture mixes Q4_K, Q6_K, Q8_0 and
        // F32 precisely so a single assumed bpw cannot satisfy all four.
        LZ_CHECK_EQ(got->bytes, want.bytes);
        LZ_CHECK_EQ(got->offset, want.offset);
        LZ_CHECK_EQ(got->ggml_type, want.ggml_type);
        LZ_CHECK_EQ(got->layer, want.layer);
        LZ_CHECK_EQ(static_cast<int>(got->cls), static_cast<int>(expected_class(want.cls)));
    }
}

LZ_TEST(router_gates_are_never_classified_as_streamable) {
    // A non-resident gate must be read before you know which experts to read:
    // a serialized round trip per layer per token that can never be overlapped.
    // Misclassifying one as RoutedExpert would not fail any byte total, so it
    // gets its own case.
    std::string err;
    const Plan p = build_plan(path(), kGenerousCap, 512, &err);
    LZ_REQUIRE(p.feasible);

    int gates = 0;
    for (const dray::plan::TensorInfo& t : p.tensors) {
        if (t.name.find("ffn_gate_inp") != std::string::npos) {
            ++gates;
            LZ_CHECK_EQ(static_cast<int>(t.cls), static_cast<int>(TensorClass::RouterGate));
        }
        // The near-miss names that a substring match would swallow.
        if (t.name.find("ffn_gate_exps") != std::string::npos) {
            LZ_CHECK_EQ(static_cast<int>(t.cls), static_cast<int>(TensorClass::RoutedExpert));
        }
        if (t.name.find("shexp") != std::string::npos) {
            LZ_CHECK_EQ(static_cast<int>(t.cls), static_cast<int>(TensorClass::UnconditionalBulk));
        }
    }
    LZ_CHECK_EQ(gates, static_cast<int>(fixture().n_moe_layers));
}

LZ_TEST(slot_classes_are_per_layer_and_come_out_non_uniform) {
    const Fixture& fx = fixture();
    std::string err;
    const Plan p = build_plan(path(), kGenerousCap, 512, &err);
    LZ_REQUIRE(p.feasible);

    LZ_REQUIRE_EQ(p.slot_classes.size(), static_cast<size_t>(fx.n_moe_layers));

    std::vector<uint64_t> seen;
    for (const dray::plan::LayerSlotClass& sc : p.slot_classes) {
        LZ_CHECK_GE(sc.layer, 0);
        LZ_CHECK_EQ(sc.n_experts, fx.spec.n_experts);

        const FixtureLayer* want = nullptr;
        for (const FixtureLayer& l : fx.layers) {
            if (l.layer == sc.layer) {
                want = &l;
                break;
            }
        }
        if (want == nullptr || !want->is_moe) {
            LZ_FAIL("slot class for a layer that has no experts: " + std::to_string(sc.layer));
            continue;
        }
        LZ_CHECK_EQ(sc.slot_bytes, want->slot_bytes);
        seen.push_back(sc.slot_bytes);
    }

    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());

    // The whole reason slot classes are per-layer: M3 UD-Q2_K_XL has three
    // distinct sizes and one global slot sized to the largest wastes ~26% of the
    // arena on 54 of 57 layers.
    LZ_CHECK_EQ(seen.size(), fx.distinct_slot_bytes().size());
    LZ_CHECK_GE(seen.size(), 3u);
    if (seen.size() >= 2) {
        LZ_CHECK_GT(seen.back(), seen.front());
    }
}

LZ_TEST(floor_totals_its_own_fields_and_the_gates_are_all_of_them) {
    const Fixture& fx = fixture();
    std::string err;
    const Plan p = build_plan(path(), kGenerousCap, 512, &err);
    LZ_REQUIRE(p.feasible);

    LZ_CHECK_EQ(p.floor.router_gates, fx.router_gate_bytes);
    LZ_CHECK_EQ(p.floor.norms_biases, fx.norm_bias_bytes);
    LZ_CHECK_EQ(p.floor.total(), sum_floor_fields(p.floor));

    // Some state term must exist for a real context.
    LZ_CHECK_GT(p.floor.kv_cache + p.floor.recurrent_state, 0u);
    LZ_CHECK_GT(p.floor.prefill_activation, 0u);
}

LZ_TEST(cache_budget_is_the_cap_minus_the_floor_never_the_cap_plus_it) {
    std::string err;
    const uint64_t cap = kGenerousCap;
    const Plan p = build_plan(path(), cap, 512, &err);
    LZ_REQUIRE(p.feasible);

    LZ_CHECK_EQ(p.cap, cap);
    LZ_CHECK_EQ(p.cache_budget, cap - p.floor.total());
    // The invariant stated as the user sees it: everything the engine will hold
    // resident fits inside the number they asked for.
    LZ_CHECK_LE(p.floor.total() + p.cache_budget, cap);
}

LZ_TEST(the_state_terms_scale_with_n_ctx) {
    std::string err;
    const Plan a = build_plan(path(), kGenerousCap, 1024, &err);
    const Plan b = build_plan(path(), kGenerousCap, 4096, &err);
    LZ_REQUIRE(a.feasible);
    LZ_REQUIRE(b.feasible);

    // KV is reserved for the REQUESTED n_ctx up front, not grown on demand, so
    // a bigger context must cost more floor before a single token is generated.
    LZ_CHECK_GT(b.floor.kv_cache + b.floor.recurrent_state,
                a.floor.kv_cache + a.floor.recurrent_state);
    LZ_CHECK_GT(b.floor.prefill_activation, a.floor.prefill_activation);
    LZ_CHECK_GT(b.floor.total(), a.floor.total());
    LZ_CHECK_LT(b.cache_budget, a.cache_budget);
}

LZ_TEST(byte_totals_split_the_unconditional_stream_from_the_routed_population) {
    const Fixture& fx = fixture();
    std::string err;
    const Plan p = build_plan(path(), kGenerousCap, 512, &err);
    LZ_REQUIRE(p.feasible);

    LZ_CHECK_EQ(p.routed_bytes, fx.routed_bytes);
    LZ_CHECK_EQ(p.unconditional_bytes, fx.unconditional_bytes);
    LZ_CHECK_EQ(p.row_sliced_bytes, fx.row_sliced_bytes);
    LZ_CHECK_EQ(p.n_moe_layers, fx.n_moe_layers);
    LZ_CHECK_EQ(p.n_experts, fx.spec.n_experts);
    LZ_CHECK_EQ(p.n_expert_used, fx.spec.n_expert_used);

    // cold_bytes_per_token = k * n_moe_layers * mean slot bytes. With per-layer
    // slot sizes that is exactly k * sum(slot_bytes); the tolerance is only for
    // integer rounding of the mean.
    const uint64_t want = fx.expected_cold_bytes_per_token();
    const uint64_t slack = static_cast<uint64_t>(fx.n_moe_layers) * fx.spec.n_expert_used;
    LZ_CHECK_GE(p.cold_bytes_per_token + slack, want);
    LZ_CHECK_LE(p.cold_bytes_per_token, want + slack);
}

LZ_TEST(projected_bytes_per_token_never_exceeds_the_fully_cold_figure) {
    std::string err;
    const Plan p = build_plan(path(), kGenerousCap, 512, &err);
    LZ_REQUIRE(p.feasible);
    LZ_CHECK_LE(p.projected_bytes_per_token(),
                p.unconditional_bytes + p.cold_bytes_per_token);
}

LZ_TEST(bigger_caps_never_cost_more_bytes_per_token) {
    const Fixture& fx = fixture();
    const Plan base = plan_at(kGenerousCap, 512);
    LZ_REQUIRE(base.feasible);
    const uint64_t floor_bytes = base.floor.total();

    // Start from min_cap, not floor+4096: a cap that covers every mandatory byte
    // but leaves no room for one MoE layer's working set is CORRECTLY refused,
    // because it could not compute a single layer. min_cap is that threshold.
    const uint64_t start_cap = base.min_cap;
    LZ_REQUIRE_GT(start_cap, floor_bytes);

    uint64_t previous = UINT64_MAX;
    for (int step = 0; step <= 8; ++step) {
        const uint64_t extra =
            (fx.unconditional_bytes + fx.routed_bytes) * static_cast<uint64_t>(step) / 4u;
        const Plan p = plan_at(start_cap + extra, 512);
        if (!p.feasible) {
            LZ_FAIL("cap above the floor was refused at step " + std::to_string(step));
            continue;
        }
        const uint64_t got = p.projected_bytes_per_token();
        LZ_CHECK_LE(got, previous);
        previous = got;
    }

    // A cap that holds the whole model resident reads nothing per token. If this
    // is not zero, some term is being charged that no longer comes off disk.
    const Plan huge = plan_at(floor_bytes + fx.unconditional_bytes + fx.routed_bytes + (1 << 20), 512);
    LZ_REQUIRE(huge.feasible);
    LZ_CHECK_EQ(huge.projected_bytes_per_token(), 0u);
}

LZ_TEST(a_cap_that_cannot_cover_the_floor_is_refused_not_shrunk) {
    // Bracket a cap between the floor at a small context and the floor at a big
    // one, so the refusal is certain to be about n_ctx and not about the model.
    const Plan small_ctx = plan_at(kGenerousCap, 256);
    const Plan big_ctx = plan_at(kGenerousCap, 8192);
    LZ_REQUIRE(small_ctx.feasible);
    LZ_REQUIRE(big_ctx.feasible);
    LZ_REQUIRE_GT(big_ctx.floor.total(), small_ctx.floor.total());

    const uint64_t cap = (small_ctx.floor.total() + big_ctx.floor.total()) / 2;

    std::string err;
    const Plan p = build_plan(path(), cap, 8192, &err);

    LZ_REQUIRE(!p.feasible);
    LZ_CHECK_EQ(p.cache_budget, 0u);
    LZ_CHECK(!p.refusal.empty());

    // "Refuses admission" is only half of it: a refusal that does not say what
    // WOULD work leaves the user guessing, and guessing here costs hours.
    LZ_REQUIRE_GT(p.max_n_ctx_for_cap, 0u);
    LZ_CHECK_LT(p.max_n_ctx_for_cap, 8192u);
    LZ_CHECK_CONTAINS(p.refusal, std::to_string(p.max_n_ctx_for_cap));

    // The number it named has to actually work at the same cap.
    const Plan retry = plan_at(cap, p.max_n_ctx_for_cap);
    LZ_CHECK(retry.feasible);
    LZ_CHECK(retry.refusal.empty());
    LZ_CHECK_LE(retry.floor.total(), cap);
}

LZ_TEST(a_hopeless_cap_still_refuses_rather_than_throwing) {
    std::string err;
    const Plan p = build_plan(path(), 4096, 32768, &err);
    LZ_CHECK(!p.feasible);
    LZ_CHECK(!p.refusal.empty());
    LZ_CHECK_EQ(p.cache_budget, 0u);
}

LZ_TEST(ram_curve_knee_lands_where_the_unconditional_set_fits) {
    const Fixture& fx = fixture();
    const Plan p = plan_at(kGenerousCap, 512);
    LZ_REQUIRE(p.feasible);

    const std::vector<Plan::CurvePoint> curve = p.ram_curve();
    LZ_REQUIRE_GE(curve.size(), 3u);

    int knees = 0;
    size_t knee_at = 0;
    for (size_t i = 0; i < curve.size(); ++i) {
        if (i > 0) {
            LZ_CHECK_GT(curve[i].cap, curve[i - 1].cap);
            LZ_CHECK_LE(curve[i].bytes_per_token, curve[i - 1].bytes_per_token);
        }
        if (curve[i].is_knee) {
            ++knees;
            knee_at = i;
        }
    }
    LZ_REQUIRE_EQ(knees, 1);

    // The knee is not a heuristic: it is the cap at which tier 1 (the
    // unconditional bulk, value 1.0) is fully resident and the next byte spent
    // starts buying tier 2 (routed experts, value k/n). At that point h_routed
    // is still 0 while h_bytes is already high -- the two hit rates diverge
    // sharply here, which is why they are never one symbol.
    const uint64_t threshold = p.floor.total() + fx.unconditional_bytes;
    LZ_CHECK_GE(curve[knee_at].cap, threshold);
    LZ_REQUIRE_GT(knee_at, 0u);
    LZ_CHECK_LT(curve[knee_at - 1].cap, threshold);

    // Routed experts are still entirely cold at the knee.
    const uint64_t slack = static_cast<uint64_t>(fx.n_moe_layers) * fx.spec.n_expert_used;
    LZ_CHECK_GE(curve[knee_at].bytes_per_token + slack, p.cold_bytes_per_token);
    LZ_CHECK_LE(curve[knee_at].bytes_per_token, p.cold_bytes_per_token + slack);
}

LZ_TEST(a_missing_file_is_reported_not_asserted) {
    std::string err;
    const Plan p = build_plan((fixture().path.parent_path() / "no_such.gguf").string(),
                              kGenerousCap, 512, &err);
    LZ_CHECK(!p.feasible);
    LZ_CHECK(!err.empty());
    LZ_CHECK_EQ(p.tensors.size(), 0u);
}

LZ_TEST(a_null_error_pointer_is_allowed) {
    // Callers that already know they are on a happy path should not have to
    // invent a string to throw away.
    const Plan good = build_plan(path(), kGenerousCap, 512, nullptr);
    LZ_CHECK(good.feasible);
    const Plan bad = build_plan("definitely/not/here.gguf", kGenerousCap, 512, nullptr);
    LZ_CHECK(!bad.feasible);
}

LZ_TEST(report_is_not_empty) {
    const Plan p = plan_at(kGenerousCap, 512);
    LZ_REQUIRE(p.feasible);
    const std::string r = p.report();
    LZ_CHECK(!r.empty());
    // The cap becomes something with a visible shape only if the report shows
    // more than one line of it.
    LZ_CHECK_NE(r.find('\n'), std::string::npos);
}
