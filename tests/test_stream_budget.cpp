// size_stream(): how the streamer divides its share of the cap.
//
// Every figure here was reached by measurement and each has been wrong at some
// cap (the comments in stream_budget.cpp carry the history). Until this was
// extracted, checking any of them meant loading a model. These cases pin the
// CURRENT arithmetic with hand-computed numbers, so a change to the sizing is
// a deliberate edit to this file rather than a surprise in a bytes-read table.

#include "harness.h"

#include <cstdint>
#include <string>

#include "backend/stream_budget.h"

using dray::backend::Config;
using dray::backend::size_stream;
using dray::backend::StreamBudget;
using dray::backend::StreamFlags;
namespace plan = dray::plan;

namespace {

constexpr uint64_t MiB = 1ull << 20;
constexpr uint64_t GiB = 1ull << 30;

void add(plan::Plan& p, const std::string& name, plan::TensorClass cls, uint64_t bytes) {
    plan::TensorInfo t;
    t.name = name;
    t.cls = cls;
    t.bytes = bytes;
    p.tensors.push_back(t);
}

// A small MoE: 64 experts, top-8, 1 MiB per expert slot, a 100 MiB output
// head, a 500 MiB embedding (row-sliced, so it bounds nothing), 1 GiB floor.
plan::Plan small_moe() {
    plan::Plan p;
    p.n_experts = 64;
    p.n_expert_used = 8;
    plan::LayerSlotClass sc;
    sc.layer = 0;
    sc.slot_bytes = 1 * MiB;
    sc.n_experts = 64;
    p.slot_classes.push_back(sc);
    add(p, "token_embd.weight", plan::TensorClass::UnconditionalBulk, 500 * MiB);
    add(p, "blk.0.attn_q.weight", plan::TensorClass::UnconditionalBulk, 10 * MiB);
    add(p, "output.weight", plan::TensorClass::UnconditionalBulk, 100 * MiB);
    add(p, "blk.0.ffn_gate_exps.weight", plan::TensorClass::RoutedExpert, 64 * MiB);
    add(p, "blk.0.ffn_gate_inp.weight", plan::TensorClass::RouterGate, 1 * MiB);
    add(p, "blk.0.attn_norm.weight", plan::TensorClass::NormOrBias, 4096);
    p.floor.router_gates = 1 * GiB;
    return p;
}

Config cfg_at(uint64_t cap, uint32_t n_seq = 1) {
    Config c;
    c.cap = cap;
    c.n_seq = n_seq;
    return c;
}

}  // namespace

LZ_TEST(single_stream_churn_is_three_regions_of_k_doubled_plus_slack) {
    const StreamBudget b = size_stream(small_moe(), cfg_at(4 * GiB), 4096, StreamFlags{});
    // 1 MiB x 8 experts x gate/up/down = 24 MiB; x2 layers in flight + 256 MiB.
    // The widest whole tensor (output, 100 MiB + 25%) is smaller, so slots win.
    LZ_CHECK_EQ(b.churn_reserve, 304 * MiB);
    LZ_CHECK_EQ(b.batch_region_bound, 8 * MiB);   // width 1: exactly k slots
    LZ_CHECK_EQ(b.scratch_bytes, 8 * MiB);        // the 8 MiB floor beats 2 slots
}

LZ_TEST(ring_is_floored_at_512_mib_even_below_the_widest_clamp) {
    // cap/8 = 512 MiB, clamped to 2 x 100 + 256 = 456 MiB by the widest tensor.
    // The whole unconditional stream (110 MiB) fits under the pinnable budget, so
    // the remainder clamp fires with remainder 0 -- and its 512 MiB floor lifts
    // the ring back ABOVE the widest-tensor clamp. Current behaviour, pinned
    // here so it is visible; whether a fully pinnable stream should get any ring
    // at all is a policy question for measurement, not for this refactor.
    const StreamBudget b = size_stream(small_moe(), cfg_at(4 * GiB), 4096, StreamFlags{});
    LZ_CHECK_EQ(b.ring_target, 512 * MiB);
}

LZ_TEST(eslot_pool_is_the_surplus_after_stream_churn_fallback_and_ring) {
    const StreamBudget b = size_stream(small_moe(), cfg_at(4 * GiB), 4096, StreamFlags{});
    // headroom = 4 GiB cap - 1 GiB floor = 3072 MiB.
    // reserved = uncond 110 + churn 304 + routed whole 64 + 25% 16 = 494 MiB.
    LZ_CHECK_EQ(b.eslot_headroom, 3072 * MiB);
    LZ_CHECK_EQ(b.eslot_reserved, 494 * MiB);
    LZ_CHECK_EQ(b.eslot_pool(512 * MiB), (3072 - 494 - 512) * MiB);
    // A refused ring allocation hands its bytes to the pool.
    LZ_CHECK_EQ(b.eslot_pool(0), (3072 - 494) * MiB);
    // Never negative.
    LZ_CHECK_EQ(b.eslot_pool(100 * GiB), 0u);
}

LZ_TEST(batch_width_scales_churn_by_the_union_and_disables_the_pool) {
    const StreamBudget b = size_stream(small_moe(), cfg_at(4 * GiB, 4), 4096, StreamFlags{});
    // ceil(64 x (1 - (1 - 8/64)^4)) = ceil(26.48) = 27 distinct experts.
    LZ_CHECK_EQ(b.batch_region_bound, 27 * MiB);
    LZ_CHECK_EQ(b.churn_reserve, (27 * 3 * 2 + 256) * MiB);
    LZ_CHECK(!b.eslot_enabled);
    LZ_CHECK_EQ(b.eslot_pool(0), 0u);
}

LZ_TEST(no_compact_reserves_a_whole_routed_tensor) {
    Config c = cfg_at(4 * GiB);
    c.no_compact = true;
    const StreamBudget b = size_stream(small_moe(), c, 4096, StreamFlags{});
    // The 64 MiB fused expert tensor replaces the 24 MiB compact trio.
    LZ_CHECK_EQ(b.churn_reserve, (64 * 2 + 256) * MiB);
}

LZ_TEST(a_huge_whole_tensor_sets_churn_and_the_ring_follows_the_remainder) {
    // K3-shaped at a large cap: the unconditional stream (60,512 MiB) exceeds
    // what can be pinned, so the ring is sized to the streamed remainder.
    plan::Plan p = small_moe();
    p.tensors.clear();
    add(p, "output.weight", plan::TensorClass::UnconditionalBulk, 4000 * MiB);
    for (int i = 0; i < 14; ++i) {
        add(p, "blk." + std::to_string(i) + ".attn_q.weight",
            plan::TensorClass::UnconditionalBulk, 4000 * MiB);
    }
    add(p, "blk.14.attn_q.weight", plan::TensorClass::UnconditionalBulk, 512 * MiB);
    const StreamBudget b = size_stream(p, cfg_at(64 * GiB), 4096, StreamFlags{});
    LZ_CHECK_EQ(b.churn_reserve, 5000 * MiB);   // 4000 + 25%
    // pinnable = (65536 - 1024) - 5000 = 59,512 MiB; remainder = 1,000 MiB.
    LZ_CHECK_EQ(b.ring_target, 1000 * MiB);
}

LZ_TEST(ring_override_is_clamped_aligned_and_can_disable) {
    const plan::Plan p = small_moe();
    StreamFlags f;
    f.ring_mb = 0;
    LZ_CHECK_EQ(size_stream(p, cfg_at(4 * GiB), 4096, f).ring_target, 0u);
    f.ring_mb = 32;   // under the 64 MiB minimum: no ring
    LZ_CHECK_EQ(size_stream(p, cfg_at(4 * GiB), 4096, f).ring_target, 0u);
    f.ring_mb = 100;
    LZ_CHECK_EQ(size_stream(p, cfg_at(4 * GiB), 4096, f).ring_target, 100 * MiB);
    f.ring_mb = 2000;   // never more than a quarter of the cap
    LZ_CHECK_EQ(size_stream(p, cfg_at(4 * GiB), 4096, f).ring_target, 1024 * MiB);
    // Rounded DOWN to the discovered device alignment (Invariant 5).
    f.ring_mb = 512;
    const uint32_t odd_align = 3 * 1024 * 1024;
    LZ_CHECK_EQ(size_stream(p, cfg_at(4 * GiB), odd_align, f).ring_target,
                (512 * MiB / odd_align) * odd_align);
}

LZ_TEST(scratch_covers_two_slots_or_the_largest_tensor_on_slow_load) {
    plan::Plan p = small_moe();
    p.slot_classes[0].slot_bytes = 6 * MiB;
    LZ_CHECK_EQ(size_stream(p, cfg_at(4 * GiB), 4096, StreamFlags{}).scratch_bytes, 12 * MiB);
    Config c = cfg_at(4 * GiB);
    c.slow_load = true;   // llama.cpp's loader writes here: fit the largest tensor
    LZ_CHECK_EQ(size_stream(p, c, 4096, StreamFlags{}).scratch_bytes, 500 * MiB);
}
