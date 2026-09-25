#include "backend/stream_budget.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace dray::backend {

namespace {

bool is_token_embd(const plan::TensorInfo& t) {
    return t.name.find("token_embd") != std::string::npos;
}

// The unconditional stream: bulk weights read every token, less the row-sliced
// embedding (only a few rows of it are ever read).
uint64_t uncond_stream_bytes(const plan::Plan& p) {
    uint64_t sum = 0;
    for (const plan::TensorInfo& t : p.tensors) {
        if (t.cls != plan::TensorClass::UnconditionalBulk) continue;
        if (is_token_embd(t)) continue;
        sum += t.bytes;
    }
    return sum;
}

// Sentinel, not a landing zone. This was sized to the largest tensor (1.6 GiB
// on Qwen3.8) only because llama.cpp's file loader wrote bytes straight into
// tensor->data. On the metadata path the loader never reads or writes anything,
// so this only has to be a non-null address satisfying ggml's "tensor not
// allocated" assertions, plus a poison pattern.
//
// That 1.6 GiB is exactly what made a 4 GiB cap infeasible: on top of a ~2.3 GiB
// floor it consumed the entire budget before a single weight could be cached.
//
// Big enough that a failed node reading slot 0 (see the zeroed private ids on
// the failure path) stays inside it: one expert stride across gate/up/down,
// with headroom. The slow-load path additionally needs it to be the LANDING
// ZONE for llama.cpp's own reads, which must fit the largest whole tensor.
uint64_t scratch_size(const plan::Plan& p, const Config& cfg) {
    uint64_t widest = 8ull << 20;
    for (const plan::LayerSlotClass& sc : p.slot_classes) {
        widest = std::max(widest, sc.slot_bytes * 2);
    }
    if (cfg.slow_load) {
        for (const plan::TensorInfo& t : p.tensors) widest = std::max(widest, t.bytes);
    }
    return widest;
}

// Experts one batched node touches: the UNION of the width's selections,
// E*(1-(1-k/E)^n_seq), not n_expert_used (measured 2026-08-19, the
// cap-independent single failure at wide distinct widths: sizing churn for one
// sequence let static pinning eat every budget at every cap and starve the
// 2.5 GB regions batch actually materialises). Reduces EXACTLY to
// n_expert_used at width 1.
uint64_t union_experts(const plan::Plan& p, const Config& cfg) {
    const uint32_t nsq = cfg.n_seq ? cfg.n_seq : 1;
    uint64_t uniq_w = p.n_expert_used;
    if (nsq > 1 && p.n_experts > 0 && p.n_expert_used > 0) {
        const double uE = static_cast<double>(p.n_experts);
        const double uk = static_cast<double>(p.n_expert_used);
        uniq_w = static_cast<uint64_t>(std::ceil(
            uE * (1.0 - std::pow(1.0 - uk / uE, static_cast<double>(nsq)))));
    }
    return uniq_w;
}

// The largest tensor that must be materialised WHOLE. Experts are compacted and
// embedding rows are sliced, so neither needs its full size; everything else
// does, and output.weight is the biggest of them.
uint64_t widest_whole_tensor(const plan::Plan& p) {
    uint64_t widest = 0;
    for (const plan::TensorInfo& t : p.tensors) {
        if (t.cls == plan::TensorClass::RoutedExpert) continue;   // compacted
        if (t.cls == plan::TensorClass::RouterGate) continue;     // pinned floor
        if (t.cls == plan::TensorClass::NormOrBias) continue;     // pinned floor
        if (is_token_embd(t)) continue;                           // row-sliced
        widest = std::max(widest, t.bytes);
    }
    return widest;
}

}  // namespace

uint64_t StreamBudget::eslot_pool(uint64_t ring_allocated) const {
    if (!eslot_enabled) return 0;
    const uint64_t reserved = eslot_reserved + ring_allocated;
    return eslot_headroom > reserved ? eslot_headroom - reserved : 0;
}

StreamBudget size_stream(const plan::Plan& p, const Config& cfg, uint32_t align,
                         const StreamFlags& flags) {
    StreamBudget b;
    b.scratch_bytes = scratch_size(p, cfg);

    // CHURN: room for the experts of the layer being computed, plus a little
    // slack; everything else may be pinned permanently. Sized for what this mode
    // actually materialises. With compaction that is the union of selected
    // experts across gate/up/down; without it, whole routed tensors, which on K3
    // are ~2 GB each. Reserving only the compacted figure while materialising
    // whole tensors starves eviction: static pinning fills the budget, nothing
    // is evictable, and the fallback fails.
    const uint64_t uniq_w = union_experts(p, cfg);
    uint64_t churn = 0;
    for (const plan::LayerSlotClass& sc : p.slot_classes) {
        churn = std::max<uint64_t>(churn, sc.slot_bytes * uniq_w * 3ull);
        // The single widest union region: admission's empirical floor. B=38
        // died with budget at 0.71x this; B=32 ran clean at 2.07x.
        b.batch_region_bound = std::max<uint64_t>(b.batch_region_bound,
                                                  sc.slot_bytes * uniq_w);
    }
    if (cfg.no_compact) {
        for (const plan::TensorInfo& t : p.tensors) {
            if (t.cls == plan::TensorClass::RoutedExpert) churn = std::max(churn, t.bytes);
        }
    }
    churn = churn * 2 + (256ull << 20);   // two layers in flight, plus slack

    // AND at least the largest tensor that must be materialised WHOLE.
    //
    // Sizing the reserve only from expert slots makes this scale wrongly with the
    // cap: static pinning grows to fill whatever budget exists while the reserve
    // stays constant, so the evictable remainder shrinks as the cap RISES. K3 at
    // 32 GiB pinned 29.27 of 30.27 GB, leaving 0.05 GB against a 0.96 GB
    // output.weight -- a configuration that works at 12 GiB and fails at 32, which
    // is the opposite of what anyone would predict.
    const uint64_t widest_whole = widest_whole_tensor(p);
    churn = std::max(churn, widest_whole + (widest_whole / 4));   // + headroom
    b.churn_reserve = churn;

    const uint64_t uncond = uncond_stream_bytes(p);
    const uint64_t budget_est = cfg.cap > p.floor.total() ? cfg.cap - p.floor.total() : 0;

    // PHASE B RING: an accounted, fixed line item -- never carved from the cache
    // at runtime, which is what killed the first prefetch. IoStaging category:
    // cache_budget() derives from cap minus non-cache, so the budget shrinks by
    // exactly this amount automatically.
    //
    // MEASURED, not derived: at an 8 GiB cap on K3, 768 MB/1280 MB/2048 MB rings
    // gave 24.5/24.5/25.0 s/tok and 528/533/539 GB. Bytes fall monotonically as
    // the ring shrinks (returned pinning) while overlap does not degrade -- the
    // widest tensor simply takes the sync path via the span > ring/2 guard, once
    // per token. So the ring should be SMALL: cap/8, floored at 512 MB, never
    // more than the widest whole tensor twice over. The old cap/4 default cost
    // ~1.3 GB/token for nothing.
    uint64_t want = cfg.cap / 8;
    if (want < (512ull << 20)) want = 512ull << 20;
    if (want > widest_whole * 2 + (256ull << 20)) want = widest_whole * 2 + (256ull << 20);

    // INVERSE scaling near the knee. The ring is a flow-through window sized by
    // bandwidth x stall, not by RAM -- but it must also never exceed the stream
    // it serves. As the cap approaches the unconditional set, pinning absorbs
    // the stream and the remainder shrinks; a 2.2 GB window over a ~3 GB/token
    // stream just displaces ~1.7 GB/token of pinning (measured, K3 at 56 GiB).
    // So cap the ring at the estimated streamed remainder.
    {
        const uint64_t pinnable  = budget_est > churn ? budget_est - churn : 0;
        const uint64_t remainder = uncond > pinnable ? uncond - pinnable : 0;
        if (want > remainder) {
            want = remainder < (512ull << 20) ? (512ull << 20) : remainder;
        }
    }

    if (flags.ring_mb) want = *flags.ring_mb << 20;   // override; 0 disables
    if (want > cfg.cap / 4) want = cfg.cap / 4;
    want = (want / align) * align;
    b.ring_target = want >= (64ull << 20) ? want : 0;

    // ESLOT POOL: the past-knee surplus, sized here and never renegotiated.
    //
    // Reserve the whole-expert FALLBACK too: churn_reserve excludes routed
    // tensors from its widest scan, and a compact failure falls back to
    // materialising one whole (2.5 GB on K3) -- slots must never eat that
    // headroom (measured: in_use 54.87/54.89, 0.67 GB evictable, refused).
    uint64_t routed_whole = 0;
    for (const plan::TensorInfo& t : p.tensors) {
        if (t.cls == plan::TensorClass::RoutedExpert && t.bytes > routed_whole)
            routed_whole = t.bytes;
    }
    b.eslot_headroom = budget_est;
    b.eslot_reserved = uncond + churn + routed_whole + routed_whole / 4;
    // Batch: eslot admission is single-stream-only (measured, see the admission
    // site), so a committed pool under batch is budget that can never earn a hit
    // -- and at caps where the unconditional set fully fits, that dead
    // commitment strangled make_room (M3 28G B=16: budget 12.30, in_use 12.21,
    // need 0.70, nothing evictable; NO_ESLOTS cleared it). No width, no pool.
    b.eslot_enabled = !(cfg.n_seq > 1);
    return b;
}

}  // namespace dray::backend
