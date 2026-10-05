// How the streamer divides its share of the cap, derived once from the Plan.
//
// Pure arithmetic over the GGUF tensor table and the configuration: no I/O, no
// allocation, no ledger. That is the point of keeping it separate -- every
// figure here was reached by measurement and each has been wrong at least once
// at some cap, so it must be checkable without loading a model
// (tests/test_stream_budget.cpp does exactly that).
//
// Figures that depend on the LIVE ledger (the cache budget, the static-pinning
// allowance) are not here: they are derived from the Accountant at the moment
// of use, because a stored copy drifts (see cache_budget in stream_buffer.cpp).

#pragma once

#include <cstdint>

#include "backend/stream_buffer.h"
#include "backend/stream_flags.h"
#include "plan/residency.h"

namespace dray::backend {

struct StreamBudget {
    // Poison sentinel: the non-null address streamed tensors point at until
    // materialised, and what a failed node computes against. On the slow-load
    // path it is also llama.cpp's landing zone, so it fits the largest tensor.
    uint64_t scratch_bytes = 0;

    // Room the node being computed is guaranteed; static pinning never takes it.
    uint64_t churn_reserve = 0;

    // The widest single batched union region (uniq(n_seq) x slot): admission's
    // measured floor. Certain death below 1x (B=38), proven clean at 2x (B=32).
    uint64_t batch_region_bound = 0;

    // The expert union the churn reserve funds per layer (expected union at the
    // funded width; k for one stream). Steps whose union fits it may take the
    // decode fast paths (ExpertCompactor::fast_path).
    uint64_t funded_union = 0;

    // Phase B ring arena to allocate, already aligned. 0 = no ring.
    uint64_t ring_target = 0;

    // The frequency expert cache's pool: the past-knee surplus. Sized against
    // the ring ACTUALLY allocated, which may be less than ring_target (0 when
    // the allocation was refused).
    uint64_t eslot_pool(uint64_t ring_allocated) const;

    // Inputs of eslot_pool(), kept so the pool can be derived after the ring
    // allocation succeeds or fails.
    uint64_t eslot_headroom = 0;   // cap - floor, the budget estimate
    uint64_t eslot_reserved = 0;   // what the pool must leave, ring excluded
    bool     eslot_enabled  = true;
};

// `align` is the discovered device alignment (Invariant 5), the max across shards.
StreamBudget size_stream(const plan::Plan&, const Config&, uint32_t align,
                         const StreamFlags&);

}  // namespace dray::backend
