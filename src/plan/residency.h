// Residency planner: classifies every tensor in a GGUF, computes the mandatory
// floor, and decides what the cache holds.
//
// At small caps this component decides almost all of the achieved performance, so
// it is where the tuning effort belongs.
//
// FILL ORDER (falls out of the arithmetic, not a heuristic):
//   floor  - router gates, norms, biases, KV/recurrent state, prefill activation,
//            scratch, staging. Reserved BEFORE the cache budget exists.
//   tier 1 - unconditional bulk weights (read every token: value 1.0)
//   tier 2 - routed experts (read only when selected: value k/n_experts)
//
// The value ratio is (n/k) weighted by the bandwidth at which the byte would
// otherwise be read; the gather class is slower than the stream class, so the raw
// n/k overstates it. Fill order is unaffected -- the margin is overwhelming either
// way -- but marginal analysis near the knee should use the weighted form.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mem/accountant.h"

namespace dray::plan {

enum class TensorClass : uint8_t {
    RouterGate,      // ffn_gate_inp: MUST be resident. A non-resident gate has to be
                     // read before you know which experts to read -- a serialized
                     // round trip per layer per token that can never be overlapped.
    NormOrBias,      // tiny and scattered; per-request overhead dominates
    RoutedExpert,    // ffn_{gate,up,down}_exps: the streamed population
    UnconditionalBulk,  // attention/mixer projections, shared experts, embd, output
};

struct TensorInfo {
    std::string name;
    TensorClass cls = TensorClass::UnconditionalBulk;
    int32_t     layer = -1;      // -1 for non-layer tensors
    // Offset is absolute WITHIN ITS SHARD, so it is meaningless without `shard`.
    // Every target ships 4-14 shards and a layer's tensors can sit in a different
    // shard from its neighbours; using offset against the wrong file reads plausible
    // garbage rather than failing.
    int32_t     shard = 0;       // index into Plan::shard_paths
    uint64_t    offset = 0;
    uint64_t    bytes = 0;       // exact size from the tensor table (Invariant 3)
    uint32_t    ggml_type = 0;
    // Repacked layout: per-expert DISK stride when the tensor lives expert-
    // interleaved in a companion file (0 = natural layout, stride == nb[2]).
    // Memory layout is unchanged either way; only where bytes come FROM moves.
    uint64_t    disk_stride = 0;
};

// Per-layer expert slot geometry. Sizes are NOT uniform across layers: M3
// UD-Q2_K_XL has three distinct per-layer expert sizes, and a single global slot
// sized to the largest wastes ~26% of the arena on 54 of 57 layers.
struct LayerSlotClass {
    int32_t  layer = -1;
    uint64_t slot_bytes = 0;   // bytes for ONE expert on this layer
    uint32_t n_experts = 0;
};

struct Floor {
    uint64_t router_gates = 0;
    uint64_t norms_biases = 0;
    uint64_t kv_cache = 0;
    uint64_t recurrent_state = 0;
    uint64_t prefill_activation = 0;  // hidden * n_ctx * 2, input and output
    uint64_t compute_scratch = 0;
    uint64_t io_staging = 0;
    uint64_t total() const;
};

struct Plan {
    std::vector<TensorInfo>     tensors;
    std::vector<LayerSlotClass> slot_classes;
    // One entry per shard, in split order. TensorInfo::shard indexes this.
    std::vector<std::string>    shard_paths;
    // Non-fatal disclosures (Invariant 6): printed on success, never mistaken
    // for failure. The error out-param means FAILURE only (swarm S10).
    std::vector<std::string>    warnings;
    // Entries [0, n_meta_shards) are GGUF files carrying tensor metadata; anything
    // past that is a raw data companion (repack) -- readable, never parsed as GGUF.
    // 0 means "all are GGUF" so plans built before this field behave unchanged.
    size_t                      n_meta_shards = 0;

    Floor    floor;
    uint64_t cap = 0;
    uint64_t cache_budget = 0;      // cap - floor.total(); 0 means "refuse"

    // Smallest cap that can actually run at this n_ctx: the floor PLUS enough
    // cache to hold one MoE layer's working set (n_expert_used slots at the widest
    // layer). A cap above the floor but below this is correctly refused -- it
    // could reserve every mandatory byte and still not compute a single layer.
    uint64_t min_cap = 0;

    uint64_t unconditional_bytes = 0;   // read every token when not cached
    uint64_t routed_bytes = 0;          // whole routed population on disk
    uint64_t cold_bytes_per_token = 0;  // k * n_moe_layers * mean slot bytes

    uint32_t n_moe_layers = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_experts = 0;
    uint32_t n_seq = 1;   // sequences this admission funds (batch decode)

    bool     feasible = false;
    uint32_t max_n_ctx_for_cap = 0;  // largest context this cap could serve
    // general.architecture from the GGUF -- the registry lookup key, so the
    // engine can apply profile-expressed load behaviour (e.g. kv_defaults).
    std::string arch = "unknown";
    // KV cache element type the floor was priced at (f16 unless --kv).
    std::string kv_type = "f16";
    std::string refusal;             // non-empty when !feasible: the computed
                                     // minimum plus what context it could serve

    // Projected bytes read per token at this plan's cache budget.
    uint64_t projected_bytes_per_token() const;

    // The RAM curve: what larger caps would buy. Reported at startup so the cap
    // becomes something with a visible shape rather than an opaque number.
    struct CurvePoint { uint64_t cap; uint64_t bytes_per_token; bool is_knee; };
    std::vector<CurvePoint> ram_curve() const;

    std::string report() const;
};

// Reads the GGUF tensor table (never assumes bits-per-weight) and builds a plan.
// Returns a plan with feasible=false and a populated refusal string rather than
// throwing, when the cap cannot cover the floor.
// n_seq: concurrent sequences the admission must fund (batch decode). Each
// sequence carries its own KV and recurrent state INSIDE the cap; 1 = the
// classic single-stream admission. Refusal is per Invariant 1: the cap means
// total bytes, and B sequences that do not fit are refused at plan time,
// never discovered at the first decode.
Plan build_plan(const std::string& gguf_path, uint64_t cap_bytes, uint32_t n_ctx,
                std::string* error, uint32_t n_seq = 1,
                // Bytes per KV element: 2.0 f16 (default), 1.0625 q8_0,
                // 0.5625 q4_0. The floor prices what the run will allocate.
                double kv_bytes_per_el = 2.0);

}  // namespace dray::plan
