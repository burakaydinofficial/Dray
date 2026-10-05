// Residency planner implementation.
//
// Every byte figure here comes from the GGUF tensor table or from GGUF metadata.
// Nothing is derived from an assumed bits-per-weight (Invariant 3) -- that is what
// makes UD-Q2_K_XL, IQ1_M and Q4_K_M all work with no per-quant code.
//
// CONCERNS WITH residency.h AS WRITTEN (implemented as-is, per instruction):
//
//   * Plan carries neither the requested n_ctx nor the model path, so report()
//     cannot restate the context the floor was sized for, nor name the file. Both
//     belong in a startup report. `uint32_t n_ctx` and `std::string path` on Plan
//     would fix it without disturbing anything else. Until then the floor section
//     says "the planned context" and leans on max_n_ctx_for_cap.
//
//   * TensorInfo has no shard id. Every model at this scale ships as
//     <name>-000NN-of-000MM.gguf, and a sharded GGUF numbers tensor offsets from
//     its own file's data section, so `offset` alone cannot address a tensor. The
//     planner reads all shards and records the within-shard absolute offset; the
//     prefetch scheduler will need an io::FileId alongside it.
//
//   * Floor has no field for tokenizer/vocab/sampler, which Invariant 1 places
//     inside the cap (mem::Category::Misc). Those bytes are counted inside
//     compute_scratch and broken out in report() rather than silently dropped --
//     omitting them would make the cap a promise we cannot keep.

#include "plan/residency.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "llama.h"  // llama_split_path / llama_split_prefix: the shard naming convention

namespace dray::plan {
namespace {

// ---------------------------------------------------------------------------
// Estimates
//
// These three are the only numbers in this file that are NOT read from the model.
// They are guesses with a plausible shape, and report() labels them ESTIMATE so a
// user never mistakes them for measurement (Invariant 6). Replace each with the
// figure named in its comment as soon as that measurement exists.
// ---------------------------------------------------------------------------

// Replace with: the arena size ggml_gallocr_get_buffer_size() reports for the
// real graph, taken once at load. A CPU-only MoE forward pass is dominated by the
// largest single intermediate (the expert FFN activation for one wave), which
// scales with n_embd and the wave width -- so a constant is wrong in principle and
// only defensible until the graph exists to measure.
// Scaled, not constant. A flat figure is wrong in both directions: it dwarfs a
// small model (making every modest cap look infeasible) and understates a wide
// one. Shape: a handful of live n_embd-wide intermediates across a batch, which is
// what a CPU MoE forward pass is actually dominated by. Still an ESTIMATE.
constexpr uint64_t kScratchBatchTokens = 512;   // llama.cpp's default n_batch
constexpr uint64_t kScratchLiveTensors = 8;     // live intermediates, order-of
constexpr uint64_t kScratchFloor = 16ull * 1024 * 1024;
constexpr uint64_t kScratchCeil  = 1024ull * 1024 * 1024;

uint64_t compute_scratch_estimate(uint64_t n_embd) {
    const uint64_t v = n_embd * kScratchBatchTokens * sizeof(float) * kScratchLiveTensors;
    return v < kScratchFloor ? kScratchFloor : (v > kScratchCeil ? kScratchCeil : v);
}

// Replace with: queue_depth * max_fragment_bytes from the storage backend, once
// Alignment is discovered at runtime (Invariant 5). Reads land directly in slab
// slots, so staging covers only the unaligned head/tail fragment of each request
// plus the per-token remap tables -- it should end up far smaller than this.
// Scales with what is actually in flight: a few widest-layer expert slots plus the
// per-token remap tables. Capped, because staging buys queue depth and nothing more
// once the queue is full -- a few tens of MB covers a deep queue at 256 KiB requests.
constexpr uint64_t kStagingSlotsInFlight = 4;
constexpr uint64_t kStagingFloor = 2ull * 1024 * 1024;
constexpr uint64_t kStagingCeil  = 64ull * 1024 * 1024;

uint64_t io_staging_estimate(uint64_t widest_slot_bytes) {
    const uint64_t v = widest_slot_bytes * kStagingSlotsInFlight;
    return v < kStagingFloor ? kStagingFloor : (v > kStagingCeil ? kStagingCeil : v);
}

// Replace with: mem::Accountant's Misc category after the vocab is built. Vocab
// strings, merges, the sampler and the cache index. Scales with vocab size, which
// is in the GGUF, but the in-memory expansion factor is not.
// Scales with vocab, which IS in the GGUF; only the in-memory expansion factor is
// a guess. ~64 B per token covers the string, its merge entries and index overhead.
constexpr uint64_t kMiscBytesPerVocabEntry = 64;
constexpr uint64_t kMiscFloor = 4ull * 1024 * 1024;

uint64_t misc_resident_estimate(uint64_t n_vocab) {
    const uint64_t v = n_vocab * kMiscBytesPerVocabEntry;
    return v < kMiscFloor ? kMiscFloor : v;
}

// The KV cache element type. "Assume F16 unless told otherwise" -- there is no
// GGUF key for it; it is a runtime choice llama.cpp makes per context. Taken from
// ggml rather than written as "2" so a future type change stays honest.
constexpr ggml_type kKvType = GGML_TYPE_F16;

// Recurrent state is F32 in llama_memory_recurrent regardless of the weight quant.
constexpr ggml_type kRecurrentStateType = GGML_TYPE_F32;

// The single-stream default. batch and serve fund n_seq sequences explicitly (the
// plan's n_seq); this is the reservation when nothing asks for more.
constexpr uint64_t kResidentSequences = 1;

// Upper bound for the max-n_ctx bisection. Above this the prefill activation term
// alone dwarfs any plausible cap, so searching further only wastes iterations.
constexpr uint64_t kNCtxSearchCeiling = 1ull << 26;  // 67,108,864 tokens

// Number of interior samples in ram_curve(). The knee and the configured cap are
// inserted on top of these.
constexpr int kCurveSamples = 16;

// ---------------------------------------------------------------------------
// Small string helpers
// ---------------------------------------------------------------------------

std::string group_digits(uint64_t v) {
    const std::string s = std::to_string(v);
    std::string out;
    out.reserve(s.size() + s.size() / 3);
    const size_t n = s.size();
    for (size_t i = 0; i < n; ++i) {
        if (i > 0 && (n - i) % 3 == 0) {
            out.push_back(',');
        }
        out.push_back(s[i]);
    }
    return out;
}

std::string human_bytes(uint64_t b) {
    static const char* const kUnit[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    double v = static_cast<double>(b);
    int u = 0;
    while (v >= 1024.0 && u < 5) {
        v /= 1024.0;
        ++u;
    }
    char buf[64];
    if (u == 0) {
        std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(b));
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f %s", v, kUnit[u]);
    }
    return std::string(buf);
}

// Rounded figure plus the exact count. Never the rounded figure alone: at these
// scales "240 GiB" hides a quarter of a gigabyte of disagreement.
std::string bytes_exact(uint64_t b) {
    return human_bytes(b) + "  (" + group_digits(b) + " B)";
}

std::string pct_of(uint64_t part, uint64_t whole) {
    if (whole == 0) {
        return "n/a";
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f%%",
                  100.0 * static_cast<double>(part) / static_cast<double>(whole));
    return std::string(buf);
}

std::string rpad(const std::string& s, size_t w) {
    if (s.size() >= w) {
        return s + " ";
    }
    return s + std::string(w - s.size(), ' ');
}

std::string lpad(const std::string& s, size_t w) {
    if (s.size() >= w) {
        return s;
    }
    return std::string(w - s.size(), ' ') + s;
}

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool contains(const std::string& s, const char* sub) {
    return s.find(sub) != std::string::npos;
}

// ---------------------------------------------------------------------------
// GGUF metadata access
//
// gguf_get_val_* aborts the process on a type mismatch, so every read checks
// gguf_get_kv_type first. Writers are inconsistent about integer width for the
// same key (u32 vs i32 vs u64 all appear in the wild), hence the widening reader.
// ---------------------------------------------------------------------------

bool kv_u64(const gguf_context* ctx, const std::string& key, uint64_t* out) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return false;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  *out = gguf_get_val_u8(ctx, id);  return true;
        case GGUF_TYPE_UINT16: *out = gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_UINT32: *out = gguf_get_val_u32(ctx, id); return true;
        case GGUF_TYPE_UINT64: *out = gguf_get_val_u64(ctx, id); return true;
        case GGUF_TYPE_INT8: {
            const int8_t v = gguf_get_val_i8(ctx, id);
            if (v < 0) return false;
            *out = static_cast<uint64_t>(v);
            return true;
        }
        case GGUF_TYPE_INT16: {
            const int16_t v = gguf_get_val_i16(ctx, id);
            if (v < 0) return false;
            *out = static_cast<uint64_t>(v);
            return true;
        }
        case GGUF_TYPE_INT32: {
            const int32_t v = gguf_get_val_i32(ctx, id);
            if (v < 0) return false;
            *out = static_cast<uint64_t>(v);
            return true;
        }
        case GGUF_TYPE_INT64: {
            const int64_t v = gguf_get_val_i64(ctx, id);
            if (v < 0) return false;
            *out = static_cast<uint64_t>(v);
            return true;
        }
        default:
            return false;
    }
}

bool kv_str(const gguf_context* ctx, const std::string& key, std::string* out) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return false;
    }
    *out = gguf_get_val_str(ctx, id);
    return true;
}

// Reads a numeric array KV. Returns false for non-arrays and for string arrays.
bool kv_u64_array(const gguf_context* ctx, const std::string& key, std::vector<uint64_t>* out) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        return false;
    }
    const gguf_type et = gguf_get_arr_type(ctx, id);
    const size_t n = gguf_get_arr_n(ctx, id);
    // TYPE FIRST. gguf_get_arr_data ASSERTS the array is not a string array
    // (ggml/src/gguf.cpp), and GGML_ASSERT is not NDEBUG-gated, so calling it
    // before this check aborts the process on a GGUF whose head_count_kv is a
    // string array. The docstring above already promised to return false for
    // string arrays; the switch that would have done so is below this call.
    if (et == GGUF_TYPE_STRING) {
        return false;
    }
    const void* raw = gguf_get_arr_data(ctx, id);
    if (raw == nullptr) {
        return false;
    }
    out->clear();
    out->reserve(n);
    for (size_t i = 0; i < n; ++i) {
        switch (et) {
            case GGUF_TYPE_UINT8:
                out->push_back(static_cast<const uint8_t*>(raw)[i]);
                break;
            case GGUF_TYPE_INT8:
                out->push_back(static_cast<uint64_t>(std::max<int8_t>(0, static_cast<const int8_t*>(raw)[i])));
                break;
            case GGUF_TYPE_UINT16:
                out->push_back(static_cast<const uint16_t*>(raw)[i]);
                break;
            case GGUF_TYPE_INT16:
                out->push_back(static_cast<uint64_t>(std::max<int16_t>(0, static_cast<const int16_t*>(raw)[i])));
                break;
            case GGUF_TYPE_UINT32:
                out->push_back(static_cast<const uint32_t*>(raw)[i]);
                break;
            case GGUF_TYPE_INT32:
                out->push_back(static_cast<uint64_t>(std::max<int32_t>(0, static_cast<const int32_t*>(raw)[i])));
                break;
            case GGUF_TYPE_UINT64:
                out->push_back(static_cast<const uint64_t*>(raw)[i]);
                break;
            case GGUF_TYPE_INT64:
                out->push_back(static_cast<uint64_t>(std::max<int64_t>(0, static_cast<const int64_t*>(raw)[i])));
                break;
            default:
                out->clear();
                return false;
        }
    }
    return true;
}

// Several hparams are per-layer arrays in some architectures and scalars in others
// (llama.cpp calls this get_key_or_arr). head_count_kv is the one that matters
// here: a hybrid model with different KV widths per layer gets the KV reservation
// badly wrong if the array form is ignored.
std::vector<uint64_t> per_layer(const gguf_context* ctx, const std::string& key,
                                uint64_t n_layer, uint64_t fallback, bool* exact) {
    *exact = false;
    std::vector<uint64_t> arr;
    if (kv_u64_array(ctx, key, &arr) && !arr.empty()) {
        if (arr.size() == static_cast<size_t>(n_layer)) {
            *exact = true;
            return arr;
        }
        // Length disagreement: broadcast the widest entry rather than guess an
        // alignment. Over-reserving KV is recoverable; under-reserving is not.
        const uint64_t widest = *std::max_element(arr.begin(), arr.end());
        return std::vector<uint64_t>(static_cast<size_t>(n_layer), widest);
    }
    uint64_t scalar = 0;
    if (kv_u64(ctx, key, &scalar)) {
        *exact = true;
        return std::vector<uint64_t>(static_cast<size_t>(n_layer), scalar);
    }
    return std::vector<uint64_t>(static_cast<size_t>(n_layer), fallback);
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

TensorClass classify(const std::string& name) {
    // Order matters. ffn_gate_inp_shexp also matches the first test and is also a
    // router gate (for the shared-expert branch), which is the wanted answer.
    if (contains(name, "ffn_gate_inp")) {
        return TensorClass::RouterGate;
    }
    if (ends_with(name, "_norm.weight") || ends_with(name, "_norm.bias") ||
        contains(name, "exp_probs_b")) {
        return TensorClass::NormOrBias;
    }
    // Every routed-expert spelling the vendored fork can emit, not just the
    // three-tensor form. A fused gate-up tensor (blk.N.ffn_gate_up_exps, real
    // and produced by convert_hf_to_gguf --fuse-gate-up-exps, and by qwen3next
    // and qwen35moe through create_tensor_gate_up_exps) contains NONE of the
    // three substrings below it, so it used to fall through to
    // UnconditionalBulk. That misfiling puts the whole routed population into
    // the unconditional stream: routed_bytes, the knee, cold_bytes_per_token
    // and the entire RAM curve become fiction, and largest_whole_read then
    // prices a multi-GB slab as a whole read and refuses caps that would run.
    // GroveMoE's *_chexps has the same shape (2026-08-24 audit).
    // Deliberately a SHAPE test, not a list of three names. The exclusions
    // matter: ffn_norm_exps is a norm (caught above only if it ends _norm.weight,
    // which it does not), and an expert bias is 2D and must not enter the slot
    // arithmetic, where dividing its bytes by n_experts produces a slot size
    // matching no real expert.
    if (contains(name, "exps") && !contains(name, "_norm") &&
        !ends_with(name, ".bias") && !ends_with(name, "_b")) {
        return TensorClass::RoutedExpert;
    }
    // Attention/mixer projections, shared experts (ffn_*_shexp -- read every token,
    // so they belong in the unconditional stream), token_embd and output.
    return TensorClass::UnconditionalBulk;
}

// "blk.<N>.<rest>" -> N. Anything else -> -1.
int32_t layer_of(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) {
        return -1;
    }
    size_t i = 4;
    uint64_t v = 0;
    bool any = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        v = v * 10 + static_cast<uint64_t>(name[i] - '0');
        ++i;
        any = true;
        // The overflow check after the loop is useless if the accumulator has
        // already wrapped: blk.18446744073709551617 wrapped to 1 and filed the
        // tensor into layer 1's slot arithmetic. GGML_MAX_NAME leaves room for
        // ~58 digits, far past 2^64 (2026-08-24 audit).
        if (v > 0x7fffffffull) return -1;
    }
    if (!any || i >= name.size() || name[i] != '.') {
        return -1;
    }
    return static_cast<int32_t>(v);
}

// ---------------------------------------------------------------------------
// Model geometry needed to size the floor
// ---------------------------------------------------------------------------

struct Geometry {
    std::string arch = "unknown";
    // KV floor scale: bytes-per-element ratio vs the f16 baseline the
    // geometry computes (set by build_plan from the requested cache type).
    double kv_scale = 1.0;

    uint64_t n_layer = 0;
    uint64_t n_embd = 0;
    uint64_t n_head = 0;
    uint64_t n_embd_head_k = 0;
    uint64_t n_embd_head_v = 0;
    std::vector<uint64_t> n_head_kv;  // per layer
    uint64_t n_ctx_train = 0;

    // Precomputed so the bisection does not re-walk the layer list: KV bytes for
    // one token across every layer.
    uint64_t kv_bytes_per_token = 0;

    uint64_t recurrent_bytes = 0;      // independent of n_ctx by construction
    uint64_t n_recurrent_layers = 0;
    uint64_t kv_attention_layers = 0;  // layers that actually carry attention KV
    bool     recurrent_unknown = false;  // recurrent layers present, size not derivable

    uint64_t router_gates = 0;
    uint64_t norms_biases = 0;

    // Inputs to the scaled floor estimates. Flat constants made a tiny model look
    // infeasible at any sane cap and understated a wide one.
    uint64_t n_vocab = 0;
    uint64_t widest_slot_bytes = 0;

    bool kv_geometry_exact = false;
};

Floor floor_at(const Geometry& g, uint64_t n_ctx, uint32_t n_seq) {
    Floor f;
    f.router_gates = g.router_gates;
    f.norms_biases = g.norms_biases;
    // Batch: KV and recurrent state are PER SEQUENCE; the floor funds all of
    // them or the admission lies. Prefill activation stays 1x -- offline batch
    // prefills sequences one at a time and the scratch is reused.
    f.kv_cache = static_cast<uint64_t>(static_cast<double>(g.kv_bytes_per_token) * g.kv_scale) * n_ctx * n_seq;
    f.recurrent_state = g.recurrent_bytes * n_seq;
    // hidden * n_ctx * 2 bytes, doubled for input and output.
    f.prefill_activation = g.n_embd * n_ctx * 2ull * 2ull;
    f.compute_scratch = compute_scratch_estimate(g.n_embd) + misc_resident_estimate(g.n_vocab);
    f.io_staging = io_staging_estimate(g.widest_slot_bytes);
    return f;
}

// ---------------------------------------------------------------------------
// Shard discovery
//
// A model that needs this engine does not fit in one file. llama.cpp's convention
// is "<prefix>-%05d-of-%05d.gguf" with split.no / split.count in every shard's
// metadata; llama_split_prefix/llama_split_path are reused rather than reimplemented
// so a convention change in the submodule cannot silently desync.
// ---------------------------------------------------------------------------

std::vector<std::string> shard_paths(const std::string& path, const gguf_context* ctx) {
    uint64_t n_split = 0;
    uint64_t idx = 0;
    if (!kv_u64(ctx, "split.count", &n_split) || n_split <= 1) {
        return {path};
    }
    // Untrusted: n_split drives a reserve() and was truncated to int32 for the
    // loop. Largest real split in this collection is 14; 4096 is generous and
    // keeps a hostile header from throwing length_error out of build_plan.
    if (n_split > 4096) {
        return {path};
    }
    if (!kv_u64(ctx, "split.no", &idx)) {
        idx = 0;
    }

    std::vector<char> buf(4096, '\0');
    const int32_t plen = llama_split_prefix(buf.data(), buf.size(), path.c_str(),
                                            static_cast<int32_t>(idx),
                                            static_cast<int32_t>(n_split));
    if (plen <= 0) {
        // The file names itself a shard but is not named like one. Trust the file
        // we were handed rather than inventing sibling paths that do not exist.
        return {path};
    }
    const std::string prefix(buf.data(), static_cast<size_t>(plen));

    std::vector<std::string> paths;
    paths.reserve(static_cast<size_t>(n_split));
    for (int32_t i = 0; i < static_cast<int32_t>(n_split); ++i) {
        const int32_t len = llama_split_path(buf.data(), buf.size(), prefix.c_str(), i,
                                             static_cast<int32_t>(n_split));
        if (len <= 0) {
            return {path};
        }
        paths.emplace_back(buf.data(), static_cast<size_t>(len));
    }
    return paths;
}

gguf_context* open_gguf(const std::string& path) {
    // no_alloc with a null ggml context: parse the header, KV block and tensor
    // table, read no tensor data. Invariant 2 permits mmap for exactly this, but
    // gguf_init_from_file uses buffered reads and the metadata is a few tens of MB
    // read once, so it is not worth a private parser.
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx = nullptr;
    return gguf_init_from_file(path.c_str(), gp);
}

// ---------------------------------------------------------------------------
// Traffic model
//
// Fill order is tier 1 (unconditional) then tier 2 (routed); see the header. This
// is the OPTIMISTIC bound: it assumes perfect eviction and uniform routing, so the
// real figure is worse. report() says so, because a number that is only ever an
// upper bound on performance must be labelled as one.
// ---------------------------------------------------------------------------

uint64_t bytes_per_token_at(const Plan& p, uint64_t budget) {
    const uint64_t uncond_cached = std::min(budget, p.unconditional_bytes);
    const uint64_t uncond_miss = p.unconditional_bytes - uncond_cached;

    const uint64_t rest = budget - uncond_cached;
    const uint64_t routed_cached = std::min(rest, p.routed_bytes);

    double routed_miss = 0.0;
    if (p.routed_bytes > 0) {
        // h_routed = cache bytes / routed-expert bytes. Never a whole-file
        // denominator -- that flatters thrashing configs (CLAUDE.md, cost model).
        const double h_routed =
            static_cast<double>(routed_cached) / static_cast<double>(p.routed_bytes);
        routed_miss = static_cast<double>(p.cold_bytes_per_token) * (1.0 - h_routed);
    }
    return uncond_miss + static_cast<uint64_t>(routed_miss + 0.5);
}

// The smallest cache that can execute one MoE layer: k slots at the largest layer's
// slot size, all live at once for ggml_mul_mat_id. Below this the engine cannot
// make forward progress at all, so it is part of feasibility, not of the floor.
uint64_t min_working_set(const Plan& p) {
    // The LARGEST SINGLE TENSOR the run must hold whole. This dominates on dense
    // models and on any model with a fat embedding or output projection, and
    // omitting it made plan optimistic in exactly the way that matters: it
    // reported the 27B feasible at 1 GiB while the engine refused up to 4,
    // quoting "output.weight needs 994 MiB whole". The instant-truth command
    // must not be more hopeful than the loader (measured 2026-08-21..23).
    uint64_t widest_whole = 0;
    for (const TensorInfo& t : p.tensors) {
        if (t.cls == TensorClass::RoutedExpert) continue;   // read per expert slot
        if (t.cls == TensorClass::RowSliced) continue;      // read per row
        widest_whole = std::max(widest_whole, t.bytes);
    }

    uint64_t widest = 0;
    for (const LayerSlotClass& sc : p.slot_classes) {
        widest = std::max(widest, sc.slot_bytes);
    }
    const uint64_t experts = (widest == 0 || p.n_expert_used == 0)
                                 ? 0
                                 : widest * p.n_expert_used;

    // Both must fit, not one or the other: a token materialises its experts AND
    // sweeps the unconditional set. Taking the max rather than the sum keeps the
    // floor honest without pretending the two peak simultaneously.
    const uint64_t need = std::max(experts, widest_whole);
    return need > 0 ? need : 1;
}

}  // namespace

// ---------------------------------------------------------------------------

uint64_t Floor::total() const {
    return router_gates + norms_biases + kv_cache + recurrent_state +
           prefill_activation + compute_scratch + io_staging;
}

uint64_t Plan::projected_bytes_per_token() const {
    return bytes_per_token_at(*this, cache_budget);
}

std::vector<Plan::CurvePoint> Plan::ram_curve() const {
    std::vector<CurvePoint> out;

    const uint64_t base = floor.total();
    const uint64_t span = unconditional_bytes + routed_bytes;
    if (span == 0) {
        out.push_back(CurvePoint{base, 0ull, true});
        return out;
    }

    std::vector<uint64_t> caps;
    caps.reserve(static_cast<size_t>(kCurveSamples) + 4);
    caps.push_back(base);
    for (int i = 1; i <= kCurveSamples; ++i) {
        const double frac = static_cast<double>(i) / static_cast<double>(kCurveSamples);
        caps.push_back(base + static_cast<uint64_t>(static_cast<double>(span) * frac));
    }

    // The knee: the cap at which the entire unconditional set fits. Everything to
    // its left is paying for bytes that are re-read on every single token, so the
    // curve is steep there and nearly flat after it.
    const uint64_t knee = base + unconditional_bytes;
    caps.push_back(knee);

    if (cap > base && cap < base + span) {
        caps.push_back(cap);
    }

    std::sort(caps.begin(), caps.end());
    caps.erase(std::unique(caps.begin(), caps.end()), caps.end());

    out.reserve(caps.size());
    for (uint64_t c : caps) {
        const uint64_t budget = c > base ? c - base : 0;
        out.push_back(CurvePoint{c, bytes_per_token_at(*this, budget), c == knee});
    }
    return out;
}

std::string Plan::report() const {
    std::ostringstream os;

    // -- classification totals, recomputed from the tensor list so the report can
    //    never disagree with the plan it describes.
    uint64_t cls_bytes[5] = {0, 0, 0, 0, 0};
    uint64_t cls_count[5] = {0, 0, 0, 0, 0};
    uint64_t all_bytes = 0;
    for (const TensorInfo& t : tensors) {
        const size_t i = static_cast<size_t>(t.cls);
        cls_bytes[i] += t.bytes;
        cls_count[i] += 1;
        all_bytes += t.bytes;
    }

    const uint64_t floor_total = floor.total();
    const uint64_t knee_cap = floor_total + unconditional_bytes;

    os << "Dray residency plan\n";
    os << "======================\n\n";

    os << "model\n";
    os << "  " << rpad("tensors", 26) << lpad(group_digits(tensors.size()), 20) << "\n";
    os << "  " << rpad("tensor data, all shards", 26) << lpad(bytes_exact(all_bytes), 20)
       << "   summed from the tensor table\n";
    os << "  " << rpad("moe layers", 26) << lpad(group_digits(n_moe_layers), 20) << "\n";
    os << "  " << rpad("experts per layer", 26) << lpad(group_digits(n_experts), 20) << "\n";
    os << "  " << rpad("experts used per token", 26) << lpad(group_digits(n_expert_used), 20)
       << "\n\n";

    os << "classification  (bytes from the GGUF tensor table, never from an assumed bpw)\n";
    static const char* const kClassName[5] = {"router gates", "norms and biases",
                                              "routed experts", "unconditional bulk",
                                              "row-sliced tables"};
    for (size_t i = 0; i < 5; ++i) {
        if (i == 4 && cls_count[i] == 0) continue;   // a model that ties token_embd has none
        os << "  " << rpad(kClassName[i], 26) << lpad(bytes_exact(cls_bytes[i]), 20)
           << "   " << lpad(group_digits(cls_count[i]), 6) << " tensors"
           << "   " << pct_of(cls_bytes[i], all_bytes) << "\n";
    }
    os << "\n";

    // -- slot geometry: collapse consecutive layers that share a slot size. M3
    //    UD-Q2_K_XL has three classes; a single global slot sized to the largest
    //    would waste ~26% of the arena on most layers.
    if (!slot_classes.empty()) {
        os << "expert slot geometry  (per-layer size classes)\n";
        size_t i = 0;
        while (i < slot_classes.size()) {
            size_t j = i;
            while (j + 1 < slot_classes.size() &&
                   slot_classes[j + 1].slot_bytes == slot_classes[i].slot_bytes &&
                   slot_classes[j + 1].layer == slot_classes[j].layer + 1) {
                ++j;
            }
            std::ostringstream range;
            if (i == j) {
                range << "layer " << slot_classes[i].layer;
            } else {
                range << "layers " << slot_classes[i].layer << "-" << slot_classes[j].layer
                      << " (" << (j - i + 1) << ")";
            }
            os << "  " << rpad(range.str(), 26)
               << lpad(group_digits(slot_classes[i].slot_bytes) + " B/slot", 20)
               << "   x " << slot_classes[i].n_experts << " experts = "
               << human_bytes(slot_classes[i].slot_bytes * slot_classes[i].n_experts) << "\n";
            i = j + 1;
        }
        os << "\n";
    }

    os << "mandatory floor  (reserved before the cache budget exists; sized for the planned context)\n";
    // The note must outlive the initializer -- a temporary's c_str() dangles.
    const std::string kv_note = kv_type + (n_seq > 1 ? ", scales with context; x n_seq"
                                                     : ", scales with context");
    struct Row { const char* label; uint64_t bytes; const char* note; };
    const Row rows[] = {
        {"router gates", floor.router_gates, "fully resident at any cap"},
        {"norms and biases", floor.norms_biases, "fully resident at any cap"},
        {"kv cache", floor.kv_cache, kv_note.c_str()},
        {"recurrent state", floor.recurrent_state, "f32, fixed per sequence"},
        {"prefill activation", floor.prefill_activation, "hidden x ctx x 2 B, in + out"},
        {"compute scratch", floor.compute_scratch, "ESTIMATE (incl. vocab/tokenizer/sampler)"},
        {"io staging", floor.io_staging, "ESTIMATE"},
    };
    for (const Row& r : rows) {
        os << "  " << rpad(r.label, 26) << lpad(bytes_exact(r.bytes), 20) << "   " << r.note
           << "\n";
    }
    os << "  " << std::string(46, '-') << "\n";
    os << "  " << rpad("floor total", 26) << lpad(bytes_exact(floor_total), 20) << "\n\n";

    os << "budget\n";
    os << "  " << rpad("cap", 26) << lpad(bytes_exact(cap), 20) << "\n";
    os << "  " << rpad("floor", 26) << lpad(bytes_exact(floor_total), 20) << "   "
       << pct_of(floor_total, cap) << " of cap\n";
    os << "  " << rpad("expert cache budget", 26) << lpad(bytes_exact(cache_budget), 20)
       << "\n";
    os << "  " << rpad("largest context this cap", 26)
       << lpad(group_digits(max_n_ctx_for_cap) + " tokens", 20) << "\n";

    if (!feasible) {
        os << "\nREFUSED\n  ";
        os << (refusal.empty()
                   ? std::string("the plan is incomplete; see the error string returned by "
                                 "build_plan()")
                   : refusal);
        os << "\n\n";
    } else {
        const uint64_t uncond_cached = std::min(cache_budget, unconditional_bytes);
        const uint64_t routed_cached = std::min(cache_budget - uncond_cached, routed_bytes);
        os << "  " << rpad("unconditional resident", 26)
           << lpad(bytes_exact(uncond_cached), 20) << "   "
           << pct_of(uncond_cached, unconditional_bytes) << " of tier 1\n";
        os << "  " << rpad("h_routed", 26) << lpad(pct_of(routed_cached, routed_bytes), 20)
           << "   cache bytes / routed bytes\n";
        if (cache_budget < unconditional_bytes) {
            os << "  " << rpad("to reach the knee", 26)
               << lpad(bytes_exact(knee_cap - cap), 20) << "   more RAM\n";
        }
        os << "\n";

        const uint64_t bpt = projected_bytes_per_token();
        const uint64_t uncond_miss = unconditional_bytes - uncond_cached;
        os << "projected traffic per token  (OPTIMISTIC: assumes perfect eviction and uniform\n";
        os << "                              routing, so the measured figure will be worse)\n";
        os << "  " << rpad("unconditional re-read", 26) << lpad(bytes_exact(uncond_miss), 20)
           << "\n";
        os << "  " << rpad("routed expert misses", 26)
           << lpad(bytes_exact(bpt > uncond_miss ? bpt - uncond_miss : 0ull), 20) << "\n";
        os << "  " << rpad("cold cost if h_routed=0", 26)
           << lpad(bytes_exact(cold_bytes_per_token), 20) << "\n";
        os << "  " << std::string(46, '-') << "\n";
        os << "  " << rpad("total per token", 26) << lpad(bytes_exact(bpt), 20) << "\n";
        os << "  seconds per token = " << human_bytes(bpt)
           << " / realized_read_bandwidth, measured on this drive at this block size\n";
        os << "  and queue depth -- never a vendor sequential rating.\n\n";
    }

    // Printed even on a refusal: "how much RAM would this actually need" is the
    // first thing a refused user wants, and the curve answers it.
    os << "what more RAM would buy\n";
    os << "  " << rpad("cap", 20) << rpad("bytes/token", 20) << "\n";
    for (const CurvePoint& p : ram_curve()) {
        os << "  " << rpad(human_bytes(p.cap), 20) << rpad(human_bytes(p.bytes_per_token), 20);
        if (p.is_knee) {
            os << "  <- knee: the whole unconditional stream is resident";
        }
        if (p.cap == cap) {
            os << "  <- configured cap";
        }
        os << "\n";
    }

    return os.str();
}

// ---------------------------------------------------------------------------

Plan build_plan(const std::string& gguf_path, uint64_t cap_bytes, uint32_t n_ctx,
                std::string* error, uint32_t n_seq, double kv_bytes_per_el) {
    if (n_seq == 0) n_seq = 1;
    Plan plan;
    plan.cap = cap_bytes;

    const auto fail = [&](const std::string& msg) -> Plan& {
        if (error != nullptr) {
            *error = msg;
        }
        plan.feasible = false;
        return plan;
    };

    gguf_context* head = open_gguf(gguf_path);
    if (head == nullptr) {
        return fail("cannot open GGUF: " + gguf_path);
    }

    const std::vector<std::string> paths = shard_paths(gguf_path, head);
    plan.shard_paths = paths;
    gguf_free(head);
    head = nullptr;

    // -- read every shard's tensor table. The first shard that carries a given key
    //    wins; in practice shard 0 carries all metadata and the rest carry only
    //    tensors, but nothing in the format guarantees that.
    std::vector<gguf_context*> ctxs;
    std::vector<uint64_t> shard_file_bytes;  // 0 when the OS would not tell us
    ctxs.reserve(paths.size());
    shard_file_bytes.reserve(paths.size());
    for (const std::string& p : paths) {
        gguf_context* c = open_gguf(p);
        if (c == nullptr) {
            for (gguf_context* d : ctxs) {
                gguf_free(d);
            }
            return fail("cannot open GGUF shard: " + p);
        }
        ctxs.push_back(c);

        std::error_code ec;
        const std::uintmax_t sz = std::filesystem::file_size(std::filesystem::path(p), ec);
        shard_file_bytes.push_back(ec ? 0ull : static_cast<uint64_t>(sz));
    }
    // A short shard is the failure mode of a 250 GB download, and it surfaces later
    // as plausible text from garbage weights rather than as an error. Cheap to catch
    // here: the tensor table says how many bytes the file must have.
    std::string truncation_warning;

    const gguf_context* meta = ctxs.front();

    Geometry g;
    if (!kv_str(meta, "general.architecture", &g.arch)) {
        g.arch = "unknown";
    }
    const std::string a = g.arch + ".";

    // -- tensor table -------------------------------------------------------
    uint64_t max_layer_seen = 0;
    bool any_layer = false;
    std::unordered_map<std::string, int32_t> seen_names;   // name -> first shard
    for (size_t s = 0; s < ctxs.size(); ++s) {
        const gguf_context* c = ctxs[s];
        const uint64_t data_off = static_cast<uint64_t>(gguf_get_data_offset(c));
        const int64_t n_tensors = gguf_get_n_tensors(c);
        uint64_t shard_end = data_off;
        for (int64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = gguf_get_tensor_name(c, i);
            // Upstream rejects duplicate names WITHIN a shard; nothing checked
            // ACROSS them. A duplicate used to be double-counted into every byte
            // total here, and then the merged metadata load GGML_ABORTed with no
            // path in the message. Refuse at plan time with both shards named
            // (2026-08-24 audit).
            {
                auto ins = seen_names.emplace(t.name, static_cast<int32_t>(s));
                if (!ins.second) {
                    const std::string msg =
                        "duplicate tensor \"" + t.name + "\" appears in shard " +
                        std::to_string(ins.first->second) + " and shard " +
                        std::to_string(s) +
                        ": the shard set is inconsistent (mixed conversions?)";
                    for (gguf_context* cc : ctxs) gguf_free(cc);
                    return fail(msg);
                }
            }
            t.cls = classify(t.name);
            t.row_gathered = contains(t.name, "token_embd");
            t.layer = layer_of(t.name);
            // Absolute WITHIN THIS SHARD, so it travels with the shard index.
            // Using it against another shard's handle reads plausible garbage
            // rather than failing, which is the worst failure mode available here.
            t.shard = static_cast<int32_t>(s);
            t.offset = data_off + static_cast<uint64_t>(gguf_get_tensor_offset(c, i));
            t.bytes = static_cast<uint64_t>(gguf_get_tensor_size(c, i));
            t.ggml_type = static_cast<uint32_t>(gguf_get_tensor_type(c, i));

            shard_end = std::max(shard_end, t.offset + t.bytes);
            if (t.layer >= 0) {
                max_layer_seen = std::max(max_layer_seen, static_cast<uint64_t>(t.layer));
                any_layer = true;
            }
            plan.tensors.push_back(std::move(t));
        }
        if (shard_file_bytes[s] != 0 && shard_file_bytes[s] < shard_end) {
            // REFUSE, do not warn. The old warning's own text admitted the
            // consequence -- "reads past the end will return garbage, not an
            // error" -- and then let the run proceed to serve exactly that.
            // A floor-integrity mismatch is fatal; a file that cannot contain
            // its own tensor table is the same defect one step earlier.
            // DRAY_ALLOW_TRUNCATED=1 keeps the old behaviour for someone
            // who genuinely wants to poke a partial download (2026-08-24).
            const bool allow = [] {
                const char* v = std::getenv("DRAY_ALLOW_TRUNCATED");
                return v && v[0] == '1';
            }();
            const std::string what =
                paths[s] + " is short: the tensor table needs " +
                group_digits(shard_end) + " B but the file is " +
                group_digits(shard_file_bytes[s]) + " B";
            if (!allow) {
                for (gguf_context* ctx : ctxs) gguf_free(ctx);
                return fail("REFUSED: " + what +
                            ". A short shard serves plausible text from garbage weights; "
                            "re-download it, or set DRAY_ALLOW_TRUNCATED=1 to proceed anyway.");
            }
            truncation_warning += "warning: " + what +
                                  ". Reads past the end will return garbage, not an error.\n";
        }
    }

    if (plan.tensors.empty()) {
        for (gguf_context* c : ctxs) {
            gguf_free(c);
        }
        return fail("GGUF contains no tensors: " + gguf_path);
    }

    // -- hparams ------------------------------------------------------------
    if (!kv_u64(meta, a + "block_count", &g.n_layer) || g.n_layer == 0) {
        g.n_layer = any_layer ? max_layer_seen + 1 : 0;
    }
    // GGUF metadata is untrusted input from the internet, and both of these
    // numbers size std::vectors directly. A block_count of 2^50 attempted an
    // 8 PB allocation (uncaught bad_alloc, process terminates); a single tensor
    // named blk.2147483000.x sized a 17 GB vector -- outside the accountant, an
    // Invariant 1 hole as well as a crash. No real architecture is within two
    // orders of magnitude of this bound (2026-08-24 audit).
    constexpr uint64_t kMaxLayers = 65536;
    if (g.n_layer > kMaxLayers || (any_layer && max_layer_seen >= kMaxLayers)) {
        for (gguf_context* c : ctxs) gguf_free(c);
        return fail("implausible layer count (block_count " + std::to_string(g.n_layer) +
                    ", highest blk." + std::to_string(max_layer_seen) +
                    "): refusing rather than sizing tables from a hostile header");
    }
    // Per-layer tables are sized from whichever is larger, the declared block_count
    // or the highest blk.<N> actually present. A GGUF where those disagree is odd,
    // but dropping the extra layers would understate routed bytes -- an optimistic
    // error, which is the one kind this project does not tolerate.
    const size_t n_layer_slots =
        static_cast<size_t>(std::max<uint64_t>(g.n_layer, any_layer ? max_layer_seen + 1 : 0)) + 1;
    if (!kv_u64(meta, a + "embedding_length", &g.n_embd)) {
        g.n_embd = 0;
    }
    if (!kv_u64(meta, a + "attention.head_count", &g.n_head)) {
        g.n_head = 0;
    }
    if (!kv_u64(meta, a + "context_length", &g.n_ctx_train)) {
        g.n_ctx_train = 0;
    }

    if (!kv_u64(meta, a + "attention.key_length", &g.n_embd_head_k)) {
        g.n_embd_head_k = (g.n_head > 0) ? g.n_embd / g.n_head : 0;
    }
    if (!kv_u64(meta, a + "attention.value_length", &g.n_embd_head_v)) {
        g.n_embd_head_v = g.n_embd_head_k;
    }

    bool head_kv_exact = false;
    g.n_head_kv = per_layer(meta, a + "attention.head_count_kv", g.n_layer, g.n_head,
                            &head_kv_exact);
    g.kv_geometry_exact = head_kv_exact && g.n_embd_head_k > 0;

    // Vocab size drives the Misc estimate. Prefer the token-list array length,
    // which is exact; fall back to the scalar key some converters write.
    {
        const int64_t kid = gguf_find_key(meta, "tokenizer.ggml.tokens");
        if (kid >= 0 && gguf_get_kv_type(meta, kid) == GGUF_TYPE_ARRAY) {
            g.n_vocab = static_cast<uint64_t>(gguf_get_arr_n(meta, kid));
        }
        if (g.n_vocab == 0) {
            uint64_t v = 0;
            if (kv_u64(meta, a + "vocab_size", &v)) g.n_vocab = v;
        }
    }

    {
        // n_head_kv * (n_embd_head_k + n_embd_head_v) * f16, summed over layers.
        //
        // Two known ways this is not the whole story, both left as-is because
        // sizing them needs architecture knowledge that belongs in the registry,
        // not here:
        //   - MiniMax M3 attaches an indexer key cache to its sparse (MSA) layers
        //     on top of ordinary KV, so this UNDER-counts. Verify against
        //     llama_kv_cache_msa before trusting the M3 number.
        //   - MLA architectures store a compressed latent instead of full K and V,
        //     so this OVER-counts, which is the safe direction.
        uint64_t per_token = 0;
        for (uint64_t h : g.n_head_kv) {
            per_token += h * (g.n_embd_head_k + g.n_embd_head_v);
        }
        g.kv_bytes_per_token = per_token * static_cast<uint64_t>(ggml_type_size(kKvType));
    }

    // -- recurrent state ----------------------------------------------------
    // Counted from the tensor table, not from an architecture branch (Invariant 4):
    // every recurrent mixer in llama.cpp names its weights blk.<N>.ssm_*, including
    // KDA and Gated DeltaNet.
    {
        std::vector<bool> recurrent(n_layer_slots, false);
        for (const TensorInfo& t : plan.tensors) {
            if (t.layer >= 0 && contains(t.name, ".ssm_")) {
                const size_t li = static_cast<size_t>(t.layer);
                if (li < recurrent.size()) {
                    recurrent[li] = true;
                }
            }
        }
        for (bool b : recurrent) {
            if (b) {
                ++g.n_recurrent_layers;
            }
        }

        // A recurrent layer has NO attention KV, and on a hybrid model most layers
        // are recurrent. head_count_kv is usually a scalar in the metadata, so the
        // per_layer() broadcast hands every layer the same value and the KV sum
        // above counts all of them.
        //
        // Measured on Qwen3.8 UD-IQ1_S (23 attention + 69 Gated DeltaNet of 93):
        // the uncorrected sum gave 380,928 B/token against a true 94,208 -- exactly
        // 93/23 too high -- which inflated the 32K floor from ~6 GiB to 14.84 GiB
        // and made caps look infeasible that are not. K3 is 24 MLA + 69 KDA and is
        // wrong by the same mechanism.
        if (g.n_recurrent_layers > 0 && !g.n_head_kv.empty()) {
            uint64_t per_token = 0;
            for (size_t li = 0; li < g.n_head_kv.size(); ++li) {
                if (li < recurrent.size() && recurrent[li]) continue;  // no KV here
                per_token += g.n_head_kv[li] * (g.n_embd_head_k + g.n_embd_head_v);
            }
            g.kv_bytes_per_token =
                per_token * static_cast<uint64_t>(ggml_type_size(kKvType));
            g.kv_attention_layers = g.n_head_kv.size() - g.n_recurrent_layers;
        }
    }

    if (g.n_recurrent_layers > 0) {
        // Mirrors llama_hparams::n_embd_r() / n_embd_s(). Element counts, not bytes.
        uint64_t d_conv = 0, d_inner = 0, d_state = 0, n_group = 0, kda_head_dim = 0;
        kv_u64(meta, a + "ssm.conv_kernel", &d_conv);
        kv_u64(meta, a + "ssm.inner_size", &d_inner);
        kv_u64(meta, a + "ssm.state_size", &d_state);
        kv_u64(meta, a + "ssm.group_count", &n_group);
        kv_u64(meta, a + "kda.head_dim", &kda_head_dim);

        uint64_t n_r = 0, n_s = 0;
        if (kda_head_dim > 0 && g.n_head > 0) {
            // KDA: conv state for q, k and v, plus a head_dim x head_dim state per head.
            const uint64_t kda_inner = g.n_head * kda_head_dim;
            n_r = 3ull * (d_conv > 0 ? d_conv - 1 : 3) * kda_inner;
            n_s = kda_head_dim * kda_head_dim * g.n_head;
        } else if (d_state > 0 && d_inner > 0) {
            n_r = (d_conv > 0 ? d_conv - 1 : 0) * (d_inner + 2 * n_group * d_state);
            n_s = d_state * d_inner;
        } else {
            // Recurrent layers exist but nothing in the metadata sizes them. Report
            // unknown rather than default (Invariant 6): a zero here would silently
            // hand the cache RAM that the mixer is going to take anyway.
            g.recurrent_unknown = true;
        }
        g.recurrent_bytes = (n_r + n_s) * static_cast<uint64_t>(ggml_type_size(kRecurrentStateType)) *
                            g.n_recurrent_layers * kResidentSequences;
    }

    // -- expert counts ------------------------------------------------------
    uint64_t n_experts = 0, n_expert_used = 0;
    kv_u64(meta, a + "expert_count", &n_experts);
    kv_u64(meta, a + "expert_used_count", &n_expert_used);

    // -- class totals and per-layer routed sums -----------------------------
    // Embedding tables are only gathered by rows -- unless the model TIES its
    // output projection to token_embd (no output.weight), in which case that one
    // table is also a whole matmul every token and stays unconditional bulk.
    // Pricing a 27 GB engram table as a whole read refused Qwen3.8-Flash-Next
    // below 27 GiB and projected 33 GB/token it never reads.
    bool has_output = false;
    for (const TensorInfo& t : plan.tensors) has_output = has_output || t.name == "output.weight";
    for (TensorInfo& t : plan.tensors) {
        if (t.row_gathered && t.cls == TensorClass::UnconditionalBulk &&
            (has_output || t.name != "token_embd.weight")) {
            t.cls = TensorClass::RowSliced;
        }
    }

    std::vector<uint64_t> routed_by_layer(n_layer_slots, 0);
    bool any_routed = false;
    for (const TensorInfo& t : plan.tensors) {
        switch (t.cls) {
            case TensorClass::RouterGate:
                g.router_gates += t.bytes;
                break;
            case TensorClass::NormOrBias:
                g.norms_biases += t.bytes;
                break;
            case TensorClass::RoutedExpert:
                plan.routed_bytes += t.bytes;
                any_routed = true;
                if (t.layer >= 0 && static_cast<size_t>(t.layer) < routed_by_layer.size()) {
                    routed_by_layer[static_cast<size_t>(t.layer)] += t.bytes;
                }
                break;
            case TensorClass::UnconditionalBulk:
                plan.unconditional_bytes += t.bytes;
                break;
            case TensorClass::RowSliced:
                plan.row_sliced_bytes += t.bytes;
                break;
        }
    }

    if (any_routed && n_experts == 0) {
        // Fall back to the tensor table before giving up: the router gate is
        // [n_embd, n_expert], so ne[1] is the expert count. Cheaper than refusing,
        // and still Invariant-3 clean.
        for (size_t s = 0; s < ctxs.size() && n_experts == 0; ++s) {
            const gguf_context* c = ctxs[s];
            const int64_t n = gguf_get_n_tensors(c);
            for (int64_t i = 0; i < n; ++i) {
                const std::string nm = gguf_get_tensor_name(c, i);
                if (!contains(nm, "ffn_gate_inp") || contains(nm, "shexp")) {
                    continue;
                }
                const int64_t* ne = gguf_get_tensor_ne(c, i);
                if (ne != nullptr && ne[1] > 0) {
                    n_experts = static_cast<uint64_t>(ne[1]);
                    break;
                }
            }
        }
    }

    for (gguf_context* c : ctxs) {
        gguf_free(c);
    }
    ctxs.clear();

    if (any_routed && n_experts == 0) {
        return fail("GGUF has routed experts but no " + a +
                    "expert_count, and the router gate shape did not yield one; "
                    "cannot size expert slots");
    }
    if (any_routed && n_expert_used == 0) {
        return fail("GGUF has routed experts but no " + a +
                    "expert_used_count; cannot size the per-token working set");
    }

    plan.n_experts = static_cast<uint32_t>(n_experts);
    plan.n_expert_used = static_cast<uint32_t>(n_expert_used);

    // -- per-layer slot classes ---------------------------------------------
    // Sizes are NOT uniform across layers. M3 UD-Q2_K_XL has three distinct
    // per-layer expert sizes (18,137,088 / 20,938,752 / 24,477,696 bytes); one
    // global slot sized to the largest wastes ~26% of the arena on 54 of 57
    // layers. So this is computed per layer, never once globally.
    //
    // Sanity target: on M3 UD-Q2_K_XL this must produce exactly three distinct
    // values -- 18,137,088 / 20,938,752 / 24,477,696 -- with nothing rounded. Any
    // other outcome means the grouping is wrong, not that the model is unusual.
    uint64_t routed_slot_sum = 0;
    bool slot_rounded = false;
    for (size_t li = 0; li < routed_by_layer.size(); ++li) {
        if (routed_by_layer[li] == 0) {
            continue;
        }
        LayerSlotClass sc;
        sc.layer = static_cast<int32_t>(li);
        sc.n_experts = plan.n_experts;
        // Round up: a slot must hold the whole expert. A GGUF whose routed tensors
        // do not divide evenly by n_experts is malformed, but truncating here would
        // hand ggml_mul_mat_id a short slot, and a short slot yields plausible text
        // from garbage weights rather than an error.
        slot_rounded = slot_rounded || (routed_by_layer[li] % n_experts != 0);
        sc.slot_bytes = (routed_by_layer[li] + n_experts - 1) / n_experts;
        routed_slot_sum += sc.slot_bytes;
        if (sc.slot_bytes > g.widest_slot_bytes) g.widest_slot_bytes = sc.slot_bytes;
        plan.slot_classes.push_back(sc);
    }
    plan.n_moe_layers = static_cast<uint32_t>(plan.slot_classes.size());

    // k * n_moe_layers * mean slot bytes, written as the exact sum so a model with
    // three size classes is not averaged into a figure that matches no layer.
    plan.cold_bytes_per_token = static_cast<uint64_t>(plan.n_expert_used) * routed_slot_sum;

    // -- floor and feasibility ----------------------------------------------
    g.kv_scale = kv_bytes_per_el / 2.0;
    plan.kv_type = kv_bytes_per_el < 0.7 ? "q4_0" : (kv_bytes_per_el < 1.5 ? "q8_0" : "f16");
    plan.arch = g.arch;
    plan.floor = floor_at(g, n_ctx, n_seq);
    plan.n_seq = n_seq;
    const uint64_t floor_total = plan.floor.total();
    plan.cache_budget = cap_bytes > floor_total ? cap_bytes - floor_total : 0;

    const uint64_t working_set = min_working_set(plan);
    const uint64_t minimum_cap = floor_total + working_set;

    // Largest context this cap could serve. floor_at() is monotone increasing in
    // n_ctx (kv_cache and prefill_activation are both linear in it, everything else
    // is constant), so bisection is exact rather than a search over a bumpy space.
    {
        uint64_t hi = std::max<uint64_t>(n_ctx, std::max<uint64_t>(g.n_ctx_train, 4096));
        hi = std::min(hi, kNCtxSearchCeiling);

        const auto fits = [&](uint64_t ctx_len) {
            const uint64_t ft = floor_at(g, ctx_len, n_seq).total();
            return cap_bytes > ft && (cap_bytes - ft) >= working_set;
        };

        uint64_t lo = 0;
        if (fits(hi)) {
            // The cap covers the whole search range, so the honest answer is "at
            // least this much" -- saturated at the training context (or the
            // requested context, or the ceiling), whichever bounded the search.
            lo = hi;
        } else {
            while (lo < hi) {
                const uint64_t mid = lo + (hi - lo + 1) / 2;
                if (fits(mid)) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
        }
        plan.max_n_ctx_for_cap = static_cast<uint32_t>(std::min<uint64_t>(lo, 0xffffffffull));
    }

    plan.min_cap  = minimum_cap;
    plan.feasible = cap_bytes >= minimum_cap && plan.cache_budget > 0;

    if (!plan.feasible) {
        // The header documents cache_budget == 0 as "refuse". Keep the two signals
        // in agreement: a caller that only checks the budget must not see a
        // non-zero one on a plan that was refused for want of a working set.
        plan.cache_budget = 0;

        std::ostringstream r;
        r << "cap of " << bytes_exact(cap_bytes) << " cannot serve a context of "
          << group_digits(n_ctx) << " tokens.\n";
        r << "  mandatory floor at that context: " << bytes_exact(floor_total) << "\n";
        // Name what the number IS. working_set is max(expert set, widest whole
        // tensor), so on a dense model this printed "0 experts at the widest
        // layer's slot size: 994 MiB" -- a refusal that misdescribed the very
        // thing it was refusing over (2026-08-24).
        if (plan.n_expert_used > 0) {
            r << "  smallest working set (the larger of " << plan.n_expert_used
              << " experts at the widest layer's slot size, and the largest single"
                 " tensor held whole): "
              << bytes_exact(working_set) << "\n";
        } else {
            r << "  largest single tensor that must be held whole: "
              << bytes_exact(working_set) << "\n";
        }
        r << "  minimum cap: " << bytes_exact(minimum_cap) << "  -- short by "
          << bytes_exact(minimum_cap > cap_bytes ? minimum_cap - cap_bytes : 0ull) << "\n";
        if (plan.max_n_ctx_for_cap > 0) {
            // Bare digits, not grouped: a context length is conventionally written
            // unseparated (32768, not 32,768), and it keeps the number greppable
            // and machine-parseable out of the refusal text.
            r << "  this cap could serve a context of up to "
              << plan.max_n_ctx_for_cap << " tokens";
        } else {
            r << "  this cap cannot serve any context: the context-independent part of "
                 "the floor (gates, norms, recurrent state, scratch, staging) is "
              << bytes_exact(plan.floor.router_gates + plan.floor.norms_biases +
                             plan.floor.recurrent_state + plan.floor.compute_scratch +
                             plan.floor.io_staging)
              << " before a single token of context";
        }
        plan.refusal = r.str();
    }

    if (error != nullptr) error->clear();
    // Non-fatal disclosures (Invariant 6) travel in plan.warnings, NEVER the
    // error out-param: six callers treat a non-empty error as fatal, which
    // turned a degraded-but-usable plan into "unservable" (swarm S10). The
    // likely live triggers are recurrent metadata gaps (Qwen3.8, K3) and
    // truncated shards.
    if (!truncation_warning.empty()) plan.warnings.push_back(truncation_warning);
    if (!g.kv_geometry_exact) {
        plan.warnings.push_back(
            "warning: KV geometry incomplete in metadata (" + a +
            "attention.head_count_kv / key_length); the KV reservation is "
            "an inference from head_count and embedding_length.");
    }
    if (g.recurrent_unknown) {
        plan.warnings.push_back(
            "warning: recurrent layers present but their state size is not "
            "derivable from metadata; recurrent_state is reported as 0 and is "
            "UNKNOWN, not zero.");
    }
    if (slot_rounded) {
        plan.warnings.push_back(
            "warning: a layer's routed-expert bytes do not divide evenly by "
            "expert_count; slot_bytes was rounded up. Either the tensor "
            "grouping here is wrong or the file is unusual -- check before "
            "trusting the arena geometry.");
    }

    return plan;
}

}  // namespace dray::plan
