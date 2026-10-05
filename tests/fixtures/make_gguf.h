// A tiny but STRUCTURALLY REAL GGUF, generated at test time.
//
// WHY this file is the backbone of the suite: the smallest real target is 143 GB
// and a real run is hours (CLAUDE.md), so nothing in CI can open a real model.
// Every claim the residency planner makes -- classification, floor arithmetic,
// slot geometry, the cache budget -- is a claim about a tensor table, and a
// tensor table is cheap to synthesise exactly.
//
// WHY the per-layer expert sizes are deliberately NON-UNIFORM: MiniMax M3
// UD-Q2_K_XL has three distinct per-layer expert sizes (18,137,088 / 20,938,752
// / 24,477,696 bytes) and a single global slot sized to the largest wastes ~26%
// of the arena on 54 of 57 layers. A fixture with uniform layers lets a planner
// that assumes uniformity pass by accident. The class assignment here is also
// deliberately NOT round-robin, for the same reason: M3's distribution is
// skewed, three layers out of fifty-seven.
//
// The size RATIO here (1 : 2 : 3) is exaggerated relative to M3's
// (1 : 1.15 : 1.35) because K-quant block granularity is 256 elements and
// matching the real ratio would need a ~22 MB fixture. What is being tested is
// non-uniformity, not the specific ratio.
//
// WHY three different quant types in one file: Invariant 3 -- never assume
// bits-per-weight. Experts are Q4_K (144 B / 256 weights), attention is Q6_K
// (210 B / 256), the embedding is Q8_0 (34 B / 32), norms and the router gate
// are F32. Any code that derives a byte figure from an assumed bpw gets three
// different wrong answers here.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace dray::testfix {

// Mirrors dray::plan::TensorClass. Deliberately a separate enum: the fixture
// states what it WROTE, the planner states what it INFERRED, and the test is the
// comparison. Sharing one enum would still be sound, but sharing one derivation
// would not be.
enum class FixtureClass : uint8_t {
    RouterGate,
    NormOrBias,
    RoutedExpert,
    UnconditionalBulk,
    RowSliced,          // gathered by rows only (token_embd beside a separate output.weight)
};

const char* fixture_class_name(FixtureClass);

struct FixtureTensor {
    std::string  name;
    FixtureClass cls = FixtureClass::UnconditionalBulk;
    int32_t      layer = -1;   // -1 for non-layer tensors
    uint32_t     ggml_type = 0;
    uint64_t     offset = 0;   // ABSOLUTE byte offset of the data in the file
    uint64_t     bytes = 0;    // ggml_nbytes(), i.e. what the tensor table says
    uint32_t     seed = 0;     // fill pattern; see fill_byte()
    int32_t      shard_index = 0;   // which split file holds it (S20)
    int64_t      ne[4] = {1, 1, 1, 1};
};

struct FixtureLayer {
    int32_t  layer = -1;
    bool     is_moe = false;
    uint32_t n_experts = 0;
    uint32_t ffn_len = 0;
    uint64_t slot_bytes = 0;  // bytes for ONE expert on this layer:
                              // gate.nb[2] + up.nb[2] + down.nb[2]
};

struct GgufSpec {
    std::string arch = "minimax-m3";   // matches llama.cpp's LLM_ARCH_MINIMAX_M3
    std::string display_name = "dray-synthetic";

    uint32_t n_layers = 8;        // total blocks
    uint32_t n_dense_lead = 1;    // leading dense blocks; the rest are MoE
    uint32_t n_experts = 4;
    uint32_t n_expert_used = 2;

    // All row lengths must be multiples of 256 (K-quant block size), otherwise
    // ggml_row_size() asserts.
    uint32_t n_embd = 256;
    uint32_t n_head = 8;
    uint32_t n_head_kv = 2;
    uint32_t n_vocab = 512;
    uint32_t n_ctx_train = 4096;
    uint32_t dense_ffn_len = 512;   // ffn width of the leading dense blocks
    uint32_t shexp_ffn_len = 256;   // shared-expert width (unconditional bulk)

    // The three size classes. Must be multiples of 256.
    uint32_t expert_ffn_len[3] = {256, 512, 768};

    uint32_t alignment = 32;   // GGUF_DEFAULT_ALIGNMENT

    bool write_shared_experts = true;
    bool write_expert_bias = true;   // blk.N.exp_probs_b.bias
};

struct Fixture {
    bool        ok = false;
    std::string error;

    std::filesystem::path path;
    GgufSpec              spec;

    std::vector<FixtureTensor> tensors;
    std::vector<FixtureLayer>  layers;   // one entry per block, in block order

    uint64_t alignment = 0;
    uint64_t data_offset = 0;   // start of the tensor data section
    std::vector<std::filesystem::path> shard_paths;   // split fixtures only (S20)
    uint64_t file_size = 0;

    // Totals a residency plan has to reproduce from the file alone.
    uint64_t router_gate_bytes = 0;
    uint64_t norm_bias_bytes = 0;
    uint64_t routed_bytes = 0;
    uint64_t unconditional_bytes = 0;
    uint64_t row_sliced_bytes = 0;
    uint32_t n_moe_layers = 0;

    const FixtureTensor* find(std::string_view name) const;

    // Sorted, deduplicated slot sizes across MoE layers. Length >= 3 is the
    // point of the fixture.
    std::vector<uint64_t> distinct_slot_bytes() const;

    // Sum of slot_bytes over MoE layers, i.e. one expert per MoE layer.
    uint64_t slot_bytes_sum() const;

    // k * n_moe_layers * mean slot bytes, exactly as residency.h defines
    // cold_bytes_per_token.
    uint64_t expected_cold_bytes_per_token() const;
};

// Writes <dir>/<filename> and returns everything a test needs to check it.
// Never throws; on failure returns ok=false with a populated error.
// S20: two real GGUF shards with split metadata and the llama naming
// convention; tensor data byte-identical to the single-file build.
Fixture make_gguf_split(const std::filesystem::path& dir, const std::string& base,
                        const GgufSpec& spec);

Fixture make_gguf(const std::filesystem::path& dir, const std::string& filename,
                  const GgufSpec& spec);

// Deterministic fill, so a test can verify bytes that came back off a disk
// without keeping a copy of the whole file in memory.
uint8_t fill_byte(uint32_t seed, uint64_t index);
void    fill_buffer(uint32_t seed, void* dst, size_t bytes, uint64_t start_index = 0);
bool    verify_buffer(uint32_t seed, const void* src, size_t bytes, uint64_t start_index = 0);

// Raw-file helpers for the storage and snapshot tests.
bool write_pattern_file(const std::filesystem::path& path, uint64_t bytes, uint32_t seed);
bool truncate_file(const std::filesystem::path& path, uint64_t new_size);
bool corrupt_byte(const std::filesystem::path& path, uint64_t offset);

}  // namespace dray::testfix
