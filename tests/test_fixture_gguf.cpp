// The fixture is the backbone of every other test in this suite, so it gets its
// own: if make_gguf() writes a file whose real offsets do not match the table it
// hands the tests, then test_residency is comparing two wrong numbers and
// agreeing with itself.
//
// This case reads the generated file back through the vendored gguf reader --
// the same code path llama.cpp uses -- and checks that every byte figure the
// fixture reported is where it said it would be.

#include "harness.h"

#include "fixtures/make_gguf.h"
#include "fixtures/temp_dir.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using dray::testfix::Fixture;
using dray::testfix::FixtureClass;
using dray::testfix::FixtureLayer;
using dray::testfix::FixtureTensor;
using dray::testfix::GgufSpec;

namespace {

struct Env {
    dray::testfix::TempDir dir{"fixture"};
    Fixture fx;
    Env() { fx = dray::testfix::make_gguf(dir.path(), "tiny.gguf", GgufSpec{}); }
};

const Fixture& fixture() {
    static Env e;
    return e.fx;
}

}  // namespace

LZ_TEST(the_fixture_writes_a_file) {
    const Fixture& fx = fixture();
    if (!fx.ok) {
        LZ_FAIL("make_gguf failed: " + fx.error);
        return;
    }
    LZ_CHECK(std::filesystem::exists(fx.path));
    LZ_CHECK_GT(fx.file_size, fx.data_offset);
    LZ_CHECK_GT(fx.tensors.size(), 20u);
}

LZ_TEST(the_generated_file_parses_as_gguf_and_agrees_with_the_fixture) {
    const Fixture& fx = fixture();
    LZ_REQUIRE(fx.ok);

    gguf_init_params ip;
    ip.no_alloc = true;
    ip.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(fx.path.string().c_str(), ip);
    LZ_REQUIRE(g != nullptr);

    LZ_CHECK_EQ(static_cast<uint64_t>(gguf_get_alignment(g)), fx.alignment);
    LZ_CHECK_EQ(static_cast<uint64_t>(gguf_get_data_offset(g)), fx.data_offset);
    LZ_CHECK_EQ(gguf_get_n_tensors(g), static_cast<int64_t>(fx.tensors.size()));

    for (const FixtureTensor& want : fx.tensors) {
        const int64_t id = gguf_find_tensor(g, want.name.c_str());
        if (id < 0) {
            LZ_FAIL("tensor missing from the written file: " + want.name);
            continue;
        }
        LZ_CHECK_EQ(static_cast<uint64_t>(gguf_get_tensor_size(g, id)), want.bytes);
        LZ_CHECK_EQ(static_cast<uint32_t>(gguf_get_tensor_type(g, id)), want.ggml_type);
        LZ_CHECK_EQ(fx.data_offset + static_cast<uint64_t>(gguf_get_tensor_offset(g, id)),
                    want.offset);
    }

    // Metadata the planner will read back.
    const std::string arch_key = "general.architecture";
    const int64_t ka = gguf_find_key(g, arch_key.c_str());
    LZ_REQUIRE_GE(ka, 0);
    LZ_CHECK_EQ(std::string(gguf_get_val_str(g, ka)), fx.spec.arch);

    const std::string blocks_key = fx.spec.arch + ".block_count";
    const int64_t kb = gguf_find_key(g, blocks_key.c_str());
    LZ_REQUIRE_GE(kb, 0);
    LZ_CHECK_EQ(gguf_get_val_u32(g, kb), fx.spec.n_layers);

    gguf_free(g);
}

LZ_TEST(the_bytes_on_disk_are_at_the_offsets_the_fixture_reported) {
    const Fixture& fx = fixture();
    LZ_REQUIRE(fx.ok);

    std::ifstream f(fx.path, std::ios::binary);
    LZ_REQUIRE(static_cast<bool>(f));

    // Spot-check the first, last and every expert tensor: an off-by-one in the
    // offset arithmetic would give the residency tests a table that is
    // self-consistent and wrong.
    std::vector<const FixtureTensor*> probe;
    probe.push_back(&fx.tensors.front());
    probe.push_back(&fx.tensors.back());
    for (const FixtureTensor& t : fx.tensors) {
        if (t.cls == FixtureClass::RoutedExpert || t.cls == FixtureClass::RouterGate) {
            probe.push_back(&t);
        }
    }

    std::vector<uint8_t> head(256);
    for (const FixtureTensor* t : probe) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(head.size(), t->bytes));
        f.seekg(static_cast<std::streamoff>(t->offset), std::ios::beg);
        LZ_REQUIRE(static_cast<bool>(f.read(reinterpret_cast<char*>(head.data()),
                                            static_cast<std::streamsize>(n))));
        if (!dray::testfix::verify_buffer(t->seed, head.data(), n, 0)) {
            LZ_FAIL("bytes at the reported offset are not this tensor's: " + t->name);
        }

        // And the tail, which is what catches a size that is too small.
        if (t->bytes > head.size()) {
            const uint64_t tail_start = t->bytes - head.size();
            f.seekg(static_cast<std::streamoff>(t->offset + tail_start), std::ios::beg);
            LZ_REQUIRE(static_cast<bool>(f.read(reinterpret_cast<char*>(head.data()),
                                                static_cast<std::streamsize>(head.size()))));
            if (!dray::testfix::verify_buffer(t->seed, head.data(), head.size(), tail_start)) {
                LZ_FAIL("tail bytes do not match for: " + t->name);
            }
        }
    }
}

LZ_TEST(per_layer_expert_sizes_are_deliberately_non_uniform) {
    const Fixture& fx = fixture();
    LZ_REQUIRE(fx.ok);

    const std::vector<uint64_t> classes = fx.distinct_slot_bytes();
    // Three size classes, mirroring M3 UD-Q2_K_XL (18,137,088 / 20,938,752 /
    // 24,477,696 B). A fixture that collapsed to one class would let a planner
    // that assumes uniform layers pass by accident.
    LZ_CHECK_EQ(classes.size(), 3u);
    if (classes.size() == 3) {
        LZ_CHECK_LT(classes[0], classes[1]);
        LZ_CHECK_LT(classes[1], classes[2]);
    }

    // And the distribution is skewed, not round-robin: most layers share the
    // smallest class, exactly as the real model does.
    size_t smallest = 0;
    for (const FixtureLayer& l : fx.layers) {
        if (l.is_moe && l.slot_bytes == classes.front()) {
            ++smallest;
        }
    }
    LZ_CHECK_GT(smallest * 2, static_cast<size_t>(fx.n_moe_layers));

    for (const FixtureLayer& l : fx.layers) {
        if (l.is_moe) {
            LZ_CHECK_GT(l.slot_bytes, 0u);
            LZ_CHECK_EQ(l.n_experts, fx.spec.n_experts);
        } else {
            LZ_CHECK_EQ(l.slot_bytes, 0u);
        }
    }
    LZ_CHECK_EQ(fx.n_moe_layers, fx.spec.n_layers - fx.spec.n_dense_lead);
}

LZ_TEST(the_totals_the_planner_has_to_reproduce_are_self_consistent) {
    const Fixture& fx = fixture();
    LZ_REQUIRE(fx.ok);

    uint64_t sum = 0;
    for (const FixtureTensor& t : fx.tensors) {
        sum += t.bytes;
    }
    LZ_CHECK_EQ(sum, fx.router_gate_bytes + fx.norm_bias_bytes + fx.routed_bytes +
                         fx.unconditional_bytes + fx.row_sliced_bytes);
    LZ_CHECK_LE(sum, fx.file_size);

    // Routed experts must dominate: a fixture where they do not is not a model
    // this engine would ever stream.
    LZ_CHECK_GT(fx.routed_bytes, fx.unconditional_bytes);
    LZ_CHECK_GT(fx.router_gate_bytes, 0u);
    LZ_CHECK_GT(fx.norm_bias_bytes, 0u);

    // One expert per MoE layer, k times, is the fully-cold per-token cost.
    LZ_CHECK_EQ(fx.expected_cold_bytes_per_token(),
                static_cast<uint64_t>(fx.spec.n_expert_used) * fx.slot_bytes_sum());
    LZ_CHECK_LT(fx.expected_cold_bytes_per_token(), fx.routed_bytes);

    // Three different quant types are present on purpose (Invariant 3): no
    // single bits-per-weight assumption can reproduce these byte figures.
    std::vector<uint32_t> types;
    for (const FixtureTensor& t : fx.tensors) {
        types.push_back(t.ggml_type);
    }
    std::sort(types.begin(), types.end());
    types.erase(std::unique(types.begin(), types.end()), types.end());
    LZ_CHECK_GE(types.size(), 4u);
}
