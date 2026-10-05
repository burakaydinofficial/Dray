#include "fixtures/make_gguf.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <system_error>

namespace dray::testfix {
namespace {

// Names follow llama.cpp's LLM_TENSOR_NAMES exactly (src/llama-arch.cpp), with
// the ".weight"/".bias" suffix the loader appends. Anything else would test the
// planner against a naming convention no real GGUF uses.
std::string blk_name(uint32_t il, const char* suffix) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%u.%s", il, suffix);
    return std::string(buf);
}

std::string kv_key(const std::string& arch, const char* suffix) {
    return arch + "." + suffix;
}

// Skewed on purpose. M3 UD-Q2_K_XL has three per-layer expert sizes with the
// largest on only a handful of layers; a round-robin fixture would let a planner
// that assumes uniform-modulo-3 layers pass by accident.
uint32_t size_class_of(const GgufSpec& s, uint32_t il) {
    if (il + 1 == s.n_layers) {
        return 2;
    }
    if ((il % 4) == 1) {
        return 1;
    }
    return 0;
}

struct Builder {
    ggml_context* ctx = nullptr;
    gguf_context* gg = nullptr;

    std::vector<ggml_tensor*>  handles;
    std::vector<FixtureTensor> records;

    ggml_tensor* add(FixtureClass cls, int32_t layer, ggml_type type, const std::string& name,
                     int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor* t = nullptr;
        if (ne2 > 1) {
            t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        } else if (ne1 > 1) {
            t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        } else {
            t = ggml_new_tensor_1d(ctx, type, ne0);
        }
        if (t == nullptr) {
            return nullptr;
        }
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(gg, t);

        FixtureTensor rec;
        rec.name = name;
        rec.cls = cls;
        rec.layer = layer;
        rec.ggml_type = static_cast<uint32_t>(type);
        rec.bytes = static_cast<uint64_t>(ggml_nbytes(t));
        rec.seed = static_cast<uint32_t>(records.size()) + 1u;  // 0 is reserved for "no fill"
        for (int i = 0; i < 4; ++i) {
            rec.ne[i] = t->ne[i];
        }

        handles.push_back(t);
        records.push_back(rec);
        return t;
    }
};

}  // namespace

const char* fixture_class_name(FixtureClass c) {
    switch (c) {
        case FixtureClass::RouterGate:        return "RouterGate";
        case FixtureClass::NormOrBias:        return "NormOrBias";
        case FixtureClass::RoutedExpert:      return "RoutedExpert";
        case FixtureClass::UnconditionalBulk: return "UnconditionalBulk";
        case FixtureClass::RowSliced:         return "RowSliced";
    }
    return "?";
}

uint8_t fill_byte(uint32_t seed, uint64_t index) {
    // splitmix64 on (seed, index). Cheap, deterministic across platforms and
    // runs, and unlike memset it catches an off-by-one in a read offset.
    uint64_t x = (static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull) ^
                 (index * 0xBF58476D1CE4E5B9ull + 0x94D049BB133111EBull);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<uint8_t>(x & 0xFFull);
}

void fill_buffer(uint32_t seed, void* dst, size_t bytes, uint64_t start_index) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    for (size_t i = 0; i < bytes; ++i) {
        p[i] = fill_byte(seed, start_index + static_cast<uint64_t>(i));
    }
}

bool verify_buffer(uint32_t seed, const void* src, size_t bytes, uint64_t start_index) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    for (size_t i = 0; i < bytes; ++i) {
        if (p[i] != fill_byte(seed, start_index + static_cast<uint64_t>(i))) {
            return false;
        }
    }
    return true;
}

bool write_pattern_file(const std::filesystem::path& path, uint64_t bytes, uint32_t seed) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    constexpr size_t kChunk = 64 * 1024;
    std::vector<uint8_t> chunk(kChunk);
    uint64_t written = 0;
    while (written < bytes) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(kChunk, bytes - written));
        fill_buffer(seed, chunk.data(), n, written);
        f.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(n));
        if (!f) {
            return false;
        }
        written += n;
    }
    f.close();
    return static_cast<bool>(f);
}

bool truncate_file(const std::filesystem::path& path, uint64_t new_size) {
    std::error_code ec;
    std::filesystem::resize_file(path, static_cast<std::uintmax_t>(new_size), ec);
    return !ec;
}

bool corrupt_byte(const std::filesystem::path& path, uint64_t offset) {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) {
        return false;
    }
    f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    char c = 0;
    if (!f.read(&c, 1)) {
        return false;
    }
    c = static_cast<char>(~static_cast<unsigned char>(c));
    f.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!f.write(&c, 1)) {
        return false;
    }
    f.close();
    return true;
}

const FixtureTensor* Fixture::find(std::string_view name) const {
    for (const FixtureTensor& t : tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

std::vector<uint64_t> Fixture::distinct_slot_bytes() const {
    std::vector<uint64_t> v;
    for (const FixtureLayer& l : layers) {
        if (l.is_moe) {
            v.push_back(l.slot_bytes);
        }
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

uint64_t Fixture::slot_bytes_sum() const {
    uint64_t sum = 0;
    for (const FixtureLayer& l : layers) {
        if (l.is_moe) {
            sum += l.slot_bytes;
        }
    }
    return sum;
}

uint64_t Fixture::expected_cold_bytes_per_token() const {
    // misses_per_token = k_routed * n_moe_layers (at h_routed = 0), and the byte
    // cost of a miss is that LAYER's slot size, not a global one -- which is the
    // whole reason slot classes are per-layer.
    return static_cast<uint64_t>(spec.n_expert_used) * slot_bytes_sum();
}

Fixture make_gguf(const std::filesystem::path& dir, const std::string& filename,
                  const GgufSpec& spec) {
    Fixture fx;
    fx.spec = spec;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    fx.path = dir / filename;

    if (spec.n_layers == 0 || spec.n_dense_lead > spec.n_layers) {
        fx.error = "invalid layer configuration";
        return fx;
    }
    if ((spec.n_embd % 256) != 0) {
        fx.error = "n_embd must be a multiple of the 256-element K-quant block";
        return fx;
    }
    if (spec.alignment != GGUF_DEFAULT_ALIGNMENT) {
        // The vendored writer cannot honour a different one: gguf_context's
        // alignment field is fixed at construction (ggml/src/gguf.cpp:223) and
        // gguf_set_val_u32("general.alignment", ...) only validates the value,
        // it does not update the context. Writing a different KV would produce a
        // file whose stated alignment contradicts its own padding.
        fx.error = "only GGUF_DEFAULT_ALIGNMENT is writable by the vendored gguf writer";
        return fx;
    }

    // Metadata only: tensor data lives in our own buffers so the fixture can
    // hand gguf a pointer per tensor without a second ggml allocation pass.
    const size_t est_tensors = static_cast<size_t>(spec.n_layers) * 20u + 16u;
    ggml_init_params ip;
    ip.mem_size = est_tensors * ggml_tensor_overhead() + 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;

    Builder b;
    b.ctx = ggml_init(ip);
    if (b.ctx == nullptr) {
        fx.error = "ggml_init failed";
        return fx;
    }
    b.gg = gguf_init_empty();
    if (b.gg == nullptr) {
        ggml_free(b.ctx);
        fx.error = "gguf_init_empty failed";
        return fx;
    }

    // Stated explicitly so a reader that honours the key and a reader that
    // assumes the default agree on this file. It matches gguf_get_alignment(),
    // which is the value the writer actually pads to.
    gguf_set_val_u32(b.gg, GGUF_KEY_GENERAL_ALIGNMENT,
                     static_cast<uint32_t>(gguf_get_alignment(b.gg)));

    gguf_set_val_str(b.gg, "general.architecture", spec.arch.c_str());
    gguf_set_val_str(b.gg, "general.name", spec.display_name.c_str());
    gguf_set_val_str(b.gg, "general.type", "model");
    gguf_set_val_u32(b.gg, "general.quantization_version", 2);
    gguf_set_val_u32(b.gg, "general.file_type", 15);  // MOSTLY_Q4_K_M, cosmetic

    const std::string& a = spec.arch;
    gguf_set_val_u32(b.gg, kv_key(a, "block_count").c_str(), spec.n_layers);
    gguf_set_val_u32(b.gg, kv_key(a, "leading_dense_block_count").c_str(), spec.n_dense_lead);
    gguf_set_val_u32(b.gg, kv_key(a, "context_length").c_str(), spec.n_ctx_train);
    gguf_set_val_u32(b.gg, kv_key(a, "embedding_length").c_str(), spec.n_embd);
    gguf_set_val_u32(b.gg, kv_key(a, "feed_forward_length").c_str(), spec.dense_ffn_len);
    // Deliberately reports only the SMALLEST expert width even though two other
    // widths exist in the file. A planner that sizes slots from this KV instead
    // of from the tensor table (Invariant 3) gets the big layers wrong, and the
    // residency test will catch it.
    gguf_set_val_u32(b.gg, kv_key(a, "expert_feed_forward_length").c_str(),
                     spec.expert_ffn_len[0]);
    gguf_set_val_u32(b.gg, kv_key(a, "expert_count").c_str(), spec.n_experts);
    gguf_set_val_u32(b.gg, kv_key(a, "expert_used_count").c_str(), spec.n_expert_used);
    gguf_set_val_u32(b.gg, kv_key(a, "attention.head_count").c_str(), spec.n_head);
    gguf_set_val_u32(b.gg, kv_key(a, "attention.head_count_kv").c_str(), spec.n_head_kv);

    const uint32_t head_dim = spec.n_head > 0 ? spec.n_embd / spec.n_head : 0;
    gguf_set_val_u32(b.gg, kv_key(a, "attention.key_length").c_str(), head_dim);
    gguf_set_val_u32(b.gg, kv_key(a, "attention.value_length").c_str(), head_dim);
    gguf_set_val_f32(b.gg, kv_key(a, "attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_u32(b.gg, kv_key(a, "rope.dimension_count").c_str(), head_dim);
    gguf_set_val_f32(b.gg, kv_key(a, "rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_u32(b.gg, kv_key(a, "vocab_size").c_str(), spec.n_vocab);

    // NOTE: no tokenizer arrays. Nothing under test loads this file through
    // llama_model_load_from_file -- the residency planner reads the tensor table
    // directly -- and a synthetic vocab would triple the fixture for no coverage.

    const int64_t n_embd = static_cast<int64_t>(spec.n_embd);
    const int64_t n_embd_kv = spec.n_head > 0
                                  ? static_cast<int64_t>(spec.n_embd) *
                                        static_cast<int64_t>(spec.n_head_kv) /
                                        static_cast<int64_t>(spec.n_head)
                                  : n_embd;
    const int64_t n_experts = static_cast<int64_t>(spec.n_experts);

    b.add(FixtureClass::RowSliced, -1, GGML_TYPE_Q8_0, "token_embd.weight", n_embd,
          static_cast<int64_t>(spec.n_vocab), 1);

    fx.layers.resize(spec.n_layers);

    for (uint32_t il = 0; il < spec.n_layers; ++il) {
        const int32_t layer = static_cast<int32_t>(il);
        const bool is_moe = il >= spec.n_dense_lead;

        FixtureLayer& L = fx.layers[il];
        L.layer = layer;
        L.is_moe = is_moe;

        b.add(FixtureClass::NormOrBias, layer, GGML_TYPE_F32, blk_name(il, "attn_norm.weight"),
              n_embd, 1, 1);
        b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q6_K,
              blk_name(il, "attn_q.weight"), n_embd, n_embd, 1);
        b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q6_K,
              blk_name(il, "attn_k.weight"), n_embd, n_embd_kv, 1);
        b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q6_K,
              blk_name(il, "attn_v.weight"), n_embd, n_embd_kv, 1);
        b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q6_K,
              blk_name(il, "attn_output.weight"), n_embd, n_embd, 1);
        b.add(FixtureClass::NormOrBias, layer, GGML_TYPE_F32, blk_name(il, "ffn_norm.weight"),
              n_embd, 1, 1);

        if (!is_moe) {
            const int64_t ff = static_cast<int64_t>(spec.dense_ffn_len);
            b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q4_K,
                  blk_name(il, "ffn_gate.weight"), n_embd, ff, 1);
            b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q4_K,
                  blk_name(il, "ffn_up.weight"), n_embd, ff, 1);
            b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q4_K,
                  blk_name(il, "ffn_down.weight"), ff, n_embd, 1);
            continue;
        }

        const uint32_t cls = size_class_of(spec, il);
        const int64_t ff = static_cast<int64_t>(spec.expert_ffn_len[cls]);
        L.n_experts = spec.n_experts;
        L.ffn_len = spec.expert_ffn_len[cls];

        // The gate is resident at ANY cap: a non-resident gate has to be read
        // before you know which experts to read, which is a serialized round
        // trip per layer per token that can never be overlapped.
        b.add(FixtureClass::RouterGate, layer, GGML_TYPE_F32,
              blk_name(il, "ffn_gate_inp.weight"), n_embd, n_experts, 1);
        if (spec.write_expert_bias) {
            b.add(FixtureClass::NormOrBias, layer, GGML_TYPE_F32,
                  blk_name(il, "exp_probs_b.bias"), n_experts, 1, 1);
        }

        const ggml_tensor* g = b.add(FixtureClass::RoutedExpert, layer, GGML_TYPE_Q4_K,
                                     blk_name(il, "ffn_gate_exps.weight"), n_embd, ff, n_experts);
        const ggml_tensor* u = b.add(FixtureClass::RoutedExpert, layer, GGML_TYPE_Q4_K,
                                     blk_name(il, "ffn_up_exps.weight"), n_embd, ff, n_experts);
        const ggml_tensor* d = b.add(FixtureClass::RoutedExpert, layer, GGML_TYPE_Q4_K,
                                     blk_name(il, "ffn_down_exps.weight"), ff, n_embd, n_experts);
        if (g == nullptr || u == nullptr || d == nullptr) {
            gguf_free(b.gg);
            ggml_free(b.ctx);
            fx.error = "ran out of ggml context space";
            return fx;
        }

        // nb[2] is the per-expert stride: exactly what a slab slot must hold for
        // ggml_mul_mat_id to index the layer's experts from one base pointer.
        L.slot_bytes = static_cast<uint64_t>(g->nb[2]) + static_cast<uint64_t>(u->nb[2]) +
                       static_cast<uint64_t>(d->nb[2]);

        if (spec.write_shared_experts) {
            const int64_t sh = static_cast<int64_t>(spec.shexp_ffn_len);
            b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q4_K,
                  blk_name(il, "ffn_gate_shexp.weight"), n_embd, sh, 1);
            b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q4_K,
                  blk_name(il, "ffn_up_shexp.weight"), n_embd, sh, 1);
            b.add(FixtureClass::UnconditionalBulk, layer, GGML_TYPE_Q4_K,
                  blk_name(il, "ffn_down_shexp.weight"), sh, n_embd, 1);
        }
    }

    b.add(FixtureClass::NormOrBias, -1, GGML_TYPE_F32, "output_norm.weight", n_embd, 1, 1);
    b.add(FixtureClass::UnconditionalBulk, -1, GGML_TYPE_Q6_K, "output.weight", n_embd,
          static_cast<int64_t>(spec.n_vocab), 1);

    for (const ggml_tensor* t : b.handles) {
        if (t == nullptr) {
            gguf_free(b.gg);
            ggml_free(b.ctx);
            fx.error = "ran out of ggml context space";
            return fx;
        }
    }

    // Offsets are only knowable once every tensor and every KV pair is present:
    // the data section starts after the padded metadata.
    fx.alignment = static_cast<uint64_t>(gguf_get_alignment(b.gg));
    fx.data_offset = static_cast<uint64_t>(gguf_get_meta_size(b.gg));

    std::vector<std::vector<uint8_t>> payload(b.records.size());
    for (size_t i = 0; i < b.records.size(); ++i) {
        FixtureTensor& rec = b.records[i];
        const int64_t id = gguf_find_tensor(b.gg, rec.name.c_str());
        if (id < 0) {
            gguf_free(b.gg);
            ggml_free(b.ctx);
            fx.error = "tensor vanished from gguf context: " + rec.name;
            return fx;
        }
        rec.offset = fx.data_offset + static_cast<uint64_t>(gguf_get_tensor_offset(b.gg, id));

        payload[i].resize(static_cast<size_t>(rec.bytes));
        fill_buffer(rec.seed, payload[i].data(), payload[i].size(), 0);
        gguf_set_tensor_data(b.gg, rec.name.c_str(), payload[i].data());
    }

    const bool written = gguf_write_to_file(b.gg, fx.path.string().c_str(), /*only_meta =*/false);

    gguf_free(b.gg);
    ggml_free(b.ctx);

    if (!written) {
        fx.error = "gguf_write_to_file failed for " + fx.path.string();
        return fx;
    }

    fx.tensors = std::move(b.records);
    fx.file_size = static_cast<uint64_t>(std::filesystem::file_size(fx.path, ec));
    if (ec) {
        fx.error = "file_size failed";
        return fx;
    }

    for (const FixtureTensor& t : fx.tensors) {
        switch (t.cls) {
            case FixtureClass::RouterGate:        fx.router_gate_bytes += t.bytes; break;
            case FixtureClass::NormOrBias:        fx.norm_bias_bytes += t.bytes; break;
            case FixtureClass::RoutedExpert:      fx.routed_bytes += t.bytes; break;
            case FixtureClass::UnconditionalBulk: fx.unconditional_bytes += t.bytes; break;
            case FixtureClass::RowSliced:         fx.row_sliced_bytes += t.bytes; break;
        }
    }
    for (const FixtureLayer& l : fx.layers) {
        if (l.is_moe) {
            ++fx.n_moe_layers;
        }
    }

    fx.ok = true;
    return fx;
}

}  // namespace dray::testfix

namespace dray::testfix {

// S20: the two-shard fixture. Builds the proven single-file fixture, then
// SPLITS it: each shard is a real GGUF with the full KV block (copied via
// gguf_set_kv, so the shards cannot drift from the single-file builder) plus
// the split.no / split.count / split.tensors.count keys residency.cpp parses,
// named by the "-%05d-of-%05d.gguf" convention llama_split_path produces.
// Tensor data is carried over byte-for-byte from the single file, so every
// FixtureTensor keeps its seed-derived fill pattern -- which is what lets a
// test prove bytes came from the RIGHT FILE, the regression this exists for.
Fixture make_gguf_split(const std::filesystem::path& dir, const std::string& base,
                        const GgufSpec& spec) {
    Fixture fx = make_gguf(dir, base + "-single-tmp.gguf", spec);
    if (!fx.ok) return fx;

    // Read the single file whole; offsets in fx.tensors index into it.
    std::vector<uint8_t> whole;
    {
        std::ifstream f(fx.path, std::ios::binary);
        f.seekg(0, std::ios::end);
        const std::streamoff sz = f.tellg();
        // T27: tellg is -1 on a bad stream; resize(-1) violates the
        // never-throws contract this header documents.
        if (!f.good() || sz <= 0) {
            fx.ok = false;
            fx.error = "could not stat single fixture";
            return fx;
        }
        whole.resize(static_cast<size_t>(sz));
        f.seekg(0);
        if (!f.read(reinterpret_cast<char*>(whole.data()),
                    static_cast<std::streamsize>(whole.size()))) {
            fx.ok = false;
            fx.error = "could not re-read single fixture";
            return fx;
        }
    }

    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context* src = gguf_init_from_file(fx.path.string().c_str(), gp);
    if (src == nullptr) {
        fx.ok = false;
        fx.error = "gguf_init_from_file failed on single fixture";
        return fx;
    }

    // T26: cut INSIDE a MoE trio when possible (gate on shard 1, up/down on
    // shard 2), so the split exercises a trio straddling files -- the exact
    // arrangement "right offset, wrong file" once shipped under. Falls back to
    // the midpoint when no trio crosses it.
    size_t cut = fx.tensors.size() / 2;
    for (size_t i = cut; i < fx.tensors.size(); ++i) {
        if (fx.tensors[i].name.find("ffn_up_exps") != std::string::npos) {
            cut = i;   // gate (before up) stays in shard 1; up/down go to shard 2
            break;
        }
    }
    const int n_split = 2;
    std::vector<FixtureTensor> updated;

    for (int s = 0; s < n_split && fx.ok; ++s) {
        char name[512];
        std::snprintf(name, sizeof(name), "%s-%05d-of-%05d.gguf",
                      base.c_str(), s + 1, n_split);
        const std::filesystem::path shard_path = dir / name;

        const size_t lo = s == 0 ? 0 : cut;
        const size_t hi = s == 0 ? cut : fx.tensors.size();

        const size_t est = (hi - lo) + 8;
        ggml_init_params ip;
        ip.mem_size = est * ggml_tensor_overhead() + 4096;
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;
        ggml_context* tctx = ggml_init(ip);
        gguf_context* gg = gguf_init_empty();
        if (tctx == nullptr || gg == nullptr) {
            // T27: the paired frees ran only on the success path.
            if (gg) gguf_free(gg);
            if (tctx) ggml_free(tctx);
            fx.ok = false;
            fx.error = "shard context init failed";
            break;
        }
        gguf_set_kv(gg, src);   // the entire KV block, no drift
        gguf_set_val_u16(gg, "split.no", static_cast<uint16_t>(s));
        gguf_set_val_u16(gg, "split.count", static_cast<uint16_t>(n_split));
        gguf_set_val_i32(gg, "split.tensors.count",
                         static_cast<int32_t>(fx.tensors.size()));

        std::vector<ggml_tensor*> tts;
        for (size_t i = lo; i < hi; ++i) {
            const FixtureTensor& r = fx.tensors[i];
            ggml_tensor* t = ggml_new_tensor_4d(tctx, static_cast<ggml_type>(r.ggml_type),
                                                r.ne[0], r.ne[1], r.ne[2], r.ne[3]);
            if (t == nullptr) { fx.ok = false; fx.error = "shard tensor alloc failed"; break; }
            ggml_set_name(t, r.name.c_str());
            gguf_add_tensor(gg, t);
            tts.push_back(t);
        }
        if (fx.ok) {
            for (size_t i = lo; i < hi; ++i) {
                const FixtureTensor& r = fx.tensors[i];
                gguf_set_tensor_data(gg, r.name.c_str(), whole.data() + r.offset);
            }
            const uint64_t data_off = static_cast<uint64_t>(gguf_get_meta_size(gg));
            if (!gguf_write_to_file(gg, shard_path.string().c_str(), false)) {
                fx.ok = false;
                fx.error = "gguf_write_to_file failed for shard";
            } else {
                for (size_t i = lo; i < hi; ++i) {
                    FixtureTensor r = fx.tensors[i];
                    const int64_t id = gguf_find_tensor(gg, r.name.c_str());
                    if (id < 0) {   // T27: mirror the sibling guard; never assert-abort
                        fx.ok = false;
                        fx.error = "tensor vanished from shard context: " + r.name;
                        break;
                    }
                    r.offset = data_off + static_cast<uint64_t>(gguf_get_tensor_offset(gg, id));
                    r.shard_index = s;
                    updated.push_back(std::move(r));
                }
                fx.shard_paths.push_back(shard_path);
            }
        }
        gguf_free(gg);
        ggml_free(tctx);
    }
    gguf_free(src);

    std::error_code ec;
    std::filesystem::remove(fx.path, ec);   // the temp single file
    if (fx.ok) {
        fx.tensors = std::move(updated);
        fx.path = fx.shard_paths.empty() ? fx.path : fx.shard_paths[0];
        // T27: file_size described a file this function just deleted. It now
        // means TOTAL bytes across shards (what cap arithmetic wants);
        // data_offset is shard 1's, refreshed from its own table.
        uint64_t total = 0;
        for (const auto& sp : fx.shard_paths) {
            total += static_cast<uint64_t>(std::filesystem::file_size(sp, ec));
        }
        fx.file_size = total;
    }
    return fx;
}

}  // namespace dray::testfix
