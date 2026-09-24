// See repack.h. Layout per MoE layer, all sizes from the tensor table:
//
//   for e in 0..n_experts-1:
//     [ gate_e  pad->4096 ][ up_e  pad->4096 ][ down_e  pad->4096 ]
//
// Every tensor's per-expert DISK stride becomes the padded trio span --
// identical for gate/up/down, which is exactly what lets the existing compact
// path address expert e at source.offset + e * disk_stride with no new code.

#include "tools/repack.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"

#include "plan/residency.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace dray::tools {

namespace {

constexpr uint64_t kAlign = 4096;
uint64_t pad(uint64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

struct Part {
    const dray::plan::TensorInfo* t = nullptr;
    uint64_t per_expert = 0;   // bytes of one expert's slice in the ORIGINAL layout
    uint64_t intra = 0;        // offset of this part inside the trio span
};

}  // namespace

int repack_write(const std::string& model_gguf, const std::string& out_dir) {
    std::string err;
    // ctx doesn't matter for classification; the plan is only the tensor table here.
    dray::plan::Plan p = dray::plan::build_plan(model_gguf, 8ull << 30, 2048, &err);
    if (!err.empty()) { std::fprintf(stderr, "repack: plan failed: %s\n", err.c_str()); return 1; }

    std::map<int32_t, uint32_t> experts_of;   // layer -> n_experts
    for (const auto& sc : p.slot_classes) experts_of[sc.layer] = sc.n_experts;

    // layer -> the trio, in a FIXED part order (gate, up, down) so the layout is
    // deterministic and the map alone describes it.
    std::map<int32_t, std::vector<Part>> layers;
    static const char* kOrder[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    for (const auto& t : p.tensors) {
        if (t.cls != dray::plan::TensorClass::RoutedExpert) continue;
        for (int s = 0; s < 3; ++s) {
            if (t.name.find(kOrder[s]) == std::string::npos) continue;
            auto& v = layers[t.layer];
            v.resize(3);
            v[static_cast<size_t>(s)].t = &t;
            break;
        }
    }

    fs::create_directories(out_dir);
    const fs::path bin_path = fs::path(out_dir) / "repack.bin";
    std::ofstream out(bin_path, std::ios::binary | std::ios::trunc);
    if (!out) { std::fprintf(stderr, "repack: cannot write %s\n", bin_path.string().c_str()); return 1; }

    // One reader per shard, opened lazily.
    std::vector<std::ifstream> shards(p.shard_paths.size());
    auto reader = [&](int32_t idx) -> std::ifstream& {
        auto& f = shards[static_cast<size_t>(idx)];
        if (!f.is_open()) f.open(p.shard_paths[static_cast<size_t>(idx)], std::ios::binary);
        return f;
    };

    json entries = json::object();
    std::vector<char> buf;
    uint64_t cursor = 0;
    uint64_t layers_done = 0;

    for (auto& [layer, parts] : layers) {
        const uint32_t ne = experts_of.count(layer) ? experts_of[layer] : 0;
        if (ne == 0 || !parts[0].t || !parts[1].t || !parts[2].t) {
            std::fprintf(stderr, "repack: layer %d incomplete trio or unknown expert count\n", layer);
            return 1;
        }
        uint64_t span = 0;
        for (auto& part : parts) {
            if (part.t->bytes % ne != 0) {
                std::fprintf(stderr, "repack: %s bytes %% n_experts != 0\n", part.t->name.c_str());
                return 1;
            }
            part.per_expert = part.t->bytes / ne;
            part.intra = span;
            span += pad(part.per_expert);
        }

        const uint64_t layer_base = cursor;   // cursor stays 4096-aligned throughout
        for (uint32_t e = 0; e < ne; ++e) {
            for (const auto& part : parts) {
                std::ifstream& in = reader(part.t->shard);
                if (!in) { std::fprintf(stderr, "repack: cannot open shard %d\n", part.t->shard); return 1; }
                in.seekg(static_cast<std::streamoff>(part.t->offset + e * part.per_expert));
                buf.resize(static_cast<size_t>(pad(part.per_expert)), 0);
                std::memset(buf.data(), 0, buf.size());
                if (!in.read(buf.data(), static_cast<std::streamsize>(part.per_expert))) {
                    std::fprintf(stderr, "repack: short read in %s expert %u\n",
                                 part.t->name.c_str(), e);
                    return 1;
                }
                out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
                cursor += buf.size();
            }
        }
        for (const auto& part : parts) {
            entries[part.t->name] = {
                { "offset", layer_base + part.intra },
                { "disk_stride", span },
            };
        }
        ++layers_done;
        std::fprintf(stderr, "\rrepack: layer %lld done, %.1f GB written",
                     static_cast<long long>(layers_done), cursor / 1e9);
    }
    out.close();
    std::fprintf(stderr, "\nrepack: %.2f GB -> %s\n", cursor / 1e9, bin_path.string().c_str());

    json map = {
        { "companion", "repack.bin" },
        { "align", kAlign },
        { "entries", entries },
    };
    std::ofstream jm(fs::path(out_dir) / "repack.json", std::ios::trunc);
    jm << map.dump(1);
    return 0;
}

bool repack_apply(dray::plan::Plan* p, const std::string& dir, std::string* err) {
    const fs::path jp = fs::path(dir) / "repack.json";
    if (!fs::exists(jp)) { if (err) *err = "no repack.json in " + dir; return false; }

    json map;
    try {
        std::ifstream f(jp);
        f >> map;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("repack.json unreadable: ") + ex.what();
        return false;
    }
    const fs::path bin = fs::path(dir) / map.value("companion", "repack.bin");
    if (!fs::exists(bin)) { if (err) *err = "companion missing: " + bin.string(); return false; }

    if (p->n_meta_shards == 0) p->n_meta_shards = p->shard_paths.size();
    const int32_t shard = static_cast<int32_t>(p->shard_paths.size());
    p->shard_paths.push_back(bin.string());

    size_t applied = 0;
    for (auto& t : p->tensors) {
        auto it = map["entries"].find(t.name);
        if (it == map["entries"].end()) continue;
        t.shard = shard;
        t.offset = (*it)["offset"].get<uint64_t>();
        t.disk_stride = (*it)["disk_stride"].get<uint64_t>();
        ++applied;
    }
    std::fprintf(stderr, "repack: %zu tensors sourced from %s\n",
                 applied, bin.string().c_str());
    return applied > 0;
}

}  // namespace dray::tools
