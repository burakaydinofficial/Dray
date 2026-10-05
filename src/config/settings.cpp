#include "config/settings.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>

#include "json.hpp"

namespace dray::config {

// Cap parsing accepts a plain byte count or a K/M/G/T suffix. GiB vs GB is a real
// trap: "64 GB of RAM" almost always means 64 GiB. We treat bare G as GiB
// (what users mean) and print both units everywhere so the ambiguity never bites.
uint64_t parse_size(const std::string& s, bool* ok) {
    *ok = false;
    if (s.empty()) return 0;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || v < 0) return 0;
    uint64_t mult = 1;
    switch (*end) {
        case 'k': case 'K': mult = 1ull << 10; ++end; break;
        case 'm': case 'M': mult = 1ull << 20; ++end; break;
        case 'g': case 'G': mult = 1ull << 30; ++end; break;
        case 't': case 'T': mult = 1ull << 40; ++end; break;
        case '\0': break;
        default: return 0;
    }
    if (*end == 'b' || *end == 'B') ++end;
    if (*end != '\0') return 0;
    *ok = true;
    return static_cast<uint64_t>(v * static_cast<double>(mult));
}

using nlohmann::json;

const char* source_name(Source s) {
    switch (s) {
        case Source::Builtin:       return "built-in";
        case Source::ShippedSystem: return "shipped system";
        case Source::UserSystem:    return "user system";
        case Source::ShippedModel:  return "shipped model";
        case Source::UserModel:     return "user model";
        case Source::Cli:           return "command line";
    }
    return "?";
}

Settings builtin_settings() {
    Settings s;
    // Measured optimum on the reference machine (2026-08-21): decode is
    // bandwidth-bound and flat from 4 threads up. A machine that differs says
    // so in its system config.
    s.decode_threads.value = 4;
    s.prefill_threads.value = 0;        // every core the process may use
    s.reserved_threads.value = 9;       // counted on the reference machine
    s.queue_depth.value = 64;
    s.prefill_chunk_cpu.value = 512;    // llama.cpp's own n_batch default
    s.prefill_chunk_gpu.value = 2048;   // amortises the PCIe weight transfer
    s.admit_byte_fraction.value = 0.10;
    s.serve_max_parallel.value = 1;               // one generation at a time until configured
    s.serve_prefill_tokens_per_step.value = 256;
    s.serve_checkpoints_per_slot.value = 2;
    s.serve_checkpoint_offset.value = 4;          // as llama-server
    s.serve_kv_idle_timeout_s.value = 600;
    s.serve_kv_pool_tokens.value = -1;           // context x max_parallel
    s.gpu_vram_cap.value = 0;                     // automatic, see below
    s.gpu_vram_auto_fraction.value = 0.25;        // the owner's rule: a part of the machine
    s.gpu_vram_auto_fraction_small.value = 0.5;
    s.gpu_vram_small_card.value = 2ull << 30;
    return s;
}

namespace {

struct Layer {
    const std::string& origin;
    Source             source;
    std::string*       err;
};

bool fail(const Layer& l, const std::string& key, const std::string& why) {
    *l.err = l.origin + ": \"" + key + "\": " + why;
    return false;
}

bool take_int(const Layer& l, const json& v, const std::string& key, int64_t lo, int64_t hi,
              Setting<int>* s) {
    if (!v.is_number_integer()) return fail(l, key, "must be an integer");
    const int64_t x = v.get<int64_t>();
    if (x < lo || x > hi) {
        return fail(l, key, "must be in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    }
    s->set(static_cast<int>(x), l.source, l.origin);
    return true;
}

bool take_fraction(const Layer& l, const json& v, const std::string& key, Setting<double>* s) {
    if (!v.is_number()) return fail(l, key, "must be a number");
    const double x = v.get<double>();
    if (!(x >= 0.0 && x <= 1.0)) return fail(l, key, "must be in [0, 1]");
    s->set(x, l.source, l.origin);
    return true;
}

bool take_strings(const Layer& l, const json& v, const std::string& key,
                  Setting<std::vector<std::string>>* s) {
    if (!v.is_array()) return fail(l, key, "must be an array of strings");
    std::vector<std::string> out;
    for (const json& e : v) {
        if (!e.is_string()) return fail(l, key, "must be an array of strings");
        out.push_back(e.get<std::string>());
    }
    s->set(out, l.source, l.origin);
    return true;
}

// A size: a K/M/G/T string ("2G") or a plain byte count.
bool take_size(const Layer& l, const json& v, const std::string& key, Setting<uint64_t>* s) {
    bool ok = false;
    uint64_t x = 0;
    if (v.is_number_integer() && v.get<int64_t>() >= 0) {
        x = static_cast<uint64_t>(v.get<int64_t>());
        ok = true;
    } else if (v.is_string()) {
        x = parse_size(v.get<std::string>(), &ok);
    }
    if (!ok) return fail(l, key, "must be a size such as \"2G\" or \"1536M\", or a byte count");
    s->set(x, l.source, l.origin);
    return true;
}

bool expect_object(const Layer& l, const json& v, const std::string& key) {
    return v.is_object() || fail(l, key, "must be an object");
}

// The policy sections, shared by the system file and a model's "overrides".
bool apply_policy_section(const Layer& l, const std::string& name, const json& v,
                          const std::string& prefix, Settings* s) {
    const std::string at = prefix + name;
    if (!expect_object(l, v, at)) return false;
    for (const auto& [k, val] : v.items()) {
        const std::string key = at + "." + k;
        bool ok = false;
        if (name == "prefill" && k == "chunk_cpu") {
            ok = take_int(l, val, key, 16, 8192, &s->prefill_chunk_cpu);
        } else if (name == "prefill" && k == "chunk_gpu") {
            ok = take_int(l, val, key, 16, 8192, &s->prefill_chunk_gpu);
        } else if (name == "cache" && k == "admit_byte_fraction") {
            ok = take_fraction(l, val, key, &s->admit_byte_fraction);
        } else if (name == "serve" && k == "max_parallel") {
            ok = take_int(l, val, key, 1, 1024, &s->serve_max_parallel);
        } else if (name == "serve" && k == "prefill_tokens_per_step") {
            ok = take_int(l, val, key, 1, 65536, &s->serve_prefill_tokens_per_step);
        } else if (name == "serve" && k == "checkpoints_per_slot") {
            ok = take_int(l, val, key, 0, 64, &s->serve_checkpoints_per_slot);
        } else if (name == "serve" && k == "checkpoint_offset") {
            ok = take_int(l, val, key, 1, 65536, &s->serve_checkpoint_offset);
        } else if (name == "serve" && k == "kv_idle_timeout_s") {
            ok = take_int(l, val, key, 0, 604800, &s->serve_kv_idle_timeout_s);
        } else if (name == "serve" && k == "kv_pool_tokens") {
            ok = take_int(l, val, key, -1, 2147483647, &s->serve_kv_pool_tokens);
        } else {
            return fail(l, key, "unknown key");
        }
        if (!ok) return false;
    }
    return true;
}

bool is_policy_section(const std::string& k) {
    return k == "prefill" || k == "cache" || k == "serve";
}

bool check_schema(const Layer& l, const json& v) {
    if (!v.is_number_integer() || v.get<int64_t>() != 1) {
        return fail(l, "schema", "unsupported schema version (this build reads 1)");
    }
    return true;
}

bool apply_system(const Layer& l, const json& root, Settings* s) {
    bool schema = false;
    for (const auto& [k, v] : root.items()) {
        if (k == "schema") {
            if (!check_schema(l, v)) return false;
            schema = true;
        } else if (k == "threads") {
            if (!expect_object(l, v, k)) return false;
            for (const auto& [tk, tv] : v.items()) {
                const std::string key = "threads." + tk;
                bool ok = false;
                if (tk == "decode")        ok = take_int(l, tv, key, 1, 1024, &s->decode_threads);
                else if (tk == "prefill")  ok = take_int(l, tv, key, 0, 1024, &s->prefill_threads);
                else if (tk == "reserved") ok = take_int(l, tv, key, 0, 1024, &s->reserved_threads);
                else return fail(l, key, "unknown key");
                if (!ok) return false;
            }
        } else if (k == "gpu") {
            if (!expect_object(l, v, k)) return false;
            for (const auto& [gk, gv] : v.items()) {
                const std::string key = "gpu." + gk;
                bool ok = false;
                if (gk == "vram_cap")                      ok = take_size(l, gv, key, &s->gpu_vram_cap);
                else if (gk == "vram_auto_fraction")       ok = take_fraction(l, gv, key, &s->gpu_vram_auto_fraction);
                else if (gk == "vram_auto_fraction_small") ok = take_fraction(l, gv, key, &s->gpu_vram_auto_fraction_small);
                else if (gk == "vram_small_card")          ok = take_size(l, gv, key, &s->gpu_vram_small_card);
                else return fail(l, key, "unknown key");
                if (!ok) return false;
            }
        } else if (k == "io") {
            if (!expect_object(l, v, k)) return false;
            for (const auto& [ik, iv] : v.items()) {
                const std::string key = "io." + ik;
                if (ik != "queue_depth") return fail(l, key, "unknown key");
                if (!take_int(l, iv, key, 1, 4096, &s->queue_depth)) return false;
            }
        } else if (is_policy_section(k)) {
            if (!apply_policy_section(l, k, v, "", s)) return false;
        } else {
            return fail(l, k, "unknown key");
        }
    }
    return schema || fail(l, "schema", "missing (this build reads 1)");
}

// A model's "overrides" object, or one file's: policy sections only.
bool apply_overrides(const Layer& l, const json& v, const std::string& at, Settings* s) {
    if (!expect_object(l, v, at)) return false;
    for (const auto& [k, sv] : v.items()) {
        if (!is_policy_section(k)) {
            return fail(l, at + "." + k,
                        "not overridable by a model (hardware settings belong to the system config)");
        }
        if (!apply_policy_section(l, k, sv, at + ".", s)) return false;
    }
    return true;
}

bool apply_model(const Layer& l, const json& root, const std::string& gguf_file, Settings* s) {
    bool schema = false;
    const json* file_entry = nullptr;
    std::string file_key;
    for (const auto& [k, v] : root.items()) {
        if (k == "schema") {
            if (!check_schema(l, v)) return false;
            schema = true;
        } else if (k == "arch") {
            if (!v.is_string()) return fail(l, k, "must be a string");
        } else if (k == "kv_defaults") {
            if (!take_strings(l, v, k, &s->kv_defaults)) return false;
        } else if (k == "overrides") {
            if (!apply_overrides(l, v, k, s)) return false;
        } else if (k == "files") {
            if (!expect_object(l, v, k)) return false;
            for (const auto& [fk, fv] : v.items()) {
                if (!expect_object(l, fv, "files." + fk)) return false;
                if (fk == gguf_file) { file_entry = &fv; file_key = fk; }
            }
        } else {
            return fail(l, k, "unknown key");
        }
    }
    if (!schema) return fail(l, "schema", "missing (this build reads 1)");
    // One GGUF file's section narrows the model's own values, so it applies last.
    if (file_entry) {
        const std::string at = "files." + file_key;
        for (const auto& [k, v] : file_entry->items()) {
            if (k == "kv_defaults") {
                if (!take_strings(l, v, at + ".kv_defaults", &s->kv_defaults)) return false;
            } else if (k == "overrides") {
                if (!apply_overrides(l, v, at + ".overrides", s)) return false;
            } else {
                return fail(l, at + "." + k, "unknown key");
            }
        }
    }
    return true;
}

bool read_file(const std::filesystem::path& p, std::string* out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (out->size() >= 3 && static_cast<unsigned char>((*out)[0]) == 0xEF &&
        static_cast<unsigned char>((*out)[1]) == 0xBB && static_cast<unsigned char>((*out)[2]) == 0xBF) {
        out->erase(0, 3);   // a UTF-8 BOM, as editors on Windows like to write
    }
    return true;
}

// An architecture string becomes a file name: nothing that could leave the
// models directory.
bool safe_arch(const std::string& a) {
    if (a.empty()) return false;
    for (char c : a) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '-' || c == '_' || c == '.';
        if (!ok) return false;
    }
    return a.find("..") == std::string::npos;
}

}  // namespace

bool apply_json(const std::string& text, const std::string& origin, Source source,
                bool model_file, const std::string& gguf_file, Settings* s, std::string* err) {
    json root;
    try {
        root = json::parse(text, nullptr, /*allow_exceptions=*/true, /*ignore_comments=*/true);
    } catch (const json::parse_error& e) {
        *err = origin + ": " + e.what();
        return false;
    }
    const Layer l{origin, source, err};
    if (!root.is_object()) return fail(l, "(top level)", "must be an object");
    // All or nothing: a layer that fails halfway (a bad key after good ones, a
    // missing schema noticed at the end) must not leave half its values behind.
    Settings next = *s;
    if (!(model_file ? apply_model(l, root, gguf_file, &next) : apply_system(l, root, &next))) {
        return false;
    }
    *s = std::move(next);
    return true;
}

bool resolve(const Locations& where, const std::string& arch, const std::string& gguf_file,
             const CliSettings& cli, Settings* out, std::string* err) {
    Settings s = builtin_settings();

    auto layer = [&](const std::filesystem::path& p, Source src, bool model) -> bool {
        std::error_code ec;
        if (p.empty() || !std::filesystem::is_regular_file(p, ec)) return true;   // absent
        std::string text;
        const std::string origin = p.string();
        if (!read_file(p, &text)) {
            *err = origin + ": cannot be read";
            return false;
        }
        if (!apply_json(text, origin, src, model, gguf_file, &s, err)) return false;
        s.applied.push_back(origin);
        return true;
    };

    if (!where.shipped.empty() && !layer(where.shipped / "system.json", Source::ShippedSystem, false)) return false;
    if (!where.user.empty() && !layer(where.user / "system.json", Source::UserSystem, false)) return false;
    if (safe_arch(arch)) {
        const std::string name = arch + ".json";
        if (!where.shipped.empty() && !layer(where.shipped / "models" / name, Source::ShippedModel, true)) return false;
        if (!where.user.empty() && !layer(where.user / "models" / name, Source::UserModel, true)) return false;
    }

    if (cli.threads > 0) {
        s.decode_threads.set(cli.threads, Source::Cli, "--threads");
        s.prefill_threads.set(cli.threads, Source::Cli, "--threads");
    }
    if (cli.prefill_chunk > 0) {
        if (cli.prefill_chunk < 16 || cli.prefill_chunk > 8192) {
            *err = "--prefill-chunk must be in [16, 8192]";
            return false;
        }
        s.prefill_chunk_cpu.set(cli.prefill_chunk, Source::Cli, "--prefill-chunk");
        s.prefill_chunk_gpu.set(cli.prefill_chunk, Source::Cli, "--prefill-chunk");
    }
    if (cli.parallel > 0) s.serve_max_parallel.set(cli.parallel, Source::Cli, "--parallel");
    if (cli.vram_cap > 0) s.gpu_vram_cap.set(cli.vram_cap, Source::Cli, "--vram-cap");
    *out = std::move(s);
    return true;
}

namespace {

template <class T>
void line(std::ostream& o, const char* key, const Setting<T>& s, const std::string& value) {
    o << "  " << key;
    for (size_t n = std::string(key).size(); n < 32; ++n) o << ' ';
    o << value;
    for (size_t n = value.size(); n < 12; ++n) o << ' ';
    o << source_name(s.source);
    if (!s.from.empty()) o << " (" << s.from << ")";
    o << "\n";
}

}  // namespace

void describe(const Settings& s, std::ostream& o) {
    o << "settings (each value's source; later layers override earlier ones)\n";
    line(o, "threads.decode", s.decode_threads, std::to_string(s.decode_threads.value));
    line(o, "threads.prefill", s.prefill_threads,
         s.prefill_threads.value ? std::to_string(s.prefill_threads.value) : std::string("all"));
    line(o, "threads.reserved", s.reserved_threads, std::to_string(s.reserved_threads.value));
    line(o, "io.queue_depth", s.queue_depth, std::to_string(s.queue_depth.value));
    line(o, "prefill.chunk_cpu", s.prefill_chunk_cpu, std::to_string(s.prefill_chunk_cpu.value));
    line(o, "prefill.chunk_gpu", s.prefill_chunk_gpu, std::to_string(s.prefill_chunk_gpu.value));
    std::ostringstream f;
    f << s.admit_byte_fraction.value;
    line(o, "cache.admit_byte_fraction", s.admit_byte_fraction, f.str());
    line(o, "serve.max_parallel", s.serve_max_parallel, std::to_string(s.serve_max_parallel.value));
    line(o, "serve.prefill_tokens_per_step", s.serve_prefill_tokens_per_step,
         std::to_string(s.serve_prefill_tokens_per_step.value));
    line(o, "serve.checkpoints_per_slot", s.serve_checkpoints_per_slot,
         std::to_string(s.serve_checkpoints_per_slot.value));
    line(o, "serve.checkpoint_offset", s.serve_checkpoint_offset,
         std::to_string(s.serve_checkpoint_offset.value));
    line(o, "serve.kv_idle_timeout_s", s.serve_kv_idle_timeout_s,
         std::to_string(s.serve_kv_idle_timeout_s.value));
    line(o, "serve.kv_pool_tokens", s.serve_kv_pool_tokens,
         s.serve_kv_pool_tokens.value < 0 ? std::string("auto") : std::to_string(s.serve_kv_pool_tokens.value));
    auto gib = [](uint64_t b) { std::ostringstream t; t.precision(3); t << (b / 1073741824.0) << " GiB"; return t.str(); };
    line(o, "gpu.vram_cap", s.gpu_vram_cap, s.gpu_vram_cap.value ? gib(s.gpu_vram_cap.value) : std::string("auto"));
    line(o, "gpu.vram_auto_fraction", s.gpu_vram_auto_fraction, std::to_string(s.gpu_vram_auto_fraction.value));
    line(o, "gpu.vram_auto_fraction_small", s.gpu_vram_auto_fraction_small,
         std::to_string(s.gpu_vram_auto_fraction_small.value));
    line(o, "gpu.vram_small_card", s.gpu_vram_small_card, gib(s.gpu_vram_small_card.value));
    line(o, "kv_defaults", s.kv_defaults,
         s.kv_defaults.value.empty() ? std::string("none")
                                     : std::to_string(s.kv_defaults.value.size()) + " keys");
    for (const std::string& k : s.kv_defaults.value) o << "    " << k << "\n";
    if (s.applied.empty()) {
        o << "  files applied: none\n";
    } else {
        for (const std::string& p : s.applied) o << "  file applied: " << p << "\n";
    }
}

}  // namespace dray::config
