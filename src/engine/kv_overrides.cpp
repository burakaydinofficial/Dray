#include "engine/kv_overrides.h"

#include <cstdio>
#include <cstdlib>

#include "models/profile.h"

namespace dray::engine {

namespace {

// One "key=type:value" string. Returns false on malformed input. A key already
// present is accepted and ignored: the earlier entry wins.
bool parse_one(const std::string& s, std::vector<llama_model_kv_override>* kvo) {
    const size_t eq = s.find('=');
    const size_t co = s.find(':', eq == std::string::npos ? 0 : eq);
    if (eq == std::string::npos || co == std::string::npos || eq == 0 ||
        co <= eq + 1 || co + 1 > s.size()) {
        return false;
    }
    const std::string key = s.substr(0, eq);
    const std::string ty = s.substr(eq + 1, co - eq - 1);
    const std::string val = s.substr(co + 1);
    if (key.size() >= 128 || val.size() >= 128 || val.empty()) return false;
    for (const auto& existing : *kvo) {
        if (key == existing.key) return true;   // earlier entry wins
    }
    llama_model_kv_override o{};
    std::snprintf(o.key, sizeof(o.key), "%s", key.c_str());
    char* end = nullptr;
    if (ty == "int") {
        o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        o.val_i64 = std::strtoll(val.c_str(), &end, 10);
        if (!end || *end != '\0') return false;
    } else if (ty == "float") {
        o.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        o.val_f64 = std::strtod(val.c_str(), &end);
        if (!end || *end != '\0') return false;
    } else if (ty == "bool") {
        o.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (val != "true" && val != "false") return false;
        o.val_bool = (val == "true");
    } else if (ty == "str") {
        o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        std::snprintf(o.val_str, sizeof(o.val_str), "%s", val.c_str());
    } else {
        return false;
    }
    kvo->push_back(o);
    return true;
}

}  // namespace

const llama_model_kv_override* KvOverrides::terminated() {
    terminated_ = entries;
    llama_model_kv_override term{};
    term.key[0] = '\0';
    terminated_.push_back(term);
    return terminated_.data();
}

bool build_kv_overrides(const std::vector<std::string>& cli,
                        const models::Profile* profile,
                        KvOverrides* out, std::string* error) {
    for (const std::string& s : cli) {
        if (!parse_one(s, &out->entries)) {
            *error = "bad --override-kv \"" + s + "\" (want key=int|float|bool|str:value)";
            return false;
        }
    }
    if (profile && profile->kv_defaults) {
        for (const char* const* d = profile->kv_defaults; *d; ++d) {
            if (!parse_one(*d, &out->entries)) {
                *error = std::string("bad profile kv_default \"") + *d + "\"";
                return false;
            }
        }
    }
    return true;
}

}  // namespace dray::engine
