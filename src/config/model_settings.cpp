#include "config/model_settings.h"

#include <filesystem>

#include "gguf.h"

#include "config/paths.h"

namespace dray::config {

namespace {

std::string gguf_architecture(const std::string& path, std::string* err) {
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx = nullptr;
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        *err = "cannot read GGUF header: " + path;
        return {};
    }
    std::string arch;
    const int64_t k = gguf_find_key(g, "general.architecture");
    if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING) arch = gguf_get_val_str(g, k);
    gguf_free(g);
    if (arch.empty()) *err = "no general.architecture in " + path;
    return arch;
}

}  // namespace

bool resolve_for_model(const std::string& model_path, const std::string& config_dir,
                       const CliSettings& cli, Settings* out, std::string* arch,
                       std::string* err) {
    const std::string a = gguf_architecture(model_path, err);
    if (a.empty()) return false;
    if (arch) *arch = a;
    const std::string gguf = std::filesystem::path(model_path).filename().string();
    return resolve(default_locations(config_dir), a, gguf, cli, out, err);
}

}  // namespace dray::config
