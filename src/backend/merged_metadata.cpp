#include "backend/merged_metadata.h"

namespace dray::backend {

MergedMetadata merge_shard_metadata(const plan::Plan& p, std::string* error) {
    MergedMetadata out;
    if (p.shard_paths.empty()) { if (error) *error = "no shards"; return out; }

    out.gguf = gguf_init_empty();
    if (!out.gguf) { if (error) *error = "gguf_init_empty failed"; return out; }

    // One ggml context holding metadata-only tensors for every shard. no_alloc, so
    // these carry shape and type but no data -- which is all llama.cpp needs to
    // build the model; the bytes arrive later, from the streamer.
    struct ggml_init_params ip = {};
    ip.mem_size   = static_cast<size_t>(p.tensors.size() + 64) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) {
        gguf_free(out.gguf); out.gguf = nullptr;
        if (error) *error = "ggml_init failed";
        return out;
    }

    const size_t n_meta = p.n_meta_shards ? p.n_meta_shards : p.shard_paths.size();
    for (size_t s = 0; s < n_meta; ++s) {
        struct ggml_context* shard_ctx = nullptr;
        struct gguf_init_params gp = {};
        gp.no_alloc = true;
        gp.ctx = &shard_ctx;
        struct gguf_context* g = gguf_init_from_file(p.shard_paths[s].c_str(), gp);
        if (!g) {
            if (error) *error = "cannot reopen shard for metadata: " + p.shard_paths[s];
            free_merged_metadata(out);
            return out;
        }
        // Shard 0 carries the model metadata; later shards carry only tensors.
        if (s == 0) gguf_set_kv(out.gguf, g);

        for (struct ggml_tensor* t = ggml_get_first_tensor(shard_ctx); t != nullptr;
             t = ggml_get_next_tensor(shard_ctx, t)) {
            struct ggml_tensor* copy = ggml_new_tensor(out.ctx, t->type, GGML_MAX_DIMS, t->ne);
            if (!copy) {
                if (error) *error = "out of metadata context space";
                gguf_free(g); ggml_free(shard_ctx);
                free_merged_metadata(out);
                return out;
            }
            ggml_set_name(copy, ggml_get_name(t));
            gguf_add_tensor(out.gguf, copy);
            ++out.n_tensors;
        }
        gguf_free(g);
        ggml_free(shard_ctx);
    }

    // Drop the split keys: there is one merged description now, and leaving them
    // makes llama.cpp hunt for shard files it will never open.
    for (const char* k : { "split.no", "split.count", "split.tensors.count" }) {
        const int64_t id = gguf_find_key(out.gguf, k);
        if (id >= 0) gguf_remove_key(out.gguf, k);
    }
    return out;
}

void free_merged_metadata(MergedMetadata& m) {
    if (m.gguf) { gguf_free(m.gguf); m.gguf = nullptr; }
    if (m.ctx)  { ggml_free(m.ctx);  m.ctx  = nullptr; }
    m.n_tensors = 0;
}

}  // namespace dray::backend
