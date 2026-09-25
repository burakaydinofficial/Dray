// Not part of streaming itself: this is what lets the engine skip llama.cpp's
// own load. It lives beside the Streamer because the two are always used together.

#pragma once

#include <cstdint>
#include <string>

#include "ggml.h"
#include "gguf.h"
#include "plan/residency.h"

namespace dray::backend {

// Builds one gguf_context describing EVERY tensor across every shard, so the model
// can be created with llama_model_init_from_user instead of being loaded from file.
//
// Why: llama_model_loader::load_all_data returns immediately when `files` is empty,
// calling only the set_tensor_data callback. init_from_user takes that path, so
// nothing is read at load. Otherwise llama.cpp reads all 508 GB into a landing
// buffer we then discard -- minutes of pure waste per start, and worse, the landing
// buffer must be as large as the biggest tensor (1.6 GiB on Qwen3.8), which at a
// 4 GiB cap consumes the entire budget before a single weight is cached.
//
// The returned gguf_context and ggml_context are owned by the caller. `split.*`
// keys are dropped so llama.cpp does not go looking for shards it will never read.
struct MergedMetadata {
    // Qualified: a bare `struct gguf_context*` here would declare a NEW type inside
    // this namespace rather than referring to ggml's.
    ::gguf_context* gguf = nullptr;
    ::ggml_context* ctx  = nullptr;
    int64_t n_tensors = 0;
};

MergedMetadata merge_shard_metadata(const plan::Plan&, std::string* error);
void free_merged_metadata(MergedMetadata&);

}  // namespace dray::backend
