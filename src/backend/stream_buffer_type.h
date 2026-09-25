// The ggml buffer type llama.cpp allocates weights from.
//
// WHERE TO INTERVENE. Not in build_moe_ffn -- that only reaches the experts. The
// right layer is ggml's buffer type, because it reaches every weight uniformly
// and needs no llama.cpp changes: llama_model_params::tensor_buft_overrides is a
// public NULL-terminated {pattern, buft} list, so weights are routed here by name.
//
// These are free functions because ggml's backend interface is a C vtable. They
// hold no logic: each resolves the Streamer::Impl and forwards to it.

#pragma once

#include "backend/stream_buffer.h"

namespace dray::backend {

// Fills in `im.buft`'s interface and device. Called once, from the Streamer's
// constructor, before llama.cpp can see the buffer type.
void init_stream_buffer_type(Streamer::Impl& im);

}  // namespace dray::backend
