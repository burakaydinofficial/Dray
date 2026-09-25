// A (2026-08-22): rewrite freshly read whole 2D weights into the interleaved
// layout ggml-cpu's optimised matmul kernels want.
//
// MEASURED NEGATIVE: on the streaming path this costs ~20% (3.85 vs 3.2 s/tok on
// a 27B, two runs each), because the transform is paid on EVERY
// materialisation while only 74 of 545 2D tensors in a UD quant even have an
// optimal repack type. OFF by default; DRAY_REPACK=1 reproduces it. The
// machinery stays because install-time repacking (pay once, on disk) is the
// design that would actually win.
//
// The kernels gate on buffer-type IDENTITY with the CPU repack buffer type,
// which a streaming engine can never satisfy, so the vendored fork exposes entry
// points to repack bytes ourselves and declare this buffer type acceptable.

#pragma once

#include <cstdint>
#include <ostream>

#include "ggml-backend.h"
#include "ggml.h"

namespace dray::backend {

class Repacker {
public:
    explicit Repacker(bool enabled) : enabled_(enabled) {}

    bool enabled() const { return enabled_; }
    // Opts the buffer type into the repacked kernels. We then guarantee the
    // bytes behind every matching tensor ARE repacked.
    void accept(ggml_backend_buffer_type_t buft) const;
    // Only 2D weights (GGML_OP_MUL_MAT); the 3D expert slabs mul_mat_id consumes
    // carry our own slot stride and are left alone. Returns true if it repacked
    // -- such a tensor no longer matches the file byte-for-byte.
    bool maybe_repack(ggml_tensor* t);

    // ", repack done/tried (n no-traits)" -- only when enabled; off, these
    // counters are noise.
    void append(std::ostream& o) const;

private:
    bool     enabled_;
    uint64_t tried_ = 0;      // 2D whole tensors reaching maybe_repack
    uint64_t done_ = 0;       // ...that actually got interleaved
    uint64_t no_traits_ = 0;  // ...declined for lack of an optimal repack
};

}  // namespace dray::backend
