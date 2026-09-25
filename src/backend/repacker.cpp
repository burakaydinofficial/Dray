#include "backend/repacker.h"

// The vendored fork's entry points (see repacker.h).
extern "C" {
bool ggml_cpu_repack_data_in_place(const struct ggml_tensor* t, void* data, size_t size);
void* ggml_cpu_repack_traits_for(const struct ggml_tensor* t);
void ggml_cpu_repack_accept_buft(ggml_backend_buffer_type_t buft);
}

namespace dray::backend {

void Repacker::accept(ggml_backend_buffer_type_t buft) const {
    if (enabled_) ggml_cpu_repack_accept_buft(buft);
}

bool Repacker::maybe_repack(ggml_tensor* t) {
    if (!enabled_ || !t || !t->data) return false;
    if (ggml_n_dims(t) != 2) return false;
    ++tried_;
    // The kernels read tensor->extra to find their traits; a buffer that
    // repacks must set it, or dispatch silently declines and the repack is
    // wasted work (measured: 2.8 vs 2.6 s/tok, i.e. nothing, before this).
    void* traits = ggml_cpu_repack_traits_for(t);
    if (!traits) { ++no_traits_; return false; }
    if (!ggml_cpu_repack_data_in_place(t, t->data, ggml_nbytes(t))) return false;
    t->extra = traits;
    ++done_;
    return true;
}

void Repacker::append(std::ostream& o) const {
    if (!enabled_) return;
    o << ", repack " << done_ << "/" << tried_ << " (" << no_traits_ << " no-traits)";
}

}  // namespace dray::backend
