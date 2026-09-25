#include "backend/poison_buffers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dray::backend {

namespace {
constexpr int kPoison = 0xA5;
}  // namespace

PoisonBuffers::~PoisonBuffers() {
    mem_.free_uncharged(emergency_, emergency_bytes_);
    mem_.free_uncharged(sentinel_, sentinel_bytes_);
}

bool PoisonBuffers::create_sentinel(uint64_t bytes, std::string* error) {
    sentinel_bytes_ = bytes;
    sentinel_ = static_cast<uint8_t*>(mem_.alloc_uncharged(bytes, kHostAlign));
    if (!sentinel_) { *error = "cannot allocate sentinel"; return false; }
    std::memset(sentinel_, kPoison, static_cast<size_t>(bytes));
    // Pass-4b: the reserve result was once DISCARDED -- a failed reservation
    // left the sentinel's bytes outside the ledger (Invariant 1). Near-dead
    // path, but honesty is not sized by reachability.
    if (!mem_.ledger().reserve(mem::Category::IoStaging, bytes)) {
        mem_.free_uncharged(sentinel_, bytes);
        sentinel_ = nullptr;
        *error = "cap cannot hold the sentinel buffer";
        return false;
    }
    return true;
}

uint8_t* PoisonBuffers::poison_for(const ggml_tensor* t) {
    const uint64_t need = ggml_nbytes(t);
    if (need <= sentinel_bytes_) return sentinel_;
    if (need > emergency_bytes_) {
        mem_.free_uncharged(emergency_, emergency_bytes_);
        emergency_ = static_cast<uint8_t*>(mem_.alloc_uncharged(need, kHostAlign));
        emergency_bytes_ = emergency_ ? need : 0;
        if (emergency_) std::memset(emergency_, kPoison, static_cast<size_t>(need));
    }
    if (!emergency_) {
        std::fprintf(stderr,
                     "[dray] FATAL: %s failed to materialise and no safe poison "
                     "buffer of %llu bytes could be allocated\n",
                     t->name, static_cast<unsigned long long>(need));
        std::abort();
    }
    return emergency_;
}

}  // namespace dray::backend
