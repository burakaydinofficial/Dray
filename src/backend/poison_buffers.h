// What a weight points at when it is NOT materialised.
//
// The SENTINEL is the address every streamed tensor holds until a node needs it,
// and again after eviction: ggml asserts `tensor->data != NULL` long before any
// of our code runs, so "unbacked" still has to be a real, readable address.
// Poisoned (0xA5) rather than zeroed: a weight computed against without being
// materialised must produce obviously broken output, not a plausible zero
// tensor. On the slow-load path it is also llama.cpp's landing zone.
//
// The EMERGENCY buffer covers a failed tensor larger than the sentinel. The
// fast-load sentinel is slot-sized (~22 MB on K3), and pointing a 0.96 GB plain
// MUL_MAT at it read 40x past the end (SIGSEGV in ggml_vec_dot on Linux, where the
// smaller budget made the failure reachable). Raw memory, outside the cap ledger:
// a dying run may briefly overshoot the cap, which is the honest price of never
// corrupting memory.
//
// Both come from the arena when there is one: failure paths READ them in
// compute, so on a GPU-mapped build they must resolve inside the one registered
// range.

#pragma once

#include <cstdint>
#include <string>

#include "backend/accounted_alloc.h"
#include "ggml.h"

namespace dray::backend {

class PoisonBuffers {
public:
    explicit PoisonBuffers(AccountedAlloc& mem) : mem_(mem) {}
    ~PoisonBuffers();
    PoisonBuffers(const PoisonBuffers&) = delete;
    PoisonBuffers& operator=(const PoisonBuffers&) = delete;

    // Allocates and poisons the sentinel, then charges it to IoStaging. False,
    // with `error` set, when memory or the cap refuses it; nothing is held then.
    bool create_sentinel(uint64_t bytes, std::string* error);

    uint8_t* sentinel() const { return sentinel_; }
    uint64_t sentinel_bytes() const { return sentinel_bytes_; }

    // A poison buffer guaranteed to cover `t`'s FULL extent: the sentinel when
    // it is big enough, else a lazily grown emergency buffer. If even that cannot
    // be allocated there is no safe way to let the node run, and a clean fatal
    // beats undefined behaviour: this aborts.
    uint8_t* poison_for(const ggml_tensor* t);

private:
    AccountedAlloc& mem_;
    uint8_t* sentinel_ = nullptr;
    uint64_t sentinel_bytes_ = 0;
    uint8_t* emergency_ = nullptr;
    uint64_t emergency_bytes_ = 0;
};

}  // namespace dray::backend
