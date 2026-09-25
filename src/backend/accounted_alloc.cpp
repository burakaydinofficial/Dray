#include "backend/accounted_alloc.h"

namespace dray::backend {

void* AccountedAlloc::alloc(mem::Category cat, uint64_t bytes, uint32_t align) {
    if (bytes == 0) return nullptr;
    if (!ledger_.reserve(cat, bytes)) return nullptr;
    void* p = arena_.alloc(bytes, align);
    if (!p) {
        // Arena disabled or VA exhausted: plain host memory is correct, merely
        // outside the registrable range. Counted, so a Metal-day audit sees it.
        if (arena_.enabled()) arena_.note_fallback();
        p = mem::aligned_alloc_host(static_cast<size_t>(bytes), align);
    }
    if (!p) ledger_.release(cat, bytes);
    return p;
}

void AccountedAlloc::free(mem::Category cat, void* p, uint64_t bytes) {
    if (!p) return;
    free_uncharged(p, bytes);
    ledger_.release(cat, bytes);
}

void* AccountedAlloc::alloc_uncharged(uint64_t bytes, uint32_t align) {
    void* p = arena_.alloc(bytes, align);
    if (!p) p = mem::aligned_alloc_host(static_cast<size_t>(bytes), align);
    return p;
}

void AccountedAlloc::free_uncharged(void* p, uint64_t bytes) {
    if (!p) return;
    if (arena_.contains(p)) arena_.free(p, bytes);
    else                    mem::aligned_free_host(p, bytes);
}

uint64_t AccountedAlloc::cache_used() const {
    return ledger_.used_in(mem::Category::ExpertCache);
}

uint64_t AccountedAlloc::cache_budget() const {
    const uint64_t nc = ledger_.non_cache();
    const uint64_t cap = ledger_.cap();
    return cap > nc ? cap - nc : 0;
}

}  // namespace dray::backend
