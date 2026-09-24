// See pool_arena.h. Platform notes:
//   Windows: VirtualAlloc MEM_RESERVE once, MEM_COMMIT / MEM_DECOMMIT per
//     block. Decommitted pages leave both RSS and commit charge immediately.
//   POSIX:   mmap PROT_NONE once, mprotect to enable per block; release with
//     madvise (MADV_FREE on macOS, MADV_DONTNEED elsewhere) + mprotect
//     PROT_NONE, which drops the pages from RSS. macOS overcommits by design,
//     so the PROT_NONE reservation costs nothing until touched.

#include "mem/pool_arena.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#endif

namespace dray::mem {

namespace {
uint64_t round_up(uint64_t n, uint64_t g) { return (n + g - 1) / g * g; }
}  // namespace

PoolArena::~PoolArena() {
#if defined(_WIN32)
    if (base_) VirtualFree(base_, 0, MEM_RELEASE);
#else
    if (base_) munmap(base_, reserved_);
#endif
}

bool PoolArena::init(uint64_t reserve_bytes) {
    if (base_ || reserve_bytes == 0) return false;
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    page_ = si.dwPageSize ? si.dwPageSize : 4096;
    reserved_ = round_up(reserve_bytes, page_);
    base_ = static_cast<uint8_t*>(
        VirtualAlloc(nullptr, reserved_, MEM_RESERVE, PAGE_NOACCESS));
#else
    long ps = sysconf(_SC_PAGESIZE);
    page_ = ps > 0 ? static_cast<uint64_t>(ps) : 4096;
    reserved_ = round_up(reserve_bytes, page_);
    // GPU-mapped runs (DRAY_METAL=1) keep the whole span PROT_READ|WRITE:
    // Metal cannot no-copy-wrap PROT_NONE pages. Overcommit makes untouched RW
    // pages free, and free() still madvises dropped blocks out of RSS, so the
    // cap reconcile keeps meaning what it says either way.
    const char* mv = std::getenv("DRAY_METAL");
    always_rw_ = mv && mv[0] == '1';
    void* m = mmap(nullptr, reserved_, always_rw_ ? (PROT_READ | PROT_WRITE) : PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    base_ = (m == MAP_FAILED) ? nullptr : static_cast<uint8_t*>(m);
#endif
    if (!base_) reserved_ = 0;
    return base_ != nullptr;
}

void* PoolArena::alloc(uint64_t bytes, uint32_t align) {
    if (!base_ || bytes == 0) return nullptr;
    // Page granularity gives every block page alignment. A stricter align
    // (never seen; Invariant 5 distrusts the drive population anyway) is
    // REFUSED rather than honored asymmetrically -- free() recomputes spans
    // from the page, so a gran-rounded block would decommit the wrong size.
    // The caller's fallback allocator honors arbitrary align natively (R13).
    if (align > page_) return nullptr;
    const uint64_t span = round_up(bytes, page_);

    uint64_t off;
    auto it = free_.find(span);
    if (it != free_.end() && !it->second.empty()) {
        off = it->second.back();
        it->second.pop_back();
        if (it->second.empty()) free_.erase(it);
    } else {
        if (bump_ + span > reserved_) return nullptr;   // VA exhausted: fall back
        off = bump_;
        bump_ += span;
    }

#if defined(_WIN32)
    if (!VirtualAlloc(base_ + off, span, MEM_COMMIT, PAGE_READWRITE)) {
        // Commit refused (system commit limit): surrender the offset to the
        // free list and let the caller fall back.
        free_[span].push_back(off);
        return nullptr;
    }
#else
    if (!always_rw_ && mprotect(base_ + off, span, PROT_READ | PROT_WRITE) != 0) {
        free_[span].push_back(off);
        return nullptr;
    }
#endif
    committed_ += span;
    return base_ + off;
}

void PoolArena::free(void* p, uint64_t bytes) {
    if (!contains(p) || bytes == 0) return;
    const uint64_t span = round_up(bytes, page_);
    const uint64_t off = static_cast<uint8_t*>(p) - base_;
#if defined(_WIN32)
    VirtualFree(p, span, MEM_DECOMMIT);
#else
#if defined(__APPLE__)
    madvise(p, span, MADV_FREE);
#else
    madvise(p, span, MADV_DONTNEED);
#endif
    if (!always_rw_) mprotect(p, span, PROT_NONE);
#endif
    committed_ -= span;
    free_[span].push_back(off);
}

}  // namespace dray::mem
