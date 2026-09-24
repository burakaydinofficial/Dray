// Implementation of the resident-byte ledger. accountant.h is the contract; this
// file only decides HOW the numbers are kept, not what they mean.
//
// Two things here are load-bearing:
//
//  * reserve() is all-or-nothing against concurrent callers. Invariant 1 says the
//    cap is TOTAL resident bytes, so the quantity being checked spans every
//    category counter -- it is not a single word. See reserve() for why that
//    forces a serialised check-and-commit rather than one CAS.
//
//  * Slab allocates ONE contiguous block and strides it uniformly. Scattered
//    per-expert allocations cannot be expressed to ggml_mul_mat_id, which reaches
//    every expert from one base pointer with one nb[2].

#include "mem/accountant.h"

#include <atomic>
#include <cstdio>
#include <thread>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#  include <malloc.h>
#  if defined(_MSC_VER)
// K32GetProcessMemoryInfo lives in kernel32 on Win7+, which is already linked;
// psapi.lib forwards to it and costs nothing. Declared here rather than in
// CMakeLists so the RSS probe stays a property of this file.
#    pragma comment(lib, "psapi.lib")
#  endif
#elif defined(__APPLE__)
#  include <stdlib.h>
#  include <mach/mach.h>
#else
#  include <stdlib.h>
#  include <unistd.h>
#  include <fstream>
#endif

namespace dray::mem {

namespace {

constexpr size_t kCategoryCount = static_cast<size_t>(Category::_Count);

// GB here is 2^30, because that is the unit caps are quoted in ("a 4-16 GB cap on
// a 16-32 GB machine") and mixing 10^9 into the same table would make the
// reconciliation against RSS look like a leak.
constexpr double kBytesPerGb = 1024.0 * 1024.0 * 1024.0;

// At or above this, allocations go to VirtualAlloc so freeing returns the pages to
// the OS instead of a CRT free list. One 64 KB allocation granularity: below it the
// rounding waste would outweigh the retention it avoids.
constexpr size_t kVirtualAllocFloor = 64 * 1024;

double gb(size_t bytes) {
    return static_cast<double>(bytes) / kBytesPerGb;
}

// Invariant 6: a percentage of an unknown whole is not 0.0, it is unknown.
void fmt_pct(char* dst, size_t n, size_t part, size_t whole) {
    if (whole == 0) {
        std::snprintf(dst, n, "%7s", "n/a");
        return;
    }
    std::snprintf(dst, n, "%6.1f%%",
                  100.0 * static_cast<double>(part) / static_cast<double>(whole));
}

// The header fixes the ledger's private state to one counter per category with no
// aggregate total, so "does this fit under the cap" reads kCategoryCount separate
// words. A CAS on the caller's own counter cannot see a concurrent reservation in
// a DIFFERENT category, and two such threads would each observe a fitting total
// and both commit -- overshooting the cap, which Invariant 1 forbids outright.
// The check-and-commit is therefore serialised by this CAS-acquired token.
//
// Concerns worth recording rather than hiding:
//   - The token is process-wide, not per-Accountant, because the header leaves no
//     per-instance storage for it. There is one Accountant per process by design,
//     and reserve() runs at load and at slab construction, never on the decode
//     path, so contention is not a cost anyone can measure.
//   - A single std::atomic<size_t> total_ member would make this one CAS loop and
//     lock-free. That is a header change, so it is noted, not made.
std::atomic<bool> g_ledger_token{false};

class LedgerToken {
public:
    LedgerToken() {
        bool expected = false;
        while (!g_ledger_token.compare_exchange_weak(
                   expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
            expected = false;
            // The critical section is a handful of relaxed loads and one store, so
            // the holder is never descheduled for long -- but yielding keeps a
            // spin from stealing a core from the decode threads if it ever is.
            std::this_thread::yield();
        }
    }
    ~LedgerToken() { g_ledger_token.store(false, std::memory_order_release); }

    LedgerToken(const LedgerToken&) = delete;
    LedgerToken& operator=(const LedgerToken&) = delete;
};

// posix_memalign and _aligned_malloc both demand a power-of-two alignment, and
// posix_memalign additionally demands a multiple of sizeof(void*). Alignment is a
// runtime value discovered from the drive (Invariant 5), so it is normalised here
// instead of trusted.
uint32_t normalize_align(uint32_t align) {
    uint32_t a = align;
    if (a < sizeof(void*)) {
        a = static_cast<uint32_t>(sizeof(void*));
    }
    if (a > 0x80000000u) {
        return 0x80000000u;
    }
    uint32_t p = 1;
    while (p < a) {
        p <<= 1;
    }
    return p;
}

}  // namespace

const char* category_name(Category c) {
    switch (c) {
        case Category::RouterGates:       return "router gates";
        case Category::NormsAndBiases:    return "norms and biases";
        case Category::KvCache:           return "KV cache";
        case Category::RecurrentState:    return "recurrent state";
        case Category::PrefillActivation: return "prefill activation";
        case Category::ComputeScratch:    return "compute scratch";
        case Category::IoStaging:         return "I/O staging";
        case Category::ExpertCache:       return "expert cache";
        case Category::Misc:              return "misc";
        case Category::_Count:            break;
    }
    return "unknown";
}

size_t Breakdown::total() const {
    size_t sum = 0;
    for (size_t i = 0; i < kCategoryCount; ++i) {
        sum += bytes[i];
    }
    return sum;
}

Accountant::Accountant(size_t cap_bytes) : cap_(cap_bytes) {
    for (size_t i = 0; i < kCategoryCount; ++i) {
        counts_[i].store(0, std::memory_order_relaxed);
    }
}

bool Accountant::reserve(Category c, size_t bytes) {
    const size_t idx = static_cast<size_t>(c);
    if (idx >= kCategoryCount) {
        return false;
    }
    if (bytes == 0) {
        return true;  // a zero reservation always fits, including when cap_ is 0
    }

    const LedgerToken token;

    // S18: unreserved_ IS resident process memory (the RSS reconcile charges
    // rss-minus-explained here every token). A gate that ignored it approved
    // reservations past the cap exactly while the process was heaviest -- the
    // cap binding on an estimate again. Refusal here means the caller evicts
    // or waits, which is the designed response to genuine pressure.
    size_t cur = unreserved_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < kCategoryCount; ++i) {
        cur += counts_[i].load(std::memory_order_relaxed);
    }

    // Subtraction rather than cur + bytes: the caller's byte count comes from a
    // GGUF tensor table and a corrupt table must refuse, not wrap.
    if (cur > cap_ || bytes > cap_ - cur) {
        return false;
    }

    counts_[idx].fetch_add(bytes, std::memory_order_acq_rel);
    return true;
}

void Accountant::release(Category c, size_t bytes) {
    const size_t idx = static_cast<size_t>(c);
    if (idx >= kCategoryCount || bytes == 0) {
        return;
    }

    // Releasing more than was reserved is a bug in the caller, but letting the
    // counter wrap to ~2^64 would turn that bug into a ledger that reports
    // exabytes and an engine that thinks it has infinite room. Clamp instead, and
    // do it with a CAS loop so a concurrent release cannot lose an update.
    //
    // Releases stay outside the reserve token deliberately: they only ever lower
    // the total, so a reserver racing one merely sums a stale-high figure and is
    // conservative. That direction can never overshoot the cap.
    size_t cur = counts_[idx].load(std::memory_order_relaxed);
    for (;;) {
        const size_t next = (cur >= bytes) ? (cur - bytes) : 0;
        if (counts_[idx].compare_exchange_weak(cur, next, std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
            return;
        }
    }
}

void Accountant::charge_unreserved(size_t bytes) {
    unreserved_.store(bytes, std::memory_order_relaxed);
}

size_t Accountant::unreserved() const {
    return unreserved_.load(std::memory_order_relaxed);
}

size_t Accountant::used_in(Category c) const {
    const size_t idx = static_cast<size_t>(c);
    if (idx >= kCategoryCount) return 0;
    return counts_[idx].load(std::memory_order_relaxed);
}

size_t Accountant::used() const {
    // Relaxed: this is a snapshot of a quantity several threads may be moving. It
    // is exact whenever it matters (startup, and the reconciliation timer).
    size_t sum = unreserved_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < kCategoryCount; ++i) {
        sum += counts_[i].load(std::memory_order_relaxed);
    }
    return sum;
}

size_t Accountant::available() const {
    const size_t u = used();
    return (u >= cap_) ? 0 : (cap_ - u);
}

Breakdown Accountant::breakdown() const {
    Breakdown b;
    for (size_t i = 0; i < kCategoryCount; ++i) {
        b.bytes[i] = counts_[i].load(std::memory_order_relaxed);
    }
    return b;
}

size_t Accountant::non_cache() const {
    // Unreserved bytes are by definition not the cache, so they reduce the cache
    // budget -- which is the whole mechanism by which the cap binds on reality.
    size_t sum = unreserved_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < kCategoryCount; ++i) {
        if (i == static_cast<size_t>(Category::ExpertCache)) {
            continue;
        }
        sum += counts_[i].load(std::memory_order_relaxed);
    }
    return sum;
}

std::string Accountant::report() const {
    // One snapshot for the whole table: re-reading the counters per line would let
    // a concurrent reservation produce a report whose rows do not sum to its total,
    // which is exactly the kind of number that costs an hour to disbelieve.
    const Breakdown b = breakdown();
    const size_t reserved = b.total();
    const size_t cache = b.bytes[static_cast<size_t>(Category::ExpertCache)];
    const size_t floor_bytes = reserved - cache;  // not named `floor`: C4459 vs ::floor
    // T22: headroom derives from used() (which includes the unreserved RSS
    // charge), not from category sums alone -- the old row printed "unreserved
    // 1.78 GB" while available() was 0 and every reservation was refused.
    const size_t u = used();
    const size_t free_bytes = (u >= cap_) ? 0 : (cap_ - u);

    char line[256];
    char pct[16];
    std::string out;

    std::snprintf(line, sizeof(line),
                  "resident memory ledger -- cap %.2f GB (total, not per-category)\n",
                  gb(cap_));
    out += line;

    for (size_t i = 0; i < kCategoryCount; ++i) {
        if (b.bytes[i] == 0) {
            continue;
        }
        fmt_pct(pct, sizeof(pct), b.bytes[i], cap_);
        std::snprintf(line, sizeof(line), "  %-20s %8.2f GB  %s\n",
                      category_name(static_cast<Category>(i)), gb(b.bytes[i]), pct);
        out += line;
    }

    out += "  -------------------------------------------\n";

    fmt_pct(pct, sizeof(pct), reserved, cap_);
    std::snprintf(line, sizeof(line), "  %-20s %8.2f GB  %s\n", "reserved", gb(reserved), pct);
    out += line;

    fmt_pct(pct, sizeof(pct), floor_bytes, cap_);
    std::snprintf(line, sizeof(line), "    %-18s %8.2f GB  %s\n", "floor + scratch",
                  gb(floor_bytes), pct);
    out += line;

    fmt_pct(pct, sizeof(pct), cache, cap_);
    std::snprintf(line, sizeof(line), "    %-18s %8.2f GB  %s\n", "expert cache",
                  gb(cache), pct);
    out += line;

    fmt_pct(pct, sizeof(pct), free_bytes, cap_);
    std::snprintf(line, sizeof(line), "  %-20s %8.2f GB  %s\n", "unreserved",
                  gb(free_bytes), pct);
    out += line;

    // RSS is the reconciliation the ledger exists to be checked against. Reported
    // as "unknown" when the OS will not say, never as 0 (Invariant 6).
    const size_t rss = process_rss();
    if (rss == 0) {
        std::snprintf(line, sizeof(line), "  %-20s %11s\n", "process RSS", "unknown");
    } else {
        std::snprintf(line, sizeof(line), "  %-20s %8.2f GB  (ledger accounts %.1f%% of it)\n",
                      "process RSS", gb(rss),
                      100.0 * static_cast<double>(reserved) / static_cast<double>(rss));
    }
    out += line;

    return out;
}

size_t Accountant::process_committed() {
#if defined(_WIN32)
    // PagefileUsage is the private commit charge. The working set omits pages the
    // allocator holds on its free lists, which on a streaming workload is where the
    // difference lives -- see the header for the measured gap.
    PROCESS_MEMORY_COUNTERS_EX pmc;
    pmc.cb = static_cast<DWORD>(sizeof(pmc));
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                              static_cast<DWORD>(sizeof(pmc)))) {
        return 0;
    }
    return static_cast<size_t>(pmc.PrivateUsage);
#else
    return 0;   // unknown here; the report says so rather than inventing a figure
#endif
}

size_t Accountant::process_rss() {
#if defined(_WIN32)
    // WorkingSetSize is the resident figure; PagefileUsage would count committed
    // pages the engine has paged out, which is not what the cap is about.
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = static_cast<DWORD>(sizeof(pmc));
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, static_cast<DWORD>(sizeof(pmc)))) {
        return 0;
    }
    return static_cast<size_t>(pmc.WorkingSetSize);
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(info.resident_size);
#else
    // /proc/self/statm field 2 is resident pages. statm is cheap and stable;
    // /proc/self/status VmRSS costs a string scan for the same number.
    const long page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return 0;
    }
    std::ifstream statm("/proc/self/statm");
    unsigned long long total_pages = 0;
    unsigned long long rss_pages = 0;
    if (!(statm >> total_pages >> rss_pages)) {
        return 0;
    }
    return static_cast<size_t>(rss_pages) * static_cast<size_t>(page);
#endif
}

// --- Slab --------------------------------------------------------------------

Slab::Slab(Accountant& acct, Category cat, size_t slot_bytes, size_t n_slots, uint32_t align)
    : acct_(acct), cat_(cat) {
    if (slot_bytes == 0 || n_slots == 0) {
        return;  // valid() == false, nothing reserved
    }
    if (n_slots > SIZE_MAX / slot_bytes) {
        return;  // the multiply would wrap; refuse rather than under-reserve
    }

    const size_t total = slot_bytes * n_slots;

    // Reserve BEFORE allocating. The reverse order would put bytes in the process
    // that the cap never approved, which is the whole failure mode Invariant 1
    // exists to prevent.
    if (!acct_.reserve(cat_, total)) {
        return;  // graceful: the caller shrinks its budget and retries
    }

    base_ = aligned_alloc_host(total, align);
    if (base_ == nullptr) {
        acct_.release(cat_, total);
        return;
    }

    // Only published once the allocation succeeded, so bytes() is 0 on failure and
    // the destructor's release is a correct no-op.
    slot_bytes_ = slot_bytes;
    n_slots_ = n_slots;

    // NOTE (concern, not a deviation): the stride is exactly slot_bytes, because
    // the header pins nb[2] = slot_bytes. Only the base is aligned to `align`, so
    // slot i is aligned for uncached I/O only when slot_bytes is a multiple of
    // `align`. That holds for the measured M3 UD-Q2_K_XL per-layer expert sizes --
    // 18,137,088 / 20,938,752 / 24,477,696 are all exact multiples of 4096 -- but
    // it is not guaranteed by anything. A caller whose slot_bytes is not a multiple
    // of the drive's memory alignment must round slot_bytes up itself before
    // constructing the Slab (which keeps the stride uniform and honest), or route
    // that layer through IoStaging. Padding silently here would change nb[2] out
    // from under the ggml tensor the caller builds from slot_bytes().
}

Slab::~Slab() {
    if (base_ != nullptr) {
        aligned_free_host(base_, slot_bytes_ * n_slots_);
        base_ = nullptr;
    }
    // bytes() is 0 unless the allocation succeeded, so this releases exactly what
    // was reserved.
    acct_.release(cat_, bytes());
    slot_bytes_ = 0;
    n_slots_ = 0;
}

const void* Slab::slot(size_t i) const {
    if (base_ == nullptr || i >= n_slots_) {
        return nullptr;
    }
    return static_cast<const char*>(base_) + i * slot_bytes_;
}

void* Slab::slot(size_t i) {
    return const_cast<void*>(static_cast<const Slab*>(this)->slot(i));
}

// --- host allocation ---------------------------------------------------------

void* aligned_alloc_host(size_t bytes, uint32_t align) {
    if (bytes == 0) {
        return nullptr;
    }
    const size_t a = normalize_align(align);
#if defined(_WIN32)
    // VirtualAlloc, NOT _aligned_malloc, for anything large.
    //
    // The streaming path allocates and frees continuously, and the CRT heap keeps
    // freed blocks on its free lists rather than returning them. Measured on
    // Qwen3.8 at a 12 GiB cap: working set 11.97 GB but private commit 13.59 GB --
    // 1.6 GB the process had taken from the system and would not give back, which
    // put it over a cap that the working-set figure said was being respected.
    //
    // MEM_RELEASE hands the pages back at once, so commit tracks actual use. The
    // allocation granularity is 64 KB and the base is always 64 KB aligned, which
    // satisfies every alignment this engine discovers (device 4096, ggml 64) with
    // no extra work. Below that granularity the waste would dominate, so small
    // requests still go to the heap -- and the sizes never mix, because the
    // threshold is a compile-time constant checked identically on free.
    if (bytes >= kVirtualAllocFloor) {
        void* p = ::VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        // `p || a <= 65536` was wrong in both directions (2026-08-24 audit): a
        // failed VirtualAlloc at the common alignment returned nullptr instead of
        // reaching the heap fallback this comment describes, and a successful one
        // was returned for a request needing MORE than 64 KB alignment, handing
        // uncached I/O an under-aligned buffer. Take the region only when it
        // actually satisfies the request; otherwise release it and fall through.
        if (p) {
            if (a <= 65536) return p;
            ::VirtualFree(p, 0, MEM_RELEASE);
        }
    }
    // std::aligned_alloc does not exist in the MSVC runtime; _aligned_malloc is
    // the only option and its blocks MUST be freed with _aligned_free.
    return _aligned_malloc(bytes, a);
#else
    void* p = nullptr;
    if (::posix_memalign(&p, a, bytes) != 0) {
        return nullptr;
    }
    // posix_memalign rather than std::aligned_alloc: the latter requires the size
    // to be a multiple of the alignment, and slot sizes come from the GGUF tensor
    // table (Invariant 3) with no such guarantee.
    return p;
#endif
}

void aligned_free_host(void* p, size_t bytes) {
    if (p == nullptr) {
        return;
    }
#if defined(_WIN32)
    // Mirror of the split in aligned_alloc_host. MEM_RELEASE requires the size
    // argument to be 0 and frees the whole reservation.
    if (bytes >= kVirtualAllocFloor) {
        if (::VirtualFree(p, 0, MEM_RELEASE)) return;
    }
    _aligned_free(p);
#else
    (void)bytes;
    ::free(p);
#endif
}

}  // namespace dray::mem
