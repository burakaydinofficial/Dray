// Windows storage backend: FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED reads,
// completions harvested through one IOCP with GetQueuedCompletionStatusEx.
//
// Why this shape:
//
//   * NO_BUFFERING is not an optimisation here, it is Invariant 2. A buffered
//     read of a 594 GB model would hand the page cache the whole file and make
//     the resident-bytes cap unenforceable -- a correctness argument, not a
//     benchmark one. The price is that offset, length AND the destination
//     address must all be sector multiples, which is why Alignment is queried
//     rather than assumed.
//
//   * OVERLAPPED + IOCP is the only Windows mechanism that keeps a real queue
//     outstanding from one thread. The cost model's realized_read_bandwidth is
//     measured at a queue depth, so a backend that cannot hold depth cannot even
//     be measured honestly.
//
//   * GetQueuedCompletionStatusEx (plural) rather than GetQueuedCompletionStatus:
//     at decode we harvest tens of completions per token and one syscall per
//     completion is pure overhead against reads we are already waiting on.
//
// Per-request state lives in a fixed pool for the backend's lifetime. An
// OVERLAPPED is owned by the KERNEL from ReadFile until its packet is dequeued;
// a stack-allocated one (or one freed on an error path) is a write-after-free the
// day a read is slow, which is every day here.

#include "io/storage.h"

#include <windows.h>
#include <winioctl.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include "io/storage_threadpool.h"

namespace dray::io {
namespace {

constexpr size_t kNoIndex = static_cast<size_t>(-1);

// One GetQueuedCompletionStatusEx call drains up to this many packets. 64 keeps
// the stack array at 1.5 KB and comfortably covers a decode step's worth of
// expert reads (k_routed x n_moe_layers is tens, not thousands, per wave).
constexpr size_t kHarvestBatch = 64;

// Last-resort alignment. Named rather than sprinkled, so that the one place we
// are guessing is visible -- Invariant 5 forbids hardcoding this, and describe()
// reports it as "assumed" when we get here.
constexpr uint32_t kAssumedAlign = 4096;

bool is_pow2(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }

bool plausible_sector(uint32_t v) { return is_pow2(v) && v >= 512 && v <= (1u << 20); }

// Win32 error codes are small positive DWORDs; Completion::status is a negative
// platform code. 0 would collide with success, so an unexplained failure becomes
// -1 rather than a silent "ok".
int32_t win_status(DWORD e) {
    return (e == 0) ? -1 : -static_cast<int32_t>(e);
}

// Win32 stops interpreting a path at MAX_PATH unless it carries the \\?\ prefix.
// Model trees are deep (D:\Models\unsloth\<repo>\<split-of-N>.gguf), so this is
// cheap insurance against a failure that would look like "file not found".
std::wstring win_path(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(p, ec);
    if (ec) abs = p;
    abs.make_preferred();
    std::wstring w = abs.native();
    if (w.size() >= 240 && w.size() > 1 && w[1] == L':' && w.compare(0, 4, L"\\\\?\\") != 0) {
        w.insert(0, L"\\\\?\\");
    }
    return w;
}

struct AlignQuery {
    Alignment   align;
    std::string detail;
};

// Fallback path: ask the volume's storage device directly. Works on drive letters
// and on volumes mounted into a folder, and needs no elevation because the volume
// handle is opened with zero access rights.
bool query_alignment_ioctl(const std::filesystem::path& p, uint32_t* logical, uint32_t* physical) {
    const std::wstring full = win_path(p);

    wchar_t mount[MAX_PATH + 1] = {};
    if (!GetVolumePathNameW(full.c_str(), mount, MAX_PATH)) return false;

    wchar_t volume[MAX_PATH + 1] = {};
    if (!GetVolumeNameForVolumeMountPointW(mount, volume, MAX_PATH)) return false;

    // "\\?\Volume{guid}\" names a directory; the device wants it without the
    // trailing separator.
    std::wstring dev = volume;
    if (!dev.empty() && dev.back() == L'\\') dev.pop_back();

    HANDLE vh = CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            OPEN_EXISTING, 0, nullptr);
    if (vh == INVALID_HANDLE_VALUE) return false;

    STORAGE_PROPERTY_QUERY q = {};
    q.PropertyId = StorageAccessAlignmentProperty;
    q.QueryType  = PropertyStandardQuery;

    STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR d = {};
    DWORD returned = 0;
    const BOOL ok = DeviceIoControl(vh, IOCTL_STORAGE_QUERY_PROPERTY, &q,
                                    static_cast<DWORD>(sizeof(q)), &d,
                                    static_cast<DWORD>(sizeof(d)), &returned, nullptr);
    CloseHandle(vh);

    if (!ok || returned < sizeof(d)) return false;
    *logical  = d.BytesPerLogicalSector;
    *physical = d.BytesPerPhysicalSector;
    return true;
}

// Alignment discovery, best source first. The offset/length requirement is the
// LOGICAL sector size (that is what NO_BUFFERING actually enforces); the buffer
// address is over-aligned to the physical sector when we know it, because a
// larger buffer alignment costs nothing and a 512e drive doing 4K-unaligned DMA
// does not.
AlignQuery query_alignment(HANDLE h, const std::filesystem::path& p) {
    AlignQuery q;
    uint32_t logical = 0;
    uint32_t physical = 0;
    const char* source = nullptr;

    FILE_STORAGE_INFO fsi = {};
    if (GetFileInformationByHandleEx(h, FileStorageInfo, &fsi, static_cast<DWORD>(sizeof(fsi)))) {
        logical = fsi.LogicalBytesPerSector;
        physical = fsi.PhysicalBytesPerSectorForPerformance;
        if (!plausible_sector(physical)) {
            physical = fsi.FileSystemEffectivePhysicalBytesPerSectorForAtomicity;
        }
        if (plausible_sector(logical)) source = "via FileStorageInfo";
    }

    if (source == nullptr) {
        uint32_t l = 0;
        uint32_t ph = 0;
        if (query_alignment_ioctl(p, &l, &ph) && plausible_sector(l)) {
            logical = l;
            physical = ph;
            source = "via IOCTL_STORAGE_QUERY_PROPERTY";
        }
    }

    if (source == nullptr) {
        logical = kAssumedAlign;
        physical = kAssumedAlign;
        source = "assumed: the OS answered neither query";
    }

    if (!plausible_sector(physical) || physical < logical) physical = logical;

    q.align.offset = logical;
    q.align.length = logical;
    q.align.memory = physical;

    q.detail = "logical " + std::to_string(logical) + ", physical " + std::to_string(physical) +
               ", " + source;
    return q;
}

class IocpBackend final : public Backend {
public:
    explicit IocpBackend(size_t queue_depth)
        : capacity_(clamp_queue_depth(queue_depth)) {
        iocp_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
        if (iocp_ == nullptr) return;

        pool_.resize(capacity_);
        for (size_t i = 0; i < capacity_; ++i) {
            pool_[i].next = (i + 1 < capacity_) ? (i + 1) : kNoIndex;
        }
        free_head_ = (capacity_ > 0) ? size_t{0} : kNoIndex;
    }

    ~IocpBackend() override {
        // Closing every file cancels and REAPS its outstanding I/O. That is the
        // only thing that makes it safe to let pool_ die: until a packet is
        // dequeued the kernel may still write to its OVERLAPPED.
        for (size_t i = 0; i < files_.size(); ++i) {
            if (files_[i].h != INVALID_HANDLE_VALUE) close(static_cast<FileId>(i));
        }
        if (iocp_ != nullptr) CloseHandle(iocp_);
    }

    IocpBackend(const IocpBackend&) = delete;
    IocpBackend& operator=(const IocpBackend&) = delete;

    bool usable() const { return iocp_ != nullptr && !pool_.empty(); }

    FileId open(const std::filesystem::path& path) override {
        const std::wstring w = win_path(path);

        // FILE_FLAG_RANDOM_ACCESS is deliberately absent: it is a cache hint and
        // there is no cache here.
        HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                               OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) return kInvalidFile;

        LARGE_INTEGER sz = {};
        if (!GetFileSizeEx(h, &sz)) sz.QuadPart = 0;

        AlignQuery aq = query_alignment(h, path);

        std::lock_guard<std::mutex> lk(m_);

        FileId id = kInvalidFile;
        if (!free_files_.empty()) {
            id = free_files_.back();
            free_files_.pop_back();
        } else {
            id = static_cast<FileId>(files_.size());
            files_.emplace_back();
        }

        if (CreateIoCompletionPort(h, iocp_, static_cast<ULONG_PTR>(id), 0) == nullptr) {
            CloseHandle(h);
            free_files_.push_back(id);
            return kInvalidFile;
        }

        FileEntry& fe = files_[static_cast<size_t>(id)];
        fe.h = h;
        fe.size = static_cast<uint64_t>(sz.QuadPart);
        fe.align = aq.align;
        fe.outstanding = 0;

        // With NO_BUFFERING every read reaches the device, so inline completion is
        // rare -- but it is legal, and if we let the kernel post a packet for it we
        // would pay a port round trip for a read that is already done. Asking for
        // SKIP_COMPLETION_PORT_ON_SUCCESS means a ReadFile that returns TRUE is OUR
        // completion to report; if the call fails we must NOT synthesise one,
        // because the packet is still coming. Hence the per-file flag.
        // The flags parameter is a UCHAR, hence the cast.
        fe.skip_on_success =
            SetFileCompletionNotificationModes(
                h, static_cast<UCHAR>(FILE_SKIP_COMPLETION_PORT_ON_SUCCESS |
                                      FILE_SKIP_SET_EVENT_ON_HANDLE)) != FALSE;

        if (aq.align.max() > max_align_seen_) {
            max_align_seen_ = aq.align.max();
            align_detail_ = aq.detail;
        }
        return id;
    }

    void close(FileId id) override {
        HANDLE h = INVALID_HANDLE_VALUE;
        {
            std::lock_guard<std::mutex> lk(m_);
            FileEntry* fe = entry_locked(id);
            if (fe == nullptr) return;
            h = fe->h;
        }

        // storage.h forbids cancel() as an interface operation, but shutdown still
        // has to get the kernel to release these OVERLAPPEDs. Cancelled reads
        // complete promptly with ERROR_OPERATION_ABORTED, and drain() parks their
        // completions so a caller still polling gets a verdict for every tag it
        // submitted rather than a silent disappearance.
        CancelIoEx(h, nullptr);
        drain_file(id);

        std::lock_guard<std::mutex> lk(m_);
        FileEntry* fe = entry_locked(id);
        if (fe == nullptr) return;
        CloseHandle(fe->h);
        fe->h = INVALID_HANDLE_VALUE;
        fe->size = 0;
        fe->outstanding = 0;
        free_files_.push_back(id);
    }

    uint64_t size(FileId id) const override {
        std::lock_guard<std::mutex> lk(m_);
        const FileEntry* fe = entry_locked(id);
        return fe ? fe->size : 0;
    }

    Alignment alignment(FileId id) const override {
        std::lock_guard<std::mutex> lk(m_);
        const FileEntry* fe = entry_locked(id);
        return fe ? fe->align : Alignment{};
    }

    size_t submit(const ReadRequest* reqs, size_t n) override {
        if (reqs == nullptr) return 0;
        std::lock_guard<std::mutex> lk(m_);

        size_t accepted = 0;
        for (; accepted < n; ++accepted) {
            const ReadRequest& r = reqs[accepted];

            // The pool IS the queue-depth gate: an allocated Op is either in the
            // kernel or waiting to be harvested, which is exactly storage.h's
            // definition of in flight.
            const size_t idx = alloc_op_locked();
            if (idx == kNoIndex) break;

            Op& op = pool_[idx];
            op.tag = r.tag;
            op.length = r.length;
            op.file = r.file;
            op.result = Completion{};
            op.result.tag = r.tag;

            FileEntry* fe = entry_locked(r.file);
            if (fe == nullptr) {
                op.result.status = win_status(ERROR_INVALID_HANDLE);
                park_ready_locked(idx, op.result);
                continue;
            }
            op.h = fe->h;

            // FILE_FLAG_NO_BUFFERING is unforgiving: offset, length and buffer
            // address must all be sector multiples or ReadFile rejects the call.
            // Callers guarantee this from alignment(); in release a violation
            // still surfaces as a Completion error rather than a crash, so these
            // are debug tripwires that name the bug, not checks the code needs.
            assert(fe->align.offset != 0 && r.offset % fe->align.offset == 0 &&
                   "unaligned offset under FILE_FLAG_NO_BUFFERING");
            assert(fe->align.length != 0 && r.length % fe->align.length == 0 &&
                   "unaligned length under FILE_FLAG_NO_BUFFERING");
            assert(fe->align.memory != 0 &&
                   reinterpret_cast<uintptr_t>(r.dst) % fe->align.memory == 0 &&
                   "unaligned destination under FILE_FLAG_NO_BUFFERING");

            std::memset(&op.ov, 0, sizeof(OVERLAPPED));
            op.ov.Offset     = static_cast<DWORD>(r.offset & 0xFFFFFFFFull);
            op.ov.OffsetHigh = static_cast<DWORD>(r.offset >> 32);
            op.ov.hEvent     = nullptr;  // unused for an IOCP-associated handle

            // lpNumberOfBytesRead MUST be null on an overlapped handle: on the
            // pending path the kernel would write through it long after the
            // caller's frame is gone.
            const BOOL ok = ReadFile(op.h, r.dst, static_cast<DWORD>(r.length), nullptr, &op.ov);
            if (ok) {
                if (fe->skip_on_success) {
                    DWORD got = 0;
                    if (GetOverlappedResult(op.h, &op.ov, &got, FALSE)) {
                        op.result.status = 0;
                        op.result.bytes = static_cast<uint32_t>(got);
                    } else {
                        op.result.status = win_status(GetLastError());
                    }
                    park_ready_locked(idx, op.result);
                } else {
                    ++fe->outstanding;  // the packet is coming anyway
                }
                continue;
            }

            const DWORD e = GetLastError();
            if (e == ERROR_IO_PENDING) {
                ++fe->outstanding;
                continue;
            }

            // A rejected read is a completion, not an exception (storage.h: short
            // reads and EIO are expected, and the caller must see the tag again).
            op.result.status = win_status(e);
            park_ready_locked(idx, op.result);
        }
        return accepted;
    }

    size_t poll(Completion* out, size_t max, size_t min_complete) override {
        if (out == nullptr || max == 0) return 0;

        size_t n = 0;
        {
            std::lock_guard<std::mutex> lk(m_);
            n = drain_ready_locked(out, max);
        }

        while (n < max) {
            DWORD timeout = 0;
            if (n < min_complete) {
                // Never block on a completion that cannot arrive: if nothing is in
                // the kernel, a caller asking for more than it submitted gets a
                // short return instead of a hang.
                if (kernel_outstanding() == 0) break;

                // Bounded rather than INFINITE. kernel_outstanding() is sampled
                // with the lock released, so a second poller could take the last
                // packet between that check and this wait; INFINITE would then be
                // a permanent hang, which over a multi-day run is the worst
                // failure this file can produce. Re-checking costs 20 wakeups a
                // second while genuinely blocked -- against reads measured in
                // milliseconds to seconds -- and nothing at all on the packet
                // path, since an already-queued packet returns immediately.
                timeout = 50;
            }

            OVERLAPPED_ENTRY ents[kHarvestBatch];
            const ULONG want = static_cast<ULONG>(std::min<size_t>(kHarvestBatch, max - n));
            ULONG removed = 0;
            if (!GetQueuedCompletionStatusEx(iocp_, ents, want, &removed, timeout, FALSE)) {
                // On the opportunistic (timeout 0) sweep this is simply "nothing
                // more is ready"; while still owing the caller completions it means
                // keep waiting.
                if (GetLastError() == WAIT_TIMEOUT && n < min_complete &&
                    kernel_outstanding() > 0) {
                    continue;
                }
                break;  // nothing ready, or a dead port
            }

            std::lock_guard<std::mutex> lk(m_);
            for (ULONG i = 0; i < removed; ++i) {
                Completion c;
                const size_t idx = reap_locked(ents[i], &c);
                if (idx == kNoIndex) continue;  // not one of ours
                // Deliver only if this packet actually retired a live op. A
                // duplicate for an already-freed slot is dropped rather than
                // handed up as a second completion for a tag the caller has
                // settled -- and free_op_locked refuses to splice it in twice.
                if (!free_op_locked(idx)) continue;
                out[n++] = c;
            }
        }
        return n;
    }

    size_t max_in_flight() const override { return capacity_; }

    size_t in_flight() const override {
        std::lock_guard<std::mutex> lk(m_);
        return alloc_count_;
    }

    std::string describe() const override {
        std::lock_guard<std::mutex> lk(m_);
        // A refused duplicate is contained, but it is never normal: it means the
        // port delivered a packet for a slot already retired. Say so rather than
        // swallowing it, because the count is the only evidence the condition
        // exists at all.
        std::string extra;
        if (double_free_refused_ > 0) {
            extra = "REFUSED " + std::to_string(double_free_refused_) +
                    " duplicate completion(s); the port delivered packets for retired ops";
        }
        return describe_backend("windows: FILE_FLAG_NO_BUFFERING|OVERLAPPED + IOCP",
                                max_align_seen_, align_detail_, capacity_, extra);
    }

private:
    struct Op {
        OVERLAPPED ov{};  // MUST be first: the completion path casts back from it
        uint64_t   tag = 0;
        uint32_t   length = 0;  // kept for diagnosis: a short read is only
                                // interpretable next to what was asked for
        FileId     file = kInvalidFile;
        HANDLE     h = INVALID_HANDLE_VALUE;
        Completion result{};
        size_t     next = kNoIndex;  // free list, then ready list
        // A slot is either the caller's or the pool's, never both. Without this
        // the pool had NO WAY to notice a second reap of the same op, and a
        // second free spliced the slot into the free list twice: free_head_
        // eventually points at a cycle, two callers are handed the same staging
        // buffer, and alloc_count_ (unsigned) underflows so kernel_outstanding()
        // reads about 2^64 and every caller spins forever. Two things can deliver
        // a duplicate completion -- FILE_SKIP_COMPLETION_PORT_ON_SUCCESS is a
        // request the kernel may decline (filter drivers, network paths), and
        // close()'s drain dequeues from the same port poll() does. This flag
        // turns both into a loud, contained refusal (2026-08-24 audit).
        bool       allocated = false;
    };

    static_assert(std::is_standard_layout<Op>::value,
                  "Op must be standard-layout to recover it from its OVERLAPPED");
    static_assert(offsetof(Op, ov) == 0,
                  "OVERLAPPED must be Op's first member: reap_locked casts an "
                  "LPOVERLAPPED straight back to Op*");

    struct FileEntry {
        HANDLE      h = INVALID_HANDLE_VALUE;
        uint64_t    size = 0;
        Alignment   align;
        size_t      outstanding = 0;  // issued to the kernel, packet not yet dequeued
        bool        skip_on_success = false;
    };

    FileEntry* entry_locked(FileId id) {
        if (id < 0 || static_cast<size_t>(id) >= files_.size()) return nullptr;
        FileEntry& fe = files_[static_cast<size_t>(id)];
        return fe.h == INVALID_HANDLE_VALUE ? nullptr : &fe;
    }

    const FileEntry* entry_locked(FileId id) const {
        if (id < 0 || static_cast<size_t>(id) >= files_.size()) return nullptr;
        const FileEntry& fe = files_[static_cast<size_t>(id)];
        return fe.h == INVALID_HANDLE_VALUE ? nullptr : &fe;
    }

    size_t alloc_op_locked() {
        if (free_head_ == kNoIndex) return kNoIndex;
        const size_t i = free_head_;
        free_head_ = pool_[i].next;
        pool_[i].next = kNoIndex;
        pool_[i].allocated = true;
        ++alloc_count_;
        return i;
    }

    // Returns false and does nothing if the slot is already free. Refusing a
    // double free is the difference between one dropped completion and a free
    // list with a cycle in it.
    bool free_op_locked(size_t i) {
        if (i >= pool_.size() || !pool_[i].allocated) {
            ++double_free_refused_;
            return false;
        }
        pool_[i].allocated = false;
        pool_[i].next = free_head_;
        free_head_ = i;
        --alloc_count_;
        return true;
    }

    void park_ready_locked(size_t i, const Completion& c) {
        // Parking a slot that is not allocated would link a free-list member into
        // the ready list, which merges the two lists -- the same corruption a
        // double free causes, reached from the drain path instead.
        if (i >= pool_.size() || !pool_[i].allocated) {
            ++double_free_refused_;
            return;
        }
        pool_[i].result = c;
        pool_[i].next = kNoIndex;
        if (ready_tail_ == kNoIndex) {
            ready_head_ = i;
        } else {
            pool_[ready_tail_].next = i;
        }
        ready_tail_ = i;
        ++ready_count_;
    }

    size_t drain_ready_locked(Completion* out, size_t max) {
        size_t n = 0;
        while (n < max && ready_head_ != kNoIndex) {
            const size_t i = ready_head_;
            ready_head_ = pool_[i].next;
            if (ready_head_ == kNoIndex) ready_tail_ = kNoIndex;
            out[n++] = pool_[i].result;
            --ready_count_;
            free_op_locked(i);
        }
        return n;
    }

    // Turns one dequeued packet into a Completion. Returns the pool index so the
    // caller decides whether to hand it to a poller or park it.
    size_t reap_locked(const OVERLAPPED_ENTRY& e, Completion* c) {
        if (e.lpOverlapped == nullptr) return kNoIndex;
        Op* op = reinterpret_cast<Op*>(e.lpOverlapped);  // ov is Op's first member

        c->tag = op->tag;
        // dwNumberOfBytesTransferred comes straight off the IRP and is authoritative
        // for short reads -- which are expected, not exceptional: a NO_BUFFERING read
        // whose sector-rounded length runs past EOF succeeds with fewer bytes.
        c->bytes = static_cast<uint32_t>(e.dwNumberOfBytesTransferred);

        DWORD got = 0;
        if (GetOverlappedResult(op->h, &op->ov, &got, FALSE)) {
            c->status = 0;
        } else {
            // Translating entry.Internal (an NTSTATUS) would need ntdll; this is the
            // documented route to a Win32 code and costs one call per multi-MB read.
            c->status = win_status(GetLastError());
        }

        FileEntry* fe = entry_locked(op->file);
        if (fe != nullptr && fe->outstanding > 0) --fe->outstanding;

        return static_cast<size_t>(op - pool_.data());
    }

    size_t kernel_outstanding() const {
        std::lock_guard<std::mutex> lk(m_);
        return alloc_count_ - ready_count_;
    }

    // Reaps until this file owns nothing in the kernel. Completions are parked,
    // never dropped. A concurrent poll() that steals the packets also decrements
    // outstanding, so this terminates either way.
    void drain_file(FileId id) {
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(m_);
                const FileEntry* fe = entry_locked(id);
                if (fe == nullptr || fe->outstanding == 0) return;
            }

            OVERLAPPED_ENTRY ents[kHarvestBatch];
            ULONG removed = 0;
            if (!GetQueuedCompletionStatusEx(iocp_, ents, static_cast<ULONG>(kHarvestBatch),
                                             &removed, 1000, FALSE)) {
                if (GetLastError() == WAIT_TIMEOUT) continue;  // cancelled I/O still completes
                return;                                        // port is gone; nothing better to do
            }

            std::lock_guard<std::mutex> lk(m_);
            for (ULONG i = 0; i < removed; ++i) {
                Completion c;
                const size_t idx = reap_locked(ents[i], &c);
                if (idx != kNoIndex) park_ready_locked(idx, c);
            }
        }
    }

    HANDLE                 iocp_ = nullptr;
    size_t                 capacity_ = 0;
    mutable std::mutex     m_;
    size_t                 double_free_refused_ = 0;
    std::vector<Op>        pool_;  // never resized after construction: Op* must be stable
    size_t                 free_head_ = kNoIndex;
    size_t                 ready_head_ = kNoIndex;
    size_t                 ready_tail_ = kNoIndex;
    size_t                 alloc_count_ = 0;  // in kernel + ready == in_flight()
    size_t                 ready_count_ = 0;
    std::vector<FileEntry> files_;
    std::vector<FileId>    free_files_;
    uint32_t               max_align_seen_ = 0;
    std::string            align_detail_;
};

}  // namespace

Alignment query_file_alignment(void* native_handle, const std::filesystem::path& path,
                               std::string* detail) {
    const AlignQuery q = query_alignment(static_cast<HANDLE>(native_handle), path);
    if (detail != nullptr) *detail = q.detail;
    return q.align;
}

std::unique_ptr<Backend> make_iocp_backend(size_t queue_depth) {
    auto b = std::make_unique<IocpBackend>(queue_depth);
    if (!b->usable()) return nullptr;  // storage.cpp falls back to the thread pool
    return b;
}

}  // namespace dray::io
