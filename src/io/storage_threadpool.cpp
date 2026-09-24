// The portable floor: N worker threads, each blocking in one positional read at a
// time, feeding one mutex+condvar completion queue.
//
// This backend is permanent on macOS, so it is written to be the real thing:
//
//   * The device sees a queue depth equal to the WORKER COUNT, not to
//     max_in_flight(). A thread blocked in pread is one outstanding request; the
//     rest of the queue is just work waiting for a thread. describe() reports both
//     numbers because the cost model's realized_read_bandwidth is only meaningful
//     next to the depth it was measured at, and reporting 64 when the drive saw 16
//     would poison every projection downstream.
//
//   * Nothing allocates after construction. Both rings are sized to
//     max_in_flight() at construction, which is sufficient because every accepted
//     request is in exactly one of {pending, executing, done} until it is
//     harvested.
//
//   * Reads are uncached (Invariant 2): FILE_FLAG_NO_BUFFERING on Windows,
//     O_DIRECT on Linux, F_NOCACHE on macOS. When the kernel refuses, describe()
//     says so out loud rather than quietly streaming 594 GB through the page
//     cache with a memory cap that then means nothing.

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE  // O_DIRECT
#endif

#include "io/storage_threadpool.h"

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace dray::io {
namespace {

#if defined(_WIN32)
using NativeHandle = HANDLE;
const NativeHandle kNoHandle = INVALID_HANDLE_VALUE;
const char* const kMechanism = "threadpool: blocking ReadFile (FILE_FLAG_NO_BUFFERING)";
#elif defined(__APPLE__)
using NativeHandle = int;
const NativeHandle kNoHandle = -1;
const char* const kMechanism = "threadpool: blocking pread (F_NOCACHE)";
#else
using NativeHandle = int;
const NativeHandle kNoHandle = -1;
const char* const kMechanism = "threadpool: blocking pread (O_DIRECT)";
#endif

// One blocked thread is one outstanding request, so the worker count IS the queue
// depth. The ceiling is a stack-reservation limit, not a throughput opinion: past
// this the threads cost more address space than the depth is worth on any drive
// we can currently measure.
constexpr size_t kMaxWorkers = 64;

#if !defined(_WIN32)
bool is_pow2(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }

// POSIX alignment discovery. st_blksize is the filesystem's preferred I/O size,
// which is >= the device block size on every mount we can construct, so using it
// as the requirement is conservative in the safe direction: over-aligned requests
// are always legal, under-aligned ones fail O_DIRECT with EINVAL.
Alignment posix_alignment(int fd, std::string* detail) {
    Alignment a;
    struct stat st;
    if (fd >= 0 && ::fstat(fd, &st) == 0) {
        const uint32_t blk = static_cast<uint32_t>(st.st_blksize);
        if (is_pow2(blk) && blk >= 512 && blk <= (1u << 20)) {
            a.memory = blk;
            a.offset = blk;
            a.length = blk;
            if (detail != nullptr) *detail = "st_blksize " + std::to_string(blk);
            return a;
        }
    }
    if (detail != nullptr) *detail = "assumed: st_blksize was not usable";
    return a;  // Alignment's own 4096 default, reported as assumed
}
#endif

Completion execute_read(NativeHandle h, const ReadRequest& r, uint32_t align, bool uncached) {
    Completion c;
    c.tag = r.tag;

#if defined(_WIN32)
    (void)align;
    (void)uncached;
    if (h == INVALID_HANDLE_VALUE) {
        c.status = -static_cast<int32_t>(ERROR_INVALID_HANDLE);
        return c;
    }
    // A handle opened WITHOUT FILE_FLAG_OVERLAPPED is a synchronous file object:
    // ReadFile does not return until the transfer is done, so this OVERLAPPED --
    // which carries nothing but the offset -- is safe on the stack. That is the
    // exact opposite of the IOCP backend, where the kernel owns the OVERLAPPED
    // past the call and it must live in a pool.
    OVERLAPPED ov = {};
    ov.Offset     = static_cast<DWORD>(r.offset & 0xFFFFFFFFull);
    ov.OffsetHigh = static_cast<DWORD>(r.offset >> 32);

    DWORD got = 0;  // ReadFile zeroes this before doing any work, so it is
                    // trustworthy on the failure path too
    if (ReadFile(h, r.dst, static_cast<DWORD>(r.length), &got, &ov)) {
        c.bytes = static_cast<uint32_t>(got);
        return c;
    }
    const DWORD e = GetLastError();
    c.bytes = static_cast<uint32_t>(got);
    c.status = (e == 0) ? -1 : -static_cast<int32_t>(e);
    return c;
#else
    if (h < 0) {
        c.status = -EBADF;
        return c;
    }
    size_t done = 0;
    while (done < r.length) {
        const ssize_t got = ::pread(h, static_cast<char*>(r.dst) + done, r.length - done,
                                    static_cast<off_t>(r.offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            c.status = -errno;  // partial progress is still reported in bytes
            break;
        }
        if (got == 0) break;  // EOF: a short read, which the caller must check for
        done += static_cast<size_t>(got);
        // O_DIRECT requires the RESUMED offset and length to stay aligned too. If a
        // partial transfer lands off a sector boundary, stop and report the short
        // read rather than issue a follow-up the kernel would reject with EINVAL --
        // an error there would mask the real event, which is the short read.
        if (uncached && align != 0 && (done % align) != 0) break;
    }
    c.bytes = static_cast<uint32_t>(done);
#if defined(__linux__)
    // Only reachable when O_DIRECT was refused. Invariant 2 is about not RETAINING
    // page cache, so pruning what this read instantiated is the nearest honest
    // thing available; describe() still says the mode is degraded.
    if (!uncached) ::posix_fadvise(h, static_cast<off_t>(r.offset), static_cast<off_t>(r.length),
                                   POSIX_FADV_DONTNEED);
#endif
    return c;
#endif
}

class ThreadPoolBackend final : public Backend {
public:
    explicit ThreadPoolBackend(size_t queue_depth)
        : capacity_(clamp_queue_depth(queue_depth)) {
        pending_.resize(capacity_);
        done_.resize(capacity_);

        const size_t want = std::min<size_t>(capacity_, kMaxWorkers);
        workers_.reserve(want);
        for (size_t i = 0; i < want; ++i) {
            try {
                workers_.emplace_back([this, i] { worker(i); });
            } catch (const std::exception&) {
                break;  // keep the threads that did start; see submit()
            }
        }
    }

    ~ThreadPoolBackend() override {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_work_.notify_all();
        cv_done_.notify_all();  // release any poller blocked on a min_complete
        for (std::thread& t : workers_) {
            if (t.joinable()) t.join();
        }
        // Only now is it certain no thread is inside a read on these handles.
        std::lock_guard<std::mutex> lk(m_);
        for (FileEntry& fe : files_) {
            if (fe.open) close_handles_locked(&fe);
        }
    }

    ThreadPoolBackend(const ThreadPoolBackend&) = delete;
    ThreadPoolBackend& operator=(const ThreadPoolBackend&) = delete;

    FileId open(const std::filesystem::path& path) override {
        FileEntry fe;
        if (!open_native(path, &fe)) return kInvalidFile;

        std::lock_guard<std::mutex> lk(m_);
        FileId id = kInvalidFile;
        if (!free_files_.empty()) {
            id = free_files_.back();
            free_files_.pop_back();
            files_[static_cast<size_t>(id)] = std::move(fe);
        } else {
            id = static_cast<FileId>(files_.size());
            files_.push_back(std::move(fe));
        }

        FileEntry& e = files_[static_cast<size_t>(id)];
        if (e.align.max() > max_align_seen_) {
            max_align_seen_ = e.align.max();
            align_detail_ = e.detail;
        }
        if (!e.uncached) uncached_refused_ = true;
        return id;
    }

    void close(FileId id) override {
        std::unique_lock<std::mutex> lk(m_);
        if (entry_locked(id) == nullptr) return;

        // storage.h deliberately has no cancel(), and a thread blocked in pread
        // genuinely cannot be interrupted -- that is the reason the interface
        // forbids cancellation at all. So closing waits for the reads already
        // running on this file. The cache guarantees it never evicts a slot with a
        // read outstanding, so this is normally a zero-length wait; it exists so a
        // caller who gets that wrong sees a stall instead of a use-after-close.
        cv_done_.wait(lk, [this, id] {
            const FileEntry* e = entry_locked(id);
            return e == nullptr || e->outstanding == 0;
        });

        // Re-resolve: the wait dropped the lock and open() may have grown files_.
        FileEntry* fe = entry_locked(id);
        if (fe == nullptr) return;
        close_handles_locked(fe);
        fe->open = false;
        fe->size = 0;
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

        size_t accepted = 0;
        {
            std::lock_guard<std::mutex> lk(m_);
            while (accepted < n && inflight_ < capacity_) {
                const ReadRequest& r = reqs[accepted];

                FileEntry* fe = entry_locked(r.file);
                if (fe != nullptr) {
                    // Uncached I/O is unforgiving about all three of these, and
                    // every shipping build defines NDEBUG, so asserts here never
                    // executed (swarm S21). A violation is refused BEFORE the OS
                    // sees it, with a named line and an EINVAL completion -- the
                    // generic OS error it replaced did not say which rule broke,
                    // and on some paths a misaligned read can silently succeed
                    // through the cache, which is the Invariant 2 failure mode.
                    const bool mis_off = fe->uncached && fe->align.offset != 0 &&
                                         r.offset % fe->align.offset != 0;
                    const bool mis_len = fe->uncached && fe->align.length != 0 &&
                                         r.length % fe->align.length != 0;
                    const bool mis_mem = fe->uncached && fe->align.memory != 0 &&
                        reinterpret_cast<uintptr_t>(r.dst) % fe->align.memory != 0;
                    if (mis_off || mis_len || mis_mem) {
                        std::fprintf(stderr,
                                     "storage: REFUSED misaligned uncached read (%s%s%s) "
                                     "off=%llu len=%u\n",
                                     mis_off ? "offset " : "", mis_len ? "length " : "",
                                     mis_mem ? "memory" : "",
                                     static_cast<unsigned long long>(r.offset), r.length);
                        Completion c{};
                        c.tag = r.tag;
                        c.status = -22;   // EINVAL, per the negative-errno contract
                        c.bytes = 0;
                        // The done ring's canonical write; inflight balances the
                        // poll-side wait predicate, decremented on harvest like
                        // any completion.
                        done_[(d_head_ + d_count_) % capacity_] = c;
                        ++d_count_;
                        ++inflight_;
                        ++accepted;
                        cv_done_.notify_all();
                        continue;
                    }
                    ++fe->outstanding;
                }
                // An invalid FileId is queued like any other request: it comes back
                // as an error Completion carrying the caller's tag, which is what
                // storage.h promises. Dropping it would strand the caller's slot.

                pending_[(p_head_ + p_count_) % capacity_] = r;
                ++p_count_;
                ++inflight_;
                ++accepted;
            }
        }

        if (accepted == 0) return 0;

        if (workers_.empty()) {
            // Degraded: not one thread could be started. Running the reads inline
            // means submit() blocks, which breaks its contract -- but the only
            // alternative is accepting work that nobody will ever do, and a hang is
            // strictly worse than a slow answer. describe() reports this state.
            run_inline();
        } else {
            for (size_t i = 0; i < accepted; ++i) cv_work_.notify_one();
        }
        return accepted;
    }

    size_t poll(Completion* out, size_t max, size_t min_complete) override {
        if (out == nullptr || max == 0) return 0;

        std::unique_lock<std::mutex> lk(m_);
        if (min_complete > 0) {
            cv_done_.wait(lk, [this, min_complete] {
                // The last clause is what stops a caller asking for more than it
                // submitted from hanging: once everything accepted has completed,
                // a short return is the truthful answer.
                return stop_ || d_count_ >= min_complete || d_count_ >= inflight_;
            });
        }

        size_t n = 0;
        while (n < max && d_count_ > 0) {
            out[n++] = done_[d_head_];
            d_head_ = (d_head_ + 1) % capacity_;
            --d_count_;
            --inflight_;  // harvested: storage.h counts in flight as un-harvested
        }
        return n;
    }

    size_t max_in_flight() const override { return capacity_; }

    size_t in_flight() const override {
        std::lock_guard<std::mutex> lk(m_);
        return inflight_;
    }

    std::string describe() const override {
        std::lock_guard<std::mutex> lk(m_);

        std::string mech = kMechanism;
        if (uncached_refused_) {
            mech += " [DEGRADED: the kernel refused uncached mode on at least one "
                    "file; page cache is being populated]";
        }

        std::string extra;
        if (workers_.empty()) {
            extra = "0 worker threads: reads run inline in submit()";
        } else {
            extra = std::to_string(workers_.size()) + " worker threads (the depth the drive sees)";
        }
        return describe_backend(mech, max_align_seen_, align_detail_, capacity_, extra);
    }

private:
    struct FileEntry {
        bool                      open = false;
        bool                      uncached = false;
        uint64_t                  size = 0;
        Alignment                 align;
        std::string               detail;
        size_t                    outstanding = 0;  // queued or executing, per file
        std::vector<NativeHandle> handles;
    };

    FileEntry* entry_locked(FileId id) {
        if (id < 0 || static_cast<size_t>(id) >= files_.size()) return nullptr;
        FileEntry& fe = files_[static_cast<size_t>(id)];
        return fe.open ? &fe : nullptr;
    }

    const FileEntry* entry_locked(FileId id) const {
        if (id < 0 || static_cast<size_t>(id) >= files_.size()) return nullptr;
        const FileEntry& fe = files_[static_cast<size_t>(id)];
        return fe.open ? &fe : nullptr;
    }

    void close_handles_locked(FileEntry* fe) {
        for (NativeHandle h : fe->handles) {
#if defined(_WIN32)
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#else
            if (h >= 0) ::close(h);
#endif
        }
        fe->handles.clear();
    }

    bool open_native(const std::filesystem::path& path, FileEntry* out) {
#if defined(_WIN32)
        // One handle PER WORKER. A handle opened without FILE_FLAG_OVERLAPPED is a
        // synchronous file object and the kernel serialises I/O on it, so N workers
        // sharing one handle would collapse this backend to queue depth 1 -- the
        // one thing it exists to avoid. Separate handles are cheap; depth is not.
        const size_t want = std::max<size_t>(workers_.size(), 1);
        std::wstring w = path.native();
        {
            std::error_code ec;
            std::filesystem::path abs = std::filesystem::absolute(path, ec);
            if (!ec) {
                abs.make_preferred();
                w = abs.native();
            }
            if (w.size() >= 240 && w.size() > 1 && w[1] == L':' && w.compare(0, 4, L"\\\\?\\") != 0) {
                w.insert(0, L"\\\\?\\");
            }
        }

        for (size_t i = 0; i < want; ++i) {
            HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
            if (h == INVALID_HANDLE_VALUE) break;  // fewer handles = less depth, still correct
            out->handles.push_back(h);
        }
        if (out->handles.empty()) return false;

        LARGE_INTEGER sz = {};
        if (GetFileSizeEx(out->handles[0], &sz)) out->size = static_cast<uint64_t>(sz.QuadPart);
        out->align = query_file_alignment(out->handles[0], path, &out->detail);
        out->uncached = true;
        out->open = true;
        return true;
#else
        int flags = O_RDONLY;
#if defined(O_CLOEXEC)
        flags |= O_CLOEXEC;
#endif
        bool uncached = false;
        int fd = -1;
#if defined(O_DIRECT)
        fd = ::open(path.c_str(), flags | O_DIRECT);
        uncached = (fd >= 0);
#endif
        if (fd < 0) fd = ::open(path.c_str(), flags);
        if (fd < 0) return false;

#if defined(__APPLE__)
        // macOS has no O_DIRECT. F_NOCACHE is the uncached path; F_RDAHEAD off
        // stops the kernel reading past what we asked for, which on a gather
        // workload is pure wasted bandwidth on the drive we are trying to measure.
        if (::fcntl(fd, F_NOCACHE, 1) == 0) uncached = true;
        ::fcntl(fd, F_RDAHEAD, 0);
#endif
        struct stat st;
        if (::fstat(fd, &st) == 0) out->size = static_cast<uint64_t>(st.st_size);
        out->align = posix_alignment(fd, &out->detail);
        out->handles.push_back(fd);  // pread is stateless: one fd serves every worker
        out->uncached = uncached;
        out->open = true;
        return true;
#endif
    }

    // Pops one job, runs it with the lock RELEASED, and records the completion.
    // Called with lk held; returns with lk held.
    void run_one(std::unique_lock<std::mutex>& lk, size_t worker_idx) {
        const ReadRequest r = pending_[p_head_];
        p_head_ = (p_head_ + 1) % capacity_;
        --p_count_;

        NativeHandle h = kNoHandle;
        uint32_t align = 0;
        bool uncached = false;
        if (const FileEntry* fe = entry_locked(r.file)) {
            if (!fe->handles.empty()) h = fe->handles[worker_idx % fe->handles.size()];
            align = fe->align.length;
            uncached = fe->uncached;
        }

        // Safe to leave the file table: close() cannot retire these handles while
        // this request is still counted in FileEntry::outstanding.
        lk.unlock();
        const Completion c = execute_read(h, r, align, uncached);
        lk.lock();

        if (FileEntry* fe = entry_locked(r.file)) {
            if (fe->outstanding > 0) --fe->outstanding;
        }
        done_[(d_head_ + d_count_) % capacity_] = c;
        ++d_count_;
        cv_done_.notify_all();
    }

    void worker(size_t idx) {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_work_.wait(lk, [this] { return stop_ || p_count_ > 0; });
            // Shutdown discards whatever is still queued: the destructor's
            // precondition is that no caller is waiting on a completion.
            if (stop_) return;
            run_one(lk, idx);
        }
    }

    void run_inline() {
        std::unique_lock<std::mutex> lk(m_);
        while (p_count_ > 0) run_one(lk, 0);
    }

    const size_t            capacity_;
    mutable std::mutex      m_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    std::vector<ReadRequest> pending_;  // ring, capacity_ entries, never resized
    size_t                   p_head_ = 0;
    size_t                   p_count_ = 0;

    std::vector<Completion> done_;  // ring, capacity_ entries, never resized
    size_t                  d_head_ = 0;
    size_t                  d_count_ = 0;

    // Accepted but not yet harvested == pending + executing + done. Every accepted
    // request is in exactly one of those, which is why both rings need only
    // capacity_ slots.
    size_t inflight_ = 0;
    bool   stop_ = false;

    std::vector<std::thread> workers_;
    std::vector<FileEntry>   files_;
    std::vector<FileId>      free_files_;
    uint32_t                 max_align_seen_ = 0;
    std::string              align_detail_;
    bool                     uncached_refused_ = false;
};

}  // namespace

std::unique_ptr<Backend> make_threadpool_backend(size_t queue_depth) {
    return std::make_unique<ThreadPoolBackend>(queue_depth);
}

}  // namespace dray::io
