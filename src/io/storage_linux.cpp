// Linux storage backend: io_uring, in the slot reserved for it since the
// factory's fall-through was written.
//
// Compiled to a real backend only when <liburing.h> is present (CMake defines
// DRAY_HAVE_URING); otherwise make_uring_backend() returns nullptr and the
// portable thread pool keeps the platform correct -- the same graceful shape as
// make_iocp_backend() on Windows.
//
// Design notes, mirroring the decisions storage.h records:
//   * polled completion queue, no callbacks; no cancel();
//   * short reads and errors are EXPECTED: Completion carries status and bytes
//     and callers must check both;
//   * alignment via statx(STATX_DIOALIGN) per file (Invariant 5), 4096 when the
//     kernel or filesystem does not report it;
//   * a mutex around ring operations: the engine drives submit/poll from one
//     thread today, and the lock prices that assumption at nanoseconds instead
//     of leaving it implicit.
//
// Deliberately NOT here yet: registered files and fixed buffers. They are the
// difference between io_uring and a syscall per read at high IOPS, but they
// constrain buffer lifetimes (the ring, regions and slots all move), and the
// first measurement should price the plain ring before optimizing it.

#include "io/storage_threadpool.h"

#if defined(DRAY_HAVE_URING)

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <liburing.h>

#include "io/storage.h"

namespace dray::io {

namespace {

class UringBackend final : public Backend {
public:
    explicit UringBackend(size_t qd) : qd_(qd) {
        // CQ sized 2x SQ so a burst of completions never drops.
        if (io_uring_queue_init(static_cast<unsigned>(qd * 2), &ring_, 0) < 0) {
            ok_ = false;
        }
    }
    ~UringBackend() override {
        if (ok_) io_uring_queue_exit(&ring_);
        for (int fd : fds_) {
            if (fd >= 0) ::close(fd);
        }
    }

    bool valid() const { return ok_; }

    FileId open(const std::filesystem::path& path) override {
        std::lock_guard<std::mutex> lk(mu_);
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) return kInvalidFile;

        Alignment al;
        struct statx sx {};
        bool discovered = false;
        if (statx(fd, "", AT_EMPTY_PATH | AT_STATX_SYNC_AS_STAT, STATX_DIOALIGN, &sx) == 0 &&
            (sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align) {
            al.memory = sx.stx_dio_mem_align ? sx.stx_dio_mem_align : 4096;
            al.offset = sx.stx_dio_offset_align;
            al.length = sx.stx_dio_offset_align;
            discovered = true;
        }
        fds_.push_back(fd);
        aligns_.push_back(al);
        statx_ok_.push_back(discovered);
        return static_cast<FileId>(fds_.size() - 1);
    }

    void close(FileId f) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (f >= 0 && static_cast<size_t>(f) < fds_.size() && fds_[f] >= 0) {
            ::close(fds_[static_cast<size_t>(f)]);
            fds_[static_cast<size_t>(f)] = -1;
        }
    }

    uint64_t size(FileId f) const override {
        std::lock_guard<std::mutex> lk(mu_);
        if (f < 0 || static_cast<size_t>(f) >= fds_.size() || fds_[f] < 0) return 0;
        struct stat st {};
        return ::fstat(fds_[static_cast<size_t>(f)], &st) == 0
                   ? static_cast<uint64_t>(st.st_size) : 0;
    }

    Alignment alignment(FileId f) const override {
        std::lock_guard<std::mutex> lk(mu_);
        if (f < 0 || static_cast<size_t>(f) >= aligns_.size()) return Alignment{};
        return aligns_[static_cast<size_t>(f)];
    }

    size_t submit(const ReadRequest* reqs, size_t n) override {
        // S25: SQEs prepped into the ring are COMMITTED -- io_uring has no
        // un-prep. If the kernel accepts fewer than we prepped, the remainder
        // stays queued in the SQ and goes with the next io_uring_submit, so
        // they must be accounted as accepted HERE; returning the kernel count
        // made the caller re-prep them and every such request read twice,
        // completing into tags the caller had already retired.
        std::lock_guard<std::mutex> lk(mu_);
        size_t took = 0;
        for (; took < n; ++took) {
            if (in_flight_.load(std::memory_order_relaxed) + took >= qd_) break;
            const ReadRequest& r = reqs[took];
            if (r.file < 0 || static_cast<size_t>(r.file) >= fds_.size() ||
                fds_[static_cast<size_t>(r.file)] < 0) {
                break;
            }
            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) break;
            io_uring_prep_read(sqe, fds_[static_cast<size_t>(r.file)], r.dst,
                               r.length, r.offset);
            io_uring_sqe_set_data64(sqe, r.tag);
        }
        if (took) {
            io_uring_submit(&ring_);
            // Whatever the kernel consumed now, the rest is in the SQ and will
            // enter on the next submit call: all `took` are in flight from the
            // caller's perspective, and none may be re-prepped.
            in_flight_.fetch_add(took, std::memory_order_relaxed);
            return took;
        }
        return 0;
    }

    size_t poll(Completion* out, size_t max, size_t min_complete) override {
        if (!out || max == 0) return 0;
        size_t got = 0;
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                io_uring_cqe* cqe = nullptr;
                while (got < max && io_uring_peek_cqe(&ring_, &cqe) == 0 && cqe) {
                    out[got].tag = io_uring_cqe_get_data64(cqe);
                    if (cqe->res < 0) {
                        out[got].status = cqe->res;   // negative errno, per contract
                        out[got].bytes = 0;
                    } else {
                        out[got].status = 0;
                        out[got].bytes = static_cast<uint32_t>(cqe->res);
                    }
                    io_uring_cqe_seen(&ring_, cqe);
                    in_flight_.fetch_sub(1, std::memory_order_relaxed);
                    ++got;
                    cqe = nullptr;
                }
            }
            if (got >= min_complete || got >= max) return got;
            if (in_flight_.load(std::memory_order_relaxed) == 0) return got;
            // Bounded wait, never INFINITE: the same multi-day discipline as the
            // other backends -- re-checking costs wakeups, a missed completion
            // under a race would cost a hang.
            __kernel_timespec ts{0, 200 * 1000 * 1000};
            std::lock_guard<std::mutex> lk(mu_);
            io_uring_cqe* cqe = nullptr;
            io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
            // Harvested on the next loop iteration under the same lock pattern.
        }
    }

    size_t max_in_flight() const override { return qd_; }
    size_t in_flight() const override {
        return in_flight_.load(std::memory_order_relaxed);
    }

    std::string describe() const override {
        Alignment al{};
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!aligns_.empty()) al = aligns_.front();
        }
        // S26: the label must say where the number CAME from (Invariant 6);
        // claiming statx for an assumed default was a small lie in a report.
        const bool any_discovered = [this] {
            for (bool b : statx_ok_) if (b) return true;
            return false;
        }();
        return describe_backend("linux: io_uring + O_DIRECT",
                                aligns_.empty() ? 0 : al.max(),
                                any_discovered ? "statx DIOALIGN" : "assumed 4096 (statx DIOALIGN unavailable)",
                                qd_, "");
    }

private:
    mutable std::mutex   mu_;
    io_uring             ring_{};
    bool                 ok_ = true;
    size_t               qd_;
    std::atomic<size_t>  in_flight_{0};
    std::vector<int>       fds_;
    std::vector<Alignment> aligns_;
    std::vector<bool>      statx_ok_;
};

}  // namespace

std::unique_ptr<Backend> make_uring_backend(size_t queue_depth) {
    // OPT-IN (DRAY_URING=1) until real hardware says otherwise. Measured on
    // the Hyper-V VM, K3 8 GiB, text identical: uring 22.9 s/tok vs thread pool
    // 19.5 -- with FEWER bytes (512B alignment cut widening waste). On
    // virtualized storage 32 blocking preads give the hypervisor 32 independent
    // requests to schedule; one ring funnels through a single virtio path. The
    // usual economics invert, and the default follows the measurement it has,
    // not the reputation io_uring earned on hardware we have not measured.
    const char* v = std::getenv("DRAY_URING");
    if (!(v && v[0] == '1')) {
        return nullptr;   // default: the thread pool, which measured faster here
    }
    auto b = std::make_unique<UringBackend>(queue_depth);
    if (!b->valid()) return nullptr;
    return b;
}

}  // namespace dray::io

#else  // !DRAY_HAVE_URING

namespace dray::io {
std::unique_ptr<Backend> make_uring_backend(size_t) { return nullptr; }
}  // namespace dray::io

#endif
