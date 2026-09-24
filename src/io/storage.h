// Storage backend: platform-specific uncached async reads behind one interface.
//
// Design decisions recorded here because they set the threading model for every
// component above this one (see CLAUDE.md, Architecture item 4):
//
//   * COMPLETION MODEL IS A POLLED QUEUE. io_uring and IOCP both hand you a queue
//     natively; the portable thread-pool floor emulates one with a lock-free ring.
//     Callbacks were rejected: they would force the thread-pool backend to invent
//     threading semantics and would move slot refcount decrements onto I/O threads.
//
//   * THERE IS NO cancel(). The portable macOS floor is a pool of blocking pread()
//     and genuinely cannot cancel an in-flight read. Rather than expose an operation
//     one backend can't honour, the interface forbids it -- which makes it the
//     cache's job never to evict a slot with a read outstanding (refcount > 0).
//
//   * SHORT READS AND EIO ARE EXPECTED, not exceptional: at design-center caps this
//     engine reads ~1 PB/day, possibly from an external drive. A partially filled
//     slot handed to ggml_mul_mat_id yields plausible text from garbage weights, so
//     Completion carries both status and bytes and callers MUST check both.
//
//   * ALIGNMENT IS A RUNTIME VALUE (Invariant 5). Never hardcode 4096.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace dray::io {

// Alignment constraints for uncached I/O on a given file. All three are queried
// from the OS at open() time; on Linux via statx(STATX_DIOALIGN), on Windows via
// STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR, on macOS via the mount's devblocksize.
struct Alignment {
    uint32_t memory = 4096;  // required alignment of the destination buffer address
    uint32_t offset = 4096;  // required alignment of the file offset
    uint32_t length = 4096;  // required alignment of the transfer length

    uint32_t max() const {
        uint32_t m = memory > offset ? memory : offset;
        return m > length ? m : length;
    }
};

using FileId = int32_t;
static constexpr FileId kInvalidFile = -1;

struct ReadRequest {
    FileId   file = kInvalidFile;
    uint64_t offset = 0;   // must satisfy Alignment::offset
    uint32_t length = 0;   // must satisfy Alignment::length
    void*    dst = nullptr;  // must satisfy Alignment::memory
    uint64_t tag = 0;      // opaque; returned verbatim in Completion
};

struct Completion {
    uint64_t tag = 0;
    int32_t  status = 0;   // 0 = success; negative = platform error code
    uint32_t bytes = 0;    // bytes actually transferred; < requested is a short read

    bool ok(uint32_t expected) const { return status == 0 && bytes == expected; }
};

// A file opened in uncached mode. GGUF tensor data only -- see Invariant 2.
class Backend {
public:
    virtual ~Backend() = default;

    // Opens in the platform's uncached mode. Returns kInvalidFile on failure.
    virtual FileId open(const std::filesystem::path& path) = 0;
    virtual void   close(FileId) = 0;
    virtual uint64_t size(FileId) const = 0;

    virtual Alignment alignment(FileId) const = 0;

    // Queues up to n requests. Returns how many were accepted; fewer than n means
    // the submission queue is full and the caller should poll() before retrying.
    // Never blocks.
    virtual size_t submit(const ReadRequest* reqs, size_t n) = 0;

    // Harvests completions. Blocks until at least min_complete are available (pass
    // 0 to poll without blocking). Returns how many were written to out.
    virtual size_t poll(Completion* out, size_t max, size_t min_complete) = 0;

    // Maximum requests that may be in flight simultaneously.
    virtual size_t max_in_flight() const = 0;

    // Requests currently submitted but not yet harvested.
    virtual size_t in_flight() const = 0;

    // Human-readable identification for the startup report, e.g.
    // "windows: FILE_FLAG_NO_BUFFERING|OVERLAPPED + IOCP, 4096B align, QD 64".
    virtual std::string describe() const = 0;
};

// Creates the best backend available on this platform. Never returns null: the
// blocking-pread thread pool is always available as a floor.
std::unique_ptr<Backend> make_backend(size_t queue_depth);

}  // namespace dray::io
