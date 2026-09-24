// The portable storage floor: a pool of worker threads doing blocking positional
// reads, pushing results into a mutex+condvar completion queue.
//
// This is not a token fallback. It is the PERMANENT macOS backend -- there is no
// io_uring there and F_NOCACHE + pread is the only uncached path -- so it honours
// exactly the same contract as the IOCP and io_uring backends:
//
//   * submit() never blocks and never allocates. It takes what fits and returns a
//     short count; the caller polls and retries.
//   * at most max_in_flight() requests are outstanding, counted as
//     accepted-but-not-yet-harvested (the same definition storage.h gives).
//   * short reads and errors surface through Completion, never as exceptions.
//   * poll(min_complete) never waits for a completion that cannot arrive, so a
//     caller that over-asks gets a short return instead of a hang.
//
// The depth the DEVICE sees is the worker count, not max_in_flight(): each worker
// blocks in one read at a time, and queueing beyond the worker count only keeps
// workers fed. describe() reports both, so nobody reads "QD 64" off a 16-thread
// pool and believes the drive saw 64 -- which would silently corrupt the
// calibration this engine's whole cost model is built on.
//
// This header also carries the cross-TU wiring for the io subsystem (the backend
// factories and the shared describe() formatter). storage.h is the PUBLIC
// contract and must not grow implementation declarations, and the subsystem owns
// no other private header.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "io/storage.h"

namespace dray::io {

// Blocking-pread thread pool. Never returns null: if not one worker thread could
// be started it degrades to executing reads inline in submit(), which is slow but
// still correct and, crucially, still terminates. See storage.h: make_backend()
// promises a backend always exists.
std::unique_ptr<Backend> make_threadpool_backend(size_t queue_depth);

// Linux io_uring (storage_linux.cpp), same graceful contract as IOCP: nullptr
// when unavailable (no liburing at build, ring init failure, or
// DRAY_URING=1 opts IN to io_uring; unset keeps the thread pool -- T28: an
// earlier comment stated the opposite polarity). Declared unguarded: the
// stub definition exists on every platform, and a declaration inside the _WIN32
// block was invisible to the only platform that calls it -- found by the Linux
// build, invisible to the Windows one.
std::unique_ptr<Backend> make_uring_backend(size_t queue_depth);

#if defined(_WIN32)
// Defined in storage_win.cpp. Returns nullptr when the completion port could not
// be created -- make_backend()'s cue to fall back to the thread pool rather than
// failing the process.
std::unique_ptr<Backend> make_iocp_backend(size_t queue_depth);

// Also defined in storage_win.cpp. Sector discovery is identical for both Windows
// backends and is not a thing to keep two copies of: GetFileInformationByHandleEx
// (FileStorageInfo), then IOCTL_STORAGE_QUERY_PROPERTY, then an ADMITTED
// assumption. *detail names which source answered, because 4096 from a query and
// 4096 from a guess are different facts and the report must not conflate them.
//
// native_handle is a HANDLE, typed as void* so this header stays free of
// windows.h.
Alignment query_file_alignment(void* native_handle, const std::filesystem::path& path,
                               std::string* detail);
#endif

// Defined in storage.cpp. One place formats every backend's describe() so the
// startup report is comparable across platforms: mechanism, alignment, queue
// depth, in that order.
//
// align_max == 0 means "not queried yet" and is rendered as unknown rather than
// as a default (Invariant 6: anything unobtainable is reported unknown; Invariant
// 5: alignment is a runtime value, never a hardcoded 4096).
std::string describe_backend(const std::string& mechanism,
                             uint32_t           align_max,
                             const std::string& align_detail,
                             size_t             queue_depth,
                             const std::string& extra);

// Queue depth comes from calibration (Invariant 5), never from a default in this
// layer. These bounds exist only so a pathological value cannot make a backend
// allocate a pathological request pool: the pool is O(depth) small structs and
// make_backend() has no Accountant to charge them to.
inline size_t clamp_queue_depth(size_t qd) {
    constexpr size_t kMaxQueueDepth = 4096;
    if (qd < 1) return 1;
    if (qd > kMaxQueueDepth) return kMaxQueueDepth;
    return qd;
}

}  // namespace dray::io
