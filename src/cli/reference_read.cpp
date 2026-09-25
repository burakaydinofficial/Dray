#include "cli/reference_read.h"

#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "mem/accountant.h"

namespace dray::cli {


// Independent uncached reference read for the diagnostics (verify/stream).
//
// Deliberately NOT the engine's storage backend: a reference that shares the
// code it checks can only confirm that code's own bugs. And deliberately not
// buffered stdio, which it replaced: the old fopen references populated the
// page cache during the exact runs whose premise is that it stays clean
// (Invariant 2) -- a diagnostic that violates the property under test is
// measuring its own footprint. One raw handle per call; performance is
// irrelevant here, independence and cleanliness are not.
bool read_reference_uncached(const std::string& path, uint64_t offset,
                             uint64_t len, uint8_t* dst) {
    const uint64_t kAl = 4096;  // conservative; diagnostics may over-align freely
    const uint64_t lo = (offset / kAl) * kAl;
    const uint64_t head = offset - lo;
    const uint64_t span = ((head + len + kAl - 1) / kAl) * kAl;
    void* buf = dray::mem::aligned_alloc_host(static_cast<size_t>(span), kAl);
    if (!buf) return false;
    bool ok = false;
#if defined(_WIN32)
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        uint64_t done = 0;
        ok = true;
        while (done < span && ok) {
            OVERLAPPED ov{};
            const uint64_t at = lo + done;
            ov.Offset = static_cast<DWORD>(at & 0xFFFFFFFFull);
            ov.OffsetHigh = static_cast<DWORD>(at >> 32);
            DWORD got = 0;
            const DWORD want = static_cast<DWORD>(
                span - done > (64u << 20) ? (64u << 20) : span - done);
            if (!ReadFile(h, static_cast<uint8_t*>(buf) + done, want, &got, &ov) ||
                got == 0) {
                // Short at EOF is legal only if the interior was covered.
                ok = done >= head + len;
                break;
            }
            done += got;
        }
        ok = ok && (done >= head + len || done == span);
        CloseHandle(h);
    }
#else
    int flags = O_RDONLY;
#if defined(__linux__)
    flags |= O_DIRECT;
#endif
    int fd = ::open(path.c_str(), flags);
#if defined(__APPLE__)
    if (fd >= 0) ::fcntl(fd, F_NOCACHE, 1);
#endif
    if (fd >= 0) {
        uint64_t done = 0;
        ok = true;
        while (done < span) {
            const ssize_t r = ::pread(fd, static_cast<uint8_t*>(buf) + done,
                                      static_cast<size_t>(span - done),
                                      static_cast<off_t>(lo + done));
            if (r <= 0) { ok = done >= head + len; break; }
            done += static_cast<uint64_t>(r);
        }
        ::close(fd);
    }
#endif
    if (ok) std::memcpy(dst, static_cast<uint8_t*>(buf) + head, static_cast<size_t>(len));
    dray::mem::aligned_free_host(buf, span);
    return ok;
}

}  // namespace dray::cli
