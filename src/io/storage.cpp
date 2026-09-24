// Backend selection and the one place describe() strings are formatted.
//
// Selection is deliberately dumb: best available, floor always present. The
// interesting decisions all live in the backends themselves.

#include "io/storage.h"

#include <string>

#include "io/storage_threadpool.h"

namespace dray::io {

std::string describe_backend(const std::string& mechanism,
                             uint32_t           align_max,
                             const std::string& align_detail,
                             size_t             queue_depth,
                             const std::string& extra) {
    std::string s = mechanism;

    s += ", ";
    if (align_max == 0) {
        // Invariant 6: unobtainable is reported unknown, never defaulted. A
        // backend with no file open yet genuinely does not know the alignment,
        // and printing "4096B" here would be a fabricated measurement.
        s += "align unknown (no file open yet)";
    } else {
        s += std::to_string(align_max);
        s += "B align";
        if (!align_detail.empty()) {
            s += " (";
            s += align_detail;
            s += ")";
        }
    }

    s += ", QD ";
    s += std::to_string(queue_depth);

    if (!extra.empty()) {
        s += ", ";
        s += extra;
    }
    return s;
}

std::unique_ptr<Backend> make_backend(size_t queue_depth) {
    const size_t qd = clamp_queue_depth(queue_depth);

#if defined(__linux__)
    // io_uring first on Linux -- the depth the thread pool cannot give from one
    // thread. Falls through identically to IOCP's pattern.
    if (std::unique_ptr<Backend> uring = make_uring_backend(qd)) {
        return uring;
    }
#endif

#if defined(_WIN32)
    // IOCP first: it is the only Windows mechanism that keeps many uncached reads
    // outstanding from one thread. If the port cannot be created we do not fail --
    // the thread pool reaches the same device, just at the depth of its worker
    // count, and a slower engine beats no engine.
    if (std::unique_ptr<Backend> iocp = make_iocp_backend(qd)) {
        return iocp;
    }
#endif

    // macOS lands here permanently, and so does any platform whose native async
    // path is unavailable at runtime. storage.h promises this is never null.
    //
    // NOTE for whoever lands src/io/storage_linux.cpp: an io_uring backend belongs
    // ahead of this line, declared next to make_iocp_backend() in
    // storage_threadpool.h and tried the same way (factory returns nullptr ->
    // fall through). Until then Linux runs on the thread pool, which is correct
    // but not the depth io_uring would give.
    return make_threadpool_backend(qd);
}

}  // namespace dray::io
