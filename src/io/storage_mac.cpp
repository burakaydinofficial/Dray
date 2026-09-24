// macOS storage backend.
//
// The portable thread pool IS the permanent macOS backend, by design rather
// than by omission -- storage_threadpool.cpp was written against exactly this
// platform's constraints (F_NOCACHE instead of O_DIRECT, blocking pread with a
// worker pool because macOS offers no io_uring/IOCP analogue worth the
// complexity at this project's size; see storage.h's no-cancel decision, which
// exists precisely because this floor cannot cancel).
//
// This file exists because the build lists one storage file per platform, and
// so that anything genuinely Apple-specific that ever earns code -- an
// F_RDAHEAD policy, an APFS alignment quirk, a dispatch_io experiment measured
// against the pool -- lands HERE with no build-system change. The Linux file
// carried the same note until io_uring arrived; if that day comes for macOS,
// declare the factory next to make_uring_backend() and try it the same way.
