// Scratch directories for tests that need real files on a real filesystem.
//
// WHY real files rather than an in-memory stub: the storage backend under test
// is uncached platform I/O (FILE_FLAG_NO_BUFFERING / O_DIRECT / F_NOCACHE) and
// its alignment constraints are runtime values discovered from the mount
// (Invariant 5). A memory stub would answer every alignment question with the
// number the test wanted to hear.
//
// Header-only and free of main(), so fixture translation units may include it.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

namespace dray::testfix {

// Unique per (process start time, counter) so parallel ctest jobs and repeated
// runs never collide, and so a crashed run's leftovers are never reused.
inline std::string unique_suffix() {
    static const uint64_t stamp = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    static std::atomic<uint32_t> counter{0};
    const uint32_t n = counter.fetch_add(1, std::memory_order_relaxed);
    return std::to_string(stamp) + "_" + std::to_string(n);
}

class TempDir {
public:
    explicit TempDir(const std::string& tag) {
        std::error_code ec;
        std::filesystem::path root = std::filesystem::temp_directory_path(ec);
        if (ec) {
            root = std::filesystem::current_path();
        }
        path_ = root / ("dray_" + tag + "_" + unique_suffix());
        std::filesystem::create_directories(path_, ec);
        ok_ = !ec;
    }

    ~TempDir() {
        // Best effort: a test that leaves a file open on Windows must not turn a
        // pass into a crash in a destructor.
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    bool ok() const { return ok_; }
    const std::filesystem::path& path() const { return path_; }
    std::filesystem::path file(const std::string& name) const { return path_ / name; }

private:
    std::filesystem::path path_;
    bool ok_ = false;
};

}  // namespace dray::testfix
