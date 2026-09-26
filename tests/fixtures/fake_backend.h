// A scripted io::Backend for driving the I/O layer without a disk.
//
// Real drives cannot be made to misbehave on demand, and the I/O paths that
// matter most -- out-of-order completion, short reads, EIO, a backend that dies
// with reads outstanding -- are exactly the ones no model run exercises. This
// fake serves in-memory "files" whose bytes follow a pattern the test can check,
// and completes requests according to a Script:
//
//   * completions come back in random order (shuffle) and in random-sized
//     batches, never more than were submitted, never the same one twice;
//   * a request may come back short (inside the data, not just at EOF) or fail
//     with an error, at a configured rate;
//   * after die_after completions the device stops answering: poll() returns
//     nothing while reads are still outstanding -- the "dead backend" case;
//   * every request is checked against the declared alignment and counted as a
//     violation if it breaks it (the real backends would fail such a read).
//
// Header-only and single-threaded by design; the threaded I/O tests wrap it.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "io/storage.h"

namespace dray::test {

class FakeBackend final : public io::Backend {
public:
    struct Script {
        uint32_t align = 4096;
        size_t   queue_depth = 8;
        uint64_t seed = 1;
        bool     shuffle = true;
        double   eio_rate = 0.0;      // chance a completion fails with an error
        double   short_rate = 0.0;    // chance a completion is cut short mid-data
        size_t   die_after = SIZE_MAX;  // completions delivered before the device dies
        // Every completion returns at most this many bytes: a deterministic short
        // read, for placing the cut exactly (e.g. past the data a caller asked
        // for but short of the alignment lead-in plus that data).
        uint64_t cap_bytes = UINT64_MAX;
    };

    FakeBackend(std::vector<std::pair<std::string, uint64_t>> files, Script s)
        : files_(std::move(files)), s_(s), rng_(s.seed) {}

    // The byte at `off` of file `f`: position-dependent, so a read that lands at
    // the wrong offset, in the wrong file or in the wrong buffer shows up.
    static uint8_t byte_at(io::FileId f, uint64_t off) {
        return static_cast<uint8_t>((off * 131u + static_cast<uint64_t>(f) * 7u + (off >> 12)) & 0xffu);
    }

    io::FileId open(const std::filesystem::path& path) override {
        for (size_t i = 0; i < files_.size(); ++i) {
            if (files_[i].first == path.string()) return static_cast<io::FileId>(i);
        }
        return io::kInvalidFile;
    }
    void close(io::FileId) override { ++closes_; }
    uint64_t size(io::FileId f) const override { return files_[static_cast<size_t>(f)].second; }
    io::Alignment alignment(io::FileId) const override { return {s_.align, s_.align, s_.align}; }

    size_t submit(const io::ReadRequest* reqs, size_t n) override {
        size_t accepted = 0;
        for (; accepted < n && queue_.size() < s_.queue_depth; ++accepted) {
            const io::ReadRequest& r = reqs[accepted];
            if (r.offset % s_.align || r.length % s_.align ||
                reinterpret_cast<uintptr_t>(r.dst) % s_.align) {
                ++violations_;
            }
            queue_.push_back(r);
            submit_log_.push_back(r.tag);
            ++submitted_;
        }
        return accepted;
    }

    size_t poll(io::Completion* out, size_t max, size_t min_complete) override {
        ++polls_;
        if (queue_.empty()) return 0;
        if (delivered_ >= s_.die_after) return 0;   // dead: nothing ever comes back
        // Deliver at least min_complete (if it can), otherwise a random number.
        const size_t cap = std::min(max, queue_.size());
        size_t k = min_complete > 0 ? std::min(cap, min_complete) : 0;
        if (k < cap) k += std::uniform_int_distribution<size_t>(0, cap - k)(rng_);
        if (k > s_.die_after - delivered_) k = s_.die_after - delivered_;
        for (size_t i = 0; i < k; ++i) {
            const size_t pick = s_.shuffle
                ? std::uniform_int_distribution<size_t>(0, queue_.size() - 1)(rng_) : 0;
            const io::ReadRequest r = queue_[pick];
            queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(pick));
            out[i] = complete(r);
            ++delivered_;
        }
        return k;
    }

    size_t max_in_flight() const override { return s_.queue_depth; }
    size_t in_flight() const override { return queue_.size(); }
    std::string describe() const override { return "fake"; }

    size_t submitted() const { return submitted_; }
    size_t delivered() const { return delivered_; }
    size_t violations() const { return violations_; }
    size_t closes() const { return closes_; }
    // Tags in the order the backend received them.
    const std::vector<uint64_t>& submit_log() const { return submit_log_; }

private:
    io::Completion complete(const io::ReadRequest& r) {
        io::Completion c;
        c.tag = r.tag;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (u(rng_) < s_.eio_rate) { c.status = -5; c.bytes = 0; return c; }
        const uint64_t fsz = files_[static_cast<size_t>(r.file)].second;
        uint64_t n = r.offset >= fsz ? 0 : std::min<uint64_t>(r.length, fsz - r.offset);
        if (n > 0 && u(rng_) < s_.short_rate) {
            n = std::uniform_int_distribution<uint64_t>(0, n - 1)(rng_);   // cut mid-data
        }
        if (n > s_.cap_bytes) n = s_.cap_bytes;
        uint8_t* d = static_cast<uint8_t*>(r.dst);
        for (uint64_t i = 0; i < n; ++i) d[i] = byte_at(r.file, r.offset + i);
        c.bytes = static_cast<uint32_t>(n);
        return c;
    }

    std::vector<std::pair<std::string, uint64_t>> files_;
    Script s_;
    std::mt19937_64 rng_;
    std::vector<io::ReadRequest> queue_;
    std::vector<uint64_t> submit_log_;
    size_t submitted_ = 0, delivered_ = 0, violations_ = 0, polls_ = 0, closes_ = 0;
};

}  // namespace dray::test
