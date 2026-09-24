// KV / recurrent-state prefix snapshot cache.
//
// WHY THIS SUBSYSTEM EXISTS, and why its I/O path is deliberately dumb:
// a 1.37 GB snapshot read is ~0.34 s at 4 GB/s. The prefill it replaces is
// 5-15 minutes on this engine. The read is three orders of magnitude cheaper
// than the work it avoids, so there is no return on making it clever -- and
// every line of cleverness here is a line that can hand the model a corrupt
// state, which produces fluent text from the wrong prefix. Straightforward
// buffered I/O, one file per snapshot, atomic rename, refuse on any doubt.
//
// STATE CAPTURE IS ONE CALL, EVEN FOR HYBRID MODELS. Verified against the
// pinned SHA (84e908c62): llama_memory_hybrid::state_write forwards to BOTH
// mem_attn->state_write and mem_recr->state_write when PARTIAL_ONLY is clear,
// so a single llama_state_seq_get_data_ext with LLAMA_STATE_SEQ_FLAGS_NONE
// covers attention KV and recurrent state in one blob. That is what lets one
// cache serve M3 (sliceable KV) and Qwen3.8 / K3 (opaque recurrent) with no
// per-state-class code.
//
// FLAGS ARE NOT A TUNING KNOB -- use LLAMA_STATE_SEQ_FLAGS_NONE and nothing
// else:
//   * PARTIAL_ONLY returns ONLY the non-rewindable parts (the SWA window plus
//     recurrent state). It is not a full state. Restoring one is silent
//     corruption, not an error.
//   * ON_DEVICE keeps tensor data in device buffers, and getting the state for
//     a seq_id with that flag INVALIDATES every prior ON_DEVICE blob for the
//     same seq_id. A cache is by definition a set of prior blobs.
//
// SNAPSHOTS ARE TAKEN AT SEMANTIC BOUNDARIES, and the caller decides where.
// A snapshot at an arbitrary token is fine for crash-resume and useless as a
// shared prefix, because nobody will ever query that exact prefix hash again.
// That distinction is the Retention argument to store().
//
// ENDIANNESS / PORTABILITY: the on-disk header is written as raw host bytes.
// This is host-local storage by construction -- the compatibility stamp pins
// the snapshot to one model file on one machine -- so a portable encoding
// would buy nothing.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "mem/accountant.h"

struct llama_context;

namespace dray::cache {

// Cumulative prefix hash. h_0 = kPrefixHashSeed; h_i = extend(h_{i-1}, tok_i).
//
// FNV-1a written out explicitly over the token's four little-endian bytes.
// std::hash is deliberately not used: it is not required to be stable across
// runs, let alone across processes, and this value names files on disk.
inline constexpr uint64_t kPrefixHashSeed = 0xcbf29ce484222325ULL;  // FNV-1a basis
inline constexpr uint64_t kPrefixHashPrime = 0x00000100000001b3ULL;  // FNV-1a prime

inline uint64_t prefix_hash_extend(uint64_t h, int32_t token) {
    const uint32_t t = static_cast<uint32_t>(token);
    for (int i = 0; i < 4; ++i) {
        h ^= static_cast<uint64_t>((t >> (i * 8)) & 0xffu);
        h *= kPrefixHashPrime;
    }
    return h;
}

uint64_t prefix_hash(const int32_t* tokens, size_t n_tokens);

// Non-cryptographic 64-bit hash over a byte range, 8 bytes at a time. Used for
// the blob integrity check; a byte-at-a-time FNV would add ~2 s to a 1.37 GB
// snapshot, which is six times the read itself.
uint64_t blob_hash64(const void* data, size_t bytes);

// Stamped into every snapshot file and re-checked on every load. Restore MUST
// refuse on any mismatch: a snapshot restored against different weights does
// not fail loudly, it produces plausible text from the wrong model.
struct CompatStamp {
    uint64_t model_file_size = 0;
    int64_t  model_mtime = 0;        // filesystem clock ticks, host-local
    uint64_t engine_build_id = 0;    // hash of the caller's build identifier
    uint32_t n_ctx = 0;              // llama_n_ctx of the context that wrote it
    char     model_id[96] = {};      // llama_model_desc: arch + quant tier
};

// Builds the stamp from the live context plus the model file on disk.
// engine_build_id_str is whatever the build system can guarantee changes when
// the state format could change (git SHA + llama.cpp SHA is the intent).
bool make_compat_stamp(const std::filesystem::path& model_path,
                       const llama_context* ctx,
                       const std::string& engine_build_id_str,
                       CompatStamp* out,
                       std::string* error);

// The two lifecycles. They must NOT share a retention policy: evicting a
// running job's checkpoint to make room for a shared prefix loses days of
// generated tokens.
enum class Retention : uint8_t {
    Reuse = 0,   // taken at a semantic boundary; shareable, LRU-evictable
    Pinned = 1,  // a running job's crash-resume checkpoint; never evicted
                 // while pinned, and not charged against the reuse budget.
                 // Persisted, so a crashed job's checkpoint survives the
                 // crash it exists for. Released by unpin() or discard().
};

struct Entry {
    uint64_t prefix_hash = 0;
    uint64_t n_tokens = 0;    // tokens covered by this prefix
    uint64_t bytes = 0;       // state blob size, excluding our file header
    Retention retention = Retention::Reuse;
    std::filesystem::path path;
};

struct Stats {
    size_t   n_reuse = 0;
    size_t   n_pinned = 0;
    uint64_t reuse_bytes = 0;
    uint64_t pinned_bytes = 0;
    uint64_t budget_bytes = 0;
};

struct Config {
    std::filesystem::path dir;          // one directory, owned exclusively by
                                        // this engine instance (see open()).
    uint64_t budget_bytes = 0;          // LRU budget over Reuse entries only
    size_t   queue_capacity = 2;        // bounded async write queue; each slot
                                        // can hold a multi-GB blob, so this is
                                        // small on purpose
    // Optional ledger (Invariant 1). Snapshot blobs are resident bytes and are
    // charged here while they exist. CONCERN, noted rather than worked around:
    // mem::Category has no Snapshot member, so they land in Misc alongside the
    // tokenizer. If snapshot bytes ever need to be visible in the startup
    // report, that enum -- which this file does not own -- needs an entry.
    mem::Accountant* acct = nullptr;
    // F4/G1: bytes of the standing checkpoint allowance the engine reserved in
    // its floor. A blob that FITS the allowance skips per-capture/restore
    // charges (the allowance covers it); a LARGER blob falls back to
    // reserve-or-refuse. 0 = no allowance, every blob charges the ledger.
    uint64_t prereserved_bytes = 0;

    // Verify the blob hash on load. Costs one extra pass over the blob
    // (~0.3 s per 1.37 GB); worth it, because the failure it catches is
    // indistinguishable from a bad model at the output.
    bool verify_on_load = true;
};

class SnapshotCache {
public:
    SnapshotCache(Config cfg, CompatStamp stamp);
    ~SnapshotCache();

    SnapshotCache(const SnapshotCache&) = delete;
    SnapshotCache& operator=(const SnapshotCache&) = delete;

    // Creates the directory if needed, deletes leftover .tmp files, and indexes
    // the snapshots already there by scanning it. A directory scan is used
    // rather than an index file precisely because there is no index to be torn:
    // a snapshot exists iff a fully-written, correctly-stamped file exists.
    //
    // The directory must not be shared with another running instance -- the
    // .tmp sweep would delete its in-progress writes. Stated rather than
    // defended against, because a shared cache directory needs a lock protocol
    // this subsystem is explicitly too small to carry.
    bool open(std::string* error);

    std::optional<Entry> lookup(uint64_t prefix_hash);

    // Restores into seq_id. If the restore itself fails, seq_id is cleared
    // rather than left half-written; if validation fails first, the context is
    // never touched and seq_id keeps whatever it already held.
    bool load(const Entry& entry, llama_context* ctx, int32_t seq_id, std::string* error);

    // Captures and writes synchronously. Safe to call only between decodes:
    // llama_state_seq_get_data_ext reads live context state.
    bool store(uint64_t prefix_hash, uint64_t n_tokens, llama_context* ctx,
               int32_t seq_id, Retention retention, std::string* error);

    // Same, but only the file write moves off the decode path. The capture
    // itself is a memcpy out of the live context and CANNOT be deferred, so
    // this still costs ~0.34 s of decode-thread time per 1.37 GB. Claiming
    // otherwise would be the optimistic figure CLAUDE.md forbids.
    // Returns false if the bounded queue is full -- the caller decides whether
    // to skip this checkpoint or fall back to store().
    bool store_async(uint64_t prefix_hash, uint64_t n_tokens, llama_context* ctx,
                     int32_t seq_id, Retention retention, std::string* error);

    // Blocks until every queued write has completed.
    void flush();

    // flush(), then stop and join the writer thread. store_async() fails after
    // this. Called by the destructor.
    void drain();

    // Evicts Reuse entries, least-recently-used first, until reuse bytes fit
    // the budget. Never touches Pinned entries, and never touches an entry with
    // a load in flight.
    void evict_to_budget();

    // Pinned -> Reuse. The job that owned the checkpoint has finished, so it
    // may now be evicted like any other prefix.
    bool unpin(uint64_t prefix_hash);

    // Removes the snapshot outright. Fails while a load is in flight.
    bool discard(uint64_t prefix_hash);

    Stats stats() const;

private:
    struct Blob;   // owns the captured bytes and their accounting
    struct Job;    // one queued async write

    // Index record. Defined here rather than in the .cpp only because
    // std::unordered_map<K, V> requires V to be complete at the point the
    // member is declared.
    struct Rec {
        uint64_t hash = 0;
        uint64_t n_tokens = 0;
        uint64_t bytes = 0;
        uint64_t tick = 0;      // logical LRU timestamp
        Retention retention = Retention::Reuse;
        int      in_use = 0;    // loads in flight; blocks eviction and discard
        std::filesystem::path path;
    };

    bool capture(llama_context* ctx, int32_t seq_id, Blob* out, std::string* error);
    bool commit(uint64_t prefix_hash, uint64_t n_tokens, Retention retention,
                Blob& blob, std::string* error);
    void evict_locked();
    void writer_loop();

    Config      cfg_;
    CompatStamp stamp_;

    mutable std::mutex      mu_;
    std::condition_variable cv_;          // queue not-full / not-empty / idle
    std::unordered_map<uint64_t, Rec> index_;
    uint64_t    clock_ = 0;               // logical LRU tick
    uint64_t    reuse_bytes_ = 0;
    uint64_t    pinned_bytes_ = 0;

    std::deque<std::unique_ptr<Job>> queue_;
    std::thread queue_thread_;
    bool        writing_ = false;         // a job is in flight, off the queue
    bool        stopping_ = false;
};

}  // namespace dray::cache
