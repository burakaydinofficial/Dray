#include "cache/snapshot_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>       // _commit, _fileno
#else
#include <unistd.h>   // fsync, fileno
#endif

#include "llama.h"

namespace dray::cache {

namespace {

// Bytes "DRAYGSNP1" in file order on a little-endian host. The trailing digit is
// part of the magic and is NOT the version -- format_version is its own field,
// so the magic never has to move.
constexpr uint64_t kMagic = 0x31504E5347445A4CULL;
constexpr uint32_t kFormatVersion = 1;

constexpr uint32_t kFlagPinned = 1u << 0;

// Written as raw host bytes; see the endianness note in the header. Field order
// is chosen so natural alignment produces no padding, which the static_assert
// below enforces -- a silently repacked header would change the on-disk format
// without changing format_version.
struct FileHeader {
    uint64_t magic;
    uint32_t format_version;
    uint32_t flags;
    uint64_t prefix_hash;
    uint64_t n_tokens;
    uint64_t blob_bytes;
    uint64_t blob_hash;
    uint64_t model_file_size;
    int64_t  model_mtime;
    uint64_t engine_build_id;
    uint32_t n_ctx;
    uint32_t reserved0;
    char     model_id[96];
};
static_assert(sizeof(FileHeader) == 176, "snapshot header layout changed; bump kFormatVersion");

void set_error(std::string* error, std::string msg) {
    if (error != nullptr) {
        *error = std::move(msg);
    }
}

std::string hex64(uint64_t v) {
    static const char digits[] = "0123456789abcdef";
    std::string s(16, '0');
    for (int i = 15; i >= 0; --i) {
        s[static_cast<size_t>(i)] = digits[static_cast<size_t>(v & 0xfULL)];
        v >>= 4;
    }
    return s;
}

bool parse_hex64(const std::string& s, uint64_t* out) {
    if (s.size() != 16u) {
        return false;
    }
    uint64_t v = 0;
    for (char c : s) {
        uint64_t d;
        if (c >= '0' && c <= '9')      { d = static_cast<uint64_t>(c - '0'); }
        else if (c >= 'a' && c <= 'f') { d = static_cast<uint64_t>(c - 'a') + 10; }
        else { return false; }
        v = (v << 4) | d;
    }
    *out = v;
    return true;
}

// Opened through the wide-char CRT entry point on Windows so model and cache
// directories outside the active code page work; path::string() would mangle
// them.
std::FILE* open_file(const std::filesystem::path& p, const char* mode) {
#ifdef _WIN32
    const std::wstring wmode(mode, mode + std::strlen(mode));
    std::FILE* f = nullptr;
    if (_wfopen_s(&f, p.c_str(), wmode.c_str()) != 0) {
        return nullptr;
    }
    return f;
#else
    return std::fopen(p.c_str(), mode);
#endif
}

// std::fflush only reaches the OS. A snapshot that survives a power cut with
// the right size and zeroed contents is exactly the torn file this cache must
// never expose, so the device is flushed before the rename publishes it.
bool sync_file(std::FILE* f) {
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}

bool read_header(const std::filesystem::path& p, FileHeader* out, std::string* error) {
    std::FILE* f = open_file(p, "rb");
    if (f == nullptr) {
        set_error(error, "snapshot: cannot open " + p.string());
        return false;
    }
    const bool got = std::fread(out, sizeof(FileHeader), 1, f) == size_t{1};
    std::fclose(f);
    if (!got) {
        set_error(error, "snapshot: short header in " + p.string());
        return false;
    }
    if (out->magic != kMagic) {
        set_error(error, "snapshot: bad magic in " + p.string());
        return false;
    }
    if (out->format_version != kFormatVersion) {
        set_error(error, "snapshot: format version " + std::to_string(out->format_version) +
                             " != " + std::to_string(kFormatVersion));
        return false;
    }
    std::error_code ec;
    const auto sz = std::filesystem::file_size(p, ec);
    if (ec || static_cast<uint64_t>(sz) != sizeof(FileHeader) + out->blob_bytes) {
        set_error(error, "snapshot: truncated file " + p.string());
        return false;
    }
    return true;
}

// Refuse on ANY mismatch. There is no partial compatibility here: a state
// restored against different weights does not fail, it generates fluent text
// from a prefix the model never saw.
bool stamp_ok(const FileHeader& h, const CompatStamp& s, std::string* error) {
    if (h.model_file_size != s.model_file_size || h.model_mtime != s.model_mtime) {
        set_error(error, "snapshot: model file changed since this snapshot was written");
        return false;
    }
    if (h.n_ctx != s.n_ctx) {
        set_error(error, "snapshot: n_ctx " + std::to_string(h.n_ctx) + " != " +
                             std::to_string(s.n_ctx));
        return false;
    }
    if (h.engine_build_id != s.engine_build_id) {
        set_error(error, "snapshot: written by a different engine build");
        return false;
    }
    if (std::memcmp(h.model_id, s.model_id, sizeof(s.model_id)) != 0) {
        set_error(error, "snapshot: different model/quant");
        return false;
    }
    return true;
}

// Demotes Pinned -> Reuse on disk. One aligned 4-byte field inside the first
// sector; a torn write of that is not a failure mode real storage exhibits, so
// this does not warrant a rewrite-and-rename of a multi-GB file.
bool rewrite_flags(const std::filesystem::path& p, uint32_t flags, const CompatStamp& stamp,
                   std::string* error) {
    FileHeader h{};
    if (!read_header(p, &h, error) || !stamp_ok(h, stamp, error)) {
        return false;
    }
    std::FILE* f = open_file(p, "r+b");
    if (f == nullptr) {
        set_error(error, "snapshot: cannot reopen " + p.string());
        return false;
    }
    bool ok = std::fseek(f, static_cast<long>(offsetof(FileHeader, flags)), SEEK_SET) == 0;
    if (ok) {
        ok = std::fwrite(&flags, sizeof(flags), 1, f) == size_t{1};
    }
    if (ok) {
        ok = std::fflush(f) == 0 && sync_file(f);
    }
    std::fclose(f);
    if (!ok) {
        set_error(error, "snapshot: failed to update flags in " + p.string());
    }
    return ok;
}

// Reads a snapshot file and pushes it into the context. On any failure after
// the context has been touched the destination sequence is cleared: a
// half-restored sequence is the same class of bug as a wrong-model restore.
bool restore_from_file(const std::filesystem::path& path, llama_context* ctx, int32_t seq_id,
                       const CompatStamp& stamp, bool verify, mem::Accountant* acct,
                       uint64_t prereserved, std::string* error) {
    FileHeader h{};
    if (!read_header(path, &h, error) || !stamp_ok(h, stamp, error)) {
        return false;
    }
    const size_t len = static_cast<size_t>(h.blob_bytes);
    if (len == 0 || static_cast<uint64_t>(len) != h.blob_bytes) {
        set_error(error, "snapshot: implausible blob size in " + path.string());
        return false;
    }
    // G2: the standing allowance covers a fitting blob on RESTORE exactly as
    // on capture; nulling acct here makes every release below symmetric.
    if (acct != nullptr && prereserved && static_cast<uint64_t>(len) <= prereserved) {
        acct = nullptr;
    }
    // Pass-4b: only the excess past the standing allowance is new pressure.
    const size_t charged = (acct && prereserved && len > prereserved)
                               ? len - static_cast<size_t>(prereserved) : len;
    if (acct != nullptr && !acct->reserve(mem::Category::Misc, charged)) {
        // G2: a cap refusal is TRANSIENT -- the blob is fine, the moment is
        // not. The prefix lets callers keep the checkpoint and the index.
        set_error(error, "TRANSIENT: restore buffer of " + std::to_string(len) +
                             " bytes does not fit the resident cap right now");
        return false;
    }
    // default-init, not value-init: a multi-GB memset before the read that
    // overwrites every byte of it is pure waste.
    std::unique_ptr<uint8_t[]> buf(new (std::nothrow) uint8_t[len]);
    if (!buf) {
        if (acct != nullptr) { acct->release(mem::Category::Misc, charged); }
        // H10: an allocation failure under a small cap is TRANSIENT -- the blob
        // is fine, the moment is not; de-indexing it orphaned a valid pinned
        // file forever.
        set_error(error, "TRANSIENT: out of memory for " + std::to_string(len) + " bytes");
        return false;
    }

    bool ok = false;
    std::FILE* f = open_file(path, "rb");
    if (f != nullptr) {
        // NOTE (Invariant 2): this read retains page cache, unlike every other
        // read in the engine. Uncached I/O would need the blob sector-aligned
        // within the file, which the header breaks. Left as-is because it is a
        // once-per-job cost, but it is a real cost under a small cap.
        ok = std::fseek(f, static_cast<long>(sizeof(FileHeader)), SEEK_SET) == 0 &&
             std::fread(buf.get(), 1, len, f) == len;
        std::fclose(f);
    }
    if (!ok) {
        if (acct != nullptr) { acct->release(mem::Category::Misc, charged); }
        set_error(error, "snapshot: short read on " + path.string());
        return false;
    }
    if (verify && blob_hash64(buf.get(), len) != h.blob_hash) {
        if (acct != nullptr) { acct->release(mem::Category::Misc, charged); }
        set_error(error, "snapshot: blob hash mismatch in " + path.string());
        return false;
    }

    // FLAGS_NONE. PARTIAL_ONLY would restore only the SWA window plus recurrent
    // state over whatever the sequence already held; ON_DEVICE would invalidate
    // sibling blobs. Neither is a full state.
    const size_t read = llama_state_seq_set_data_ext(ctx, buf.get(), len,
                                                     static_cast<llama_seq_id>(seq_id),
                                                     LLAMA_STATE_SEQ_FLAGS_NONE);
    if (acct != nullptr) { acct->release(mem::Category::Misc, charged); }

    // Strict equality on purpose. Both counters include llama.cpp's own 8-byte
    // magic + seq_id preamble (state_seq_{read,write}_data return io.n_bytes()
    // after it), so a well-formed blob consumes exactly what it occupied.
    // Anything else means the state and the reader disagree, and the cost of
    // being wrong here is a re-prefill, not a wrong answer.
    if (read == 0 || read != len) {
        (void)llama_memory_seq_rm(llama_get_memory(ctx), static_cast<llama_seq_id>(seq_id), -1, -1);
        set_error(error, "snapshot: llama_state_seq_set_data_ext consumed " +
                             std::to_string(read) + " of " + std::to_string(len) + " bytes");
        return false;
    }
    return true;
}

}  // namespace

uint64_t prefix_hash(const int32_t* tokens, size_t n_tokens) {
    uint64_t h = kPrefixHashSeed;
    for (size_t i = 0; i < n_tokens; ++i) {
        h = prefix_hash_extend(h, tokens[i]);
    }
    return h;
}

uint64_t blob_hash64(const void* data, size_t bytes) {
    // splitmix64 finalizer over 8-byte words. Not cryptographic and not
    // collision-proof against an adversary -- it exists to catch storage
    // corruption, which the compatibility stamp cannot.
    const auto mix = [](uint64_t x) -> uint64_t {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    };
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(bytes);
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, 8);
        h = mix(h ^ w) + 0x9e3779b97f4a7c15ULL;
    }
    uint64_t tail = 0;
    for (size_t k = 0; i + k < bytes; ++k) {
        tail |= static_cast<uint64_t>(p[i + k]) << (k * 8);
    }
    return mix(h ^ tail);
}

bool make_compat_stamp(const std::filesystem::path& model_path, const llama_context* ctx,
                       const std::string& engine_build_id_str, CompatStamp* out,
                       std::string* error) {
    if (out == nullptr || ctx == nullptr) {
        set_error(error, "snapshot: make_compat_stamp needs a context and an output");
        return false;
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(model_path, ec);
    if (ec) {
        set_error(error, "snapshot: cannot stat model " + model_path.string());
        return false;
    }
    const auto mtime = std::filesystem::last_write_time(model_path, ec);
    if (ec) {
        set_error(error, "snapshot: cannot read mtime of " + model_path.string());
        return false;
    }
    const llama_model* model = llama_get_model(ctx);
    if (model == nullptr) {
        set_error(error, "snapshot: context has no model");
        return false;
    }

    CompatStamp s;
    s.model_file_size = static_cast<uint64_t>(size);
    // Filesystem-clock ticks, compared only against ticks from the same host.
    s.model_mtime = static_cast<int64_t>(mtime.time_since_epoch().count());
    s.n_ctx = llama_n_ctx(ctx);
    s.engine_build_id = blob_hash64(engine_build_id_str.data(), engine_build_id_str.size());

    // llama_model_desc is arch + size + quant tier, e.g. "minimax-m3 235B.A22B Q2_K - Medium".
    // Truncation past 95 chars is tolerable only because file size and mtime
    // are also pinned; on its own this string is not a unique model identity.
    char desc[192] = {};
    const int32_t n = llama_model_desc(model, desc, sizeof(desc));
    if (n <= 0) {
        set_error(error, "snapshot: llama_model_desc failed");
        return false;
    }
    const size_t copy = std::min(sizeof(s.model_id) - 1, std::strlen(desc));
    std::memcpy(s.model_id, desc, copy);

    *out = s;
    return true;
}

// ---------------------------------------------------------------------------

// Captured state plus its ledger reservation. Move-only: two Blobs holding the
// same reservation would double-release it.
struct SnapshotCache::Blob {
    std::unique_ptr<uint8_t[]> data;
    size_t cap = 0;              // bytes reserved with the accountant
    size_t len = 0;              // bytes actually written by llama.cpp
    mem::Accountant* acct = nullptr;

    Blob() = default;
    ~Blob() { reset(); }

    Blob(const Blob&) = delete;
    Blob& operator=(const Blob&) = delete;

    Blob(Blob&& o) noexcept
        : data(std::move(o.data)), cap(o.cap), len(o.len), acct(o.acct) {
        o.cap = 0;
        o.len = 0;
        o.acct = nullptr;
    }
    Blob& operator=(Blob&& o) noexcept {
        if (this != &o) {
            reset();
            data = std::move(o.data);
            cap = o.cap;
            len = o.len;
            acct = o.acct;
            o.cap = 0;
            o.len = 0;
            o.acct = nullptr;
        }
        return *this;
    }

    void reset() {
        if (acct != nullptr && cap != 0) {
            acct->release(mem::Category::Misc, cap);
        }
        data.reset();
        cap = 0;
        len = 0;
        acct = nullptr;
    }
};

struct SnapshotCache::Job {
    uint64_t  prefix_hash = 0;
    uint64_t  n_tokens = 0;
    Retention retention = Retention::Reuse;
    Blob      blob;
};

SnapshotCache::SnapshotCache(Config cfg, CompatStamp stamp)
    : cfg_(std::move(cfg)), stamp_(stamp) {}

SnapshotCache::~SnapshotCache() {
    drain();
}

bool SnapshotCache::open(std::string* error) {
    if (cfg_.dir.empty()) {
        set_error(error, "snapshot: no cache directory configured");
        return false;
    }
    // Refused rather than defaulted: a zero budget could mean "unbounded" or
    // "keep nothing", and guessing either way silently either fills the disk or
    // throws away every snapshot the moment it is written.
    if (cfg_.budget_bytes == 0) {
        set_error(error, "snapshot: budget_bytes must be set");
        return false;
    }
    if (cfg_.queue_capacity == 0) {
        cfg_.queue_capacity = 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(cfg_.dir, ec);
    if (!std::filesystem::is_directory(cfg_.dir, ec)) {
        set_error(error, "snapshot: cannot create " + cfg_.dir.string());
        return false;
    }

    struct Found {
        std::filesystem::file_time_type mtime;
        Rec rec;
    };
    std::vector<Found> found;

    std::filesystem::directory_iterator it(cfg_.dir, ec);
    if (ec) {
        set_error(error, "snapshot: cannot scan " + cfg_.dir.string());
        return false;
    }
    const std::filesystem::directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            break;
        }
        const std::filesystem::path p = it->path();
        std::error_code stat_ec;
        if (!std::filesystem::is_regular_file(p, stat_ec) || stat_ec) {
            continue;
        }
        const std::string ext = p.extension().string();
        if (ext == ".tmp") {
            // An uncommitted write from a previous run. Safe to delete because
            // a .tmp is never discoverable as a snapshot; see the shared-
            // directory caveat on open() in the header.
            std::error_code rm_ec;
            std::filesystem::remove(p, rm_ec);
            continue;
        }
        if (ext != ".lzs") {
            continue;
        }

        FileHeader h{};
        // Foreign, torn or wrong-model files are left on disk and simply not
        // indexed. Nothing on a read path deletes a user's data.
        if (!read_header(p, &h, nullptr) || !stamp_ok(h, stamp_, nullptr)) {
            continue;
        }
        uint64_t name_hash = 0;
        if (!parse_hex64(p.stem().string(), &name_hash) || name_hash != h.prefix_hash) {
            continue;
        }

        Found f;
        f.mtime = std::filesystem::last_write_time(p, stat_ec);
        if (stat_ec) {
            f.mtime = std::filesystem::file_time_type{};
        }
        f.rec.hash = h.prefix_hash;
        f.rec.n_tokens = h.n_tokens;
        f.rec.bytes = h.blob_bytes;
        f.rec.retention = (h.flags & kFlagPinned) != 0 ? Retention::Pinned : Retention::Reuse;
        f.rec.path = p;
        found.push_back(std::move(f));
    }

    // mtime order seeds the LRU across restarts; the logical clock only exists
    // within a run.
    std::sort(found.begin(), found.end(),
              [](const Found& a, const Found& b) { return a.mtime < b.mtime; });

    {
        std::lock_guard<std::mutex> lk(mu_);
        for (Found& f : found) {
            f.rec.tick = ++clock_;
            if (f.rec.retention == Retention::Pinned) {
                pinned_bytes_ += f.rec.bytes;
            } else {
                reuse_bytes_ += f.rec.bytes;
            }
            const uint64_t key = f.rec.hash;
            index_.emplace(key, std::move(f.rec));
        }
        evict_locked();
        stopping_ = false;
    }

    if (!queue_thread_.joinable()) {
        queue_thread_ = std::thread(&SnapshotCache::writer_loop, this);
    }
    return true;
}

std::optional<Entry> SnapshotCache::lookup(uint64_t hash) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(hash);
    if (it == index_.end()) {
        return std::nullopt;
    }
    it->second.tick = ++clock_;

    Entry e;
    e.prefix_hash = it->second.hash;
    e.n_tokens = it->second.n_tokens;
    e.bytes = it->second.bytes;
    e.retention = it->second.retention;
    e.path = it->second.path;
    return e;
}

bool SnapshotCache::load(const Entry& entry, llama_context* ctx, int32_t seq_id,
                         std::string* error) {
    if (ctx == nullptr) {
        set_error(error, "snapshot: load needs a context");
        return false;
    }

    std::filesystem::path path;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = index_.find(entry.prefix_hash);
        if (it == index_.end()) {
            set_error(error, "snapshot: entry was evicted before it could be loaded");
            return false;
        }
        // in_use pins the file for the duration of the read. The storage
        // interface has no cancel and eviction is an unlink, so this is the
        // only thing standing between a long read and its own file vanishing.
        it->second.in_use += 1;
        it->second.tick = ++clock_;
        path = it->second.path;
    }

    std::string local;
    const bool ok = restore_from_file(path, ctx, seq_id, stamp_, cfg_.verify_on_load,
                                      cfg_.acct, cfg_.prereserved_bytes, &local);

    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = index_.find(entry.prefix_hash);
        if (it != index_.end() && it->second.in_use > 0) {
            it->second.in_use -= 1;
        }
        // G2: a TRANSIENT failure (cap refusal) is NOT permanent -- de-indexing
        // it would orphan a valid pinned blob forever. Keep it indexed.
        const bool transient = local.rfind("TRANSIENT:", 0) == 0;
        if (!ok && !transient && it != index_.end() && it->second.in_use == 0) {
            // It failed validation once; it will fail every time. De-index so
            // it stops being a permanent false hit -- but leave the file, since
            // the reason may be a model the user will point back at.
            if (it->second.retention == Retention::Pinned) {
                pinned_bytes_ -= it->second.bytes;
            } else {
                reuse_bytes_ -= it->second.bytes;
            }
            index_.erase(it);
        }
        evict_locked();
    }
    cv_.notify_all();

    if (!ok) {
        set_error(error, local);
    }
    return ok;
}

bool SnapshotCache::capture(llama_context* ctx, int32_t seq_id, Blob* out, std::string* error) {
    if (ctx == nullptr || out == nullptr) {
        set_error(error, "snapshot: capture needs a context");
        return false;
    }
    // LLAMA_STATE_SEQ_FLAGS_NONE, always. Verified against 84e908c62:
    // llama_memory_hybrid::state_write emits attention KV and recurrent state
    // in one blob under this flag, so hybrid models need no special case here.
    const size_t need = llama_state_seq_get_size_ext(ctx, static_cast<llama_seq_id>(seq_id),
                                                     LLAMA_STATE_SEQ_FLAGS_NONE);
    if (need == 0) {
        set_error(error, "snapshot: llama_state_seq_get_size_ext returned 0");
        return false;
    }
    // G1: "covered" means the blob genuinely fits the standing allowance the
    // engine reserved; anything larger falls back to reserve-or-refuse. The
    // previous boolean flag skipped the ledger for EVERY blob on the strength
    // of an allowance sized from an empty context (16 bytes) -- multi-GB
    // captures ran entirely outside the cap.
    const bool covered = cfg_.acct != nullptr && cfg_.prereserved_bytes != 0 &&
                         need <= cfg_.prereserved_bytes;
    // Pass-4b: when NOT covered, the allowance is still standing -- reserving
    // the full need on top of it double-charged by the allowance's size. Only
    // the EXCESS past the allowance is new pressure.
    const size_t excess = covered ? 0
        : (need > cfg_.prereserved_bytes ? need - static_cast<size_t>(cfg_.prereserved_bytes)
                                         : need);
    if (cfg_.acct != nullptr && !covered &&
        !cfg_.acct->reserve(mem::Category::Misc, excess)) {
        set_error(error, "snapshot: capture buffer of " + std::to_string(need) +
                             " bytes does not fit the resident cap");
        return false;
    }

    Blob b;
    b.acct = covered ? nullptr : cfg_.acct;   // G1: covered = inside the allowance
    b.cap = excess;   // pass-4b: release exactly what was reserved, never more
    b.data.reset(new (std::nothrow) uint8_t[need]);
    if (!b.data) {
        set_error(error, "snapshot: out of memory for " + std::to_string(need) + " bytes");
        return false;  // ~Blob releases the reservation
    }

    // This is a memcpy out of live context state: it must happen between
    // decodes, on the thread that owns the context, and it cannot be deferred.
    const size_t got = llama_state_seq_get_data_ext(ctx, b.data.get(), need,
                                                    static_cast<llama_seq_id>(seq_id),
                                                    LLAMA_STATE_SEQ_FLAGS_NONE);
    if (got == 0 || got > need) {
        set_error(error, "snapshot: llama_state_seq_get_data_ext wrote " + std::to_string(got) +
                             " of " + std::to_string(need) + " bytes");
        return false;
    }
    b.len = got;
    *out = std::move(b);
    return true;
}

bool SnapshotCache::commit(uint64_t hash, uint64_t n_tokens, Retention retention, Blob& blob,
                           std::string* error) {
    const std::string name = hex64(hash);
    const std::filesystem::path final_path = cfg_.dir / (name + ".lzs");
    const std::filesystem::path tmp_path = cfg_.dir / (name + ".lzs.tmp");

    FileHeader h{};
    h.magic = kMagic;
    h.format_version = kFormatVersion;
    h.flags = (retention == Retention::Pinned) ? kFlagPinned : 0u;
    h.prefix_hash = hash;
    h.n_tokens = n_tokens;
    h.blob_bytes = static_cast<uint64_t>(blob.len);
    h.blob_hash = blob_hash64(blob.data.get(), blob.len);
    h.model_file_size = stamp_.model_file_size;
    h.model_mtime = stamp_.model_mtime;
    h.engine_build_id = stamp_.engine_build_id;
    h.n_ctx = stamp_.n_ctx;
    h.reserved0 = 0;
    std::memcpy(h.model_id, stamp_.model_id, sizeof(h.model_id));

    std::error_code ec;
    std::FILE* f = open_file(tmp_path, "wb");
    if (f == nullptr) {
        set_error(error, "snapshot: cannot create " + tmp_path.string());
        return false;
    }
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == size_t{1};
    if (ok && blob.len != 0) {
        ok = std::fwrite(blob.data.get(), 1, blob.len, f) == blob.len;
    }
    if (ok) {
        ok = std::fflush(f) == 0 && sync_file(f);
    }
    std::fclose(f);
    if (!ok) {
        std::filesystem::remove(tmp_path, ec);
        set_error(error, "snapshot: write failed for " + tmp_path.string());
        return false;
    }

    // Publish atomically. Anything discoverable as ".lzs" is therefore complete.
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        std::error_code rm_ec;
        std::filesystem::remove(tmp_path, rm_ec);
        set_error(error, "snapshot: cannot publish " + final_path.string());
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = index_.find(hash);
        if (it != index_.end()) {
            // Same prefix hash means the same tokens means the same state, so
            // the file we just wrote is equivalent to the one it replaced.
            // Only retention and LRU position change; in_use carries over,
            // because a reader is still holding that path open.
            Rec& r = it->second;
            if (r.retention == Retention::Pinned) {
                pinned_bytes_ -= r.bytes;
            } else {
                reuse_bytes_ -= r.bytes;
            }
            r.n_tokens = n_tokens;
            r.bytes = static_cast<uint64_t>(blob.len);
            r.tick = ++clock_;
            r.retention = retention;
            r.path = final_path;
            if (retention == Retention::Pinned) {
                pinned_bytes_ += r.bytes;
            } else {
                reuse_bytes_ += r.bytes;
            }
        } else {
            Rec r;
            r.hash = hash;
            r.n_tokens = n_tokens;
            r.bytes = static_cast<uint64_t>(blob.len);
            r.tick = ++clock_;
            r.retention = retention;
            r.path = final_path;
            if (retention == Retention::Pinned) {
                pinned_bytes_ += r.bytes;
            } else {
                reuse_bytes_ += r.bytes;
            }
            index_.emplace(hash, std::move(r));
        }
        evict_locked();
    }

    blob.reset();  // the bytes are on disk; stop charging the ledger for them
    return true;
}

bool SnapshotCache::store(uint64_t hash, uint64_t n_tokens, llama_context* ctx, int32_t seq_id,
                          Retention retention, std::string* error) {
    Blob b;
    if (!capture(ctx, seq_id, &b, error)) {
        return false;
    }
    return commit(hash, n_tokens, retention, b, error);
}

bool SnapshotCache::store_async(uint64_t hash, uint64_t n_tokens, llama_context* ctx,
                                int32_t seq_id, Retention retention, std::string* error) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopping_ || !queue_thread_.joinable()) {
            set_error(error, "snapshot: writer thread is not running");
            return false;
        }
        if (queue_.size() >= cfg_.queue_capacity) {
            set_error(error, "snapshot: write queue full");
            return false;
        }
    }

    auto job = std::make_unique<Job>();
    job->prefix_hash = hash;
    job->n_tokens = n_tokens;
    job->retention = retention;
    if (!capture(ctx, seq_id, &job->blob, error)) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(mu_);
        // Re-checked because capture() took real time and the queue is bounded
        // in slots that each hold gigabytes; blocking here would put the write
        // back on the decode path, which is the whole point of this call.
        if (stopping_ || queue_.size() >= cfg_.queue_capacity) {
            set_error(error, "snapshot: write queue full");
            return false;  // ~Job releases the capture reservation
        }
        queue_.push_back(std::move(job));
    }
    cv_.notify_all();
    return true;
}

void SnapshotCache::writer_loop() {
    for (;;) {
        std::unique_ptr<Job> job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;  // only reachable with stopping_ set
            }
            job = std::move(queue_.front());
            queue_.pop_front();
            writing_ = true;
        }
        cv_.notify_all();  // a producer may be waiting on a free slot

        // A failed background write means the snapshot will not be there next
        // time. It must never take the engine down mid-generation.
        std::string err;
        (void)commit(job->prefix_hash, job->n_tokens, job->retention, job->blob, &err);
        job.reset();

        {
            std::lock_guard<std::mutex> lk(mu_);
            writing_ = false;
        }
        cv_.notify_all();
    }
}

void SnapshotCache::flush() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return queue_.empty() && !writing_; });
}

void SnapshotCache::drain() {
    if (queue_thread_.joinable()) {
        flush();
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        queue_thread_.join();
    }
    std::lock_guard<std::mutex> lk(mu_);
    stopping_ = true;
}

void SnapshotCache::evict_to_budget() {
    std::lock_guard<std::mutex> lk(mu_);
    evict_locked();
}

void SnapshotCache::evict_locked() {
    // Linear scan per eviction. The index holds tens of entries, not millions --
    // each one is gigabytes -- so a heap would be more code than it saves.
    while (reuse_bytes_ > cfg_.budget_bytes) {
        auto victim = index_.end();
        for (auto it = index_.begin(); it != index_.end(); ++it) {
            // Pinned entries are invisible to eviction: they are the crash-
            // resume checkpoint of a job that is still running, and reclaiming
            // one to make room for a shared prefix trades days of generation
            // for minutes of prefill.
            if (it->second.retention != Retention::Reuse) {
                continue;
            }
            if (it->second.in_use != 0) {
                continue;
            }
            if (victim == index_.end() || it->second.tick < victim->second.tick) {
                victim = it;
            }
        }
        if (victim == index_.end()) {
            return;  // over budget with nothing evictable; visible in stats()
        }
        std::error_code ec;
        std::filesystem::remove(victim->second.path, ec);
        // De-indexed even if the unlink failed: an undeletable file is a disk
        // problem, but leaving it indexed makes it the permanent LRU victim and
        // stalls eviction forever.
        reuse_bytes_ -= victim->second.bytes;
        index_.erase(victim);
    }
}

bool SnapshotCache::unpin(uint64_t hash) {
    std::filesystem::path path;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = index_.find(hash);
        if (it == index_.end() || it->second.retention != Retention::Pinned) {
            return false;
        }
        path = it->second.path;
    }

    // Persist the demotion before it takes effect in memory. If the process
    // dies between the two, the entry stays pinned -- the safe direction, since
    // a stale pin only costs disk while a lost pin costs a job.
    if (!rewrite_flags(path, 0u, stamp_, nullptr)) {
        return false;
    }

    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(hash);
    if (it == index_.end() || it->second.retention != Retention::Pinned) {
        return false;
    }
    pinned_bytes_ -= it->second.bytes;
    it->second.retention = Retention::Reuse;
    reuse_bytes_ += it->second.bytes;
    it->second.tick = ++clock_;
    evict_locked();
    return true;
}

bool SnapshotCache::discard(uint64_t hash) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = index_.find(hash);
    if (it == index_.end() || it->second.in_use != 0) {
        return false;
    }
    std::error_code ec;
    std::filesystem::remove(it->second.path, ec);
    if (it->second.retention == Retention::Pinned) {
        pinned_bytes_ -= it->second.bytes;
    } else {
        reuse_bytes_ -= it->second.bytes;
    }
    index_.erase(it);
    return true;
}

Stats SnapshotCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s;
    s.budget_bytes = cfg_.budget_bytes;
    s.reuse_bytes = reuse_bytes_;
    s.pinned_bytes = pinned_bytes_;
    for (const auto& kv : index_) {
        if (kv.second.retention == Retention::Pinned) {
            ++s.n_pinned;
        } else {
            ++s.n_reuse;
        }
    }
    return s;
}

}  // namespace dray::cache
