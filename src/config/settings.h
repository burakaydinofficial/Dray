// Tuning settings: every value that is not a fact read from the model or
// measured at runtime, with where it came from.
//
// Three rules (owner, 2026-09-29):
//   * A property of the MACHINE lives in the system config -- one file per
//     machine, a commented default shipped with the binary, edited by the user.
//   * A property of a MODEL is DECLARED in that model's config file, keyed by
//     GGUF general.architecture (optionally narrowed to one GGUF file).
//   * Nothing tuned on one model or one machine sits in engine code; the
//     built-in values here are the documented fallback when no file says
//     otherwise, and the readout always says which layer a value came from.
//
// Layers, each overriding the one before, per key:
//   built-in < shipped system < user system < shipped model < user model
//            < the model file's per-GGUF-file section < command line
// A model file may override only the POLICY fields (prefill, cache); hardware
// fields (threads, io) are system-only and a model file naming one is an error.
// Parsing is strict: unknown keys and wrong types are errors, so a typo can
// never silently do nothing.
#pragma once

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace dray::config {

enum class Source : uint8_t {
    Builtin,
    ShippedSystem,
    UserSystem,
    ShippedModel,
    UserModel,
    Cli,
};
const char* source_name(Source s);

template <class T>
struct Setting {
    T           value{};
    Source      source = Source::Builtin;
    std::string from;   // the file (or flag) that set it; empty for built-in

    void set(const T& v, Source s, const std::string& f) {
        value = v;
        source = s;
        from = f;
    }
};

struct Settings {
    // --- system only: the machine -------------------------------------------
    // Compute threads for decode, and for prefill (0 = every core the process
    // may use: hardware minus threads.reserved).
    Setting<int> decode_threads;
    Setting<int> prefill_threads;
    // Threads this process runs besides ggml's pool (the oversubscription
    // margin; ggml's pool spin-waits, so exceeding the cores collapses).
    Setting<int> reserved_threads;
    // Uncached reads kept in flight.
    Setting<int> queue_depth;

    // --- policy: system, a model file may override ---------------------------
    // Prefill physical batch (tokens), CPU-only and with --gpu.
    Setting<int> prefill_chunk_cpu;
    Setting<int> prefill_chunk_gpu;
    // --gpu prefill: once an expert copy lands, read the next layer's experts of that
    // kind ahead, guessing the same routing (ExpertCompactor::read_ahead_next_layer).
    // Fills drive idle time at the price of the guesses the next layer does not use
    // (Qwen3.6 35B-A3B: 12-19% more bytes for ~2-4% beyond merged reads) -- a model
    // that gains more may turn it on in its own file.
    Setting<bool> prefill_gpu_read_ahead;
    // Share of the bytes a pass reads that the frequency cache may copy into
    // slots in that pass (see backend::Config::admit_byte_fraction).
    Setting<double> admit_byte_fraction;
    // serve: requests generating at once (each its own slot and KV, funded at
    // load), and prompt tokens a step may carry besides the decode tokens.
    // A model may override both: its per-sequence state decides how many fit.
    Setting<int> serve_max_parallel;
    Setting<int> serve_prefill_tokens_per_step;
    // Recurrent models: checkpoints kept per slot for conversation reuse (0 =
    // off; each is one sequence's recurrent state, reserved at load), and how
    // many tokens before a prompt's end the checkpoint is taken (a next turn
    // re-tokenizes the end of this one). Attention-only models need neither.
    Setting<int> serve_checkpoints_per_slot;
    Setting<int> serve_checkpoint_offset;
    // A kept conversation idle longer than this is no longer reused (seconds).
    Setting<int> serve_kv_idle_timeout_s;
    // Tokens the pool of parked conversations (kept outside the slots) may hold:
    // -1 = the slots' context x max_parallel, 0 = no pool.
    Setting<int> serve_kv_pool_tokens;

    // gpu (system only -- a property of this machine): the HARD limit on what --gpu may
    // hold in VRAM, bytes; 0 = automatic: vram_auto_fraction of the device's VRAM,
    // or vram_auto_fraction_small on a card of at most vram_small_card.
    Setting<uint64_t> gpu_vram_cap;
    Setting<double>   gpu_vram_auto_fraction;
    Setting<double>   gpu_vram_auto_fraction_small;
    Setting<uint64_t> gpu_vram_small_card;

    // --- model only ------------------------------------------------------------
    // llama --override-kv strings the released conversions lack; an explicit
    // --override-kv for the same key wins.
    Setting<std::vector<std::string>> kv_defaults;

    // Which files were applied, in order (for the readout).
    std::vector<std::string> applied;
};

// The documented fallbacks, all sources Builtin.
Settings builtin_settings();

struct Locations {
    std::filesystem::path shipped;   // config/ next to the binary (may not exist)
    std::filesystem::path user;      // the user's config directory (may not exist)
};

// What the command line set; an unset field leaves the files' value.
// A byte count or a K/M/G/T suffix (bare G is GiB): --cap, --vram-cap, and
// size-valued settings share this one parser.
uint64_t parse_size(const std::string& s, bool* ok);

struct CliSettings {
    int threads = 0;          // --threads: decode AND prefill, as always
    int prefill_chunk = 0;    // --prefill-chunk: both chunk sizes
    int parallel = 0;         // --parallel: serve.max_parallel
    uint64_t vram_cap = 0;    // --vram-cap: gpu.vram_cap
};

// Resolves every layer for a model of architecture `arch` whose first GGUF
// file is named `gguf_file` (file name only). Missing files are simply absent
// layers; a file that exists but is invalid is an error, with its path and the
// offending key in `*err`.
bool resolve(const Locations& where, const std::string& arch, const std::string& gguf_file,
             const CliSettings& cli, Settings* out, std::string* err);

// One layer, from JSON text (with comments). `model_file` selects the model
// schema (arch, kv_defaults, overrides, files) instead of the system one.
// Exposed for tests; resolve() is the production entry point.
bool apply_json(const std::string& text, const std::string& origin, Source source,
                bool model_file, const std::string& gguf_file, Settings* s, std::string* err);

// Every effective value with its source, one per line.
void describe(const Settings& s, std::ostream& o);

}  // namespace dray::config
