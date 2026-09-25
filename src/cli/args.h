// Command-line arguments: one struct for every subcommand, parsed once.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dray::cli {

struct Args {
    std::string cmd;
    std::string model;
    std::string prompt = "Hello";
    // The default cap is deliberately polite (CLAUDE.md): a tool that grabs memory
    // by default gets uninstalled by exactly the audience it is for, and the
    // failure is silent because the user blames the slowdown on the model.
    uint64_t cap = 4ull << 30;
    uint32_t n_ctx = 0;          // 0 = take the profile default
    int32_t  n_predict = 32;
    uint32_t seed = 1234;
    bool     greedy = true;
    std::string status_path;
    // Load through llama.cpp normally instead of streaming: the reference half of a
    // differential test. Only usable on models that fit.
    bool     no_stream = false;
    int      port = 8080;        // serve: OpenAI-compatible API bind port
    uint32_t n_batch_seq = 1;    // batch: concurrent sequences
    std::string prompts_file;    // batch: one prompt per line (else -p replicated)
    std::vector<std::string> stop_seqs;  // batch: --stop, repeatable
    int32_t rotate_span = 0;             // batch: --rotate SPAN (0 = off)
    std::string state_dir;               // batch: --state-dir (required with --rotate)
    std::vector<std::string> kv_overrides;  // --override-kv key=type:value, repeatable
    bool gpu = false;                       // --gpu: runtime GPU consent (backend-carrying builds)
    std::string kv;                         // --kv q4|q8|f16: KV cache quantization (floor priced accordingly)
    int32_t prefill_chunk = 0;              // --prefill-chunk: tokens per prefill pass (GPU buffer scales with it)
    int32_t n_threads = 0;                  // --threads: compute threads (0 = measured optimum)
    bool force_stream = false;              // --force-stream: pin the streaming path even if the model fits
    bool resident = false;                  // --resident: EXPERIMENTAL llama-native allocation when it fits
    std::string out_dir;         // repack: output directory
    std::string repack_dir;      // every engine command: apply a repacked companion
    std::string jobs_dir;        // serve: checkpoint+resume dir (empty = off)
    std::string api_key;         // serve: bearer token for /v1/* (empty = open, localhost-only)
    bool     help = false;
};

// Cap parsing accepts a plain byte count or a K/M/G/T suffix; bare G means GiB.
uint64_t parse_size(const std::string& s, bool* ok);
bool parse_args(int argc, char** argv, Args* a);
void usage();

}  // namespace dray::cli
