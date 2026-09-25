#include "cli/args.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace dray::cli {


// Cap parsing accepts a plain byte count or a K/M/G suffix. GiB vs GB is a real
// trap: "64 GB of RAM" almost always means 64 GiB. We treat bare G as GiB
// (what users mean) and print both units everywhere so the ambiguity never bites.
uint64_t parse_size(const std::string& s, bool* ok) {
    *ok = false;
    if (s.empty()) return 0;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || v < 0) return 0;
    uint64_t mult = 1;
    switch (*end) {
        case 'k': case 'K': mult = 1ull << 10; ++end; break;
        case 'm': case 'M': mult = 1ull << 20; ++end; break;
        case 'g': case 'G': mult = 1ull << 30; ++end; break;
        case 't': case 'T': mult = 1ull << 40; ++end; break;
        case '\0': break;
        default: return 0;
    }
    if (*end == 'b' || *end == 'B') ++end;
    if (*end != '\0') return 0;
    *ok = true;
    return static_cast<uint64_t>(v * static_cast<double>(mult));
}

void usage() {
    std::printf(
        "dray - frontier-scale MoE inference from SSD\n\n"
        "usage:\n"
        "  dray plan      -m <model.gguf> [--cap 4G] [--ctx N] [--batch N] [--kv q4|q8|f16]\n"
        "                    residency plan and RAM curve; reads only the tensor table\n"
        "  dray calibrate -m <model.gguf>\n"
        "                    measure this drive per traffic class; saved beside the model\n"
        "  dray run       -m <model.gguf> [-p PROMPT | --prompt-file F] [-n N] [--cap 4G]\n"
        "                    [--ctx N] [--seed N] [--temp] [--stop S] [--status FILE]\n"
        "  dray batch     -m <model.gguf> --batch N [--prompts FILE] [-n N] [--stop S]\n"
        "                    [--rotate SPAN --state-dir DIR]\n"
        "  dray serve     -m <model.gguf> [--cap 4G] [--ctx N] [--port 8080]\n"
        "                    [--jobs-dir DIR] [--api-key KEY] [--repack DIR]\n"
        "                    OpenAI-compatible API (/v1/chat/completions, SSE);\n"
        "                    admission serialized, one generation at a time.\n"
        "                    --jobs-dir enables background jobs with crash-resume;\n"
        "                    --api-key requires Authorization: Bearer KEY on /v1/*.\n"
        "  diagnostics:      verify | stream | snaptest | repack --out DIR\n\n"
        "  --cap            total resident byte budget. The mandatory floor and KV\n"
        "                   reservation are subtracted from this to yield the expert\n"
        "                   cache; the cap is never exceeded. Default 4G.\n"
        "  --ctx            context length. Costs KV, which competes with the cache.\n"
        "  --kv             KV cache type: f16 (default), q8, q4 (q4/q8 force flash attention)\n"
        "  --gpu            runtime GPU consent: prefill on the GPU, weights stay in RAM\n"
        "  --threads N      compute threads (default: 4 decode, cores-9 prefill)\n"
        "  --prefill-chunk  tokens per prefill pass (default 512 CPU, 2048 GPU)\n"
        "  --force-stream   keep the streaming path even when the model fits\n"
        "  --override-kv    key=int|float|bool|str:value, GGUF metadata override\n"
        "  --repack DIR     use the repacked companion made by `repack --out DIR`\n"
        "  --no-stream      reference path: llama.cpp loads normally (differential tests)\n"
        "  --status         append machine-readable JSONL status for detached runs.\n");
}

bool parse_args(int argc, char** argv, Args* a) {
    if (argc < 2) { a->help = true; return true; }
    a->cmd = argv[1];
    if (a->cmd == "-h" || a->cmd == "--help" || a->cmd == "help") { a->help = true; return true; }
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](std::string* dst) -> bool {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); return false; }
            *dst = argv[++i]; return true;
        };
        // Checked integer parse. atoi returns 0 on garbage with no way to tell,
        // and this file already documented that bug class for env vars (S11/H16)
        // without applying the fix to its own command line: `--ctx 32k` silently
        // became ctx 32 -- a truthy value, so not even the default saved it
        // (2026-08-24 audit).
        auto next_int = [&](long long lo, long long hi, long long* out) -> bool {
            std::string s;
            if (!next(&s)) return false;
            errno = 0;
            char* end = nullptr;
            const long long n = std::strtoll(s.c_str(), &end, 10);
            if (errno != 0 || end == s.c_str() || *end != '\0' || n < lo || n > hi) {
                std::fprintf(stderr, "%s wants an integer in [%lld, %lld], got \"%s\"\n",
                             k.c_str(), lo, hi, s.c_str());
                return false;
            }
            *out = n;
            return true;
        };
        std::string v;
        long long nv = 0;
        if (k == "-m" || k == "--model") { if (!next(&v)) return false; a->model = v; }
        else if (k == "-p" || k == "--prompt") { if (!next(&v)) return false; a->prompt = v; }
        // Long contexts are unreachable via -p: Windows caps a command line
        // near 32 KB, which is ~8k tokens. A file has no such bound.
        else if (k == "--prompt-file") {
            if (!next(&v)) return false;
            std::ifstream pf(v, std::ios::binary);
            if (!pf.good()) { std::fprintf(stderr, "cannot read prompt file %s\n", v.c_str()); return false; }
            std::ostringstream ss; ss << pf.rdbuf(); a->prompt = ss.str();
            if (a->prompt.empty()) { std::fprintf(stderr, "prompt file %s is empty\n", v.c_str()); return false; }
        }
        else if (k == "-n" || k == "--n-predict") { if (!next_int(1, 1 << 24, &nv)) return false; a->n_predict = static_cast<int32_t>(nv); }
        else if (k == "--cap") {
            if (!next(&v)) return false;
            bool ok = false; a->cap = parse_size(v, &ok);
            if (!ok) { std::fprintf(stderr, "bad --cap value: %s\n", v.c_str()); return false; }
        }
        else if (k == "--ctx") { if (!next_int(1, 1 << 24, &nv)) return false; a->n_ctx = static_cast<uint32_t>(nv); }
        else if (k == "--seed") { if (!next_int(0, 4294967295LL, &nv)) return false; a->seed = static_cast<uint32_t>(nv); }
        else if (k == "--temp") { a->greedy = false; }
        else if (k == "--no-stream") { a->no_stream = true; }
        else if (k == "--status") { if (!next(&v)) return false; a->status_path = v; }
        else if (k == "--port") {
            if (!next(&v)) return false;
            const long pn = std::strtol(v.c_str(), nullptr, 10);
            if (pn < 1 || pn > 65535) return false;   // swarm S12: 0 = silent ephemeral bind
            a->port = static_cast<int>(pn);
        }
        else if (k == "--out") { if (!next(&v)) return false; a->out_dir = v; }
        else if (k == "--repack") { if (!next(&v)) return false; a->repack_dir = v; }
        else if (k == "--jobs-dir") { if (!next(&v)) return false; a->jobs_dir = v; }
        else if (k == "--api-key") { if (!next(&v)) return false; a->api_key = v; }
        else if (k == "--batch") { if (!next(&v)) return false; a->n_batch_seq = static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 10)); }
        else if (k == "--prompts") { if (!next(&v)) return false; a->prompts_file = v; }
        else if (k == "--stop") { if (!next(&v)) return false; a->stop_seqs.push_back(v); }
        else if (k == "--rotate") { if (!next_int(1, 1 << 24, &nv)) return false; a->rotate_span = static_cast<int32_t>(nv); }
        else if (k == "--state-dir") { if (!next(&v)) return false; a->state_dir = v; }
        else if (k == "--override-kv") { if (!next(&v)) return false; a->kv_overrides.push_back(v); }
        else if (k == "--gpu") { a->gpu = true; }
        else if (k == "--kv") {
            if (!next(&v)) return false;
            // Refuse a typo here rather than silently pricing f16 downstream:
            // `--kv q4_0` or `--kv Q4` used to run the whole measurement at the
            // f16 default while the operator believed the cache was quantised
            // (2026-08-24 audit).
            if (v != "f16" && v != "q8" && v != "q4") {
                std::fprintf(stderr, "--kv accepts f16, q8 or q4 (got \"%s\")\n", v.c_str());
                return false;
            }
            a->kv = v;
        }
        else if (k == "--prefill-chunk") { if (!next_int(1, 65536, &nv)) return false; a->prefill_chunk = static_cast<int32_t>(nv); }
        else if (k == "--threads" || k == "-t") { if (!next_int(1, 1024, &nv)) return false; a->n_threads = static_cast<int32_t>(nv); }
        else if (k == "--force-stream") { a->force_stream = true; }
        else if (k == "--resident") { a->resident = true; }
        else if (k == "-h" || k == "--help") { a->help = true; return true; }
        else { std::fprintf(stderr, "unknown argument: %s\n", k.c_str()); return false; }
    }
    return true;
}

}  // namespace dray::cli
