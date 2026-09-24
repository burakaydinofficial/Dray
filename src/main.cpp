// dray - run frontier-scale MoE models from SSD on a polite slice of RAM.
//
// Subcommands:
//   plan       read the GGUF tensor table, print the residency plan and RAM curve.
//              Does not load the model, so it is instant even on a 594 GB file.
//   calibrate  measure this drive's realized bandwidth, per traffic class.
//   run        generate, with the live readout.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backend/stream_buffer.h"
#include "cache/expert_cache.h"
#include "calib/calibrate.h"
#include "io/storage.h"
#include "mem/accountant.h"
#include "models/profile.h"
#include "plan/residency.h"
#include "server/engine.h"
#include "tools/repack.h"
#include "cache/snapshot_cache.h"
#include "config/env_report.h"
#include "report/readout.h"

#include "llama.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

// 64-bit file seek. std::fseek's offset is a 32-bit long on Windows, which silently
// wraps on the tens-of-GB offsets every model here uses.
bool dray_fseek64(std::FILE* f, uint64_t off) {
#if defined(_WIN32)
    return _fseeki64(f, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

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
    std::string repack_dir;      // run/serve: apply a repacked companion
    std::string jobs_dir;        // serve: checkpoint+resume dir (empty = off)
    std::string api_key;         // serve: bearer token for /v1/* (empty = open, localhost-only)
    bool     help = false;
};

void usage() {
    std::printf(
        "dray - frontier-scale MoE inference from SSD\n\n"
        "usage:\n"
        "  dray plan      -m <model.gguf> [--cap 4G] [--ctx N]\n"
        "  dray calibrate -m <model.gguf>\n"
        "  dray run       -m <model.gguf> [-p PROMPT] [-n N] [--cap 4G] [--ctx N]\n"
        "                    [--seed N] [--temp] [--status FILE]\n"
        "  dray serve     -m <model.gguf> [--cap 4G] [--ctx N] [--port 8080]\n"
        "                    [--jobs-dir DIR] [--api-key KEY] [--repack DIR]\n"
        "                    OpenAI-compatible API (/v1/chat/completions, SSE);\n"
        "                    admission serialized, one generation at a time.\n"
        "                    --jobs-dir enables background jobs with crash-resume;\n"
        "                    --api-key requires Authorization: Bearer KEY on /v1/*.\n\n"
        "  --cap     total resident byte budget. The mandatory floor and KV\n"
        "            reservation are subtracted from this to yield the expert\n"
        "            cache; the cap is never exceeded. Default 4G.\n"
        "  --ctx     context length. Costs KV, which competes with the cache.\n"
        "  --status  append machine-readable JSONL status for detached runs.\n");
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

int cmd_plan(const Args& a) {
    std::string err;
    uint32_t ctx = a.n_ctx ? a.n_ctx : 32768;
    // plan --batch N: the admission maths at width N, instantly, from the
    // tensor table alone -- the same floor and refusal the engine would
    // compute at load, without touching the drive for the weights.
    const uint32_t plan_seq = a.n_batch_seq > 1 ? a.n_batch_seq : 1;
    // --kv must reach the planner too: the floor it prints is the floor the
    // engine will reserve (this was silently dropped when --kv landed).
    const double plan_kv_el = a.kv == "q4" ? 0.5625 : (a.kv == "q8" ? 1.0625 : 2.0);
    dray::plan::Plan p = dray::plan::build_plan(a.model, a.cap, ctx, &err, plan_seq, plan_kv_el);
    if (!err.empty()) { std::fprintf(stderr, "plan failed: %s\n", err.c_str()); return 1; }
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    std::printf("%s\n", p.report().c_str());
    // T28: the README says plan output names the pulled levers; make it true.
    {
        const std::string envs_plan = dray::config::env_report();
        if (!envs_plan.empty()) std::printf("%s\n", envs_plan.c_str());
    }

// Batching PROJECTION (Invariant 8: projections are labeled, never quoted
    // as results; and GRADED -- the K3 measurement confirmed the sharing at
    // B=8 within 20% and FALSIFIED the old feasibility column, which counted
    // only KV state: the per-step expert-UNION working set must also fit the
    // cache or the step thrashes within itself, measured at B=32/28G reading
    // 1.8x its own cold bound). Pure arithmetic from this file's own numbers. The mechanism:
    // the unconditional stream is read ONCE per decode step regardless of how
    // many sequences share it, while routed-expert overlap between sequences
    // is the birthday-ish 1-(1-k/E)^B, near-nil at frontier sparsity. So per-
    // sequence bytes fall as uncond/B + routed*u(B)/B, and at large B every
    // step converges to one full-model scan shared by all sequences: decode
    // becomes prefill. Cold bound (h_routed = 0, uncond fully streaming);
    // caching only improves it. Feasibility: each extra sequence carries its
    // own KV + recurrent state inside the cap.
    if (p.n_moe_layers > 0 && p.n_experts > 0 && p.n_expert_used > 0) {
        const double E = static_cast<double>(p.n_experts);
        const double k = static_cast<double>(p.n_expert_used);
        const double uncond = static_cast<double>(p.unconditional_bytes);
        const double routed = static_cast<double>(p.routed_bytes);
        const uint64_t seq_state = p.floor.kv_cache + p.floor.recurrent_state;
        const uint64_t f_total = p.floor.total();
        const uint64_t spare = a.cap > f_total ? a.cap - f_total : 0;
        const uint64_t b_max = seq_state ? 1 + spare / seq_state : 1;
        const double per1 = uncond + routed * (1.0 - std::pow(1.0 - k / E, 1.0));
        std::printf("\nbatched-decode projection (cold bound; PROJECTION, not a measurement):\n");
        std::printf("  %-6s %14s %14s %10s %s\n",
                    "batch", "bytes/tok/seq", "step bytes", "aggregate", "fits this cap?");
        for (uint64_t B : { 1ull, 2ull, 4ull, 8ull, 16ull, 32ull, 64ull }) {
            const double u = 1.0 - std::pow(1.0 - k / E, static_cast<double>(B));
            const double step = uncond + routed * u;
            const double per_seq = step / static_cast<double>(B);
            // J2: the previous "union working set" verdict compared one layer's
            // union against the WHOLE cache budget (n_layers too permissive)
            // and spent the same spare bytes on state and cache at once -- it
            // could never fire on the measured B=32 thrash it was added for.
            // No validated intra-step thrash model exists yet, so the honest
            // column is DISCLOSURE: the per-B cache remainder (state funded
            // first, per row) and the union/cache pressure ratio, with the
            // measured anchor in the footnote. "measure" is the verdict where
            // pressure is material; fake precision was worse than none.
            const uint64_t extra = (B - 1) * seq_state;
            const uint64_t cache_b = a.cap > f_total + extra ? a.cap - f_total - extra : 0;
            const double union_all = routed * u;
            const bool state_ok = B <= b_max;
            const double pressure = cache_b ? union_all / static_cast<double>(cache_b) : 1e9;
            std::printf("  %-6llu %11.2f GB %11.2f GB %9.2fx %s\n",
                        static_cast<unsigned long long>(B), per_seq / 1e9, step / 1e9,
                        per1 / per_seq,
                        !state_ok ? "no (per-seq state)"
                                  : (pressure > 4.0 ? "measure (union >> cache)" : "yes"));
        }
        std::printf("  per-seq state %.2f GB (KV + recurrent); max batch at this cap ~%llu\n",
                    seq_state / 1e9, static_cast<unsigned long long>(b_max));
        std::printf("  at large batch every step is one full-model scan shared by all\n"
                    "  sequences: decode converges to prefill and the SEQUENTIAL ceiling\n"
                    "  (calibrate's stream class) becomes the relevant one.\n");
        // plan --batch N: preview the engine's own admission verdict at this
        // width -- the measured bounds (below 1x the widest union region is
        // certain death, 2x proven clean, between is the untested band).
        if (plan_seq > 1) {
            const double uw = E > 0.0
                ? E * (1.0 - std::pow(1.0 - k / E, static_cast<double>(plan_seq))) : k;
            uint64_t max_slot = 0;
            for (const auto& sc : p.slot_classes) max_slot = std::max(max_slot, sc.slot_bytes);
            const uint64_t region = static_cast<uint64_t>(std::ceil(uw)) * max_slot;
            const uint64_t remainder = a.cap > f_total ? a.cap - f_total : 0;
            const char* verdict = region > remainder
                ? "REFUSED at load (one union region exceeds the cache remainder)"
                : (region * 2 > remainder
                       ? "admits with WARNING (1x-2x the union region: untested band)"
                       : "admits");
            std::printf("  batch %u admission preview: floor %.2f GB, cache remainder %.2f GB,\n"
                        "  widest union region %.2f GB -> %s\n",
                        plan_seq, f_total / 1e9, remainder / 1e9, region / 1e9, verdict);
        }
    }
    if (!p.feasible) {
        std::printf("\nREFUSED: %s\n", p.refusal.c_str());
        return 2;
    }
    return 0;
}

int cmd_calibrate(const Args& a) {
    auto backend = dray::io::make_backend(64);
    if (!backend) { std::fprintf(stderr, "no storage backend\n"); return 1; }
    dray::io::FileId f = backend->open(a.model);
    if (f == dray::io::kInvalidFile) {
        std::fprintf(stderr, "cannot open %s in uncached mode\n", a.model.c_str());
        return 1;
    }
    std::printf("backend: %s\n", backend->describe().c_str());
    dray::calib::Options opt;
    dray::calib::Calibration c = dray::calib::calibrate(*backend, f, backend->size(f), opt);
    std::printf("%s\n", c.report().c_str());
    // T5: persist the summary so run/serve can quote real ceilings (with age).
    const std::string calp = a.model + ".lzcal.json";
    std::string serr;
    if (dray::calib::save_calibration(c, calp, &serr)) {
        std::printf("calibration saved: %s\n", calp.c_str());
    } else {
        std::fprintf(stderr, "calibration NOT saved: %s\n", serr.c_str());
    }
    backend->close(f);
    return 0;
}

// Proves the streaming geometry against the real model without loading it: reads
// real expert slices through the uncached backend into slab slots, then re-reads
// the same file ranges through an ordinary buffered path and compares bytes.
//
// This is the check that catches the failure mode that matters most here -- wrong
// offsets or strides do not crash, they return plausible-looking garbage weights.
// gate/up/down have DIFFERENT per-expert strides, so an evenly-divided slot would
// pass a size check and still be wrong.
int cmd_verify(const Args& a) {
    std::string err;
    dray::plan::Plan p = dray::plan::build_plan(a.model, a.cap, a.n_ctx ? a.n_ctx : 512, &err);
    if (!err.empty()) { std::fprintf(stderr, "plan failed: %s\n", err.c_str()); return 1; }
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    if (p.slot_classes.empty()) { std::fprintf(stderr, "no MoE layers found\n"); return 1; }

    auto backend = dray::io::make_backend(32);
    // EVERY shard, not just the one named on the command line. Routed experts
    // live wherever the split put them -- on DeepSeek V4 Flash all 129 of them
    // are in shards 2 and 3, so a shard-1-only check reported "INCONCLUSIVE --
    // nothing compared" while verifying precisely nothing.
    std::vector<dray::io::FileId> handles(p.shard_paths.size(), dray::io::kInvalidFile);
    std::vector<uint64_t> sizes(p.shard_paths.size(), 0);
    for (size_t i = 0; i < p.shard_paths.size(); ++i) {
        handles[i] = backend->open(p.shard_paths[i]);
        if (handles[i] != dray::io::kInvalidFile) sizes[i] = backend->size(handles[i]);
    }
    dray::io::FileId f = handles.empty() ? dray::io::kInvalidFile : handles[0];
    if (f == dray::io::kInvalidFile) {
        std::fprintf(stderr, "cannot open %s uncached\n", a.model.c_str());
        return 1;
    }
    const dray::io::Alignment al = backend->alignment(f);
    std::printf("backend: %s\n", backend->describe().c_str());

    // Walk every routed tensor that lives in THIS shard, and for a sample of
    // experts compare an aligned uncached read against a buffered reference.
    size_t checked = 0, mismatched = 0, skipped = 0;
    std::vector<uint8_t> direct, reference;

    std::FILE* ref = std::fopen(a.model.c_str(), "rb");
    if (!ref) { std::fprintf(stderr, "cannot open reference handle\n"); backend->close(f); return 1; }

    for (const dray::plan::TensorInfo& t : p.tensors) {
        if (t.cls != dray::plan::TensorClass::RoutedExpert) continue;
        if (p.n_experts == 0 || t.bytes == 0) continue;
        const size_t si = static_cast<size_t>(t.shard >= 0 ? t.shard : 0);
        if (si >= handles.size() || handles[si] == dray::io::kInvalidFile) { ++skipped; continue; }
        if (t.offset + t.bytes > sizes[si]) { ++skipped; continue; }
        const dray::io::FileId fh = handles[si];

        const uint64_t stride = t.bytes / p.n_experts;
        if (stride == 0) { ++skipped; continue; }

        for (uint32_t e = 0; e < p.n_experts; e += (p.n_experts / 4 ? p.n_experts / 4 : 1)) {
            const uint64_t want_off = t.offset + static_cast<uint64_t>(e) * stride;

            // Widen to the alignment boundary and slice the interior back out --
            // GGUF tensor offsets are aligned to general.alignment (32 by default),
            // which is NOT the device's I/O alignment.
            const uint64_t lo = (want_off / al.offset) * al.offset;
            const uint64_t head = want_off - lo;
            uint64_t span = head + stride;
            span = ((span + al.length - 1) / al.length) * al.length;
            if (lo + span > sizes[si]) { ++skipped; continue; }
            if (span > (64u << 20)) { ++skipped; continue; }

            void* buf = dray::mem::aligned_alloc_host(static_cast<size_t>(span), al.memory);
            if (!buf) { ++skipped; continue; }

            dray::io::ReadRequest r;
            r.file = fh; r.offset = lo; r.length = static_cast<uint32_t>(span);
            r.dst = buf; r.tag = 1;
            bool ok = backend->submit(&r, 1) == 1;
            if (ok) {
                dray::io::Completion c;
                ok = backend->poll(&c, 1, 1) == 1 && c.ok(static_cast<uint32_t>(span));
            }
            if (ok) {
                reference.assign(static_cast<size_t>(stride), 0);
                ok = read_reference_uncached(p.shard_paths[si], want_off, stride, reference.data());
                if (ok) {
                    const uint8_t* got = static_cast<const uint8_t*>(buf) + head;
                    if (std::memcmp(got, reference.data(), static_cast<size_t>(stride)) != 0) {
                        ++mismatched;
                        std::printf("  MISMATCH %s expert %u @ %llu (%llu B)\n", t.name.c_str(),
                                    e, static_cast<unsigned long long>(want_off),
                                    static_cast<unsigned long long>(stride));
                    }
                    ++checked;
                } else { ++skipped; }
            } else { ++skipped; }
            dray::mem::aligned_free_host(buf, span);
        }
    }

    // SECOND PASS: the unconditional weights. Routed experts are sampled above
    // because there are hundreds of thousands of slices, but the unconditional
    // set -- attention projections, shared experts, embedding, output -- is read
    // on EVERY token, so a misread there corrupts every token rather than the
    // ones that happen to route through a bad expert. Checking only the routed
    // half left the more damaging half unchecked.
    size_t un_checked = 0, un_mismatched = 0, un_skipped = 0;
    for (const dray::plan::TensorInfo& t : p.tensors) {
        if (t.cls != dray::plan::TensorClass::UnconditionalBulk) continue;
        if (t.bytes == 0) continue;
        const size_t si = static_cast<size_t>(t.shard >= 0 ? t.shard : 0);
        if (si >= handles.size() || handles[si] == dray::io::kInvalidFile) { ++un_skipped; continue; }
        if (t.offset + t.bytes > sizes[si]) { ++un_skipped; continue; }

        // A prefix is enough: these are contiguous reads, so an offset or
        // alignment error shows up in the first bytes. Bounded so output.weight
        // does not cost a gigabyte of comparison.
        const uint64_t want = std::min<uint64_t>(t.bytes, 4ull << 20);
        const uint64_t lo = (t.offset / al.length) * al.length;
        const uint64_t head = t.offset - lo;
        uint64_t span = ((head + want + al.length - 1) / al.length) * al.length;
        if (lo + span > sizes[si]) { ++un_skipped; continue; }

        void* buf = dray::mem::aligned_alloc_host(static_cast<size_t>(span), al.memory);
        if (!buf) { ++un_skipped; continue; }
        dray::io::ReadRequest r;
        r.file = handles[si]; r.offset = lo; r.length = static_cast<uint32_t>(span);
        r.dst = buf; r.tag = 1;
        if (backend->submit(&r, 1) == 1) {
            dray::io::Completion c;
            if (backend->poll(&c, 1, 1) == 1 && c.ok(static_cast<uint32_t>(span))) {
                std::vector<uint8_t> ref(static_cast<size_t>(want), 0);
                if (read_reference_uncached(p.shard_paths[si], t.offset, want, ref.data())) {
                    if (std::memcmp(static_cast<const uint8_t*>(buf) + head, ref.data(),
                                    static_cast<size_t>(want)) != 0) {
                        ++un_mismatched;
                        std::printf("  MISMATCH %s @ %llu (%llu B prefix)\n", t.name.c_str(),
                                    static_cast<unsigned long long>(t.offset),
                                    static_cast<unsigned long long>(want));
                    }
                    ++un_checked;
                } else { ++un_skipped; }
            } else { ++un_skipped; }
        } else { ++un_skipped; }
        dray::mem::aligned_free_host(buf, span);
    }

    for (dray::io::FileId h : handles) {
        if (h != dray::io::kInvalidFile) backend->close(h);
    }

    std::printf("\nunconditional weight verification  (read every token)\n");
    std::printf("  tensors compared   %zu\n", un_checked);
    std::printf("  mismatches         %zu\n", un_mismatched);
    std::printf("  skipped            %zu\n", un_skipped);

    std::printf("\nexpert slice verification\n");
    std::printf("  shard              %s\n", a.model.c_str());
    std::printf("  slices compared    %zu\n", checked);
    std::printf("  mismatches         %zu\n", mismatched);
    std::printf("  skipped            %zu  (other shard, or unaligned span too large)\n", skipped);
    // BOTH passes decide the verdict. An earlier edit added the unconditional
    // pass, printed its mismatch count, and left the verdict reading only the
    // routed counters -- so a run in which every unconditional weight differed
    // still printed PASS and returned 0. A verifier whose second half cannot
    // fail is the exact defect its first half was written to fix (2026-08-24).
    const size_t all_checked = checked + un_checked;
    const size_t all_bad     = mismatched + un_mismatched;
    std::printf("  verdict            %s\n",
                (all_checked > 0 && all_bad == 0) ? "PASS - uncached reads land byte-identical"
                                                  : (all_checked == 0 ? "INCONCLUSIVE - nothing compared"
                                                                      : "FAIL"));
    return (all_checked > 0 && all_bad == 0) ? 0 : 1;
}

// Exercises the real decode hot path against the real model, without the matmul:
// for each MoE layer, pick the experts a router would pick, ensure() them through
// the cache (slab + LRU + refcounts + multi-shard reads), verify a sample against a
// buffered reference, release, repeat. This is everything the streaming engine does
// except hand the bytes to ggml.
int cmd_stream(const Args& a) {
    std::string err;
    const uint32_t ctx = a.n_ctx ? a.n_ctx : 4096;
    dray::plan::Plan p = dray::plan::build_plan(a.model, a.cap, ctx, &err);
    if (!err.empty()) { std::fprintf(stderr, "plan failed: %s\n", err.c_str()); return 1; }
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    if (!p.feasible) { std::printf("REFUSED\n  %s\n", p.refusal.c_str()); return 2; }

    auto backend = dray::io::make_backend(64);
    std::printf("backend: %s\n", backend->describe().c_str());
    std::printf("shards:  %zu\n", p.shard_paths.size());

    dray::mem::Accountant acct(a.cap);
    // Reserve the floor first: the cap means total resident bytes, so the cache
    // gets what is left, never the whole cap.
    acct.reserve(dray::mem::Category::RouterGates, p.floor.router_gates);
    acct.reserve(dray::mem::Category::NormsAndBiases, p.floor.norms_biases);
    acct.reserve(dray::mem::Category::KvCache, p.floor.kv_cache);
    acct.reserve(dray::mem::Category::RecurrentState, p.floor.recurrent_state);
    acct.reserve(dray::mem::Category::PrefillActivation, p.floor.prefill_activation);
    acct.reserve(dray::mem::Category::ComputeScratch, p.floor.compute_scratch);
    acct.reserve(dray::mem::Category::IoStaging, p.floor.io_staging);

    dray::cache::ExpertCache cache(acct, *backend, p, 4096);
    if (!cache.valid()) {
        std::fprintf(stderr, "expert cache unavailable: %s\n", cache.error().c_str());
        return 1;
    }


    const int32_t k = static_cast<int32_t>(p.n_expert_used);
    const int32_t tokens = a.n_predict > 0 ? a.n_predict : 8;
    uint64_t rng = 0x9E3779B97F4A7C15ull ^ a.seed;
    auto next = [&rng]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };

    size_t verified = 0, bad = 0, stalls = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (int32_t tok = 0; tok < tokens; ++tok) {
        for (const dray::plan::LayerSlotClass& sc : p.slot_classes) {
            std::vector<int32_t> ids(static_cast<size_t>(k));
            for (int32_t i = 0; i < k; ++i) {
                ids[static_cast<size_t>(i)] = static_cast<int32_t>(next() % p.n_experts);
            }
            dray::cache::Remap rm = cache.ensure(sc.layer, ids.data(), ids.size());
            if (!rm.complete) { ++stalls; continue; }

            // Spot-check the first slot of the first layer of each token.
            if (verified < 64 && sc.layer == p.slot_classes.front().layer) {
                dray::cache::LayerCache* lc = cache.layer_cache(sc.layer);
                const uint8_t* got = static_cast<const uint8_t*>(lc->slot_ptr(rm.slot_of_selected[0]));
                if (got) {
                    // Compare the gate slice, which sits at the front of the slot.
                    for (const dray::plan::TensorInfo& t : p.tensors) {
                        if (t.layer != sc.layer || t.cls != dray::plan::TensorClass::RoutedExpert) continue;
                        if (t.name.find("ffn_gate_exps") == std::string::npos) continue;
                        const uint64_t stride = t.bytes / p.n_experts;
                        const uint64_t off = t.offset + static_cast<uint64_t>(ids[0]) * stride;
                        if (t.shard >= 0 && static_cast<size_t>(t.shard) < p.shard_paths.size()) {
                            std::vector<uint8_t> ref(static_cast<size_t>(stride));
                            if (read_reference_uncached(p.shard_paths[static_cast<size_t>(t.shard)],
                                                        off, stride, ref.data())) {
                                if (std::memcmp(got, ref.data(), ref.size()) != 0) ++bad;
                                ++verified;
                            }
                        }
                        break;
                    }
                }
            }
            cache.release(sc.layer, rm);
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const double gb = static_cast<double>(cache.bytes_read()) / 1e9;
    std::printf("\n%s\n", cache.report().c_str());
    std::printf("\nstreaming simulation (uniform-random routing, no matmul)\n");
    std::printf("  tokens                %d over %u MoE layers\n", tokens, p.n_moe_layers);
    std::printf("  slot spot-checks      %zu verified, %zu WRONG\n", verified, bad);
    std::printf("  stalls (all pinned)   %zu\n", stalls);
    std::printf("  read                  %.2f GB in %.2f s = %.2f GB/s\n", gb, secs, secs > 0 ? gb / secs : 0.0);
    std::printf("  per token             %.2f GB  ->  %.2f s/tok at this rate\n",
                gb / tokens, secs / tokens);
    // T21: a wedged run (stalls) must not print PASS beside its own stall count.
    const bool pass = bad == 0 && verified > 0 && stalls == 0;
    std::printf("  verdict               %s\n",
                pass ? "PASS - cached slots match the file"
                     : (stalls > 0 ? "FAIL - stalled (all pinned); see stalls count" : "FAIL"));
    std::printf("\n%s\n", acct.report().c_str());
    return pass ? 0 : 1;
}

// Reconciles the resident-byte ledger against what the OS says the process holds,
// and REPORTS the result either way.
//
// Invariant 1 -- the cap means total resident bytes -- is this project's central
// claim, so a run that quietly exceeded it would be precisely the dishonesty the
// readout rules exist to prevent. Called after the context is built and again
// during generation, because the gap does not appear all at once: llama.cpp's
// buffers land at context creation, while staging and fragmentation accumulate.
void reconcile_cap(dray::backend::Streamer& s, uint64_t cap, const char* when) {
    const size_t rss = dray::mem::Accountant::process_rss();
    const size_t com = dray::mem::Accountant::process_committed();
    if (rss == 0 && com == 0) {
        std::printf("resident (%s): unknown -- cannot verify the cap here\n", when);
        return;   // Invariant 6: unobtainable is reported unknown, never defaulted
    }
    // Bind on the WORSE of the two. Committed memory the allocator is holding is
    // unavailable to the rest of the system even when it is out of our working set,
    // and a cap that ignores it reports success while the machine is short.
    const uint64_t worst = std::max<uint64_t>(rss, com);
    const uint64_t budget = s.rebudget_against_rss(worst);

    char cbuf[32];
    if (com) std::snprintf(cbuf, sizeof(cbuf), "%.2f GB", com / 1e9);
    else     std::snprintf(cbuf, sizeof(cbuf), "unknown");
    std::printf("resident %.2f GB, committed %s, of %.2f GB cap (%s); cache budget %.2f GB%s\n",
                rss / 1e9, cbuf, cap / 1e9, when, budget / 1e9,
                s.over_cap() ? "  ** OVER CAP **" : "");
    // Where it actually went. A cap that leaves no cache is a fact the user has to
    // be able to act on -- "budget 0.18 GB" without the breakdown tells them the
    // engine is broken rather than that the floor is too big for what they asked.
    std::printf("%s\n", s.accountant_report().c_str());
}

// THE REFERENCE PATH, deliberately NOT consolidated onto the server engine: a
// reference that shares the code it checks can only confirm that code's own
// bugs (the same principle as read_reference_uncached). --no-stream lands here
// and nowhere else; the body is the original cmd_run, byte-for-byte.
// S38 note: a.no_stream is ALWAYS true in here, so the `if (!a.no_stream)`
// blocks below are statically dead -- kept verbatim for now because this body
// is frozen as the reference; excising them is queued as its own careful pass
// with the difftest as the gate.
int cmd_run_reference(const Args& a) {
    // The plan is built first and printed, because on this design the plan is the
    // single best predictor of what the user is about to experience.
    std::string err;
    uint32_t ctx = a.n_ctx ? a.n_ctx : 32768;
    dray::plan::Plan p = dray::plan::build_plan(a.model, a.cap, ctx, &err);
    if (!err.empty()) { std::fprintf(stderr, "plan failed: %s\n", err.c_str()); return 1; }
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    if (!a.repack_dir.empty()) {
        std::string rerr;
        if (!dray::tools::repack_apply(&p, a.repack_dir, &rerr)) {
            std::fprintf(stderr, "repack apply failed: %s\n", rerr.c_str());
            return 1;
        }
    }
    std::printf("%s\n", p.report().c_str());
    if (!p.feasible) { std::printf("\nREFUSED: %s\n", p.refusal.c_str()); return 2; }

    // NEVER refuse a model for not fitting in RAM. Not fitting is the premise of
    // this project, not an error condition: every target is 143-594 GB and the cap
    // is 4-16 GB by design. A "model too large" refusal here would make dray
    // just another engine that runs what already fits, which is a solved problem
    // nobody needs solved again.
    //
    // The only correct response to "it does not fit" is to stream it.

    llama_backend_init();

    // Route EVERY model weight through the streaming buffer type. Nothing is
    // resident except the mandatory floor; the rest is read from disk, uncached,
    // immediately before the node that needs it.
    dray::mem::Accountant acct(a.cap);
    acct.reserve(dray::mem::Category::KvCache, p.floor.kv_cache);
    acct.reserve(dray::mem::Category::RecurrentState, p.floor.recurrent_state);
    acct.reserve(dray::mem::Category::ComputeScratch, p.floor.compute_scratch);

    dray::backend::Config scfg;
    scfg.cap = a.cap;
    scfg.queue_depth = 64;
    // The metadata path is the default: it reads nothing at load, and dropping the
    // landing buffer it made necessary is what puts a 4 GiB cap within reach.
    // DRAY_SLOW_LOAD=1 falls back to llama.cpp's file reader, useful only for
    // bisecting a suspected difference between the two paths.
    // Reference mode MUST take the file path: the metadata path deliberately reads
    // nothing, so with the buffer-type override also disabled llama.cpp would hold no
    // weights at all and emit confident garbage -- which is exactly what it did, and
    // briefly looked like evidence about llama.cpp rather than about this flag.
    const bool fast_load = [&] {
        if (a.no_stream) return false;
        const char* v = std::getenv("DRAY_SLOW_LOAD");
        return !(v && v[0] == '1');
    }();
    scfg.slow_load = !fast_load;
    scfg.no_compact = [] {
        const char* v = std::getenv("DRAY_NO_COMPACT");
        return v && v[0] == '1';
    }();
    dray::backend::Streamer streamer(acct, p, scfg);
    if (!streamer.valid()) {
        std::fprintf(stderr, "streamer unavailable: %s\n", streamer.error().c_str());
        llama_backend_free();
        return 1;
    }

    // The pattern is a REGEX, not a glob: a bare "*" is error_badrepeat. ".*"
    // matches every tensor; the streamer then decides per tensor whether it is
    // floor (pinned) or streamed, from the classification the planner already did.
    const llama_model_tensor_buft_override overrides[] = {
        { ".*", streamer.buft() },
        { nullptr, nullptr },
    };

    llama_model_params mp = llama_model_default_params();
    // REFERENCE MODE. Skips the override entirely, so llama.cpp loads and holds the
    // model the ordinary way and our streamer is never consulted.
    //
    // This exists because the project had no ground truth. K3 and Qwen3.8 cannot run
    // through stock llama.cpp on this machine, so for every correctness question the
    // best available evidence was "the output looks wrong" -- never what it should
    // have been. On a model small enough to hold, --no-stream and the streaming path
    // differ ONLY in where weights come from, so any difference in the tokens they
    // emit is ours. Small models are useless as a target and indispensable as a
    // control; conflating those two things cost most of a day.
    if (!a.no_stream) mp.tensor_buft_overrides = overrides;
    // Invariant 2: tensor data is never mmap'd. While a mapping exists its
    // page-cache footprint cannot be bounded, which makes the memory cap
    // unenforceable -- a correctness argument, not a benchmark one.
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;

    // Two ways in. The file path reads the whole model into a landing buffer we
    // then discard -- ~5 min per start, and it forces that buffer to be as large as
    // the biggest tensor, which is what makes a 4 GiB cap infeasible. The metadata
    // path reads nothing: llama_model_loader::load_all_data returns immediately when
    // it has no files.
    //
    // Three llama.cpp fixes were needed to make the metadata path usable, all on our
    // branch: buft_for_tensor returning null for architecturally-unused tensors made
    // that branch assert instead of skip; done_getting_tensors compared against
    // weights_map, which is empty without files; and an OPTIONAL tensor absent from
    // the metadata was invented at a defaulted F32 rather than reported missing --
    // which for a fused blk.N.ffn_gate_up_exps.weight (a tensor that exists only
    // after load-time fusion, never in any GGUF) meant demanding an F32-sized
    // allocation of a fused expert tensor.
    llama_model* model = nullptr;
    dray::backend::MergedMetadata meta;
    if (fast_load) {
        std::string merr;
        meta = dray::backend::merge_shard_metadata(p, &merr);
        if (!meta.gguf) {
            std::fprintf(stderr, "metadata merge failed: %s\n", merr.c_str());
            llama_backend_free();
            return 1;
        }
        std::printf("metadata: %lld tensors across %zu shards, nothing read\n",
                    static_cast<long long>(meta.n_tensors), p.shard_paths.size());
        // Deliberately does nothing: every streamed tensor's bytes stay on disk
        // until a node needs them.
        auto no_load = [](ggml_tensor*, void*) {};
        model = llama_model_init_from_user(meta.gguf, no_load, nullptr, mp);
    } else {
        std::printf("loading (direct I/O, no mmap)...\n");
        model = llama_model_load_from_file(a.model.c_str(), mp);
    }
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        dray::backend::free_merged_metadata(meta);
        llama_backend_free();
        return 1;
    }

    // Integrity check before generating anything. Cheap (the floor is small) and it
    // catches the failure mode that has twice reached the user as fluent garbage:
    // a tensor present but wrong, with nothing in the engine able to notice.
    // Reference mode never populates the streamer, so there is nothing to check and
    // nothing that could be wrong -- llama.cpp holds the weights itself.
    if (!a.no_stream) {
        const dray::backend::Streamer::SelfCheck sc = streamer.self_check();
        if (!sc.ok()) {
            std::fprintf(stderr,
                         "\nREFUSING TO GENERATE: floor integrity check failed "
                         "(%zu of %zu resident tensors do not match the file).\n"
                         "Generating now would produce fluent text from wrong weights.\n",
                         sc.mismatched, sc.checked);
            llama_model_free(model);
            dray::backend::free_merged_metadata(meta);
            llama_backend_free();
            return 1;
        }
        std::printf("floor integrity: %zu resident tensors match the file\n", sc.checked);
    }

    // Make the cap bind on TOTAL resident bytes (Invariant 1), not on our share.
    // llama.cpp has now allocated its KV cache, compute buffers and vocab, none of
    // which pass through our accountant -- so this is the first moment the real
    // figure is knowable, and the cache budget shrinks to fit what is left.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctx;
    cp.n_batch = 512;
    // Materialise a node's weights just before it runs, release just after. This
    // is where streaming actually happens: the graph is built normally, and the
    // weights it references appear from disk on demand.
    cp.cb_eval = [](ggml_tensor* t, bool ask, void* ud) -> bool {
        auto* s = static_cast<dray::backend::Streamer*>(ud);
        if (ask) {
            s->materialise(t);
            // ALWAYS true, regardless of whether materialise succeeded. This return
            // value controls BATCHING in ggml_backend_sched, not error reporting:
            //
            //     bool need = callback(t, true);
            //     while (!need && j1 < n_nodes-1) { t = nodes[++j1]; need = callback(t, true); }
            //     compute nodes j0..j1 together
            //
            // Returning false makes sched accumulate the next node into the same
            // batch -- so materialise() runs for node B, clears the protected set
            // and evicts node A's weights, and only then are A and B computed
            // together. That is a use-after-free, and it is what crashed 5-token
            // prefill while a single token survived: it needs memory pressure to
            // trigger. One node per batch is a requirement of this design, not a
            // performance choice.
            //
            // Failures are surfaced through Streamer::failures() instead, which
            // marks the whole run's output untrustworthy.
            return true;
        }
        s->release(t);
        return true;
    };
    cp.cb_eval_user_data = &streamer;
    cp.abort_callback = &dray::backend::Streamer::abort_cb;
    cp.abort_callback_data = &streamer;
    llama_context* lctx = llama_init_from_model(model, cp);
    if (!lctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model); llama_backend_free(); return 1;
    }

    // Reconcile the ledger against the OS now that llama.cpp has allocated its KV
    // cache, compute buffers and vocab -- the first moment the true figure exists.
    // Doing this before the context was created was measuring nothing: RSS read
    // 1.70 GB against a 12.88 GB cap, so it never had anything to correct.
    reconcile_cap(streamer, scfg.cap, "after context");

    // ADMISSION CHECK. Refuse a cap that cannot hold the largest single streamed
    // tensor, and say what would work.
    //
    // This is refusing a CAP, not a model -- the model always streams. But a cache
    // smaller than the biggest tensor cannot run: that tensor is indexed linearly
    // by shape, so unlike experts (compacted) or embedding rows (sliced) there is
    // nothing to take a subset of. Without this the run dies mid-prefill with an
    // access violation, because the failure path points a 0.18 GB tensor at an 8 MB
    // poison buffer and MUL_MAT reads off the end. A clear refusal at load beats a
    // crash 40 layers in.
    if (!a.no_stream) {
        const uint64_t need = streamer.largest_streamed_bytes();
        const uint64_t have = streamer.rebudget_against_rss(
            std::max<uint64_t>(dray::mem::Accountant::process_rss(),
                               dray::mem::Accountant::process_committed()));
        if (need > have) {
            const uint64_t suggest = scfg.cap + (need - have);
            std::fprintf(stderr,
                "\nREFUSED: --cap %.2f GB leaves %.2f GB of cache, but the largest single\n"
                "tensor that must be materialised whole is %.2f GB (%s).\n"
                "Try --cap %.0fG or more. Everything else about this model streams fine;\n"
                "it is this one allocation that does not fit.\n",
                scfg.cap / 1e9, have / 1e9, need / 1e9,
                streamer.largest_streamed_name().c_str(),
                std::ceil(suggest / (1024.0 * 1024.0 * 1024.0)));
            llama_free(lctx); llama_model_free(model); llama_backend_free();
            return 2;
        }
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(a.prompt.size() + 8);
    int32_t n = llama_tokenize(vocab, a.prompt.c_str(), static_cast<int32_t>(a.prompt.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);
    if (n < 0) {
        toks.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, a.prompt.c_str(), static_cast<int32_t>(a.prompt.size()),
                           toks.data(), static_cast<int32_t>(toks.size()), true, true);
    }
    if (n <= 0) {
        std::fprintf(stderr, "tokenization failed\n");
        llama_free(lctx); llama_model_free(model); llama_backend_free(); return 1;
    }
    toks.resize(static_cast<size_t>(n));

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    if (a.greedy) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(a.seed));
    }

    namespace rep = dray::report;
    rep::Readout::Options ropt;
    ropt.model = a.model;
    ropt.jsonl_path = a.status_path;
    rep::Readout ro(ropt);
    ro.set_context("llama.cpp direct-io load", p.report());

    rep::Counters c;
    c.tokens_requested = a.n_predict;
    c.resident_cap = static_cast<int64_t>(a.cap);
    c.context_cap = static_cast<int64_t>(ctx);

    // The terminal line goes to stderr so its carriage returns never chew through
    // the generated text on stdout.
    std::fprintf(stderr, "prefill: %d tokens\n", n);
    c.phase = rep::Phase::Prefill;
    ro.update(c, rep::monotonic_ns());

    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(lctx, batch) != 0) {
        std::fprintf(stderr, "prefill failed\n");
        llama_sampler_free(smpl); llama_free(lctx); llama_model_free(model);
        llama_backend_free(); return 1;
    }

    c.phase = rep::Phase::Decode;
    ro.update(c, rep::monotonic_ns());

    std::string out;
    for (int32_t i = 0; i < a.n_predict; ++i) {
        llama_token t = llama_sampler_sample(smpl, lctx, -1);
        if (llama_vocab_is_eog(vocab, t)) break;

        char buf[256];
        int32_t np = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (np > 0) out.append(buf, static_cast<size_t>(np));

        llama_batch b1 = llama_batch_get_one(&t, 1);
        if (llama_decode(lctx, b1) != 0) { std::fprintf(stderr, "\ndecode failed\n"); break; }
        if (streamer.aborted()) {
            std::fprintf(stderr, "\nABORTED: a weight failed to materialise mid-decode. "
                                 "Everything after this point would be computed against "
                                 "weights that are not there.\n");
            break;
        }

        c.tokens_done = i + 1;
        c.context_len = static_cast<int64_t>(n) + c.tokens_done;
        const size_t rss = dray::mem::Accountant::process_rss();
        // Keep the ledger honest as the run proceeds: allocator retention and
        // staging accumulate during generation, so a single check at load cannot
        // hold the cap for a run measured in days. Bind on the worse figure.
        const size_t com = dray::mem::Accountant::process_committed();
        if (rss) c.rss_bytes = static_cast<int64_t>(rss);
        const uint64_t worst = std::max<uint64_t>(rss, com);
        if (worst) streamer.rebudget_against_rss(worst);
        ro.update(c, rep::monotonic_ns());
        ro.render_terminal(std::cerr);
        ro.write_jsonl();
    }
    ro.end_terminal(std::cerr);

    std::printf("\n%s\n", out.c_str());
    std::fprintf(stderr, "\n%s\n", ro.final_summary().c_str());

    // The streamer's own accounting. Printed unconditionally: if any weight failed
    // to materialise, the generated text above came partly from poison and must not
    // be reported as a result.
    std::fprintf(stderr, "%s\n", streamer.report().c_str());
    if (streamer.failures() > 0) {
        std::fprintf(stderr,
                     "\nOUTPUT NOT TRUSTWORTHY: %llu weights failed to materialise.\n",
                     static_cast<unsigned long long>(streamer.failures()));
    }

    llama_sampler_free(smpl);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// SNAPSHOT ROUND-TRIP PROOF -- the integration test the unit suite registered
// as debt (store/load need a live context; here is one, through the same
// engine_open every consumer uses).
//
// Shape: generate k tokens greedily, capture the state BEFORE decoding the
// pending token and remember that token -- the exact resume shape the job
// layer uses, because a snapshot taken after a decode has no logits and the
// pending-token re-decode is what regenerates them. Continue to the end for
// the direct text; then clear, restore, re-decode the pending token, continue
// again. Greedy determinism makes the verdict binary: the two continuations
// must match token for token, or persistence is broken.

// Batched offline decode: N prompts, lockstep steps, one shared stream. The
// server stays serialized (Invariant 7); this is the throughput operating
// point the projection table models, and the run that grades that table.
int cmd_batch(const Args& a) {
    namespace srv = dray::server;
    if (a.n_batch_seq < 1) { std::fprintf(stderr, "batch: --batch must be >= 1\n"); return 1; }

    std::vector<std::string> prompts;
    if (!a.prompts_file.empty()) {
        std::ifstream pf(a.prompts_file);
        if (!pf.good()) { std::fprintf(stderr, "batch: cannot read %s\n", a.prompts_file.c_str()); return 1; }
        std::string line;
        while (std::getline(pf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) prompts.push_back(line);
        }
        if (prompts.empty()) { std::fprintf(stderr, "batch: %s has no prompts\n", a.prompts_file.c_str()); return 1; }
        if (prompts.size() > a.n_batch_seq && a.rotate_span <= 0) {
            std::fprintf(stderr, "batch: %zu prompts but --batch %u; refusing to truncate"
                                 " (or add --rotate SPAN --state-dir DIR to serve them in cohorts)\n",
                         prompts.size(), a.n_batch_seq);
            return 1;
        }
    } else {
        prompts.assign(a.n_batch_seq, a.prompt);
    }
    if (a.rotate_span > 0 && a.state_dir.empty()) {
        std::fprintf(stderr, "batch: --rotate writes parked state; --state-dir is a required,"
                             " explicit choice (a different device than the model is wise)\n");
        return 1;
    }

    srv::EngineConfig ec;
    ec.model_path = a.model;
    ec.cap = a.cap;
    ec.n_ctx = a.n_ctx;
    ec.repack_dir = a.repack_dir;
    ec.kv_overrides = a.kv_overrides;
    ec.gpu = a.gpu;
    ec.kv_quant = a.kv;
    ec.prefill_chunk = a.prefill_chunk;
    ec.n_threads = a.n_threads;
    ec.force_stream = a.force_stream;
    ec.resident = a.resident;
    ec.plan_to_stdout = true;
    // Rotation: the context funds the COHORT width (--batch); prompts beyond
    // it rotate through. Unrotated: every prompt is resident, as before.
    ec.n_seq = a.rotate_span > 0 ? a.n_batch_seq
                                 : static_cast<uint32_t>(prompts.size());
    std::string err;
    srv::Engine* eng = srv::engine_open(ec, &err);
    if (!eng) { std::fprintf(stderr, "batch: %s\n", err.c_str()); return 1; }

    if (a.rotate_span > 0) {
        // Cohort rotation: N prompts through a --batch-wide window, state
        // parked to --state-dir between residencies. Writes are reported
        // with the same rigor as reads -- the honesty contract covers them.
        srv::RotateParams rp;
        rp.prompts = prompts;
        rp.max_tokens = a.n_predict;
        rp.temperature = a.greedy ? 0.0f : 0.8f;
        rp.seed = a.seed;
        rp.stop = a.stop_seqs;
        rp.span = a.rotate_span;
        rp.state_dir = a.state_dir;
        rp.on_round = [](int32_t round, int32_t cohort, int32_t done) {
            std::fprintf(stderr, "rotate: round %d, cohort %d, %d sequences done\n",
                         round, cohort, done);
        };
        const auto ec_open = srv::engine_counters(eng);
        const auto t0 = std::chrono::steady_clock::now();
        srv::RotateResult rres = srv::engine_generate_rotated(eng, rp);
        const auto ec1 = srv::engine_counters(eng);
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!rres.error.empty()) {
            std::fprintf(stderr, "rotate: %s\n", rres.error.c_str());
            srv::engine_close(eng);
            return 1;
        }
        bool any_bad = false;
        uint64_t total_out = 0;
        for (size_t i = 0; i < rres.seqs.size(); ++i) {
            const auto& r = rres.seqs[i];
            std::printf("--- seq %zu (%d tokens%s%s%s) ---\n%s\n", i, r.tokens_out,
                        r.truncated_by_eog ? ", eog/stop" : "",
                        r.ctx_wall ? ", ctx-wall" : "",
                        r.cancelled ? ", cancelled" : "",
                        r.text.c_str());
            if (!r.error.empty()) std::printf("    ERROR: %s\n", r.error.c_str());
            total_out += static_cast<uint64_t>(r.tokens_out);
            any_bad = any_bad || r.aborted || !r.error.empty();
        }
        const uint64_t bytes_r = ec1.bytes_streamed - ec_open.bytes_streamed;
        const uint64_t fails = ec1.failures - ec_open.failures;
        std::printf("\nrotate summary: %zu seqs in cohorts of %u, %llu residencies, %llu tokens\n",
                    rres.seqs.size(), a.n_batch_seq,
                    static_cast<unsigned long long>(rres.rounds),
                    static_cast<unsigned long long>(total_out));
        std::printf("  model bytes read    %.3f GB\n", bytes_r / 1e9);
        std::printf("  state bytes WRITTEN %.3f GB, read back %.3f GB (%.1f MB/token amortized)\n",
                    rres.state_bytes_written / 1e9, rres.state_bytes_read / 1e9,
                    total_out ? rres.state_bytes_written / 1e6 / static_cast<double>(total_out) : 0.0);
        std::printf("  wall                %.1f s  (%.2f tok/s aggregate; wall is noisy, bytes decide)\n",
                    secs, secs > 0 ? total_out / secs : 0.0);
        if (fails) std::printf("  *** %llu materialise FAILURES -- OUTPUT NOT TRUSTWORTHY ***\n",
                               static_cast<unsigned long long>(fails));
        const bool over = srv::engine_live_over_cap(eng);
        if (over) std::printf("  *** CAP BREACH occurred during this run ***\n");
        std::printf("\n%s\n", srv::engine_accountant_report(eng).c_str());
        srv::engine_close(eng);
        return (!any_bad && fails == 0 && !over) ? 0 : 1;
    }

    srv::BatchParams bp;
    bp.prompts = prompts;
    bp.max_tokens = a.n_predict;
    bp.temperature = a.greedy ? 0.0f : 0.8f;
    bp.seed = a.seed;
    bp.stop = a.stop_seqs;   // battery-caught: the engine implemented stop, the CLI never plumbed it

    const auto ec_open = srv::engine_counters(eng);
    const auto t0 = std::chrono::steady_clock::now();
    bp.on_step = [&](int32_t step, int32_t live) {
        if (step % 8 == 0) {
            const auto ecs = srv::engine_counters(eng);
            std::fprintf(stderr, "batch: step %d, %d live, %.2f GB read\n", step, live,
                         (ecs.bytes_streamed - ec_open.bytes_streamed) / 1e9);
        }
    };
    srv::BatchResult br = srv::engine_generate_batch(eng, bp);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto ec1 = srv::engine_counters(eng);

    if (!br.error.empty()) {
        std::fprintf(stderr, "batch: %s\n", br.error.c_str());
        srv::engine_close(eng);
        return 1;
    }

    uint64_t total_out = 0;
    bool any_bad = false;
    for (size_t s = 0; s < br.seqs.size(); ++s) {
        const auto& r = br.seqs[s];
        std::printf("--- seq %zu (%d tokens%s%s%s) ---\n%s\n", s, r.tokens_out,
                    r.aborted ? ", ABORTED" : "",
                    r.cancelled ? ", cancelled" : "",
                    r.error.empty() ? "" : ", ERROR",
                    r.text.c_str());
        if (!r.error.empty()) std::printf("    error: %s\n", r.error.c_str());
        total_out += static_cast<uint64_t>(r.tokens_out);
        any_bad = any_bad || r.aborted || !r.error.empty();
    }
    // J7: prefill and decode reported SEPARATELY -- the first benchmark
    // divided prefill-inclusive bytes by decode-only steps, inflating every
    // per-step figure by a term that grows with B.
    const uint64_t total_bytes = ec1.bytes_streamed - ec_open.bytes_streamed;
    const uint64_t prefill_bytes = br.prefill_end_bytes > ec_open.bytes_streamed
                                       ? br.prefill_end_bytes - ec_open.bytes_streamed : 0;
    const uint64_t decode_bytes = total_bytes > prefill_bytes ? total_bytes - prefill_bytes : 0;
    const uint64_t fails = ec1.failures - ec_open.failures;
    std::printf("\nbatch summary: %zu seqs, %llu steps, %llu tokens total\n",
                br.seqs.size(), static_cast<unsigned long long>(br.steps),
                static_cast<unsigned long long>(total_out));
    std::printf("  prefill bytes  %.3f GB (all %zu sequences)\n",
                prefill_bytes / 1e9, br.seqs.size());
    std::printf("  decode bytes   %.3f GB  (%.3f GB/step, %.3f GB/token-aggregate)\n",
                decode_bytes / 1e9,
                br.steps ? decode_bytes / 1e9 / static_cast<double>(br.steps) : 0.0,
                total_out ? decode_bytes / 1e9 / static_cast<double>(total_out) : 0.0);
    std::printf("  total bytes    %.3f GB\n", total_bytes / 1e9);
    std::printf("  wall           %.1f s  (%.2f tok/s aggregate; wall is noisy, bytes decide)\n",
                secs, secs > 0 ? total_out / secs : 0.0);
    if (fails) std::printf("  *** %llu materialise FAILURES -- OUTPUT NOT TRUSTWORTHY ***\n",
                           static_cast<unsigned long long>(fails));
    // J4: the LATCHED breach signal, not a point sample of a self-clearing
    // flag -- and the loud line, so text greps see it like cmd_run's.
    const bool over = srv::engine_live_over_cap(eng);
    if (over) std::printf("  *** CAP BREACH occurred during this run ***\n");
    std::printf("\n%s\n", srv::engine_accountant_report(eng).c_str());
    srv::engine_close(eng);
    return (!any_bad && fails == 0 && !over) ? 0 : 1;
}

int cmd_snaptest(const Args& a) {
    namespace cache = dray::cache;
    namespace srv = dray::server;

    srv::EngineConfig ec;
    ec.model_path = a.model;
    ec.cap = a.cap;
    ec.n_ctx = a.n_ctx ? a.n_ctx : 512;
    ec.kv_overrides = a.kv_overrides;
    ec.gpu = a.gpu;
    ec.kv_quant = a.kv;
    ec.prefill_chunk = a.prefill_chunk;
    ec.n_threads = a.n_threads;
    ec.force_stream = a.force_stream;
    ec.resident = a.resident;
    std::string err;
    srv::Engine* eng = srv::engine_open(ec, &err);
    if (!eng) { std::fprintf(stderr, "snaptest: %s\n", err.c_str()); return 1; }
    llama_context* lctx = srv::engine_ctx(eng);
    const llama_vocab* vocab = srv::engine_vocab(eng);

    auto tokenize = [&](const std::string& s) {
        std::vector<llama_token> t(s.size() + 8);
        int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(),
                                   t.data(), (int32_t)t.size(), true, true);
        t.resize(n > 0 ? (size_t)n : 0);
        return t;
    };
    auto greedy_next = [&]() {
        llama_sampler* g = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(g, llama_sampler_init_greedy());
        llama_token t = llama_sampler_sample(g, lctx, -1);
        llama_sampler_free(g);
        return t;
    };
    auto decode1 = [&](llama_token t) {
        llama_batch b = llama_batch_get_one(&t, 1);
        return llama_decode(lctx, b) == 0;
    };

    std::vector<llama_token> toks = tokenize(a.prompt);
    if (toks.empty()) { srv::engine_close(eng); return 1; }
    llama_batch pre = llama_batch_get_one(toks.data(), (int32_t)toks.size());
    if (llama_decode(lctx, pre) != 0) { srv::engine_close(eng); return 1; }

    uint64_t h = cache::prefix_hash(toks.data(), toks.size());
    const int32_t k = 8, tail = 8;

    // k-1 tokens decoded; the k-th becomes the pending token at the capture point.
    for (int32_t i = 0; i < k - 1; ++i) {
        llama_token t = greedy_next();
        h = cache::prefix_hash_extend(h, t);
        if (!decode1(t)) { srv::engine_close(eng); return 1; }
    }
    const llama_token pending = greedy_next();

    cache::CompatStamp stamp;
    if (!cache::make_compat_stamp(a.model, lctx, "snaptest-build", &stamp, &err)) {
        std::fprintf(stderr, "snaptest: stamp: %s\n", err.c_str());
        srv::engine_close(eng);
        return 1;
    }
    cache::Config cc;
    cc.dir = std::filesystem::temp_directory_path() / "dray_snaptest";
    std::filesystem::remove_all(cc.dir);
    cc.budget_bytes = 1ull << 30;
    cc.acct = srv::engine_accountant(eng);   // F4: gated path = tested path
    cache::SnapshotCache snap(cc, stamp);
    if (!snap.open(&err)) { std::fprintf(stderr, "snaptest: %s\n", err.c_str()); srv::engine_close(eng); return 1; }
    if (!snap.store(h, toks.size() + k - 1, lctx, 0, cache::Retention::Pinned, &err)) {
        std::fprintf(stderr, "snaptest: store: %s\n", err.c_str());
        srv::engine_close(eng);
        return 1;
    }
    std::printf("stored: %.1f MB pinned\n", snap.stats().pinned_bytes / 1e6);

    // Direct continuation: pending + tail more.
    std::vector<llama_token> direct;
    direct.push_back(pending);
    if (!decode1(pending)) { srv::engine_close(eng); return 1; }
    for (int32_t i = 0; i < tail; ++i) {
        llama_token t = greedy_next();
        direct.push_back(t);
        if (!decode1(t)) { srv::engine_close(eng); return 1; }
    }

    // Nuke and restore.
    llama_memory_clear(llama_get_memory(lctx), true);
    auto entry = snap.lookup(h);
    if (!entry) { std::fprintf(stderr, "snaptest: lookup miss after store\n"); srv::engine_close(eng); return 1; }
    if (!snap.load(*entry, lctx, 0, &err)) {
        std::fprintf(stderr, "snaptest: load: %s\n", err.c_str());
        srv::engine_close(eng);
        return 1;
    }

    std::vector<llama_token> resumed;
    resumed.push_back(pending);
    if (!decode1(pending)) { srv::engine_close(eng); return 1; }
    for (int32_t i = 0; i < tail; ++i) {
        llama_token t = greedy_next();
        resumed.push_back(t);
        if (!decode1(t)) { srv::engine_close(eng); return 1; }
    }

    const bool same = direct == resumed;
    std::printf("SNAPSHOT ROUND-TRIP: %s (%zu tokens compared)\n",
                same ? "IDENTICAL" : "MISMATCH", direct.size());

    // Stamp refusal: a cache opened under a different build id must not see
    // this snapshot -- serving a checkpoint across an incompatible state format
    // is fluent garbage with extra steps.
    cache::CompatStamp other = stamp;
    other.engine_build_id ^= 0xDEAD;
    cache::SnapshotCache snap2(cc, other);
    std::string e2;
    const bool refused = snap2.open(&e2) && !snap2.lookup(h).has_value();
    std::printf("STAMP REFUSAL: %s\n", refused ? "correct (foreign stamp invisible)"
                                               : "BROKEN (foreign snapshot visible)");

    std::filesystem::remove_all(cc.dir);
    srv::engine_close(eng);
    return (same && refused) ? 0 : 1;
}

// THE STREAMING RUN, consolidated onto the server engine: one setup sequence,
// one generate loop, two consumers (this and serve). The readout drives through
// the observer hooks; its output format is byte-compatible with the pre-
// consolidation path. (T28: difftest reads only the generated text after the
// last readout line; no fuller parser exists in scripts/ today.)
// One display-only difference, deliberate: the Decode phase mark shifts by one
// token's sampling (~ms), because the engine has no hook between the prefill
// decode and the first sample.
int cmd_run(const Args& a) {
    if (a.no_stream) return cmd_run_reference(a);

    dray::server::EngineConfig ec;
    ec.model_path = a.model;
    ec.cap = a.cap;
    ec.n_ctx = a.n_ctx;
    ec.repack_dir = a.repack_dir;
    ec.kv_overrides = a.kv_overrides;
    ec.gpu = a.gpu;
    ec.kv_quant = a.kv;
    ec.prefill_chunk = a.prefill_chunk;
    ec.n_threads = a.n_threads;
    ec.force_stream = a.force_stream;
    ec.resident = a.resident;
    ec.plan_to_stdout = true;   // the plan is FOR the user here, not the operator

    std::string err;
    dray::server::Engine* eng = dray::server::engine_open(ec, &err);
    if (!eng) {
        if (err.rfind("REFUSED", 0) == 0) { std::printf("\n%s\n", err.c_str()); return 2; }
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    namespace rep = dray::report;
    rep::Readout::Options ropt;
    ropt.model = a.model;
    ropt.jsonl_path = a.status_path;
    rep::Readout ro(ropt);
    ro.set_context("llama.cpp direct-io load",
                   dray::server::engine_plan_report(eng) + "\n" +
                       dray::config::env_report());

    // T5: the README's limiter promise, finally wired. A persisted calibration
    // sets the per-class ceilings, quoted WITH its age on every readout line;
    // no calibration means the verdict stays honestly unknown, and says how to
    // get one. Never a default, never fresh-truth theatre.
    {
        dray::calib::Calibration cal;
        std::string when, lerr;
        if (dray::calib::load_calibration(&cal, &when, a.model + ".lzcal.json", &lerr)) {
            rep::Ceilings ceil;
            ceil.set(rep::ByteClass::Stream, cal.stream_bw);
            ceil.set(rep::ByteClass::Gather, cal.gather_bw);
            ceil.provenance = "calibrated " + when;
            ro.set_ceilings(ceil);
            std::fprintf(stderr,
                         "ceilings: %s; thermal state may differ, re-run calibrate to refresh\n",
                         ceil.provenance.c_str());
        } else {
            // A calibration measures the DRIVE, not the model, but it is stored
            // beside the model file -- so sweeping one model taught the engine
            // nothing about the next one on the same disk, and most runs went
            // out with "ceiling unknown". That is how a 2.75 GB/s realised rate
            // sat unremarked next to a 6.8 GB/s drive for days (2026-08-24).
            //
            // So fall back to any calibration on the SAME VOLUME, and name the
            // file it came from. Borrowed, disclosed, never silent.
            std::string borrowed, bwhen;
            try {
                const std::filesystem::path mp(a.model);
                const auto root = mp.root_name();
                // Walk up, and at each level look one directory deeper too:
                // models are laid out as <root>/<repo>/<quant>/file.gguf, so a
                // calibration for a different model is a SIBLING subtree away,
                // never in a shared parent's own file list.
                auto scan = [&](const std::filesystem::path& dir) {
                    std::error_code ec;
                    for (const auto& e : std::filesystem::directory_iterator(
                             dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
                        if (ec) return;
                        if (!e.is_regular_file(ec)) continue;
                        const std::string p = e.path().string();
                        if (p.size() > 11 && p.compare(p.size() - 11, 11, ".lzcal.json") == 0) {
                            dray::calib::Calibration c2;
                            std::string e2;
                            if (dray::calib::load_calibration(&c2, &bwhen, p, &e2)) {
                                cal = c2;
                                borrowed = e.path().filename().string();
                                return;
                            }
                        }
                    }
                };
                for (std::filesystem::path dir = mp.parent_path();
                     !dir.empty() && dir.root_name() == root; dir = dir.parent_path()) {
                    scan(dir);
                    if (!borrowed.empty()) break;
                    std::error_code ec;
                    for (const auto& e : std::filesystem::directory_iterator(
                             dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
                        if (ec) break;
                        if (e.is_directory(ec)) { scan(e.path()); if (!borrowed.empty()) break; }
                    }
                    if (!borrowed.empty()) break;
                    if (dir == dir.parent_path()) break;
                }
            } catch (const std::exception&) { /* no calibration, say so below */ }

            if (!borrowed.empty()) {
                rep::Ceilings ceil;
                ceil.set(rep::ByteClass::Stream, cal.stream_bw);
                ceil.set(rep::ByteClass::Gather, cal.gather_bw);
                ceil.provenance = "calibrated " + bwhen + " for a DIFFERENT file on this volume (" +
                                  borrowed + ")";
                ro.set_ceilings(ceil);
                std::fprintf(stderr,
                             "ceilings: %s. The drive is the same; the model is not. Re-run "
                             "calibrate on THIS model to remove the borrow.\n",
                             ceil.provenance.c_str());
            } else {
                std::fprintf(stderr,
                             "no drive calibration; limiter verdicts unknown. run: dray calibrate -m <model>\n");
            }
        }
    }

    rep::Counters c;
    c.tokens_requested = a.n_predict;
    c.resident_cap = static_cast<int64_t>(a.cap);
    c.context_cap = static_cast<int64_t>(a.n_ctx ? a.n_ctx : 32768);

    int32_t n_prompt = 0;
    dray::server::GenParams gp;
    gp.prompt = a.prompt;
    gp.max_tokens = a.n_predict;
    // --stop was parsed, plumbed into batch and rotate, and dropped HERE -- the
    // one path a user trying the flag reaches first. A silently ignored flag is
    // indistinguishable from a honoured one (2026-08-24 audit).
    gp.stop = a.stop_seqs;
    gp.temperature = a.greedy ? 0.0f : 0.8f;
    gp.seed = a.seed;
    gp.on_prefill = [&](int32_t n) {
        n_prompt = n;
        std::fprintf(stderr, "prefill: %d tokens\n", n);
        c.phase = rep::Phase::Prefill;
        ro.update(c, rep::monotonic_ns());
    };

    // Swarm R6/R7/S13/S37: the streaming path now feeds the readout the same
    // truths the reference path always had -- byte classes from the streamer's
    // own counters, ledger residency, and the cap-honesty verdict, per token.
    uint64_t prev_stream = 0, prev_gather = 0;
    bool over_cap_seen = false;
    dray::server::GenResult r = dray::server::engine_generate(
        eng, gp, [&](const std::string&) {
            if (c.phase != rep::Phase::Decode) {
                c.phase = rep::Phase::Decode;
            }
            ++c.tokens_done;
            c.context_len = static_cast<int64_t>(n_prompt) + c.tokens_done;
            const size_t rss = dray::mem::Accountant::process_rss();
            if (rss) c.rss_bytes = static_cast<int64_t>(rss);
            const auto ec2 = dray::server::engine_counters(eng);
            const uint64_t g = ec2.bytes_gather;
            const uint64_t s = ec2.bytes_streamed - ec2.bytes_gather;
            c.add_bytes(rep::ByteClass::Gather, static_cast<int64_t>(g - prev_gather));
            c.add_bytes(rep::ByteClass::Stream, static_cast<int64_t>(s - prev_stream));
            prev_gather = g;
            prev_stream = s;
            c.resident_bytes = static_cast<int64_t>(ec2.resident_bytes);
            // R7 residual: the two hit rates, from the streamer's own ledgers.
            c.routed_bytes_demanded = static_cast<int64_t>(ec2.routed_needed);
            c.routed_bytes_hit = static_cast<int64_t>(
                ec2.routed_needed - std::min(ec2.bytes_gather, ec2.routed_needed));
            const uint64_t active_needed = ec2.routed_needed + ec2.uncond_needed;
            const uint64_t active_read = ec2.bytes_gather + ec2.uncond_read;
            c.active_bytes_demanded = static_cast<int64_t>(active_needed);
            c.active_bytes_hit = static_cast<int64_t>(
                active_needed - std::min(active_read, active_needed));
            if (ec2.over_cap && !over_cap_seen) {
                over_cap_seen = true;
                std::fprintf(stderr, "\nCAP BREACH: resident bytes exceed the configured cap; see final accountant report\n");
            }
            ro.update(c, rep::monotonic_ns());
            ro.render_terminal(std::cerr);
            ro.write_jsonl();
        });
    ro.end_terminal(std::cerr);

    if (!r.error.empty()) {
        std::fprintf(stderr, "%s\n", r.error.c_str());
    }
    std::printf("\n%s\n", r.text.c_str());
    std::fprintf(stderr, "\n%s\n", ro.final_summary().c_str());
    std::fprintf(stderr, "%s\n", dray::server::engine_report(eng).c_str());
    // R6: the per-category ledger, previously printed only by the reference
    // path, closes every streaming run too.
    std::fprintf(stderr, "%s\n", dray::server::engine_accountant_report(eng).c_str());
    if (over_cap_seen) {
        std::fprintf(stderr, "CAP BREACH occurred during this run; figures above are the ledger's.\n");
    }
    const uint64_t failures = dray::server::engine_failures(eng);
    if (failures > 0) {
        std::fprintf(stderr,
                     "\nOUTPUT NOT TRUSTWORTHY: %llu weights failed to materialise.\n",
                     static_cast<unsigned long long>(failures));
    }
    dray::server::engine_close(eng);
    // T19: an untrustworthy run must not score as clean on the primary gate.
    return (r.error.empty() && failures == 0 && !over_cap_seen && !r.aborted) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, &a)) return 1;
    if (a.help || a.cmd.empty()) { usage(); return a.help ? 0 : 1; }
    if (a.model.empty()) { std::fprintf(stderr, "-m <model.gguf> is required\n"); return 1; }

    if (a.cmd == "plan")      return cmd_plan(a);
    if (a.cmd == "calibrate") return cmd_calibrate(a);
    if (a.cmd == "verify")    return cmd_verify(a);
    if (a.cmd == "stream")    return cmd_stream(a);
    if (a.cmd == "repack") {
        if (a.out_dir.empty()) { std::fprintf(stderr, "repack: --out DIR required\n"); return 1; }
        return dray::tools::repack_write(a.model, a.out_dir);
    }
    if (a.cmd == "snaptest")  return cmd_snaptest(a);
    if (a.cmd == "batch")     return cmd_batch(a);
    if (a.cmd == "run")       return cmd_run(a);
    if (a.cmd == "serve") {
        dray::server::EngineConfig ec;
        ec.model_path = a.model;
        ec.cap = a.cap;
        ec.n_ctx = a.n_ctx;
        ec.repack_dir = a.repack_dir;   // S29: was parsed, then silently dropped
        ec.kv_overrides = a.kv_overrides;
    ec.gpu = a.gpu;
    ec.kv_quant = a.kv;
    ec.prefill_chunk = a.prefill_chunk;
    ec.n_threads = a.n_threads;
    ec.force_stream = a.force_stream;
    ec.resident = a.resident;
        ec.reserve_checkpoint = !a.jobs_dir.empty();   // F4
        return dray::server::serve_main(ec, a.port, a.jobs_dir, a.api_key);
    }

    std::fprintf(stderr, "unknown subcommand: %s\n", a.cmd.c_str());
    usage();
    return 1;
}
