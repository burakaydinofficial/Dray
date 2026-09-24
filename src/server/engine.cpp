// See engine.h. THE setup sequence: cmd_run runs OVER this engine since the
// consolidation (S38 retired the old "change both places" rule -- there is one
// place now). Only cmd_run_reference keeps its own independent copy, BY DESIGN:
// it is the differential test's reference half and must not share the code it
// checks.

#include "server/engine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#include "llama.h"

#include "backend/stream_buffer.h"
#include "mem/accountant.h"
#include "models/profile.h"
#include "plan/residency.h"
#include "tools/repack.h"
#include "config/env_report.h"

namespace dray::server {

// One prefill chunk. Client input must never be able to reach
// GGML_ASSERT(n_tokens <= n_batch) -- that is an abort, not an error return --
// so engine_generate decodes prompts in chunks of exactly this many tokens.
static constexpr int32_t kPrefillBatchDefault = 512;
// Runtime prefill chunk. The GPU compute buffer scales with it: 512 tokens
// cost ~1.5 GiB of VRAM on a 27B, which excludes older 2 GiB cards from the
// prefill win entirely. Smaller chunks trade a little amortization for a
// proportionally smaller buffer (measured 2026-08-21).
static int32_t g_prefill_batch = kPrefillBatchDefault;

struct Engine {
    // T13: written per token by the generating thread, readable lock-free by
    // /health while busy -- a cap breach hours into a server generation must
    // not wait for the generation to end to be reportable.
    std::atomic<int32_t> live_tokens{0};
    std::atomic<bool>    live_over_cap{false};
    uint64_t             ckpt_allowance = 0;   // G1: standing Misc reservation
    dray::plan::Plan            plan;
    dray::mem::Accountant*      acct = nullptr;
    dray::backend::Streamer*    streamer = nullptr;
    dray::backend::MergedMetadata meta;
    llama_model*                   model = nullptr;
    llama_context*                 lctx = nullptr;
    const llama_vocab*             vocab = nullptr;
    std::string                    model_id;
    uint64_t                       cap = 0;
};

Engine* engine_open(const EngineConfig& cfg, std::string* err) {
    auto fail = [&](const std::string& m) -> Engine* {
        if (err) *err = m;
        return nullptr;
    };

    std::string perr;
    const uint32_t ctx = cfg.n_ctx ? cfg.n_ctx : 32768;
    const uint32_t n_seq = cfg.n_seq ? cfg.n_seq : 1;
    // --kv: price the floor at what the run will actually allocate.
    double kv_el = 2.0;
    ggml_type kv_t = GGML_TYPE_F16;
    if (cfg.kv_quant == "q4") { kv_el = 0.5625; kv_t = GGML_TYPE_Q4_0; }
    else if (cfg.kv_quant == "q8") { kv_el = 1.0625; kv_t = GGML_TYPE_Q8_0; }
    else if (!cfg.kv_quant.empty() && cfg.kv_quant != "f16") {
        return fail("bad --kv \"" + cfg.kv_quant + "\" (want q4, q8 or f16)");
    }
    dray::plan::Plan p = dray::plan::build_plan(cfg.model_path, cfg.cap, ctx, &perr, n_seq, kv_el);
    if (!perr.empty()) return fail("plan failed: " + perr);
    if (!cfg.repack_dir.empty()) {
        std::string rerr;
        if (!dray::tools::repack_apply(&p, cfg.repack_dir, &rerr)) {
            return fail("repack apply failed: " + rerr);
        }
    }
    std::fprintf(cfg.plan_to_stdout ? stdout : stderr, "%s\n", p.report().c_str());
    // S30: every run names its pulled levers, or it is not reproducible from
    // its own log.
    const std::string envs = dray::config::env_report();
    if (!envs.empty()) std::fprintf(stderr, "%s\n", envs.c_str());
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    if (!p.feasible) return fail("REFUSED: " + p.refusal);

    // Runtime GPU consent, needed here because the prefill chunk default
    // follows the device (see below). --gpu means the platform GPU: Metal on
    // Apple, Vulkan elsewhere; the legacy envs remain censused alternatives.
    const bool gpu_consent = cfg.gpu ||
        [] { const char* v = std::getenv("DRAY_VULKAN"); return v && v[0] == '1'; }() ||
        [] { const char* v = std::getenv("DRAY_METAL"); return v && v[0] == '1'; }();

    // MEASURED (2026-08-23, 6594-token prompt on the 27B): chunk size matters
    // far more on GPU than CPU, because a larger physical batch amortises the
    // PCIe weight transfer -- GPU 43s at 512 vs 23s at 2048, CPU 47m23s vs
    // 44m03s. So the default follows the device: big when a GPU will do the
    // work, modest on CPU where the win is small and the buffer costs capped
    // RAM. --prefill-chunk still overrides, which is what small-VRAM cards
    // need (1.3 GiB at chunk 64, measured).
    if (cfg.prefill_chunk > 0) {
        if (cfg.prefill_chunk < 16 || cfg.prefill_chunk > 8192) {
            return fail("--prefill-chunk must be 16..8192");
        }
        g_prefill_batch = cfg.prefill_chunk;
    } else {
        g_prefill_batch = gpu_consent ? 2048 : kPrefillBatchDefault;
    }

    llama_backend_init();

    auto* e = new Engine();
    e->plan = std::move(p);
    e->cap = cfg.cap;
    e->model_id = cfg.model_path.substr(cfg.model_path.find_last_of("/\\") + 1);

    e->acct = new dray::mem::Accountant(cfg.cap);
    // ALL FOUR floor terms, and CHECKED. PrefillActivation was priced by the
    // plan (up to ~940 MB at 32k context) but never reserved here, so the
    // streamer's cache budget quietly exceeded what plan.cache_budget promised
    // by exactly that amount -- the over-claim rebudget_against_rss then has to
    // claw back. And every reserve() result was discarded: on a refusal the
    // floor bytes stayed off the ledger at exactly the moment the cap was
    // tightest (2026-08-24 audit).
    const bool floor_ok =
        e->acct->reserve(dray::mem::Category::KvCache, e->plan.floor.kv_cache) &&
        e->acct->reserve(dray::mem::Category::RecurrentState, e->plan.floor.recurrent_state) &&
        e->acct->reserve(dray::mem::Category::ComputeScratch, e->plan.floor.compute_scratch) &&
        e->acct->reserve(dray::mem::Category::PrefillActivation, e->plan.floor.prefill_activation);
    if (!floor_ok) {
        engine_close(e);
        return fail("REFUSED: the mandatory floor does not fit the cap (the plan said it "
                    "would; this is a planner/ledger disagreement worth reporting)");
    }

    dray::backend::Config scfg;
    scfg.cap = cfg.cap;
    scfg.queue_depth = 64;
    scfg.slow_load = [] {
        const char* v = std::getenv("DRAY_SLOW_LOAD");
        return v && v[0] == '1';
    }();
    scfg.no_compact = [] {
        const char* v = std::getenv("DRAY_NO_COMPACT");
        return v && v[0] == '1';
    }();
    scfg.n_seq = n_seq;   // churn reserve scales with the union working set
#if defined(__APPLE__)
    // Tier 2 v1: Metal computes correctly over the mapped arena but its
    // MUL_MAT_ID kernels assume canonical strides -- compacted slot tensors
    // (nb[2] = slot size) read at the wrong stride and produce fluent garbage
    // (measured: testbed, compact on = "Takd", compact off = CPU-identical).
    // Until the kernel honors nb[2], Metal forces whole-tensor mode, loudly.
    // This is also why GPU stays opt-IN on Apple rather than the owner's
    // preferred opt-OUT: a silent default must never change the bytes story.
    if (gpu_consent) {
        if (!scfg.no_compact) {
            std::fprintf(stderr, "metal: forcing whole-tensor mode (compaction strides unsupported by Metal MUL_MAT_ID)\n");
            scfg.no_compact = true;
        }
    }
#endif
    e->streamer = new dray::backend::Streamer(*e->acct, e->plan, scfg);
    if (!e->streamer->valid()) {
        std::string m = "streamer unavailable: " + e->streamer->error();
        engine_close(e);
        return fail(m);
    }

    // RESIDENT MODE (owner decision 2026-08-22, opt-OUT): when the whole model
    // fits the cache budget, streaming buys nothing and costs ~3x -- our buffer
    // type is not one of ggml-cpu's "extra buffer types", so it never reaches
    // the repacked-weight matmul kernels (measured: stock 0.70 s/tok vs ours
    // 2.2 on the same 27B and CPU). In that case hand allocation to llama and
    // get out of the way. --force-stream (or DRAY_FORCE_STREAM=1) pins the
    // streaming path regardless: EVERY correctness gate and measurement must
    // use it, or it silently stops testing this engine.
    uint64_t model_bytes = 0;
    for (const auto& t : e->plan.tensors) model_bytes += t.bytes;
    const bool forced = cfg.force_stream ||
        [] { const char* v = std::getenv("DRAY_FORCE_STREAM"); return v && v[0] == '1'; }();
    // OPT-OUT (owner, 2026-08-22): when the whole model fits, get out of the
    // way by default. Proven byte-correct by scripts/residenttest.ps1 (greedy
    // text identical to the streaming path) and measured 3x faster on a 27B
    // (1.2 vs 3.6 s/token). --force-stream/DRAY_FORCE_STREAM opts out and
    // is MANDATORY in every correctness gate and measurement.
    (void)cfg.resident;   // explicit request; the default already does this
    const bool resident_mode = !forced && model_bytes > 0 &&
                               model_bytes <= e->plan.cache_budget;

    static const llama_model_tensor_buft_override kNone = { nullptr, nullptr };
    llama_model_tensor_buft_override overrides[2] = {
        { ".*", e->streamer->buft() }, kNone,
    };
    llama_model_params mp = llama_model_default_params();
    // WEIGHTS NEVER LIVE IN VRAM. This engine's premise is that the model does
    // not fit anywhere convenient, least of all an 8 GB card, and the RAM cap
    // is the promise. --gpu means "use the GPU for the work it is good at",
    // not "move the model there": with n_gpu_layers 0 the weights stay in our
    // accounted host memory and ggml still offloads ops above its batch-size
    // threshold, which is prefill and not decode. Without this, resident mode
    // plus --gpu tried to allocate a 17.5 GB model into 8 GB of VRAM and died.
    if (gpu_consent) mp.n_gpu_layers = 0;   // CPU-only runs keep llama's own default:
                                            // forcing 0 there changed its preferred
                                            // buffer type and diverged residenttest.
    if (resident_mode) {
        // Invariant 2 still holds: no mmap of tensor data, ever. DIRECT_IO
        // reads the weights without retaining page cache, exactly as the
        // streaming path does -- resident mode changes WHO allocates, not
        // how the bytes reach memory.
        mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
        // Account the weights we are about to hand off, so the cap keeps
        // meaning what it says even though llama does the allocating.
        e->acct->reserve(dray::mem::Category::ExpertCache, model_bytes);
        std::fprintf(stderr,
                     "resident mode: the whole model (%.2f GB) fits the cache budget, so "
                     "llama allocates it natively and reaches its repacked-weight kernels "
                     "(~3x faster than our streaming buffer). Pass --force-stream to "
                     "measure the streaming path instead.\n",
                     model_bytes / 1e9);
    } else {
        mp.tensor_buft_overrides = overrides;
        mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    }

    // Without runtime consent (--gpu / censused env), restrict llama to CPU
    // devices -- a backend-carrying binary must never engage a GPU by surprise.
    static std::vector<ggml_backend_dev_t> cpu_devs;
    if (!gpu_consent) {
        cpu_devs.clear();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) cpu_devs.push_back(d);
        }
        cpu_devs.push_back(nullptr);
        mp.devices = cpu_devs.data();
    }

    // Metadata overrides: CLI --override-kv first, then the profile's
    // kv_defaults for keys the CLI did not name (released conversions
    // sometimes LACK keys the loader requires -- M3's five indexer
    // hyperparameters were the first case). Dedupe is ours so precedence
    // never depends on llama's override-application order. Parse failures
    // are load failures, loudly -- a silently dropped override would load a
    // model with wrong hyperparameters and generate fluent nonsense.
    std::vector<llama_model_kv_override> kvo;
    {
        auto parse_one = [&](const std::string& s) -> bool {
            const size_t eq = s.find('=');
            const size_t co = s.find(':', eq == std::string::npos ? 0 : eq);
            if (eq == std::string::npos || co == std::string::npos || eq == 0 ||
                co <= eq + 1 || co + 1 > s.size()) {
                return false;
            }
            const std::string key = s.substr(0, eq);
            const std::string ty = s.substr(eq + 1, co - eq - 1);
            const std::string val = s.substr(co + 1);
            if (key.size() >= 128 || val.size() >= 128 || val.empty()) return false;
            for (const auto& existing : kvo) {
                if (key == existing.key) return true;   // earlier entry wins
            }
            llama_model_kv_override o{};
            std::snprintf(o.key, sizeof(o.key), "%s", key.c_str());
            char* end = nullptr;
            if (ty == "int") {
                o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
                o.val_i64 = std::strtoll(val.c_str(), &end, 10);
                if (!end || *end != '\0') return false;
            } else if (ty == "float") {
                o.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
                o.val_f64 = std::strtod(val.c_str(), &end);
                if (!end || *end != '\0') return false;
            } else if (ty == "bool") {
                o.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
                if (val != "true" && val != "false") return false;
                o.val_bool = (val == "true");
            } else if (ty == "str") {
                o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                std::snprintf(o.val_str, sizeof(o.val_str), "%s", val.c_str());
            } else {
                return false;
            }
            kvo.push_back(o);
            return true;
        };
        for (const std::string& s : cfg.kv_overrides) {
            if (!parse_one(s)) {
                engine_close(e);
                return fail("bad --override-kv \"" + s + "\" (want key=int|float|bool|str:value)");
            }
        }
        const models::Profile* prof = models::find_profile(e->plan.arch);
        if (prof && prof->kv_defaults) {
            for (const char* const* d = prof->kv_defaults; *d; ++d) {
                if (!parse_one(*d)) {
                    engine_close(e);
                    return fail(std::string("bad profile kv_default \"") + *d + "\"");
                }
            }
        }
        if (!kvo.empty()) {
            for (const auto& o : kvo) {
                std::fprintf(stderr, "kv-override: %s (%s)\n", o.key,
                             prof && prof->kv_defaults ? "profile default or CLI" : "CLI");
            }
            llama_model_kv_override term{};
            term.key[0] = '\0';
            kvo.push_back(term);
            mp.kv_overrides = kvo.data();
        }
    }

    if (resident_mode) {
        // THE reason resident mode emitted garbage: the fast path below hands
        // llama a NO-OP tensor loader, because in streaming mode weights are
        // materialised on demand and must never be read at load. With no
        // streamer to fill them, those tensors stay uninitialised. Resident
        // mode therefore uses llama's REAL loader, which reads the weights
        // itself (DIRECT_IO, so Invariant 2 holds: no retained page cache).
        e->model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    } else if (!scfg.slow_load) {
        std::string merr;
        e->meta = dray::backend::merge_shard_metadata(e->plan, &merr);
        if (!e->meta.gguf) {
            engine_close(e);
            return fail("metadata merge failed: " + merr);
        }
        auto no_load = [](ggml_tensor*, void*) {};
        e->model = llama_model_init_from_user(e->meta.gguf, no_load, nullptr, mp);
    } else {
        e->model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    }
    if (!e->model) { engine_close(e); return fail("failed to load model"); }

    // Same refusal as cmd_run: serving fluent text from wrong weights is worse
    // than not serving.
    // In resident mode we own no tensors -- llama allocated and verified them
    // through its own loader -- so our self-check has nothing to inspect and
    // ok() correctly reports "checked 0" as failure. Say what is true instead
    // of pretending we verified something: the check DOES NOT APPLY, and the
    // run is trusting llama's loader rather than our byte comparison.
    const dray::backend::Streamer::SelfCheck sc = e->streamer->self_check();
    if (resident_mode && sc.checked == 0) {
        std::fprintf(stderr,
                     "floor integrity: NOT APPLICABLE in resident mode (llama owns the "
                     "weights; its loader verified them, we did not)\n");
    } else if (!sc.ok()) {
        engine_close(e);
        return fail("floor integrity check failed: " + std::to_string(sc.mismatched) +
                    " of " + std::to_string(sc.checked) + " resident tensors mismatch");
    } else {
        std::fprintf(stderr, "floor integrity: %zu resident tensors match the file\n", sc.checked);
    }

    // T16: the backend identity on every run path, and the Invariant 2 gate.
    // A DEGRADED backend retains page cache, which makes the cap meaningless
    // and every measured byte suspect -- that is a refusal, not a warning,
    // unless the operator explicitly accepts it.
    const std::string iod = e->streamer->io_describe();
    std::fprintf(stderr, "backend: %s\n", iod.c_str());
    if (iod.find("DEGRADED") != std::string::npos) {
        const char* allow = std::getenv("DRAY_ALLOW_DEGRADED");
        if (!(allow && allow[0] == '1')) {
            engine_close(e);
            return fail("REFUSED: storage backend is DEGRADED (uncached mode "
                        "unavailable), so the memory cap cannot be enforced "
                        "(Invariant 2). Set DRAY_ALLOW_DEGRADED=1 to run "
                        "anyway; measurements from such a run are contaminated.");
        }
        std::fprintf(stderr, "WARNING: DEGRADED backend accepted by "
                             "DRAY_ALLOW_DEGRADED=1; the cap is not "
                             "enforceable and measurements are contaminated\n");
    }

    llama_context_params cp = llama_context_default_params();
    // J1: llama splits this across n_seq PRIVATE per-sequence rings of ctx
    // cells each (kv_unified false); the plan funded exactly that.
    cp.n_ctx = ctx * n_seq;
    // SEAMLESS GPU (2026-08-22): keep the KV cache and attention on the CPU.
    // Measured, --gpu on a 27B: with KQV offloaded, decode ran 2.6 s/token
    // against 2.0 on CPU because attention followed the cache into VRAM and
    // the graph split 852 ways. Vulkan already offloads only ops above
    // op_offload_min_batch_size, so PREFILL (512-token chunks) still goes to
    // the GPU while DECODE (batch 1) stays on CPU -- the split we measured as
    // correct, now automatic. It also keeps KV inside the accounted RAM cap.
    if (gpu_consent) {
        cp.offload_kqv = false;
    }
    if (kv_t != GGML_TYPE_F16) {
        cp.type_k = kv_t;
        cp.type_v = kv_t;
        // llama requires flash attention for a quantized V cache.
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    cp.n_seq_max = n_seq;
    cp.n_batch = g_prefill_batch;
    // PHYSICAL batch, which is what actually gets computed in one graph. It
    // defaulted to 512 no matter what n_batch said, so raising --prefill-chunk
    // alone could never buy anything: the work still arrived 512 at a time.
    // Larger ubatch means more parallel work per graph launch and, under --gpu,
    // fewer weight transfers across PCIe per prompt token.
    cp.n_ubatch = g_prefill_batch;
    // llama defaults n_threads to 4 (GGML_DEFAULT_N_THREADS) no matter how
    // many cores exist. Use the machine unless the caller says otherwise.
    {
        const int hw = static_cast<int>(std::thread::hardware_concurrency());
        // MEASURED (2026-08-22): this process runs ~9 threads of its own (IOCP
        // completions, main, backend). ggml's pool SPIN-waits, so once total
        // threads exceed cores the spinners starve the workers and throughput
        // collapses -- 22 requested on 22 cores measured 50-61 s/token against
        // 2.3 at 4. Clamp so the pool cannot oversubscribe, and say so.
        const int kOurThreads = 9;
        const int ceiling = hw > kOurThreads ? hw - kOurThreads : 1;
        // MEASURED 2026-08-21: this engine does NOT scale with threads and
        // COLLAPSES at high counts. Qwen3.6-35B-A3B, everything resident:
        // 4 threads 2.2 s/tok, 6 -> 2.3, 8 -> 2.3, 12 -> 3.4, 22 -> 62.
        // Dense 27B is the same shape (4 -> 2.2, 12 -> 3.7), so the fault is
        // engine-wide, not MoE-specific. Until the cause is found, the
        // default is the MEASURED optimum, not the core count.
        int want = cfg.n_threads > 0 ? cfg.n_threads : 4;
        if (want > ceiling) {
            std::fprintf(stderr,
                         "compute threads: %d requested, CLAMPED to %d -- this process runs "
                         "~%d threads of its own and ggml spin-waits, so exceeding %d cores "
                         "collapses throughput (measured 20x+).\n",
                         want, ceiling, kOurThreads, hw);
            want = ceiling;
        }
        // SEAMLESS (2026-08-22): decode and prefill are different workloads and
        // llama already separates them. Decode is bandwidth-bound and measured
        // flat from 4 threads upward; PREFILL is compute-bound over 512-token
        // chunks and should use the machine. So they get different defaults,
        // both under the oversubscription ceiling. An explicit --threads sets
        // both, because someone measuring wants one variable, not two.
        cp.n_threads = want;
        cp.n_threads_batch = cfg.n_threads > 0 ? want : ceiling;
        std::fprintf(stderr, "compute threads: %d decode, %d prefill (ceiling %d)\n",
                     cp.n_threads, cp.n_threads_batch, ceiling);
    }
    // Resident mode: llama owns the weights, so the streamer must NOT touch
    // graph nodes -- repointing tensors it does not own produced fluent
    // garbage ("!!!!!!!!") on the first resident run, which is exactly the
    // failure mode this project refuses to ship. No callback, no repointing.
    if (!resident_mode) {
        cp.cb_eval = [](ggml_tensor* t, bool ask, void* ud) -> bool {
            auto* s = static_cast<dray::backend::Streamer*>(ud);
            if (ask) {
                // False lets ggml batch this node with its neighbours into ONE
                // compute call; true forces a single-node call plus a full
                // thread-pool barrier. needs() is a narrow allowlist because a
                // broad version diverged both gates.
                if (!s->needs(t)) return false;
                s->materialise(t);
                return true;
            }
            s->release(t);
            return true;
        };
        cp.cb_eval_user_data = e->streamer;
    }
    cp.abort_callback = &dray::backend::Streamer::abort_cb;
    cp.abort_callback_data = e->streamer;
    e->lctx = llama_init_from_model(e->model, cp);
    if (!e->lctx) { engine_close(e); return fail("failed to create context"); }

    // F4/G1: the checkpoint allowance, reserved ONCE so it subtracts from the
    // cache budget like every other floor item (Invariant 1). Sized from the
    // PLANNER's geometry for the admitted n_ctx -- the honest upper bound of a
    // full state blob -- NOT from llama_state_seq_get_size on the freshly
    // created (empty) context, which returns 16 bytes and turned the previous
    // version of this gate into dead code while its flag disabled per-capture
    // accounting entirely (pass-4 G1, the sharpest self-inflicted bug of the
    // campaign). Blobs larger than the allowance fall back to per-capture
    // reserve-or-refuse inside SnapshotCache.
    if (cfg.reserve_checkpoint) {
        const uint64_t blob = e->plan.floor.kv_cache + e->plan.floor.recurrent_state +
                              (16u << 20);   // serializer framing headroom
        if (!e->acct->reserve(dray::mem::Category::Misc, blob)) {
            engine_close(e);
            return fail("REFUSED: cap cannot hold a checkpoint blob (" +
                        std::to_string(blob >> 20) + " MiB) on top of the floor; "
                        "raise --cap, lower --ctx, or serve without --jobs-dir");
        }
        e->ckpt_allowance = blob;
        std::fprintf(stderr, "checkpoint allowance: %llu MiB reserved (jobs enabled)\n",
                     static_cast<unsigned long long>(blob >> 20));
    }

    // Bind the cap on the worse of RSS and commit, same as cmd_run's reconcile.
    const uint64_t worst = std::max<uint64_t>(
        dray::mem::Accountant::process_rss(),
        dray::mem::Accountant::process_committed());
    if (worst) e->streamer->rebudget_against_rss(worst);

    const uint64_t need = e->streamer->largest_streamed_bytes();
    const uint64_t have = e->streamer->rebudget_against_rss(worst);
    // The batch working set is admission's problem too, bounded by MEASURED
    // lines only: below 1x the widest union region the run certainly dies
    // mid-step (B=38, budget 0.71x, died); at 2x it ran clean (B=32). The
    // full churn reserve is a worst-case SIZING figure, not an admission
    // bound -- requiring it refused a configuration that measures 5.45x.
    const uint64_t region = e->streamer->batch_region_bytes();
    if (n_seq > 1 && region > have) {
        const uint64_t suggest = cfg.cap + (region - have);
        std::string m = "REFUSED: cap leaves " + std::to_string(have >> 20) +
                        " MiB of cache but one batch-" + std::to_string(n_seq) +
                        " union region needs " + std::to_string(region >> 20) +
                        " MiB; lower --batch or try --cap " +
                        std::to_string((unsigned long long)std::ceil(
                            static_cast<double>(suggest) / (1024.0 * 1024 * 1024))) + "G";
        engine_close(e);
        return fail(m);
    }
    if (n_seq > 1 && region * 4 > have) {
        std::fprintf(stderr,
                     "[dray] WARNING: cache budget %llu MiB is between 1x and 2x the batch-%u\n"
                     "[dray] union region (%llu MiB): untested regime -- proven clean at 2x, proven\n"
                     "[dray] fatal below 1x. The run may fail mid-step; bytes remain honest.\n",
                     (unsigned long long)(have >> 20), n_seq,
                     (unsigned long long)(region >> 20));
    }
    if (need > have) {
        const uint64_t suggest = cfg.cap + (need - have);
        // S36: one unit family per message. --cap parses bare G as GiB, so the
        // figures here are MiB/GiB (binary) throughout -- decimal MB beside a
        // binary G suggestion made the same bytes look like two numbers.
        std::string m = "REFUSED: cap leaves " + std::to_string(have >> 20) +
                        " MiB of cache but " + e->streamer->largest_streamed_name() +
                        " needs " + std::to_string(need >> 20) +
                        " MiB whole; try --cap " +
                        std::to_string((unsigned long long)std::ceil(
                            static_cast<double>(suggest) / (1024.0 * 1024 * 1024))) + "G";
        engine_close(e);
        return fail(m);
    }

    // GPU consent moves the KV cache and recurrent state into VRAM (measured
    // 2026-08-21: "Vulkan0 KV buffer", "Vulkan0 RS buffer" -- weights stay in
    // our host arena, state does not). The RAM floor reserved both; holding
    // that reservation would starve the weight cache of memory nothing is
    // using. Hand it back, loudly, so the ledger describes reality.
    // (The KV reservation is NOT released under --gpu any more: offload_kqv is
    // false, so the cache stays in CPU RAM where the ledger already counts it.)

    e->vocab = llama_model_get_vocab(e->model);
    return e;
}

void engine_close(Engine* e) {
    if (!e) return;
    if (e->lctx) llama_free(e->lctx);
    if (e->model) llama_model_free(e->model);
    dray::backend::free_merged_metadata(e->meta);
    delete e->streamer;
    delete e->acct;
    llama_backend_free();
    delete e;
}

// Swarm S2: a llama token can be a bare byte, so a multi-byte codepoint can
// straddle two pieces; nlohmann dump() throws on the fragment and the throw
// escapes to std::terminate through the SSE and worker threads. Returns the
// length of the longest prefix ending on a codepoint boundary; the caller
// holds the tail back until it completes.
static size_t utf8_complete_prefix(const std::string& s) {
    size_t i = s.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0x80) == 0) return i;                // ASCII tail: complete
        if ((c & 0xC0) == 0xC0) {                     // lead byte at i-1
            const size_t need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3
                              : (c & 0xF8) == 0xF0 ? 4 : 1;
            return (s.size() - (i - 1)) >= need ? s.size() : i - 1;
        }
        --i; ++back;                                  // continuation: keep scanning
    }
    return s.size();  // malformed anyway; the dump-side replace backstop holds
}

GenResult engine_generate(Engine* e, const GenParams& p,
                          const std::function<void(const std::string&)>& token_cb) {
    GenResult r;
    int32_t max_out = p.max_tokens;
    std::string pend;  // bytes held back until their codepoint completes
    auto feed = [&](const char* b, size_t np) {
        const size_t base = r.text.size();   // delivered + pend, before this piece
        r.text.append(b, np);
        // T10/F8: stop sequences match against the growing decoded text; on a
        // hit the text is trimmed at the match AND the not-yet-delivered span
        // up to the trim point is emitted, so SSE and the job registry carry
        // the same bytes r.text keeps (content inside the final piece but
        // before the match was previously lost to both). A partial stop that
        // straddles earlier pieces can still leak its prefix to the wire;
        // full holdback is a post-tag item, noted in DECISIONS.
        // Match against seed+text, not text alone: a stop whose prefix was
        // generated before a crash and whose suffix arrives after resume lives
        // across that boundary, and matching only the new session's text made
        // it invisible -- a resumed greedy job could run PAST its stop and
        // produce longer text than the uninterrupted run the README promises
        // it matches (2026-08-24 audit). The seed cannot contain a full stop:
        // the pre-crash session would have trimmed and finished on it.
        const size_t seedn = p.stop_seed.size();
        for (const std::string& st : p.stop) {
            if (st.empty() || seedn + r.text.size() < st.size()) continue;
            size_t at;
            if (seedn == 0) {
                at = r.text.rfind(st);
                if (at == std::string::npos || at + st.size() < r.text.size() - np) continue;
            } else {
                const std::string comb = p.stop_seed + r.text;
                const size_t cat = comb.rfind(st);
                if (cat == std::string::npos ||
                    cat + st.size() < comb.size() - np) continue;
                if (cat < seedn) {
                    // Straddle: the stop begins inside the persisted prefix.
                    // Nothing of this session survives, and the caller must
                    // drop the overhang from its stored text.
                    r.base_trim = seedn - cat;
                    r.text.clear();
                    r.stop_hit = true;
                    pend.clear();
                    return;
                }
                at = cat - seedn;
            }
            r.text.erase(at);
            r.stop_hit = true;
            if (token_cb) {
                const size_t delivered = base - pend.size();
                if (at > delivered) token_cb(r.text.substr(delivered, at - delivered));
            }
            pend.clear();
            return;
        }
        if (!token_cb) return;
        pend.append(b, np);
        const size_t ok = utf8_complete_prefix(pend);
        if (ok) { token_cb(pend.substr(0, ok)); pend.erase(0, ok); }
    };

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    if (p.temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(p.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(
            p.seed == 0 ? LLAMA_DEFAULT_SEED : p.seed));
    }

    if (p.resume) {
        // I3: a spent budget must be decided BEFORE the pending decode -- the
        // old order decoded, fed and counted the token, then declared a clean
        // finish one token past the bound the client set.
        if (max_out < 1) {
            llama_sampler_free(smpl);
            r.tokens_out = 0;
            r.truncated_by_eog = false;   // finish_reason: length, honestly
            return r;
        }
        // The caller restored a snapshot taken at a safepoint: state excludes
        // the pending token, whose decode here regenerates the logits. No
        // clear, no prefill -- the restored state IS the prefix.
        llama_token pending = p.resume_pending;
        char buf0[256];
        int32_t np0 = llama_token_to_piece(e->vocab, pending, buf0, sizeof(buf0), 0, true);
        llama_batch b0 = llama_batch_get_one(&pending, 1);
        if (llama_decode(e->lctx, b0) != 0) {
            llama_sampler_free(smpl);
            r.error = "resume redecode failed";
            return r;
        }
        if (np0 > 0) feed(buf0, static_cast<size_t>(np0));
        r.tokens_out = 1;
        // G3 FIRST (H7c): a stop completed by the pending token's own piece
        // completes the job regardless of context arithmetic below.
        if (r.stop_hit) {
            r.truncated_by_eog = true;
            if (token_cb && !pend.empty()) { token_cb(pend); pend.clear(); }
            llama_sampler_free(smpl);
            return r;
        }
        // Pass-4b correction + H7: the fresh branch clamps max_out against the
        // context; resume did not. seq_len is read AFTER the pending decode so
        // it already counts that slot -- the bound for the remaining
        // (max_out - 1) decodes is room + 1, and the exact fresh-branch clamp
        // is the model (off by one here cost a resumed greedy run its last
        // token against the byte-identity claim). max_out can also arrive at 0
        // legitimately (the budget was spent exactly at the interruption):
        // that is a CLEAN FINISH, not an error -- erroring resurrected the job
        // every restart until quarantine.
        {
            if (max_out < 1) {
                r.truncated_by_eog = false;   // finish_reason: length, honestly
                if (token_cb && !pend.empty()) { token_cb(pend); pend.clear(); }
                llama_sampler_free(smpl);
                return r;
            }
            const int32_t seq_len = static_cast<int32_t>(
                llama_memory_seq_pos_max(llama_get_memory(e->lctx), 0)) + 1;
            const int32_t room = static_cast<int32_t>(llama_n_ctx_seq(e->lctx)) - seq_len;
            if (max_out > room + 1) max_out = room + 1;
            if (max_out < 1) {
                if (token_cb && !pend.empty()) { token_cb(pend); pend.clear(); }
                llama_sampler_free(smpl);
                r.error = "resume: context is full; nothing can be generated";
                return r;
            }
        }
    } else {
        // Requests are independent: clear KV and recurrent state, then prefill.
        llama_memory_clear(llama_get_memory(e->lctx), true);

        std::vector<llama_token> toks(p.prompt.size() + 8);
        int32_t n = llama_tokenize(e->vocab, p.prompt.c_str(),
                                   static_cast<int32_t>(p.prompt.size()),
                                   toks.data(), static_cast<int32_t>(toks.size()), true, true);
        if (n < 0) {
            toks.resize(static_cast<size_t>(-n));
            n = llama_tokenize(e->vocab, p.prompt.c_str(),
                               static_cast<int32_t>(p.prompt.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);
        }
        if (n <= 0) { llama_sampler_free(smpl); r.error = "tokenization failed"; return r; }
        toks.resize(static_cast<size_t>(n));

        // Admission against the context (swarm S3): a prompt that cannot fit is
        // an error the client can act on, never an abort hours of work deep.
        // J3 latent: under n_seq>1 llama_n_ctx() is the AGGREGATE; the
        // per-sequence bound is llama_n_ctx_seq (identical when n_seq==1).
        const int32_t nctx = static_cast<int32_t>(llama_n_ctx_seq(e->lctx));
        if (n >= nctx) {
            llama_sampler_free(smpl);
            r.error = "prompt is " + std::to_string(n) + " tokens but the context is " +
                      std::to_string(nctx) + "; raise --ctx or shorten the prompt";
            r.bad_request = true;
            return r;
        }

        // Generation cannot outrun the KV either: clamp instead of dying at
        // "decode failed" hours in (the client sees the honest finish_reason).
        if (max_out > nctx - n) max_out = nctx - n;

        r.tokens_in = n;   // T9
        if (p.on_prefill) p.on_prefill(n);
        // Chunked prefill: n_batch is a compute-sizing knob, not a prompt limit.
        for (int32_t i0 = 0; i0 < n; i0 += g_prefill_batch) {
            if (p.on_prefill_progress) p.on_prefill_progress(i0);   // F13
            // T3: prefill is minutes on the large models; cancel, disconnect
            // and shutdown must not be ignored for its whole duration.
            if (p.should_continue && !p.should_continue()) {
                llama_sampler_free(smpl);
                r.cancelled = true;
                return r;
            }
            const int32_t take = std::min(g_prefill_batch, n - i0);
            llama_batch batch = llama_batch_get_one(toks.data() + i0, take);
            if (llama_decode(e->lctx, batch) != 0) {
                llama_sampler_free(smpl);
                // Same reasoning as the decode loop: prefill never consulted
                // aborted() at all, so a weight that failed to materialise during
                // the prompt returned a bare "prefill failed".
                if (e->streamer && e->streamer->aborted()) {
                    r.aborted = true;
                    r.error = "prefill failed: a weight failed to materialise";
                } else {
                    r.error = "prefill failed";
                }
                return r;
            }
        }
    }

    for (int32_t i = r.tokens_out; i < max_out; ++i) {
        llama_token t = llama_sampler_sample(smpl, e->lctx, -1);
        if (llama_vocab_is_eog(e->vocab, t)) { r.truncated_by_eog = true; break; }
        if (p.on_safepoint) {
            // T8: flush the held-back UTF-8 tail into the consumer BEFORE the
            // checkpoint reads its text, or resumed text loses those bytes and
            // gains a U+FFFD. Safepoints exist only on the job path, whose
            // token_cb appends to the registry -- no SSE sees a partial glyph.
            if (token_cb && !pend.empty()) { token_cb(pend); pend.clear(); }
            p.on_safepoint(t, p.resume_tokens_done + i);   // F5: absolute
        }

        char buf[256];
        int32_t np = llama_token_to_piece(e->vocab, t, buf, sizeof(buf), 0, true);

        llama_batch b1 = llama_batch_get_one(&t, 1);
        if (llama_decode(e->lctx, b1) != 0) {
            // ASK WHY FIRST. A materialise failure makes ggml return ABORTED, so
            // llama_decode fails and this branch used to fire with only a generic
            // message -- leaving r.aborted false on the very path that means "a
            // weight did not arrive". Callers keying on r.aborted (the streaming
            // error frame, the job status) then saw a plain error and could not
            // tell an untrustworthy run from a broken request (2026-08-24 audit).
            if (e->streamer->aborted()) {
                r.aborted = true;
                r.error = "decode failed: a weight failed to materialise";
            } else {
                r.error = "decode failed";
            }
            break;
        }
        e->live_tokens.store(i + 1, std::memory_order_relaxed);
        if (e->streamer->over_cap()) e->live_over_cap.store(true, std::memory_order_relaxed);
        if (e->streamer->aborted()) { r.aborted = true; break; }
        if (p.should_continue && !p.should_continue()) {
            r.cancelled = true;
            r.tokens_out = i + 1;
            // T1: a cancellation that can resume needs the pending-token shape
            // (state BEFORE a sampled token's decode). Sample it and offer one
            // final safepoint; the caller's gate decides whether to store.
            // F6/pass-4b: token t was DECODED (it is in the snapshot state) but
            // its piece had not been fed. Feed it for EVERY cancel -- the guard
            // that scoped this to safepoint callers made foreground cancels
            // count a token whose characters they never delivered.
            if (np > 0) feed(buf, static_cast<size_t>(np));
            if (r.stop_hit) {
                r.cancelled = false;
                r.truncated_by_eog = true;
                break;
            }
            if (p.on_safepoint) {
                // G4: the feed above is the stop matcher; the stop-hit exit
                // before this block keeps a checkpoint from recording a
                // pending token past the stop.
                if (token_cb && !pend.empty()) { token_cb(pend); pend.clear(); }
                // I3: never checkpoint a pending token PAST the client's budget
                // -- the restarted session would compute max_tokens 0 and the
                // poll would report orig_max + 1.
                if (i + 1 < max_out) {
                    const llama_token pt = llama_sampler_sample(smpl, e->lctx, -1);
                    if (!llama_vocab_is_eog(e->vocab, pt)) {
                        // F5: absolute count, base + session-local.
                        p.on_safepoint(pt, p.resume_tokens_done + i + 1);
                    }
                }
            }
            break;
        }
        r.tokens_out = i + 1;
        // Post-decode on purpose: a readout ticking here matches the verified
        // measurement path's timing exactly.
        if (np > 0) feed(buf, static_cast<size_t>(np));
        if (r.stop_hit) { r.truncated_by_eog = true; break; }   // T10: honest "stop"

        // The multi-day discipline, per token, same as cmd_run: bind the cap on
        // the worse of RSS and commit as the run proceeds.
        const uint64_t worst = std::max<uint64_t>(
            dray::mem::Accountant::process_rss(),
            dray::mem::Accountant::process_committed());
        if (worst) e->streamer->rebudget_against_rss(worst);
    }

    // Flush any held-back tail; if genuinely malformed, the dump-side replace
    // backstop renders it as U+FFFD instead of killing the process.
    if (token_cb && !pend.empty()) token_cb(pend);
    llama_sampler_free(smpl);
    return r;
}

struct llama_context* engine_ctx(Engine* e) { return e->lctx; }
const llama_vocab*    engine_vocab(Engine* e) { return e->vocab; }
const llama_model* engine_model(Engine* e) { return e->model; }
// T13: always-safe live fields (atomics written by the generating thread).
int32_t engine_live_tokens(Engine* e) { return e->live_tokens.load(std::memory_order_relaxed); }
bool engine_live_over_cap(Engine* e) { return e->live_over_cap.load(std::memory_order_relaxed); }

EngineCounters engine_counters(Engine* e) {
    EngineCounters c;
    c.bytes_streamed = e->streamer->bytes_streamed();
    c.bytes_gather   = e->streamer->routed_bytes_read();
    c.routed_needed  = e->streamer->routed_bytes_needed();
    c.uncond_needed  = e->streamer->uncond_bytes_needed();
    c.uncond_read    = e->streamer->uncond_bytes_read();
    c.nodes          = e->streamer->nodes_materialised();
    c.failures       = e->streamer->failures();
    c.resident_bytes = e->acct->used();
    c.over_cap       = e->streamer->over_cap();
    return c;
}
std::string engine_accountant_report(Engine* e) { return e->streamer->accountant_report(); }
dray::mem::Accountant* engine_accountant(Engine* e) { return e->acct; }
uint64_t engine_checkpoint_allowance(Engine* e) { return e->ckpt_allowance; }
std::string engine_plan_report(Engine* e) { return e->plan.report(); }
std::string engine_report(Engine* e)   { return e->streamer->report(); }
std::string engine_model_id(Engine* e) { return e->model_id; }
bool engine_tainted(Engine* e)         { return e->streamer->failures() > 0; }
uint64_t engine_failures(Engine* e)    { return e->streamer->failures(); }

}  // namespace dray::server

namespace dray::server {

// Batched lockstep decode. Correctness before speed: B copies of one greedy
// prompt MUST produce B byte-identical outputs matching the single-stream
// control (the batchdiff gate), because the ids tensors run [k, B] through the
// compaction machinery here -- the ne[1]>1 paths three audits called untested
// get their exercise WITH an oracle attached.
BatchResult engine_generate_batch(Engine* e, const BatchParams& p) {
    BatchResult br;
    const int32_t B = static_cast<int32_t>(p.prompts.size());
    if (B < 1) { br.error = "batch: no prompts"; return br; }
    if (static_cast<uint32_t>(B) > e->plan.n_seq) {
        // Invariant 1: the admission funded plan.n_seq sequences; more would
        // run KV outside the cap. Refused here, not discovered mid-decode.
        br.error = "batch: " + std::to_string(B) + " sequences but the plan admitted " +
                   std::to_string(e->plan.n_seq) + "; re-plan with --batch " +
                   std::to_string(B);
        return br;
    }
    br.seqs.resize(static_cast<size_t>(B));

    llama_memory_clear(llama_get_memory(e->lctx), true);
    const int32_t nctx_total = static_cast<int32_t>(llama_n_ctx(e->lctx));

    // Per-sequence machinery: sampler (seed+s, disclosed), UTF-8-free feed
    // (offline mode buffers whole texts; no SSE hold-back needed), stop
    // matcher, liveness.
    struct Seq {
        llama_sampler* smpl = nullptr;
        llama_token pending = 0;
        bool live = false;
        bool admitted = false;
        int32_t n_prompt = 0;
        std::vector<llama_token> toks;
    };
    std::vector<Seq> seqs(static_cast<size_t>(B));
    auto feed_seq = [&](int32_t s, const char* b, size_t np) {
        GenResult& r = br.seqs[static_cast<size_t>(s)];
        r.text.append(b, np);
        for (const std::string& st : p.stop) {
            if (st.empty() || r.text.size() < st.size()) continue;
            const size_t at = r.text.rfind(st);
            if (at != std::string::npos && at + st.size() >= r.text.size() - np) {
                r.text.erase(at);
                r.stop_hit = true;
                return;
            }
        }
    };

    // Prefill each sequence in turn (chunked; activation scratch reused, which
    // is why the planner funds it once).
    for (int32_t s = 0; s < B; ++s) {
        GenResult& r = br.seqs[static_cast<size_t>(s)];
        Seq& q = seqs[static_cast<size_t>(s)];
        const std::string& pr = p.prompts[static_cast<size_t>(s)];
        std::vector<llama_token> toks(pr.size() + 8);
        // J3: per-sequence admission BEFORE the prefill is paid for in drive
        // time -- against the PER-SEQUENCE ring, not the aggregate.
        int32_t n = llama_tokenize(e->vocab, pr.c_str(), static_cast<int32_t>(pr.size()),
                                   toks.data(), static_cast<int32_t>(toks.size()), true, true);
        if (n < 0) {
            toks.resize(static_cast<size_t>(-n));
            n = llama_tokenize(e->vocab, pr.c_str(), static_cast<int32_t>(pr.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);
        }
        if (n <= 0) { r.error = "batch: tokenize failed"; continue; }
        {
            const int32_t cseq = static_cast<int32_t>(llama_n_ctx_seq(e->lctx));
            if (n >= cseq) {
                r.bad_request = true;
                r.error = "batch: prompt of " + std::to_string(n) + " tokens >= per-sequence context " +
                          std::to_string(cseq) + "; raise --ctx or shorten the prompt";
                continue;
            }
        }
        q.n_prompt = n;
        r.tokens_in = n;
        q.toks.assign(toks.begin(), toks.begin() + n);
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        q.smpl = llama_sampler_chain_init(sp);
        if (p.temperature > 0.0f) {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_temp(p.temperature));
            const uint32_t sseed = (p.seed ? p.seed : 1234u) + static_cast<uint32_t>(s);
            llama_sampler_chain_add(q.smpl, llama_sampler_init_dist(sseed));
        } else {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_greedy());
        }
        q.admitted = true;
    }

    // JOINT prefill: pack ALL sequences' prompt tokens into shared
    // g_prefill_batch chunks -- ceil(total/512) model sweeps instead of one
    // sweep per sequence (measured: 32 sequential prefills were 72% of the
    // crown run's bytes). Chunks carry ne[1] > n_seq, so the streamer keeps
    // them on the prefill SCAN path (whole tensors through the ring, no
    // union-region materialisation) -- the same path the sequential version
    // ran on, with the same hazards, just fewer sweeps; never more than the
    // sequential count in any prompt/batch shape. Each token carries its own
    // seq_id and position; a sequence's last prompt token requests logits so
    // its first generated token samples from that row.
    {
        const int32_t chunk_cap = g_prefill_batch;
        llama_batch pb = llama_batch_init(chunk_cap, 0, 1);
        std::vector<int32_t> fed(static_cast<size_t>(B), 0);
        std::vector<int32_t> logit_seq(static_cast<size_t>(chunk_cap));
        std::vector<int32_t> logit_row(static_cast<size_t>(chunk_cap));
        bool cancelled = false;
        for (;;) {
            if (p.should_continue && !p.should_continue()) { cancelled = true; break; }
            pb.n_tokens = 0;
            int32_t n_logit = 0;
            for (int32_t s = 0; s < B && pb.n_tokens < chunk_cap; ++s) {
                Seq& q = seqs[static_cast<size_t>(s)];
                if (!q.admitted) continue;
                while (fed[static_cast<size_t>(s)] < q.n_prompt && pb.n_tokens < chunk_cap) {
                    const int32_t i0 = fed[static_cast<size_t>(s)]++;
                    const int32_t j = pb.n_tokens++;
                    pb.token[j] = q.toks[static_cast<size_t>(i0)];
                    pb.pos[j] = i0;
                    pb.n_seq_id[j] = 1;
                    pb.seq_id[j][0] = s;
                    const bool last = (i0 == q.n_prompt - 1);
                    pb.logits[j] = static_cast<int8_t>(last);
                    // llama_get_logits_ith wants the BATCH TOKEN index (it maps
                    // through output_ids itself); a dense logits ordinal aborts.
                    if (last) { logit_seq[static_cast<size_t>(n_logit)] = s;
                                logit_row[static_cast<size_t>(n_logit++)] = j; }
                }
            }
            if (pb.n_tokens == 0) break;
            if (llama_decode(e->lctx, pb) != 0) {
                for (int32_t s = 0; s < B; ++s) {
                    Seq& q = seqs[static_cast<size_t>(s)];
                    if (q.admitted && fed[static_cast<size_t>(s)] > 0) {
                        q.admitted = false;
                        br.seqs[static_cast<size_t>(s)].error = "batch: prefill decode failed";
                    }
                }
                break;
            }
            // Sample the first token of every sequence whose prompt completed
            // in this chunk; logits rows appear in request order.
            for (int32_t li = 0; li < n_logit; ++li) {
                const int32_t s = logit_seq[static_cast<size_t>(li)];
                Seq& q = seqs[static_cast<size_t>(s)];
                q.pending = llama_sampler_sample(q.smpl, e->lctx, logit_row[static_cast<size_t>(li)]);
                q.live = !llama_vocab_is_eog(e->vocab, q.pending);
                if (!q.live) br.seqs[static_cast<size_t>(s)].truncated_by_eog = true;
            }
        }
        llama_batch_free(pb);
        if (cancelled) {
            for (int32_t s = 0; s < B; ++s) {
                if (seqs[static_cast<size_t>(s)].admitted)
                    br.seqs[static_cast<size_t>(s)].cancelled = true;
            }
        }
    }

    br.prefill_end_bytes = engine_counters(e).bytes_streamed;   // J7 boundary

    // Lockstep decode: one llama_decode per step carrying every live
    // sequence's pending token; the streamer sees ids [k, live] and reads the
    // union once.
    llama_batch db = llama_batch_init(B, 0, 1);
    std::vector<int32_t> row_of(static_cast<size_t>(B), -1);
    // J1: this build does NOT use a unified KV pool -- kv_unified defaults
    // false, so each sequence owns a PRIVATE ring of llama_n_ctx_seq() cells
    // and the binding constraint is PER SEQUENCE. (An earlier total-cells
    // guard here was n_seq times too loose; every proven run was symmetric,
    // which made its algebra accidentally correct.)
    const int32_t ctx_seq = static_cast<int32_t>(llama_n_ctx_seq(e->lctx));
    for (int32_t step = 0; step < p.max_tokens; ++step) {
        if (p.should_continue && !p.should_continue()) {
            for (int32_t s = 0; s < B; ++s) {
                if (seqs[static_cast<size_t>(s)].live) br.seqs[static_cast<size_t>(s)].cancelled = true;
            }
            break;
        }
        int32_t live = 0;
        db.n_tokens = 0;
        for (int32_t s = 0; s < B; ++s) {
            Seq& q = seqs[static_cast<size_t>(s)];
            row_of[static_cast<size_t>(s)] = -1;
            if (!q.live) continue;
            // J1: the per-sequence wall, checked on the sequence it binds --
            // only THIS row retires, honestly marked, the rest decode on.
            if (q.n_prompt + step + 1 > ctx_seq) {
                q.live = false;
                br.seqs[static_cast<size_t>(s)].ctx_wall = true;
                continue;
            }
            const int32_t j = db.n_tokens++;
            db.token[j] = q.pending;
            db.pos[j] = q.n_prompt + step;
            db.n_seq_id[j] = 1;
            db.seq_id[j][0] = s;
            db.logits[j] = 1;
            row_of[static_cast<size_t>(s)] = j;
            ++live;
        }
        if (live == 0) break;
        if (llama_decode(e->lctx, db) != 0) {
            const bool tainted = e->streamer->aborted();
            for (int32_t s = 0; s < B; ++s) {
                if (!seqs[static_cast<size_t>(s)].live) continue;
                if (tainted) br.seqs[static_cast<size_t>(s)].aborted = true;
                else br.seqs[static_cast<size_t>(s)].error = "batch: decode failed";
            }
            break;
        }
        ++br.steps;
        e->live_tokens.store(static_cast<int32_t>(br.steps), std::memory_order_relaxed);
        if (e->streamer->over_cap()) e->live_over_cap.store(true, std::memory_order_relaxed);
        // J5: the multi-day discipline, per STEP -- bind the cap on the worse
        // of RSS and commit exactly as the single-stream loop does; without
        // this the unreserved charge froze at open and over_cap was blinded.
        {
            const uint64_t worst = std::max<uint64_t>(
                dray::mem::Accountant::process_rss(),
                dray::mem::Accountant::process_committed());
            if (worst) e->streamer->rebudget_against_rss(worst);
        }
        for (int32_t s = 0; s < B; ++s) {
            Seq& q = seqs[static_cast<size_t>(s)];
            if (!q.live) continue;
            GenResult& r = br.seqs[static_cast<size_t>(s)];
            char buf[256];
            const int32_t np = llama_token_to_piece(e->vocab, q.pending, buf, sizeof(buf), 0, true);
            if (np > 0) feed_seq(s, buf, static_cast<size_t>(np));
            r.tokens_out = step + 1;
            if (r.stop_hit) { r.truncated_by_eog = true; q.live = false; continue; }
            const llama_token nxt =
                llama_sampler_sample(q.smpl, e->lctx, row_of[static_cast<size_t>(s)]);
            if (llama_vocab_is_eog(e->vocab, nxt)) {
                r.truncated_by_eog = true;
                q.live = false;
            } else {
                q.pending = nxt;
            }
        }
        if (p.on_step) p.on_step(step + 1, live);
    }
    llama_batch_free(db);
    for (auto& q : seqs) { if (q.smpl) llama_sampler_free(q.smpl); }
    if (e->streamer->aborted()) {
        for (auto& r : br.seqs) r.aborted = true;
    }
    return br;
}


RotateResult engine_generate_rotated(Engine* e, const RotateParams& p) {
    RotateResult rr;
    const int32_t N = static_cast<int32_t>(p.prompts.size());
    const int32_t W = static_cast<int32_t>(e->plan.n_seq);
    if (N < 1) { rr.error = "rotate: no prompts"; return rr; }
    if (W < 1) { rr.error = "rotate: plan funded no sequences"; return rr; }
    if (p.span < 1) { rr.error = "rotate: span must be >= 1"; return rr; }
    // MEASURED (2026-08-19, testbed bisection): cohort slot recycling is
    // deterministic; the llama_state_seq park/restore roundtrip is NOT --
    // set_data reports success yet reruns of the identical command diverge,
    // the signature of restored-state bookkeeping desync reading
    // uninitialized cells. Until the fork's multi-stream state serialization
    // is proven, rotation refuses mid-generation parking: each cohort runs
    // to completion, then rotates. That is the throughput use case; spans
    // below max_tokens buy only latency fairness and are not worth wrong
    // tokens.
    const bool rot_unsafe = [] {
        const char* v = std::getenv("DRAY_ROTATE_UNSAFE");
        return v && v[0] == '1';
    }();
    if (p.span < p.max_tokens && !rot_unsafe) {
        rr.error = "rotate: span " + std::to_string(p.span) + " < max_tokens " +
                   std::to_string(p.max_tokens) + " requires mid-generation state parking, which is "
                   "NOT yet proven bit-exact in this build (reruns diverge); use --rotate >= " +
                   std::to_string(p.max_tokens) + " (run-to-completion cohorts)";
        return rr;
    }
    if (p.state_dir.empty()) { rr.error = "rotate: --state-dir is required (rotation writes; where is an explicit choice)"; return rr; }
    rr.seqs.resize(static_cast<size_t>(N));

    llama_memory_clear(llama_get_memory(e->lctx), true);
    const int32_t ctx_seq = static_cast<int32_t>(llama_n_ctx_seq(e->lctx));

    // Host-persistent per-GLOBAL-sequence machinery. Only KV/recurrent state
    // rotates through llama slots; samplers, pending tokens and text live in
    // host RAM for the whole run, so a residency is pure state scheduling.
    struct RSeq {
        llama_sampler* smpl = nullptr;
        llama_token pending = 0;
        bool live = false;
        bool admitted = false;
        bool prefilled = false;
        int32_t n_prompt = 0;
        int32_t done = 0;               // decode steps completed
        std::vector<llama_token> toks;
    };
    std::vector<RSeq> gs(static_cast<size_t>(N));

    auto feed_g = [&](int32_t g, const char* b, size_t np) {
        GenResult& r = rr.seqs[static_cast<size_t>(g)];
        r.text.append(b, np);
        for (const std::string& st : p.stop) {
            if (st.empty()) continue;
            const size_t pos = r.text.find(st);
            if (pos != std::string::npos) {
                r.text.resize(pos);
                r.stop_hit = true;
                return;
            }
        }
    };

    // Tokenize + admit + build samplers up front, exactly the batch path's
    // discipline: refuse per sequence before any drive time is spent.
    for (int32_t g = 0; g < N; ++g) {
        GenResult& r = rr.seqs[static_cast<size_t>(g)];
        RSeq& q = gs[static_cast<size_t>(g)];
        const std::string& pr = p.prompts[static_cast<size_t>(g)];
        std::vector<llama_token> toks(pr.size() + 8);
        int32_t n = llama_tokenize(e->vocab, pr.c_str(), static_cast<int32_t>(pr.size()),
                                   toks.data(), static_cast<int32_t>(toks.size()), true, true);
        if (n < 0) {
            toks.resize(static_cast<size_t>(-n));
            n = llama_tokenize(e->vocab, pr.c_str(), static_cast<int32_t>(pr.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);
        }
        if (n <= 0) { r.error = "rotate: tokenize failed"; continue; }
        if (n >= ctx_seq) {
            r.bad_request = true;
            r.error = "rotate: prompt of " + std::to_string(n) + " tokens >= per-sequence context " +
                      std::to_string(ctx_seq);
            continue;
        }
        q.n_prompt = n;
        r.tokens_in = n;
        q.toks.assign(toks.begin(), toks.begin() + n);
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        q.smpl = llama_sampler_chain_init(sp);
        if (p.temperature > 0.0f) {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_temp(p.temperature));
            // Seed by GLOBAL index so a rotated run and split unrotated runs
            // of the same prompts sample identically -- the oracle depends on it.
            const uint32_t sseed = (p.seed ? p.seed : 1234u) + static_cast<uint32_t>(g);
            llama_sampler_chain_add(q.smpl, llama_sampler_init_dist(sseed));
        } else {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_greedy());
        }
        q.admitted = true;
    }

    auto state_path = [&](int32_t g) {
        return p.state_dir + "/g" + std::to_string(g) + ".state";
    };
    const bool keep_states = rot_unsafe;   // diagnostics: copy each park aside
    int32_t park_no = 0;
    auto park = [&](int32_t g, int32_t slot) -> bool {
        const size_t sz = llama_state_seq_get_size(e->lctx, slot);
        if (sz == 0) return false;
        std::vector<uint8_t> buf(sz);
        const size_t got = llama_state_seq_get_data(e->lctx, buf.data(), sz, slot);
        if (got == 0) return false;
        FILE* f = std::fopen(state_path(g).c_str(), "wb");
        if (!f) return false;
        const size_t wr = std::fwrite(buf.data(), 1, got, f);
        std::fclose(f);
        if (wr != got) return false;
        rr.state_bytes_written += wr;
        if (keep_states) {
            FILE* k = std::fopen((state_path(g) + ".park" + std::to_string(park_no++)).c_str(), "wb");
            if (k) { std::fwrite(buf.data(), 1, got, k); std::fclose(k); }
        }
        return true;
    };
    auto unpark = [&](int32_t g, int32_t slot) -> bool {
        FILE* f = std::fopen(state_path(g).c_str(), "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz <= 0) { std::fclose(f); return false; }
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        const size_t rd = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        if (rd != buf.size()) return false;
        if (llama_state_seq_set_data(e->lctx, buf.data(), buf.size(), slot) == 0) return false;
        rr.state_bytes_read += rd;
        return true;
    };

    const int32_t cohorts = (N + W - 1) / W;
    bool all_done = false;
    bool cancelled = false;
    while (!all_done && !cancelled) {
        all_done = true;
        for (int32_t c = 0; c < cohorts && !cancelled; ++c) {
            const int32_t g0 = c * W;
            const int32_t cw = std::min(W, N - g0);
            // Anything left to do in this cohort this round?
            bool work = false;
            for (int32_t s = 0; s < cw; ++s) {
                const RSeq& q = gs[static_cast<size_t>(g0 + s)];
                if (q.admitted && (!q.prefilled || (q.live && q.done < p.max_tokens))) { work = true; break; }
            }
            if (!work) continue;
            if (p.should_continue && !p.should_continue()) { cancelled = true; break; }

            // RESTORE: previously-parked live sequences re-enter their slots.
            for (int32_t s = 0; s < cw; ++s) {
                const int32_t g = g0 + s;
                RSeq& q = gs[static_cast<size_t>(g)];
                if (!q.admitted || !q.prefilled || !q.live) continue;
                if (!unpark(g, s)) {
                    q.live = false;
                    rr.seqs[static_cast<size_t>(g)].error = "rotate: state restore failed";
                }
            }

            // FIRST RESIDENCY: joint prefill for this cohort's unprefilled
            // sequences -- the batch path's shared-chunk scheme, slots local.
            {
                llama_batch pb = llama_batch_init(g_prefill_batch, 0, 1);
                std::vector<int32_t> fed(static_cast<size_t>(cw), 0);
                std::vector<int32_t> lseq(static_cast<size_t>(g_prefill_batch));
                std::vector<int32_t> lrow(static_cast<size_t>(g_prefill_batch));
                for (;;) {
                    pb.n_tokens = 0;
                    int32_t nl = 0;
                    for (int32_t s = 0; s < cw && pb.n_tokens < g_prefill_batch; ++s) {
                        RSeq& q = gs[static_cast<size_t>(g0 + s)];
                        if (!q.admitted || q.prefilled) continue;
                        while (fed[static_cast<size_t>(s)] < q.n_prompt && pb.n_tokens < g_prefill_batch) {
                            const int32_t i0 = fed[static_cast<size_t>(s)]++;
                            const int32_t j = pb.n_tokens++;
                            pb.token[j] = q.toks[static_cast<size_t>(i0)];
                            pb.pos[j] = i0;
                            pb.n_seq_id[j] = 1;
                            pb.seq_id[j][0] = s;
                            const bool last = (i0 == q.n_prompt - 1);
                            pb.logits[j] = static_cast<int8_t>(last);
                            if (last) { lseq[static_cast<size_t>(nl)] = s; lrow[static_cast<size_t>(nl++)] = j; }
                        }
                    }
                    if (pb.n_tokens == 0) break;
                    if (llama_decode(e->lctx, pb) != 0) {
                        for (int32_t s = 0; s < cw; ++s) {
                            RSeq& q = gs[static_cast<size_t>(g0 + s)];
                            if (q.admitted && !q.prefilled && fed[static_cast<size_t>(s)] > 0) {
                                q.admitted = false;
                                rr.seqs[static_cast<size_t>(g0 + s)].error = "rotate: prefill decode failed";
                            }
                        }
                        break;
                    }
                    for (int32_t li = 0; li < nl; ++li) {
                        const int32_t s = lseq[static_cast<size_t>(li)];
                        const int32_t g = g0 + s;
                        RSeq& q = gs[static_cast<size_t>(g)];
                        q.pending = llama_sampler_sample(q.smpl, e->lctx, lrow[static_cast<size_t>(li)]);
                        q.prefilled = true;
                        q.live = !llama_vocab_is_eog(e->vocab, q.pending);
                        if (!q.live) rr.seqs[static_cast<size_t>(g)].truncated_by_eog = true;
                    }
                }
                llama_batch_free(pb);
            }

            // SPAN of lockstep decode, the batch loop's exact shape on slots.
            llama_batch db = llama_batch_init(cw, 0, 1);
            std::vector<int32_t> row_of(static_cast<size_t>(cw), -1);
            for (int32_t t = 0; t < p.span; ++t) {
                if (p.should_continue && !p.should_continue()) { cancelled = true; break; }
                int32_t live = 0;
                db.n_tokens = 0;
                for (int32_t s = 0; s < cw; ++s) {
                    const int32_t g = g0 + s;
                    RSeq& q = gs[static_cast<size_t>(g)];
                    row_of[static_cast<size_t>(s)] = -1;
                    if (!q.admitted || !q.live || q.done >= p.max_tokens) continue;
                    if (q.n_prompt + q.done + 1 > ctx_seq) {
                        q.live = false;
                        rr.seqs[static_cast<size_t>(g)].ctx_wall = true;
                        continue;
                    }
                    const int32_t j = db.n_tokens++;
                    db.token[j] = q.pending;
                    db.pos[j] = q.n_prompt + q.done;
                    db.n_seq_id[j] = 1;
                    db.seq_id[j][0] = s;
                    db.logits[j] = 1;
                    row_of[static_cast<size_t>(s)] = j;
                    ++live;
                }
                if (live == 0) break;
                if (llama_decode(e->lctx, db) != 0) {
                    const bool tainted = e->streamer->aborted();
                    for (int32_t s = 0; s < cw; ++s) {
                        RSeq& q = gs[static_cast<size_t>(g0 + s)];
                        if (!q.live) continue;
                        if (tainted) rr.seqs[static_cast<size_t>(g0 + s)].aborted = true;
                        else rr.seqs[static_cast<size_t>(g0 + s)].error = "rotate: decode failed";
                        q.live = false;
                    }
                    break;
                }
                if (e->streamer->over_cap()) e->live_over_cap.store(true, std::memory_order_relaxed);
                {
                    const uint64_t worst = std::max<uint64_t>(
                        dray::mem::Accountant::process_rss(),
                        dray::mem::Accountant::process_committed());
                    if (worst) e->streamer->rebudget_against_rss(worst);
                }
                for (int32_t s = 0; s < cw; ++s) {
                    const int32_t g = g0 + s;
                    RSeq& q = gs[static_cast<size_t>(g)];
                    if (row_of[static_cast<size_t>(s)] < 0) continue;
                    GenResult& r = rr.seqs[static_cast<size_t>(g)];
                    char buf[256];
                    const int32_t np = llama_token_to_piece(e->vocab, q.pending, buf, sizeof(buf), 0, true);
                    if (np > 0) feed_g(g, buf, static_cast<size_t>(np));
                    q.done += 1;
                    r.tokens_out = q.done;
                    if (r.stop_hit) { r.truncated_by_eog = true; q.live = false; continue; }
                    if (q.done >= p.max_tokens) { q.live = false; continue; }
                    const llama_token nxt =
                        llama_sampler_sample(q.smpl, e->lctx, row_of[static_cast<size_t>(s)]);
                    if (llama_vocab_is_eog(e->vocab, nxt)) {
                        r.truncated_by_eog = true;
                        q.live = false;
                        continue;
                    }
                    q.pending = nxt;
                }
            }
            llama_batch_free(db);
            ++rr.rounds;

            // PARK live sequences; finished ones just vacate. Slots must be
            // empty either way -- the next cohort owns them.
            for (int32_t s = 0; s < cw; ++s) {
                const int32_t g = g0 + s;
                RSeq& q = gs[static_cast<size_t>(g)];
                if (q.admitted && q.prefilled && q.live && q.done < p.max_tokens) {
                    // Diagnostics under DRAY_ROTATE_UNSAFE: keep a copy of
                    // each park for byte-comparison across reruns.
                    if (!park(g, s)) {
                        q.live = false;
                        rr.seqs[static_cast<size_t>(g)].error = "rotate: state park failed";
                    }
                    all_done = false;
                }
                llama_memory_seq_rm(llama_get_memory(e->lctx), s, -1, -1);
            }
            int32_t done_ct = 0;
            for (int32_t g = 0; g < N; ++g) {
                const RSeq& q = gs[static_cast<size_t>(g)];
                if (!q.admitted || !q.live || q.done >= p.max_tokens) ++done_ct;
            }
            if (p.on_round) p.on_round(static_cast<int32_t>(rr.rounds), c, done_ct);
        }
    }
    if (cancelled) {
        for (int32_t g = 0; g < N; ++g) {
            if (gs[static_cast<size_t>(g)].live) rr.seqs[static_cast<size_t>(g)].cancelled = true;
        }
    }
    for (int32_t g = 0; g < N; ++g) {
        RSeq& q = gs[static_cast<size_t>(g)];
        if (q.smpl) llama_sampler_free(q.smpl);
        std::remove(state_path(g).c_str());
    }
    return rr;
}

}  // namespace dray::server
