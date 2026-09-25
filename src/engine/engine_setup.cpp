// Engine::open and its steps: the setup sequence every consumer shares.
//
//   plan -> floor -> streamer -> model -> integrity -> backend -> context
//        -> checkpoint allowance -> admission
//
// Each step refuses with a message naming what would work; nothing here ever
// refuses a model for being large -- not fitting in RAM is the premise.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "llama.h"

#include "backend/stream_buffer.h"
#include "config/env_report.h"
#include "engine/engine.h"
#include "engine/engine_internal.h"
#include "engine/thread_policy.h"
#include "mem/accountant.h"
#include "models/profile.h"
#include "plan/residency.h"
#include "tools/repack.h"

namespace dray::engine {

namespace {

constexpr int32_t kPrefillBatchDefault = 512;

bool env_is_one(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}

// --kv: bytes per KV element for the floor, and the ggml type for the context.
struct KvChoice { double bytes_per_el = 2.0; ggml_type type = GGML_TYPE_F16; bool ok = true; };

KvChoice kv_choice(const std::string& kv) {
    KvChoice c;
    if (kv == "q4")      { c.bytes_per_el = 0.5625; c.type = GGML_TYPE_Q4_0; }
    else if (kv == "q8") { c.bytes_per_el = 1.0625; c.type = GGML_TYPE_Q8_0; }
    else if (!kv.empty() && kv != "f16") c.ok = false;
    return c;
}

uint32_t context_cells(const EngineConfig& c) { return c.n_ctx ? c.n_ctx : 32768; }
uint32_t sequences(const EngineConfig& c)     { return c.n_seq ? c.n_seq : 1; }

}  // namespace

Engine::Engine() = default;

std::unique_ptr<Engine> Engine::open(const EngineConfig& cfg, std::string* err) {
    auto fail = [&](const std::string& m) -> std::unique_ptr<Engine> {
        if (err) *err = m;
        return nullptr;
    };

    // --- plan: the admission maths, from the tensor table alone.
    const KvChoice kv = kv_choice(cfg.kv_quant);
    if (!kv.ok) return fail("bad --kv \"" + cfg.kv_quant + "\" (want q4, q8 or f16)");
    std::string perr;
    plan::Plan p = plan::build_plan(cfg.model_path, cfg.cap, context_cells(cfg), &perr,
                                    sequences(cfg), kv.bytes_per_el);
    if (!perr.empty()) return fail("plan failed: " + perr);
    if (!cfg.repack_dir.empty()) {
        std::string rerr;
        if (!tools::repack_apply(&p, cfg.repack_dir, &rerr)) {
            return fail("repack apply failed: " + rerr);
        }
    }
    std::fprintf(cfg.plan_to_stdout ? stdout : stderr, "%s\n", p.report().c_str());
    // S30: every run names its pulled levers, or it is not reproducible from
    // its own log.
    const std::string envs = config::env_report();
    if (!envs.empty()) std::fprintf(stderr, "%s\n", envs.c_str());
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    if (!p.feasible) return fail("REFUSED: " + p.refusal);

    // --- GPU consent: --gpu means the platform GPU (Metal on Apple, Vulkan
    //     elsewhere); the legacy envs remain censused alternatives.
    const bool gpu_consent = cfg.gpu || env_is_one("DRAY_VULKAN") || env_is_one("DRAY_METAL");

    // --- prefill chunk. MEASURED (2026-08-23, 6594-token prompt on the 27B):
    //     chunk size matters far more on GPU than CPU, because a larger physical
    //     batch amortises the PCIe weight transfer -- GPU 43s at 512 vs 23s at
    //     2048, CPU 47m23s vs 44m03s. --prefill-chunk overrides; small-VRAM cards
    //     need it (1.3 GiB at chunk 64, measured).
    int32_t chunk = gpu_consent ? 2048 : kPrefillBatchDefault;
    if (cfg.prefill_chunk > 0) {
        if (cfg.prefill_chunk < 16 || cfg.prefill_chunk > 8192) {
            return fail("--prefill-chunk must be 16..8192");
        }
        chunk = cfg.prefill_chunk;
    }

    llama_backend_init();

    std::unique_ptr<Engine> e(new Engine());
    e->plan_ = std::make_unique<plan::Plan>(std::move(p));
    e->cap_ = cfg.cap;
    e->model_id_ = cfg.model_path.substr(cfg.model_path.find_last_of("/\\") + 1);
    e->prefill_batch_ = chunk;
    e->load_ = std::make_unique<LoadState>();

    std::string step_err;
    if (!e->reserve_floor(&step_err) ||
        !e->start_streamer(cfg, gpu_consent, &step_err) ||
        !e->load_model(cfg, gpu_consent, &step_err) ||
        !e->verify_floor(&step_err) ||
        !e->check_storage_backend(&step_err) ||
        !e->create_context(cfg, gpu_consent, &step_err) ||
        !e->reserve_checkpoint_allowance(cfg, &step_err) ||
        !e->admit(cfg, &step_err)) {
        return fail(step_err);   // e's destructor releases what was acquired
    }
    e->vocab_ = llama_model_get_vocab(e->model_);
    return e;
}

// ALL FOUR floor terms, and CHECKED. PrefillActivation was priced by the plan
// (up to ~940 MB at 32k context) but never reserved, so the streamer's cache
// budget quietly exceeded what plan.cache_budget promised by exactly that amount
// -- the over-claim rebudget_against_rss then had to claw back. And every
// reserve() result was discarded: on a refusal the floor bytes stayed off the
// ledger at exactly the moment the cap was tightest (2026-08-24 audit).
bool Engine::reserve_floor(std::string* err) {
    acct_ = std::make_unique<mem::Accountant>(cap_);
    const plan::Floor& f = plan_->floor;
    const bool ok =
        acct_->reserve(mem::Category::KvCache, f.kv_cache) &&
        acct_->reserve(mem::Category::RecurrentState, f.recurrent_state) &&
        acct_->reserve(mem::Category::ComputeScratch, f.compute_scratch) &&
        acct_->reserve(mem::Category::PrefillActivation, f.prefill_activation);
    if (!ok) {
        *err = "REFUSED: the mandatory floor does not fit the cap (the plan said it "
               "would; this is a planner/ledger disagreement worth reporting)";
    }
    return ok;
}

bool Engine::start_streamer(const EngineConfig& cfg, bool gpu_consent, std::string* err) {
    backend::Config scfg;
    scfg.cap = cfg.cap;
    scfg.queue_depth = 64;
    scfg.slow_load = env_is_one("DRAY_SLOW_LOAD");
    scfg.no_compact = env_is_one("DRAY_NO_COMPACT");
    scfg.n_seq = sequences(cfg);   // churn reserve scales with the union working set
#if defined(__APPLE__)
    // Tier 2 v1: Metal computes correctly over the mapped arena but its
    // MUL_MAT_ID kernels assume canonical strides -- compacted slot tensors
    // (nb[2] = slot size) read at the wrong stride and produce fluent garbage
    // (measured: testbed, compact on = "Takd", compact off = CPU-identical).
    // Until the kernel honors nb[2], Metal forces whole-tensor mode, loudly.
    // This is also why GPU stays opt-IN on Apple: a silent default must never
    // change the bytes story.
    if (gpu_consent && !scfg.no_compact) {
        std::fprintf(stderr, "metal: forcing whole-tensor mode (compaction strides unsupported by Metal MUL_MAT_ID)\n");
        scfg.no_compact = true;
    }
#else
    (void)gpu_consent;
#endif
    streamer_ = std::make_unique<backend::Streamer>(*acct_, *plan_, scfg);
    if (!streamer_->valid()) {
        *err = "streamer unavailable: " + streamer_->error();
        return false;
    }
    return true;
}

// RESIDENT vs STREAMING, then the load itself.
//
// RESIDENT MODE (owner decision 2026-08-22, opt-OUT): when the whole model fits
// the cache budget, streaming buys nothing and costs ~3x -- our buffer type is
// not one of ggml-cpu's "extra buffer types", so it never reaches the
// repacked-weight matmul kernels (measured: stock 0.70 s/tok vs ours 2.2 on the
// same 27B and CPU). Proven byte-correct by scripts/residenttest.ps1 and
// measured 3x faster on a 27B (1.2 vs 3.6 s/token). --force-stream (or
// DRAY_FORCE_STREAM=1) pins the streaming path and is MANDATORY in every
// correctness gate and measurement, or they silently stop testing this engine.
bool Engine::load_model(const EngineConfig& cfg, bool gpu_consent, std::string* err) {
    uint64_t model_bytes = 0;
    for (const auto& t : plan_->tensors) model_bytes += t.bytes;
    const bool forced = cfg.force_stream || env_is_one("DRAY_FORCE_STREAM");
    resident_mode_ = !forced && model_bytes > 0 && model_bytes <= plan_->cache_budget;

    load_->buft_overrides[0] = { ".*", streamer_->buft() };
    load_->buft_overrides[1] = { nullptr, nullptr };

    llama_model_params mp = llama_model_default_params();
    // WEIGHTS NEVER LIVE IN VRAM. --gpu means "use the GPU for the work it is
    // good at", not "move the model there": with n_gpu_layers 0 the weights stay
    // in our accounted host memory and ggml still offloads ops above its
    // batch-size threshold, which is prefill and not decode. Without this,
    // resident mode plus --gpu tried to put a 17.5 GB model into 8 GB of VRAM.
    // CPU-only runs keep llama's own default: forcing 0 there changed its
    // preferred buffer type and diverged residenttest.
    if (gpu_consent) mp.n_gpu_layers = 0;
    // Invariant 2 on both paths: tensor data is never mmap'd. DIRECT_IO reads
    // without retaining page cache.
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    if (resident_mode_) {
        // Account the weights we are about to hand off, so the cap keeps
        // meaning what it says even though llama does the allocating.
        acct_->reserve(mem::Category::ExpertCache, model_bytes);
        std::fprintf(stderr,
                     "resident mode: the whole model (%.2f GB) fits the cache budget, so "
                     "llama allocates it natively and reaches its repacked-weight kernels "
                     "(~3x faster than our streaming buffer). Pass --force-stream to "
                     "measure the streaming path instead.\n",
                     model_bytes / 1e9);
    } else {
        mp.tensor_buft_overrides = load_->buft_overrides;
    }

    // Without runtime consent, restrict llama to CPU devices -- a
    // backend-carrying binary must never engage a GPU by surprise.
    if (!gpu_consent) {
        auto& devs = load_->cpu_devices;
        devs.clear();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) devs.push_back(d);
        }
        devs.push_back(nullptr);
        mp.devices = devs.data();
    }

    const models::Profile* prof = models::find_profile(plan_->arch);
    if (!build_kv_overrides(cfg.kv_overrides, prof, &load_->kv_overrides, err)) return false;
    if (!load_->kv_overrides.entries.empty()) {
        for (const auto& o : load_->kv_overrides.entries) {
            std::fprintf(stderr, "kv-override: %s (%s)\n", o.key,
                         prof && prof->kv_defaults ? "profile default or CLI" : "CLI");
        }
        mp.kv_overrides = load_->kv_overrides.terminated();
    }

    if (resident_mode_) {
        // llama's REAL loader: the streaming path's no-op loader below would
        // leave resident tensors uninitialised (the original garbage bug).
        model_ = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    } else if (!env_is_one("DRAY_SLOW_LOAD")) {
        // The metadata path reads nothing: every streamed tensor's bytes stay on
        // disk until a node needs them.
        std::string merr;
        meta_ = std::make_unique<backend::MergedMetadata>(
            backend::merge_shard_metadata(*plan_, &merr));
        if (!meta_->gguf) {
            *err = "metadata merge failed: " + merr;
            return false;
        }
        auto no_load = [](ggml_tensor*, void*) {};
        model_ = llama_model_init_from_user(meta_->gguf, no_load, nullptr, mp);
    } else {
        model_ = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    }
    if (!model_) { *err = "failed to load model"; return false; }
    return true;
}

// Serving fluent text from wrong weights is worse than not serving. In resident
// mode we own no tensors -- llama allocated and verified them through its own
// loader -- so the check DOES NOT APPLY, and we say so instead of pretending.
bool Engine::verify_floor(std::string* err) {
    const backend::Streamer::SelfCheck sc = streamer_->self_check();
    if (resident_mode_ && sc.checked == 0) {
        std::fprintf(stderr,
                     "floor integrity: NOT APPLICABLE in resident mode (llama owns the "
                     "weights; its loader verified them, we did not)\n");
    } else if (!sc.ok()) {
        *err = "floor integrity check failed: " + std::to_string(sc.mismatched) +
               " of " + std::to_string(sc.checked) + " resident tensors mismatch";
        return false;
    } else {
        std::fprintf(stderr, "floor integrity: %zu resident tensors match the file\n", sc.checked);
    }
    return true;
}

// T16: the backend identity on every run path, and the Invariant 2 gate. A
// DEGRADED backend retains page cache, which makes the cap meaningless and every
// measured byte suspect -- a refusal, not a warning, unless explicitly accepted.
bool Engine::check_storage_backend(std::string* err) {
    const std::string iod = streamer_->io_describe();
    std::fprintf(stderr, "backend: %s\n", iod.c_str());
    if (iod.find("DEGRADED") == std::string::npos) return true;
    if (!env_is_one("DRAY_ALLOW_DEGRADED")) {
        *err = "REFUSED: storage backend is DEGRADED (uncached mode "
               "unavailable), so the memory cap cannot be enforced "
               "(Invariant 2). Set DRAY_ALLOW_DEGRADED=1 to run "
               "anyway; measurements from such a run are contaminated.";
        return false;
    }
    std::fprintf(stderr, "WARNING: DEGRADED backend accepted by "
                         "DRAY_ALLOW_DEGRADED=1; the cap is not "
                         "enforceable and measurements are contaminated\n");
    return true;
}

bool Engine::create_context(const EngineConfig& cfg, bool gpu_consent, std::string* err) {
    llama_context_params cp = llama_context_default_params();
    const uint32_t n_seq = sequences(cfg);
    // J1: llama splits this across n_seq PRIVATE per-sequence rings of ctx
    // cells each (kv_unified false); the plan funded exactly that.
    cp.n_ctx = context_cells(cfg) * n_seq;
    // SEAMLESS GPU (2026-08-22): keep the KV cache and attention on the CPU.
    // With KQV offloaded, decode ran 2.6 s/token against 2.0 on CPU because
    // attention followed the cache into VRAM and the graph split 852 ways.
    // Vulkan offloads only ops above op_offload_min_batch_size, so PREFILL
    // still goes to the GPU while DECODE stays on CPU. It also keeps KV inside
    // the accounted RAM cap.
    if (gpu_consent) cp.offload_kqv = false;
    const KvChoice kv = kv_choice(cfg.kv_quant);
    if (kv.type != GGML_TYPE_F16) {
        cp.type_k = kv.type;
        cp.type_v = kv.type;
        // llama requires flash attention for a quantized V cache.
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    cp.n_seq_max = n_seq;
    cp.n_batch = prefill_batch_;
    // PHYSICAL batch, which is what actually gets computed in one graph. It
    // defaulted to 512 no matter what n_batch said, so raising --prefill-chunk
    // alone could never buy anything.
    cp.n_ubatch = prefill_batch_;

    const ThreadChoice th = choose_threads(
        cfg.n_threads, static_cast<int>(std::thread::hardware_concurrency()));
    if (th.clamped) {
        std::fprintf(stderr,
                     "compute threads: %d requested, CLAMPED to %d -- this process runs "
                     "~%d threads of its own and ggml spin-waits, so exceeding %d cores "
                     "collapses throughput (measured 20x+).\n",
                     th.requested, th.ceiling, kOwnThreads,
                     static_cast<int>(std::thread::hardware_concurrency()));
    }
    cp.n_threads = th.decode;
    cp.n_threads_batch = th.prefill;
    std::fprintf(stderr, "compute threads: %d decode, %d prefill (ceiling %d)\n",
                 cp.n_threads, cp.n_threads_batch, th.ceiling);

    // The interception hook. Resident mode: llama owns the weights, so the
    // streamer must NOT touch graph nodes (repointing tensors it does not own
    // produced "!!!!!!!!" on the first resident run).
    if (!resident_mode_) {
        cp.cb_eval = [](ggml_tensor* t, bool ask, void* ud) -> bool {
            auto* s = static_cast<backend::Streamer*>(ud);
            if (ask) {
                // False lets ggml batch this node with its neighbours into ONE
                // compute call; true forces a single-node call plus a full
                // thread-pool barrier. After materialising it MUST be true:
                // batching would let the next node's materialise evict this
                // one's weights before it computes.
                if (!s->needs(t)) return false;
                s->materialise(t);
                return true;
            }
            s->release(t);
            return true;
        };
        cp.cb_eval_user_data = streamer_.get();
    }
    cp.abort_callback = &backend::Streamer::abort_cb;
    cp.abort_callback_data = streamer_.get();
    lctx_ = llama_init_from_model(model_, cp);
    if (!lctx_) { *err = "failed to create context"; return false; }
    return true;
}

// F4/G1: the checkpoint allowance, reserved ONCE so it subtracts from the cache
// budget like every other floor item (Invariant 1). Sized from the PLANNER's
// geometry -- the honest upper bound of a full state blob -- NOT from
// llama_state_seq_get_size on the fresh (empty) context, which returns 16 bytes
// and once turned this gate into dead code. Larger blobs fall back to
// per-capture reserve-or-refuse inside SnapshotCache.
bool Engine::reserve_checkpoint_allowance(const EngineConfig& cfg, std::string* err) {
    if (!cfg.reserve_checkpoint) return true;
    const uint64_t blob = plan_->floor.kv_cache + plan_->floor.recurrent_state +
                          (16u << 20);   // serializer framing headroom
    if (!acct_->reserve(mem::Category::Misc, blob)) {
        *err = "REFUSED: cap cannot hold a checkpoint blob (" +
               std::to_string(blob >> 20) + " MiB) on top of the floor; "
               "raise --cap, lower --ctx, or serve without --jobs-dir";
        return false;
    }
    ckpt_allowance_ = blob;
    std::fprintf(stderr, "checkpoint allowance: %llu MiB reserved (jobs enabled)\n",
                 static_cast<unsigned long long>(blob >> 20));
    return true;
}

// ADMISSION, now that llama's KV and compute buffers exist and the ledger can be
// reconciled against the OS (the cap binds on the worse of RSS and commit).
bool Engine::admit(const EngineConfig& cfg, std::string* err) {
    const uint64_t worst = std::max<uint64_t>(
        mem::Accountant::process_rss(),
        mem::Accountant::process_committed());
    if (worst) streamer_->rebudget_against_rss(worst);

    const uint32_t n_seq = sequences(cfg);
    const uint64_t need = streamer_->largest_streamed_bytes();
    const uint64_t have = streamer_->rebudget_against_rss(worst);
    // The batch working set is admission's problem too, bounded by MEASURED
    // lines only: below 1x the widest union region the run certainly dies
    // mid-step (B=38, budget 0.71x, died); at 2x it ran clean (B=32).
    const uint64_t region = streamer_->batch_region_bytes();
    if (n_seq > 1 && region > have) {
        const uint64_t suggest = cfg.cap + (region - have);
        *err = "REFUSED: cap leaves " + std::to_string(have >> 20) +
               " MiB of cache but one batch-" + std::to_string(n_seq) +
               " union region needs " + std::to_string(region >> 20) +
               " MiB; lower --batch or try --cap " +
               std::to_string((unsigned long long)std::ceil(
                   static_cast<double>(suggest) / (1024.0 * 1024 * 1024))) + "G";
        return false;
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
        // figures here are MiB/GiB (binary) throughout.
        *err = "REFUSED: cap leaves " + std::to_string(have >> 20) +
               " MiB of cache but " + streamer_->largest_streamed_name() +
               " needs " + std::to_string(need >> 20) +
               " MiB whole; try --cap " +
               std::to_string((unsigned long long)std::ceil(
                   static_cast<double>(suggest) / (1024.0 * 1024 * 1024))) + "G";
        return false;
    }
    return true;
}

}  // namespace dray::engine
