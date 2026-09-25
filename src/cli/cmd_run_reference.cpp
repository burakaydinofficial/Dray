#include "cli/commands.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

#include "backend/stream_buffer.h"
#include "mem/accountant.h"
#include "plan/residency.h"
#include "report/readout.h"
#include "tools/repack.h"

namespace dray::cli {


// Reconciles the resident-byte ledger against what the OS says the process holds,
// and REPORTS the result either way.
//
// Invariant 1 -- the cap means total resident bytes -- is this project's central
// claim, so a run that quietly exceeded it would be precisely the dishonesty the
// readout rules exist to prevent. Called after the context is built and again
// during generation, because the gap does not appear all at once: llama.cpp's
// buffers land at context creation, while staging and fragmentation accumulate.
static void reconcile_cap(dray::backend::Streamer& s, uint64_t cap, const char* when) {
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


}  // namespace dray::cli
