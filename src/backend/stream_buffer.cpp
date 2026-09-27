// The Streamer's coordination: what happens at load, and per graph node.
//
// The components it coordinates, and the member order that fixes their
// lifetimes, are in streamer_impl.h. ggml reaches this file two ways: the
// buffer-type callbacks (stream_buffer_type.cpp), which forward load-time events
// to on_init_tensor / on_get_tensor, and the cb_eval hook the engine installs,
// which calls needs() / materialise() before a node computes and release() after.

#include "backend/stream_buffer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"

#include "backend/stream_budget.h"
#include "backend/stream_buffer_type.h"
#include "backend/streamer_impl.h"

namespace dray::backend {

// ---------------------------------------------------------------------------
// load time

ggml_status Streamer::Impl::on_init_tensor(ggml_tensor* t) {
    // VIEWS FIRST. init_tensor is not only called at load: ggml_gallocr_init_tensor
    // calls ggml_backend_view_init for every view whose source lives in our buffer,
    // on EVERY graph allocation. That has already set
    //     t->data = t->view_src->data + t->view_offs
    // and then dispatches here -- so the old code overwrote a correct alias with
    // fresh uninitialised memory, and registered a permanently-pinned entry keyed
    // by a per-graph tensor pointer. Two consequences: the view stopped aliasing
    // its weight (reading uninitialised bytes, not even the 0xA5 poison, so it
    // failed plausibly rather than loudly), and resident entries accumulated every
    // decode -- 746 -> 751 -> 754 across successive runs, unaccounted against the
    // cap that Invariant 1 exists to hold.
    if (t->view_src != nullptr) {
        return GGML_STATUS_SUCCESS;
    }

    const Source* found = tensors.source_named(t->name);
    if (!found) {
        // NO DISK SOURCE. llama.cpp creates a few tensors of its own, and those are
        // fine here: give them real memory so they behave normally. A real WEIGHT
        // reaching this path is silent corruption: it gets memory and is never
        // read, so on the no-read load path it holds whatever the allocator
        // returned. That is the "!!!! with zero reported failures" failure mode,
        // and nothing downstream can detect it.
        //
        // So say so, loudly and once per name. Anything listed here that looks like
        // a model weight is a classifier gap, not a curiosity.
        if (tensors.note_unsourced(t->name)) {
            std::fprintf(stderr, "[dray] NO SOURCE: %s (%.3f MB) -- allocated, never read\n",
                         t->name, ggml_nbytes(t) / 1e6);
            std::fflush(stderr);
        }
        void* m = mem.alloc(mem::Category::Misc, ggml_nbytes(t), kHostAlign);
        t->data = m;
        Resident r; r.mem = m; r.bytes = ggml_nbytes(t); r.pinned = true;
        r.cat = mem::Category::Misc;
        cache.add_permanent(t, r);
        return m ? GGML_STATUS_SUCCESS : GGML_STATUS_ALLOC_FAILED;
    }

    const Source s = *found;
    tensors.bind(t, s);

    if (s.pinned) {
        // Mandatory floor: resident for the life of the run. Router gates in
        // particular MUST be resident -- a non-resident gate has to be read before
        // you know which experts to read, adding a serialized round trip per layer.
        // Floor tensors are charged to their real category, so the startup report
        // shows where a user's 4 GB actually went rather than lumping them in cache.
        const mem::Category fcat = s.is_router_gate ? mem::Category::RouterGates
                                                    : mem::Category::NormsAndBiases;
        void* m = mem.alloc(fcat, ggml_nbytes(t), kHostAlign);
        if (!m) return GGML_STATUS_ALLOC_FAILED;

        // READ IT HERE, from disk, rather than waiting for set_tensor to hand us
        // bytes. This file's whole premise is that we know where every tensor lives
        // and fetch it ourselves; relying on set_tensor quietly made the floor
        // depend on llama.cpp reading the model first. On the no-read path nothing
        // calls set_tensor, so the router gates and norms held uninitialised memory
        // and the model emitted "!!!!" with zero reported failures -- silent
        // corruption produced by an inconsistency between two load paths.
        if (!io.read_exact(s, m, ggml_nbytes(t))) {
            mem.free(fcat, m, ggml_nbytes(t));
            ++failures;
            return GGML_STATUS_FAILED;
        }
        t->data = m;
        Resident r; r.mem = m; r.bytes = ggml_nbytes(t); r.pinned = true;
        r.cat = fcat;
        r.repacked = repacker.maybe_repack(t);
        cache.add_permanent(t, r);
        return GGML_STATUS_SUCCESS;
    }

    // Streamed: no storage now, and none until the node that needs it runs.
    //
    // data must still be NON-NULL: ggml_backend_tensor_set asserts
    // `tensor->data != NULL && "tensor not allocated"` before it ever reaches our
    // set_tensor, so a null here aborts the load. The sentinel is never
    // dereferenced -- our set_tensor drops the bytes, and materialise() installs a
    // real pointer before any node computes.
    t->data = poison.sentinel();
    return GGML_STATUS_SUCCESS;
}

void Streamer::Impl::on_get_tensor(const ggml_tensor* t, void* data, size_t offset, size_t size) {
    const Resident* r = cache.find(t);
    if (r && r->mem) {
        std::memcpy(data, static_cast<const uint8_t*>(r->mem) + offset, size);
        return;
    }
    // Zeros presented as data, silently, was the worst available behaviour: any
    // llama.cpp read-back of a streamed (non-resident) weight got fabricated
    // bytes with no failure counted. Still zero-fill -- callers do not check --
    // but count it and say so once, so a run that hit this path can never again
    // look clean (2026-08-24 audit).
    ++failures;
    if (failures <= 8) {
        std::fprintf(stderr,
                     "[dray] GET_TENSOR MISS: %s read back before materialisation; "
                     "returning zeros and counting a failure\n",
                     t->name);
    }
    std::memset(data, 0, size);
}

// ---------------------------------------------------------------------------
// lifetime

Streamer::Streamer(mem::Accountant& acct, const plan::Plan& p, Config cfg)
    : impl_(new Impl(acct, p, cfg)) {
    Impl& im = *impl_;

    // The backend, every shard uncached, and the discovered alignment.
    if (!im.io.open(p.shard_paths, cfg.queue_depth, &error_)) return;

    // Every tensor's true location, from the GGUF tensor table. Router gates and
    // norms are pinned; everything else streams, including attention and shared
    // experts -- at a 4 GB cap those do not fit either.
    for (const plan::TensorInfo& t : p.tensors) {
        Source s;
        s.shard  = t.shard;
        s.offset = t.offset;
        s.bytes  = t.bytes;
        s.pinned = (t.cls == plan::TensorClass::RouterGate ||
                    t.cls == plan::TensorClass::NormOrBias);
        s.is_router_gate = (t.cls == plan::TensorClass::RouterGate);
        s.disk_stride = t.disk_stride;
        // Routed experts are compacted; the token embedding is row-sliced. Neither
        // ever needs its full size resident, so neither bounds the minimum cap.
        s.sliceable = (t.cls == plan::TensorClass::RoutedExpert) ||
                      (t.name.find("token_embd") != std::string::npos);
        s.routed = (t.cls == plan::TensorClass::RoutedExpert);   // I2: class matters per-lever
        im.tensors.add_source(t.name, s);
    }

    im.fake_base = mem::aligned_alloc_host(1 << 16, kHostAlign);
    if (!im.fake_base) { error_ = "cannot allocate base sentinel"; return; }

    // Every size the streamer carves from its share of the cap, from the Plan
    // alone (stream_budget.h). Pure arithmetic: nothing is allocated yet.
    const StreamBudget budget = size_stream(p, cfg, im.io.align(), im.flags);

    // The poison sentinel every streamed tensor points at until materialised
    // (and the slow-load landing zone), charged to IoStaging.
    if (!im.poison.create_sentinel(budget.scratch_bytes, &error_)) return;

    // The budget is whatever the cap has left after every non-cache category, read
    // live from the Accountant. It is NOT stored: the floor is charged during
    // init_tensor, long after this point, and a stored copy would not see it.
    if (im.mem.cache_budget() == 0) { error_ = "cap leaves nothing for the paging cache"; return; }

    // Churn reserve, ring and eslot pool: every figure is derived from the Plan
    // alone, and each one's measured history lives with its arithmetic in
    // stream_budget.cpp. Only the ring allocation itself happens here, and the
    // eslot pool is sized against the ring actually allocated.
    im.cache.set_churn_reserve(budget.churn_reserve);
    im.batch_region_bound = budget.batch_region_bound;
    if (budget.ring_target) im.ring.allocate(budget.ring_target, im.io.align());
    im.slots.set_pool(budget.eslot_pool(im.ring.bytes()));

    // Static pinning takes everything except the current layer's reserve. It does
    // NOT set aside room for speculative reads. The per-tensor prefetch that
    // preceded the ring (replaced in 89cfcdc, deleted later) measured why, on
    // Qwen3.8 UD-IQ1_S at 12 GiB, 4 tokens:
    //
    //   batched only             31.5 s/tok   206.3 GB
    //   + prefetch, no evict     35.2 s/tok   208.8 GB   304/304 hits
    //   + prefetch, may evict    36.7 s/tok   211.5 GB   673/673 hits
    //
    // A perfect hit rate that loses on BOTH axes: every byte a speculative read
    // holds is a byte not pinned, re-read every token. Below the knee holding a
    // byte beats overlapping a read. The ring gets its own accounted line item
    // above instead of competing with pinning at runtime.

    // The ggml buffer type llama.cpp will allocate weights from.
    init_stream_buffer_type(im);
    // A: opt this buffer type into the optimised matmul kernels when repacking
    // is enabled (see repacker.h: measured negative, off by default).
    im.repacker.accept(&im.buft);
    im.buft.context = &im;
}

Streamer::~Streamer() {
    if (!impl_) return;
    Impl& im = *impl_;
    // Teardown order is load-bearing only where DMA is involved: pending regions
    // and ring segments settle every outstanding read before their memory is
    // released, and leak it -- loudly -- when the backend died with reads still
    // in flight. Everything returns memory by origin without crediting the
    // ledger; the poison buffers free themselves when Impl is destroyed.
    // The cache is destroyed before the scheduler that points at it: nothing
    // reclaims from here on.
    im.io.set_reclaimer(nullptr);
    im.cache.release_all();
    im.compactor.release_all();
    im.ring.release_all();
    im.slots.release_all();
    if (im.fake_base) mem::aligned_free_host(im.fake_base, 1 << 16);
}

ggml_backend_buffer_type_t Streamer::buft() { return &impl_->buft; }

// ---------------------------------------------------------------------------
// materialisation

// The ring producer runs inside both callbacks; its CPU time is not a wait on
// the drive, so it is counted on its own.
static void produce_timed(Streamer::Impl& im) {
    const auto t0 = std::chrono::steady_clock::now();
    im.ring.produce();
    im.ns_produce += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

bool Streamer::Impl::bring_in(ggml_tensor* t, uint64_t* streamed) {
    const auto t0 = std::chrono::steady_clock::now();
    auto took = [&](Source_ from) {
        ns_from[from] += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        ++n_from[from];
    };
    ring.record_consumption(t);

    // R13: a COMPACTED partial region (uniq non-empty: only the router-selected
    // experts, remapped) must never be served as the whole tensor -- a plain
    // MUL_MAT reading it would see k experts where n_experts belong. Both known
    // triggers are near-dead, but this file exists to prevent exactly that.
    // T20: and the refusal must RECLAIM, not orphan -- falling through to a
    // whole-tensor read that copy-assigns over the entry left the old region
    // permanently charged and invisible to make_room (t is protected).
    if (const Resident* r = cache.find(t); r && r->mem && !r->uniq.empty()) {
        cache.drop(t);
    }
    if (const Resident* r = cache.find(t); r && r->mem && r->uniq.empty()) {
        ring.retire_pending(t);
        t->data = static_cast<uint8_t*>(r->mem) + r->head;
        // H8: the RAM-hit sibling of the miss-path split -- a routed tensor's
        // hits and misses must land in ONE class or h_routed lies (Metal's
        // forced whole-tensor mode printed a hard 0% no matter the pinning).
        hits.add_for(t, ggml_nbytes(t), 0);   // served from RAM: read 0
        took(kFromCache);
        return true;
    }
    // The ring: by now the bytes are usually already in the arena, and this is a
    // pointer assignment. On failure fall through to the synchronous read.
    if (ring.consume(t, streamed)) { took(kFromRing); return true; }

    const Source* src = tensors.source_of(t);
    if (!src) return t->data != nullptr;   // not ours

    const uint64_t bytes = ggml_nbytes(t);

    // On ANY failure below, data must be left pointing at the poisoned landing
    // zone -- never at null. A null here is dereferenced by the CPU backend and
    // access-violates; the poison at least fails loudly in the output instead.
    auto fail = [&](const char* why) {
        if (failures < 8) {
            std::fprintf(stderr,
                "[dray] MATERIALISE FAIL %s: %s  need=%.2fGB in_use=%.2fGB budget=%.2fGB lru=%zu\n",
                t->name, why, bytes / 1e9, mem.cache_used() / 1e9, mem.cache_budget() / 1e9,
                cache.lru_size());
            std::fflush(stderr);
        }
        t->data = poison.poison_for(t);
        return false;
    };

    if (!cache.make_room(bytes)) return fail("no room");
    uint64_t alloc_bytes = 0;
    uint32_t head = 0;
    void* m = io.read_whole(mem::Category::ExpertCache, *src, bytes, &alloc_bytes, &head);
    if (!m) return fail("read");

    Resident nr; nr.mem = m; nr.bytes = alloc_bytes; nr.head = head;
    nr.refs = 0;
    // Whole-tensor materialisation is an unconditional weight: attention, shared
    // experts, embedding, lm_head. Read every token, so worth ~51x a routed byte.
    nr.prio = Prio::Unconditional;
    nr.pinned = cache.claim_static(alloc_bytes);
    t->data = static_cast<uint8_t*>(m) + head;
    nr.repacked = repacker.maybe_repack(t);
    cache.add(t, nr);
    *streamed += bytes;
    // Pass-4b: a whole-tensor read of a ROUTED (expert-fused, ne[2]>1) tensor
    // reaches here via the compaction-decline fallthrough; charging it to the
    // unconditional class conflated the two hit rates the readout exists to
    // keep apart. Bytes-read is unchanged; only the class split is.
    hits.add_for(t, bytes, bytes);
    bytes_from_disk += bytes;
    took(kFromDisk);
    return true;
}

// Does this node need anything from us? ggml executes NODE BY NODE with a full
// ggml_backend_synchronize after every node for which the eval callback answers
// true (ggml-backend.cpp, the callback_eval branch). Answering true for every
// node -- which this engine did until 2026-08-22 -- means one thread-pool
// barrier per node, ~4300 per token, and it is the whole of the 3x gap against
// stock: it also explains why threads never helped and why 22 of them
// collapsed (barrier cost scales with pool size). Nodes with no disk-backed
// source need nothing from us and must be batched by ggml instead.
bool Streamer::needs(ggml_tensor* node) {
    if (!node || !impl_) return false;
    Impl& im = *impl_;
    // Claiming every node costs a thread-pool barrier each. DRAY_FAST_NODES=1
    // opts into the predicate below, which is now CORRECT on all five
    // architectures in archgate -- the earlier divergence was a real bug, found
    // by reading rather than guessing: a skipped node never gets its VIEWS
    // re-pointed, and ggml fixes a view's data pointer at graph-allocation time.
    // See the K3 ssm_a note in materialise().
    //
    // It is worth 11-15%%, not the 42%% first reported: that figure came from the
    // BROKEN version, which was fast partly because it skipped work it needed to
    // do. Claiming view nodes takes back about half the skips (122B: 152,295 ->
    // 78,903 claims, 2.7 -> 2.4 s/token; 27B: 65,399 -> 36,771, 2.6 -> 2.2).
    // Re-verified 2026-08-24 after the instrumentation edits: archgate 5/5 with
    // the flag ON, so the correctness claim is current, not inherited.
    if (!im.flags.fast_nodes) return true;
    // Loud once per run. The flag is opt-in because archgate can only vouch for
    // the architectures it has: an untested graph shape could still reach a
    // skipped node through a path the predicate below does not follow.
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "\n*** DRAY_FAST_NODES=1: experimental node skipping ***\n"
                     "    Verified on the five archgate architectures (2026-08-24).\n"
                     "    Check YOUR model with scripts/archgate.ps1 before believing a token.\n\n");
    }

    // EXACT TEST, replacing the op enumeration that broke DeepSeek: a node needs
    // us if and only if it reads a tensor allocated in OUR buffer type. Buffer
    // ownership is a fact ggml already tracks; "which op kinds can reach our
    // memory" was a guess, and a new architecture falsified it. Views are
    // followed to their root because a view of a streamed weight is a streamed
    // weight.
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        for (ggml_tensor* v = node->src[i]; v; v = v->view_src) {
            if (v->buffer && v->buffer->buft == &im.buft) return true;
        }
    }
    for (ggml_tensor* v = node; v; v = v->view_src) {
        if (v->buffer && v->buffer->buft == &im.buft) return true;
    }
    if (im.ids.owns(node)) return true;
    // HYPOTHESIS UNDER TEST: a node whose src is a VIEW must claim even if the
    // view root is not ours. materialise re-points views EVERY node, because
    // ggml fixes a view's data pointer at graph-allocation time (this is the K3
    // ssm_a bug documented below). A skipped node never gets that repointing.
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] && node->src[i]->view_src) return true;
    }
    return false;
}

bool Streamer::materialise(ggml_tensor* node) {
    const auto t_enter = std::chrono::steady_clock::now();
    struct TimeIt {
        Streamer::Impl* im; const ggml_tensor* node; std::chrono::steady_clock::time_point t0;
        ~TimeIt() {
            const uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
            im->ns_materialise += ns;
            ++im->n_materialise;
            const int c = wait_class();
            im->ns_wait[c] += ns;
            ++im->n_wait[c];
        }
        int wait_class() const {
            if (!node) return Streamer::Impl::kWaitOther;
            if (node->op == GGML_OP_MUL_MAT_ID) return Streamer::Impl::kWaitRouted;
            if (node->op == GGML_OP_GET_ROWS && node->src[0] && im->tensors.is_bound(node->src[0]))
                return Streamer::Impl::kWaitRows;
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (node->src[j] && im->tensors.is_bound(node->src[j])) return Streamer::Impl::kWaitWeights;
            }
            return Streamer::Impl::kWaitOther;
        }
    } timeit{impl_.get(), node, t_enter};
    if (!node) return true;
    Impl& im = *impl_;
    ++nodes_;

    // Loading is over by the first node (floor reads and self_check drove the
    // queue themselves): from here the I/O thread owns the device.
    if (im.flags.io_thread && !im.io.threaded()) im.io.start_thread();

    // DRAY_TRACE=1 prints every node as it is materialised, unbuffered, so the
    // last line before a fault names the node that faulted. Guessing at which node
    // crashed from an exit code cost several wrong diagnoses.
    if (im.flags.trace) {
        std::fprintf(stderr, "[dray] node=%s op=%s ne2=%lld in_use=%.2fGB\n",
                     node->name, ggml_op_name(node->op),
                     static_cast<long long>(node->ne[2]), im.mem.cache_used() / 1e9);
        std::fflush(stderr);
    }

    // ONLY MUL_MAT_ID is compacted, and only its src[0].
    //
    // ADD_ID (bias, indexed at dim 1) and the GET_ROWS on the per-expert scale
    // vector inside build_lora_mm_id also consume the ids -- but because the shared
    // ids are never modified, they simply see real expert ids and index whole
    // tensors correctly. They need no special handling, which is the point: the
    // previous design required knowing about every ids consumer, and that set is
    // not knowable from here.
    //
    // Bisect switch. DRAY_NO_COMPACT=1 materialises whole expert tensors
    // instead: much slower, but it isolates a fault to compaction vs paging.
    // cfg.no_compact ALSO gates here: the Metal interlock forces whole-tensor
    // mode through the config, and a gate that only read the env made that
    // force vacuous (swarm finding R4 -- the sibling-path law, again).
    const bool no_compact = im.flags.no_compact || im.cfg.no_compact;

    // Separately gateable from expert compaction: they are different mechanisms on
    // different node types, and a bisect that cannot tell them apart wastes a cycle.
    const bool no_rowslice = im.flags.no_rowslice;

    // Forward-pass boundary: the embedding lookup is the first op of every pass.
    if (node->op == GGML_OP_GET_ROWS && node->src[0]) {
        const Source* bs = im.tensors.source_of(node->src[0]);
        if (bs && bs->sliceable) {
            ++im.pass_count;
            im.slots.on_pass();
            im.ring.on_pass(im.pass_count);
        }
    }

    // GET_ROWS on a streamed table: read only the rows this node asks for. Without
    // this, token_embd.weight (1.09 GiB) is materialised whole every token to read
    // a handful of ~7.5 KB rows, and it alone sets the floor under the minimum cap.
    if (node->op == GGML_OP_GET_ROWS && node->src[0] && node->src[1] &&
        im.tensors.is_bound(node->src[0]) && !no_compact && !no_rowslice) {
        ggml_tensor* w = node->src[0];
        im.cache.protect_none();
        im.cache.protect(w);
        if (im.compactor.compact_rows(node, w, im.ids.original(node, node->src[1]),
                                      &bytes_streamed_)) {
            ++im.compacted;
            return true;
        }
        // Declined (too many rows, or no room). F9: the NODE may still hold a
        // previous token's private compact ids (never restored); reading the
        // whole table through them is rows 0..k-1, fluent garbage. Restore the
        // original ids before the fallthrough.
        im.ids.restore(node, 1);
        if (im.bring_in(w, &bytes_streamed_)) return true;

        // Failed outright. Same guard as the MUL_MAT_ID path: all-zero private ids
        // so every lookup lands in the first row and cannot read past the poison
        // buffer. Real token ids reach 151,936 against an 8 MB sentinel, which is
        // an access violation rather than a wrong answer.
        im.ids.install_zeros(node, 1);
        w->data = im.poison.sentinel();
        ++im.failures;
        return true;
    }

    // The disk_stride refusal below lives inside the !no_compact branch, so with
    // compaction disabled -- by the env lever, or by the Metal interlock that
    // forces whole-tensor mode -- a repacked companion fell straight through to
    // the generic path and was read CONTIGUOUSLY. Its experts are interleaved on
    // disk, so that read succeeds at full length and assembles a tensor out of
    // the wrong bytes: exactly the silent-wrong-weights outcome the guard was
    // written to prevent, reachable by going around it (2026-08-24 audit).
    if (no_compact && node->op == GGML_OP_MUL_MAT_ID && node->src[0]) {
        const Source* bsrc = im.tensors.source_of(node->src[0]);
        if (bsrc && bsrc->disk_stride) {
            std::fprintf(stderr,
                "[dray] REFUSED: %s is expert-interleaved on disk (a repacked "
                "companion), and compaction is disabled, so it cannot be read "
                "whole. Drop DRAY_NO_COMPACT, or use the original model file.\n",
                node->src[0]->name);
            std::fflush(stderr);
            ++im.failures;
            node->src[0]->data = im.poison.poison_for(node->src[0]);
            return false;
        }
    }
    if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] && node->src[2] &&
        im.tensors.is_bound(node->src[0]) && !no_compact) {
        ggml_tensor* w = node->src[0];
        im.cache.protect_none();
        im.cache.protect(w);
        if (im.compactor.compact_experts(node, w, im.ids.original(node, node->src[2]),
                                         &bytes_streamed_)) {
            ++im.compacted;
            return true;
        }
        // Compaction declined. The SHARED ids were never touched -- but the NODE
        // may still point at a previous token's private compact ids (install is
        // never restored), and a whole tensor read through 0..k-1 ids is rows
        // 0..k-1 of the full tensor: fluent garbage with failures==0, the one
        // failure this file exists to prevent (F9). Restore the original ids
        // BEFORE the fallthrough; then the whole tensor is correctly indexable
        // -- expensive but right, and a tight cap keeps working.
        //
        // EXCEPT under a repacked companion: there the tensor's bytes are
        // expert-interleaved on disk, so a contiguous whole read would assemble
        // plausible garbage. Refuse loudly instead -- fluent output from a
        // misassembled tensor is the one failure this engine must never ship.
        im.ids.restore(node, 2);
        const Source* bsrc = im.tensors.source_of(w);
        if (bsrc && bsrc->disk_stride) {
            std::fprintf(stderr,
                "[dray] REPACK: whole-tensor fallback impossible for %s "
                "(expert-interleaved on disk); failing the node instead\n",
                w->name);
        } else if (im.bring_in(w, &bytes_streamed_)) {
            return true;
        }

        // Failed. Make the node structurally incapable of reading out of bounds
        // before the abort takes effect: point the weights at the poison buffer AND
        // give the node all-zero private ids, so every lookup lands in the first
        // stride. Leaving real expert ids against a small poison buffer is what
        // turned a materialise failure into an access violation. The run aborts
        // regardless -- this only guarantees the in-flight node cannot fault
        // before it does.
        im.ids.install_zeros(node, 2);
        w->data = im.poison.poison_for(w);
        ++im.failures;
        return true;   // see the cb_eval comment: the return value controls batching
    }

    // No refcounts. An earlier version incremented on materialise and decremented
    // in the post-callback, but the post-callback only fires when the ask returns
    // true -- so any failure leaked a reference, eviction starved, and every
    // subsequent tensor failed to materialise. Instead: protect exactly the
    // tensors this node needs, for the duration of this node.
    im.cache.protect_none();
    // VIEWS OVER STREAMED WEIGHTS.
    //
    // A src may be a view whose view_src is a weight we own. ggml_backend_view_init
    // fixes the view's data pointer from its parent at GRAPH-ALLOCATION time, which
    // is before anything is materialised -- so the view captures the poison sentinel
    // and keeps it. The parent is never a direct src of any node, so walking srcs
    // alone never materialises it and never notices.
    //
    // This is what broke Kimi K3. Its only use of ssm_a is
    //     A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1)
    // and ggml_reshape_3d returns a view. ssm_a was therefore never read from disk
    // on any layer of any token, and the KDA gate computed against 0xA5 -- fluent,
    // confidently wrong output with zero reported failures, because every mechanism
    // that could have caught it (failure counters, floor integrity, byte
    // verification) was looking at tensors that WERE materialised.
    //
    // Qwen3.8 never reshapes a streamed weight, which is why it was unaffected and
    // why "it works on the other model" proved nothing.
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> views;   // (view, parent)
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        ggml_tensor* s = node->src[i];
        if (!s) continue;
        if (im.tensors.is_bound(s)) { im.cache.protect(s); continue; }
        // Follow the view chain to its root; reshapes can nest.
        ggml_tensor* root = s->view_src;
        while (root && !im.tensors.is_bound(root) && root->view_src) {
            root = root->view_src;
        }
        if (root && im.tensors.is_bound(root)) {
            im.cache.protect(root);
            views.emplace_back(s, root);
        }
    }

    bool ok = true;
    for (ggml_tensor* s : im.cache.protected_tensors()) {
        if (!im.bring_in(s, &bytes_streamed_)) { ok = false; ++im.failures; }
    }
    // Re-point every view at where its parent actually landed. Must run AFTER
    // bring_in, which is what moves the parent.
    for (const auto& vp : views) {
        if (vp.second->data) {
            vp.first->data = static_cast<uint8_t*>(vp.second->data) + vp.first->view_offs;
        }
    }

    // Keep the drive fed: the ring producer replaces the old per-tensor prefetch,
    // whose admission guard was unsatisfiable below the knee (304 issues in 44,655
    // nodes). The ring's memory is its own line item, so the guard cannot starve.
    produce_timed(im);
    return ok;
}

void Streamer::release(ggml_tensor* node) {
    const auto t_enter_rel = std::chrono::steady_clock::now();
    struct RelTimeIt {
        Streamer::Impl* im; std::chrono::steady_clock::time_point t0;
        ~RelTimeIt() {
            im->ns_release += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
        }
    } reltimeit{impl_.get(), t_enter_rel};
    // Protection is implicit in the protected set, which the next materialise()
    // replaces. But a release callback is CPU time the drive can use.
    if (!impl_) return;
    Impl& im = *impl_;

    // PHASE C: if this node IS a router's ids tensor, its data was computed an
    // instant ago -- the earliest moment the layer's expert addresses exist.
    // Submit all three regions unwaited; the MUL_MAT_ID nodes a few small ops
    // later adopt them mid-flight. Refusals degrade to the Phase A path.
    im.compactor.unlock_early(node);

    produce_timed(im);
}

// ---------------------------------------------------------------------------
// integrity and budget

Streamer::SelfCheck Streamer::self_check(size_t max_tensors) {
    SelfCheck out;
    if (!impl_) return out;
    Impl& im = *impl_;

    // Re-read through the UNCACHED backend on FRESH handles, not through fopen.
    //
    // Invariant 2 is not negotiable and this runs on every start: buffered reads of
    // up to 32 resident tensors would dirty hundreds of MB of the user's page cache
    // each time (K3's router gates alone are 2.2 GB across 92 tensors). An earlier
    // version of this check used fopen/fread and did exactly that.
    //
    // Fresh handles rather than the streamer's own: it keeps the open/offset path
    // independently exercised. It is weaker than comparing against a completely
    // different I/O API -- a bug inside read_batch itself would be invisible here --
    // and that is the deliberate trade for not touching the page cache. The
    // differential compaction test covers the read path's logic separately.
    std::vector<io::FileId> refs;
    refs.reserve(im.plan.shard_paths.size());
    for (const std::string& sp : im.plan.shard_paths) {
        refs.push_back(im.io.open_file(sp));
    }

    std::vector<uint8_t> buf;
    for (const auto& kv : im.cache.entries()) {
        if (out.checked >= max_tensors) break;
        const Resident& r = kv.second;
        if (!r.pinned || !r.mem || r.bytes == 0) continue;
        if (r.repacked) { ++out.skipped_repacked; continue; }   // no longer file-shaped
        const Source* b = im.tensors.source_of(kv.first);
        if (!b) continue;          // no disk source to compare to
        const Source s = *b;
        if (s.shard < 0 || static_cast<size_t>(s.shard) >= refs.size()) continue;
        if (refs[static_cast<size_t>(s.shard)] == io::kInvalidFile) continue;

        // Through the fresh handle, with read_exact's alignment widening reused
        // verbatim.
        buf.assign(static_cast<size_t>(r.bytes), 0);
        const bool got = im.io.read_exact_via(refs[static_cast<size_t>(s.shard)], s,
                                              buf.data(), r.bytes);
        if (!got) continue;

        ++out.checked;
        if (std::memcmp(buf.data(), r.mem, buf.size()) != 0) ++out.mismatched;
    }

    for (io::FileId f : refs) if (f != io::kInvalidFile) im.io.close_file(f);
    return out;
}

uint64_t Streamer::rebudget_against_rss(uint64_t rss) {
    if (!impl_ || rss == 0) return impl_ ? impl_->mem.cache_budget() : 0;
    Impl& im = *impl_;

    // Reconcile the ledger against the OS, then let the derived budget do the rest.
    //
    // Whatever the process holds beyond what the ledger explains is real resident
    // memory -- llama.cpp's KV and compute buffers, the vocab, allocator overhead,
    // fragmentation -- and it counts against the cap like anything else. Recording
    // it shrinks cache_budget() automatically, because that is defined as cap minus
    // every non-cache category.
    //
    // charge_unreserved SETS rather than accumulates, so calling this on a timer
    // tracks the figure instead of compounding it.
    const uint64_t explained = im.mem.ledger().used() - im.mem.ledger().unreserved();
    im.mem.ledger().charge_unreserved(rss > explained ? rss - explained : 0);

    // Hand back anything now over the line. Pinned entries are not evictable, so
    // this can fail to reach the target -- report_over_cap() is what tells the user
    // that happened rather than leaving them to infer it from RSS. FIRST give the
    // shrink teeth: static pins claimed against the LOAD-TIME allowance are
    // demoted (never the floor) until the claim fits the allowance it now has.
    im.cache.demote_static_to_allowance();
    im.cache.shrink_to_budget();
    return im.mem.cache_budget();
}

std::string Streamer::accountant_report() const {
    if (!impl_) return "";
    std::string r = impl_->mem.ledger().report();
    const uint64_t unres = impl_->mem.ledger().unreserved();
    if (unres) {
        std::ostringstream o;
        o << "  unreserved (llama.cpp buffers, vocab, allocator overhead)  "
          << (unres / 1e9) << " GB";
        r += std::string("\n") + o.str();
    }
    return r;
}

// I2: ONE traversal for size and name, so the pair can never drift again --
// H9 fixed the drift, H14 reintroduced it one commit later by widening one
// sibling. And TWO predicates, not one: routed experts are compacted under
// NO_ROWSLICE alone (the gates differ, the streamer's own dispatch is the
// authority), so counting them "whole" there refused caps that would run.
static void largest_whole_read(const Streamer::Impl& im, uint64_t* bytes,
                               std::string* name) {
    const bool no_compact = im.flags.no_compact || im.cfg.no_compact;
    const bool env_no_rowslice = im.flags.no_rowslice;
    *bytes = 0;
    name->clear();
    for (const auto& kv : im.tensors.sources()) {
        if (kv.second.pinned) continue;          // floor is resident, not cached
        if (kv.second.sliceable) {
            const bool whole = kv.second.routed ? no_compact
                                                : (no_compact || env_no_rowslice);
            if (!whole) continue;
        }
        if (kv.second.bytes > *bytes) { *bytes = kv.second.bytes; *name = kv.first; }
    }
}

uint64_t Streamer::churn_reserve_bytes() const {
    return impl_ ? impl_->cache.churn_reserve() : 0;
}

uint64_t Streamer::batch_region_bytes() const {
    return impl_ ? impl_->batch_region_bound : 0;
}

uint64_t Streamer::largest_streamed_bytes() const {
    if (!impl_) return 0;
    uint64_t b = 0;
    std::string n;
    largest_whole_read(*impl_, &b, &n);
    return b;
}

std::string Streamer::largest_streamed_name() const {
    if (!impl_) return "";
    uint64_t b = 0;
    std::string n;
    largest_whole_read(*impl_, &b, &n);
    return n;
}

bool Streamer::over_cap() const {
    if (!impl_) return false;
    return impl_->mem.ledger().used() > impl_->mem.ledger().cap();
}

bool Streamer::aborted() const { return impl_ && impl_->failures > 0; }

bool Streamer::abort_cb(void* p) {
    auto* s = static_cast<Streamer*>(p);
    return s && s->aborted();
}

uint64_t Streamer::failures() const  { return impl_ ? impl_->failures.load(std::memory_order_relaxed) : 0; }
uint64_t Streamer::routed_bytes_read() const { return impl_ ? impl_->hits.routed_read : 0; }
std::string Streamer::io_describe() const {
    return impl_ ? impl_->io.describe() : std::string("no backend");
}
uint64_t Streamer::routed_bytes_needed() const { return impl_ ? impl_->hits.routed_needed : 0; }
uint64_t Streamer::uncond_bytes_needed() const { return impl_ ? impl_->hits.uncond_needed : 0; }
uint64_t Streamer::uncond_bytes_read() const   { return impl_ ? impl_->hits.uncond_read : 0; }
uint64_t Streamer::compacted() const { return impl_ ? impl_->compacted : 0; }

std::string Streamer::report() const {
    const Impl& im = *impl_;
    std::ostringstream o;
    o << "streamer: budget " << (im.mem.cache_budget() / 1e9) << " GB, in use "
      << (im.mem.cache_used() / 1e9) << " GB, " << im.cache.size() << " tensors resident, "
      << (bytes_streamed_ / 1e9) << " GB streamed over " << nodes_ << " nodes, "
      << im.compacted << " expert-compacted, "
      << "pool " << (im.mem.arena().enabled() ? im.mem.arena().committed() / 1e9 : -1.0) << " GB committed "
      << im.mem.arena().fallbacks() << " fallbacks, ";
    im.ring.append(o);
    o << (im.cache.static_used() / 1e9) << " GB pinned static of "
      << (im.cache.static_allowance() / 1e9) << " GB allowed, "
      << im.compactor.early_unlocks() << " early-unlocked, " << im.ring.promotions() << " promoted, "
      << im.slots.hits() << " eslot hits (" << (im.slots.bytes() / 1e9) << " of "
      << (im.slots.pool() / 1e9) << " GB pool)";

    // I/O FORENSICS, off unless asked for (DRAY_IO_STATS=1). These counters
    // located the bandwidth defect on 2026-08-24 and are worth keeping, but they
    // add 250 characters to a line that was already long. Default output is what
    // it was before that investigation.
    if (im.flags.io_stats) {
        o << ", cb " << (im.ns_materialise / 1000000)
          << "ms mat/" << (im.ns_release / 1000000) << "ms rel over " << im.n_materialise
          << " calls (mat by wait: routed " << (im.ns_wait[Impl::kWaitRouted] / 1000000)
          << "ms/" << im.n_wait[Impl::kWaitRouted] << ", rows "
          << (im.ns_wait[Impl::kWaitRows] / 1000000) << "ms/" << im.n_wait[Impl::kWaitRows]
          << ", weights " << (im.ns_wait[Impl::kWaitWeights] / 1000000) << "ms/"
          << im.n_wait[Impl::kWaitWeights] << ", other "
          << (im.ns_wait[Impl::kWaitOther] / 1000000) << "ms/" << im.n_wait[Impl::kWaitOther] << ")"
          << "; whole weights from cache " << (im.ns_from[Impl::kFromCache] / 1000000) << "ms/"
          << im.n_from[Impl::kFromCache] << ", ring " << (im.ns_from[Impl::kFromRing] / 1000000)
          << "ms/" << im.n_from[Impl::kFromRing] << ", disk " << (im.ns_from[Impl::kFromDisk] / 1000000)
          << "ms/" << im.n_from[Impl::kFromDisk] << " (" << (im.bytes_from_disk / 1000000000.0)
          << " GB); ring producer " << (im.ns_produce / 1000000) << "ms";
        im.io.stats().append(o);
    }
    im.repacker.append(o);
    im.hits.append(o);
    im.skew.append(o);
    if (im.failures) {
        o << "  [" << im.failures << " MATERIALISE FAILURES -- output is not trustworthy]";
    }
    return o.str();
}

}  // namespace dray::backend
