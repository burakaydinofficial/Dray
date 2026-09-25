// The streaming weight buffer: the mechanism the whole project exists for.
//
// PREMISE. Models do not fit and never will. K3 is 594 GB, the cap is 4-16 GB.
// "Does not fit in RAM" is not an error condition here, it is the input. Any code
// path that refuses a model for being too large is wrong by construction -- engines
// that run what already fits are a solved problem.
//
// CONSEQUENCE THAT AN EARLIER DESIGN GOT WRONG. Streaming only the routed experts
// is not enough. K3's unconditional weights -- attention, shared experts, embedding,
// lm_head -- are 48.68 GB measured, and Qwen3.8's are 36.87 GB. At a 4 GB cap those
// do not fit either. EVERY weight streams. Only the mandatory floor is resident:
// router gates (2.36 GB on K3, and they must be resident or every layer gains a
// serialized round trip), norms, KV and recurrent state.
//
// WHERE TO INTERVENE. Not in build_moe_ffn -- that only reaches the experts. The
// right layer is ggml's buffer type, because it reaches every weight uniformly and
// needs no llama.cpp changes: llama_model_params::tensor_buft_overrides is a public
// NULL-terminated {pattern, buft} list, so weights can be routed here by name.
//
// HOW IT WORKS.
//   1. llama.cpp allocates model weights from this buffer type.
//   2. init_tensor does NOT reserve storage. It looks the tensor up by name in the
//      residency Plan -- which already carries an exact (shard, offset, bytes) for
//      every tensor, read from the GGUF tensor table -- and records that source.
//      tensor->data is left unbacked until the tensor is actually needed.
//   3. set_tensor is a no-op. llama.cpp is handing us bytes it read from the file;
//      we already know where those bytes live on disk and will read them ourselves,
//      uncached, when they are needed. Accepting the write would mean materialising
//      the whole model, which is the thing we are avoiding.
//   4. During graph execution, before a node runs, every weight src it touches is
//      materialised into a slab slot and tensor->data is pointed at it. Resident
//      tensors (the floor) are pinned and skip this entirely.
//
// MUL_MAT_ID. A layer's experts are one fused tensor indexed by uniform stride, so
// a partially-resident tensor cannot be expressed by repointing data. Only the
// selected experts are read, into one compact region at the tensor's own stride,
// and the node gets private remapped ids (expert_compactor.h, private_ids.h).
//
// WHERE THINGS LIVE. This header is the whole public surface. Behind it,
// streamer_impl.h composes the parts: accounted_alloc, poison_buffers,
// io_scheduler, tensor_registry, residency_cache, expert_compactor,
// expert_slots + routing_skew, uncond_ring, hit_rates, repacker, private_ids;
// stream_budget and stream_flags are the setup they are built from, and
// stream_buffer_type.cpp is the ggml callback glue.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend/merged_metadata.h"
#include "backend/tensor_source.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "io/storage.h"
#include "mem/accountant.h"
#include "plan/residency.h"

namespace dray::backend {

struct Config {
    uint64_t cap = 0;             // total resident bytes (Invariant 1)
    uint32_t queue_depth = 64;
    // True when the model is loaded through llama.cpp's file reader rather than
    // from merged metadata. Only that path needs a landing buffer as large as the
    // biggest tensor; reserving one otherwise costs 1.6 GiB of a small cap.
    bool     slow_load = false;
    // Mirrors DRAY_NO_COMPACT. Whole-tensor materialisation needs a far larger
    // eviction reserve than compacted experts do.
    bool     no_compact = false;
    // Batch width the admission funded; sizes the churn reserve for the
    // UNION working set. 1 = classic single-stream (bit-identical behaviour).
    uint32_t n_seq = 1;
    // Slab sizing is derived from the plan, not guessed: enough slots to hold the
    // widest single node's working set, plus whatever the cap allows for reuse.
    uint64_t min_slots_per_layer = 0;
};

// Owns the slab, the shard handles and the tensor->Source map. One per model.
class Streamer {
public:
    Streamer(mem::Accountant&, const plan::Plan&, Config);
    ~Streamer();

    bool valid() const { return error_.empty(); }
    const std::string& error() const { return error_; }

    // The buffer type to hand llama.cpp via tensor_buft_overrides. Never null.
    ggml_backend_buffer_type_t buft();

    // Materialises every weight src of `node` that this Streamer owns, pointing
    // tensor->data at slab memory. Returns false if any read failed -- callers MUST
    // NOT compute on a partially materialised node, because a short read produces
    // plausible text from the wrong weights rather than an error.
    bool materialise(struct ggml_tensor* node);

    // Releases the refcounts taken by materialise() for this node.
    void release(struct ggml_tensor* node);

    uint64_t bytes_streamed() const { return bytes_streamed_; }
    uint64_t nodes_materialised() const { return nodes_; }
    // Routed-expert bytes actually read from disk (the readout's Gather class;
    // Stream = bytes_streamed() - this). Same counter h_routed already uses.
    uint64_t routed_bytes_read() const;
    // T16: the storage backend's self-description (mechanism, alignment,
    // queue depth, and its DEGRADED marker when the kernel refused uncached
    // mode) -- reachable on every run path, not only from cmd_calibrate.
    std::string io_describe() const;
    // R7 residual: the raw inputs of the two hit rates (CLAUDE.md: never one
    // symbol). needed = bytes demanded; read = the miss side, from disk.
    uint64_t routed_bytes_needed() const;
    uint64_t uncond_bytes_needed() const;
    uint64_t uncond_bytes_read() const;
    // Non-zero means at least one node computed against poison rather than real
    // weights. Any output from such a run is untrustworthy and must be labelled so.
    uint64_t failures() const;
    uint64_t compacted() const;

    // True once any weight failed to materialise. The run MUST stop: there is no
    // safe way to continue.
    //
    // Pointing the tensor at a poison buffer and carrying on is not safe. The
    // buffer is a 1 MiB sentinel while a real expert tensor is ~2 GB, and the ids
    // still hold real expert indices, so mul_mat_id reads hundreds of strides past
    // the end. Sizing the poison to the largest tensor would only convert the crash
    // into fluent garbage, which is worse.
    bool aborted() const;

    // Install as llama_set_abort_callback so ggml stops the graph cleanly at the
    // next node instead of computing against weights that are not there.
    static bool abort_cb(void* streamer);
    std::string report() const;

    // Post-load integrity check on the mandatory floor: re-reads a sample of the
    // resident tensors through an ordinary buffered path and compares them against
    // what is in memory.
    //
    // This exists because "zero reported failures, garbage output" has happened
    // twice. Both times a tensor was present but wrong -- once uninitialised
    // because only one of two load paths filled it -- and nothing in the engine
    // could tell. A counter that only counts what it knows went wrong cannot
    // detect what it never knew about; this compares against the file instead.
    struct SelfCheck {
        size_t checked = 0;
        size_t mismatched = 0;
        // Repacked tensors cannot be compared to the file (their bytes were
        // rewritten for the optimised kernels). Counted, never hidden.
        size_t skipped_repacked = 0;
        bool ok() const { return checked > 0 && mismatched == 0; }
    };
    SelfCheck self_check(size_t max_tensors = 32);

    // Re-derives the cache budget from what the process ACTUALLY has resident, so
    // the cap means total resident bytes rather than the streamer's share of it.
    //
    // The floor is an estimate of what llama.cpp will allocate; llama.cpp then
    // allocates whatever it actually needs -- its own KV cache, compute buffers,
    // the vocab -- none of which pass through our accountant. Measured on Qwen3.8
    // at a 12 GiB cap: 13.57 GB private bytes, 0.69 GB over. Called once the
    // context exists, which is the first moment the real figure is knowable.
    //
    // Returns the new budget. Shrinking it is the only lever available: the
    // external allocations already exist and are not ours to release.
    uint64_t rebudget_against_rss(uint64_t rss_bytes);

    // True when the ledger says the process is over the cap. Reported, never
    // silently tolerated: Invariant 1 is the project's central claim, and a run
    // that exceeded it while printing nothing would be the exact dishonesty the
    // readout rules exist to prevent.
    bool over_cap() const;

    // Per-category ledger dump, including bytes charged as unreserved. This is what
    // makes a small cap actionable instead of mysterious.
    std::string accountant_report() const;

    // Bytes of the largest single streamed tensor. A cache smaller than this cannot
    // run the model at all: that tensor has to be materialised whole (it is indexed
    // linearly by shape, so unlike experts or embedding rows there is nothing to
    // slice), and failing to do so mid-graph means computing against a poison buffer
    // that is orders of magnitude too small.
    uint64_t largest_streamed_bytes() const;
    // The churn reserve: room one batched node's union working set requires.
    // Admission must refuse when the cache budget cannot fund it (measured
    // 2026-08-19: B=38 passed the output.weight check with a 1.81 GB budget,
    // then died mid-run needing a 2.54 GB region -- a check the data already
    // knew at load).
    uint64_t churn_reserve_bytes() const;
    uint64_t batch_region_bytes() const;
    // True when this node has a disk-backed source and the engine must act on
    // it. Answering false lets ggml batch consecutive nodes into ONE compute
    // call instead of one per node with a barrier between (see the definition).
    bool needs(ggml_tensor* node);
    std::string largest_streamed_name() const;

    // Public because the ggml buffer-type callbacks are free functions with C
    // linkage semantics and must reach it; there is nothing to encapsulate from.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    std::string error_;
    uint64_t bytes_streamed_ = 0;
    uint64_t nodes_ = 0;
};

}  // namespace dray::backend
