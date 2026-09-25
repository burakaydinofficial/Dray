// Private, remapped ids tensors for compacted nodes.
//
// MUL_MAT_ID indexes all experts by one uniform stride, and GET_ROWS all rows by
// another, so a compact region holding only the selected experts (or rows) is
// addressed by COMPACT indices 0..k-1. The node gets its own ids tensor holding
// those; the graph's shared ids tensor is never written.
//
// NEVER THE SHARED IDS. An earlier design rewrote selected_experts in place,
// which requires EVERY consumer of those ids to be compaction-aware. Three were
// found at three different indexed axes -- MUL_MAT_ID (dim 2), ADD_ID (dim 1), and
// GET_ROWS on the per-expert scale vector inside build_lora_mm_id -- each only
// after assuming the set was complete. The set is not enumerable from here: it
// depends on which architecture is loaded. With private ids the original keeps
// real ids, so scales and biases stay correct by construction.
//
// PER NODE, not shared. A single shared private tensor is only safe if exactly
// one node computes between one materialise and the next -- an assumption about
// ggml_backend_sched batching this engine has already been burned by once.
// unordered_map gives stable element addresses, which node->src relies on.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ggml.h"

namespace dray::backend {

// The ids tensor is a VIEW: selected_experts comes from ggml_argsort_top_k,
// which views the first n_expert_used entries of each row of an
// [n_expert, n_tokens] argsort result. So nb[1] is the FULL row, not
// ne[0]*nb[0], and the elements are not contiguous. Treating it as a flat array
// read the wrong ids and, worse, wrote through the view into neighbouring
// memory -- one token worked (one row is accidentally contiguous), five faulted.
inline int32_t* ids_at(ggml_tensor* ids, int64_t i0, int64_t i1) {
    return reinterpret_cast<int32_t*>(static_cast<char*>(ids->data) +
                                      i1 * ids->nb[1] + i0 * ids->nb[0]);
}

// Derives the compact ordering (`uniq`, first-seen order) and each position's
// compact index (`remapped`, row-major over ne[0] x ne[1]) from the ORIGINAL
// ids. Reads only. False when the ids are unreadable or out of range -- never
// guess.
bool derive_ids_map(ggml_tensor* ids, int64_t n_expert,
                    std::vector<int32_t>* uniq, std::vector<int32_t>* remapped);

class PrivateIds {
public:
    // `trace` is DRAY_TRACE_COMPACT: read each installed tensor back through
    // its own strides and report mismatches (first four installs).
    explicit PrivateIds(bool trace) : trace_(trace) {}

    // The ids the GRAPH meant for `node`, not the private tensor we may have
    // installed on a previous token. llama.cpp reuses graphs across decode steps
    // and we never restore node->src, so on the next token src already points at
    // OUR tensor -- deriving from it would remap an already-remapped set (the
    // router asks for 5,12,33 and we fetch 0,1,2: first token right, every later
    // token wrong).
    ggml_tensor* original(ggml_tensor* node, ggml_tensor* ids);

    // Builds `node`'s private tensor from `remapped` and repoints node->src[src]
    // at it. MUL_MAT_ID takes ids at src[2], GET_ROWS at src[1].
    void install(ggml_tensor* node, ggml_tensor* ids, const std::vector<int32_t>& remapped,
                 int src = 2);
    // F9: puts the graph's original ids back on node->src[src] if our private
    // tensor is there. A node falling back to the WHOLE tensor must index it
    // with real ids: a previous token's compact ids against the whole tensor
    // are rows 0..k-1 -- fluent garbage with zero failures counted.
    void restore(ggml_tensor* node, int src);
    // A failed node's last line of defence: private ids that are all zero, so
    // every lookup lands in the first stride of the poison buffer and cannot read
    // past it. F10: routed through original(), so a node whose src already IS the
    // private tensor never zeroes its own source.
    void install_zeros(ggml_tensor* node, int src);

    // True when `node` carries (or has carried) a private ids tensor; such a
    // node must always be given to the streamer.
    bool owns(const ggml_tensor* node) const { return by_node_.count(node) != 0; }

private:
    struct Priv {
        ggml_tensor          t{};
        std::vector<int32_t> buf;
        ggml_tensor*         orig = nullptr;   // the ids tensor the graph supplied
    };

    std::unordered_map<const ggml_tensor*, Priv> by_node_;
    bool trace_;
    int  traced_ = 0;
};

}  // namespace dray::backend
