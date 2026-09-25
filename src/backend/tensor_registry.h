// Where every tensor lives on disk, and which ggml tensors we have bound to it.
//
// Two views of one fact. `sources` is keyed by NAME and filled from the Plan
// (the GGUF tensor table) before llama.cpp creates anything; `bound` is keyed by
// the ggml tensor POINTER and filled as init_tensor meets each weight. The
// streamer reasons in pointers on the hot path, and in names only at load and
// for sibling lookup (blk.N.ffn_gate_exps -> blk.N.ffn_up_exps).

#pragma once

#include <set>
#include <string>
#include <unordered_map>

#include "backend/tensor_source.h"
#include "ggml.h"

namespace dray::backend {

class TensorRegistry {
public:
    using ByName = std::unordered_map<std::string, Source>;

    void add_source(const std::string& name, const Source& s) { sources_[name] = s; }
    const ByName& sources() const { return sources_; }
    const Source* source_named(const char* name) const {
        auto it = sources_.find(name);
        return it == sources_.end() ? nullptr : &it->second;
    }

    // init_tensor: this ggml tensor IS that disk range.
    void bind(ggml_tensor* t, const Source& s) {
        bound_[t] = s;
        by_name_[t->name] = t;   // sibling lookup for fused per-layer expert reads
    }
    // Null when the tensor is not ours (llama.cpp's own tensors, activations).
    const Source* source_of(const ggml_tensor* t) const {
        auto it = bound_.find(t);
        return it == bound_.end() ? nullptr : &it->second;
    }
    bool is_bound(const ggml_tensor* t) const { return bound_.count(t) != 0; }
    ggml_tensor* named(const std::string& name) const {
        auto it = by_name_.find(name);
        return it == by_name_.end() ? nullptr : it->second;
    }

    // A tensor that reached init_tensor with no disk source. Returns true the
    // first time a name is seen, so it is reported once. A real weight here is
    // uninitialised memory -- a classifier gap, not a curiosity.
    bool note_unsourced(const char* name) { return unsourced_.insert(name).second; }

private:
    ByName                                           sources_;
    std::unordered_map<const ggml_tensor*, Source>   bound_;
    std::unordered_map<std::string, ggml_tensor*>    by_name_;
    std::set<std::string>                            unsourced_;
};

}  // namespace dray::backend
