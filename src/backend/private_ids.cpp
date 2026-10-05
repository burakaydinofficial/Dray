#include "backend/private_ids.h"

#include <algorithm>
#include <cstdio>

#include "ggml-backend.h"

namespace dray::backend {

bool derive_ids_map(ggml_tensor* ids, int64_t n_expert,
                    std::vector<int32_t>* uniq, std::vector<int32_t>* remapped) {
    if (!ids->data) return false;
    // Ids computed on another device (a GPU split in --gpu prefill) live in its
    // memory: `data` is a device handle, and reading it here was an access
    // violation that killed the server on a second request (2026-09-30).
    if (ids->buffer && !ggml_backend_buffer_is_host(ids->buffer)) return false;
    const int64_t n0 = ids->ne[0], n1 = ids->ne[1];
    if (n0 <= 0 || n1 <= 0) return false;

    uniq->clear();
    remapped->assign(static_cast<size_t>(n0 * n1), 0);

    size_t k = 0;
    for (int64_t i1 = 0; i1 < n1; ++i1) {
        for (int64_t i0 = 0; i0 < n0; ++i0, ++k) {
            const int32_t e = *ids_at(ids, i0, i1);
            if (e < 0 || e >= n_expert) return false;   // out of range: never guess
            auto f = std::find(uniq->begin(), uniq->end(), e);
            if (f == uniq->end()) { uniq->push_back(e); f = uniq->end() - 1; }
            (*remapped)[k] = static_cast<int32_t>(f - uniq->begin());
        }
    }
    return !uniq->empty();
}

ggml_tensor* PrivateIds::original(ggml_tensor* node, ggml_tensor* ids) {
    auto it = by_node_.find(node);
    if (it != by_node_.end() && ids == &it->second.t && it->second.orig) {
        return it->second.orig;
    }
    return ids;
}

void PrivateIds::install(ggml_tensor* node, ggml_tensor* ids,
                         const std::vector<int32_t>& remapped, int src) {
    Priv& p = by_node_[node];
    if (p.orig == nullptr || ids != &p.t) p.orig = ids;

    // A CLEAN tensor, not a doctored copy of a view. Copying *ids brought its op
    // (GGML_OP_VIEW), its src[] pointing at the argsort result, and its flags
    // along with the shape -- then nulling view_src and buffer left an object
    // claiming to be a view of nothing. Proven to be the fault: with an identity
    // mapping, where the remapped ids are numerically identical to the originals,
    // installing that tensor still produced wrong output, and skipping the
    // install produced text matching the reference exactly.
    //
    // SAME STRIDES AS THE ORIGINAL, not a re-laid-out contiguous copy. The MoE
    // ids are a strided view: ne[0] is n_expert_used but nb[1] is a full n_expert
    // row. A contiguous tensor with the same ne[] and different nb[] SHOULD be
    // equivalent -- mul_mat_id reads through nb[] -- and demonstrably is not.
    // GET_ROWS ids are already contiguous and those always worked. So mirror the
    // original byte layout exactly and change only the VALUES; the buffer is sized
    // from the original stride and the gaps are left as they are.
    const size_t words = static_cast<size_t>(ggml_nbytes(ids) / sizeof(int32_t)) + 4;
    p.buf.assign(words, 0);
    p.t = ggml_tensor{};
    p.t.type = ids->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        p.t.ne[i] = ids->ne[i];
        p.t.nb[i] = ids->nb[i];
    }
    p.t.op = GGML_OP_NONE;
    p.t.buffer   = ids->buffer;
    p.t.data     = p.buf.data();
    p.t.view_src = nullptr;
    p.t.view_offs = 0;
    // Write the remapped values at the ORIGINAL strided positions.
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const size_t k = static_cast<size_t>(i1 * ids->ne[0] + i0);
            if (k >= remapped.size()) continue;
            *reinterpret_cast<int32_t*>(static_cast<char*>(p.t.data) +
                                        i1 * p.t.nb[1] + i0 * p.t.nb[0]) = remapped[k];
        }
    }

    // Read the installed tensor back through its OWN strides and compare against
    // what we meant to write. If these disagree the tensor is not what we think
    // we built, whatever the field values look like.
    if (trace_ && traced_ < 4) {
        ++traced_;
        int bad = 0;
        for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                const int32_t got = *reinterpret_cast<int32_t*>(
                    static_cast<char*>(p.t.data) + i1 * p.t.nb[1] + i0 * p.t.nb[0]);
                const size_t k = static_cast<size_t>(i1 * ids->ne[0] + i0);
                if (k >= remapped.size() || got != remapped[k]) ++bad;
            }
        }
        std::fprintf(stderr, "[dray] PRIVIDS ne=[%lld,%lld] nb=[%zu,%zu] buf=%zu bad=%d src=%d\n",
                     (long long)p.t.ne[0], (long long)p.t.ne[1], p.t.nb[0], p.t.nb[1],
                     p.buf.size(), bad, src);
        std::fflush(stderr);
    }

    node->src[src] = &p.t;
}

void PrivateIds::restore(ggml_tensor* node, int src) {
    auto pv = by_node_.find(node);
    if (pv != by_node_.end() && node->src[src] == &pv->second.t && pv->second.orig != nullptr) {
        node->src[src] = pv->second.orig;
    }
}

void PrivateIds::install_zeros(ggml_tensor* node, int src) {
    if (!node->src[src]) return;
    ggml_tensor* oids = original(node, node->src[src]);
    const std::vector<int32_t> zeros(static_cast<size_t>(ggml_nelements(oids)), 0);
    install(node, oids, zeros, src);
}

}  // namespace dray::backend
