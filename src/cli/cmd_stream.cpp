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

#include "cache/expert_cache.h"
#include "cli/reference_read.h"
#include "io/storage.h"
#include "mem/accountant.h"
#include "plan/residency.h"

namespace dray::cli {

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


}  // namespace dray::cli
