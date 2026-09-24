// MiniMax-M3. The profile is a plain struct literal: no parser, no lookup, no
// dependency beyond profile.h. See CLAUDE.md, "Adding a model must be additive".
//
// M3 is target #1 because it is the only one of the three on mainline llama.cpp
// and the only one whose state is prefix-sliceable, so it pays for the whole
// engine with no fork and no recurrent-state checkpointing.
//
// ---------------------------------------------------------------------------
// VERIFIED FACTS
//
// Structural facts (arch string, layer split, tensor inventory, cache class) are
// from the vendored llama.cpp pinned at 84e908c62: src/llama-arch.cpp (the
// LLM_ARCH_NAMES table), src/models/minimax-m3.cpp (hparam + tensor loading),
// src/llama-model.cpp (KV cache construction). The counts and sizes are from the
// released model's own GGUF metadata -- llama.cpp only reads them, it does not
// define them, and neither does this file.
//
// These are recorded here because they were expensive to establish and are easy
// to get wrong, NOT because the engine reads them from here. Everything GGUF
// exposes as metadata is read from the GGUF at load; the Profile schema
// deliberately carries none of it (Invariant 3 -- never assume, always read).
//
//   general.architecture   "minimax-m3"        (llama-arch.cpp, LLM_ARCH_NAMES)
//   size class             428B total / 23B active
//                          (LLM_TYPE_428B_A23B, selected on n_layer == 60)
//   layers                 60 -- blk.0-2 dense, blk.3-59 MoE (57 MoE layers).
//                          leading_dense_block_count = 3; llama.cpp splits on
//                          `i < hparams.n_layer_dense_lead` when creating tensors.
//   experts                128 routed, top-4, 1 shared
//   hidden size            6144
//   routed expert ffn      3072  (expert_feed_forward_length)
//   vocab                  200064
//   native context         1,048,576
//
// MIXER -- the fact the whole build order rests on.
// M3 is block-sparse GQA (MiniMax Sparse Attention). That is ORDINARY attention
// with an index choosing which blocks of the past to attend to: the past is still
// stored per token, still addressable by prefix, still sliceable. It is NOT
// linear attention. Verified structurally rather than from the paper -- the M3
// tensor table contains no conv1d, no dt_bias, no A_log and no ssm_* tensor
// anywhere. load_arch_tensors creates exactly: attention QKV/output projections,
// per-head QK-norms, norms, dense FFN on blk.0-2, routed+shared experts on
// blk.3-59, and four indexer tensors (index_{q,k}_proj, index_{q,k}_norm) per
// sparse layer. Nothing else. llama.cpp then gives it a llama_kv_cache_msa --
// a KV cache. Hence StateClass::SliceableKv, and no snapshot path is needed.
//
// The sparse/dense split coincides with the MoE/dense split: the indexer key
// cache is filtered to `il >= n_layer_dense_lead`, so blk.0-2 run full attention
// and blk.3-59 run block-sparse. One boundary, not two.
//
// CONTEXT -- why default_n_ctx is 32768 and not 1,048,576.
// M3's native window is unreachable on any in-scope machine and we do not pretend
// otherwise. KV is 122,880 B/token at F16, so the full 1,048,576-token window is
// 128.8 GB (120 GiB) of KV alone, before one weight is resident -- more than the
// total RAM of the 16-32 GB machines this engine targets. 32768 is the default;
// it still costs 3.75 GiB of KV (122,880 x 32,768 = 4,026,531,840 B) and will
// itself be refused under a 4 GB cap. That refusal is correct and is reported
// with the largest context the cap could actually serve
// (plan::Plan::max_n_ctx_for_cap), never silently truncated. A user asking for 1M
// should be told the number, not handed a quiet 32k.
//
// 122,880 B/token is the plain K/V figure. llama.cpp allocates the MSA indexer
// key cache in ADDITION to it, over blk.3-59 only. The residency planner must add
// that to the KV floor rather than assume it is included -- unverified either
// way, so it must be measured, not guessed.
//
// COST MODEL (CLAUDE.md): misses_per_token = k_routed x n_moe_layers x (1-h_routed)
//   = 4 x 57 = 228 cold expert reads per token at h_routed = 0.
// Against the per-expert slot sizes measured on UD-Q2_K_XL (18,137,088 /
// 20,938,752 / 24,477,696 B, recorded in mem/accountant.h) that is roughly 4.2 GB
// read per token with a cold cache. This is where "seconds per token" comes from,
// and it is a property of the model, not of the engine.
//
// Do not back out a bits-per-weight from those slot sizes. A UD quant mixes types
// across a single layer's gate/up/down, and the per-expert size varies by layer;
// every byte figure comes from the GGUF tensor table (Invariant 3).

#include "models/profile.h"

namespace dray::models {

// External linkage is given explicitly: a namespace-scope `const` object would
// otherwise be internal, and registry.cpp links against this symbol. The
// registry declares it rather than profile.h, because profile.h exposes the
// lookup API only -- models are linked into the registry, never #included.
//
// All initialisers are constant expressions, so this is constant-initialised and
// there is no static-initialisation-order hazard with the registry's table.
extern const Profile kMiniMaxM3;

// The released UD conversions predate the loader's five REQUIRED indexer
// hyperparameter keys (first-light 2026-08-20: "key not found in model:
// minimax-m3.attention.indexer.head_count"). Values verified against
// MiniMaxAI/MiniMax-M3 config.json, field-by-field: sparse_num_index_heads=4,
// sparse_index_dim=128, sparse_topk_blocks=16, sparse_block_size=128,
// sparse_local_block=1. The fork's converter has no minimax-m3 support, so
// re-converting was not an option; profile-expressed defaults are.
static const char* const kM3KvDefaults[] = {
    "minimax-m3.attention.indexer.head_count=int:4",
    "minimax-m3.attention.indexer.key_length=int:128",
    "minimax-m3.attention.indexer.top_k=int:16",
    "minimax-m3.attention.indexer.block_size=int:128",
    "minimax-m3.attention.indexer.local_blocks=int:1",
    nullptr,
};

extern const Profile kMiniMaxM3 = {
    .arch          = "minimax-m3",
    .display_name  = "MiniMax-M3",

    // Block-sparse GQA is ordinary sliceable KV -- see the MIXER note above.
    .state_class   = StateClass::SliceableKv,

    // 1,048,576 native, but 122,880 B/token of KV puts the full window at
    // 128.8 GB. 32768 costs 3.75 GiB and is the largest honest default.
    .default_n_ctx = 32768,

    // Verified: llama.cpp creates ffn_{gate,up,down}_exps with ne[2] = n_expert
    // for every MoE layer, i.e. one tensor per matrix with all 128 experts
    // uniformly strided. That stride is exactly what the slab must reproduce for
    // ggml_mul_mat_id, so this is a geometry fact, not a formatting detail.
    .fused_experts = true,

    .kv_defaults   = kM3KvDefaults,
};

}  // namespace dray::models
