// Model profiles. Adding a model must be additive (CLAUDE.md): a new model that
// reuses an existing mixer class is a Profile entry and one registry line, nothing
// else. A model name appears ONLY in the registry (Invariant 4) -- never branch on
// architecture identity anywhere else in the codebase.
//
// Almost everything about a model is read from GGUF metadata at load. A Profile
// carries only what GGUF cannot tell us, plus defaults.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dray::models {

enum class MixerKind : uint8_t {
    Attention,        // ordinary KV: prefix-addressable, sliceable, resumable by slicing
    LinearRecurrent,  // KDA / Gated DeltaNet: a running summary of the whole prefix.
                      // Cannot be sliced -- resume requires a snapshot or a re-prefill.
};

enum class StateClass : uint8_t {
    SliceableKv,      // M3
    OpaqueRecurrent,  // Qwen3.8 (Gated DeltaNet), K3 (KDA)
};

struct Profile {
    const char* arch = nullptr;          // must equal GGUF general.architecture
    const char* display_name = nullptr;

    StateClass state_class = StateClass::SliceableKv;

    // Defaults, overridable by CLI; real values come from GGUF where available.
    uint32_t default_n_ctx = 32768;

    // True when the GGUF is known to fuse per-layer experts into
    // ffn_{gate,up,down}_exps (the llama.cpp convention). Drives slab geometry.
    bool fused_experts = true;

    // GGUF metadata keys the released conversions are known to LACK but the
    // loader requires, as "key=type:value" strings in llama --override-kv
    // syntax. Applied at load when general.architecture matches this profile;
    // an explicit CLI --override-kv for the same key wins. Values here must be
    // verified from the model's own published config, never guessed
    // (Invariant 3) -- the profile records where.
    // Null-terminated array; nullptr = none.
    const char* const* kv_defaults = nullptr;
};

// The registry. This is the only file in the project that names models.
const Profile* find_profile(const std::string& arch);
std::vector<const Profile*> all_profiles();

}  // namespace dray::models
