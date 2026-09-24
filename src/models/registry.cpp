// The registry. This is the ONLY file in the project that lists models
// (Invariant 4). Nothing else may branch on architecture identity: if a component
// needs to behave differently per model, that difference belongs in the Profile
// schema or behind a hook interface, never behind the name.
//
// The table is a flat array scanned linearly. With three entries a map would cost
// more in allocation and static-init order than it saves in comparisons, and the
// scan order is also the reporting order.

#include "models/profile.h"

#include <iterator>
#include <string>
#include <vector>

namespace dray::models {

// Each model is defined as a struct literal in its own translation unit and
// linked, not #included -- profile.h carries the lookup API only, so adding a
// model never edits a header that the rest of the engine depends on.
extern const Profile kMiniMaxM3;

namespace {

// ---------------------------------------------------------------------------
// THE TABLE
// ---------------------------------------------------------------------------
const Profile* const kRegistry[] = {
    &kMiniMaxM3,
    // S34 truth note: Kimi K3, GLM-5.2 and Qwen3.8 all generate correct text
    // TODAY with no entry here -- the planner is architecture-blind and their
    // llama.cpp graph support suffices. An entry becomes necessary only when a
    // model needs profile-expressed behaviour (per-layer schedule overrides,
    // policy defaults, a new state class). The commented lines below used to
    // read as "unsupported until this file grows", which inverted reality.
    // &kQwen38,   // only if/when Qwen3.8 needs profile-expressed behaviour
    // &kKimiK3,   // only if/when K3 needs profile-expressed behaviour
};

// ---------------------------------------------------------------------------
// WHAT A FUTURE MODEL COSTS
//
// The two blocks below are commented out on purpose: together with the two table
// lines above they are the COMPLETE diff for adding a model, so the price is
// visible rather than asserted. One struct literal in its own file, one extern
// declaration, one table line. No other file in the engine learns the name.
//
// The honest part of the price: both of the next two targets are
// StateClass::OpaqueRecurrent -- Qwen3.8 uses Gated DeltaNet, Kimi K3 uses KDA --
// T28 truth note: the paragraph below predates events. Qwen3.8 and K3 both
// GENERATE today with no state class -- llama.cpp holds their recurrent state
// and the arch-blind planner needs nothing from this file. What still needs
// src/state/recurrent_opaque.cpp is recurrent CHECKPOINT SLICING (an opaque
// running summary cannot be sliced, so resume needs a snapshot or re-prefill),
// which the SnapshotCache currently covers by whole-state snapshots. A model reusing
// an existing mixer class is free; being the FIRST model of a class is not. Once
// Qwen3.8 has paid for recurrent_opaque.cpp, K3 costs exactly what is written
// below and nothing more -- which is the whole point of the build order.
//
// --- src/models/qwen38.cpp -------------------------------------------------
//
//   // The arch string is verified present in the pinned llama.cpp (84e908c62):
//   // LLM_ARCH_QWEN35MOE, "qwen35moe", in src/llama-arch.cpp's LLM_ARCH_NAMES.
//   // The profile is cheap; the model is not. CLAUDE.md calls Qwen3.8 the
//   // expensive step because its q1 dtypes need a non-mainline branch and its
//   // Gated DeltaNet needs the checkpoint path. Neither of those costs is
//   // expressible in -- or caused by -- this literal, and neither is hidden by it.
//   extern const Profile kQwen38;
//   const Profile kQwen38 = {
//       .arch          = "qwen35moe",
//       .display_name  = "Qwen3.8 MoE",
//       .state_class   = StateClass::OpaqueRecurrent,  // Gated DeltaNet
//       .default_n_ctx = 32768,
//       .fused_experts = true,   // VERIFY against the tensor table before trusting
//   };
//
// --- src/models/kimi_k3.cpp ------------------------------------------------
//
//   // K3's general.architecture is NOT in the pinned tree: the only Kimi entry
//   // there is LLM_ARCH_KIMI_LINEAR, "kimi-linear". Take the string from the
//   // actual GGUF's general.architecture. Do not guess it, and do not assume the
//   // converter reuses "kimi-linear" -- lookup is exact, so a wrong string here
//   // fails loudly at load, which is the intended failure mode.
//   //
//   // K3 has 2 SHARED EXPERTS, not 1 like M3. That is not a trivia difference:
//   // shared experts are read every token, so they are UnconditionalBulk (tier 1
//   // of the fill order), not routed population. The second one is not free, and
//   // it does not appear in misses_per_token -- it appears in the unconditional
//   // stream that h_bytes is measured against.
//   extern const Profile kKimiK3;
//   const Profile kKimiK3 = {
//       .arch          = "kimi-k3",          // PROVISIONAL -- verify against the GGUF
//       .display_name  = "Kimi K3",
//       .state_class   = StateClass::OpaqueRecurrent,  // KDA
//       .default_n_ctx = 32768,
//       .fused_experts = true,   // VERIFY against the tensor table before trusting
//   };
// ---------------------------------------------------------------------------

// Architectures we know by name but have not implemented. Purely so the error
// message can distinguish "never heard of it" from "recognised, not built yet" --
// a user holding a Qwen3.8 GGUF deserves the second answer, not the first. Naming
// them here rather than at a call site is what keeps Invariant 4 intact.
struct PlannedArch {
    const char* arch;
    const char* why_not_yet;
};

const PlannedArch kPlanned[] = {
    { "qwen35moe",
      "Qwen3.8 MoE - planned. Needs the opaque-recurrent state class (Gated "
      "DeltaNet) and a non-mainline llama.cpp branch for its q1 dtypes." },
    // Provisional string (see the K3 block above); if the real GGUF disagrees the
    // user falls back to the generic message, which is no worse than silence.
    { "kimi-k3",
      "Kimi K3 - planned. Needs the opaque-recurrent state class (KDA), which "
      "Qwen3.8 lands first." },
};

// ASCII-only, locale-free, and takes the char through unsigned char: std::tolower
// on a plain (possibly signed) char is undefined for bytes above 0x7F, and GGUF
// metadata is arbitrary bytes. Used ONLY to explain a near miss, never to match.
char ascii_lower(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return (u >= 'A' && u <= 'Z') ? static_cast<char>(u - 'A' + 'a') : c;
}

bool equal_ignoring_ascii_case(const std::string& a, const char* b) {
    if (b == nullptr) {
        return false;
    }
    std::string::size_type i = 0;
    for (; i < a.size(); ++i) {
        if (b[i] == '\0' || ascii_lower(a[i]) != ascii_lower(b[i])) {
            return false;
        }
    }
    return b[i] == '\0';
}

}  // namespace

const Profile* find_profile(const std::string& arch) {
    // Case-sensitive exact match, deliberately. general.architecture is not user
    // text: it is emitted by the converter from a fixed table (llama.cpp's
    // LLM_ARCH_NAMES), so any deviation means the file is not what we think it
    // is. Loose matching would let a near-miss arch load with the wrong profile,
    // and the failure would surface as plausible text from wrong weights -- the
    // worst failure mode this engine has.
    for (const Profile* p : kRegistry) {
        if (p != nullptr && p->arch != nullptr && arch == p->arch) {
            return p;
        }
    }
    return nullptr;
}

std::vector<const Profile*> all_profiles() {
    return std::vector<const Profile*>(std::begin(kRegistry), std::end(kRegistry));
}

// Message for a failed lookup. profile.h has no error channel -- find_profile
// returns a pointer and nothing else -- so this lives here, where the table is;
// building it at a call site would mean listing models outside the registry.
// Callers reach it with:
//
//     std::string unknown_arch_error(const std::string& arch);
//
// which is the one line profile.h should carry when it is next revised.
std::string unknown_arch_error(const std::string& requested) {
    std::string msg = "unknown model architecture \"" + requested + "\".\n\n";

    msg += "This build implements:\n";
    for (const Profile* p : kRegistry) {
        if (p == nullptr || p->arch == nullptr) {
            continue;
        }
        msg += "  ";
        msg += p->arch;
        msg += "  (";
        msg += (p->display_name != nullptr) ? p->display_name : "unnamed";
        msg += ", ";
        msg += (p->state_class == StateClass::SliceableKv) ? "sliceable KV"
                                                           : "opaque recurrent state";
        msg += ", default n_ctx " + std::to_string(p->default_n_ctx) + ")\n";
    }

    // Name the near miss explicitly. The alternative -- printing the list and
    // letting the user diff two near-identical strings by eye -- is how a case
    // difference costs somebody an afternoon.
    for (const Profile* p : kRegistry) {
        if (p == nullptr || p->arch == nullptr || requested == p->arch) {
            continue;  // an exact match never reaches this function
        }
        if (equal_ignoring_ascii_case(requested, p->arch)) {
            msg += "\n\"";
            msg += p->arch;
            msg += "\" differs from your string only in letter case. "
                   "general.architecture is matched exactly.\n";
        }
    }
    for (const PlannedArch& planned : kPlanned) {
        if (requested == planned.arch) {
            msg += "\nRecognised, but not implemented in this build: ";
            msg += planned.why_not_yet;
            msg += "\n";
        }
    }

    msg += "\nThe architecture is the GGUF general.architecture string, matched "
           "exactly (case-sensitive).\n";
    msg += "Adding a model is additive: one Profile literal in "
           "src/models/<name>.cpp plus one line in src/models/registry.cpp -- see "
           "the worked examples in that file.\n";
    return msg;
}

}  // namespace dray::models
