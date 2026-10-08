// Settings layers: built-in < shipped system < user system < shipped model <
// user model < per-GGUF-file section < command line, per key; strict parsing;
// hardware settings are system-only.

#include "harness.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "config/settings.h"
#include "engine/thread_policy.h"

namespace fs = std::filesystem;
using dray::config::apply_json;
using dray::config::builtin_settings;
using dray::config::CliSettings;
using dray::config::Locations;
using dray::config::resolve;
using dray::config::Settings;
using dray::config::Source;

namespace {

// A scratch tree with shipped/ and user/ directories, removed on scope exit.
struct Tree {
    fs::path root;
    Tree() {
        root = fs::temp_directory_path() / ("dray-settings-" + std::to_string(std::rand()));
        fs::create_directories(root / "shipped" / "models");
        fs::create_directories(root / "user" / "models");
    }
    ~Tree() { std::error_code ec; fs::remove_all(root, ec); }
    void write(const fs::path& rel, const std::string& text) const {
        std::ofstream f(root / rel, std::ios::binary);
        f << text;
    }
    Locations where() const { return {root / "shipped", root / "user"}; }
};

bool system_json(const std::string& text, Settings* s, std::string* err) {
    return apply_json(text, "test.json", Source::UserSystem, false, "", s, err);
}
bool model_json(const std::string& text, const std::string& gguf, Settings* s, std::string* err) {
    return apply_json(text, "model.json", Source::UserModel, true, gguf, s, err);
}

}  // namespace

LZ_TEST(no_files_means_built_in_values) {
    Tree t;
    Settings s;
    std::string err;
    LZ_REQUIRE(resolve(t.where(), "some-arch", "m.gguf", {}, &s, &err));
    const Settings b = builtin_settings();
    LZ_CHECK_EQ(s.decode_threads.value, b.decode_threads.value);
    LZ_CHECK_EQ(s.queue_depth.value, b.queue_depth.value);
    LZ_CHECK(s.decode_threads.source == Source::Builtin);
    LZ_CHECK(s.applied.empty());
}

LZ_TEST(layers_override_per_key_in_order) {
    Tree t;
    // Shipped system sets two keys; the user overrides one of them. The shipped
    // model overrides a policy key, the user model another, a file section a
    // third, and the command line wins over all.
    t.write("shipped/system.json",
            "{ \"schema\": 1, \"threads\": {\"decode\": 6}, \"io\": {\"queue_depth\": 32},"
            "  \"prefill\": {\"chunk_cpu\": 256} }");
    t.write("user/system.json", "// mine\n{ \"schema\": 1, \"io\": {\"queue_depth\": 16} }");
    t.write("shipped/models/arch-x.json",
            "{ \"schema\": 1, \"arch\": \"arch-x\", \"kv_defaults\": [\"a=int:1\"],"
            "  \"overrides\": {\"cache\": {\"admit_byte_fraction\": 0.2}} }");
    t.write("user/models/arch-x.json",
            "{ \"schema\": 1, \"overrides\": {\"prefill\": {\"chunk_gpu\": 1024}},"
            "  \"files\": { \"q4.gguf\": {\"overrides\": {\"prefill\": {\"chunk_cpu\": 128}}} } }");
    CliSettings cli;
    cli.threads = 3;
    Settings s;
    std::string err;
    LZ_REQUIRE(resolve(t.where(), "arch-x", "q4.gguf", cli, &s, &err));
    LZ_CHECK_EQ(s.queue_depth.value, 16);                  // user system beat shipped
    LZ_CHECK(s.queue_depth.source == Source::UserSystem);
    LZ_CHECK_EQ(s.decode_threads.value, 3);                // command line beat shipped
    LZ_CHECK(s.decode_threads.source == Source::Cli);
    LZ_CHECK_EQ(s.prefill_threads.value, 3);               // --threads sets both
    LZ_CHECK_EQ(s.prefill_chunk_cpu.value, 128);           // file section beat everything below it
    LZ_CHECK_EQ(s.prefill_chunk_gpu.value, 1024);          // user model
    LZ_CHECK(s.admit_byte_fraction.value == 0.2);          // shipped model
    LZ_CHECK(s.admit_byte_fraction.source == Source::ShippedModel);
    LZ_CHECK_EQ(s.kv_defaults.value.size(), 1u);
    LZ_CHECK_EQ(s.applied.size(), 4u);                     // every file, in order

    // A different GGUF file of the same model does not get that file's section.
    Settings other;
    LZ_REQUIRE(resolve(t.where(), "arch-x", "q8.gguf", cli, &other, &err));
    LZ_CHECK_EQ(other.prefill_chunk_cpu.value, 256);
}

LZ_TEST(a_model_file_cannot_set_hardware) {
    Settings s = builtin_settings();
    std::string err;
    LZ_CHECK(!model_json("{ \"schema\": 1, \"overrides\": {\"threads\": {\"decode\": 8}} }", "", &s, &err));
    LZ_CHECK(err.find("not overridable by a model") != std::string::npos);
    LZ_CHECK(!model_json("{ \"schema\": 1, \"threads\": {\"decode\": 8} }", "", &s, &err));
    LZ_CHECK_EQ(s.decode_threads.value, builtin_settings().decode_threads.value);
}

LZ_TEST(gpu_read_ahead_is_off_unless_the_machine_or_a_model_turns_it_on) {
    Settings s = builtin_settings();
    std::string err;
    LZ_CHECK(!s.prefill_gpu_read_ahead.value);                       // costs bytes: off
    // A model whose prefill gains from it opts in in its own file ...
    LZ_REQUIRE(model_json("{ \"schema\": 1, \"overrides\": {\"prefill\": {\"gpu_read_ahead\": true}} }",
                          "", &s, &err));
    LZ_CHECK(s.prefill_gpu_read_ahead.value);
    LZ_CHECK(s.prefill_gpu_read_ahead.source == Source::UserModel);
    // ... and the machine may set it for everything.
    Settings m = builtin_settings();
    LZ_REQUIRE(system_json("{ \"schema\": 1, \"prefill\": {\"gpu_read_ahead\": true} }", &m, &err));
    LZ_CHECK(m.prefill_gpu_read_ahead.value);
    LZ_CHECK(!system_json("{ \"schema\": 1, \"prefill\": {\"gpu_read_ahead\": 1} }", &m, &err));
    LZ_CHECK(err.find("true or false") != std::string::npos);
}

LZ_TEST(typos_and_wrong_types_are_errors_never_ignored) {
    Settings s = builtin_settings();
    std::string err;
    LZ_CHECK(!system_json("{ \"schema\": 1, \"threads\": {\"decoed\": 8} }", &s, &err));
    LZ_CHECK(err.find("threads.decoed") != std::string::npos);   // names the bad key
    LZ_CHECK(!system_json("{ \"schema\": 1, \"io\": {\"queue_depth\": \"64\"} }", &s, &err));
    LZ_CHECK(err.find("integer") != std::string::npos);
    LZ_CHECK(!system_json("{ \"schema\": 1, \"cache\": {\"admit_byte_fraction\": 1.5} }", &s, &err));
    LZ_CHECK(!system_json("{ \"threads\": {\"decode\": 8} }", &s, &err));   // no schema
    LZ_CHECK(!system_json("{ \"schema\": 2 }", &s, &err));                  // a newer schema
    LZ_CHECK(!system_json("{ \"schema\": 1, ", &s, &err));                  // not JSON
    LZ_CHECK_EQ(s.decode_threads.value, builtin_settings().decode_threads.value);
}

LZ_TEST(an_invalid_file_stops_resolution_with_its_path) {
    Tree t;
    t.write("user/system.json", "{ \"schema\": 1, \"io\": {\"depth\": 8} }");
    Settings s;
    std::string err;
    LZ_CHECK(!resolve(t.where(), "a", "m.gguf", {}, &s, &err));
    LZ_CHECK(err.find("system.json") != std::string::npos);
    LZ_CHECK(err.find("io.depth") != std::string::npos);
}

LZ_TEST(an_architecture_name_never_escapes_the_models_directory) {
    Tree t;
    t.write("shipped/evil.json", "{ \"schema\": 1, \"kv_defaults\": [\"x=int:1\"] }");
    Settings s;
    std::string err;
    LZ_REQUIRE(resolve(t.where(), "../evil", "m.gguf", {}, &s, &err));
    LZ_CHECK(s.kv_defaults.value.empty());
}

LZ_TEST(threads_follow_the_settings_and_clamp_to_the_ceiling) {
    using dray::engine::choose_threads;
    // Defaults on a 22-thread machine with 9 reserved: 4 decode, 13 prefill.
    auto c = choose_threads(4, 0, 9, 22);
    LZ_CHECK_EQ(c.decode, 4);
    LZ_CHECK_EQ(c.prefill, 13);
    LZ_CHECK(!c.clamped);
    // --threads 22: both clamped to the ceiling, reported.
    c = choose_threads(22, 22, 9, 22);
    LZ_CHECK_EQ(c.decode, 13);
    LZ_CHECK_EQ(c.prefill, 13);
    LZ_CHECK(c.clamped);
    // A machine that reserves nothing.
    c = choose_threads(4, 0, 0, 8);
    LZ_CHECK_EQ(c.prefill, 8);
}

LZ_TEST(describe_names_every_source) {
    Tree t;
    t.write("user/system.json", "{ \"schema\": 1, \"io\": {\"queue_depth\": 16} }");
    Settings s;
    std::string err;
    LZ_REQUIRE(resolve(t.where(), "a", "m.gguf", {}, &s, &err));
    std::ostringstream o;
    dray::config::describe(s, o);
    LZ_CHECK(o.str().find("io.queue_depth") != std::string::npos);
    LZ_CHECK(o.str().find("user system") != std::string::npos);
    LZ_CHECK(o.str().find("built-in") != std::string::npos);
}

LZ_TEST(the_vram_cap_is_a_hard_size_set_only_by_the_machine) {
    Settings s = builtin_settings();
    std::string err;
    LZ_CHECK_EQ(s.gpu_vram_cap.value, 0u);                        // automatic by default
    LZ_CHECK(s.gpu_vram_auto_fraction.value == 0.40);             // the owner's rule since 2026-10-07
    LZ_REQUIRE(system_json("{ \"schema\": 1, \"gpu\": {\"vram_cap\": \"2G\"} }", &s, &err));
    LZ_CHECK_EQ(s.gpu_vram_cap.value, 2ull << 30);
    LZ_REQUIRE(system_json("{ \"schema\": 1, \"gpu\": {\"vram_cap\": \"1536M\"} }", &s, &err));
    LZ_CHECK_EQ(s.gpu_vram_cap.value, 1536ull << 20);
    LZ_REQUIRE(system_json("{ \"schema\": 1, \"gpu\": {\"vram_cap\": 1073741824} }", &s, &err));
    LZ_CHECK_EQ(s.gpu_vram_cap.value, 1ull << 30);                // a plain byte count
    LZ_CHECK(!system_json("{ \"schema\": 1, \"gpu\": {\"vram_cap\": \"lots\"} }", &s, &err));
    LZ_CHECK(err.find("gpu.vram_cap") != std::string::npos);
    LZ_CHECK(!system_json("{ \"schema\": 1, \"gpu\": {\"vram_capp\": \"2G\"} }", &s, &err));
    // A property of the machine: no model file may set it.
    LZ_CHECK(!model_json("{ \"schema\": 1, \"overrides\": {\"gpu\": {\"vram_cap\": \"8G\"}} }", "", &s, &err));
    LZ_CHECK_EQ(s.gpu_vram_cap.value, 1ull << 30);
}

LZ_TEST(vram_cap_on_the_command_line_wins) {
    Tree t;
    t.write("user/system.json", "{ \"schema\": 1, \"gpu\": {\"vram_cap\": \"2G\"} }");
    CliSettings cli;
    cli.vram_cap = 3ull << 30;
    Settings s;
    std::string err;
    LZ_REQUIRE(resolve(t.where(), "llama", "m.gguf", cli, &s, &err));
    LZ_CHECK_EQ(s.gpu_vram_cap.value, 3ull << 30);
    LZ_CHECK(s.gpu_vram_cap.source == Source::Cli);
}
