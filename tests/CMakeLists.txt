# Test suite.
#
# This project cannot casually run its own subject matter: the smallest target is
# 143 GB and a real run is hours, so the test strategy IS the verification
# strategy and all of it stands on synthetic fixtures. Two rules follow:
#
#   * every test must finish in seconds and must not need a real model;
#   * no external test framework -- the dependency budget is the C++ standard
#     library plus vendored llama.cpp/ggml (CLAUDE.md, Toolchain). harness.h is
#     the whole runner.
#
# Each test is one executable and one add_test(), so `ctest -R residency` works.

# std::thread is used by the accountant's concurrency cases.
find_package(Threads REQUIRED)

# The synthetic GGUF writer, shared by every test that needs a tensor table.
add_library(dray_test_fixtures STATIC
    fixtures/make_gguf.cpp
)

# PUBLIC so the test executables get "harness.h" and "fixtures/..." for free.
target_include_directories(dray_test_fixtures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# dray_core carries llama+ggml and the src/ include dir transitively; ggml is
# named explicitly because the fixture calls the gguf writer directly.
target_link_libraries(dray_test_fixtures PUBLIC dray_core ggml)

function(dray_test_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:preprocessor /utf-8)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
endfunction()

dray_test_flags(dray_test_fixtures)

# name -> tests/test_<name>.cpp -> executable test_<name> -> `ctest -R <name>`
function(dray_add_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE dray_test_fixtures)
    dray_test_flags(test_${name})
    target_link_libraries(test_${name} PRIVATE Threads::Threads)

    add_test(NAME ${name}
             COMMAND test_${name}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # Generous relative to the seconds these should take: the ceiling exists to
    # catch a poll() that blocks forever, not to police performance.
    set_tests_properties(${name} PROPERTIES TIMEOUT 120 LABELS unit)
endfunction()

# The fixture is the backbone of everything else, so it is verified first.
dray_add_test(fixture_gguf)

dray_add_test(accountant)
dray_add_test(pool_arena)
dray_add_test(graph_reuse)
dray_add_test(env_census)
dray_add_test(pressure)
dray_add_test(readout)
dray_add_test(two_shard)
dray_add_test(residency)
# ResidencyCache::take_region: region reuse must behave exactly as an eviction.
dray_add_test(region_reuse)
# ExpertSlots: the admission byte budget (the storm guard).
dray_add_test(expert_slots)
# Settings layers, strict parsing, system-only hardware fields.
dray_add_test(settings)
# Scheduler: continuous batching over a scripted stepper.
dray_add_test(scheduler)
dray_add_test(storage)
# Differential check on expert compaction: the streamer's central trick, and the
# one place where being wrong yields fluent text from the wrong weights rather
# than any error. Runs against ggml directly, so it needs no model.
dray_add_test(compaction)
# The streamer's sizing arithmetic (churn, ring, eslot pool, sentinel), which
# until 2026-09-25 could only be checked by loading a model.
dray_add_test(stream_budget)
# Stop sequences on every path: the one find_stop() rule and StreamedText.
dray_add_test(stop_match)
# The I/O scheduler against a scripted fake backend: out-of-order, short, failing
# and dead reads -- the cases no model run can produce on demand.
dray_add_test(io_scheduler)

# The snapshot cache had no header when these tests were written (the root
# CMakeLists.txt lists src/cache/snapshot_cache.cpp, but src/cache/ was empty).
# test_snapshot_cache.cpp documents the contract it assumes; it is compiled only
# once a header of that shape exists, so a differently-shaped real API shows up
# here as a skip with a message rather than as a broken build for everyone.
set(_snapshot_header ${CMAKE_SOURCE_DIR}/src/cache/snapshot_cache.h)
if (EXISTS ${_snapshot_header})
    file(READ ${_snapshot_header} _snapshot_header_text)
    string(FIND "${_snapshot_header_text}" "class SnapshotCache" _has_class)
    # Reconciled 2026-08-15: the test now targets the REAL header (the old note
    # below this line argued the reverse and predated the header's design; the
    # store/load surface is context-bound by design, so the test moved). The
    # gate now only requires the class and the exported prefix_hash.
    string(FIND "${_snapshot_header_text}" "prefix_hash" _has_hash)
    if (_has_class GREATER -1 AND _has_hash GREATER -1)
        dray_add_test(snapshot_cache)
    else()
        message(STATUS
            "tests: skipping snapshot_cache -- src/cache/snapshot_cache.h exists but does "
            "not declare class SnapshotCache + prefix_hash. Reconcile "
            "tests/test_snapshot_cache.cpp with the real header and re-enable it here.")
    endif()
else()
    message(STATUS
        "tests: skipping snapshot_cache -- src/cache/snapshot_cache.h does not exist yet. "
        "tests/test_snapshot_cache.cpp is written and waiting for it.")
endif()

# ChatFormat: the model's own chat template (tools, reasoning) and the output
# parser, through llama.cpp's chat library, on real templates from its repo.
dray_add_test(chat_format)
target_link_libraries(test_chat_format PRIVATE dray_chatfmt)
target_include_directories(test_chat_format PRIVATE ${CMAKE_SOURCE_DIR}/third_party)   # json.hpp
target_compile_definitions(test_chat_format PRIVATE
    DRAY_LLAMA_TEMPLATES="${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates")

# chat_request / chat_response: the /v1/chat/completions contract, pure.
dray_add_test(chat_api)
target_link_libraries(test_chat_api PRIVATE dray_server dray_chatfmt)
target_include_directories(test_chat_api PRIVATE ${CMAKE_SOURCE_DIR}/third_party)   # json.hpp
target_compile_definitions(test_chat_api PRIVATE
    DRAY_LLAMA_TEMPLATES="${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates")
