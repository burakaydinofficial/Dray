cmake_minimum_required(VERSION 3.21)
project(dray LANGUAGES C CXX)

# NOTE: the C++ standard is set PER TARGET, never globally. A global
# CMAKE_CXX_STANDARD 20 propagates into the vendored llama.cpp subdirectory, where
# C++20 retypes u8"" literals as char8_t and breaks llama-chat.cpp with a wall of
# C2664/C2280. llama.cpp builds itself at C++17; leave it there.
set(CMAKE_C_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Everything -- our exes, the test exes, and llama.cpp's DLLs -- lands in one
# directory. llama.cpp builds shared by default on Windows, so a test exe in its
# own subdirectory dies at startup with 0xc0000135 (DLL not found) before main()
# is ever reached, which reads like a crash rather than a packaging problem.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- vendored llama.cpp -------------------------------------------------------
# CPU-only for v1 (CLAUDE.md, Toolchain). A Metal/CUDA backend would require slab
# slots carved from device buffers with their own residency rules, and on discrete
# GPUs "uncached reads land directly in cache-owned memory" stops being possible.
# Owner decision 2026-08-21: ONE binary, runtime GPU opt-in (--gpu). On Apple
# the Metal backend needs no extra SDK, so it is compiled in by default and
# gated at runtime; opt-OUT (GPU on by default) is deferred until the Metal
# MUL_MAT_ID kernel honors nb[2] -- today consent forces whole-tensor mode,
# which must never become a silent default.
if (APPLE)
    option(DRAY_METAL_BUILD "build ggml Metal backend (runtime-gated by --gpu / DRAY_METAL=1)" ON)
else()
    option(DRAY_METAL_BUILD "build ggml Metal backend (runtime-gated by --gpu / DRAY_METAL=1)" OFF)
endif()
if (DRAY_METAL_BUILD)
    set(GGML_METAL           ON  CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
else()
    set(GGML_METAL           OFF CACHE BOOL "" FORCE)
endif()
set(GGML_CUDA            OFF CACHE BOOL "" FORCE)
# GPU is a BUILD-TIME opt-in (owner decision 2026-09-26, replacing the
# 2026-08-21 "one binary, runtime opt-in"). An unused Vulkan backend is not
# free: ggml registers it eagerly at startup (ggml_vk_instance_init), which
# loads the GPU driver and queries the device before --gpu is ever checked.
# Measured on K3 (laptop, Core Ultra 7 + RTX 4070): same code, same bytes and
# text, 18.4 s/token CPU-only vs 20.3 with Vulkan compiled in and unused. So
# the default build is CPU-only; configure with -DDRAY_VULKAN_BUILD=ON (the
# Vulkan SDK must be installed) to get --gpu. A binary without a GPU backend
# refuses --gpu instead of silently running on the CPU.
option(DRAY_VULKAN_BUILD "build the ggml Vulkan backend (enables --gpu; costs ~11% on CPU runs when unused)" OFF)
if (DRAY_VULKAN_BUILD)
    find_package(Vulkan QUIET)
endif()
if (DRAY_VULKAN_BUILD AND Vulkan_FOUND)
    set(GGML_VULKAN      ON  CACHE BOOL "" FORCE)
    message(STATUS "dray: Vulkan backend IN (--gpu available)")
else()
    if (DRAY_VULKAN_BUILD)
        message(WARNING "DRAY_VULKAN_BUILD=ON but the Vulkan SDK was not found: building CPU-only")
    endif()
    set(GGML_VULKAN      OFF CACHE BOOL "" FORCE)
    message(STATUS "dray: CPU-only build (default; -DDRAY_VULKAN_BUILD=ON for --gpu)")
endif()
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
# Tools are OFF here: at this SHA the unified `llama` app links llama-server-impl,
# so LLAMA_BUILD_TOOLS=ON drags in the server we do not want. The reference CLI
# used for token-for-token correctness checking is built separately in
# build/llamacpp with -DLLAMA_BUILD_SERVER=ON; see scripts/build-reference.ps1.
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
# The common library, for its chat templates and tool-call parsing only
# (dray_chatfmt). It also defines a cpp-httplib target; nothing of ours
# references it, so the static link never pulls it in (checked on the binary).
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# --- dray modules ----------------------------------------------------------
# One static library per concept, each linking ONLY what it uses, so the
# dependency graph is declared here and a layering violation is a link error:
#
#   config  mem  io                          (leaves: depend on nothing of ours)
#   plan        -> mem, ggml(gguf), llama(shard naming)
#   report      -> plan
#   calib       -> io, mem
#   snapshot    -> mem, llama
#   repack      -> plan
#   stream      -> io, mem, plan, ggml        (the streaming engine)
#   engine      -> stream, plan, config, repack, llama
#   server      -> engine, snapshot, httplib
#   dray.exe -> all of the above
#
# Headers are still addressed as "module/file.h" from src/, so the graph is
# enforced at link time, not at include time.
function(dray_module name)
    add_library(${name} STATIC ${ARGN})
    target_compile_features(${name} PUBLIC cxx_std_20)
    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if (WIN32)
        target_compile_definitions(${name} PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    if (MSVC)
        target_compile_options(${name} PRIVATE /W4 /permissive- /Zc:preprocessor /utf-8)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
endfunction()

dray_module(dray_config src/config/env_report.cpp src/config/settings.cpp src/config/paths.cpp
                              src/config/model_settings.cpp)
target_link_libraries(dray_config PRIVATE ggml)   # gguf.h: the header read for general.architecture
target_include_directories(dray_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)  # json.hpp

dray_module(dray_mem src/mem/accountant.cpp src/mem/pool_arena.cpp)

# The thread pool is unconditional, NOT in the WIN32 branch: make_backend() falls
# back to it on every platform, and on macOS it is the permanent answer.
dray_module(dray_io src/io/storage.cpp src/io/storage_threadpool.cpp)
if (WIN32)
    target_sources(dray_io PRIVATE src/io/storage_win.cpp)
elseif (APPLE)
    target_sources(dray_io PRIVATE src/io/storage_mac.cpp)
else()
    target_sources(dray_io PRIVATE src/io/storage_linux.cpp)
    include(CheckIncludeFile)
    check_include_file("liburing.h" DRAY_HAVE_LIBURING_H)
    if (DRAY_HAVE_LIBURING_H)
        target_compile_definitions(dray_io PRIVATE DRAY_HAVE_URING=1)
        target_link_libraries(dray_io PUBLIC uring)
    else()
        message(STATUS "liburing.h not found -- Linux runs on the thread pool")
    endif()
endif()


dray_module(dray_plan src/plan/residency.cpp)
target_link_libraries(dray_plan PUBLIC dray_mem PRIVATE ggml llama)

dray_module(dray_report src/report/readout.cpp)
target_link_libraries(dray_report PUBLIC dray_plan)

dray_module(dray_calib src/calib/calibrate.cpp)
target_link_libraries(dray_calib PUBLIC dray_io PRIVATE dray_mem)
target_include_directories(dray_calib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)  # json.hpp

dray_module(dray_snapshot src/cache/snapshot_cache.cpp)
target_link_libraries(dray_snapshot PUBLIC dray_mem llama)

dray_module(dray_repack src/tools/repack.cpp)
target_link_libraries(dray_repack PRIVATE dray_plan)
target_include_directories(dray_repack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)  # json.hpp

dray_module(dray_stream
    src/backend/accounted_alloc.cpp
    src/backend/expert_compactor.cpp
    src/backend/expert_slots.cpp
    src/backend/hit_rates.cpp
    src/backend/io_scheduler.cpp
    src/backend/merged_metadata.cpp
    src/backend/poison_buffers.cpp
    src/backend/private_ids.cpp
    src/backend/residency_cache.cpp
    src/backend/routing_skew.cpp
    src/backend/stream_budget.cpp
    src/backend/stream_buffer.cpp
    src/backend/stream_buffer_type.cpp
    src/backend/stream_flags.cpp
    src/backend/uncond_ring.cpp)
target_link_libraries(dray_stream PUBLIC dray_io dray_mem dray_plan ggml)
# ggml-backend-impl.h is private to ggml/src. Implementing a buffer type requires
# it -- every in-tree ggml backend includes it the same way. This is a real
# coupling to a non-public header and will break on upstream churn; that is the
# price of streaming weights, and it is cheaper than forking the loader.
target_include_directories(dray_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src)


dray_module(dray_engine
    src/engine/engine.cpp
    src/engine/engine_setup.cpp
    src/engine/batch_generator.cpp
    src/engine/cohort_rotator.cpp
    src/engine/slot.cpp
    src/engine/scheduler.cpp
    src/engine/ram_kv_store.cpp
    src/engine/llama_stepper.cpp
    src/engine/streamed_text.cpp
    src/engine/kv_overrides.cpp
    src/engine/thread_policy.cpp)
target_link_libraries(dray_engine
    PUBLIC  llama
    PRIVATE dray_stream dray_plan dray_mem dray_config dray_repack ggml-cpu)
# llama-ext.h (llama_get_memory_breakdown) lives beside llama.cpp's sources.
target_include_directories(dray_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src)

# llama.cpp's chat library (templates, tool calls, reasoning), behind strings: its
# JSON and HTTP libraries stay out of every other translation unit.
dray_module(dray_chatfmt src/server/chat_format.cpp)
target_link_libraries(dray_chatfmt PUBLIC llama PRIVATE llama-common)

dray_module(dray_server
    src/server/serve.cpp
    src/server/request_guard.cpp
    src/server/chat_request.cpp
    src/server/chat_response.cpp
    src/server/chat_completions.cpp
    src/server/http_routes.cpp
    src/server/job_registry.cpp
    src/server/job_store.cpp
    src/server/job_worker.cpp
    src/server/shutdown.cpp)
target_link_libraries(dray_server PUBLIC dray_engine PRIVATE dray_snapshot dray_chatfmt)
target_include_directories(dray_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)  # httplib, json
if (WIN32)
    # httplib needs winsock; no TLS in v1, so ws2_32 alone suffices.
    target_link_libraries(dray_server PRIVATE ws2_32)
endif()

# Everything, for consumers that exercise several modules at once (the tests).
add_library(dray_core INTERFACE)
target_link_libraries(dray_core INTERFACE
    dray_config dray_mem dray_io dray_plan dray_report
    dray_calib dray_snapshot dray_repack dray_stream
    dray_engine llama ggml)

# --- cli ----------------------------------------------------------------------
add_executable(dray
    src/main.cpp
    src/cli/args.cpp
    src/cli/engine_config.cpp
    src/cli/reference_read.cpp
    src/cli/cmd_plan.cpp
    src/cli/cmd_calibrate.cpp
    src/cli/cmd_verify.cpp
    src/cli/cmd_run.cpp
    src/cli/cmd_run_reference.cpp
    src/cli/cmd_batch.cpp
    src/cli/cmd_snaptest.cpp
    src/cli/cmd_serve.cpp
    src/cli/cmd_repack.cpp
    src/cli/cmd_config.cpp)
target_include_directories(dray PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(dray PRIVATE
    dray_server dray_engine dray_stream dray_plan
    dray_mem dray_io dray_calib dray_report dray_snapshot dray_repack
    dray_config llama ggml)
if (WIN32)
    target_compile_definitions(dray PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
# The shipped settings (config/system.json, config/models/<arch>.json) live next
# to the binary, where config::default_locations looks for them; copied on every
# build so an edit to a shipped default is never silently stale. A target of its
# own, always run: as a POST_BUILD step of dray it ran only when the binary
# relinked, so a config-only edit (a new models/<arch>.json) never arrived.
add_custom_target(dray_shipped_config ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/config $<TARGET_FILE_DIR:dray>/config
    COMMENT "Copying the shipped config next to dray")
add_dependencies(dray dray_shipped_config)

# --- tests --------------------------------------------------------------------
include(CTest)
if (BUILD_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()

# --- install ------------------------------------------------------------------
# T25: `cmake --install` places the binary and its runtime DLLs; without this
# the README invoked a binary the build never put anywhere nameable. llama.cpp
# is EXCLUDE_FROM_ALL, whose own install rules CMake documents as undefined --
# so only OUR targets install, deliberately.
install(TARGETS dray RUNTIME DESTINATION bin)
if (WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:dray> DESTINATION bin)
endif()
install(DIRECTORY config/ DESTINATION bin/config)
install(FILES LICENSE THIRD_PARTY.md
        third_party/LICENSE-cpp-httplib third_party/LICENSE-nlohmann-json
        DESTINATION share/dray)
