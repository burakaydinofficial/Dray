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
# GPU (owner decision 2026-08-21): ONE binary, runtime opt-in. The Vulkan
# backend is compiled in whenever the SDK is present so users opt in AT START
# (--gpu or DRAY_VULKAN=1); without consent the engine restricts llama to
# CPU devices and stays byte-identical to a CPU-only build (difftest-gated).
# Machines without the SDK (CI runners included) build CPU-only, silently
# correct. DRAY_VULKAN_BUILD=OFF still forces it out entirely.
option(DRAY_VULKAN_BUILD "build ggml Vulkan backend when the SDK is found (runtime-gated by --gpu)" ON)
find_package(Vulkan QUIET)
if (DRAY_VULKAN_BUILD AND Vulkan_FOUND)
    set(GGML_VULKAN      ON  CACHE BOOL "" FORCE)
    message(STATUS "dray: Vulkan backend IN (runtime opt-in via --gpu / DRAY_VULKAN=1)")
else()
    set(GGML_VULKAN      OFF CACHE BOOL "" FORCE)
    message(STATUS "dray: Vulkan backend OUT (SDK not found or disabled); CPU-only build")
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

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# --- dray core -------------------------------------------------------------
add_library(dray_core STATIC
    # The Streamer: one owner per concern, composed in src/backend/streamer_impl.h.
    src/backend/accounted_alloc.cpp
    src/backend/expert_compactor.cpp
    src/backend/expert_slots.cpp
    src/backend/hit_rates.cpp
    src/backend/io_scheduler.cpp
    src/backend/merged_metadata.cpp
    src/backend/poison_buffers.cpp
    src/backend/private_ids.cpp
    src/backend/repacker.cpp
    src/backend/residency_cache.cpp
    src/backend/routing_skew.cpp
    src/backend/stream_budget.cpp
    src/backend/stream_buffer.cpp
    src/backend/stream_buffer_type.cpp
    src/backend/stream_flags.cpp
    src/backend/uncond_ring.cpp
    src/mem/accountant.cpp
    src/mem/pool_arena.cpp
    src/config/env_report.cpp
    src/io/storage.cpp
    # Unconditional, NOT in the WIN32 branch: make_backend() falls back to the
    # thread-pool floor on every platform, and on macOS it is the permanent answer.
    src/io/storage_threadpool.cpp
    src/models/registry.cpp
    src/models/minimax_m3.cpp
    src/plan/residency.cpp
    src/cache/expert_cache.cpp
    src/cache/snapshot_cache.cpp
    src/calib/calibrate.cpp
    src/report/readout.cpp
)

target_compile_features(dray_core PUBLIC cxx_std_20)

if (WIN32)
    target_sources(dray_core PRIVATE src/io/storage_win.cpp)
    target_compile_definitions(dray_core PUBLIC DRAY_WIN=1 NOMINMAX WIN32_LEAN_AND_MEAN)
elseif (APPLE)
    target_sources(dray_core PRIVATE src/io/storage_mac.cpp)
    target_compile_definitions(dray_core PUBLIC DRAY_MAC=1)
else()
    target_sources(dray_core PRIVATE src/io/storage_linux.cpp)
    target_compile_definitions(dray_core PUBLIC DRAY_LINUX=1)
    include(CheckIncludeFile)
    check_include_file("liburing.h" DRAY_HAVE_LIBURING_H)
    if (DRAY_HAVE_LIBURING_H)
        target_compile_definitions(dray_core PRIVATE DRAY_HAVE_URING=1)
        target_link_libraries(dray_core PUBLIC uring)
    else()
        message(STATUS "liburing.h not found -- Linux runs on the thread pool")
    endif()
endif()

target_include_directories(dray_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# ggml-backend-impl.h is private to ggml/src. Implementing a buffer type requires
# it -- every in-tree ggml backend includes it the same way. This is a real
# coupling to a non-public header and will break on upstream churn; that is the
# price of streaming weights, and it is cheaper than forking the loader.
target_include_directories(dray_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)   # json.hpp (T5 calibration persistence)
target_link_libraries(dray_core PUBLIC llama ggml)

if (MSVC)
    target_compile_options(dray_core PRIVATE /W4 /permissive- /Zc:preprocessor /utf-8)
else()
    target_compile_options(dray_core PRIVATE -Wall -Wextra)
endif()

# --- cli ----------------------------------------------------------------------
add_executable(dray
    src/main.cpp
    src/server/engine.cpp
    src/server/serve.cpp
    src/tools/repack.cpp)
target_include_directories(dray PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(dray PRIVATE dray_core)
if (WIN32)
    # httplib needs winsock; no TLS in v1, so ws2_32 alone suffices.
    target_link_libraries(dray PRIVATE ws2_32)
endif()

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
install(FILES LICENSE THIRD_PARTY.md
        third_party/LICENSE-cpp-httplib third_party/LICENSE-nlohmann-json
        DESTINATION share/dray)
