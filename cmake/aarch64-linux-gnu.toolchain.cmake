# aarch64 cross toolchain. UNTESTED ON HARDWARE -- this file exists so the day an
# ARM64 box (or Apple silicon under Linux) is available, the build is one preset
# away, not a research project. ggml selects NEON paths by architecture; nothing
# in dray itself is ISA-specific (the I/O backends are platform-, not
# architecture-, conditional).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
