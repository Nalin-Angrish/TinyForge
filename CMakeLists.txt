cmake_minimum_required(VERSION 3.10)
project(TinyForge VERSION 0.2.0 LANGUAGES C CXX)

# ──────────────────────────────────────────────────────────────
# TinyForge — embeddable inference library for Cortex-M4F
# Mirrors FlexNN's structure:
#   include/tinyforge/  → public headers
#   lib/               → library sources (like FlexNN/lib/)
#   examples/          → example firmware mains (like FlexNN/src/main.cpp)
#   tests/             → host unit tests
# This CMake builds the library for either host (scalar, for tests)
# or cross-compiled MCU (scalar or CMSIS-NN via toolchain file).
# FlexNN is *not* required to build the runtime itself — it is only
# needed to generate generated/model_data.h via its tinyforge-compile CLI.
# ──────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options — similar style to FlexNN
option(TINYFORGE_BACKEND_SCALAR "Build scalar backend (portable, host-capable)" ON)
option(TINYFORGE_BACKEND_CMSIS_NN "Build CMSIS-NN backend (requires CMSIS)" OFF)
option(TINYFORGE_BUILD_TESTS "Build host unit tests" OFF)
option(TINYFORGE_BUILD_EXAMPLES "Build example firmware binaries" OFF)
option(TINYFORGE_USE_FLEXNN "Use vendored FlexNN for host tests/tools (optional)" OFF)

# Default to scalar if neither chosen
if(NOT TINYFORGE_BACKEND_SCALAR AND NOT TINYFORGE_BACKEND_CMSIS_NN)
  set(TINYFORGE_BACKEND_SCALAR ON)
endif()

# ── Library sources — mirrors FlexNN/lib/ ──
set(TINYFORGE_LIB_SOURCES
  lib/runtime.cpp
  lib/backend_scalar.cpp
  lib/quant.cpp
  lib/activations.cpp
)

if(TINYFORGE_BACKEND_CMSIS_NN)
  list(APPEND TINYFORGE_LIB_SOURCES lib/backend_cmsis.cpp)
endif()

add_library(TinyForge STATIC ${TINYFORGE_LIB_SOURCES})
add_library(TinyForge::TinyForge ALIAS TinyForge)

target_include_directories(TinyForge
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Compiler flags — no heap, no exceptions, no RTTI for MCU-like builds
# Host builds keep sanitizers optional; MCU builds are optimized via toolchain
target_compile_options(TinyForge PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>
  $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -Werror>
)

# For MCU builds, the user supplies a toolchain file (e.g., cmake/toolchains/arm-cortex-m4f.cmake)
# that sets -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard etc.
# We do NOT hardcode those here so host builds remain native.

# ── Backend defines — compile-time selection, no runtime branch
if(TINYFORGE_BACKEND_CMSIS_NN)
  target_compile_definitions(TinyForge PUBLIC TINYFORGE_BACKEND=1 TINYFORGE_BACKEND_CMSIS=1)
  # Find CMSIS-NN/DSP if available (optional, not required for scalar host tests)
  # find_package(CMSISNN QUIET)  # vendored via lib/ or external/ if needed
  message(STATUS "TinyForge: CMSIS-NN backend enabled — ensure CMSIS packs are available for MCU builds")
else()
  target_compile_definitions(TinyForge PUBLIC TINYFORGE_BACKEND=0 TINYFORGE_BACKEND_SCALAR=1)
  message(STATUS "TinyForge: scalar backend enabled (portable)")
endif()

# ── Hide FlexNN from runtime — only for host tools
# FlexNN is only needed to generate generated/model_data.h, not to link TinyForge itself.
if(TINYFORGE_USE_FLEXNN)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/FlexNN/CMakeLists.txt)
    message(STATUS "TinyForge: vendored FlexNN found at external/FlexNN — available for tests/tools")
    # Do NOT add_subdirectory unconditionally; FlexNN is a PC library with Eigen/OpenMP.
    # Host tests that need it can do it themselves. We just expose the path.
    set(TINYFORGE_FLEXNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/FlexNN)
  else()
    message(WARNING "TINYFORGE_USE_FLEXNN=ON but external/FlexNN not found. Run: git submodule update --init --recursive")
  endif()
endif()

# ── Examples — like FlexNN/src/main.cpp ──
if(TINYFORGE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ── Tests — like FlexNN/tests/ ──
if(TINYFORGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── Install rules — makes TinyForge importable via add_subdirectory or find_package
include(GNUInstallDirs)
install(TARGETS TinyForge
  EXPORT TinyForgeTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT TinyForgeTargets
  FILE TinyForgeTargets.cmake
  NAMESPACE TinyForge::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinyForge
)

# ── Summary
message(STATUS "TinyForge ${PROJECT_VERSION} configured")
message(STATUS "  Backend scalar: ${TINYFORGE_BACKEND_SCALAR}")
message(STATUS "  Backend CMSIS-NN: ${TINYFORGE_BACKEND_CMSIS_NN}")
message(STATUS "  Build tests: ${TINYFORGE_BUILD_TESTS}")
message(STATUS "  Build examples: ${TINYFORGE_BUILD_EXAMPLES}")
message(STATUS "  Use FlexNN: ${TINYFORGE_USE_FLEXNN} (only for model generation)")
message(STATUS "  Usage: FlexNN builds tinyforge-compile → generates TinyForge/generated/model_data.h →")
message(STATUS "         TinyForge builds libTinyForge for your firmware: target_link_libraries(firmware PRIVATE TinyForge::TinyForge)")
