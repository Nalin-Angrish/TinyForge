cmake_minimum_required(VERSION 3.10)
project(TinyForge)

# Set the C++ standard — mirrors FlexNN
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories — like FlexNN's include_directories(include)
include_directories(include)
# For generated model header (from FlexNN's tinyforge-compile)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ── TinyForge library — mirrors FlexNN/lib/ ──
# Public headers: include/tinyforge/
# Sources: lib/
set(LIB_SOURCES
    lib/runtime.cpp
    lib/backend_scalar.cpp
    lib/quant.cpp
    lib/activations.cpp
)

# CMSIS-NN backend is optional — enable with -DTINYFORGE_USE_CMSIS_NN=ON
# and a toolchain that provides arm_nnfunctions.h. For host builds the
# scalar backend is used (default, portable).
option(TINYFORGE_USE_CMSIS_NN "Build with CMSIS-NN backend" OFF)
if(TINYFORGE_USE_CMSIS_NN)
  list(APPEND LIB_SOURCES lib/backend_cmsis.cpp)
  add_definitions(-DTINYFORGE_BACKEND=1 -DTINYFORGE_BACKEND_CMSIS=1)
  message(STATUS "TinyForge: CMSIS-NN backend enabled")
else()
  add_definitions(-DTINYFORGE_BACKEND=0 -DTINYFORGE_BACKEND_SCALAR=1)
  message(STATUS "TinyForge: scalar backend enabled")
endif()

add_library(TinyForge ${LIB_SOURCES})
add_library(TinyForge::TinyForge ALIAS TinyForge)

target_include_directories(TinyForge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ── Example firmware — mirrors FlexNN/src/main.cpp ──
# src/main.cpp is a minimal example that links against the library
# and a generated model (generated/model_data.h from FlexNN).
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE TinyForge)

# ── Compiler flags — no heap/exceptions for MCU builds via toolchain
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(TinyForge PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# ── TinyForge compiler — lives HERE (TinyForge depends on FlexNN, fetched via CMake)
# Parses FlexNN's model.bin (ModelIO format) and emits generated/model_data.h.
# FlexNN is NOT a submodule — it is fetched from git when the compiler is built.
# We only need FlexNN's *headers* (ModelIO format), not its full CMake+Eig en build.
option(TINYFORGE_BUILD_COMPILER "Build tinyforge-compile" ON)
option(TINYFORGE_FETCH_FLEXNN "Fetch FlexNN headers from git for compiler" ON)
if(TINYFORGE_BUILD_COMPILER)
  if(TINYFORGE_FETCH_FLEXNN)
    include(FetchContent)
    FetchContent_Declare(
      FlexNN_src
      GIT_REPOSITORY https://github.com/Nalin-Angrish/FlexNN.git
      GIT_TAG 78b83a5cfb478f3def3f70304ea7dc35d3c0f814 # pinned SHA, like submodule did
      GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(FlexNN_src)
    if(NOT flexnn_src_POPULATED)
      FetchContent_Populate(FlexNN_src)
      message(STATUS "TinyForge: fetched FlexNN headers at ${flexnn_src_SOURCE_DIR} for compiler")
    endif()
    # Expose as interface so compiler can find FlexNN headers without needing Eigen
    add_library(FlexNN_headers INTERFACE)
    target_include_directories(FlexNN_headers INTERFACE ${flexnn_src_SOURCE_DIR}/include)
    # Also make FlexNN_SOURCE_DIR available for compiler's CMake
    set(FlexNN_SOURCE_DIR ${flexnn_src_SOURCE_DIR} CACHE INTERNAL "")
  else()
    find_package(FlexNN REQUIRED)
    message(STATUS "TinyForge: using system FlexNN for compiler")
  endif()
  add_subdirectory(tools/compiler)
endif()

# ── Tests — host unit tests (like FlexNN, optional)
option(TINYFORGE_BUILD_TESTS "Build host unit tests" OFF)
if(TINYFORGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── Install — makes TinyForge importable as a package
#   add_subdirectory(TinyForge) + target_link_libraries(fw PRIVATE TinyForge::TinyForge)
#   or find_package(TinyForge)
include(GNUInstallDirs)
install(TARGETS TinyForge main
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

message(STATUS "TinyForge configured — TinyForge depends on FlexNN (fetched via FetchContent), not vice versa")
message(STATUS "  FlexNN (standalone): git clone https://github.com/Nalin-Angrish/FlexNN && cmake -S FlexNN -B build && cmake --build build  # trains, exports model.bin")
message(STATUS "  TinyForge compiler: cmake -S . -B build && ./build/tools/compiler/tinyforge-compile model.bin -o generated")
