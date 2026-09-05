# TinyForge Compiler — tinyforge-compile
# Lives in TinyForge (not FlexNN) so TinyForge depends on FlexNN, not vice versa.
# Parses FlexNN's model.bin (via external/FlexNN's ModelIO structs) and emits
# generated/model_data.h for TinyForge's runtime.

cmake_minimum_required(VERSION 3.10)
project(tinyforge-compile)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FlexNN is fetched by parent TinyForge via FetchContent (or find_package)
# It provides ModelIO format and Eigen types for parsing model.bin.
# No vendored external/ needed.

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
# FlexNN headers from FetchContent (parent) — via interface target or source dir
if(TARGET FlexNN_headers)
  # Header-only, no Eigen needed to build TinyForge's compiler stub
  get_target_property(_inc FlexNN_headers INTERFACE_INCLUDE_DIRECTORIES)
  include_directories(${_inc})
elseif(TARGET FlexNN)
  # Full FlexNN was fetched (old path)
elseif(DEFINED flexnn_src_SOURCE_DIR)
  include_directories(${flexnn_src_SOURCE_DIR}/include)
elseif(DEFINED FlexNN_SOURCE_DIR)
  include_directories(${FlexNN_SOURCE_DIR}/include)
endif()

set(COMPILER_SOURCES
  src/main.cpp
  src/parser.cpp
  src/validate.cpp
  src/quantize.cpp
  src/codegen.cpp
)

add_executable(tinyforge-compile ${COMPILER_SOURCES})
target_link_libraries(tinyforge-compile PRIVATE TinyForge)

# For host calibration, link FlexNN if available
if(TARGET FlexNN)
  target_link_libraries(tinyforge-compile PRIVATE FlexNN)
endif()

install(TARGETS tinyforge-compile DESTINATION bin)
