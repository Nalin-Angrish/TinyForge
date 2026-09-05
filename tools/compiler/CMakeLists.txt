# TinyForge Compiler — tinyforge-compile
# Lives in TinyForge (not FlexNN) so TinyForge depends on FlexNN, not vice versa.
# Parses FlexNN's model.bin (via external/FlexNN's ModelIO structs) and emits
# generated/model_data.h for TinyForge's runtime.

cmake_minimum_required(VERSION 3.10)
project(tinyforge-compile)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FlexNN is needed only for ModelIO format / Eigen types to parse model.bin
# We vendor it as external/FlexNN (TinyForge submodule)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../external/FlexNN/CMakeLists.txt)
  # We only need FlexNN's headers for ModelIO, not its main target
  # For now, just add its include path
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../external/FlexNN/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
else()
  message(WARNING "external/FlexNN not found — compiler will need it to parse model.bin")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

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
