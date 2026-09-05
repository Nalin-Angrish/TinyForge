cmake_minimum_required(VERSION 3.10)
project(TinyForge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- FlexNN (PC-side training library) as submodule ---
# Provides FlexNN target and Eigen dependency.
# Used by the TinyForge compiler/tools. NOT for on-device firmware.
option(TINYFORGE_USE_FLEXNN "Build with FlexNN submodule" ON)
if(TINYFORGE_USE_FLEXNN)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/FlexNN/CMakeLists.txt)
        add_subdirectory(external/FlexNN)
    else()
        message(WARNING "external/FlexNN not found. Run: git submodule update --init --recursive")
    endif()
endif()

# --- TinyForge Compiler (PC-side CLI) ---
# Placeholder: will parse FlexNN-exported models, validate, quantize, emit C header.
# Uncomment and flesh out tools/compiler/CMakeLists.txt when ready.
# add_subdirectory(tools/compiler)

# --- Example: host-side test that links FlexNN ---
# add_executable(tinyforge_host_test tools/host_test.cpp)
# target_link_libraries(tinyforge_host_test PRIVATE FlexNN Eigen3::Eigen)
# target_include_directories(tinyforge_host_test PRIVATE external/FlexNN/include)

message(STATUS "TinyForge CMake configured. FlexNN: ${TINYFORGE_USE_FLEXNN}")
