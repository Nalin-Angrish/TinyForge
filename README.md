# TinyForge

Embeddable inference library for STM32F407 (Cortex-M4F). Run neural networks compiled by [FlexNN](https://github.com/Nalin-Angrish/FlexNN) on-device with a tiny, no-heap, no-interpreter runtime.

**FlexNN** is the PC-side library + CLI that trains, exports, validates, quantizes, and emits a C header. **TinyForge** is the MCU-side library that *only* runs that header. FlexNN lives at `github.com/Nalin-Angrish/FlexNN`; TinyForge vendors it as `external/FlexNN` for codegen and for keeping the op registry in sync.

See `ref/OVERVIEW.md` and `ref/LLD.md` (local, gitignored) for the full spec and low-level design.

## Architecture

```
FlexNN (PC, CMake, Eigen)                     TinyForge (MCU, CMake)
  train with FlexNN.h/Layer.h                   #include "tinyforge/tinyforge.h"
       │                                         #include "generated/model_data.h"
       ▼                                                │
  exportModel("model.bin")                              │
       │                                                │
       └────► tinyforge-compile (from FlexNN) ───────────┘
                   │  validates, quantizes int8,
                   │  emits generated/model_data.h
                   ▼
              tinyforge::Model::run(input, output)  ──►  scalar or CMSIS-NN
```

TinyForge itself has no training, no parser, no Eigen. It just consumes `generated/model_data.h`.

## Repository Layout — mirrors FlexNN

```
TinyForge/  (this repo, private)
├── CMakeLists.txt              # builds libTinyForge (like FlexNN/CMakeLists.txt)
├── include/
│   └── tinyforge/
│       ├── tinyforge.h         # aggregator (like FlexNN.h)
│       ├── runtime.hpp         # Model class
│       ├── types.hpp           # LayerType, Activation, Status (mirrors FlexNN/LayerTypes.hpp)
│       ├── backend.hpp         # scalar vs CMSIS-NN compile-time switch
│       ├── quant.hpp
│       └── kernels/{dense,conv1d,activations}.hpp
├── lib/                        # library sources (like FlexNN/lib/)
│   ├── runtime.cpp
│   ├── backend_scalar.cpp
│   ├── backend_cmsis.cpp
│   ├── quant.cpp
│   └── activations.cpp
├── examples/                   # example firmware mains (like FlexNN/src/main.cpp)
│   ├── anomaly_detection/main.cpp  # 7.1 vibration
│   └── gesture/main.cpp            # 7.2 gesture
├── tests/                      # host unit tests (like FlexNN/tests/)
│   └── test_runtime.cpp
├── generated/                  # output of FlexNN's compiler (gitignored)
│   └── model_data.h
├── external/FlexNN/            # submodule → github.com/Nalin-Angrish/FlexNN @ main
│   └── build/tools/compiler/tinyforge-compile  # after `cmake --build` in FlexNN
└── ref/{OVERVIEW.md, LLD.md}   # local docs, gitignored
```

## Prerequisites

- **For TinyForge library (MCU or host):** `cmake >=3.10`, `arm-none-eabi-gcc` (for MCU), or `g++`/`clang++` (for host tests)
- **For FlexNN + compiler (to generate the model):** `cmake`, `libeigen3-dev`, `OpenMP` (see `external/FlexNN/README.md`)
- `git` with submodule support

## Clone

```bash
git clone --recurse-submodules https://github.com/Nalin-Angrish/TinyForge.git
cd TinyForge
# or: git submodule update --init --recursive
```

## 1) Generate a model with FlexNN (PC)

FlexNN now owns the compiler (`FlexNN/tools/compiler`). TinyForge only consumes its output.

```bash
# Build FlexNN + its compiler
cmake -S external/FlexNN -B external/FlexNN/build
cmake --build external/FlexNN/build -j

# Train & export (FlexNN example)
./external/FlexNN/build/main  # or your own FlexNN program that calls exportModel("model.bin")

# Compile to a header for TinyForge (validates, quantizes, checks arena)
external/FlexNN/build/tools/compiler/tinyforge-compile model.bin \
  --backend cmsis --calib data/calib.csv --arena 98304 \
  -o generated
# → writes generated/model_data.h
```

See `ref/LLD.md §5-§7` for the flat binary format and int8 quantization.

## 2) Use TinyForge as a library in your firmware

### Option A: add_subdirectory (recommended, mirrors FlexNN usage)

```cmake
# In your firmware's CMakeLists.txt
add_subdirectory(path/to/TinyForge)
target_link_libraries(your_firmware PRIVATE TinyForge::TinyForge)
target_include_directories(your_firmware PRIVATE
  path/to/TinyForge/include
  path/to/TinyForge/generated
)
```

```cpp
// In your firmware's main.cpp (see examples/anomaly_detection/main.cpp)
#include "tinyforge/tinyforge.h"
#include "generated/model_data.h"

int main() {
  tinyforge::Model model;
  if (model.init() != tinyforge::Status::Ok) { /* blink error */ }
  int8_t input[TINYFORGE_INPUT_DIM];
  int8_t output[TINYFORGE_OUTPUT_DIM];
  // fill input from sensor...
  model.run(input, output);
  // output is logits; argmax if classification
}
```

### Option B: Install & find_package

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/tinyforge
cmake --build build && cmake --install build
# then in firmware: find_package(TinyForge REQUIRED)
```

### Option C: Vendor as a static library

Just copy `include/tinyforge/` and `lib/` into your project and add them to your build.

## Build TinyForge itself

### Host (scalar, for unit tests — like FlexNN host build)

```bash
cmake -S . -B build -DTINYFORGE_BUILD_TESTS=ON -DTINYFORGE_BUILD_EXAMPLES=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tests/test_runtime
./build/examples/anomaly_detection/anomaly_detection
```

### Cross-compiled for STM32F407 (scalar)

Supply a toolchain file for `arm-none-eabi-gcc` (e.g., `cmake/toolchains/arm-cortex-m4f.cmake`):

```cmake
# cmake/toolchains/arm-cortex-m4f.cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--gc-sections -Wl,--wrap=malloc -Wl,--wrap=free")
```

```bash
cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake \
  -DTINYFORGE_BACKEND_SCALAR=ON
cmake --build build-mcu -j
arm-none-eabi-size build-mcu/libTinyForge.a
```

### With CMSIS-NN (STM32Cube + CMSIS packs)

```bash
cmake -S . -B build-mcu-cmsis \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake \
  -DTINYFORGE_BACKEND_CMSIS_NN=ON
cmake --build build-mcu-cmsis -j
# Ensure CMSIS-NN/CMSIS-DSP are available via -I or lib/ vendor
```

No `platformio.ini` — TinyForge is now a plain CMake library like FlexNN. If you use PlatformIO, add it as `lib_deps` or `lib_extra_dirs`.

## Developing FlexNN via the submodule

This repo tracks FlexNN at `external/FlexNN` on `branch main`. You can develop FlexNN (including its compiler) in place:

```bash
git submodule update --remote external/FlexNN
cd external/FlexNN
# ... edit Layer.h, tools/compiler/src/*.cpp, etc. ...
git commit -m "feat: add Conv1D+Tanh"
git push origin main
cd ../..
git add external/FlexNN
git commit -m "chore: bump FlexNN to <sha>"
git push
```

`ref/LLD.md §2, §8` explains why the compiler lives in FlexNN and why TinyForge includes `external/FlexNN/include/tinyforge/op_registry.hpp` for the single source of truth.

## Testing

- **FlexNN host:** `ctest --test-dir external/FlexNN/build` (training + compiler `tinyforge-compile --check-only`)
- **TinyForge host:** `ctest --test-dir build` (scalar vs float reference, see `tests/test_runtime.cpp`)
- **On-device:** examples in `examples/` are flashed and checked via UART (`DWT_CYCCNT` latency, `arm-none-eabi-size` for RAM/flash, accuracy vs FlexNN float)

## Toolchain

- Host: `g++`/`clang++`, `Eigen3`, `OpenMP` (for FlexNN)
- MCU: `arm-none-eabi-gcc`, STM32F407G-DISC1, optional `CMSIS-NN`/`CMSIS-DSP` for accelerated backend

## License

Apache-2.0 (same as FlexNN). See `LICENSE` when added.
