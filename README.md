# TinyForge

Embeddable inference library for Cortex-M4F — run FlexNN models on STM32 with no heap, no interpreter.

TinyForge is the MCU-side companion to [FlexNN](https://github.com/Nalin-Angrish/FlexNN). **FlexNN** trains and exports (`FlexNN → model.bin`); **TinyForge** compiles and runs (`tinyforge-compile → model_data.h → Model::run()`). FlexNN is standalone — TinyForge depends on FlexNN, not vice versa.

## Example Usage

```cpp
#include "tinyforge/tinyforge.h"
#include "generated/model_data.h" // from FlexNN's tinyforge-compile

int main() {
  tinyforge::Model model;
  if (model.init() != tinyforge::Status::Ok) return -1;

  int8_t input[TINYFORGE_INPUT_DIM] = {0};   // fill from sensor
  int8_t output[TINYFORGE_OUTPUT_DIM];

  model.run(input, output);
  // output holds logits; argmax for classification
  int best = 0;
  for (int i = 1; i < TINYFORGE_OUTPUT_DIM; ++i)
    if (output[i] > output[best]) best = i;
}
```

**Note:**
- `generated/model_data.h` is produced by **TinyForge's** `tinyforge-compile` (in `tools/compiler`, which fetches FlexNN via `FetchContent` for `ModelIO` format), not by FlexNN. FlexNN only writes `model.bin`.
- Dimensions and quant params come from the header; no filesystem on-device.

## Requirements

- `cmake >=3.10` (3.14 for `FetchContent_MakeAvailable`) and C++17 compiler (`g++`/`clang++` host, `arm-none-eabi-g++` MCU)
- For model generation: FlexNN needs `libeigen3-dev` and `OpenMP`; TinyForge fetches it via `FetchContent` when `TINYFORGE_BUILD_COMPILER=ON` (no manual clone)
- Optional for MCU: `CMSIS-NN`/`CMSIS-DSP`

## Repository Layout

This mirrors FlexNN's layout (`include/` → `lib/` → `src/`):

```
TinyForge/  (CMake library, like FlexNN)
├── CMakeLists.txt              # add_library(TinyForge) + FetchContent for FlexNN
├── include/tinyforge/          # public headers (like FlexNN/include/)
│   ├── tinyforge.h             # → FlexNN.h
│   ├── types.hpp / runtime.hpp / backend.hpp / op_registry.hpp
│   └── kernels/
├── lib/                        # library sources (like FlexNN/lib/)
│   ├── runtime.cpp / backend_scalar.cpp / quant.cpp / activations.cpp
│   └── backend_cmsis.cpp       # only with -DTINYFORGE_USE_CMSIS_NN=ON
├── src/main.cpp                # example firmware (like FlexNN/src/main.cpp)
├── tests/test_runtime.cpp      # host tests
├── tools/compiler/             # tinyforge-compile — parses FlexNN model.bin (via FetchContent)
├── docs/{OVERVIEW.md, LLD_FLEXNN.md, LLD_TINYFORGE.md}
├── generated/                  # model_data.h from tools/compiler (gitignored)
└── cmake/toolchains/           # cross toolchain (replaces platformio.ini)
```

## Building and Running

### Host (scalar, for tests)

```bash
cmake -S . -B build
cmake --build build -j
./build/main
# with tests:
cmake -S . -B build -DTINYFORGE_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
```

### Cross for STM32F407

```bash
cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake
cmake --build build-mcu -j
arm-none-eabi-size build-mcu/libTinyForge.a
```

With CMSIS-NN:

```bash
cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake -DTINYFORGE_USE_CMSIS_NN=ON
cmake --build build-mcu -j
```

### Generate a model

```bash
# Train and export with FlexNN (standalone, no TinyForge needed)
git clone https://github.com/Nalin-Angrish/FlexNN && cmake -S FlexNN -B FlexNN/build && cmake --build FlexNN/build -j
./FlexNN/build/main  # → model.bin (via FlexNN::exportModel)

# Compile with TinyForge (fetches FlexNN for ModelIO format via FetchContent)
cmake -S . -B build && cmake --build build -j
./build/tools/compiler/tinyforge-compile model.bin -o generated --backend cmsis
# → generated/model_data.h
```

## Using as a Library Package

### add_subdirectory (recommended)

```cmake
add_subdirectory(path/to/TinyForge)
target_link_libraries(your_firmware PRIVATE TinyForge::TinyForge)
target_include_directories(your_firmware PRIVATE path/to/TinyForge/generated)
```

### find_package (after install)

```bash
cmake --install build --prefix /opt/tinyforge
find_package(TinyForge REQUIRED)
target_link_libraries(your_firmware PRIVATE TinyForge::TinyForge)
```

## API Reference

- `include/tinyforge/runtime.hpp` — `class Model { Status init(); Status run(int8_t*,int8_t*); }`
- `include/tinyforge/types.hpp` — `enum class Activation/LayerType/Status`, `struct QuantParams`
- `include/tinyforge/backend.hpp` — `TINYFORGE_BACKEND` compile-time switch

See `docs/LLD_FLEXNN.md` (FlexNN training & export) and `docs/LLD_TINYFORGE.md` (TinyForge compilation & kernels) for the full low-level designs.

## License

Apache-2.0, same as FlexNN. See `LICENSE` (when added) and `external/FlexNN/LICENSE`.
