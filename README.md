# TinyForge

Embeddable inference library for Cortex-M4F — run FlexNN models on STM32 with no heap, no interpreter.

TinyForge is the MCU-side companion to [FlexNN](https://github.com/Nalin-Angrish/FlexNN). **FlexNN** trains and compiles (`FlexNN → model.bin → tinyforge-compile → model_data.h`); **TinyForge** runs the header on-device (`Model::run()`).

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
- `generated/model_data.h` is produced by FlexNN, not committed. See `generated/README.md`.
- Dimensions and quant params come from the header; no filesystem on-device.

## Requirements

- `cmake >=3.10` and a C++17 compiler (`g++`/`clang++` for host, `arm-none-eabi-g++` for MCU)
- For model generation: FlexNN (`external/FlexNN`) needs `libeigen3-dev` and `OpenMP` (see `external/FlexNN/README.md`)
- Optional for MCU: `CMSIS-NN`/`CMSIS-DSP` for the accelerated backend

## Repository Layout

This mirrors FlexNN's layout (`include/` → `lib/` → `src/`):

```
TinyForge/
├── CMakeLists.txt              # like FlexNN/CMakeLists.txt
├── include/tinyforge/          # public headers (like FlexNN/include/)
│   ├── tinyforge.h             # → FlexNN.h
│   ├── types.hpp / runtime.hpp / backend.hpp
│   └── kernels/
├── lib/                        # library sources (like FlexNN/lib/)
│   ├── runtime.cpp / backend_scalar.cpp / quant.cpp / activations.cpp
│   └── backend_cmsis.cpp       # only with -DTINYFORGE_USE_CMSIS_NN=ON
├── src/main.cpp                # example firmware (like FlexNN/src/main.cpp)
├── tests/test_runtime.cpp      # host tests
├── docs/{OVERVIEW.md,LLD.md}   # spec + low-level design
├── generated/                  # model_data.h from FlexNN (gitignored)
└── external/FlexNN/            # submodule
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
cmake -S external/FlexNN -B external/FlexNN/build && cmake --build external/FlexNN/build -j
./external/FlexNN/build/main  # trains and calls exportModel("model.bin")
external/FlexNN/build/tools/compiler/tinyforge-compile model.bin -o generated --backend cmsis
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

See `docs/LLD.md` for the full low-level design.

## License

Apache-2.0, same as FlexNN. See `LICENSE` (when added) and `external/FlexNN/LICENSE`.
