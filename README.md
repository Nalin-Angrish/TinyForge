# TinyForge

Purpose-built neural network compiler + inference runtime for STM32F407 (Cortex-M4F). Train on desktop with [FlexNN](https://github.com/Nalin-Angrish/FlexNN) → compile/validate → quantize → deploy → run on-device.

See [`docs/OVERVIEW.md`](docs/OVERVIEW.md) for the full technical specification (draft v0.1).

## System Architecture

```
[FlexNN, PC, Eigen-based]  train model
  |
[Model export]             serialize arch + weights to flat binary
  |
[TinyForge Compiler, PC]   validate ops, quantize int8, emit C header
  |
[TinyForge Runtime, MCU]   scalar reference | CMSIS-NN backend
  |
[STM32F407G-DISC1 firmware]
```

## Repository Layout

```
TinyForge/
├── docs/OVERVIEW.md       # technical spec
├── external/FlexNN/       # git submodule → github.com/Nalin-Angrish/FlexNN (PC-side training lib)
├── include/               # runtime headers (on-device)
├── src/                   # runtime source (on-device)
├── lib/                   # PlatformIO private libs
├── test/                  # PlatformIO unit tests
├── platformio.ini         # PlatformIO project (ststm32, disco_f407vg, stm32cube)
└── tools/compiler/        # (planned) PC-side compiler CLI
```

`external/FlexNN` is a **git submodule**. It is the same upstream FlexNN repo you develop independently, now vendored here for tight integration without copy-paste drift.

## Prerequisites

- `arm-none-eabi-gcc`, PlatformIO (`pio`)
- Eigen3 + OpenMP (for FlexNN / compiler builds on PC)
- `git` with submodule support, `gh` CLI (optional, for repo management)

## Clone

```bash
# fresh clone with submodules
git clone --recurse-submodules https://github.com/Nalin-Angrish/TinyForge.git
cd TinyForge

# or if you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

## Developing FlexNN via the Submodule

This repo tracks FlexNN as a submodule at `external/FlexNN` on branch `main`. You can develop FlexNN *in place* and push upstream without leaving this repo:

```bash
# pull latest upstream FlexNN
git submodule update --remote external/FlexNN

# edit inside the submodule
cd external/FlexNN
# ... make changes, commit, push (you have push rights as Nalin-Angrish) ...
git push origin main
cd ../..

# record the new FlexNN commit in the parent repo
git add external/FlexNN
git commit -m "chore: bump FlexNN to <short-sha>"
git push
```

Conversely, to pull parent-repo changes that bump FlexNN:

```bash
git pull --recurse-submodules
git submodule update --init --recursive
```

### Useful git config for submodule workflow

```bash
git config status.submodulesummary 1
git config push.recurseSubmodules on-demand
git config submodule.recurse true
git config diff.submodule log
```

## Using FlexNN in TinyForge

- **PC side** (compiler / training): add `external/FlexNN` as a CMake subdirectory or include path.
  Example in `tools/compiler/CMakeLists.txt`:
  ```cmake
  add_subdirectory(${CMAKE_SOURCE_DIR}/external/FlexNN flexnn_build)
  target_link_libraries(tinyforge-compiler PRIVATE FlexNN Eigen3::Eigen)
  target_include_directories(tinyforge-compiler PRIVATE ${CMAKE_SOURCE_DIR}/external/FlexNN/include)
  ```
  Or simply: `-I external/FlexNN/include` + link `external/FlexNN/build/libFlexNN.a`.

- **On-device** (`src/`/`include/`): **do not** include FlexNN/Eigen. The runtime is pure C++ with no heap, no STL, no exceptions — it only consumes the compiler's emitted `model_data.h`.

PlatformIO is configured to **ignore** `external/` (PC-only code). See `platformio.ini` `lib_extra_dirs` / build filters if you add PC tools.

## Build (on-device)

```bash
pio run -e disco_f407vg
pio run -e disco_f407vg -t upload   # flash via ST-Link
```

## Build (compiler + FlexNN, PC)

```bash
cmake -S external/FlexNN -B external/FlexNN/build && cmake --build external/FlexNN/build
# ... compiler build TBD (tools/compiler/) ...
```

## Toolchain

- Board: STM32F407G-DISC1 (Cortex-M4F, 1 MB flash, 192 KB RAM)
- Framework: STM32Cube (HAL/LL) via PlatformIO `ststm32`
- Acceleration: CMSIS-NN, CMSIS-DSP
- Baseline: TFLite Micro with CMSIS-NN kernels enabled

## License

TBD — FlexNN is Apache-2.0. This repo will follow the same unless noted otherwise.
