# TinyForge — Low-Level Design (LLD) v0.3

> **Parent spec:** `docs/OVERVIEW.md` (Draft v0.1, now `ref/` local-only). This LLD is the implementation blueprint for the v0.1 spec. It specifies *exact* data structures, file formats, APIs, build integration, and rationale for every decision. Any deviation in code must update this doc.
>
> **Revision 2026-09-05 (v0.2):** **Compiler moved from TinyForge to FlexNN.** All compilation-related work — parsing the flat binary, op-registry validation, arena checks, int8 quantization/calibration, BatchNorm folding, and `model_data.h` codegen — now lives in **`FlexNN/tools/compiler`** (built as part of FlexNN's CMake). TinyForge (this repo) contains only the on-device runtime, platform drivers, and `generated/` header consumed from FlexNN. Section 2, 6, 8, 15, 19 and Appendix D are updated; a migration rationale is added below. v0.1 text that assumed `TinyForge/tools/compiler` is retained as struck-through where useful for history but the new topology is authoritative.

> **Revision 2026-09-05 (v0.3):** **TinyForge is now a CMake library, not a PlatformIO project.** File layout now mirrors FlexNN (`include/tinyforge/`, `lib/`, `src/main.cpp`, `tests/`, `docs/`). `platformio.ini` and PlatformIO `test/`/`lib/` scaffolding removed. Build is `cmake -S . -B build && cmake --build build` (host) or `cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake` (cross). See §2, §19. All compilation-related work — parsing the flat binary, op-registry validation, arena checks, int8 quantization/calibration, BatchNorm folding, and `model_data.h` codegen — now lives in **`FlexNN/tools/compiler`** (built as part of FlexNN's CMake). TinyForge (this repo) contains only the on-device runtime, platform drivers, and `generated/` header consumed from FlexNN. Section 2, 6, 8, 15, 19 and Appendix D are updated; a migration rationale is added below. v0.1 text that assumed `TinyForge/tools/compiler` is retained as struck-through where useful for history but the new topology is authoritative.

**Status:** Draft · **Target:** STM32F407VGT6 (Cortex-M4F, 1 MB Flash, 192 KB RAM: 112 KB SRAM + 64 KB CCM + 16 KB backup), ST-Link V2-A, LIS3DSH/LIS302DL accel, MP45DT02 mic · **Toolchain:** `arm-none-eabi-gcc` via `cmake/toolchains/arm-cortex-m4f.cmake`, CMake 3.10+, CMSIS-NN 6.x, CMSIS-DSP, Eigen3 3.4+, OpenMP (host) · **Repos:** `Nalin-Angrish/FlexNN` (public, PC host, owns compiler) + `Nalin-Angrish/TinyForge` (private, MCU runtime) with submodule `external/FlexNN @ main`

---

## Table of Contents

1. [Design Principles & Invariants](#1-design-principles--invariants)
2. [Repository & Build Topology](#2-repository--build-topology)
3. [FlexNN — Current Implementation Audit](#3-flexnn--current-implementation-audit)
4. [FlexNN — Refactors & Extensions (PC, now including Compiler)](#4-flexnn--refactors--extensions-pc-now-including-compiler)
5. [Model Export Format (Flat Binary)](#5-model-export-format-flat-binary)
6. [FlexNN Compiler (PC CLI, lives in FlexNN)](#6-flexnn-compiler-pc-cli-lives-in-flexnn)
7. [Quantization Design (int8 per-tensor)](#7-quantization-design-int8-per-tensor)
8. [Op-Support Registry (Single Source of Truth)](#8-op-support-registry-single-source-of-truth)
9. [TinyForge Runtime — Constraints & Memory Model](#9-tinyforge-runtime--constraints--memory-model)
10. [Runtime Backend Abstraction](#10-runtime-backend-abstraction)
11. [Runtime Kernels — Scalar & CMSIS-NN](#11-runtime-kernels--scalar--cmsis-nn)
12. [Activation Handling on Cortex-M4F](#12-activation-handling-on-cortex-m4f)
13. [Runtime Public API & Error Model](#13-runtime-public-api--error-model)
14. [Platform Layer (HAL/LL, Clock, DMA, Sensors, DSP)](#14-platform-layer-halll-clock-dma-sensors-dsp)
15. [End-to-End Pipeline & CLI Examples](#15-end-to-end-pipeline--cli-examples)
16. [Target Applications — Wiring to Runtime](#16-target-applications--wiring-to-runtime)
17. [Benchmarking Harness (vs TFLite Micro)](#17-benchmarking-harness-vs-tflite-micro)
18. [Testing Strategy](#18-testing-strategy)
19. [File Tree (Planned) & Generated Artifacts](#19-file-tree-planned--generated-artifacts)
20. [Implementation Order & Bottom-Up Gating](#20-implementation-order--bottom-up-gating)
21. [Risks & Mitigations](#21-risks--mitigations)
22. [Appendix A: Binary Format Reference](#appendix-a-binary-format-reference)
23. [Appendix B: Op Registry Example](#appendix-b-op-registry-example)
24. [Appendix C: Generated Header Example](#appendix-c-generated-header-example)
25. [Appendix D: Decisions Log (extends §9 of OVERVIEW)](#appendix-d-decisions-log-extends-9-of-overview)

---

## 1. Design Principles & Invariants

1. **FlexNN-only ingestion — by design, not debt.** Generic ONNX/PyTorch import is an open-ended compiler problem (arbitrary graph, control flow, dynamic shapes). FlexNN-only bounds the compiler to a *linear stack* of layers, fully testable in a semester. *Why:* preserves the "every line you can explain" narrative and keeps the validation problem decidable. Rejected alternative: ONNX Runtime conversion — 100+ ops, multi-year effort.
2. **PC-side compilation lives with FlexNN, runtime lives with TinyForge.** Training (`FlexNN.h`/`Layer.h`), serialization (`ModelIO`), and compilation (parse → validate → quantize → codegen) are all *host* concerns that share the same `LayerType`/`Activation` enums, `Eigen` types, and export format. Co-locating them in `FlexNN` eliminates drift: `FlexNN::exportModel()` can literally `#include "tinyforge/op_registry.hpp"` and refuse to emit an unsupported op before a `model.bin` ever touches a USB stick. TinyForge stays minimal — no binary parser, no Eigen, no `lib/` for the compiler. *Why (v0.2 change):* In v0.1 the compiler was planned as `TinyForge/tools/compiler` with a separate CMake that had to re-define `LayerType`/`Activation` and parse the binary independently. That split forced two copies of the registry and two places to bump `LayerType` when adding `Conv1D`. Moving the compiler into `FlexNN/tools/compiler` makes the registry a single header in one repo, lets `isSupportedForExport()` and `tinyforge-compile` share the same `kRegistry`, and lets FlexNN's CI gate "every exported model is immediately compilable" without needing TinyForge. TinyForge's `external/FlexNN` submodule then provides the *authoritative* registry to the runtime at the pinned SHA. Rejected alternative: keep compiler in TinyForge — would require either duplicating enums or making TinyForge depend on FlexNN's internal headers via a relative `../external` path that breaks standalone FlexNN builds.
3. **Bottom-up layer unlocking.** A `(layer_type, activation)` is only added to FlexNN + compiler allow-list *after* both runtime backends (scalar + CMSIS-NN) have a tested kernel. Invariant: `FlexNN_exportable ⊆ Runtime_executable`. *Why:* prevents training a model that cannot be deployed; fails at training time, not at flash time. Enforcement now is trivial because `FlexNN/tools/compiler` and `FlexNN/include/Layer.h` live in the same repo and the same PR can add the kernel stub + registry entry + FlexNN enum together.
4. **Compile-time, not run-time, polymorphism.** Backend (`scalar` vs `cmsis-nn`) and model architecture are baked at compile time (`-DTINYFORGE_BACKEND=...` + generated `model_data.h`). No `malloc`, no `virtual` in hot path, no interpreter loop. *Why:* on 168 MHz M4F, every branch in the inner MAC loop costs cycles; deterministic footprint enables static arena sizing and honest TFLM comparison.
5. **Fail loudly on the host, never silently on device.** The compiler is a *verifier*: unknown op, OOM arena, mismatched dims → non-zero exit with `layer_index + reason + fix hint`. The runtime never does filesystem I/O or dynamic allocation; it `static_assert`s its arena at build time. *Why:* debugging on SWD UART is expensive; catching errors on the PC saves days. With the compiler in FlexNN, the failure happens even earlier: `FlexNN::exportModel()` can call `validateForTinyForge()` and return `Status::ErrUnsupportedOp` before writing `model.bin`.
6. **No heap, no exceptions, no RTTI, no STL heap containers in runtime.** Even `std::vector` is banned in `src/`/`include/` runtime code. Host tools (FlexNN library + FlexNN's compiler) may use STL/Eigen freely. *Why:* heap fragmentation is untestable on 192 KB RAM; exceptions/RTTI bloat `.text` and are non-deterministic.
7. **Two backends, one golden reference.** Scalar C++ is the correctness oracle; CMSIS-NN is the performance variant. They must agree within int8 rounding (`±1 LSB` after requant). *Why:* before benchmarking, we must prove functional equivalence.
8. **Version everything.** Model binary has `magic + version`, header has `TINYFORGE_MODEL_VERSION`, op registry has its own version. *Why:* allows forward-compatible compiler and clear error when a stale firmware (old submodule SHA) meets a new model (new FlexNN). With the compiler in FlexNN, the registry version is the FlexNN version.

---

## 2. Repository & Build Topology

### 2.1 Two repos, one submodule

```
FlexNN/  (github.com/Nalin-Angrish/FlexNN, public, PC host) — NOW OWNS COMPILER
├── include/
│   ├── FlexNN.h, Layer.h, Utility.h          # existing
│   ├── LayerTypes.hpp                        # NEW: LayerType enum + param structs
│   ├── ModelIO.hpp                           # NEW: exportModel/importModel
│   └── tinyforge/                            # NEW: shared with TinyForge runtime
│       ├── op_registry.hpp                   # SINGLE SOURCE OF TRUTH (see §8)
│       ├── model.hpp                         # compiler's in-memory Model
│       ├── quant.hpp                         # quant math
│       └── codegen.hpp
├── lib/
│   ├── FlexNN.cpp, Layer.cpp, Utility.cpp
│   └── ModelIO.cpp                           # NEW: flat binary writer
├── tools/
│   └── compiler/                             # MOVED HERE from TinyForge/tools/compiler
│       ├── CMakeLists.txt                    # add_executable(tinyforge-compile)
│       ├── src/{main.cpp, parser.cpp, validate.cpp, quantize.cpp, codegen.cpp}
│       └── tests/                            # compiler unit tests (GoogleTest)
├── tests/                                    # FlexNN unit tests
├── CMakeLists.txt                            # add_library(FlexNN ...) + add_subdirectory(tools/compiler)
└── ... (Doxyfile, data, etc.)

TinyForge/  (github.com/Nalin-Angrish/TinyForge, private, CMake library) — THIS REPO
├── CMakeLists.txt                            # like FlexNN/CMakeLists.txt — add_library(TinyForge)
├── include/tinyforge/                        # public headers (like FlexNN/include/)
│   ├── tinyforge.h                           # aggregator
│   ├── types.hpp / runtime.hpp / backend.hpp / quant.hpp
│   └── kernels/{dense,conv1d,activations}.hpp
├── lib/                                      # library sources (like FlexNN/lib/)
│   ├── runtime.cpp / backend_scalar.cpp / quant.cpp / activations.cpp
│   └── backend_cmsis.cpp                     # only with -DTINYFORGE_USE_CMSIS_NN=ON
├── src/main.cpp                              # example firmware (like FlexNN/src/main.cpp)
├── tests/test_runtime.cpp                    # host tests
├── docs/{OVERVIEW.md, LLD.md (this)}         # committed spec (ref/ is local ignored copy)
├── generated/                                # OUTPUT of FlexNN's compiler (gitignored)
│   └── model_data.h                          # from FlexNN/tools/compiler/tinyforge-compile
├── cmake/toolchains/arm-cortex-m4f.cmake     # cross toolchain (like -mcpu=cortex-m4)
└── external/FlexNN/                          # submodule @ main — provides tinyforge-compile + op_registry.hpp
```

**Old v0.1 topology (for history):**

```
# v0.1 had:
TinyForge/tools/compiler/  ← compiler lived here, with its own CMake and duplicated enums
FlexNN/                    ← only libFlexNN, no compiler
# Problem: two copies of LayerType/Activation, two registries, drift risk.
```

**Why the move (detailed):**

- **Single definition of `LayerType`/`Activation`:** FlexNN's `Layer.h` defines `enum class Activation` and `enum class LayerType` (see §4.1-4.2). The compiler's parser and validator must switch on those exact enums. If the compiler lived in TinyForge, it would either `#include "../external/FlexNN/include/Layer.h"` (a fragile relative path that breaks when FlexNN is built standalone) or duplicate the enums (drift). In `FlexNN/tools/compiler` the compiler just `#include "LayerTypes.hpp"` and `#include "tinyforge/op_registry.hpp"` with a normal include path.
- **Export-time validation:** `FlexNN::exportModel()` can now call `tinyforge::isSupported(type, act, quant)` *before* writing `model.bin` and return `Status{false, "layer 2 Conv1D+Tanh not yet in TinyForge runtime"}`. In v0.1 the export would succeed and the later `TinyForge/tools/compiler --check-only` would fail — a worse UX (write then fail).
- **One CMake, one CI:** `FlexNN/CMakeLists.txt` does `add_library(FlexNN ...)` and `add_subdirectory(tools/compiler)` which does `target_link_libraries(tinyforge-compile PRIVATE FlexNN Eigen3::Eigen)`. FlexNN's own CI can now `ctest` both the training tests and the compiler's parser/quant tests in one build, and `tinyforge-compile --check-only` can be a `add_test()` that runs on every example `model.bin`.
- **TinyForge stays minimal and is now a plain CMake library (no PlatformIO):** `TinyForge` builds as `add_library(TinyForge)` from `lib/` with headers in `include/tinyforge/` — exactly like `FlexNN` (`add_library(FlexNN)` from `lib/`). The `external/FlexNN` submodule is *not* built as part of TinyForge's library; it is only used to produce `generated/model_data.h` via its `tinyforge-compile` binary. To keep host builds native, TinyForge does not hardcode MCU flags; a toolchain file (`cmake/toolchains/arm-cortex-m4f.cmake`) supplies `-mcpu=cortex-m4 -mfpu=...` for cross builds.
- **Submodule pinning gives reproducibility:** TinyForge pins a FlexNN SHA. That SHA pins both a `libFlexNN` version and a compiler version and a registry version. When TinyForge bumps the submodule, it automatically bumps the runtime's `op_registry.hpp` (because `runtime/include/tinyforge/runtime.hpp` does `#include "external/FlexNN/include/tinyforge/op_registry.hpp"`). No manual copy.
- *Rejected:* keep compiler in TinyForge but make it `#include "external/FlexNN/..."` — works for TinyForge builds but breaks `FlexNN` standalone builds (no TinyForge checkout), and FlexNN's own `exportModel` cannot validate without the compiler.

### 2.2 Build systems, clearly separated and now both CMake

- **FlexNN host build (CMake, PC):**
  ```bash
  # Inside FlexNN repo (or TinyForge/external/FlexNN)
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  # produces:
  #   build/libFlexNN.a
  #   build/tools/compiler/tinyforge-compile
  #   build/main (example)
  ctest --test-dir build
  ```
  Flags: `-O3 -march=native -fopenmp -std=c++17` (host). No MCU flags. Tests: `tests/` (FlexNN) + `tools/compiler/tests/`.

- **TinyForge host build (CMake, scalar, like FlexNN):**
  ```bash
  # Inside TinyForge repo — no FlexNN needed to build the library itself
  cmake -S . -B build
  cmake --build build -j
  # produces:
  #   build/libTinyForge.a
  #   build/main  # from src/main.cpp
  ctest --test-dir build  # runs tests/test_runtime.cpp
  ```

- **TinyForge cross build (CMake + toolchain, like FlexNN but for MCU):**
  ```bash
  # First, ensure FlexNN's compiler is built (it generates the model header):
  cmake -S external/FlexNN -B external/FlexNN/build && cmake --build external/FlexNN/build -j
  external/FlexNN/build/tools/compiler/tinyforge-compile model.bin --backend cmsis --calib data/calib.csv --arena 98304 -o generated
  # → writes generated/model_data.h

  # Then cross-compile TinyForge as a library for STM32F407:
  cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake
  cmake --build build-mcu -j
  # or with CMSIS-NN:
  cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake -DTINYFORGE_USE_CMSIS_NN=ON
  cmake --build build-mcu -j
  ```
  Toolchain `cmake/toolchains/arm-cortex-m4f.cmake` supplies `-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -Wl,--gc-sections`. No `platformio.ini`; no `build_src_filter` needed — `external/` is never part of TinyForge's library sources.

**TinyForge's `CMakeLists.txt` (now, mirrors FlexNN — `add_library` + `add_executable`):**

```cmake
cmake_minimum_required(VERSION 3.10)
project(TinyForge)
set(CMAKE_CXX_STANDARD 17)
include_directories(include)
set(LIB_SOURCES lib/runtime.cpp lib/backend_scalar.cpp lib/quant.cpp lib/activations.cpp)
add_library(TinyForge ${LIB_SOURCES})
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE TinyForge::TinyForge)
# Tests: add_subdirectory(tests) when -DTINYFORGE_BUILD_TESTS=ON
```

*Why compile-time backend, not runtime switch:* same as before — two library variants (`scalar` vs `cmsis-nn`) are built by choosing the toolchain/CMake option, not by an `if` in the hot loop. Each firmware chooses one backend at CMake configure time.

---

## 3. FlexNN — Current Implementation Audit

Reading `external/FlexNN/include/Layer.h:37-151`, `lib/Layer.cpp:25-82`, `include/FlexNN.h:37-128`, `lib/FlexNN.cpp:29-142`:

**Current architecture:**

- `Layer` is a *monomorphic* dense-only class: `W: MatrixXd (out×in)`, `b: VectorXd (out)`, `activationFunction: std::string`. No layer-type tag; every layer is implicitly `Dense`.
- `NeuralNetwork` holds `vector<Layer> layers`. `forward()` returns `vector<MatrixXd>` interleaving `Z` and `A` (`Z0,A0,Z1,A1...`). `backward()` and `updateWeights()` assume this layout and dense gradients.
- Training: `train()` one-hot encodes `Y` via `Utility::oneHotEncode`, loops `forward→backward→update` with fixed LR, no batches, no optimizer choice, no shuffling per-epoch. `accuracy()` argmaxes cols.
- Activations: only `"relu"` and `"softmax"` (stringly-typed). No `tanh/sigmoid/leaky`. Softmax is column-wise stable (subtract max, exp, normalize) — correct forward. Backward for softmax at `Layer.cpp:72-75` is **incorrect**: `dZ = (nextW^T * nextdZ) * (expZ/expZ.sum())` — element-wise multiply by softmax output is not the Jacobian (`diag(s) - s s^T`). Works only by accident when last layer + cross-entropy, but will break for hidden softmax or when we add `tanh`. Needs fixing or restricting softmax to last layer + fusing with loss.
- `Utility` provides CSV reading and `splitXY` with shuffle; no model I/O.

**Pain points that force the LLD:**

1. **String activations** (`Layer.h:137`): typo `“ReLU” vs “relu”` compiles; `if (activationFunction=="relu")` linear search + `std::string` heap alloc in every forward on MCU — unacceptable. *Fix:* `enum class Activation`.
2. **No layer polymorphism:** cannot add `Conv1D` without `if (layerType==...)` spaghetti. *Fix:* `enum class LayerType` + tagged struct or `std::variant<DenseParams, Conv1DParams...>` (host only; MCU will use generated static structs).
3. **No serialization:** "no saving model" is a known limitation. *Fix:* `exportModel(path)` + `importModel(path)` (host only, now in `FlexNN/lib/ModelIO.cpp` and shares registry with compiler).
4. **Backward incorrect for softmax:** will be fixed as part of activation refactor; we will also document that softmax is only valid on final layer with cross-entropy and that int8 runtime will use **argmax without explicit softmax** (softmax monotonic, so argmax preserves class; saves `expf` on MCU).
5. **No batching, no validation split, no learning-rate schedule:** out of scope for this LLD (training quality is not the thesis), but we will not make it worse; keep current `train()` signature and overload for export.
6. **Weight init `Random *0.5` (`Layer.h:55`):** coarse; keep for v0.1 to avoid churn, but note that int8 quantization calibration will expose init sensitivity (documented).

---

## 4. FlexNN — Refactors & Extensions (PC, now including Compiler)

> **Gating rule (from OVERVIEW §5.1):** For each new `(LayerType, Activation)` we implement in FlexNN, the scalar kernel + CMSIS-NN kernel + compiler registry entry must exist *first*. FlexNN's `exportModel()` must refuse to export a model containing an unsupported combo (checked against the same registry header the compiler uses — now trivial because both live in `FlexNN`).

### 4.1 Activation Refactor — `enum class Activation`

**File:** `FlexNN/include/Layer.h` (breaking change, major version bump; now also included by `FlexNN/tools/compiler`)

```cpp
namespace FlexNN {
enum class Activation : uint8_t {
  None = 0,      // linear (no activation) — used for hidden layers that are pure matmul before BN fusion
  ReLU = 1,
  LeakyReLU = 2, // alpha = 0.01 fixed for v0.1 (no per-layer alpha to keep export simple)
  Sigmoid = 3,
  Tanh = 4,
  Softmax = 5    // only valid on last layer; int8 runtime will optionally skip
};
inline const char* to_string(Activation a) { /* switch → "relu", ... */ }
inline bool try_parse_activation(std::string_view s, Activation& out);
}
```

**Why enum:**

- Compile-time exhaustiveness: `switch(a) { case ... }` with `-Wswitch-enum` catches missing cases.
- No heap, no string compare in hot path (host forward still uses Eigen but now switches on uint8).
- Export binary encodes `uint8_t activation` (see §5), not variable-length string.
- **New benefit (v0.2):** `FlexNN/tools/compiler` can `switch` on the same enum without duplicating strings, and `isSupportedForExport()` and the compiler's `validate()` share the same `case` list.

**Migration:**

- Keep backward-compat ctor `Layer(int,int,const std::string&)` that parses and delegates to `Layer(int,int,Activation)`, but mark `[[deprecated]]`. Remove in v0.2.
- `Layer.h:51` new ctor: `Layer(int in, int out, Activation act = Activation::ReLU, LayerType type = LayerType::Dense)`.
- Update `lib/Layer.cpp:25` forward/backward to `switch(activation_)` (see §4.3 for formulas).

### 4.2 Layer Type System (Host Only)

**New header:** `FlexNN/include/LayerTypes.hpp` (also included by `FlexNN/tools/compiler` and, via submodule, by TinyForge runtime for the registry)

```cpp
enum class LayerType : uint8_t {
  Dense = 0,
  Conv1D = 1,
  BatchNorm1D = 2,   // or fused; kept explicit for training, folded at compile time
  MaxPool1D = 3,
  AvgPool1D = 4
};

struct DenseParams {
  int inputSize;   // in_features
  int outputSize;  // out_features
  // W: [outputSize × inputSize], b: [outputSize]
};

struct Conv1DParams {
  int inChannels;    // C_in
  int outChannels;   // C_out
  int kernelSize;    // K
  int stride;        // default 1
  int padding;       // default 0 (valid), symmetric
  int dilation;      // default 1 (v0.1 only 1)
  // W: [outChannels × inChannels × kernelSize] row-major flattened as [outChannels][inChannels*K]
  // b: [outChannels]
  // input shape: [inChannels × seqLen], output: [outChannels × outLen] where outLen = floor((seqLen + 2*pad - dilation*(K-1) -1)/stride +1)
};

struct Pool1DParams {
  int channels;      // typically equals inChannels == outChannels
  int kernelSize;
  int stride;
  int padding;
  // no weights
};

struct BatchNormParams {
  int numFeatures;
  float epsilon = 1e-5f;
  // per-feature: gamma, beta, runningMean, runningVar (training) / folded into prev layer (inference)
};
```

**Why structs, not inheritance:**

- Host training code stays simple, Eigen-friendly, no vtable. For MCU we don't reuse these structs at all — the FlexNN compiler emits flat C arrays (see §6.6) that TinyForge's runtime consumes.
- `Layer` becomes a tagged variant for v0.1:
  ```cpp
  class Layer {
    LayerType type_;
    Activation activation_;
    // common dims for quick validation
    int inputSize_, outputSize_; // for Dense; for Conv1D, logical flattened dims
    // per-type payload (only one active):
    DenseParams dense_;
    Conv1DParams conv1d_;
    // ... plus W,b as Eigen matrices sized per type
  };
  ```
  *Rejected:* full `class Conv1DLayer : public Layer` hierarchy — would require virtual `forward()` and complicate serialization. Variant is enough for a linear stack.

**Public API additions on `Layer`:**

```cpp
LayerType type() const;
Activation activation() const;
bool isSupportedForExport() const; // checks registry (see §8) — now calls tinyforge::find() directly
std::pair<Eigen::MatrixXd,Eigen::MatrixXd> forward(const Eigen::MatrixXd& input) const; // now dispatches on type_
```

For v0.1, `forward()` will support `Dense` + activations (all) and `Conv1D` + `ReLU/None` (Tanh/Sigmoid for Conv deferred). Pool/BN are phase 2.

### 4.3 New Activations — Formulas & Derivatives

| Activation | Forward `A = f(Z)` | Backward `dZ = dA ⊙ f'(Z)` | Notes |
|---|---|---|---|
| `None` | `A=Z` | `dZ=dA` | linear, used pre-BN |
| `ReLU` | `max(0,Z)` | `dZ = dA if Z>0 else 0` | current |
| `LeakyReLU` | `Z if Z>0 else 0.01*Z` | `1 if Z>0 else 0.01` | α=0.01 fixed, avoids dead ReLU; cheap on MCU (`x>0?x:0.01*x`) |
| `Sigmoid` | `1/(1+exp(-Z))` | `dA * A*(1-A)` (cache A) | Numerically: clamp Z to [-15,15] before exp to avoid overflow; host double, MCU uses LUT (see §12) |
| `Tanh` | `tanh(Z) = (exp(2Z)-1)/(exp(2Z)+1)` | `dA * (1 - A^2)` | Host uses `std::tanh`; MCU uses `arm` approx or LUT; range [-1,1] |
| `Softmax` | `exp(Z - max(Z)) / sum(exp(Z-max(Z)))` col-wise | `dZ = ...` (Jacobian) if hidden; if last+cross-entropy, `dZ = A - Y_onehot` (fused) | **Restrict:** only final layer, loss must be cross-entropy; document that runtime int8 skips softmax and does argmax on logits |

*Why fixed α, why restrict softmax:* keeps export format minimal (no per-layer float for Leaky α in v0.1) and avoids implementing a general softmax Jacobian on MCU (which would be O(C²) and unnecessary when we can argmax logits).

**Backward correctness fix:**

- Current `Layer.cpp:72-75` does `dZ = (nextW^T * nextdZ) * (expZ/expZ.sum())` — wrong. New code will, for hidden softmax (if ever allowed), compute `J = diag(s) - s s^T` per column; for final softmax+CE, the `NeuralNetwork::backward()` will fuse: `dZ_last = A_last - Y_onehot` and not call `Layer::backward` for that layer. We will add `assert(activation != Softmax || isLastLayer)` and FlexNN's compiler will reject non-last softmax.

### 4.4 New Layer Types — Spec & Priority

**Priority order (§5.1 of spec):** `Conv1D` → `BatchNorm` (folded) → `MaxPool1D`/`AvgPool1D`. *Why:* vibration/gesture (7.1,7.2) benefit most from temporal convolution; BN improves training stability but is free at inference via folding; pooling reduces seq length and compute.

#### 4.4.1 `Dense` (existing, baseline)

- **Host forward:** `Z = W * X + b` (W:`out×in`, X:`in×batch`, b broadcast col-wise). Activation applied per §4.3.
- **Host backward:** `dW = dZ * X^T / batch`, `db = rowwiseMean(dZ)`, `dX_prev = W^T * dZ`.
- **Export:** `W` row-major `float32`, `b` `float32` (via `FlexNN/lib/ModelIO.cpp`).
- **Runtime (TinyForge):** scalar loop + `arm_fully_connected_s8` (see §11).

#### 4.4.2 `Conv1D` (new, highest priority)

- **Semantics:** 1D temporal convolution over sensor time axis. Input `[C_in × L_in]` per sample (batch dim separate, Eigen `MatrixXd` with `in = C_in * L_in` flattened). Output `[C_out × L_out]`.
- **Params:** `C_in, C_out, K, stride, padding, dilation=1`. `W: [C_out][C_in*K]` (each out-channel has `C_in*K` weights), `b: [C_out]`.
- **Forward (host, FlexNN):** For each sample `n`, for each `oc`, `ol` (0..L_out-1):
  ```
  Z[oc][ol] = b[oc] + Σ_{ic=0..C_in-1} Σ_{k=0..K-1} W[oc][ic*K + k] * X_padded[ic][ol*stride - padding + k*dilation]
  ```
  where `X_padded` is zero-padded. Activation applied per-element.
- **Backward (FlexNN):** im2col + GEMM style; leverage Eigen: unfold `X` to `col` matrix `[C_in*K × L_out]` then `dW = dZ * col^T`, `dX` via col2im. Keep for host only; no MCU backward.
- **Why Conv1D before Conv2D:** sensor data is 1D time series; Conv1D maps to `arm_convolve_1d_s8` / `arm_convolve_s8` with `ch_im_in=C_in`, `x/y` dims set to `(L_in,1)`. Simpler than 2D and fits 192 KB RAM.
- **Edge:** `L_in` up to 256 (accelerometer window 1 sec @ 100 Hz ×3 axes = 300 samples), `C_in` up to 8, `C_out` up to 16, `K=3..7`, `stride=1..2`. FlexNN's compiler validates `L_out` against TinyForge arena (see §9).

#### 4.4.3 `BatchNorm1D` (training-time; folded at compile time)

- **Host training (FlexNN):** per-feature (per-channel for Conv, per-neuron for Dense) with `gamma, beta, eps`, running `mean/var` (momentum 0.1). Forward: `y = gamma * (x - mean)/sqrt(var+eps) + beta` during training (batch stats) vs `runningMean/Var` at eval. Keep `mean/var` as buffers.
- **Inference folding (FlexNN compiler):** never shipped as a separate layer. Compiler fuses into preceding `Dense` or `Conv1D` (see §7.5). *Why:* zero runtime cost, avoids needing a BN kernel on MCU. If BN follows a layer without preceding linear (edge), compiler rejects (BN must be foldable).
- **Export:** we export BN as its own layer type but with flag `foldable=true`; FlexNN's compiler either folds or errors.

#### 4.4.4 `MaxPool1D` / `AvgPool1D` (new, phase 2)

- **Params:** `kernel, stride, padding` (no weights).
- **Forward:** `Y[oc][ol] = max/mean_{k} X[oc][ol*stride - pad + k]`.
- **Why after Conv/BN:** pooling is needed to reduce sequence length after Conv (e.g., 128 → 64), but can be simulated by stride-2 Conv in v0.1 if pooling not ready; so pooling is lower priority.

### 4.5 Model I/O — `exportModel` / `importModel` (FlexNN)

**New file:** `FlexNN/lib/ModelIO.cpp`, header `FlexNN/include/ModelIO.hpp` (lives with FlexNN, shares `op_registry.hpp`)

```cpp
namespace FlexNN {
struct ExportOptions {
  int formatVersion = 1;
  bool includeOptimizerState = false; // v0.1 false
};

Status exportModel(const NeuralNetwork& net, const std::string& path, ExportOptions opts = {});
Status importModel(NeuralNetwork& net, const std::string& path); // for testing round-trip
// Status is {bool ok; std::string error;}
}
```

**Behavior (now co-located with compiler):**

1. Validate via `isSupportedForExport()` which directly calls `tinyforge::find(type, act, quant, backend)` from `FlexNN/include/tinyforge/op_registry.hpp` (the same header the compiler uses) before writing; on failure, return error naming the offending layer index + `(type, activation)`. *New in v0.2:* this check is no longer a cross-repo call; it's a direct header include.
2. Open `std::ofstream(path, std::ios::binary)`, write header + layers per §5, `fsync`, close.
3. `importModel` is only for host tests (read back and `EXPECT_EQ` weights); not used on MCU. FlexNN's compiler's `parser.cpp` reuses the same `ModelIO` structs to read (no duplicated parsing logic).

**Why separate `ModelIO.cpp`:** keeps `FlexNN.h`/`Layer.h` focused; serialization is not a `Layer` responsibility. With the compiler in FlexNN, `ModelIO.cpp` and `tools/compiler/src/parser.cpp` can share the `LayerHeader` struct and CRC code.

**Why binary, not JSON:** deterministic, no parser bloat (compiler's parser is 50 LOC and shares code with ModelIO), 2–5× smaller than JSON, no float-to-string rounding issues. *Rejected:* direct C header generation from FlexNN — would couple FlexNN to TinyForge's codegen expectations; flat binary keeps FlexNN minimal (just serialize) and lets the compiler (still in FlexNN) own all validation/quant/codegen.

### 4.6 Utility Keep

- Keep `Utility::readCSV_XY` and `splitXY` as is. Add `FlexNN::setRandomSeed(uint32_t)` for reproducibility in tests (currently `Random` uses nondeterministic seed).

### 4.7 Testing FlexNN Changes (including Compiler)

- **Host unit tests** (new, GoogleTest in `FlexNN/tests/` and `FlexNN/tools/compiler/tests/`):
  - `activation_test`: ReLU/Leaky/Sigmoid/Tanh/Softmax forward+backward vs finite differences (`eps=1e-5`).
  - `dense_test`: forward/backward vs naive loops.
  - `conv1d_test`: small cases `C_in=1..2, L=4..8, K=3` brute-forced.
  - `export_import_test`: train tiny 2-layer net, export, import, `EXPECT_NEAR` weights, then run `FlexNN/build/tools/compiler/tinyforge-compile` on exported file and check header (now a FlexNN `ctest`).
  - **Compiler tests (now in FlexNN):** `parser_test` (bad magic/CRC/version), `validate_test` (unsupported op → `EXPECT_EXIT 2`), `quant_test` (golden), `codegen_test` (compile `model_data.h` with `arm-none-eabi-gcc`).
- **Bottom-up gate:** no FlexNN PR merges a new `(type,act)` unless `TinyForge/runtime/tests` scalar vs cmsis test for that kernel passes (see §18). Since the compiler is now in FlexNN, the FlexNN PR that adds `Conv1D+Tanh` must also add the TinyForge kernel test as a submodule bump — the PR description will note the TinyForge runtime test link.

---

## 5. Model Export Format (Flat Binary)

**Decision:** custom flat binary, little-endian, versioned, with CRC. Owned by `FlexNN` (`ModelIO.cpp` writer + `tools/compiler/src/parser.cpp` reader share the same header). TinyForge runtime never parses this binary — it only consumes the generated `model_data.h`. *Why:* see §4.5; also allows `mmap`-free streaming parser in FlexNN's compiler (≈50 LOC) and trivial `hexdump` debugging.

**File layout (all ints little-endian):**

```
Offset  Size  Field
0       4     magic = 0x54464E47 ('T' 'F' 'N' 'G' = TinyForge Neural Graph) — chosen distinct from TFLite's 'TFL3'
4       2     version = 0x0001 (uint16) — bump on breaking change
6       2     header_len = 10 (uint16) — bytes from offset 8 to end of header (allows future header fields without breaking parser)
8       4     layer_count = N (uint32)
12      4     flags = 0 (uint32) — bit0: has_quant_params (v0.1 always 0; compiler adds), bit1-31 reserved
16      4     header_crc32 = CRC32 of bytes [0..15] (IEEE, init 0) — ensures header not corrupted

# Then N layer records, back-to-back, no padding between layers:
Per layer:
  0     1     layer_type (uint8) — LayerType enum (FlexNN/include/LayerTypes.hpp)
  1     1     activation (uint8) — Activation enum
  2     2     dtype = 0x0000 = float32 (uint16) — only float32 for v0.1; int8 only appears after FlexNN compiler quant
  4     4     input_dim (uint32) — for Dense: in_features; for Conv1D: C_in * L_in (flattened) or logical dims (see below)
  8     4     output_dim (uint32) — analogous
 12     4     weight_count (uint32) — number of float32 weights (e.g., Dense: out*in, Conv1D: outCh * inCh * K)
 16     4     bias_count (uint32) — number of float32 biases (typically output_dim or C_out; 0 if no bias)
 20     4     aux_count (uint32) — number of float32 aux params (e.g., BN gamma/beta/mean/var = 4*features, Pool = 0)
 24     4     reserved (uint32) — 0
 28     4     weight_offset (uint32) — absolute offset from file start to weights blob (allows future non-contiguous layout)
 32     4     bias_offset (uint32)
 36     4     aux_offset (uint32) — 0 if aux_count==0
 40     ?     — padding to 8-byte align next blob (FlexNN compiler can ignore, but writer aligns)

# Blobs at declared offsets, each blob is:
  weight_count * 4 bytes of float32 little-endian, row-major as described in §4.4
  bias_count * 4 bytes
  aux_count * 4 bytes (if any) — for BN: [gamma, beta, mean, var] concatenated

# End of file:
  4     file_crc32 = CRC32 of entire file except this field (useful for flash integrity via UART)
```

**For Conv1D, logical dims encoding:**

- `input_dim = C_in * L_in` and `output_dim = C_out * L_out` are *flattened* sizes. To recover `C* L`, the layer's payload includes `aux` for Conv? Instead we encode Conv geometry as extra header fields to avoid needing aux. Simpler: extend per-layer header for Conv to include `c_in, c_out, k, stride, pad, dilation` as 6× `uint16` after `reserved`. But to keep parser simple, v0.1 will *always* emit them as 12 bytes after offset 40, even for Dense (zeroed). So per-layer fixed header size = 52 bytes, always 8-byte aligned.

```
Per-layer extended (v0.1):
 40    2  conv_c_in
 42    2  conv_c_out
 44    2  conv_k
 46    2  conv_stride
 48    2  conv_pad
 50    2  conv_dilation
```

*Why fixed size:* parser (`FlexNN/tools/compiler/src/parser.cpp`) can `struct __attribute__((packed)) LayerHeader { ... }` and `static_assert(sizeof==52)`, no variable-length parsing. Dense layers just zero those fields. Future pool/BN use same slots with different meaning (documented).

**Versioning policy:**

- Bump `version` on any breaking change (add layer type, change blob layout, change meaning of dims). FlexNN's compiler checks `version == supported_version` and fails with `unsupported version X, expected Y — re-export with newer FlexNN` if mismatch. Minor additive changes that keep old files readable use `header_len` to skip unknown tail bytes.

**Endian & float:**

- Little-endian (matches x86 host and M4). FlexNN writes via `memcpy` + `htole32` if needed (but on x86 it's no-op). FlexNN's compiler reads with `le32_to_host`. Float32 is IEEE-754 little-endian (same as host).

**Why not FlatBuffers/Protobuf:**

- No schema compiler, no runtime dependency on MCU, no reflection bloat. Our model is a linear stack, not a graph with optionals. FlatBuffers would add 30 KB of parser code to the compiler for no benefit. Rejected. And since the parser now lives in FlexNN, we don't want to pull Protobuf into FlexNN's minimal build.

---

## 6. FlexNN Compiler (PC CLI, lives in FlexNN)

> **v0.2 move:** This entire section was `TinyForge Compiler` in v0.1. It is now `FlexNN Compiler`. All paths `tools/compiler/...` are now `FlexNN/tools/compiler/...`. TinyForge no longer builds the compiler; it *invokes* the FlexNN-built `tinyforge-compile`.

### 6.1 Responsibilities & Non-Responsibilities

**Does (in FlexNN repo):**

- Parse flat binary (§5) via shared `ModelIO` structs → in-memory `Model` struct.
- Validate every `(type, activation, dims)` against **op registry** (§8, also in FlexNN). Fail fast.
- Validate arena fit (§9.2) against user-supplied `max_arena_bytes` or derived from target (`disco_f407vg: 112 KB SRAM budget`).
- Optional calibration: read representative dataset (CSV or raw binary) and record per-layer activation ranges (reuses `FlexNN` forward, no duplicate math).
- Quantize (§7) to int8 per-tensor (weights + activations) with optional BN folding.
- Emit `model_data.h` (+ `model_data.cpp` if needed) as compile-time constants for TinyForge.

**Does NOT:**

- Train. Training stays in `FlexNN.h`/`Layer.h`.
- Do graph optimization beyond BN folding and dead-code elimination of unused layers (none in linear stack).
- Link or flash. It just emits a header; TinyForge's CMake (via `cmake --build build-mcu` with the toolchain) compiles it into firmware.
- Live in TinyForge. *Why v0.2:* see §2.1. Keeping it in FlexNN lets `exportModel()` and `tinyforge-compile` share `LayerTypes.hpp` and `op_registry.hpp`, makes `isSupportedForExport()` a one-liner, and lets FlexNN's `ctest` gate export→compile in one CI job.

**Why co-located, not separate (inverted from v0.1):**

- *v0.1 argued* to keep compiler separate from FlexNN for testability without Eigen. *v0.2 corrects* that the compiler *does* need Eigen (for calibration) and *does* need the exact `LayerType` enums — separating them forced duplication. Co-location lets `tools/compiler/src/quantize.cpp` reuse `FlexNN`'s `forward()` for calibration instead of re-implementing layer math, and lets `validate.cpp` call `tinyforge::find()` without a relative `../external` path. The compiler is still testable in isolation (`FlexNN/tools/compiler/tests` links only `libFlexNN`), but it is built and versioned with FlexNN.

### 6.2 CLI (built as `FlexNN/build/tools/compiler/tinyforge-compile`)

```
tinyforge-compile --help
Usage: tinyforge-compile <model.bin> [options]

Positional:
  model.bin               Input flat binary from FlexNN::exportModel (FlexNN/lib/ModelIO.cpp)

Options:
  -o, --output <dir>      Output dir (default: ./generated) — writes model_data.h for TinyForge
  --backend <scalar|cmsis>  Target backend for registry check (default: scalar, checks TinyForge runtime)
  --check-only            Validate only, no quant/codegen (exit 0 if valid)
  --calib <csv>           Representative dataset for activation calibration (CSV, first col ignored)
  --calib-samples <N>     Max samples for calibration (default: 500)
  --arena <bytes>         Max arena bytes for validation (default: 98304 = 96 KB, leaves headroom for stack/audio)
  --no-quant              Emit float32 header (no int8) — useful for accuracy baseline
  --fold-bn               Enable BatchNorm folding (default: on)
  --verbose               Human-readable log per layer
  --version               Print FlexNN + compiler + registry version (all from same repo)
```

**Exit codes and `--check-only` use remain as v0.1** (0 success, 2 validation, 3 parse, 4 calib, 5 codegen). *New v0.2:* `FlexNN::exportModel()` will itself call `--check-only` internally and return `Status::ErrUnsupportedOp` before writing, so the CLI's `--check-only` is now also used as a pre-commit check in FlexNN's CI: `tinyforge-compile model.bin --check-only --backend cmsis` ensures the *just-exported* model is deployable to TinyForge's CMSIS backend.

### 6.3 In-Memory Model Representation (FlexNN compiler)

```cpp
// FlexNN/include/tinyforge/model.hpp  (compiler's model, shares LayerType/Activation)
namespace tinyforge {
enum class DType { F32=0, I8=1, I32=2 };

struct TensorDesc {
  DType dtype;
  std::vector<float> f32_data; // for F32; for I8, store int8 as float for calibration then quant
};

struct LayerDesc {
  LayerType type; // from FlexNN/include/LayerTypes.hpp
  Activation act;
  int inputDim, outputDim; // flattened
  Conv1DParams conv; // valid if type==Conv1D
  TensorDesc weights, biases, aux; // aux for BN
};

struct Model {
  uint16_t version;
  std::vector<LayerDesc> layers;
  struct Calib {
    std::vector<float> minAct, maxAct; // per-layer activation range (after activation)
  } calib;
};
}
```

Parser (`FlexNN/tools/compiler/src/parser.cpp`) does `read(fd, &header, sizeof(header))`, validates magic/version/CRC, then loops `layer_count` times reading `LayerHeader` (52 bytes) and then `pread` blobs at declared offsets. Validates `weight_count == expected`. *Shares* `LayerHeader` struct with `FlexNN/lib/ModelIO.cpp` — no duplication.

### 6.4 Validation Passes (FlexNN compiler)

**Pass 1 — Registry check** (see §8, header now `FlexNN/include/tinyforge/op_registry.hpp`): for each layer `i`, `lookup(type, act, quant_scheme)` must exist. If not, error:
```
error: layer 2 (Conv1D, act=Tanh) not in TinyForge registry for backend=cmsis, quant=int8.
  supported: (Conv1D, ReLU), (Conv1D, None)
  fix: change FlexNN model to ReLU or implement kernel in TinyForge runtime and bump registry
```

**Pass 2 — Dimension sanity:** same as v0.1 (`inputDim`, `weight_count`, Conv `L_out`, sequential `outputDim == next inputDim`).

**Pass 3 — Arena fit:** same as v0.1, `max_arena` computed with CMSIS temp, checked against `--arena`. *Why in FlexNN's compiler, not TinyForge runtime:* gives immediate feedback during FlexNN training iteration, without needing to copy `model.bin` to TinyForge and run `pio`.

### 6.5 Calibration (Activation Ranges, in FlexNN)

When `--calib` is supplied and quant is enabled:

1. Load CSV via `FlexNN::readCSV_XY` (now *in the same repo*, no wrapper) and reuse `FlexNN::NeuralNetwork::forward()` for calibration — no re-implementation.
2. For `N = min(calib_samples, dataset_rows)`, run `model.forward(sample)` in float, record per-layer `min/max` of post-activation output (`A`).
3. Compute per-layer `scale = (max - min) / 255`, `zero_point = clamp(round(-min/scale), -128,127)`.
4. If `--no-quant`, skip; if no calib data but quant requested, use weight range + heuristic `[-6,6]` for Tanh/Sigmoid.

*Why per-tensor for v0.1:* same as v0.1, one scale/zp per tensor, simplest CMSIS path.

### 6.6 Code Generation (FlexNN compiler emits for TinyForge)

**Output files (default `generated/` or `FlexNN/generated/` then copied to `TinyForge/generated/`):**

- `model_data.h` — single header, include-guard `TINYFORGE_MODEL_DATA_H`, contains `TINYFORGE_MODEL_VERSION`, `TINYFORGE_LAYER_COUNT`, `TINYFORGE_ARENA_BYTES`, `tinyforge_scale_l0`, `tinyforge_weights_l0[8192]`, `tinyforge_layers[]`, etc. (see Appendix C). TinyForge's runtime `#include "generated/model_data.h"` has no idea it was produced by FlexNN — it just sees constants.
- Reproducibility comment: `// Generated by FlexNN tinyforge-compile v0.2.0 (FlexNN SHA ...) from model.bin (sha256: ...) on 2026-09-05`.

**Quantized encoding and header-vs-blob rationale unchanged from v0.1** (affine, `model_data.h` in flash, no SD).

### 6.7 Compiler Testing (now in FlexNN)

- **Parser tests:** corrupt magic, version mismatch, CRC fail → exit 3 (in `FlexNN/tools/compiler/tests/parser_test`).
- **Validation tests:** each unsupported `(type,act)` → exit 2.
- **Quant tests:** golden.
- **Golden header test:** compile generated header with `arm-none-eabi-gcc -c`.
- **CI (FlexNN):** `ctest --test-dir build` runs *both* `FlexNN/tests` and `tools/compiler/tests` in one job; a `add_test(NAME tinyforge_compile_check COMMAND tinyforge-compile ... --check-only)` gates every example model.

---

## 7. Quantization Design (int8 per-tensor)

*This section lives logically in FlexNN's compiler (`FlexNN/tools/compiler/src/quantize.cpp`) but is documented here because TinyForge's kernels must agree.*

### 7.1 Math

Affine quantization (TFLite style):

```
real = scale * (q - zero_point)
q = clamp(round(real/scale + zero_point), -128, 127)

For matmul with bias:
  acc_int32 = Σ (input_q - zp_in) * (weight_q - zp_w) + bias_int32
  bias_int32 = round(bias_float / (scale_in * scale_w))
  output_q = requantize(acc_int32, multiplier, shift, zp_out)
```

*Why affine, not symmetric-only:* affine handles ReLU output (non-negative) without wasting half the int8 range. For weights we use symmetric (`zp_w=0`) to simplify CMSIS path; for activations we use affine (`zp` may be non-zero).

**Requantization:**

CMSIS-NN expects `arm_nn_requantize(acc, multiplier, shift)` where `multiplier` is Q31 and `shift` is 31 - `right_shift`. FlexNN's compiler computes:
```
effective_scale = (scale_in * scale_w) / scale_out
multiplier = quantize_multiplier(effective_scale) // frexp to Q31
shift = 31 - exponent
```

### 7.2 Calibration Details

- **Per-layer activation range:** after forward+activation, track `min/max` across calibration samples. For `ReLU`, `min` will be ≥0, so `zp` often 0; for `Tanh`, range ≈ [-1,1] → scale≈0.0078.
- **Weight range:** `scale_w = max(|W_min|, |W_max|) / 127` (symmetric). No calibration needed.
- **Clipping:** percentile 0.1% and 99.9% to avoid outlier blow-up (optional, v0.1 uses min/max; document as future improvement).

### 7.3 Accuracy vs Size

Example: Dense 128→64 float: weights 8192 *4=32 KB, biases 256 B. Int8: weights 8 KB, biases 256 B (int32). Header overhead ~200 B. Flash saving 75%. Accuracy drop target: <1% vs float on test set (measured in §17).

### 7.4 Non-Quantized Path

`--no-quant` emits `float` arrays and runtime uses `float` kernels (for Phase 2 accuracy baseline, or for layers where int8 would hurt, e.g., softmax). Runtime will have `TINYFORGE_USE_FLOAT` env.

### 7.5 BatchNorm Folding Math

If `BN` follows `Dense/Conv`:

```
W_fold = W * gamma / sqrt(var + eps)
b_fold = (b - mean) * gamma / sqrt(var+eps) + beta
```

FlexNN's compiler does this in float before quantization, then deletes the BN layer from the architecture descriptor. *Why in FlexNN's compiler, not TinyForge runtime:* zero runtime cost, no BN kernel needed, and FlexNN already has `gamma/beta/mean/var`.

---

## 8. Op-Support Registry (Single Source of Truth)

**File: `FlexNN/include/tinyforge/op_registry.hpp`** (authoritative; TinyForge runtime includes it via submodule)

```cpp
// FlexNN/include/tinyforge/op_registry.hpp — single header for both FlexNN and TinyForge
#pragma once
#include "LayerTypes.hpp" // LayerType, Activation

namespace tinyforge {
enum class QuantScheme { Float32, Int8PerTensor, Int8PerChannel };
enum class Backend { Scalar = 0, CmsisNN = 1, Both = 2 };

struct OpEntry {
  LayerType type;
  Activation act;
  QuantScheme quant;
  Backend backend; // SCALAR, CMSIS_NN, BOTH
  const char* kernel_symbol; // e.g., "tinyforge_dense_s8_scalar" (for TinyForge runtime)
  uint8_t version_added;
};

static constexpr OpEntry kRegistry[] = {
  {LayerType::Dense,  Activation::ReLU,     QuantScheme::Int8PerTensor, Backend::Both, "dense_relu_s8", 1},
  {LayerType::Dense,  Activation::None,     QuantScheme::Int8PerTensor, Backend::Both, "dense_none_s8", 1},
  {LayerType::Dense,  Activation::Sigmoid,  QuantScheme::Int8PerTensor, Backend::Both, "dense_sigmoid_s8", 1},
  {LayerType::Dense,  Activation::Tanh,     QuantScheme::Int8PerTensor, Backend::Both, "dense_tanh_s8", 1},
  {LayerType::Dense,  Activation::LeakyReLU,QuantScheme::Int8PerTensor, Backend::Both, "dense_lrelu_s8", 1},
  // Conv1D initially only ReLU/None int8, float later
  {LayerType::Conv1D, Activation::ReLU,     QuantScheme::Int8PerTensor, Backend::Both, "conv1d_relu_s8", 1},
  {LayerType::Conv1D, Activation::None,     QuantScheme::Int8PerTensor, Backend::Both, "conv1d_none_s8", 1},
  // Pool, BN folded — not standalone in registry for v0.1
};

inline const OpEntry* find(LayerType t, Activation a, QuantScheme q, Backend b) {
  for (auto &e : kRegistry) if (e.type==t && e.act==a && e.quant==q && (e.backend==b || e.backend==Backend::Both)) return &e;
  return nullptr;
}
} // namespace tinyforge
```

**TinyForge runtime usage (via submodule):**

```cpp
// TinyForge/runtime/include/tinyforge/runtime.hpp
#include "external/FlexNN/include/tinyforge/op_registry.hpp" // <-- same header, pinned SHA

Status Model::init() {
  for (auto &l : tinyforge_layers) {
    if (!tinyforge::find((LayerType)l.type, (Activation)l.act, QuantScheme::Int8PerTensor, kBackend))
      return Status::ErrUnsupportedOp;
  }
}
```

**Why single header in FlexNN (v0.2):**

- Both FlexNN's `exportModel()` validation and `tools/compiler` and TinyForge's `runtime.cpp` include the *same* file. Add a new op → add one line in `FlexNN/include/tinyforge/op_registry.hpp`, implement kernel in TinyForge `runtime/src/backend_*.cpp`, add tests in both repos, bump submodule SHA in TinyForge. CI fails if kernel symbol not found (`static_assert` on `kRegistry` size).
- Versioned: `version_added` lets FlexNN's compiler emit `requires registry v2` error if a model was exported with a newer FlexNN but TinyForge pins an older SHA (runtime `init()` will also fail).
- *v0.1 had* `TinyForge/tools/compiler/include/tinyforge/op_registry.hpp` duplicated from FlexNN — now deleted. The authoritative path is FlexNN's.

**Generation vs manual:**

- Manual for v0.1 (6–8 entries). Future: generate from TinyForge kernel implementations via `extract_registry.py` that greps `TINYFORGE_REGISTER_OP(...)` macros and updates `FlexNN/include/tinyforge/op_registry.hpp` via PR.

---

## 9. TinyForge Runtime — Constraints & Memory Model

### 9.1 MCU Constraints (STM32F407VGT6)

- **Core:** Cortex-M4F, 168 MHz, FPU SP (FPv4-SP-D16, `mfloat-abi=hard`), DSP (SIMD `SMLA*`, `SMLAD`).
- **Flash:** 1 MB (0x08000000), `.text` + `.rodata` (weights). 1 MB is ample for 3 models (each <100 KB).
- **RAM:** 192 KB split:
  - `0x20000000` SRAM1 112 KB (DMA-capable, general purpose)
  - `0x10000000` CCM 64 KB (core-coupled, *not* DMA-capable, faster, zero wait)
  - Backup 4 KB (not used)
  *Why matters:* I2S DMA (audio) buffers *must* be in SRAM1, not CCM. Arena for NN should be in CCM for speed (no DMA), but if model needs >64 KB arena, spill to SRAM1.

**Linker script nuance:** default PIO linker puts `.data/.bss` in SRAM1, `._ccmram` section in CCM if `__attribute__((section("._ccmram")))`. We will place `tinyforge_arena` in CCM via:
```cpp
static int8_t tinyforge_arena[TINYFORGE_ARENA_BYTES] __attribute__((section(".ccmram")));
```
and provide fallback `__attribute__((section(".ram")))`.

### 9.2 Static Arena & Double Buffering

For linear stack, we need two buffers (ping-pong):

```
arena[0]: input of layer i        (size = max_input_dim)
arena[1]: output of layer i       (size = max_output_dim)
After layer i: swap pointers (no memcpy)
```

**Sizing:**

- FlexNN's compiler computes `max_arena = max_i (sizeof(q) * max(in_dim_i, out_dim_i) + cmsis_temp)` where `cmsis_temp` for `arm_convolve_s8` may need `2 * C_in * K` extra (documented in CMSIS-NN). For Dense, temp = 0. TinyForge's runtime `static_assert`s the header's `TINYFORGE_ARENA_BYTES`.
- Example: model `Dense 128→64→10` int8, worst `128` bytes → arena = 128 B (!) + overhead → fits in 1 KB. Conv `C_in=3, L=100, C_out=8, K=5, stride=2` → `in=300, out=8*50=400` → arena ~400 B.
- Runtime `static_assert(TINYFORGE_ARENA_BYTES >= TINYFORGE_REQUIRED_ARENA, "increase arena")` at compile time.

*Why not per-layer `static Layer<In,Out>` templates:* templates would require compile-time `In,Out` as `constexpr` for each layer, leading to code bloat and no sharing of buffers. Arena is simpler and enables one `Model` class to run any model (as long as header's `TINYFORGE_ARENA_BYTES` matches the model's max). Rejected alternative: `std::array` per layer — wastes RAM.

**No heap invariant enforcement:**

- `-Wl,--wrap=malloc -Wl,--wrap=free -Wl,--wrap=_sbrk` + `static_assert` that no `new/malloc` symbols are linked (check via `nm` in CI).
- Code review checklist: no `new`, no `std::vector` in `runtime/`, only `span`-like views.

---

## 10. Runtime Backend Abstraction

**Compile-time selection (TinyForge runtime):**

```cpp
// TinyForge/runtime/include/tinyforge/backend.hpp
#pragma once
#include "external/FlexNN/include/tinyforge/op_registry.hpp" // registry from FlexNN submodule
namespace tinyforge {
enum class Backend { Scalar = 0, CmsisNN = 1 };

#if TINYFORGE_BACKEND == 1
constexpr Backend kBackend = Backend::CmsisNN;
#else
constexpr Backend kBackend = Backend::Scalar;
#endif

// Dispatch — no virtual, just constexpr if
template<LayerType T, Activation A>
int run_layer(const int8_t* input, const int8_t* weights, const int32_t* bias,
              int8_t* output, const QuantParams& qp, const ConvParams* conv = nullptr);
}
```

**Implementation:** `backend_scalar.cpp` implements all, `backend_cmsis.cpp` implements same symbols but inner loops call CMSIS. At link time only one `.cpp` is compiled per `TINYFORGE_BACKEND` (via `build_src_filter`).

*Why not virtual interface `class Backend { virtual void dense(...)=0; }`:* vtable indirection costs cycles and prevents inlining; also needs heap for object. `constexpr if` / `if constexpr` + templates give zero-cost and let compiler dead-strip unused backend.

**QuantParams struct (per-layer, from FlexNN-generated header):**

```cpp
struct QuantParams {
  float input_scale, weight_scale, output_scale;
  int32_t input_zp, weight_zp, output_zp; // weight_zp==0 symmetric
  int32_t multiplier; // Q31
  int shift;          // 1..31
};
```

---

## 11. Runtime Kernels — Scalar & CMSIS-NN

### 11.1 Dense (Fully Connected)

**Float path (baseline):**

```cpp
for (int o=0; o<out_dim; ++o) {
  float acc = bias[o];
  for (int i=0; i<in_dim; ++i) acc += weights[o*in_dim + i] * input[i];
  output[o] = activate(acc, act); // see §12
}
```

**Int8 scalar:**

```cpp
for (int o=0; o<out_dim; ++o) {
  int32_t acc = bias[o]; // bias already scaled (see §7)
  for (int i=0; i<in_dim; ++i) {
    acc += (int32_t)(input[i] - zp_in) * (int32_t)(weights[o*in_dim + i] - zp_w);
  }
  int32_t rescaled = arm_nn_requantize(acc, qp.multiplier, qp.shift); // or scalar equivalent
  rescaled += qp.output_zp;
  rescaled = std::clamp(rescaled, -128, 127);
  output[o] = (int8_t)rescaled;
  // activation after requant? For ReLU, clamp below zp; for Tanh/Sigmoid, need LUT on int8 domain
}
```

*Why bias is int32:* bias scale is `scale_in * scale_w`, so `bias_q = round(bias_float / scale_in*scale_w)` fits in 32 bits for typical ranges (need 32-bit to avoid overflow when `in_dim` up to 512).

**CMSIS-NN:**

```cpp
#include "arm_nnfunctions.h"
arm_fully_connected_s8(
  /*ctx=*/ &ctx, &fc_params, &quant_params,
  &input_dims, input,
  &filter_dims, weights,
  &bias_dims, bias,
  &output_dims, output);
```

where `fc_params` holds `input_offset = -zp_in`, `filter_offset = 0`, `output_offset = zp_out`, `activation.min/max = -128/127` (or `0/127` for ReLU). CMSIS handles the `SMLAD` SIMD (4× int8 MAC per cycle) via `__SMLAD`. *Why CMSIS wins:* 4× throughput on MACs, plus prefetching.

### 11.2 Conv1D

**Mapping to CMSIS:**

CMSIS has no `arm_convolve_1d_s8` in older releases; we map 1D to 2D with `h=1`:

```
arm_convolve_wrapper_s8(
  ctx, conv_params, quant_params,
  input_dims  = {1, L_in, C_in},  // CMSIS expects {batch, h, w, ch}
  input,
  filter_dims = {C_out, 1, K, C_in},
  weights,
  bias_dims = {C_out},
  bias,
  output_dims = {1, L_out, C_out},
  output
);
```

*Why this mapping works:* `h=1` collapses 2D to 1D. Need `arm_convolve_wrapper_s8` (auto picks `1x1`/`3x3` fast paths) or `arm_convolve_s8` generic. For v0.1 we use `arm_convolve_s8` to keep simple.

**Scalar fallback:**

Triple loop `oc → ol → k,ic` as in §4.4.2, with `int32_t acc`. No im2col needed for naive scalar (acceptable for small `L`); for speed we could add `col` buffer but not needed for correctness baseline.

**CMSIS temp buffer:**

`arm_convolve_s8` needs `ctx.buf` sized `2 * C_in * K` int16 + alignment. FlexNN's compiler arena calc must include this; TinyForge's runtime `ctx.buf = arena_temp`.

### 11.3 Pool1D (Future)

- Scalar: `output[oc][ol] = max_{k} input[oc][ol*stride + k]` (or mean).
- CMSIS: `arm_max_pool_s8` / `arm_avgpool_s8` with `h=1`.

### 11.4 Kernel Correctness Gate (§5.3)

For each kernel, host test (in TinyForge, but also gated by FlexNN PR) does:

```
for random input in [-128,127], random weights:
  out_scalar = kernel_scalar(input, weights, qp)
  out_cmsis  = kernel_cmsis(input, weights, qp)  // via compiled for host with CMSIS stub or on device via UART dump
  assert max(abs(out_scalar - out_cmsis)) <= 1
```

*Why ±1:* requantize rounding may differ by 1 LSB between scalar `round` and CMSIS `NN_ROUND`.

---

## 12. Activation Handling on Cortex-M4F

### 12.1 ReLU / LeakyReLU

- **ReLU int8:** `q_out = max(q, zp)` if `zp` is ReLU's zero. For symmetric `zp=0`, it's `max(q,0)`. Scalar: `if (x < zp) x=zp;`.
- **LeakyReLU int8:** piecewise: `if (x >= zp) y=x else y = zp + (x - zp)*alpha` with `alpha=0.01` → `y = zp + ((x - zp)*13 >> 10)` approx (`0.0126`). Use `arm_nn_requantize` trick? For v0.1 we can implement float fallback then requant, but note that Leaky on int8 is rare; we may restrict Leaky to float path initially and registry will mark `LeakyReLU int8` as scalar-only until CMSIS supports.

### 12.2 Sigmoid / Tanh — The Hard Part (TBD in spec §11)

**Constraints:** M4F has FPU, so `expf`/`tanhf` work but are *slow* (~50 cycles each) and bring `libm`. For 64-neuron layer, `64*expf` = 3200 cycles, acceptable for 1 Hz inference but not for 50 Hz keyword spotting. We must choose per use-case.

**Options and decision for LLD v0.2:**

A) **Exact `expf` + `tanhf` (float path):** simplest, most accurate. Use for float models and for int8 when accuracy matters. Code: `float y = 0.5f*(1+tanhf(x))` for sigmoid via `arm`? Might just call `expf`.

B) **LUT + linear interpolation (int8 path):** 256-entry `int8` LUT for `sigmoid` over `[-8,8]` → index `i = clamp((x - zp)*scale, -8,8)` → `y = lut[(i+8)*16]`. Interpolation between entries via `y0 + (frac)*(y1-y0)`. Size 256 B, ~5 cycles.

C) **Piecewise linear (PWL):** CMSIS-NN uses `arm_nn_activations_direct`? Actually CMSIS provides `arm_sigmoid_s8` with `int16` LUT.

**LLD decision for v0.2:**

- **Float runtime:** use `sigmoid = 1/(1+expf(-x))`, `tanh = tanhf(x)` (allow `-ffast-math`).
- **Int8 scalar:** use **LUT 256** for both. Generated LUT at compile time in `model_data.h` as `static const int8_t sigmoid_lut[256]` (emitted by FlexNN's compiler).
- **Int8 CMSIS:** use `arm_nnfunctions.h` `arm_sigmoid_s8` if available in CMSIS-NN 6.x; else fallback to LUT.

*Why LUT for int8:* int8 domain is already quantized; LUT is O(1) and branch-free. Accuracy loss <1% vs expf, acceptable for edge.

**Softmax on MCU:** not computed; we **argmax logits** (`int8` or `float`) directly. For float softmax, if user really needs probabilities, they can compute softmax on host after UART dump. Document: `Activation::Softmax` on last layer is treated as `None` + argmax in TinyForge runtime.

---

## 13. Runtime Public API & Error Model

**Header:** `TinyForge/runtime/include/tinyforge/runtime.hpp` (includes `generated/model_data.h` which was produced by FlexNN's compiler)

```cpp
#pragma once
#include <stdint.h>
#include "generated/model_data.h" // from FlexNN's compiler

namespace tinyforge {

enum class Status : int32_t {
  Ok = 0,
  ErrBadDims = -1,
  ErrArenaTooSmall = -2,
  ErrUnsupportedOp = -3, // registry miss — should have been caught by FlexNN's compiler, but re-checked
  ErrNullPtr = -4
};

// No heap, no exceptions — all errors are Status
class Model {
public:
  // Binds to static arena; does not allocate
  Status init(); // validates model_data.h vs registry (via external/FlexNN/include/tinyforge/op_registry.hpp), checks arena
  // Run inference: input and output are caller-owned buffers (int8* or float* depending on quant)
  // For int8: sizes are TINYFORGE_INPUT_DIM / OUTPUT_DIM bytes
  Status run(const int8_t* input, int8_t* output);
  Status run(const float* input, float* output); // float overload when TINYFORGE_USE_FLOAT
  int input_dim() const { return TINYFORGE_INPUT_DIM; }
  int output_dim() const { return TINYFORGE_OUTPUT_DIM; }
private:
  int8_t* arena0_;
  int8_t* arena1_;
};

} // namespace tinyforge
```

**Why class, not free functions:** encapsulates arena pointers, allows `static Model instance;` plus `init()` pattern familiar from HAL.

**Error handling:**

- `init()` checks `TINYFORGE_LAYER_COUNT <= kMaxLayers (16)` and each `type/act` in registry (via FlexNN's header); on fail returns `ErrUnsupportedOp` and also `printf` via UART if `TINYFORGE_DEBUG`. This is a second line of defense — FlexNN's compiler should have already rejected, but runtime re-checks in case a stale `model_data.h` from an older FlexNN SHA is flashed with newer TinyForge.
- `run()` checks null, then loops layers calling `run_layer`. No `assert`, no `exit`; caller can `if (status != Ok) blink_error_led(status)`.
- No C++ exceptions: compile with `-fno-exceptions`.

**Input/Output ownership:**

- Caller owns `input`/`output` buffers (e.g., sensor window or UART test vector). Runtime only uses arena for intermediate. For `Dense 128→64→10`, caller provides `float/int8[128]` and `float/int8[10]`.

---

## 14. Platform Layer (HAL/LL, Clock, DMA, Sensors, DSP)

### 14.1 Clock & Power

- Use `SystemClock_Config()` from CubeMX: HSE 8 MHz → PLL 168 MHz, `FLASH_LATENCY=5`, `AHB=168`, `APB1=42`, `APB2=84`. Enable FPU `SCB->CPACR |= (3UL<<20)|(3UL<<22)`.
- Enable DWT cycle counter for latency: `DWT->CTRL|=1; DWT->CYCCNT=0;` then `latency = DWT->CYCCNT`.

### 14.2 UART Logging

- `USART2` via ST-Link virtual COM (115200 baud). Non-blocking `HAL_UART_Transmit` for logs, DMA for bulk test vectors.

### 14.3 Sensor Drivers

**Accelerometer — LIS3DSH vs LIS302DL (TBD):**

- Both are SPI. We'll write a thin `AccelDriver` that probes `WHO_AM_I` (0x3F for LIS3DSH, 0x3B for LIS302DL) at boot and branches. SPI1, CS=PE3.
- Poll at 100 Hz (7.1, 7.2): `HAL_SPI_Receive` 6 bytes (X/Y/Z each 16-bit). Window 64–128 samples per axis → input dim 192–384.
- *Why polling, not interrupt:* simpler, deterministic latency; 100 Hz leaves 10 ms slack.

**Microphone — MP45DT02 PDM → PCM:**

- PDM clock 2 MHz via `I2S2` + `DMA1_Stream3`, decimation via `libPDMFilter` or CMSIS `arm_pdm_to_pcm`? Cube provides `PDM2PCM` library.
- DMA double buffer in SRAM1 (DMA-capable): `int16_t mic_buf[2][1024]` 16 kHz mono. Half-complete callback fills inference window.
- **Feature extraction (7.3):** MFCC via CMSIS-DSP (`arm_mfcc_init`, `arm_rfft_fast_f32`, mel filterbank). Window 25 ms (400 samples @16 kHz), hop 10 ms (160), 13 MFCCs → classifier input dim e.g., 13*10=130.

*Why SRAM1 for DMA buffers:* CCM cannot be DMA source; placing them there would hard-fault.

### 14.4 DSP Preprocessing Contract

- Preprocessing is *outside* the NN runtime but inside the same firmware image. It writes into the runtime's `input` buffer. For 7.3, `mfcc_compute(mic_buf, input_f32)` then `quantize(input_f32 → input_int8)` if int8 model.

---

## 15. End-to-End Pipeline & CLI Examples

```bash
# --- In FlexNN repo (or TinyForge/external/FlexNN when vendored) ---

# 1) Train in FlexNN (host)
cd FlexNN  # or cd external/FlexNN inside TinyForge
cmake -S . -B build && cmake --build build -j
./build/main  # trains, calls FlexNN::exportModel("model.bin") — now validates via registry before writing

# 2) Compile/validate/quantize — NOW WITH FlexNN's compiler
./build/tools/compiler/tinyforge-compile model.bin --backend cmsis --calib data/calib.csv --arena 98304 -o /path/to/TinyForge/generated
# or if inside TinyForge:
#   external/FlexNN/build/tools/compiler/tinyforge-compile model.bin --backend cmsis --calib data/calib.csv --arena 98304 -o generated
# emits generated/model_data.h (for TinyForge)

# --- In TinyForge repo (this repo, MCU) ---

# 3) Build firmware (MCU, consumes generated header from FlexNN)
cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake
cmake --build build-mcu -j
# flash via OpenOCD / ST-Link: openocd -f board/stm32f4discovery.cfg -c "program build-mcu/main.elf verify reset exit"

# 4) Validate on-device (UART)
# firmware prints: "init ok, arena 400B, latency 12450 cycles, output [0.1, 0.8, ...]"
```

**Variant: validate without quant/codegen (fast CI):**

```bash
./build/tools/compiler/tinyforge-compile model.bin --check-only --backend cmsis
# exit 0 if FlexNN's model is deployable to TinyForge's CMSIS backend
# FlexNN's own ctest runs this on every example model.bin
```

**Why `TinyForge/generated/` is gitignored (`build/` pattern):** generated header is artifact, not source; FlexNN's compiler regenerates it. CI regenerates it from the pinned FlexNN SHA. You may commit an example `TinyForge/generated/example_model_data.h` for tests with `// GENERATED by FlexNN v...` comment, but the live `model_data.h` is ignored.

---

## 16. Target Applications — Wiring to Runtime

### 16.1 7.1 Vibration Anomaly (Binary, Accelerometer)

- **Data:** window `64 × 3 axes = 192` float samples, normalized `[-1,1]` (accel g).
- **Model:** `Dense 192→32 ReLU → Dense 32→1 Sigmoid` (or `Sigmoid` → threshold 0.5). Float size ~25 KB, int8 ~6 KB (header from FlexNN compiler).
- **Wiring (TinyForge):** `accel_read(window)` → `model.run(window, &out)` → `if out>0.5 anomaly`.
- **Why MLP first, Conv later:** smallest, validates pipeline; Conv1D variant `Conv1D C_in=3→C_out=8 K=5 stride2 → Dense 8*~30=240→32→1` adds temporal locality but needs Conv kernel in TinyForge and FlexNN `Conv1D` layer.

### 16.2 7.2 Gesture/Activity (Multi-class, Accelerometer)

- Same input windowing, larger output `Dense 32→6 Softmax` (walk/idle/shake/...). Softmax on host only, MCU does argmax on logits (TinyForge runtime skips softmax).
- Needs more data collection (UART dump to PC, label, train in FlexNN — then FlexNN's compiler emits header).

### 16.3 7.3 Keyword Spotting (Mic + MFCC)

- **Fallback:** Voice Activity Detection `Dense 130→32→1` if MFCC too heavy.
- **Full KWS:** `Dense 130→64 ReLU → Dense 64→12 Softmax` (12 keywords from SpeechCommands subset).
- **Wiring (TinyForge):** `i2s_dma_callback` fills `pcm[400]` → `mfcc(pcm)→mfcc_feat[13]` accumulate 10 frames → `model.run(feat, logits)` → `argmax`.
- **Why MFCC extra step:** raw 16kHz × 1 sec = 16000 dim too large for MCU; MFCC compresses to 130 dim with perceptually relevant features. CMSIS-DSP provides `arm_mfcc_*` with fixed-point.

---

## 17. Benchmarking Harness (vs TFLite Micro)

Per §8 of spec, three firmware variants **per application**, separately flashed:

1. `tinyforge_scalar` (our scalar, TinyForge)
2. `tinyforge_cmsis` (our CMSIS-NN, TinyForge)
3. `tflm_cmsis` (TFLite Micro + CMSIS-NN kernels, *not* reference kernels — invalid comparison otherwise)

**Metrics (per image, same test vectors via UART):**

- **Latency:** `DWT_CYCCNT` start/end around `model.run`. Report mean/median/p95 over 1000 runs. Unit: cycles + ms (cycles/168MHz).
- **RAM:** `arm-none-eabi-size -B` → `.data + .bss + arena`. For TFLM, include `tensor_arena` (e.g., `constexpr int kArena = 10*1024`). Also stack high-water via `0xAA` paint.
- **Flash:** `size` → `.text + .rodata` split `weights vs code` by `nm --size-sort` or `arm-none-eabi-objdump`.
- **Accuracy:** same test set (500 samples) run on PC (FlexNN float via `FlexNN/build/main`) vs on-device int8 (TinyForge); compute `correct / total`. TFLM int8 vs TinyForge int8 vs float baseline.

**Expected (honest):** latency similar (both use CMSIS MACs), TinyForge smaller RAM/flash (no interpreter, no FlatBuffer, compile-time graph from FlexNN). Report numbers regardless.

**Hardware note:** Audio DMA buffers in SRAM1 not counted as NN arena but reported separately.

---

## 18. Testing Strategy

### Host (PC) — GoogleTest

- **`FlexNN` tests (in `FlexNN/tests/` and `FlexNN/tools/compiler/tests/`):**
  - `activation_test`, `dense_test`, `conv1d_test` (FlexNN)
  - `export_import_test` (FlexNN, then runs `FlexNN/build/tools/compiler/tinyforge-compile` on exported file)
  - `parser_test` (FlexNN compiler, bad magic/CRC/version → exit 3)
  - `validate_test` (FlexNN compiler, unsupported `(type,act)` → exit 2)
  - `quant_test` (FlexNN compiler, golden), `codegen_test` (compile `model_data.h` with `arm-none-eabi-gcc`)
- **TinyForge host tests (in `TinyForge/CMakeLists.txt` when `TINYFORGE_USE_FLEXNN`):**
  - May link `external/FlexNN` for golden float reference, but not required; TinyForge's host tests can just include `generated/model_data.h` and run scalar vs cmsis stubs.

### MCU — `tests/` on host (TinyForge, scalar reference)

- `test_runtime_dense`: scalar vs cmsis vs float reference on same input (input.bin from FlexNN's `exportImport` via UART or embedded array).
- `test_runtime_conv`: same.
- `test_arena`: `static_assert` arena size, `paint` stack.

### On-device vs Host Parity

- Run inference on PC (FlexNN float) and on MCU (TinyForge int8) for same `input.bin` (from FlexNN), capture MCU output via UART, compare on host with tolerance `≤1 LSB`.

### CI (GitHub Actions, planned)

- **`FlexNN` CI:** `cmake -S . -B build && ctest` — builds `libFlexNN` + `tinyforge-compile` and runs all host tests including `--check-only` on example models. This is where export→compile is gated.
- **`TinyForge` CI:** `cmake -S . -B build && cmake --build build` (host scalar) and `cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake && cmake --build build-mcu` (cross, no hardware) — verifies that the pinned `external/FlexNN` SHA's `op_registry.hpp` + `generated/model_data.h` (regenerated via FlexNN's compiler) still compiles for MCU. Also `registry-gate`: run `external/FlexNN/build/tools/compiler/tinyforge-compile model.bin --check-only`.

---

## 19. File Tree (Planned) & Generated Artifacts

```
FlexNN/  (public, owns compiler)
├── include/
│   ├── FlexNN.h, Layer.h, LayerTypes.hpp, ModelIO.hpp, Utility.h
│   └── tinyforge/
│       ├── op_registry.hpp          # SINGLE SOURCE OF TRUTH
│       ├── model.hpp                # compiler's Model
│       ├── quant.hpp
│       └── codegen.hpp
├── lib/
│   ├── FlexNN.cpp, Layer.cpp, ModelIO.cpp, Utility.cpp
│   └── tinyforge/                   # optional: registry helpers
├── tools/
│   └── compiler/
│       ├── CMakeLists.txt           # add_executable(tinyforge-compile)
│       ├── src/{main.cpp, parser.cpp, validate.cpp, quantize.cpp, codegen.cpp}
│       └── tests/{parser_test.cpp, quant_test.cpp, ...}
├── tests/                           # FlexNN tests
├── CMakeLists.txt                   # super-build
└── ...

TinyForge/  (private, runtime only)
├── external/
│   └── FlexNN/                      # submodule → FlexNN @ main
│       ├── include/tinyforge/op_registry.hpp  # <-- runtime includes this
│       └── build/tools/compiler/tinyforge-compile  # binary after building FlexNN
├── runtime/
│   ├── include/tinyforge/{runtime.hpp, backend.hpp, kernels/{dense.hpp, conv.hpp, activations.hpp}, quant.hpp}
│   ├── src/{runtime.cpp, backend_scalar.cpp, backend_cmsis.cpp}
│   └── platform/{clock.cpp, uart.cpp, accel_driver.cpp, mic_driver.cpp, dsp_mfcc.cpp}
├── generated/                       # .gitignore, OUTPUT of FlexNN's compiler
│   └── model_data.h                 # generated by external/FlexNN/build/tools/compiler/tinyforge-compile
├── docs/{OVERVIEW.md,LLD.md}          # committed spec (ref/ is local ignored copy)
├── cmake/toolchains/arm-cortex-m4f.cmake  # cross toolchain (replaces platformio.ini)
├── CMakeLists.txt                   # super-build for TinyForge host tests (add_subdirectory(external/FlexNN) optional)
└── ref/{OVERVIEW.md, LLD.md}        # local-only, gitignored
```

Generated `model_data.h` example: see Appendix C (emitted by `FlexNN/tools/compiler`, consumed by `TinyForge/runtime`).

---

## 20. Implementation Order & Bottom-Up Gating

**Phase 0 (done):** Repo + submodule + super-build + LLD v0.1 → **v0.2: moved compiler to FlexNN, updated LLD.**

**Phase 1 — Minimal E2E (Dense + ReLU):**

1. In `FlexNN`: `Activation::ReLU` enum, `LayerTypes.hpp` Dense, `ModelIO.cpp` export, `include/tinyforge/op_registry.hpp` with Dense+ReLU, `tools/compiler` (parser/validate/quant/codegen) for Dense.
2. In `TinyForge`: scalar kernel → cmsis kernel `arm_fully_connected_s8` → `runtime.cpp` that includes `external/FlexNN/include/tinyforge/op_registry.hpp` → `Model::run` for Dense → app 7.1 MLP that consumes `FlexNN`-generated `model_data.h`.
*Gate:* `TinyForge test_runtime_dense` scalar vs cmsis `±1` **and** `FlexNN ctest` `export_import + tinyforge-compile --check-only` passes. *No FlexNN export allowed until TinyForge kernel exists.*

**Phase 2 — Activations:** add `LeakyReLU, Sigmoid, Tanh` in `FlexNN` (host forward/backward + FlexNN compiler quant ranges + LUT/CMS-S kernels in TinyForge) → registry → Quant activation ranges → test (both repos).

**Phase 3 — Conv1D:** scalar Conv in TinyForge → cmsis `arm_convolve_s8` → registry Conv+ReLU (in FlexNN) → `FlexNN` `Conv1D` layer + export → FlexNN compiler Conv dims + arena temp → test.

**Phase 4 — BN folding + Pool:** folding pass in FlexNN's `tools/compiler/src/quantize.cpp`, pool kernels in TinyForge → registry.

**Phase 5 — Preprocessing & Apps 7.2/7.3:** accel/mic drivers in TinyForge, MFCC, wiring → benchmarks.

**Phase 6 — TFLM baseline & paper:** build `tflm_cmsis` variants, measure per §17.

*No FlexNN layer is “unlocked” for export until its TinyForge kernels + registry entry exist — enforced by `FlexNN::isSupportedForExport()` which directly checks the shared `FlexNN/include/tinyforge/op_registry.hpp`.*

---

## 21. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Softmax backward incorrect blocks training of new activations | Medium | High | Fix hidden softmax Jacobian, or restrict Softmax to last layer + fuse with CE; add finite-diff tests in FlexNN |
| Arena exceeds CCM 64 KB | Low (models small) | Medium | FlexNN's compiler checks arena vs `--arena`; fallback to SRAM1 via `__attribute__((section(".ram")))`; reduce model dims |
| CMSIS-NN API mismatch (version 5 vs 6) | Medium | Medium | Pin `CMSIS-NN@1.3.0` via `lib_deps` in TinyForge; wrapper that `#if` checks `ARM_CMSIS_NN_VERSION` |
| LUT accuracy for Sigmoid/Tanh degrades to >2% drop | Medium | Medium | Fallback to `expf` float path, report tradeoff; increase LUT to 512 with interpolation (FlexNN compiler emits larger LUT) |
| FlexNN CSV reading slow for large calibration set | Low | Low | `--calib-samples 500` default; stream, don't load all (FlexNN's compiler reuses `Utility::readCSV_XY`) |
| DMA buffers in CCM hard-fault | High if ignored | High | `static_assert` DMA buffers in `0x20000000` SRAM1 via linker check script in TinyForge |
| TFLM comparison unfair (forgot to enable CMSIS) | High | High | `tflm_cmsis` variant explicitly sets `CMSIS_NN=1`; CI builds both and checks `arm_fully_connected_s8` symbol present via `nm` |
| **New v0.2:** FlexNN and TinyForge drift (registry out of sync) | Medium | High | **Registry lives only in FlexNN**; TinyForge includes it via `external/FlexNN/include/tinyforge/op_registry.hpp` at the pinned SHA. TinyForge's `init()` re-checks registry. Bump submodule SHA on every registry change. CI in both repos checks `git submodule status` is clean. |
| **New v0.2:** TinyForge needs FlexNN to build but FlexNN is a submodule not yet built | Low | Medium | Documented two-step build: `cmake -S external/FlexNN -B external/FlexNN/build` then `tinyforge-compile`. TinyForge's CI builds FlexNN first. |

---

## Appendix A: Binary Format Reference

See §5. Magic `0x54464E47`, version `1`, little-endian, CRC32 IEEE, per-layer 52-byte header + blobs. Example hexdump:

```
000000: 47 4E 46 54 01 00 0A 00 03 00 00 00 00 00 00 00  CRC...
000010: # layer 0: Dense 128->64 ReLU, 8192 weights @0x100, 64 biases @0x8100 ...
```

Parser pseudocode in `FlexNN/tools/compiler/src/parser.cpp:17`:

```cpp
LayerHeader h; in.read((char*)&h, sizeof(h));
h.layer_type = le8(h.layer_type); // no swap
h.input_dim = le32(h.input_dim);
assert(h.weight_count == expected_for[h.layer_type]);
```

*Note:* Parser shares `LayerHeader` struct with `FlexNN/lib/ModelIO.cpp` writer — no duplication.

---

## Appendix B: Op Registry Example

`FlexNN/include/tinyforge/op_registry.hpp:10` (authoritative, included by TinyForge via submodule)

```cpp
static constexpr OpEntry kRegistry[] = {
  {LayerType::Dense, Activation::ReLU, QuantScheme::Int8PerTensor, Backend::Both, "dense_relu", 1},
  {LayerType::Dense, Activation::None, QuantScheme::Int8PerTensor, Backend::Both, "dense_none", 1},
  {LayerType::Conv1D, Activation::ReLU, QuantScheme::Int8PerTensor, Backend::Both, "conv1d_relu", 1},
};
static constexpr size_t kRegistrySize = sizeof(kRegistry)/sizeof(kRegistry[0]);
```

TinyForge runtime dispatch (via submodule header):

```cpp
#include "external/FlexNN/include/tinyforge/op_registry.hpp"
for (i in 0..layer_count-1) {
  const auto* e = tinyforge::find(layers[i].type, layers[i].act, QuantScheme::Int8PerTensor, kBackend);
  if (!e) return Status::ErrUnsupportedOp;
  run_layer_dispatch(e->kernel_symbol, ...);
}
```

---

## Appendix C: Generated Header Example

`TinyForge/generated/model_data.h` (tiny model `Dense 4→3 ReLU → Dense 3→2 Softmax`, int8, **emitted by `FlexNN/build/tools/compiler/tinyforge-compile`**):

```c
#pragma once
#include <stdint.h>
#define TINYFORGE_MODEL_VERSION 1
#define TINYFORGE_LAYER_COUNT 2
#define TINYFORGE_INPUT_DIM 4
#define TINYFORGE_OUTPUT_DIM 2
#define TINYFORGE_ARENA_BYTES 8
static const float tinyforge_scale_l0_in = 0.05f;
static const int32_t tinyforge_zp_l0_in = 0;
static const float tinyforge_scale_l0_w = 0.023f;
static const float tinyforge_scale_l0_out = 0.04f;
static const int32_t tinyforge_mult_l0 = 1073741824; // Q31
static const int tinyforge_shift_l0 = 3;
static const int8_t tinyforge_weights_l0[12] __attribute__((aligned(4))) = {12, -3, 5, 8, -10, 2, 6, 4, -1, 9, 0, -5};
static const int32_t tinyforge_bias_l0[3] = {123, -45, 67};
static const int8_t tinyforge_weights_l1[6] __attribute__((aligned(4))) = {7, -2, 3, -4, 1, 8};
static const int32_t tinyforge_bias_l1[2] = {10, -20};
typedef struct { uint8_t type; uint8_t act; uint16_t in_dim; uint16_t out_dim; } tinyforge_layer_desc_t;
static const tinyforge_layer_desc_t tinyforge_layers[2] = {{0,1,4,3},{0,5,3,2}};
```

`TinyForge/runtime/src/runtime.cpp:42` iterates `tinyforge_layers` and calls `run_layer`.

---

## Appendix D: Decisions Log (extends §9 of OVERVIEW)

- **Flat binary over FlatBuffers/Protobuf (§5):** 50 LOC parser vs 30 KB dep, linear stack needs no graph flexibility. Chose magic+version+CRC for debuggability. Now shared between `FlexNN/lib/ModelIO.cpp` and `FlexNN/tools/compiler/src/parser.cpp` — no duplication.
- **Affine per-tensor int8 first (§7):** simplest CMSIS path (`arm_fully_connected_s8` expects per-tensor), 75% size saving, <1% accuracy loss expected; per-channel later for +0.5% accuracy. Quant code lives in `FlexNN/tools/compiler/src/quantize.cpp` so FlexNN can reuse it for calibration.
- **Header codegen (§6.6):** MCU has no FS; `const` in flash, no parser, deterministic. Rejected SD-card blob (extra hardware, runtime parsing cost). Header is now emitted by FlexNN, consumed by TinyForge — clean producer/consumer.
- **Backend enum compile-time (§10):** `constexpr if` eliminates branch, enables DCE, saves cycles; runtime switch would cost ~2% latency.
- **LUT for Sigmoid/Tanh int8 (§12):** 256 B table, 5 cycles vs 50 cycles `expf`; accuracy tradeoff documented, fallback to `expf` for float. LUT is emitted by FlexNN's compiler into `model_data.h`.
- **Argmax for Softmax (§12):** softmax monotonic, argmax preserves class, saves `exp` entirely; probabilities not needed on edge, can be computed on host if logged via UART.
- **Arena double-buffer (§9.2):** minimal RAM for linear stack, no per-layer template bloat, one `Model` class for any model.
- **Bottom-up gating (§4):** prevents training a non-deployable model; now enforced by `FlexNN::isSupportedForExport()` which directly checks `FlexNN/include/tinyforge/op_registry.hpp` (no cross-repo call).
- **NEW v0.2 — Compiler lives in FlexNN, not TinyForge (§2, §6, §8):** Training, serialization, and compilation share `LayerType`/`Activation` and `op_registry.hpp`. Co-location lets `exportModel()` validate before writing, lets `tools/compiler` reuse `FlexNN` forward for calibration without re-implementing math, makes a single `FlexNN` CI (`ctest`) gate export→compile, and keeps TinyForge minimal (only runtime + `generated/model_data.h`). TinyForge pins FlexNN via `external/FlexNN` and includes the same `op_registry.hpp` for its `Model::init()` second check. Rejected: keeping compiler in TinyForge (would duplicate enums, require relative `../external` includes that break standalone FlexNN builds).

---

*End of LLD v0.2 — next step: implement FlexNN `LayerTypes.hpp` + `ModelIO.cpp` + `tools/compiler` (Phase 1 Dense+ReLU) per §20, then TinyForge runtime kernels; update this doc with measured latency/RAM/accuracy.*

