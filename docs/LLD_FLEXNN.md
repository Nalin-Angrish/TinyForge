# FlexNN — Low-Level Design (LLD_FLEXNN) v1.0

> **Parent spec:** `docs/OVERVIEW.md` §5.1 and `docs/LLD_TINYFORGE.md` (consumer). This document is the complete LLD for **FlexNN itself**: what it is today, what it must become, and exactly how. FlexNN is a **standalone** C++ training library (Eigen + OpenMP) that trains and exports `model.bin` via `ModelIO`. It has **no dependency on TinyForge** — TinyForge's compiler validates and consumes FlexNN's export, not the other way.

**Status:** Draft · **Repo:** `Nalin-Angrish/FlexNN` (public, standalone) · **Target:** Host x86_64 Linux (Eigen 3.4+, OpenMP, CMake 3.10+, C++17) · **Version:** 1.0 · **Date:** 2026-09-05

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Current Implementation Audit](#2-current-implementation-audit)
3. [Design Principles](#3-design-principles)
4. [Activation Refactor — `enum class Activation`](#4-activation-refactor--enum-class-activation)
5. [Layer Type System — `enum class LayerType`](#5-layer-type-system--enum-class-layertype)
6. [New Layer Types](#6-new-layer-types)
7. [Model I/O — `exportModel` / `importModel` & Flat Binary Format](#7-model-io--exportmodel--importmodel--flat-binary-format)
8. [Utility & Training Improvements](#8-utility--training-improvements)
9. [Build System — CMake](#9-build-system--cmake)
10. [Testing Strategy](#10-testing-strategy)
11. [File Tree & Artifacts](#11-file-tree--artifacts)
12. [Implementation Order](#12-implementation-order)
13. [Risks & Mitigations](#13-risks--mitigations)
14. [Appendix A: Layer Header Format (shared with TinyForge)](#appendix-a-layer-header-format-shared-with-tinyforge)
15. [Appendix B: Example Export Flow](#appendix-b-example-export-flow)

---

## 1. Overview & Goals

**FlexNN today:** a from-scratch fully-connected neural network library that trains `Dense` layers with `ReLU`/`Softmax` on `Eigen::MatrixXd`, using `OpenMP` for parallelism. It reads `data/mnist-digit-recognition.csv`, normalizes, trains via `NeuralNetwork::train()`, and evaluates via `accuracy()`. It has **no** save/load, no `Conv1D`, no `BatchNorm`, no `Pool`, and uses `std::string` for activations.

**FlexNN must become:** a flexible *training* library that (a) supports `Dense`, `Conv1D`, `BatchNorm1D`, `MaxPool1D`/`AvgPool1D` and `ReLU`/`LeakyReLU`/`Sigmoid`/`Tanh`/`Softmax` via a type-safe `enum class` system, and (b) can **export** any trained `NeuralNetwork` to a **flat binary** `model.bin` that TinyForge (or any future runtime) can parse without Eigen. FlexNN must remain **standalone**: no `tinyforge/` headers, no validation against TinyForge's registry at export time — it just writes the requested `(type, act)` and returns `Status{ok, error}` if the caller asks for an impossible export (e.g., `Conv1D` with `dilation !=1` in v0.1).

**Why this scope:** keep FlexNN's responsibility minimal (train + serialize). All validation for a specific runtime (TinyForge's arena, CMSIS-NN kernel coverage) lives in that runtime's compiler (`TinyForge/tools/compiler`). This keeps FlexNN reusable for other runtimes and avoids coupling FlexNN to `arm_nnfunctions.h`. Rejected: make `FlexNN::exportModel()` call `tinyforge::find()` — would make FlexNN depend on TinyForge.

### 1.1 Non-Goals

- No on-device training, no optimizer beyond SGD (Adam is future, not v0.1).
- No generic ONNX import — FlexNN is the source format by design.
- No quantization in FlexNN — quantization is `TinyForge/tools/compiler`'s job (see `LLD_TINYFORGE.md`).

### 1.2 Success Criteria

- `FlexNN` builds on host with `cmake -S . -B build && cmake --build build && ctest` (existing `src/main.cpp` still works, plus new `tests/`).
- `FlexNN::Layer` can be constructed as `Layer(3*64, 16, Activation::ReLU, LayerType::Conv1D)` and `forward()`/`backward()` pass finite-difference tests.
- `FlexNN::exportModel(net, "model.bin")` writes a `model.bin` that `TinyForge`'s `tinyforge-compile` can parse (see `LLD_TINYFORGE.md` §5) and that `FlexNN::importModel()` can round-trip.

---

## 2. Current Implementation Audit

Reading `FlexNN/include/Layer.h:37-151`, `FlexNN/lib/Layer.cpp:25-82`, `FlexNN/include/FlexNN.h:37-131`, `FlexNN/lib/FlexNN.cpp:29-142`, `FlexNN/include/Utility.h:1-69`, `FlexNN/src/main.cpp:1-111`:

### 2.1 `Layer` — monomorphic, stringly-typed

```cpp
// FlexNN/include/Layer.h:37
class Layer {
  int inputSize, outputSize;
  std::string activationFunction; // "relu" or "softmax"
  Eigen::MatrixXd W; // [out × in]
  Eigen::VectorXd b; // [out]
  Layer(int in, int out, const std::string& act="relu") { W=Random*0.5; b=Random*0.5; }
  std::pair<MatrixXd,MatrixXd> forward(const MatrixXd& input);
};
```

- **Only `Dense`:** `W` is `out×in`, `forward` is `W*input + b` then `act`. No `LayerType` tag — every layer is implicitly `Dense`. Cannot add `Conv1D` without `if` spaghetti.
- **String activation (`Layer.h:137`):** `if (activationFunction=="relu")` does heap allocation and linear compare in every forward. Typo `"ReLU"` silently falls to `else { activation=output; }` (no activation). No exhaustiveness check.
- **Backward bug (`Layer.cpp:72-75`):** for `softmax`, `dZ = (nextW^T * nextdZ) * (expZ/expZ.sum())` — element-wise multiply by softmax output is not `diag(s)-s s^T`. Works only for last-layer + cross-entropy by accident; will break for hidden `softmax` or new activations. Must fix and restrict `Softmax` to last layer.

### 2.2 `NeuralNetwork` — linear stack with interleaved Z/A

```cpp
// FlexNN/include/FlexNN.h:37
class NeuralNetwork {
  std::vector<Layer> layers;
  void train(const MatrixXd& input, const MatrixXd& target, double lr, int epochs);
  std::vector<MatrixXd> forward(const MatrixXd& input); // returns [input, Z0,A0, Z1,A1, ...]
};
```

- `forward()` pushes `input`, then for each layer pushes `Z` and `A` (`result.first`, `result.second`). `backward()` and `updateWeights()` assume this `2*layers.size()+1` layout and dense gradients (`dW = dZ * X^T / m`, `db = rowwiseMean(dZ)`).
- `train()` does `oneHotEncode(target)` → loop `forward→backward→update` with fixed LR, no batches, no validation split, no shuffling per epoch. `accuracy()` argmaxes columns. Weight init `Random *0.5` (`Layer.h:55`) is coarse but kept for v0.1.

### 2.3 `Utility` — CSV and split

`readCSV_XY` and `splitXY` work but have no seed control (`std::random_device` in `Utility.cpp:116` is nondeterministic). No model I/O — known limitation in `README.md`.

### 2.4 Build — simple CMake

```cmake
# FlexNN/CMakeLists.txt:1
find_package(Eigen3 REQUIRED)
add_library(FlexNN lib/FlexNN.cpp lib/Layer.cpp lib/Utility.cpp)
add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC FlexNN OpenMP::OpenMP_CXX Eigen3::Eigen)
```

Host-only, `-O3 -march=native`, `OpenMP` required, `Doxygen` optional. No `tests/` yet.

---

## 3. Design Principles

1. **Type-safe enums, not strings.** `enum class Activation : uint8_t` and `enum class LayerType : uint8_t` give `switch` exhaustiveness (`-Wswitch-enum`), 1-byte encoding in `model.bin`, and no heap. Keep a deprecated `Layer(int,int,const std::string&)` shim for one release.
2. **Tagged variant, not inheritance, for `Layer`.** Host training stays simple (Eigen, no vtable). For MCU we don't reuse these structs at all — the export is flat arrays. `class Layer { LayerType type_; Activation act_; DenseParams dense_; Conv1DParams conv1d_; Eigen::MatrixXd W; ... }` is enough for a linear stack. Rejected: `class Conv1DLayer : public Layer` (needs virtual `forward()` and complicates `ModelIO`).
3. **FlexNN writes, TinyForge validates.** `exportModel()` just serializes the requested `(type, act)`; it does not `#include "tinyforge/op_registry.hpp"`. Validation (`tinyforge::find()`) lives in `TinyForge/tools/compiler`. This keeps FlexNN standalone and reusable. Rejected: `exportModel()` calling TinyForge's registry (would couple FlexNN to TinyForge).
4. **Flat binary contract, not C header.** FlexNN's `ModelIO` writes `model.bin` (magic + version + CRC + per-layer header + blobs). TinyForge's compiler parses it. Flat binary is deterministic, `hexdump`able, 50 LOC parser, no JSON float-rounding, no FlatBuffers dep.

---

## 4. Activation Refactor — `enum class Activation`

**File:** `FlexNN/include/Layer.h` (breaking, major bump)

```cpp
// FlexNN/include/Layer.h
namespace FlexNN {
enum class Activation : uint8_t {
  None      = 0, // linear — pre-BN or debugging
  ReLU      = 1,
  LeakyReLU = 2, // alpha=0.01 fixed for v0.1
  Sigmoid   = 3,
  Tanh      = 4,
  Softmax   = 5  // only on last layer; TinyForge runtime will argmax logits
};
inline const char* to_string(Activation a) {
  switch(a){case Activation::ReLU:return "relu"; case Activation::LeakyReLU:return "leaky_relu"; case Activation::Sigmoid:return "sigmoid"; case Activation::Tanh:return "tanh"; case Activation::Softmax:return "softmax"; default:return "none";}
}
inline bool try_parse(std::string_view s, Activation& out);
}
```

**Why 1 byte and fixed Leaky alpha:** `model.bin` encodes activation as `uint8_t` (see §7), no per-layer `float` for alpha in v0.1. Keeps writer simple. Future `LeakyReLU` with per-layer alpha would add `aux_count` float.

**Migration:**
- Add `Layer(int in, int out, Activation act, LayerType type=LayerType::Dense)`; keep old `Layer(int,int,const std::string&)` as `[[deprecated]]` that calls `try_parse` then delegates.
- `lib/Layer.cpp:25` `forward()`/`backward()` become `switch(activation_)` — see §6 for formulas. `backward()` for `Softmax` is fixed: if last layer + cross-entropy, `dZ = A - Y_onehot` is done in `NeuralNetwork::backward()`, not in `Layer::backward()`; add `assert(act != Softmax || is_last)`.

---

## 5. Layer Type System — `enum class LayerType`

**New file:** `FlexNN/include/LayerTypes.hpp` (standalone, included by `FlexNN.h` and `ModelIO.hpp`)

```cpp
// FlexNN/include/LayerTypes.hpp
#pragma once
#include <cstdint>
namespace FlexNN {
enum class LayerType : uint8_t { Dense=0, Conv1D=1, BatchNorm1D=2, MaxPool1D=3, AvgPool1D=4 };

struct DenseParams { int inputSize, outputSize; }; // W[out×in], b[out]
struct Conv1DParams {
  int inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1;
  // W[outChannels][inChannels*kernelSize] row-major, b[outChannels]
  // in: [C_in × L_in], out: [C_out × L_out], L_out = (L_in+2*pad - dilation*(K-1)-1)/stride+1
};
struct Pool1DParams { int channels, kernelSize, stride, padding; };
struct BatchNormParams { int numFeatures; float epsilon=1e-5f; };
}
```

**Why `LayerType` lives in FlexNN:** FlexNN must be able to *construct* a `Conv1D` layer for training. TinyForge has a mirror `TinyForge/include/tinyforge/types.hpp` (copied, not included) for its runtime — the two are kept in sync via the `model.bin` version field and by bumping `FlexNN` SHA in `TinyForge`'s `FetchContent`.

**`Layer` becomes tagged variant:**

```cpp
class Layer {
  LayerType type_;
  Activation act_;
  int inputSize_, outputSize_; // flattened
  DenseParams dense_;
  Conv1DParams conv1d_;
  Eigen::MatrixXd W; Eigen::VectorXd b; // sized per type
public:
  Layer(int in, int out, Activation act=ReLU, LayerType t=Dense);
  Layer(Conv1DParams p, Activation act); // for Conv1D
  LayerType type() const; Activation activation() const;
  std::pair<MatrixXd,MatrixXd> forward(const MatrixXd& input) const; // dispatches on type_
};
```

---

## 6. New Layer Types

**Priority:** `Conv1D` → `BatchNorm1D` (folded at compile time by TinyForge, but FlexNN must support training with it) → `Pool1D`. Vibration/gesture (TinyForge 7.1/7.2) need temporal conv; pooling reduces length.

### 6.1 `Dense` (existing, baseline)

- **Forward:** `Z = W*X + b` (`W` out×in, `X` in×batch, broadcast `b`), then `act`.
- **Backward:** `dW = dZ * X^T / batch`, `db = rowMean(dZ)`, `dX = W^T * dZ`.
- **Export:** `W` row-major `float32`, `b` `float32`.

### 6.2 `Conv1D` (new, highest priority)

- **Forward (host):** For each sample, `oc` in `0..C_out-1`, `ol` in `0..L_out-1`:
  ```
  Z[oc][ol] = b[oc] + Σ_{ic} Σ_{k=0}^{K-1} W[oc][ic*K+k] * X_padded[ic][ol*stride - pad + k]
  ```
  zero-padded. Then `act`.
- **Backward (host only):** im2col: unfold `X` to `col [C_in*K × L_out]`, `dW = dZ * col^T`, `dX` via `col2im`.
- **Why before Conv2D:** sensor is 1D time series (`C_in=3` axes, `L_in=64..256`). `Conv1D` maps to `arm_convolve_s8` with `h=1` on TinyForge.

### 6.3 `BatchNorm1D` (training-time; TinyForge folds at compile time)

- **FlexNN training:** per-feature `gamma, beta, eps, runningMean/Var` (momentum 0.1). `y = gamma*(x-mean)/sqrt(var+eps)+beta` (train uses batch stats, eval uses running).
- **Export:** as its own `LayerType::BatchNorm1D` with `aux_count = 4*features` (`gamma, beta, mean, var`). TinyForge's `tools/compiler` will fold it into preceding `Dense`/`Conv1D` (`W_fold = W*gamma/sqrt(var+eps)`, `b_fold = (b-mean)*gamma/sqrt(var+eps)+beta`) and delete the BN layer. If no preceding linear, TinyForge rejects.

### 6.4 `Pool1D` (new, phase 2)

- **Params:** `kernel, stride, padding` (no weights). `Y[oc][ol] = max/avg_{k} X[oc][ol*stride - pad + k]`.

### 6.5 Activations (formulas for forward/backward in `lib/Layer.cpp`)

| Act | Forward `A=f(Z)` | Backward `dZ` | Notes |
|-----|------------------|---------------|-------|
| None | `A=Z` | `dA` | |
| ReLU | `max(0,Z)` | `dA*(Z>0)` | |
| LeakyReLU | `Z>0?Z:0.01*Z` | `Z>0?1:0.01` | |
| Sigmoid | `1/(1+exp(-Z))` clamped `Z∈[-15,15]` | `dA*A*(1-A)` | host `double`, TinyForge LUT |
| Tanh | `tanh(Z)` | `dA*(1-A^2)` | `std::tanh` |
| Softmax | stable col-wise `exp(Z-max)/sum` | fused `A-Y_onehot` if last+CE else `J=diag(s)-s s^T` | **only last layer** |

---

## 7. Model I/O — `exportModel` / `importModel` & Flat Binary Format

**New files:** `FlexNN/include/ModelIO.hpp`, `FlexNN/lib/ModelIO.cpp` (standalone, no TinyForge headers)

```cpp
// FlexNN/include/ModelIO.hpp
namespace FlexNN {
struct ExportOptions { int formatVersion=1; bool includeOptimizerState=false; };
struct Status { bool ok; std::string error; };
Status exportModel(const NeuralNetwork& net, const std::string& path, ExportOptions opts={});
Status importModel(NeuralNetwork& net, const std::string& path);
}
```

**Behavior:**
1. No registry check — just write whatever `(type, act)` the network contains. If TinyForge later rejects it, the error surfaces in `TinyForge`'s compiler, not here.
2. Open `ofstream(path, binary)`, write header + per-layer headers (52 B, packed, 8-byte aligned) + blobs at `weight_offset`/`bias_offset`/`aux_offset`, then `file_crc32`. All little-endian, `float32` IEEE. See Appendix A for layout.
3. `importModel` for host tests: read back and `EXPECT_NEAR` weights.

**Flat binary (shared contract with TinyForge, see `LLD_TINYFORGE.md` §5 for TinyForge's parser):**

```
0: 4 magic 0x54464E47 ('TFNG')
4: 2 version 0x0001
6: 2 header_len 10
8: 4 layer_count N
12:4 flags
16:4 header_crc32
# then N * 52 B LayerHeader:
0:1 type, 1:1 act, 2:2 dtype(f32), 4:4 in_dim, 8:4 out_dim, 12:4 w_cnt, 16:4 b_cnt, 20:4 aux_cnt, 24:4 reserved, 28:4 w_off, 32:4 b_off, 36:4 aux_off, 40:2 c_in, 42:2 c_out, 44:2 k, 46:2 stride, 48:2 pad, 50:2 dilation
# then blobs at offsets
# end: 4 file_crc32
```

**Why binary:** 50 LOC parser in `TinyForge`, deterministic, `hexdump`able, no JSON/Protobuf dep.

---

## 8. Utility & Training Improvements

- Keep `readCSV_XY`, `splitXY`; add `setRandomSeed(uint32_t)` for determinism (currently `random_device` nondeterministic).
- No batching/Adam in v0.1 — keep `train()` signature, add overload for `exportModel` path.

---

## 9. Build System — CMake

```cmake
# FlexNN/CMakeLists.txt — like before, plus ModelIO
find_package(Eigen3 REQUIRED)
add_library(FlexNN lib/FlexNN.cpp lib/Layer.cpp lib/Utility.cpp lib/ModelIO.cpp)
target_include_directories(FlexNN PUBLIC include)
find_package(OpenMP REQUIRED)
target_link_libraries(FlexNN PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE FlexNN)
# Future: add_subdirectory(tests) when tests/ exists
```

No `TinyForge` subdirectory, no `tinyforge/` headers. Host only.

---

## 10. Testing Strategy

`FlexNN/tests/` (new, GoogleTest):

- `activation_test` — forward/backward vs finite diff `eps=1e-5`.
- `dense_test` — vs naive loops.
- `conv1d_test` — brute force small `C_in=1..2, L=4..8, K=3`.
- `export_import_test` — train tiny net, `exportModel`, `importModel`, `EXPECT_NEAR`, and run `TinyForge`'s `tinyforge-compile` on the `model.bin` as an integration check (via `FetchContent` in `TinyForge`, not in `FlexNN`).

---

## 11. File Tree & Artifacts

```
FlexNN/
├── CMakeLists.txt              # add_library(FlexNN) + add_executable(main)
├── include/
│   ├── FlexNN.h
│   ├── Layer.h                 # now Activation enum, LayerType
│   ├── LayerTypes.hpp          # new
│   ├── ModelIO.hpp             # new
│   └── Utility.h
├── lib/
│   ├── FlexNN.cpp
│   ├── Layer.cpp               # switch(act) + Conv1D
│   ├── ModelIO.cpp             # new
│   └── Utility.cpp
├── src/main.cpp                # example MNIST, now uses Activation::ReLU
├── tests/                      # new
├── docs/ (ignored via .gitignore)
└── ... (Doxyfile, data/mnist-digit-recognition.csv)
```

---

## 12. Implementation Order

1. **Activation enum** (`Layer.h`, `Layer.cpp` switch) + keep string shim + `Softmax` fix + tests.
2. **`LayerTypes.hpp` + `Layer` variant** for `Dense` with new ctor + tests (still Dense only, but type-tagged).
3. **`Conv1D` forward/backward** in `FlexNN` + tests.
4. **`ModelIO` export/import** + flat binary writer + `importModel` + golden + `TinyForge` parser integration.
5. **`BatchNorm` training** + export as `BatchNorm1D` with aux; TinyForge will fold.
6. **`Pool1D`** + export.

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Softmax backward breaks existing `src/main.cpp` | Restrict `Softmax` to last layer, fuse `A-Y` in `NeuralNetwork::backward()`, keep string shim for one release |
| String ctor removal breaks users | Keep deprecated shim one version, `[[deprecated]]` |
| `ModelIO` version drift with TinyForge | `version` field + `header_crc32`; TinyForge checks `version` and fails with `re-export with newer FlexNN` |

---

## Appendix A: Layer Header Format (shared with TinyForge)

See §7 and `LLD_TINYFORGE.md` Appendix A. `FlexNN` writes, `TinyForge` reads. `FlexNN` never validates against TinyForge's registry.

## Appendix B: Example Export Flow

```cpp
#include "FlexNN.h"
#include "ModelIO.hpp"
FlexNN::NeuralNetwork net({
  FlexNN::Layer(192, 32, FlexNN::Activation::ReLU, FlexNN::LayerType::Dense),
  FlexNN::Layer(32, 1, FlexNN::Activation::Sigmoid)
});
net.train(X, Y, 0.5, 100);
auto st = FlexNN::exportModel(net, "model.bin");
if (!st.ok) std::cerr << st.error << "\n";
```

This `model.bin` is then consumed by `TinyForge` (see `LLD_TINYFORGE.md` §5).

