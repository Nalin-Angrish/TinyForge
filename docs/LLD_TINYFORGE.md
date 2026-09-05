# TinyForge — Low-Level Design (LLD_TINYFORGE) v1.0

> **Parent spec:** `docs/OVERVIEW.md` and `docs/LLD_FLEXNN.md` (FlexNN export contract). This is the complete LLD for **TinyForge**: how it loads FlexNN's `model.bin`, compiles it for an embedded target (STM32F407, Cortex-M4F), and runs it with interchangeable scalar / CMSIS-NN kernels. TinyForge **depends on FlexNN** via `FetchContent` for `ModelIO` format only; FlexNN never depends on TinyForge.

**Status:** Draft · **Target:** STM32F407VGT6 (168 MHz, FPU SP, DSP, 1 MB Flash, 192 KB RAM: 112 KB SRAM + 64 KB CCM) · **Toolchain:** `arm-none-eabi-gcc` + `cmake/toolchains/arm-cortex-m4f.cmake`, CMake 3.10+, CMSIS-NN 6.x, CMSIS-DSP · **Repo:** `Nalin-Angrish/TinyForge` (private, CMake library) · **Date:** 2026-09-05

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Design Principles](#2-design-principles)
3. [Repository & Build Topology](#3-repository--build-topology)
4. [Model Import Contract — FlexNN's Flat Binary](#4-model-import-contract--flexnns-flat-binary)
5. [TinyForge Compiler — `tools/compiler/tinyforge-compile`](#5-tinyforge-compiler--toolscompilertinyforge-compile)
6. [Quantization — int8 per-tensor, with CMSIS-NN requant](#6-quantization--int8-per-tensor-with-cmsis-nn-requant)
7. [Op-Support Registry — `include/tinyforge/op_registry.hpp`](#7-op-support-registry--includetinyforgeop_registryhpp)
8. [Runtime — Constraints & Memory Model](#8-runtime--constraints--memory-model)
9. [Backend Abstraction — Compile-Time Selection](#9-backend-abstraction--compile-time-selection)
10. [Kernels — Scalar Reference & CMSIS-NN Accelerated](#10-kernels--scalar-reference--cmsis-nn-accelerated)
11. [Activation Handling on Cortex-M4F](#11-activation-handling-on-cortex-m4f)
12. [Runtime API — `include/tinyforge/runtime.hpp`](#12-runtime-api--includetinyforgeruntimehpp)
13. [Platform Layer — Clock, DMA, Sensors, DSP](#13-platform-layer--clock-dma-sensors-dsp)
14. [End-to-End Pipeline](#14-end-to-end-pipeline)
15. [Target Applications](#15-target-applications)
16. [Benchmarking vs TFLite Micro](#16-benchmarking-vs-tflite-micro)
17. [Testing Strategy](#17-testing-strategy)
18. [File Tree & Artifacts](#18-file-tree--artifacts)
19. [Implementation Order](#19-implementation-order)
20. [Risks & Mitigations](#20-risks--mitigations)
21. [Appendix A: Generated Header Example](#appendix-a-generated-header-example)
22. [Appendix B: Cross Toolchain Example](#appendix-b-cross-toolchain-example)

---

## 1. Overview & Goals

**TinyForge is an embeddable library, not a framework.** You train in FlexNN on a PC (`FlexNN::exportModel("model.bin")`), you compile with TinyForge on the PC (`./build/tools/compiler/tinyforge-compile model.bin -o generated` → `generated/model_data.h`), you link `TinyForge::TinyForge` into your firmware (`#include "tinyforge/tinyforge.h"` + `#include "generated/model_data.h"` + `model.run(input, output)`), and you flash the STM32F407G-DISC1. No interpreter, no filesystem, no heap.

**Why not TFLite Micro:** TFLite Micro is a general interpreter (op resolver, flatbuffer parsing, dynamic `MicroAllocator` arena) that runs any graph. TinyForge is a **purpose-built, compile-time-baked** engine for one model and one backend. For the same quantized `Dense → ReLU → Dense` on the same board and same CMSIS-NN kernels, TinyForge should be **smaller in flash/RAM** (no interpreter) and similar in latency (same `SMLAD` MACs). That's the thesis to prove, not an assumed win.

**Success criteria:**
- `cmake -S . -B build && cmake --build build && ./build/main` runs on host with scalar backend and a dummy `model_data.h`.
- `cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake && cmake --build build-mcu` produces `libTinyForge.a` that links for `arm-none-eabi` and fits `<100 KB` flash for a 192→32→1 model.
- For each of 7.1/7.2/7.3 (see `OVERVIEW.md`), TinyForge scalar and TinyForge CMSIS-NN produce bitwise-equivalent outputs within `±1 LSB` on the same `model_data.h`.

---

## 2. Design Principles

1. **TinyForge depends on FlexNN, not vice versa.** FlexNN writes `model.bin` (see `LLD_FLEXNN.md` §7) and knows nothing about `tinyforge/`. TinyForge's `tools/compiler` includes FlexNN's headers from `FetchContent` *only* to parse `model.bin`. The op registry (`include/tinyforge/op_registry.hpp`) lives in TinyForge. Rejected: compiler in FlexNN (would make FlexNN depend on TinyForge's kernel coverage).

2. **Compile-time everything.** Backend (`scalar` vs `cmsis-nn`) and model are chosen at CMake configure time (`-DTINYFORGE_USE_CMSIS_NN=ON`, `generated/model_data.h`). The hot loop has no `virtual`, no `if (backend==)` branch. Each firmware image is one model, one backend, one arena.

3. **No heap, no exceptions, no RTTI, no `std::vector` in `lib/`/`include/`.** Host tools (`tools/compiler`) may use STL/Eigen freely. `lib/runtime.cpp` uses static `tinyforge_arena` and `internal` double buffering.

4. **Two backends, one golden reference.** Scalar is the oracle; CMSIS-NN is the accelerator. They must agree within `±1` after requant (see §10.4). Gate before benchmarking.

5. **Fail loudly on the host.** `tinyforge-compile` exits non-zero with `layer_index + supported list` on unknown op, bad `magic`, arena overflow. Runtime `Model::init()` re-checks the header and `static_assert`s `TINYFORGE_ARENA_BYTES`.

6. **Version everything.** `model.bin` has `magic + version + header_crc32`; `model_data.h` has `TINYFORGE_MODEL_VERSION`; `op_registry.hpp` has `version_added`. Mismatched SHA → clear error, not silent mis-inference.

---

## 3. Repository & Build Topology

```
TinyForge/  (this repo, private, CMake library)
├── CMakeLists.txt              # add_library(TinyForge) + add_subdirectory(tools/compiler)
├── include/tinyforge/          # public headers
│   ├── tinyforge.h             # aggregator
│   ├── types.hpp               # LayerType/Activation/Backend/QuantScheme/Status
│   ├── op_registry.hpp         # SINGLE SOURCE OF TRUTH
│   ├── runtime.hpp
│   ├── backend.hpp
│   ├── quant.hpp
│   └── kernels/{dense.hpp,conv1d.hpp,activations.hpp}
├── lib/                        # runtime sources
│   ├── runtime.cpp             # Model::init/run, static arena
│   ├── backend_scalar.cpp      # scalar kernels
│   ├── backend_cmsis.cpp       # CMSIS-NN wrappers (only with -DTINYFORGE_USE_CMSIS_NN=ON)
│   ├── quant.cpp
│   └── activations.cpp
├── tools/compiler/             # tinyforge-compile — HOST, links FlexNN from FetchContent
│   ├── CMakeLists.txt
│   ├── src/{main.cpp, parser.cpp, validate.cpp, quantize.cpp, codegen.cpp}
│   └── tests/
├── src/main.cpp                # example firmware (like FlexNN/src/main.cpp)
├── tests/test_runtime.cpp      # host tests (scalar vs CMSIS stub)
├── docs/{OVERVIEW.md, LLD_FLEXNN.md, LLD_TINYFORGE.md (this)}
├── generated/                  # model_data.h from tools/compiler (gitignored)
├── cmake/toolchains/arm-cortex-m4f.cmake
└── (FetchContent) FlexNN_src/  # fetched at configure if TINYFORGE_BUILD_COMPILER=ON
```

**Why this mirrors FlexNN (`FlexNN/include/`, `FlexNN/lib/`, `FlexNN/src/main.cpp`, `FlexNN/CMakeLists.txt`):** both are CMake libraries with `add_library` + `add_executable(main)`. TinyForge adds `tools/compiler` because its compiler is a host tool that happens to live in the same repo as the runtime, not because FlexNN needs it. TinyForge's runtime never includes `FlexNN.h`.

**Build — two distinct invocations:**

*Host (scalar, for `ctest`):*
```bash
cmake -S . -B build
cmake --build build -j
# → build/libTinyForge.a, build/main, build/tools/compiler/tinyforge-compile
ctest --test-dir build
```

*Cross (STM32F407, library only, for firmware):*
```bash
cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake
cmake --build build-mcu -j
# or: -DTINYFORGE_USE_CMSIS_NN=ON  → builds lib/backend_cmsis.cpp with CMSIS
arm-none-eabi-size build-mcu/libTinyForge.a
```

*Model generation (requires FlexNN headers, but not FlexNN's build for the runtime):*
```bash
# Train in FlexNN (standalone)
git clone https://github.com/Nalin-Angrish/FlexNN && cmake -S FlexNN -B FlexNN/build && cmake --build FlexNN/build
./FlexNN/build/main  # → model.bin (via FlexNN::exportModel)

# Compile with TinyForge's own compiler (fetches FlexNN at configure if needed)
cmake -S . -B build && cmake --build build -j
./build/tools/compiler/tinyforge-compile model.bin -o generated --backend cmsis --calib data/calib.csv
# → generated/model_data.h  (then firmware does #include "generated/model_data.h")
```

Top-level `CMakeLists.txt:1` handles `TINYFORGE_USE_CMSIS_NN`, `TINYFORGE_BUILD_COMPILER` (`FetchContent` for `FlexNN_src`), and `TINYFORGE_BUILD_TESTS`. No `platformio.ini`, no `build_src_filter`.

---

## 4. Model Import Contract — FlexNN's Flat Binary

TinyForge never trains. It **imports** FlexNN's `model.bin` whose format is defined in `LLD_FLEXNN.md` §7 and summarized here for self-containment. TinyForge's `tools/compiler/src/parser.cpp` is the only file that touches `model.bin`; the runtime only sees `model_data.h`.

**Flat binary, little-endian, versioned:**

```
0: 4 magic 0x54464E47 ('TFNG')
4: 2 version 0x0001
6: 2 header_len 10
8: 4 layer_count N
12:4 flags
16:4 header_crc32 (IEEE of 0..15)
# then N * 52 B LayerHeader (packed, 8-byte aligned):
# 0:1 type (FlexNN::LayerType), 1:1 act (FlexNN::Activation), 2:2 dtype (0=f32), 4:4 in_dim, 8:4 out_dim,
# 12:4 w_cnt, 16:4 b_cnt, 20:4 aux_cnt, 24:4 reserved, 28:4 w_off, 32:4 b_off, 36:4 aux_off,
# 40:2 c_in, 42:2 c_out, 44:2 k, 46:2 stride, 48:2 pad, 50:2 dilation
# then blobs at w_off/b_off/aux_off: w_cnt*4 bytes f32, b_cnt*4, aux_cnt*4 (BN: gamma/beta/mean/var)
# end: 4 file_crc32
```

**TinyForge's parser (`tools/compiler/src/parser.cpp`):** `read(fd, &header, 20)`, check `magic==TFNG`, `version==1`, `header_crc32`, then `N * 52` loop with `pread` at offsets, check `w_cnt == C_out*C_in*K` for `Conv1D` etc., check `file_crc32`. On any mismatch: exit `3` with `offset + expected`.

**Why not `FlexNN`-owned parser:** TinyForge's parser is a 50 LOC reader that copies `LayerHeader` struct from FlexNN's `ModelIO` — no Eigen, no `FlexNN.h` include in `lib/`. FlexNN's writer and TinyForge's reader are kept in sync by `version` and by bumping `FlexNN`'s `FetchContent` SHA in TinyForge when `FlexNN` changes `LayerHeader`.

---

## 5. TinyForge Compiler — `tools/compiler/tinyforge-compile`

**Location:** `TinyForge/tools/compiler` (not in `FlexNN`). **Dependency direction:** `TinyForge/tools/compiler` → `FlexNN_src` (via `FetchContent` for `ModelIO` structs + Eigen for calibration if needed). FlexNN never depends on TinyForge.

### 5.1 Responsibilities

- **Parse** `model.bin` (§4) → `Model { vector<LayerDesc> }` (`tools/compiler/include/model.hpp`).
- **Validate** every `(type, act, dims, quant)` against `TinyForge`'s `op_registry.hpp` (§7). Unknown `Conv1D+Tanh` → exit `2` with `layer 2: supported (Conv1D, ReLU), (Conv1D, None)`.
- **Validate** `inputDim>0`, `weight_count == C_out*C_in*K` etc., `layers[i].out_dim == layers[i+1].in_dim`, and **arena fit** (`max_arena = max_i (max(in_dim,out_dim)*1 + cmsis_temp)` where `cmsis_temp = 2*C_in*K` for `arm_convolve_s8`). If `max_arena > --arena` (default `98304` = 96 KB, leaves 32 KB for stack/audio), exit `2` with `needs 18.2 KB` hint.
- **Calibrate & quantize** (§6) — per-tensor affint `int8`, `scale/zp/multiplier/shift`, optional `BatchNorm` folding.
- **Codegen** `generated/model_data.h` — `static const int8_t weights_lN[]`, `int32_t bias_lN[]`, `QuantParams`, `tinyforge_layer_desc_t layers[]`.

**Not responsible:** training (FlexNN), linking/flashing (firmware CMake).

### 5.2 CLI

```
tinyforge-compile --help
Usage: tinyforge-compile <model.bin> [options]
  <model.bin> from FlexNN::exportModel
  -o, --output <dir> (default ./generated)
  --backend scalar|cmsis (registry check)
  --check-only (validate only, exit 0 if valid)
  --calib <csv> --calib-samples 500
  --arena <bytes> (default 98304)
  --no-quant (emit f32)
  --fold-bn (default on)
  --verbose
```

Exit: `0` ok, `2` validation, `3` parse, `4` calib, `5` codegen.

### 5.3 Calibration

When `--calib` given: load CSV via `FlexNN::readCSV_XY` logic (linked from `FlexNN_src` for reuse) or via a tiny Eigen-free forward that reuses `FlexNN`'s `forward` (now available via `FetchContent`). For `N = min(calib_samples, rows)`, `batch=1`, record per-layer `A` `min/max`, compute `scale=(max-min)/255`, `zp=clamp(round(-min/scale), -128,127)`. Weight scale: `scale_w = max|W|/127` (symmetric, `zp=0`). If no calib but quant requested, use weight range + heuristic `[-6,6]` for `Tanh`/`Sigmoid`.

### 5.4 Codegen

```c
// generated/model_data.h (see Appendix A)
#pragma once
#include <stdint.h>
#define TINYFORGE_MODEL_VERSION 1
#define TINYFORGE_LAYER_COUNT 2
#define TINYFORGE_ARENA_BYTES 400
static const float tinyforge_scale_l0_in = 0.05f;
static const int32_t tinyforge_zp_l0_in = 0;
static const int32_t tinyforge_mult_l0 = 1073741824; // Q31
static const int tinyforge_shift_l0 = 3;
static const int8_t tinyforge_weights_l0[12] __attribute__((aligned(4))) = {12,-3,...};
static const int32_t tinyforge_bias_l0[3] = {123,-45,...};
typedef struct { uint8_t type, act; uint16_t in_dim, out_dim; } tinyforge_layer_desc_t;
static const tinyforge_layer_desc_t tinyforge_layers[2] = {{0,1,4,3},{0,5,3,2}};
```

Header is `const` → `.rodata` in flash, `aligned(4)` for DMA. Includes `// Generated by tinyforge-compile v0.4.0 (FlexNN SHA ...)`.

---

## 6. Quantization — int8 per-tensor, with CMSIS-NN requant

**Math (TFLite-style, see `LLD_FLEXNN.md` for FlexNN's `f32`):**

```
real = scale * (q - zp)
q = clamp(round(real/scale + zp), -128,127)
acc = Σ (in_q - zp_in)*(w_q - zp_w) + bias_q,  bias_q = round(b_float / (scale_in*scale_w))
out_q = requantize(acc, mult, shift) + zp_out,  clamp
effective_scale = scale_in*scale_w / scale_out
mult = frexp(effective_scale) → Q31, shift = 31 - exp
```

CMSIS expects `arm_nn_requantize(acc, mult, shift)` with `Q31` `mult` and `shift` as above.

**TinyForge choices:**

- **Per-tensor, affine for activations, symmetric for weights (`zp_w=0`).** One `scale`/`zp` per tensor → one `mult`/`shift` per layer, simple `arm_fully_connected_s8` `input_offset`/`output_offset`. Per-channel (`C_out` scales) is future `QuantScheme::Int8PerChannel` with `C_out` `mult` array.
- **Calibration:** per-layer `A` `min/max` from `model.bin` + calib CSV (see §5.3). `ReLU` `min≥0` → `zp=0`; `Tanh` `[-1,1]` → `scale≈0.0078`.
- **BatchNorm folding (compiler, before quant):** if `BN` follows `Dense`/`Conv1D`, `W_fold = W*gamma/sqrt(var+eps)`, `b_fold = (b-mean)*gamma/sqrt(var+eps)+beta`, delete BN layer. Zero runtime cost; if BN without preceding linear, reject.
- **Float fallback:** `--no-quant` → `float` arrays and `lib/backend_*` float path (for `Softmax` or accuracy baseline). `TINYFORGE_USE_FLOAT` selects.

**Size example:** `Dense 128→64` `f32` `32 KB` weights → `int8` `8 KB` + `256 B` bias `int32` + `200 B` header = `75%` saving, target `<1%` accuracy drop.

---

## 7. Op-Support Registry — `include/tinyforge/op_registry.hpp`

**Owned by TinyForge, not FlexNN.** Both `tools/compiler` and `lib/runtime.cpp` include the same header.

```cpp
// TinyForge/include/tinyforge/op_registry.hpp
#pragma once
#include "tinyforge/types.hpp"
namespace tinyforge {
enum class QuantScheme { Float32, Int8PerTensor, Int8PerChannel };
struct OpEntry { LayerType type; Activation act; QuantScheme quant; Backend backend; const char* sym; uint8_t ver; };
static constexpr OpEntry kRegistry[] = {
  {LayerType::Dense, Activation::ReLU,      QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_relu_s8", 1},
  {LayerType::Dense, Activation::None,      QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_none_s8", 1},
  {LayerType::Dense, Activation::Sigmoid,   QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_sigmoid_s8", 1},
  {LayerType::Dense, Activation::Tanh,      QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_tanh_s8", 1},
  {LayerType::Dense, Activation::LeakyReLU, QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_lrelu_s8", 1},
  {LayerType::Conv1D, Activation::ReLU,     QuantScheme::Int8PerTensor, Backend::Scalar,  "conv1d_relu_s8", 1},
  {LayerType::Conv1D, Activation::None,     QuantScheme::Int8PerTensor, Backend::Scalar,  "conv1d_none_s8", 1},
  {LayerType::Dense,  Activation::ReLU,     QuantScheme::Int8PerTensor, Backend::CmsisNN, "dense_relu_s8_cmsis", 1},
  {LayerType::Conv1D, Activation::ReLU,     QuantScheme::Int8PerTensor, Backend::CmsisNN, "conv1d_relu_s8_cmsis", 1},
};
inline const OpEntry* find(LayerType t, Activation a, QuantScheme q, Backend b) { /* loop kRegistry */ }
}
```

**Why single header, not generated:** small table (≤10 entries), `static_assert` on `kRegistrySize` vs kernel `switch` catches drift. Future `extract_registry.py` can grep `TINYFORGE_REGISTER_OP`.

**TinyForge runtime check:**

```cpp
#include "tinyforge/op_registry.hpp"
Status Model::init() {
  for (auto &l : tinyforge_layers)
    if (!find((LayerType)l.type,(Activation)l.act,QuantScheme::Int8PerTensor,kBackend))
      return ErrUnsupportedOp;
}
```

---

## 8. Runtime — Constraints & Memory Model

**MCU (STM32F407VGT6):** `Cortex-M4F` `168 MHz`, `FPU SP` `FPv4-SP-D16`, `DSP` `SMLAD`, `1 MB` flash `.rodata` weights, `192 KB` RAM (`0x20000000` `112 KB` SRAM `DMA-capable` + `0x10000000` `64 KB` CCM `not DMA` + `4 KB` backup). `M4F` `FPU` is enabled via `SCB->CPACR`, `DWT->CYCCNT` for latency.

**No heap invariant:** `lib/` has no `new`/`malloc`/`std::vector`. `CMake` adds `-Wl,--wrap=malloc` and `tests` does `nm` check.

**Static arena, double-buffered (linear stack):**

```
arena[2*h] : ping-pong, no memcpy
for layer i: input = arena[parity], output = arena[1-parity]; swap parity
```

- **Size:** `TinyForge`'s compiler computes `max_arena = max_i (max(in_dim,out_dim)*1 + cmsis_temp)` where `cmsis_temp = 2*C_in*K` for `arm_convolve_s8`. Example `128→64→10` `int8` → `128 B` → `<1 KB`. `Conv C_in=3,L=100→C_out=8,K=5,stride=2` → `in 300, out 400` → `400 B`.
- **Placement:** `__attribute__((section(".ccmram"))) static int8_t arena[TINYFORGE_ARENA_BYTES];` (CCM, faster, zero wait). If `>64 KB`, spill to `section(".ram")` (SRAM). `static_assert(TINYFORGE_ARENA_BYTES >= TINYFORGE_REQUIRED_ARENA)`.
- **Why not `Layer<In,Out>` templates:** would need `constexpr In,Out` per layer, code bloat, no sharing. Arena is one `Model` class for any model that fits.

---

## 9. Backend Abstraction — Compile-Time Selection

```cpp
// include/tinyforge/backend.hpp
#pragma once
#include "tinyforge/types.hpp"
#ifndef TINYFORGE_BACKEND
#define TINYFORGE_BACKEND 0 // 0 scalar (host), 1 cmsis-nn
#endif
namespace tinyforge {
#if TINYFORGE_BACKEND==1
constexpr Backend kBackend = Backend::CmsisNN;
#else
constexpr Backend kBackend = Backend::Scalar;
#endif
Status run_dense_s8(const int8_t* in, const int8_t* w, const int32_t* b, int8_t* out, int in_dim, int out_dim, const QuantParams& qin, const QuantParams& qw, const QuantParams& qout, Activation act);
Status run_conv1d_s8(const int8_t* in, const int8_t* w, const int32_t* b, int8_t* out, const Conv1DParams& p, const QuantParams& qin, const QuantParams& qw, const QuantParams& qout, Activation act);
}
```

`lib/backend_scalar.cpp` and `lib/backend_cmsis.cpp` implement the same symbols; `CMake` compiles only one per `TINYFORGE_USE_CMSIS_NN` (`build_src_filter` not needed — just `if` in `CMakeLists.txt`). No `virtual`; `if constexpr` + `template` gives zero-cost DCE.

**QuantParams per layer (from `generated/model_data.h`):**

```cpp
struct QuantParams { float scale; int32_t zp; int32_t mult; int shift; };
// per layer: qin, qw, qout
```

---

## 10. Kernels — Scalar Reference & CMSIS-NN Accelerated

### 10.1 Dense

**Float (baseline):**

```cpp
for(o) { float acc=b[o]; for(i) acc+= w[o*in+i]*in[i]; out[o]=activate_f32(acc, act); }
```

**Int8 scalar (`lib/backend_scalar.cpp`):**

```cpp
for(o){ int32_t acc=bias[o]; for(i) acc+=(in[i]-zp_in)*(w[o*in+i]-zp_w); acc=requantize(acc, mult, shift)+zp_out; acc=clamp(acc,-128,127); if(act==ReLU && acc<zp_out) acc=zp_out; out[o]=acc; }
```

**CMSIS-NN (`lib/backend_cmsis.cpp`):**

```cpp
#include "arm_nnfunctions.h"
cmsis_nn_context ctx{nullptr,0}; // TinyForge's compiler ensures ctx.buf = arena_temp if needed
cmsis_nn_fc_params fc{input_offset=-zp_in, filter_offset=0, output_offset=zp_out, activation{ -128, 127 }};
arm_fully_connected_s8(&ctx,&fc,&quant,&in_dims,in,&filter_dims,w,&bias_dims,bias,&out_dims,out);
```

`arm_fully_connected_s8` uses `SMLAD` (4× `int8` MAC/cycle). `bias` is `int32` with scale `scale_in*scale_w`.

### 10.2 Conv1D → `arm_convolve_s8` with `h=1`

**Host naive scalar:** triple loop `oc→ol→ic,k` with `int32 acc`, as in `LLD_FLEXNN.md` §6.2.

**CMSIS mapping (generic, works even without `arm_convolve_1d_s8`):**

```cpp
arm_convolve_wrapper_s8(&ctx, &conv_params{c, stride, pad}, &quant,
  /*in_dims*/{1, L_in, C_in}, in,
  /*filter_dims*/{C_out,1,K,C_in}, w,
  /*bias_dims*/{C_out}, bias,
  /*out_dims*/{1, L_out, C_out}, out);
```

`h=1` collapses 2D to 1D; `wrapper` picks fast `1x1`/`3x3` paths. `ctx.buf` needs `2*C_in*K` `int16` — compiler adds to `max_arena`.

### 10.3 Pool1D (future)

Scalar `max/avg_{k} in[oc][ol*stride - pad + k]`; CMSIS `arm_max_pool_s8` / `arm_avgpool_s8` with `h=1`.

### 10.4 Correctness Gate (must pass before benchmarking)

Host loop (also `tests/test_runtime.cpp`):

```
for random in [-128,127], w:
  out_scalar = dense_s8_scalar(...)
  out_cmsis  = dense_s8_cmsis(...) // via TinyForge's compiler host stub or UART dump
  assert max_abs(out_scalar - out_cmsis) <= 1
```

`±1 LSB` for `round` vs `NN_ROUND`.

---

## 11. Activation Handling on Cortex-M4F

- **`ReLU` int8:** `max(q, zp)` (`zp=0` for symmetric). Scalar `if(q<zp) q=zp;`.
- **`LeakyReLU` int8:** `y = q>=zp ? q : zp + (q-zp)*alpha`, `alpha=0.01≈13/1024` → `((q-zp)*13>>10)`. Restrict to `Scalar` in v0.1 if CMSIS lacks it (registry will mark `CmsisNN` unsupported).
- **`Sigmoid`/`Tanh` (see `OVERVIEW.md` TBD):** `M4F` has `FPU`, so `expf`/`tanhf` work but `~50` cycles each. `64` neurons → `3200` cycles, okay for 1 Hz but not 50 Hz KWS. TinyForge compiler emits a **256-entry `int8` LUT** for `Sigmoid` over `[-8,8]` (and `Tanh` over `[-4,4]`) as `static const int8_t lut[256]` in `model_data.h`. `Float` runtime uses `1/(1+expf(-x))` / `tanhf(x)`; `CMSIS` uses `arm_sigmoid_s8` if `CMSIS-NN 6.x` provides it, else LUT. LUT error `<1%` vs `expf`.
- **`Softmax` on MCU:** never computed — **argmax logits**. `Softmax` is monotonic, so `argmax(softmax(logits)) == argmax(logits)`. Saves `exp` entirely. Host can compute `Softmax` for UART logging.

---

## 12. Runtime API — `include/tinyforge/runtime.hpp`

```cpp
#pragma once
#include <stdint.h>
#include "tinyforge/types.hpp"
namespace tinyforge {
enum class Status : int32_t { Ok=0, ErrBadDims=-1, ErrArenaTooSmall=-2, ErrUnsupportedOp=-3, ErrNullPtr=-4, ErrBadModel=-5 };
class Model {
  bool initialized=false; int8_t *arena0=nullptr, *arena1=nullptr;
public:
  Status init(); // checks TINYFORGE_LAYER_COUNT<=16, registry, arena
  Status run(const int8_t* in, int8_t* out); // int8 logits
  Status run(const float* in, float* out);   // float fallback
  int input_dim() const; int output_dim() const; int layer_count() const;
};
}
```

`Model::init()` checks `TINYFORGE_LAYER_COUNT` from `generated/model_data.h`, loops `tinyforge_layers[]` via `op_registry.hpp`, and `static_assert`s arena. `run()` double-buffers through `arena0/1`, calls `run_dense_s8`/`run_conv1d_s8`, returns `Status` (never `assert`/`exit`). Caller owns `input`/`output` (`192` for `64×3` accel window). No `new`.

---

## 13. Platform Layer — Clock, DMA, Sensors, DSP

- **Clock:** `SystemClock_Config()` `HSE 8 MHz` → `PLL 168 MHz` `FLASH_LATENCY 5`, `AHB 168`, `APB1 42`, `APB2 84`, `SCB->CPACR` FPU enable, `DWT->CTRL` cycle counter for `latency = CYCCNT`.
- **UART:** `USART2` `115200` via ST-Link VCP, `HAL_UART_Transmit` for logs, DMA for bulk test vectors.
- **Accel (LIS3DSH/LIS302DL):** `SPI1` `CS PE3`, probe `WHO_AM_I` (`0x3F` vs `0x3B`), poll `100 Hz` `6 B` `X/Y/Z`, window `64..128` samples → `in_dim 192..384`.
- **Mic (MP45DT02 PDM → PCM):** `I2S2` `DMA1_Stream3` `2 MHz` PDM, `libPDMFilter`/`CMSIS` decimation, double `int16_t mic_buf[2][1024]` in `SRAM1` (CCM is not DMA-capable), `16 kHz` mono, `MFCC` via `CMSIS-DSP` `arm_mfcc_init`/`arm_rfft_fast_f32` `25 ms` window `160` hop `13` coeffs → `130` dim (`13*10`).
- **DSP contract:** `mfcc_compute(pcm, input_f32)` → `quantize` → `model.run()`.

---

## 14. End-to-End Pipeline

```bash
# 1) FlexNN (standalone) — train + export
git clone https://github.com/Nalin-Angrish/FlexNN && cmake -S FlexNN -B FlexNN/build && cmake --build FlexNN/build
./FlexNN/build/main  # → model.bin

# 2) TinyForge (depends on FlexNN via FetchContent) — compile + run
git clone https://github.com/Nalin-Angrish/TinyForge && cd TinyForge
cmake -S . -B build && cmake --build build -j
# → build/libTinyForge.a, build/main, build/tools/compiler/tinyforge-compile
./build/tools/compiler/tinyforge-compile model.bin -o generated --backend cmsis --calib data/calib.csv
# → generated/model_data.h

# 3) Tanner's firmware (or TinyForge's src/main.cpp) — link TinyForge
cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-cortex-m4f.cmake
cmake --build build-mcu -j
# flash via ST-Link / OpenOCD

# 4) Validate
# UART: "init ok, arena 400B, latency 12450 cycles, output [12, -3, ...]"
```

---

## 15. Target Applications

- **7.1 Vibration anomaly (binary, accel):** `window 64×3=192` `f32` `[-1,1]` → `Dense 192→32 ReLU → 32→1 Sigmoid` (`Sigmoid` = threshold `0.5`). `float` `25 KB` → `int8` `6 KB`. `accel_read` → `run` → `anomaly`.
- **7.2 Gesture (multi-class, accel):** same window → `Dense 32→6` `Softmax` (argmax on MCU). Needs UART dump → PC labeling → FlexNN training.
- **7.3 KWS (mic + MFCC):** `pcm[400]` `25 ms` → `MFCC 13` × `10` frames `130` → `Dense 130→64 ReLU → 64→12 Softmax` (subset `SpeechCommands`). Fallback `VAD` `130→32→1` if `MFCC` too heavy.

---

## 16. Benchmarking vs TFLite Micro

Per `OVERVIEW.md` §8, three firmware variants per app, separately flashed: `tinyforge_scalar`, `tinyforge_cmsis`, `tflm_cmsis` (with `CMSIS-NN` kernels, not reference). Same `model.bin` → same `test vectors` via UART.

- **Latency:** `DWT->CYCCNT` around `run()`, `mean/median/p95` over `1000`, `ms = cycles/168M`.
- **RAM:** `size -B` `.data+.bss+arena` (+ `tensor_arena` for `TFLM`, `0xAA` paint for stack high-water).
- **Flash:** `size` `.text+.rodata` split `weights vs code` via `nm`.
- **Accuracy:** same test set `500` on PC `f32` vs MCU `int8`.

**Expected:** latency similar (same `SMLAD`), **flash/RAM smaller** for TinyForge (no interpreter/FlatBuffer, compile-time graph). Report honestly.

---

## 17. Testing Strategy

- **Host (TinyForge, `tests/`):** `test_runtime_dense` scalar vs float, `test_runtime_conv` scalar vs float, `test_arena` `static_assert`. Run `ctest --test-dir build`.
- **Compiler (`tools/compiler/tests/`):** `parser` (bad magic), `validate` (unsupported op → `2`), `quant` (golden), `codegen` (compile `model_data.h` with `arm-none-eabi-gcc`).
- **Parity:** same `input.bin` on PC `f32` (FlexNN reference) and MCU `int8` (TinyForge) via UART, `≤1 LSB`.
- **CI (TinyForge):** `cmake -S . -B build && ctest` (host) and `cmake -S . -B build-mcu -DCMAKE_TOOLCHAIN_FILE=... && cmake --build build-mcu` (cross, `nm` checks `arm_fully_connected_s8` when `CMSIS`).

---

## 18. File Tree & Artifacts

```
TinyForge/
├── CMakeLists.txt              # add_library(TinyForge) + add_subdirectory(tools/compiler)
├── include/tinyforge/
│   ├── tinyforge.h
│   ├── types.hpp / runtime.hpp / backend.hpp / quant.hpp / op_registry.hpp
│   └── kernels/{dense.hpp,conv1d.hpp,activations.hpp}
├── lib/
│   ├── runtime.cpp / backend_scalar.cpp / quant.cpp / activations.cpp
│   └── backend_cmsis.cpp
├── tools/compiler/
│   ├── CMakeLists.txt
│   ├── src/{main.cpp, parser.cpp, validate.cpp, quantize.cpp, codegen.cpp}
│   └── tests/
├── src/main.cpp                # example firmware (like FlexNN/src/main.cpp)
├── tests/test_runtime.cpp
├── docs/{OVERVIEW.md, LLD_FLEXNN.md, LLD_TINYFORGE.md (this)}
├── generated/                  # model_data.h from tools/compiler (gitignored)
├── cmake/toolchains/arm-cortex-m4f.cmake
└── (FetchContent) FlexNN_src/  # fetched only for tools/compiler at configure
```

`generated/model_data.h` example: see Appendix A.

---

## 19. Implementation Order

**Phase 0 (done):** `TinyForge` `v0.3` CMake library (`include/lib/src/main.cpp`) + `docs/` + `FetchContent` for `FlexNN`.

**Phase 1 — Minimal E2E (Dense+ReLU):** `TinyForge` `Dense` scalar → `CMSIS` `arm_fully_connected_s8` → `op_registry` `Dense+ReLU` → `tools/compiler` (parser/validate/quant/codegen) → `Model::run` Dense → `src/main.cpp` 7.1 `192→32→1`. Gate: `test_runtime_dense` `±1`.

**Phase 2 — Activations:** `LeakyReLU,Sigmoid,Tanh` (`FlexNN` forward + `TinyForge` LUT/CMS-S kernels + `tools/compiler` quant ranges) → registry.

**Phase 3 — Conv1D:** scalar `Conv` → `arm_convolve_s8` → registry `Conv+ReLU` → `FlexNN` `Conv1D` (for training) → `TinyForge` compiler `Conv` dims + arena `temp` → test.

**Phase 4 — BN folding + Pool:** `tools/compiler` folding pass, `Pool` kernels.

**Phase 5 — Drivers & Apps 7.2/7.3:** `SPI` accel, `I2S` mic, `MFCC`.

**Phase 6 — TFLM baseline.**

*No `FlexNN` export is “locked until TinyForge kernel exists — `TinyForge`'s `tinyforge-compile` is the gate, not `FlexNN`.*

---

## 20. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Arena >64 KB CCM | `TinyForge`'s compiler checks `--arena`; spill to `SRAM` via `section(".ram")` |
| CMSIS-NN API 5 vs 6 | Pin `CMSIS-NN@1.3.0` via vendoring, `#if ARM_CMSIS_NN_VERSION` wrapper |
| LUT <1% accuracy | Fallback to `expf` float, or 512 LUT + interp (compiler emits larger) |
| DMA in CCM hard-fault | `static_assert` `mic_buf` in `0x20000000` |
| TFLM unfair (no CMSIS) | `tflm_cmsis` explicitly `CMSIS_NN=1`, `nm` checks `arm_fully_connected_s8` |
| Registry drift (TinyForge vs FlexNN) | Registry lives only in `TinyForge`; `FlexNN` has no registry. TinyForge pins `FlexNN` SHA for `model.bin` format, not for registry. |

---

## Appendix A: Generated Header Example

```c
// generated/model_data.h — from TinyForge/tools/compiler
#pragma once
#include <stdint.h>
#define TINYFORGE_MODEL_VERSION 1
#define TINYFORGE_LAYER_COUNT 2
#define TINYFORGE_ARENA_BYTES 400
static const float tinyforge_scale_l0_in = 0.05f;
static const int32_t tinyforge_zp_l0_in = 0;
static const int32_t tinyforge_mult_l0 = 1073741824; // Q31
static const int tinyforge_shift_l0 = 3;
static const int8_t tinyforge_weights_l0[12] __attribute__((aligned(4))) = {12,-3,5,8,-10,2,6,4,-1,9,0,-5};
static const int32_t tinyforge_bias_l0[3] = {123,-45,67};
typedef struct { uint8_t type, act; uint16_t in_dim, out_dim; } tinyforge_layer_desc_t;
static const tinyforge_layer_desc_t tinyforge_layers[2] = {{0,1,4,3},{0,5,3,2}};
```

---

## Appendix B: Cross Toolchain Example

```cmake
# cmake/toolchains/arm-cortex-m4f.cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
find_program(ARM_CC arm-none-eabi-gcc)
set(CMAKE_C_COMPILER ${ARM_CC})
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--gc-sections -Wl,--wrap=malloc")
```

