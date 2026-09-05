# TinyForge — Technical Specification (Draft v0.1)

*Project name: TinyForge.*

## 1. Overview

TinyForge is a two-part system:

1. **FlexNN** (existing, PC-side): a from-scratch neural network training
   library. Extended as part of this project to support additional layer
   types/activations and to export trained models to a portable format.
2. **TinyForge runtime** (new, this project): a compiler + inference engine
   that ingests FlexNN-exported models, statically validates that every
   operation is supported, and runs inference on a resource-constrained
   embedded target (STM32F407, Cortex-M4F) with two interchangeable
   compute backends: a portable scalar reference and a CMSIS-NN
   SIMD-accelerated backend.

**Core narrative:** train on desktop with FlexNN → compile/validate →
quantize → deploy → run on-device, with a purpose-built engine that is
smaller and leaner than a general-purpose interpreter framework
(TensorFlow Lite Micro) for the same model, same accuracy.

**Explicitly not in scope:** generic multi-framework model import (PyTorch,
ONNX, Keras, etc.). FlexNN is the only supported source format. This is a
deliberate scope decision, not an oversight — see Section 9.

## 2. Goals

- Demonstrate a full model lifecycle: train (FlexNN) → export → compile-time
  validate → quantize → deploy → run on real embedded hardware.
- Support two compute backends behind one interface: portable scalar C++
  reference, and CMSIS-NN accelerated (SIMD int8 MACs on Cortex-M4F).
- Benchmark against TensorFlow Lite Micro (with its own CMSIS-NN kernel
  backend enabled) on the same model, same board, same test inputs —
  latency, RAM footprint, flash footprint, accuracy.
- Ship three working target applications (Section 7) proving the pipeline
  end-to-end at increasing complexity.
- Produce a compiler that fails loudly and specifically at compile time
  (on the dev machine) if a model uses an unsupported op — never silently
  produce broken firmware.

## 3. Non-Goals

- No generic model import (PyTorch/.h5/ONNX). FlexNN-only ingestion.
- No on-device training or online learning. Inference only; any RL policy
  (Section 7.4) is trained off-device and deployed for inference-only.
- No dynamic memory allocation anywhere in the inference path. No
  `malloc`/`new` at runtime.
- No recurrent layers (LSTM/GRU) or attention mechanisms. Static,
  compile-time-sized buffers only — recurrence/attention breaks this
  assumption and is out of scope for this project.
- No ESP32/Xtensa backend in this version. Architecture should not
  preclude it later, but it is not being built now.
- No multi-model runtime (loading/swapping models at runtime). Each
  firmware image is compiled for exactly one model.

## 4. System Architecture

```
[FlexNN, PC, Eigen-based]
  train model
  |
  v
[Model export] -- new addition to FlexNN
  serialize architecture + weights + activations to flat binary
  |
  v
[TinyForge Compiler, PC, CLI tool]
  parse exported model
  validate every op against supported-op registry
  quantize (int8, per-tensor scale + zero-point, calibrated)
  emit C header: weights as const arrays + architecture metadata
  |
  v
[TinyForge Runtime, on-device, C++, no heap]
  Portable core (scalar reference)      <-> Backend interface
  Cortex-M CMSIS-NN backend (SIMD)      <-> Backend interface
  |
  v
[STM32F407G-DISC1 firmware image]
```

Backend selection is a **compile-time** CMake option, not a runtime branch.
Each firmware image targets exactly one backend and one model.

## 5. Components

### 5.1 FlexNN extensions (PC side)

- Refactor `Layer`'s activation field from `std::string` comparison to
  `enum class Activation`. Existing values: `ReLU`, `Softmax`. To add:
  `Tanh`, `Sigmoid`, `LeakyReLU`.
- New layer types to add (priority order): `Conv1D`, `BatchNorm` (or
  fixed-scale/shift equivalent), `MaxPool1D` / `AvgPool1D`.
- `NeuralNetwork::exportModel(path)`: serializes, per layer — layer type,
  input/output dimensions, activation, weights (`W`), biases (`b`) — to a
  flat binary format. Format TBD (Section 6), but must be simple to parse
  from a separate, dependency-free tool.
- Layer support must be added bottom-up: scalar kernel implemented and
  tested in TinyForge runtime first, then CMSIS-NN kernel, then unlocked in
  FlexNN's training side and the compiler's op-registry allow-list
  together. FlexNN must never support (for export purposes) a layer the
  runtime cannot execute.

### 5.2 TinyForge Compiler (PC side, new CLI tool)

- Separate small binary, e.g. `tinyforge-compile model.bin --backend=cmsis
  --check`. Not folded into FlexNN's export function — validation logic
  stays independent of serialization logic so the op-registry can be
  extended without touching export code.
- **Op-support registry**: explicit table of supported
  `{layer_type, activation, quantization_scheme}` combinations. Single
  source of truth, referenced by both compiler validation and (ideally)
  generated/checked against the actual backend kernel implementations, so
  the registry cannot silently drift from what the runtime can execute.
- **Validation checks:**
  - Every layer's `(type, activation)` pair exists in the registry.
  - Every layer's declared input/output dimensions fit the static
    buffer/arena size the target build is configured for.
  - Fails with a specific, actionable error identifying the offending
    layer and reason — never silently emits output on failure.
- **Quantization pass:**
  - Post-training int8 quantization. Per-tensor scale + zero-point for
    weights and activations.
  - Calibration: run forward passes over a representative dataset sample
    supplied alongside the model, record activation ranges per layer.
  - Where applicable: fold BatchNorm into the preceding Conv1D/Dense layer's
    weights at this stage (compile-time fusion, zero runtime cost).
- **Output:** a generated C header (`model_data.h` or similar) containing
  `static const int8_t weights_lN[] = {...}`, quantization params, and
  architecture metadata (layer sizes, sequence of ops) as compile-time
  constants — no filesystem access needed on-device.

### 5.3 TinyForge Runtime (on-device)

- Pure C++, target Cortex-M4F, no STL heap containers, no exceptions,
  no RTTI, no dynamic allocation in the inference path.
- Fixed-size layers via templates (`Layer<InSize, OutSize>`) or a
  preallocated static arena sized at compile time from model metadata.
- `float` for the non-quantized reference path (Phase 2 equivalent);
  `int8` accumulate-in-`int32` for the quantized path.
- Forward pass only. No backward pass, no weight updates — this engine
  has one job.
- **Backend interface:** a common API (e.g. `run_layer(LayerType, ...)`)
  implemented twice:
  - **Portable scalar backend**: plain C++ loops, compiles anywhere,
    correctness baseline.
  - **CMSIS-NN backend**: same interface, inner loops replaced with
    CMSIS-NN calls (`arm_fully_connected_s8`, `arm_convolve_*_s8`, etc.)
    for SIMD int8 MACs via Cortex-M4F DSP instructions.
- Backends must produce matching outputs (within int8 rounding tolerance)
  on identical input — this is the correctness gate before any
  benchmarking is meaningful.

## 6. Model Export Format (TBD — decide before implementation)

Options to choose between:
- **Flat binary format** (custom): compact, requires a small parser in
  the compiler tool. Straightforward versioning via a header/magic number.
- **Generated C header directly from FlexNN** (skip an intermediate binary):
  simpler pipeline (one less format to maintain), but couples FlexNN's
  export code more tightly to the compiler's expectations.

Recommendation: start with the flat binary + separate compiler tool — keeps
FlexNN's export responsibility minimal (just serialize) and keeps all
validation/quantization/codegen logic in one place (the compiler), which is
easier to test and extend independently.

Must include, at minimum: format version/magic number, layer count, per
layer (type, input dim, output dim, activation, weight/bias arrays, dtype).

## 7. Target Applications

Three primary scenarios, deliberately spanning a complexity gradient, all
using only the STM32F407G-DISC1's onboard peripherals (digital MEMS
microphone, 3-axis accelerometer) — no additional hardware procurement.

Build order: 7.1 → 7.2 → 7.3. Case 7.1 is smallest/fastest, used to
validate the full pipeline end-to-end once before tackling the harder
cases. Case 7.4 is an optional stretch goal, attempted only after 7.1–7.3
are complete.

### 7.1 Vibration anomaly / fault detection (accelerometer, binary)
- Input: windowed raw or lightly-featurized accelerometer time-series.
- Output: normal vs. anomalous vibration signature (binary).
- Architecture: small MLP (Dense + ReLU), or Conv1D + Dense once Conv1D
  support lands in FlexNN.
- Narrative: predictive maintenance / motor-health monitoring — ties
  directly to drone/robotics background.

### 7.2 Gesture / activity recognition (accelerometer, multi-class)
- Input: windowed accelerometer data.
- Output: one of a small fixed set of motion patterns (e.g. walking, idle,
  shake, a specific gesture).
- Architecture: similar shape to 7.1, more output classes, slightly larger
  model — the "medium" data point in the complexity gradient.

### 7.3 Keyword spotting (microphone, multi-class + preprocessing)
- Input: raw audio samples from the onboard mic via I2S + DMA.
- Preprocessing: MFCC/FFT feature extraction (CMSIS-DSP) ahead of the
  classifier — the most complex preprocessing pipeline of the three cases.
- Output: keyword class (subset of Google Speech Commands dataset,
  training done in FlexNN).
- Fallback if MFCC/full KWS proves too time-consuming: Voice Activity
  Detection (VAD) — binary speech-vs-silence, same mic input, much simpler
  preprocessing than full keyword classification.

### 7.4 (Stretch goal) RL policy inference
- Input: accelerometer state (orientation/acceleration).
- Output: action from a policy trained off-device in MuJoCo/Gazebo via RL
  (PPO/SAC), exported as a plain MLP (Dense + Tanh/ReLU) — structurally
  identical to the other scenarios' models, no new op types required if the
  policy network is a standard MLP.
- **Inference only.** No on-device training/policy updates.
- Version 1 (fits current no-extra-hardware constraint): log the policy's
  chosen action over UART alongside live sensor readings, no actuation.
- Version 2 (requires procuring a servo + driver, real-time control loop
  work — only attempt if ahead of schedule): closed-loop physical
  demo (e.g. simple balancing/stabilization rig).

## 8. Benchmarking Methodology

For each of the three (or four, if 7.4 is attempted) target applications,
build **separate firmware images**, flashed and measured one at a time
(not multiplexed in one binary):

1. Portable scalar C++ reference backend
2. CMSIS-NN SIMD backend
3. TensorFlow Lite Micro, **with its CMSIS-NN kernel backend explicitly
   enabled** (not default reference kernels — using TFLite Micro's
   unoptimized kernels against TinyForge's optimized backend would be an
   invalid, inflated comparison)

**Metrics per image:**
- Inference latency (via on-device cycle counter / timer)
- RAM footprint: static allocation + stack high-water mark (TFLite Micro:
  include tensor arena size)
- Flash footprint: binary size, weights vs. code split where possible
- Accuracy: compare FP32-on-PC baseline vs. int8-on-device output, same
  test set, for both TinyForge and TFLite Micro builds

**Expected/target result:** once both TinyForge and TFLite Micro use
CMSIS-NN kernels, raw compute latency should be similar (same underlying
MACs). The differentiator is expected to be RAM/flash footprint (TFLite
Micro's interpreter, op resolver, and flatbuffer parsing overhead) and
init/startup overhead (runtime graph parsing vs. compile-time-baked
architecture). Report actual numbers honestly regardless of whether they
match this expectation.

**Hardware note:** STM32F407's 192 KB RAM is split — 112 KB main SRAM
(DMA-capable) + 64 KB CCM (not DMA-capable) + a smaller backup region.
Audio (I2S/DMA) buffers must live in the DMA-capable region, not CCM.

## 9. Design Decisions / Rationale Log

Keep this section updated as decisions are made — useful both for your own
reference and as material for interview discussions about why the project
is built the way it is.

- **FlexNN-only ingestion, no generic PyTorch/ONNX import.** A generic
  importer means building an open-ended model-graph compiler (arbitrary
  op coverage, arbitrary graph topology) — a multi-year-scale problem in
  its mature form (this is what TFLite's converter and ONNX Runtime's
  conversion tooling actually are). Given project timeline, FlexNN-only
  keeps the compiler's validation problem bounded and fully testable, and
  keeps the "trained on desktop with my own training library, deployed
  end-to-end" narrative intact — every layer of the pipeline is something
  you built and can explain completely.
- **Compile-time backend selection, not runtime.** Avoids branching cost
  in the hot inference loop on a resource-constrained target.
- **Jacobi-style / stateless-per-cell computation preferred where relevant**
  for future parallelizable extensions — not directly applicable to Dense/
  Conv1D forward passes on a single core, but worth keeping in mind if
  any future op introduces iterative solving.
- **TFLite Micro comparison uses its CMSIS-NN backend, not default
  kernels.** Ensures the comparison isolates "purpose-built minimal
  engine" vs. "general-purpose interpreter overhead" rather than
  "optimized kernels" vs. "unoptimized kernels," which would be invalid.
- **STM32Cube (HAL/LL) chosen over libopencm3 and Zephyr.** Zephyr's RTOS
  overhead would undermine the project's own footprint-comparison premise.
  libopencm3 has thinner community support for this board's onboard
  audio/accelerometer peripherals specifically. STM32Cube has vendor
  reference code for exactly this hardware.
- **Standard Peripheral Library (SPL) not used.** Deprecated by ST, no
  current examples/support; STM32Cube is the maintained successor.

## 10. Toolchain / Environment

- **Build system:** PlatformIO, `platform = ststm32`,
  `board = disco_f407vg`, `framework = stm32cube`
- **Compiler:** `arm-none-eabi-gcc`
- **Flashing/debugging:** onboard ST-Link/V2-A, OpenOCD or
  STM32CubeProgrammer
- **NN acceleration library:** CMSIS-NN
- **DSP (feature extraction, KWS scenario):** CMSIS-DSP
- **Comparison framework:** TensorFlow Lite Micro (CMSIS-NN kernel backend
  enabled)
- **Board:** STM32F407G-DISC1 — STM32F407VGT6, Cortex-M4F, 1 MB flash,
  192 KB RAM, onboard ST-Link/V2-A, onboard digital MEMS microphone,
  onboard 3-axis digital accelerometer

## 11. Open Questions / TBD

- [ ] Finalize export format: custom flat binary vs. direct C header
      generation from FlexNN (Section 6)
- [ ] Exact op-registry data structure/representation (Section 5.2)
- [ ] Confirm accelerometer part number on received board revision
      (LIS3DSH vs LIS302DL — affects SPI driver code)
- [ ] Decide exact activation approximation strategy for Tanh/Sigmoid on
      Cortex-M4F (exact `expf()` vs. lookup-table/piecewise-linear
      approximation) — document as an accuracy-vs-latency tradeoff once
      decided
- [ ] Decide whether Section 7.4 (RL policy inference) is attempted at all
      given timeline, and if so, whether Version 1 (logged-only) or
      Version 2 (closed-loop with servo) is targeted