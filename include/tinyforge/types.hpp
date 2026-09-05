/**
 * @file types.hpp
 * @brief Core types for TinyForge runtime — mirrors FlexNN's LayerTypes.hpp.
 *
 * This header defines the minimal enums and structs needed by the runtime
 * to interpret generated/model_data.h without pulling in FlexNN/Eigen.
 * It intentionally duplicates FlexNN's LayerType/Activation definitions
 * (kept in sync via submodule SHA) so the runtime can be built standalone
 * without FlexNN as a dependency. The canonical definitions live in
 * FlexNN/include/LayerTypes.hpp; TinyForge's copy is a lightweight
 * mirror for MCU builds (no Eigen, no STL heap).
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_TYPES_HPP
#define TINYFORGE_TYPES_HPP

#include <stdint.h>

namespace tinyforge {

/**
 * @brief Layer kinds supported by the runtime.
 * Must stay identical to FlexNN::LayerType.
 */
enum class LayerType : uint8_t {
  Dense = 0,
  Conv1D = 1,
  BatchNorm1D = 2,  // folded at compile time, never appears in model_data.h
  MaxPool1D = 3,
  AvgPool1D = 4
};

/**
 * @brief Activation functions.
 * Must stay identical to FlexNN::Activation.
 */
enum class Activation : uint8_t {
  None = 0,
  ReLU = 1,
  LeakyReLU = 2,
  Sigmoid = 3,
  Tanh = 4,
  Softmax = 5  // only on last layer; runtime does argmax on logits
};

/**
 * @brief Quantization scheme for a tensor.
 * v0.1: only Int8PerTensor; Int8PerChannel planned.
 */
enum class QuantScheme : uint8_t {
  Float32 = 0,
  Int8PerTensor = 1,
  Int8PerChannel = 2
};

/**
 * @brief Compute backend selected at compile time.
 */
enum class Backend : uint8_t {
  Scalar = 0,
  CmsisNN = 1
};

/**
 * @brief Runtime status codes — no exceptions, no heap.
 */
enum class Status : int32_t {
  Ok = 0,
  ErrBadDims = -1,
  ErrArenaTooSmall = -2,
  ErrUnsupportedOp = -3,
  ErrNullPtr = -4,
  ErrBadModel = -5
};

/**
 * @brief Per-tensor quantization parameters.
 * Produced by FlexNN's tinyforge-compile (FlexNN/tools/compiler) and
 * baked into generated/model_data.h.
 */
struct QuantParams {
  float scale = 1.0f;
  int32_t zero_point = 0;
  // For matmul requantization: effective_scale = scale_in * scale_w / scale_out
  int32_t multiplier = 1; // Q31
  int shift = 0;          // 1..31
};

/**
 * @brief Convolution geometry for Conv1D (also used for quantized Conv).
 * Int fields avoid pulling <cstddef>.
 */
struct Conv1DParams {
  int16_t in_channels = 0;
  int16_t out_channels = 0;
  int16_t kernel_size = 0;
  int16_t stride = 1;
  int16_t padding = 0;
  int16_t dilation = 1;
};

} // namespace tinyforge

#endif // TINYFORGE_TYPES_HPP
