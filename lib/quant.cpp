/**
 * @file quant.cpp
 * @brief Quantization helpers — shared math with FlexNN's compiler.
 * See ref/LLD.md §7.
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include "tinyforge/quant.hpp"

namespace tinyforge {

int32_t requantize(int32_t acc, int32_t multiplier, int shift) {
  // Simplified scalar version of arm_nn_requantize
  // Real CMSIS does: acc * multiplier with 64-bit, rounding, then shift
  int64_t total = (int64_t)acc * multiplier;
  // Rounding: add (1 << (31 - shift - 1)) before shift
  // This is a stub; full impl will match CMSIS's rounding.
  int32_t result = (int32_t)(total >> (31 - shift));
  return result;
}

float dequantize(int8_t q, float scale, int32_t zp) {
  return scale * ((int32_t)q - zp);
}

int8_t quantize(float f, float scale, int32_t zp) {
  int32_t q = (int32_t)(f / scale + zp + 0.5f);
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

} // namespace tinyforge
