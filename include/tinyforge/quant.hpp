/**
 * @file quant.hpp
 * @brief Quantization helpers — requantization, scale/zp math.
 * Matches FlexNN/tools/compiler quant math (§7 of LLD) so TinyForge and
 * FlexNN agree on multiplier/shift.
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_QUANT_HPP
#define TINYFORGE_QUANT_HPP

#include <stdint.h>
#include "types.hpp"

namespace tinyforge {

// Q31 requantize as used by CMSIS-NN's arm_nn_requantize
int32_t requantize(int32_t acc, int32_t multiplier, int shift);

// Dequant/quant helpers for host tests
float dequantize(int8_t q, float scale, int32_t zp);
int8_t quantize(float f, float scale, int32_t zp);

} // namespace tinyforge

#endif // TINYFORGE_QUANT_HPP
