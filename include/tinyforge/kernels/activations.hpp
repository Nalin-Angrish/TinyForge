/**
 * @file activations.hpp
 * @brief Activation kernels — ReLU, LeakyReLU, Sigmoid, Tanh, Softmax (argmax).
 * Float uses expf/tanhf; int8 uses 256-entry LUT (see ref/LLD.md §12).
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_KERNELS_ACTIVATIONS_HPP
#define TINYFORGE_KERNELS_ACTIVATIONS_HPP

#include "tinyforge/types.hpp"

namespace tinyforge {
namespace kernels {

// Float
float activate_f32(float x, Activation act);
void activate_f32_inplace(float* data, int size, Activation act);

// Int8 — operates on requantized domain; LUT for Sigmoid/Tanh
int8_t activate_s8(int8_t q, Activation act, const QuantParams& qp);
void activate_s8_inplace(int8_t* data, int size, Activation act, const QuantParams& qp);

} // namespace kernels
} // namespace tinyforge

#endif // TINYFORGE_KERNELS_ACTIVATIONS_HPP
