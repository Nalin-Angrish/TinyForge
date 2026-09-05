/**
 * @file dense.hpp
 * @brief Dense (fully-connected) kernel declarations.
 * Scalar and CMSIS-NN implementations live in lib/backend_*.cpp.
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_KERNELS_DENSE_HPP
#define TINYFORGE_KERNELS_DENSE_HPP

#include "tinyforge/types.hpp"

namespace tinyforge {
namespace kernels {

// Float reference — for host tests and accuracy baseline
void dense_f32(const float* input, const float* weights, const float* bias,
               float* output, int in_dim, int out_dim, Activation act);

// Int8 quantized — primary MCU path
void dense_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
              int8_t* output, int in_dim, int out_dim,
              const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
              Activation act);

} // namespace kernels
} // namespace tinyforge

#endif // TINYFORGE_KERNELS_DENSE_HPP
