/**
 * @file conv1d.hpp
 * @brief Conv1D kernel declarations (1D temporal over sensor axis).
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_KERNELS_CONV1D_HPP
#define TINYFORGE_KERNELS_CONV1D_HPP

#include "tinyforge/types.hpp"

namespace tinyforge {
namespace kernels {

void conv1d_f32(const float* input, const float* weights, const float* bias,
                float* output, const Conv1DParams& params, Activation act);

void conv1d_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
               int8_t* output, const Conv1DParams& params,
               const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
               Activation act);

} // namespace kernels
} // namespace tinyforge

#endif // TINYFORGE_KERNELS_CONV1D_HPP
