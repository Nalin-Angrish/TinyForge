/**
 * @file backend_cmsis.cpp
 * @brief CMSIS-NN backend — SIMD accelerated for Cortex-M4F.
 * Compiled when TINYFORGE_BACKEND==1. Wraps arm_nnfunctions.h.
 * See ref/LLD.md §11 for CMSIS mapping.
 *
 * This file is *not* built for host tests; it requires CMSIS packs
 * which are vendored or fetched via CMake. For host builds the scalar
 * backend is used.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include "tinyforge/backend.hpp"

#if TINYFORGE_BACKEND == 1

// CMSIS-NN headers — available when cross-compiling for STM32
// #include "arm_nnfunctions.h"
// #include "arm_nnsupportfunctions.h"

namespace tinyforge {

Status run_dense_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
                    int8_t* output, int in_dim, int out_dim,
                    const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
                    Activation act) {
  if (!input || !weights || !bias || !output) return Status::ErrNullPtr;
  // TODO: call arm_fully_connected_s8
  // cmsis_nn_context ctx; ctx.buf = nullptr; ctx.size = 0;
  // cmsis_nn_fc_params fc_params = { .input_offset = -qp_in.zero_point, ... };
  // arm_fully_connected_s8(&ctx, &fc_params, &quant_params, &dims, input, ...);
  (void)input; (void)weights; (void)bias; (void)output;
  (void)in_dim; (void)out_dim; (void)qp_in; (void)qp_w; (void)qp_out; (void)act;
  return Status::Ok;
}

Status run_conv1d_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
                     int8_t* output, const Conv1DParams& conv,
                     const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
                     Activation act) {
  // TODO: map Conv1D to arm_convolve_wrapper_s8 with h=1 (see LLD §11.2)
  (void)input; (void)weights; (void)bias; (void)output;
  (void)conv; (void)qp_in; (void)qp_w; (void)qp_out; (void)act;
  return Status::Ok;
}

Status run_dense_f32(const float* input, const float* weights, const float* bias,
                     float* output, int in_dim, int out_dim, Activation act) {
  // CMSIS backend still uses scalar for float (baseline)
  if (!input || !weights || !bias || !output) return Status::ErrNullPtr;
  for (int o = 0; o < out_dim; ++o) {
    float acc = bias[o];
    for (int i = 0; i < in_dim; ++i) acc += weights[o * in_dim + i] * input[i];
    // TODO: activate via kernels
    (void)act;
    output[o] = acc;
  }
  return Status::Ok;
}

} // namespace tinyforge

#endif // TINYFORGE_BACKEND == 1
