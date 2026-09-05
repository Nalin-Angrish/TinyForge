/**
 * @file backend.hpp
 * @brief Backend abstraction for TinyForge runtime.
 *
 * Compile-time backend selection — no virtual, no heap, no runtime branch
 * in the hot loop. Mirrors the design in ref/LLD.md §10.
 *
 * Usage:
 *   // In CMake: -DTINYFORGE_BACKEND=0 (scalar) or 1 (CMSIS-NN)
 *   // At runtime: tinyforge::run_layer<...>(...)
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_BACKEND_HPP
#define TINYFORGE_BACKEND_HPP

#include "types.hpp"

#ifndef TINYFORGE_BACKEND
#define TINYFORGE_BACKEND 0 // 0=scalar (default, host-portable), 1=CMSIS-NN
#endif

namespace tinyforge {

#if TINYFORGE_BACKEND == 1
constexpr Backend kBackend = Backend::CmsisNN;
#else
constexpr Backend kBackend = Backend::Scalar;
#endif

/**
 * @brief Generic layer runner — dispatches to scalar or CMSIS-NN at compile time.
 *
 * Implemented in lib/backend_scalar.cpp and lib/backend_cmsis.cpp.
 * The header declares the interface; the correct .cpp is compiled per backend.
 *
 * @param input      Input buffer (int8 or float, per quant)
 * @param weights    Weight buffer (int8 or float)
 * @param bias       Bias buffer (int32 for int8 path, float for float path)
 * @param output     Output buffer
 * @param qp_in      Input quant params
 * @param qp_w       Weight quant params
 * @param qp_out     Output quant params
 * @param conv       Conv geometry (null for Dense)
 * @param act        Activation to apply after matmul
 * @return Status
 */
// Int8 path — primary for MCU
Status run_dense_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
                    int8_t* output, int in_dim, int out_dim,
                    const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
                    Activation act);

Status run_conv1d_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
                     int8_t* output, const Conv1DParams& conv,
                     const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
                     Activation act);

// Float path — for accuracy baseline and host tests
Status run_dense_f32(const float* input, const float* weights, const float* bias,
                     float* output, int in_dim, int out_dim, Activation act);

} // namespace tinyforge

#endif // TINYFORGE_BACKEND_HPP
