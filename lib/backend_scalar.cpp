/**
 * @file backend_scalar.cpp
 * @brief Scalar (portable) backend — reference implementation.
 * Compiled when TINYFORGE_BACKEND==0 (default). Mirrors FlexNN's lib/Layer.cpp
 * but for inference only, no Eigen, no heap.
 * See ref/LLD.md §11 for math.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include "tinyforge/backend.hpp"
#include "tinyforge/kernels/dense.hpp"
#include "tinyforge/kernels/conv1d.hpp"
#include "tinyforge/quant.hpp"

#if TINYFORGE_BACKEND == 0

namespace tinyforge {

Status run_dense_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
                    int8_t* output, int in_dim, int out_dim,
                    const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
                    Activation act) {
  if (!input || !weights || !bias || !output) return Status::ErrNullPtr;
  if (in_dim <= 0 || out_dim <= 0) return Status::ErrBadDims;
  kernels::dense_s8(input, weights, bias, output, in_dim, out_dim, qp_in, qp_w, qp_out, act);
  return Status::Ok;
}

Status run_conv1d_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
                     int8_t* output, const Conv1DParams& conv,
                     const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
                     Activation act) {
  if (!input || !weights || !bias || !output) return Status::ErrNullPtr;
  kernels::conv1d_s8(input, weights, bias, output, conv, qp_in, qp_w, qp_out, act);
  return Status::Ok;
}

Status run_dense_f32(const float* input, const float* weights, const float* bias,
                     float* output, int in_dim, int out_dim, Activation act) {
  if (!input || !weights || !bias || !output) return Status::ErrNullPtr;
  kernels::dense_f32(input, weights, bias, output, in_dim, out_dim, act);
  return Status::Ok;
}

} // namespace tinyforge

// ── Kernel stubs — to be filled with actual loops ──
#include "tinyforge/kernels/activations.hpp"
namespace tinyforge {
namespace kernels {

void dense_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
              int8_t* output, int in_dim, int out_dim,
              const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
              Activation act) {
  for (int o = 0; o < out_dim; ++o) {
    int32_t acc = bias[o];
    for (int i = 0; i < in_dim; ++i) {
      acc += (int32_t)(input[i] - qp_in.zero_point) * (int32_t)(weights[o * in_dim + i] - qp_w.zero_point);
    }
    acc = requantize(acc, qp_out.multiplier, qp_out.shift);
    acc += qp_out.zero_point;
    if (acc < -128) acc = -128;
    if (acc > 127) acc = 127;
    // TODO: activation LUT for Sigmoid/Tanh on int8 domain
    if (act == Activation::ReLU && acc < qp_out.zero_point) acc = qp_out.zero_point;
    output[o] = (int8_t)acc;
  }
}

void conv1d_s8(const int8_t* input, const int8_t* weights, const int32_t* bias,
               int8_t* output, const Conv1DParams& conv,
               const QuantParams& qp_in, const QuantParams& qp_w, const QuantParams& qp_out,
               Activation act) {
  // TODO: implement per ref/LLD.md §11.2
  (void)input; (void)weights; (void)bias; (void)output;
  (void)conv; (void)qp_in; (void)qp_w; (void)qp_out; (void)act;
}

void dense_f32(const float* input, const float* weights, const float* bias,
               float* output, int in_dim, int out_dim, Activation act) {
  for (int o = 0; o < out_dim; ++o) {
    float acc = bias[o];
    for (int i = 0; i < in_dim; ++i) acc += weights[o * in_dim + i] * input[i];
    output[o] = activate_f32(acc, act);
  }
}

} // namespace kernels
} // namespace tinyforge

#endif // TINYFORGE_BACKEND == 0
