/**
 * @file activations.cpp
 * @brief Activation kernels — see ref/LLD.md §12.
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include "tinyforge/kernels/activations.hpp"
#include <cmath>

namespace tinyforge {
namespace kernels {

float activate_f32(float x, Activation act) {
  switch (act) {
    case Activation::None: return x;
    case Activation::ReLU: return x > 0 ? x : 0;
    case Activation::LeakyReLU: return x > 0 ? x : 0.01f * x;
    case Activation::Sigmoid: {
      if (x < -15) return 0;
      if (x > 15) return 1;
      return 1.0f / (1.0f + std::exp(-x));
    }
    case Activation::Tanh: return std::tanh(x);
    case Activation::Softmax: return x; // Softmax is argmax on logits at runtime
    default: return x;
  }
}

void activate_f32_inplace(float* data, int size, Activation act) {
  for (int i = 0; i < size; ++i) data[i] = activate_f32(data[i], act);
}

// TODO: LUT for int8 Sigmoid/Tanh (256 entries, see LLD)
int8_t activate_s8(int8_t q, Activation act, const QuantParams& qp) {
  switch (act) {
    case Activation::ReLU: return q > qp.zero_point ? q : (int8_t)qp.zero_point;
    default: return q;
  }
}

void activate_s8_inplace(int8_t* data, int size, Activation act, const QuantParams& qp) {
  for (int i = 0; i < size; ++i) data[i] = activate_s8(data[i], act, qp);
}

} // namespace kernels
} // namespace tinyforge
