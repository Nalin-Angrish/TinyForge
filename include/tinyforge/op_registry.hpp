/**
 * @file op_registry.hpp
 * @brief Single source of truth for TinyForge supported ops.
 * Owned by TinyForge (not FlexNN). TinyForge's compiler and runtime
 * both include this header. FlexNN has no copy — it just exports
 * whatever it was trained with; TinyForge's compiler will reject
 * unsupported combos.
 *
 * Add a new op: add one line to kRegistry, implement kernel in
 * lib/backend_*.cpp, add tests in tests/ and tools/compiler/tests/.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_OP_REGISTRY_HPP
#define TINYFORGE_OP_REGISTRY_HPP

#include <cstddef>
#include <cstdint>
#include "tinyforge/types.hpp"

namespace tinyforge {

struct OpEntry {
  LayerType type;
  Activation act;
  QuantScheme quant;
  Backend backend;
  const char* kernel_symbol;
  uint8_t version_added;
};

static constexpr OpEntry kRegistry[] = {
  {LayerType::Dense,  Activation::ReLU,      QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_relu_s8",  1},
  {LayerType::Dense,  Activation::None,      QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_none_s8",  1},
  {LayerType::Dense,  Activation::Sigmoid,   QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_sigmoid_s8", 1},
  {LayerType::Dense,  Activation::Tanh,      QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_tanh_s8",  1},
  {LayerType::Dense,  Activation::LeakyReLU, QuantScheme::Int8PerTensor, Backend::Scalar,  "dense_lrelu_s8", 1},
  {LayerType::Dense,  Activation::ReLU,      QuantScheme::Float32,       Backend::Scalar,  "dense_relu_f32", 1},
  {LayerType::Conv1D, Activation::ReLU,      QuantScheme::Int8PerTensor, Backend::Scalar,  "conv1d_relu_s8", 1},
  {LayerType::Conv1D, Activation::None,      QuantScheme::Int8PerTensor, Backend::Scalar,  "conv1d_none_s8", 1},
  // CMSIS-NN variants — same ops but with Backend::CmsisNN
  {LayerType::Dense,  Activation::ReLU,      QuantScheme::Int8PerTensor, Backend::CmsisNN, "dense_relu_s8_cmsis", 1},
  {LayerType::Conv1D, Activation::ReLU,      QuantScheme::Int8PerTensor, Backend::CmsisNN, "conv1d_relu_s8_cmsis", 1},
};

static constexpr size_t kRegistrySize = sizeof(kRegistry) / sizeof(kRegistry[0]);

inline const OpEntry* find(LayerType t, Activation a, QuantScheme q, Backend b) {
  for (size_t i = 0; i < kRegistrySize; ++i) {
    const auto& e = kRegistry[i];
    if (e.type == t && e.act == a && e.quant == q && (e.backend == b || e.backend == Backend::Scalar)) {
      // For now, scalar entries satisfy both; CMSIS entries are explicit
      // A more precise check would be: e.backend == b || e.backend == Backend::Both
      // But we use simple scalar fallback for host builds
      return &e;
    }
  }
  return nullptr;
}

} // namespace tinyforge

#endif // TINYFORGE_OP_REGISTRY_HPP
