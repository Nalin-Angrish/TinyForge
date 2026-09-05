/**
 * @file runtime.hpp
 * @brief TinyForge runtime public API — the library you link into firmware.
 *
 * This is the only header a firmware project needs to include to run a
 * FlexNN-compiled model. It is deliberately tiny: no Eigen, no STL heap,
 * no exceptions, no RTTI. The generated header `generated/model_data.h`
 * (produced by FlexNN's tinyforge-compile) is the only model-specific
 * dependency.
 *
 * Typical firmware usage (see examples/anomaly_detection/main.cpp):
 * @code
 * #include "tinyforge/runtime.hpp"
 * #include "generated/model_data.h" // from FlexNN/tools/compiler
 *
 * int main() {
 *   tinyforge::Model model;
 *   if (model.init() != tinyforge::Status::Ok) { error_blink(); }
 *   int8_t input[TINYFORGE_INPUT_DIM] = { ... }; // from sensor
 *   int8_t output[TINYFORGE_OUTPUT_DIM];
 *   model.run(input, output);
 *   // output is logits (argmax for classification)
 * }
 * @endcode
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#ifndef TINYFORGE_RUNTIME_HPP
#define TINYFORGE_RUNTIME_HPP

#include <stdint.h>
#include "types.hpp"

namespace tinyforge {

/**
 * @brief Embedded inference engine — one model per firmware image.
 *
 * The Model is a lightweight handle over a static arena (no malloc).
 * `init()` validates `generated/model_data.h` against the op registry
 * and checks arena size; `run()` does a linear stack forward pass
 * with double-buffered arena (no heap, no virtual).
 *
 * Thread safety: not thread-safe; call `run()` from one context only
 * (e.g., main loop or sensor ISR, not both).
 */
class Model {
public:
  Model() = default;
  ~Model() = default;

  // Non-copyable, non-movable (owns static arena pointers)
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /**
   * @brief Validate the compiled model and bind the static arena.
   * @return Status::Ok on success, ErrUnsupportedOp/ErrArenaTooSmall/etc. on failure
   *
   * Checks:
   *  - TINYFORGE_LAYER_COUNT <= kMaxLayers
   *  - each (type, act) is in the registry for the current backend
   *  - TINYFORGE_ARENA_BYTES >= required (from header)
   * No heap is allocated; the arena is a static buffer in lib/runtime.cpp
   * (placed in .ccmram if available).
   */
  Status init();

  /**
   * @brief Run inference — int8 path (primary).
   * @param input  Caller-owned, size TINYFORGE_INPUT_DIM (int8)
   * @param output Caller-owned, size TINYFORGE_OUTPUT_DIM (int8, logits)
   * @return Status::Ok or ErrNullPtr/ErrBadDims
   */
  Status run(const int8_t* input, int8_t* output);

  /**
   * @brief Run inference — float path (for host tests / accuracy baseline).
   * @param input  Caller-owned, size TINYFORGE_INPUT_DIM (float)
   * @param output Caller-owned, size TINYFORGE_OUTPUT_DIM (float)
   */
  Status run(const float* input, float* output);

  /**
   * @brief Dimensions from generated/model_data.h
   */
  int input_dim() const;
  int output_dim() const;
  int layer_count() const;

private:
  bool initialized_ = false;
  // Arena is static in lib/runtime.cpp; we keep pointers for double-buffering
  int8_t* arena0_ = nullptr;
  int8_t* arena1_ = nullptr;
};

} // namespace tinyforge

#endif // TINYFORGE_RUNTIME_HPP
