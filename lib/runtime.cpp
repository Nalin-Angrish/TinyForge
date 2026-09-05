/**
 * @file runtime.cpp
 * @brief TinyForge Model implementation — static arena, double-buffering.
 * This mirrors FlexNN's lib/FlexNN.cpp structure but for inference only.
 * See ref/LLD.md §9, §13 for memory model and API.
 *
 * Platform notes:
 *  - No malloc/new, no exceptions, no RTTI.
 *  - Arena placed in .ccmram if available (faster, not DMA-capable)
 *    else .bss. Size is TINYFORGE_ARENA_BYTES from generated/model_data.h.
 *  - For host tests, arena is plain static.
 *
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include "tinyforge/runtime.hpp"
#include "tinyforge/backend.hpp"

// Generated header — produced by FlexNN's tinyforge-compile
// For host builds without a model, we provide a minimal fallback.
#if __has_include("generated/model_data.h")
#include "generated/model_data.h"
#elif __has_include("../generated/model_data.h")
#include "../generated/model_data.h"
#else
// Fallback for library-only builds (no model): define dummy dims
#ifndef TINYFORGE_INPUT_DIM
#define TINYFORGE_INPUT_DIM 0
#endif
#ifndef TINYFORGE_OUTPUT_DIM
#define TINYFORGE_OUTPUT_DIM 0
#endif
#ifndef TINYFORGE_LAYER_COUNT
#define TINYFORGE_LAYER_COUNT 0
#endif
#ifndef TINYFORGE_ARENA_BYTES
#define TINYFORGE_ARENA_BYTES 0
#endif
#endif

namespace tinyforge {

// ── Static arena — double-buffered ──
// Like FlexNN's static buffers but for inference. Placed in CCM for speed.
#if defined(__ARMCC_VERSION) || defined(__CC_ARM)
static int8_t tinyforge_arena[TINYFORGE_ARENA_BYTES] __attribute__((section("RW_IRAM1")));
#elif defined(__ICCARM__)
#pragma location="CCMRAM"
static int8_t tinyforge_arena[TINYFORGE_ARENA_BYTES];
#elif defined(__GNUC__)
#if TINYFORGE_ARENA_BYTES > 0
// Try CCM first; linker script must define .ccmram
__attribute__((section(".ccmram"))) static int8_t tinyforge_arena[TINYFORGE_ARENA_BYTES];
#else
static int8_t tinyforge_arena[1];
#endif
#else
static int8_t tinyforge_arena[TINYFORGE_ARENA_BYTES > 0 ? TINYFORGE_ARENA_BYTES : 1];
#endif

// ── Model implementation ──
Status Model::init() {
  if (initialized_) return Status::Ok;
  if (TINYFORGE_LAYER_COUNT == 0) return Status::ErrBadModel;
  if (TINYFORGE_ARENA_BYTES == 0) return Status::ErrArenaTooSmall;
  // TODO: validate each (type, act) against op_registry from FlexNN
  //       #include "external/FlexNN/include/tinyforge/op_registry.hpp"
  //       for each layer: tinyforge::find(type, act, quant, kBackend)
  // TODO: check header magic/version if we embed it

  // Bind arena for double-buffering
  // For now, simple: arena0 = &tinyforge_arena[0], arena1 split
  // Real impl: see ref/LLD.md §9.2 — max(in_dim, out_dim) ping-pong
  arena0_ = tinyforge_arena;
  // arena1_ will be set per run() based on dims
  initialized_ = true;
  return Status::Ok;
}

Status Model::run(const int8_t* input, int8_t* output) {
  if (!initialized_) {
    Status s = init();
    if (s != Status::Ok) return s;
  }
  if (!input || !output) return Status::ErrNullPtr;
  if (TINYFORGE_LAYER_COUNT == 0) return Status::ErrBadModel;

  // TODO: linear stack forward pass
  //  - ping-pong between arena0_/arena1_
  //  - for each layer i: call run_dense_s8 / run_conv1d_s8 via backend
  //  - handle quant params from generated header (scale/zp/multiplier/shift)
  //  - final layer → output (logits); caller does argmax if needed

  // Stub: for now, just copy input to output if dims match (for host tests)
  (void)input;
  (void)output;
  return Status::Ok;
}

Status Model::run(const float* input, float* output) {
  if (!initialized_) {
    Status s = init();
    if (s != Status::Ok) return s;
  }
  if (!input || !output) return Status::ErrNullPtr;
  // TODO: float path
  (void)input;
  (void)output;
  return Status::Ok;
}

int Model::input_dim() const { return TINYFORGE_INPUT_DIM; }
int Model::output_dim() const { return TINYFORGE_OUTPUT_DIM; }
int Model::layer_count() const { return TINYFORGE_LAYER_COUNT; }

} // namespace tinyforge
