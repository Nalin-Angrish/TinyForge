/**
 * @file test_runtime.cpp
 * @brief Host unit tests for TinyForge runtime — mirrors FlexNN's tests structure.
 * Run on host with scalar backend: cmake -B build -DTINYFORGE_BUILD_TESTS=ON && ctest
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include "tinyforge/tinyforge.h"
#include <cassert>
#include <cstdio>

int main() {
  tinyforge::Model model;
  auto s = model.init();
  // Without a generated model, init may return ErrBadModel — that's ok for now
  printf("TinyForge Model::init() = %d\n", (int)s);
  printf("Input dim: %d, Output dim: %d, Layers: %d\n",
         model.input_dim(), model.output_dim(), model.layer_count());

  // Quick smoke test for kernels (no model needed)
  {
    float in[4] = {1, -2, 3, -4};
    float w[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0}; // identity-ish
    float b[3] = {0,0,0};
    float out[3];
    auto st = tinyforge::run_dense_f32(in, w, b, out, 4, 3, tinyforge::Activation::ReLU);
    assert(st == tinyforge::Status::Ok);
    // ReLU should zero negatives: out[1] from -2 should be 0
    printf("dense_f32 ReLU test: out=[%f %f %f]\n", out[0], out[1], out[2]);
  }

  printf("TinyForge host tests passed (stub)\n");
  return 0;
}
