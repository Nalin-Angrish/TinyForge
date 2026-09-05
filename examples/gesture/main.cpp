/**
 * @file main.cpp
 * @brief Example: gesture recognition (7.2) — multi-class accel.
 * See anomaly_detection/main.cpp for build instructions.
 * @author Nalin Angrish <nalin@nalinangrish.me>
 */
#include "tinyforge/tinyforge.h"

#if __has_include("generated/model_data.h")
#include "generated/model_data.h"
#endif

int main() {
  tinyforge::Model model;
  if (model.init() != tinyforge::Status::Ok) return -1;
  int8_t input[192] = {0};
  int8_t output[6] = {0}; // 6 gestures
  // accel_read_window(input);
  auto s = model.run(input, output);
  if (s != tinyforge::Status::Ok) return -1;
  // argmax
  int best = 0;
  for (int i = 1; i < 6; ++i) if (output[i] > output[best]) best = i;
  (void)best;
  return 0;
}
