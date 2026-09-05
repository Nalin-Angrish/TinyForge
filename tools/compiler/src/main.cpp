/**
 * @file main.cpp
 * @brief tinyforge-compile CLI — parses FlexNN model.bin and emits model_data.h
 * Lives in TinyForge/tools/compiler (TinyForge depends on FlexNN).
 */
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: tinyforge-compile <model.bin> -o generated [--backend scalar|cmsis]\n";
    return 0;
  }
  std::cout << "TinyForge compiler stub — will parse " << argv[1] << " and emit model_data.h\n";
  std::cout << "FlexNN exports model.bin; TinyForge validates & quantizes for its runtime.\n";
  return 0;
}
