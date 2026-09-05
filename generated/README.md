# generated/

This directory holds `model_data.h` (and optionally `model_data.cpp`) produced by
**FlexNN's** `tinyforge-compile` CLI.

It is **gitignored** — not committed. Regenerate from FlexNN:

```bash
# From FlexNN repo (or TinyForge/external/FlexNN)
cmake -S . -B build && cmake --build build
./build/tools/compiler/tinyforge-compile model.bin --backend cmsis -o /path/to/TinyForge/generated

# Or from TinyForge with vendored FlexNN
cmake -S external/FlexNN -B external/FlexNN/build && cmake --build external/FlexNN/build
external/FlexNN/build/tools/compiler/tinyforge-compile model.bin --backend cmsis -o generated
```

The header contains `static const int8_t weights[]`, `quant params`, and
`tinyforge_layers[]` as compile-time constants. TinyForge's runtime
(`include/tinyforge/runtime.hpp`) includes it as `generated/model_data.h`.

For host tests, a dummy header is used if this directory is empty.
