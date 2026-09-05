# generated/

This directory holds `model_data.h` (and optionally `model_data.cpp`) produced by
**TinyForge's** `tinyforge-compile` CLI (in `tools/compiler`, which links FlexNN from `external/` for parsing).

It is **gitignored** — not committed. Regenerate:

```bash
# Train and export with FlexNN (standalone)
cmake -S external/FlexNN -B external/FlexNN/build && cmake --build external/FlexNN/build -j
./external/FlexNN/build/main  # → model.bin

# Compile with TinyForge (validates, quantizes, emits header)
cmake -S . -B build && cmake --build build -j
./build/tools/compiler/tinyforge-compile model.bin --backend cmsis -o generated
# → generated/model_data.h
```

The header contains `static const int8_t weights[]`, `quant params`, and
`tinyforge_layers[]` as compile-time constants. TinyForge's runtime
(`include/tinyforge/runtime.hpp`) includes it as `generated/model_data.h`.

For host tests, a dummy header is used if this directory is empty.
