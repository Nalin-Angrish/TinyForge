# TinyForge Compiler

`tinyforge-compile` lives in **TinyForge** (not FlexNN) so that **TinyForge depends on FlexNN**, not vice versa.

- **FlexNN** (`external/FlexNN`): standalone training library. Exports `model.bin` via `ModelIO` — no TinyForge headers, no validation.
- **TinyForge compiler** (`tools/compiler`): parses `model.bin` (reusing FlexNN's `ModelIO` structs via `external/`), validates against `TinyForge/include/tinyforge/op_registry.hpp`, quantizes, and emits `generated/model_data.h` for `TinyForge`'s runtime.

Build (host, also builds TinyForge lib):

```bash
cmake -S . -B build
cmake --build build -j
# → build/tools/compiler/tinyforge-compile
./build/tools/compiler/tinyforge-compile model.bin -o generated --backend cmsis
```

See `docs/LLD.md §6` for details.
