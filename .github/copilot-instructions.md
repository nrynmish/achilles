# Achilles Copilot Instructions

- Use C++20 and CMake.
- Keep public interfaces in include/achilles/.
- Keep implementations in src/.
- Keep deterministic offline tests in tests/.
- Keep benchmarks separate from correctness tests.
- Do not add external C++ dependencies without approval.
- Do not implement optimizations until the naive baseline passes tests.
- Never claim unmeasured performance improvements.
