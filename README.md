# Achilles

AI Kernel Optimization & Performance Engineering

A C++20 project for implementing, testing, benchmarking, and optimizing computational kernels.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
