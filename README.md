# Achilles

AI Kernel Optimization & Performance Engineering

Achilles is a C++20 project for implementing, validating, benchmarking, and optimizing matrix-multiplication kernels. The project focuses on correctness first, then performance tuning with deterministic release-mode benchmarks. The public API remains in `include/achilles/`, implementations live in `src/`, and tests live in `tests/`.

## Project architecture

- `include/achilles/matmul.hpp`: public matrix multiplication API
- `src/matmul.cpp`: naive, blocked, SIMD, and blocked-SIMD implementations
- `tests/test_matmul.cpp`: deterministic correctness and validation tests
- `benchmarks/matmul_benchmark.cpp`: benchmark harness for median latency and speedup reporting
- `CMakeLists.txt`: C++20 configuration and test registration

## Build, test, and benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/achilles_tests
./build/achilles_benchmark
```

## Kernel overview

- `matmul_naive`: straightforward scalar reference implementation for correctness and baseline timing.
- `matmul_blocked`: cache-aware tiled kernel that reduces poor locality in the full-matrix scalar path.
- `matmul_simd`: AVX2/FMA-accelerated implementation guarded by runtime CPU detection and a function-specific target attribute.
- `matmul_blocked_simd`: cache-blocked SIMD path that reuses tiles across the reduction dimension to improve locality while remaining correct for non-multiple-of-8 cases.

## AVX2/FMA note

AVX2/FMA support is enabled through a function-specific target attribute and runtime CPU detection, not through global compiler flags. This keeps the code portable while still allowing the optimized path to be selected when the current CPU supports it. The project intentionally avoids library-wide `-mavx2 -mfma` flags.

## Benchmark methodology

The benchmark harness uses deterministic matrices, 2 warm-up iterations, 9 timed repetitions, and the median latency as the reported value. Each run checks kernel correctness before printing performance metrics. GFLOP/s is computed for square matrices as:

$\mathrm{GFLOP/s} = \frac{2 \cdot M \cdot N \cdot K}{t \cdot 10^9}$

with $M = N = K = n$ for the square benchmark cases. Speedup is reported relative to the naive baseline for the same matrix size.

The benchmark is designed for comparison on the same machine and same build configuration; it is not a claim that the same speedups will hold across different CPUs or compilers.

## Release benchmark results

These are results from one Release benchmark run on the developer's machine. They are included as a reporting example and should not be interpreted as universal performance claims for other hardware.

| Matrix size | Best implementation | Block size | Median latency (ms) | Speedup vs naive |
| --- | --- | ---: | ---: | ---: |
| 64 | Blocked SIMD | 16 | 0.02032 | 5.8867x |
| 128 | Blocked SIMD | 16 | 0.15936 | 13.2754x |
| 256 | Blocked SIMD | 16 | 1.3668 | 12.6391x |
| 512 | Blocked SIMD | 16 | 14.6228 | 18.4033x |
| 1024 | Blocked SIMD | 8 | 127.114 | 47.7873x |

## Environment inspection commands

Use the commands below to inspect the local build environment on any machine:

```bash
uname -a
cmake --version
c++ --version
lscpu | head -n 20
cat /proc/cpuinfo | grep -E "model name|flags" | head -n 20
```

These commands help readers verify their own environment and compare results without assuming the same hardware or compiler settings as the developer machine.
