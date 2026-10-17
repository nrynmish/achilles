#pragma once

#include <cstddef>
#include <vector>

// Matrix multiply a naive CPU implementation.
// A is an M x K matrix, B is a K x N matrix, and the result is an M x N matrix.
// All matrices are stored contiguously in row-major order.
std::vector<float> matmul_naive(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k);

// Matrix multiply using a cache-blocked CPU implementation.
// A is an M x K matrix, B is a K x N matrix, and the result is an M x N matrix.
// All matrices are stored contiguously in row-major order. The block_size controls
// the size of the tiles used to improve cache locality while multiplying.
std::vector<float> matmul_blocked(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t block_size = 32);

// Matrix multiply using an AVX2/FMA-accelerated implementation.
// A is an M x K matrix, B is a K x N matrix, and the result is an M x N matrix.
// All matrices use contiguous row-major storage. This implementation uses an 8-wide
// float vector path for rows of the output when the CPU supports AVX2 + FMA; otherwise
// it safely falls back to the naive scalar implementation.
std::vector<float> matmul_simd(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k);

// Matrix multiply using cache-blocked AVX2/FMA vectorization.
// A is an M x K matrix, B is a K x N matrix, and the result is an M x N matrix.
// The implementation tiles the result into cache-friendly blocks and then processes
// each output-column strip with a width-8 AVX2/FMA vector path when supported.
std::vector<float> matmul_blocked_simd(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t block_size = 32);
