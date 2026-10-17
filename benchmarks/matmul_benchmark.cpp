#include "achilles/matmul.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kWarmupIterations = 2;
constexpr std::size_t kTimedRepetitions = 5;
constexpr float kTolerance = 1.0e-4f;

std::vector<float> make_deterministic_matrix(std::size_t rows, std::size_t cols, std::size_t seed) {
    std::vector<float> matrix(rows * cols, 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const std::size_t index = row * cols + col;
            const int value = static_cast<int>((((row + 1) * 17 + (col + 1) * 13 + seed) % 23) - 11);
            matrix[index] = static_cast<float>(value) / 7.0f;
        }
    }
    return matrix;
}

float checksum_result(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0f);
}

double median_latency(const std::vector<double>& timings) {
    std::vector<double> sorted = timings;
    std::sort(sorted.begin(), sorted.end());

    const std::size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2.0;
    }
    return sorted[mid];
}

bool nearly_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, float tolerance) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::vector<float> naive_kernel(const std::vector<float>& a,
                               const std::vector<float>& b,
                               std::size_t m,
                               std::size_t n,
                               std::size_t k) {
    return matmul_naive(a, b, m, n, k);
}

std::vector<float> blocked_kernel(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 std::size_t m,
                                 std::size_t n,
                                 std::size_t k) {
    return matmul_blocked(a, b, m, n, k, 32);
}

std::vector<float> simd_kernel(const std::vector<float>& a,
                              const std::vector<float>& b,
                              std::size_t m,
                              std::size_t n,
                              std::size_t k) {
    return matmul_simd(a, b, m, n, k);
}

template <typename KernelFunc>
std::vector<double> benchmark_impl(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  std::size_t matrix_size,
                                  const std::string& label,
                                  KernelFunc&& kernel) {
    std::vector<double> timings;
    timings.reserve(kTimedRepetitions);

    for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        const auto warmup_result = kernel(a, b, matrix_size, matrix_size, matrix_size);
        if (warmup_result.size() != matrix_size * matrix_size) {
            throw std::runtime_error(label + " warm-up result has an unexpected size.");
        }
        (void)checksum_result(warmup_result);
    }

    for (std::size_t i = 0; i < kTimedRepetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = kernel(a, b, matrix_size, matrix_size, matrix_size);
        const auto end = std::chrono::steady_clock::now();

        if (result.size() != matrix_size * matrix_size) {
            throw std::runtime_error(label + " timed result has an unexpected size.");
        }

        timings.push_back(std::chrono::duration<double>(end - start).count());
        (void)checksum_result(result);
    }

    return timings;
}

void benchmark_case(std::size_t matrix_size) {
    const auto a = make_deterministic_matrix(matrix_size, matrix_size, 11);
    const auto b = make_deterministic_matrix(matrix_size, matrix_size, 17);

    const auto naive_timings = benchmark_impl(a, b, matrix_size, "naive", naive_kernel);
    const auto blocked_timings = benchmark_impl(a, b, matrix_size, "blocked", blocked_kernel);
    const auto simd_timings = benchmark_impl(a, b, matrix_size, "simd", simd_kernel);

    const auto naive_result = matmul_naive(a, b, matrix_size, matrix_size, matrix_size);
    const auto blocked_result = matmul_blocked(a, b, matrix_size, matrix_size, matrix_size, 32);
    const auto simd_result = matmul_simd(a, b, matrix_size, matrix_size, matrix_size);

    if (!nearly_equal(naive_result, blocked_result, kTolerance) ||
        !nearly_equal(naive_result, simd_result, kTolerance)) {
        throw std::runtime_error("Naive and optimized results differ beyond the tolerance for benchmark size " +
                                 std::to_string(matrix_size) + ".");
    }

    const double naive_median = median_latency(naive_timings);
    const double blocked_median = median_latency(blocked_timings);
    const double simd_median = median_latency(simd_timings);

    const double naive_gflops = (2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
                                static_cast<double>(matrix_size)) /
                               (naive_median * 1.0e9);
    const double blocked_gflops = (2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
                                 static_cast<double>(matrix_size)) /
                                (blocked_median * 1.0e9);
    const double simd_gflops = (2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
                              static_cast<double>(matrix_size)) /
                             (simd_median * 1.0e9);

    const double blocked_speedup = naive_median / blocked_median;
    const double simd_speedup = naive_median / simd_median;

    std::cout << "size=" << matrix_size
              << " | naive_median_ms=" << (naive_median * 1.0e3)
              << " | blocked_median_ms=" << (blocked_median * 1.0e3)
              << " | simd_median_ms=" << (simd_median * 1.0e3)
              << " | naive_gflops=" << naive_gflops
              << " | blocked_gflops=" << blocked_gflops
              << " | simd_gflops=" << simd_gflops
              << " | blocked_speedup=" << blocked_speedup
              << " | simd_speedup=" << simd_speedup << "\n";
}

}  // namespace

int main() {
    std::cout << "size | naive_median_ms | blocked_median_ms | simd_median_ms | naive_gflops | blocked_gflops | simd_gflops | blocked_speedup | simd_speedup\n";
    for (const std::size_t size : {32u, 64u, 128u, 256u, 512u}) {
        benchmark_case(size);
    }
    return 0;
}
