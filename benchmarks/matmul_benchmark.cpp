#include "achilles/matmul.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

#ifndef ACHILLES_BUILD_TYPE
#define ACHILLES_BUILD_TYPE "Unknown"
#endif

// The benchmark uses a stable warm-up phase followed by a median-of-N timing pass to
// reduce noise from startup effects and scheduling jitter. The timed count is set above
// the earlier default to improve stability while remaining deterministic.
constexpr std::size_t kWarmupIterations = 2;
constexpr std::size_t kTimedRepetitions = 9;
constexpr float kTolerance = 1.0e-4f;
const std::array<std::size_t, 5> kBlockSizes = {8u, 16u, 32u, 64u, 128u};
const std::array<std::size_t, 5> kMatrixSizes = {64u, 128u, 256u, 512u, 1024u};

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
                                 std::size_t k,
                                 std::size_t block_size = 32) {
    return matmul_blocked(a, b, m, n, k, block_size);
}

std::vector<float> simd_kernel(const std::vector<float>& a,
                              const std::vector<float>& b,
                              std::size_t m,
                              std::size_t n,
                              std::size_t k) {
    return matmul_simd(a, b, m, n, k);
}

std::vector<float> blocked_simd_kernel(const std::vector<float>& a,
                                      const std::vector<float>& b,
                                      std::size_t m,
                                      std::size_t n,
                                      std::size_t k,
                                      std::size_t block_size = 8) {
    return matmul_blocked_simd(a, b, m, n, k, block_size);
}

template <typename KernelFunc>
std::vector<float> invoke_kernel(const std::vector<float>& a,
                                const std::vector<float>& b,
                                std::size_t matrix_size,
                                KernelFunc&& kernel,
                                std::size_t block_size) {
    if constexpr (std::is_invocable_r_v<std::vector<float>, KernelFunc, const std::vector<float>&,
                                        const std::vector<float>&, std::size_t, std::size_t, std::size_t>) {
        return kernel(a, b, matrix_size, matrix_size, matrix_size);
    } else {
        return kernel(a, b, matrix_size, matrix_size, matrix_size, block_size);
    }
}

template <typename KernelFunc>
std::vector<double> benchmark_kernel(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    std::size_t matrix_size,
                                    const std::string& label,
                                    KernelFunc&& kernel,
                                    std::size_t block_size = 0) {
    std::vector<double> timings;
    timings.reserve(kTimedRepetitions);

    const auto run_once = [&]() {
        return invoke_kernel(a, b, matrix_size, kernel, block_size);
    };

    // Warm-up passes prime caches and branch predictors without contributing to the reported
    // latency; the checksum is consumed after each warm-up to prevent the compiler from
    // discarding the call entirely in optimized builds.
    for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        const auto warmup_result = run_once();
        if (warmup_result.size() != matrix_size * matrix_size) {
            throw std::runtime_error(label + " warm-up result has an unexpected size.");
        }
        (void)checksum_result(warmup_result);
    }

    for (std::size_t i = 0; i < kTimedRepetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = run_once();
        const auto end = std::chrono::steady_clock::now();

        if (result.size() != matrix_size * matrix_size) {
            throw std::runtime_error(label + " timed result has an unexpected size.");
        }

        timings.push_back(std::chrono::duration<double>(end - start).count());
        // Checksum is intentionally consumed outside the timed section to prevent the compiler
        // from optimizing away the work while preserving the same measurement setup for each kernel.
        (void)checksum_result(result);
    }

    return timings;
}

void benchmark_case(std::size_t matrix_size) {
    const auto a = make_deterministic_matrix(matrix_size, matrix_size, 11);
    const auto b = make_deterministic_matrix(matrix_size, matrix_size, 17);

    const auto naive_timings = benchmark_kernel(a, b, matrix_size, "naive", naive_kernel);
    const auto naive_result = matmul_naive(a, b, matrix_size, matrix_size, matrix_size);
    const double naive_median = median_latency(naive_timings);

    std::cout << "build_type=" << ACHILLES_BUILD_TYPE << "\n";
    std::cout << "matrix_size=" << matrix_size << " | warmup_iterations=" << kWarmupIterations
              << " | timed_repetitions=" << kTimedRepetitions << "\n";
    std::cout << "matrix_size | implementation | block_size | median_latency_ms | gflops | speedup_vs_naive\n";

    const auto print_row = [&](const std::string& implementation,
                              std::size_t block_size,
                              double median_latency_ms,
                              double gflops,
                              double speedup) {
        std::cout << matrix_size << " | " << implementation << " | "
                  << (block_size == 0 ? "N/A" : std::to_string(block_size)) << " | "
                  << median_latency_ms << " | " << gflops << " | " << speedup << "\n";
    };

    // The naive kernel is used as the reference baseline for all speedup calculations.
    print_row("naive", 0,
              naive_median * 1.0e3,
              (2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
               static_cast<double>(matrix_size)) / (naive_median * 1.0e9),
              1.0);

    for (const std::size_t block_size : kBlockSizes) {
        const auto blocked_timings = benchmark_kernel(a, b, matrix_size, "blocked", blocked_kernel, block_size);
        const auto blocked_result = matmul_blocked(a, b, matrix_size, matrix_size, matrix_size, block_size);

        if (!nearly_equal(naive_result, blocked_result, kTolerance)) {
            throw std::runtime_error("Naive and blocked results differ beyond the tolerance for benchmark size " +
                                     std::to_string(matrix_size) + " and block size " + std::to_string(block_size) + ".");
        }

        const double blocked_median = median_latency(blocked_timings);
        const double blocked_gflops = (2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
                                     static_cast<double>(matrix_size)) /
                                    (blocked_median * 1.0e9);
        const double blocked_speedup = naive_median / blocked_median;

        print_row("blocked", block_size, blocked_median * 1.0e3, blocked_gflops, blocked_speedup);
    }

    const auto simd_timings = benchmark_kernel(a, b, matrix_size, "simd", simd_kernel);
    const auto simd_result = matmul_simd(a, b, matrix_size, matrix_size, matrix_size);

    if (!nearly_equal(naive_result, simd_result, kTolerance)) {
        throw std::runtime_error("Naive and SIMD results differ beyond the tolerance for benchmark size " +
                                 std::to_string(matrix_size) + ".");
    }

    const double simd_median = median_latency(simd_timings);
    const double simd_gflops = (2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
                               static_cast<double>(matrix_size)) /
                              (simd_median * 1.0e9);
    const double simd_speedup = naive_median / simd_median;

    print_row("simd", 0, simd_median * 1.0e3, simd_gflops, simd_speedup);

    for (const std::size_t block_size : kBlockSizes) {
        const auto blocked_simd_timings = benchmark_kernel(a, b, matrix_size, "blocked_simd", blocked_simd_kernel, block_size);
        const auto blocked_simd_result = matmul_blocked_simd(a, b, matrix_size, matrix_size, matrix_size, block_size);

        if (!nearly_equal(naive_result, blocked_simd_result, kTolerance)) {
            throw std::runtime_error("Naive and blocked SIMD results differ beyond the tolerance for benchmark size " +
                                     std::to_string(matrix_size) + " and block size " + std::to_string(block_size) + ".");
        }

        const double blocked_simd_median = median_latency(blocked_simd_timings);
        const double blocked_simd_gflops = (2.0 * static_cast<double>(matrix_size) * static_cast<double>(matrix_size) *
                                           static_cast<double>(matrix_size)) /
                                          (blocked_simd_median * 1.0e9);
        const double blocked_simd_speedup = naive_median / blocked_simd_median;

        print_row("blocked_simd", block_size, blocked_simd_median * 1.0e3, blocked_simd_gflops, blocked_simd_speedup);
    }
}

}  // namespace

int main() {
    std::cout << "build_type=" << ACHILLES_BUILD_TYPE << "\n";
    std::cout << "tested_matrix_sizes=";
    for (std::size_t i = 0; i < kMatrixSizes.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << kMatrixSizes[i];
    }
    std::cout << "\n";
    std::cout << "tested_block_sizes=";
    for (std::size_t i = 0; i < kBlockSizes.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << kBlockSizes[i];
    }
    std::cout << "\n";

    for (const std::size_t size : kMatrixSizes) {
        benchmark_case(size);
    }
    return 0;
}
