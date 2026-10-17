#include "achilles/matmul.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#endif

#if defined(ACHILLES_ENABLE_AVX2) && (defined(__GNUC__) || defined(__clang__)) && \
    ((defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)))
#define ACHILLES_SIMD_AVX2 __attribute__((target("avx2,fma")))
#else
#define ACHILLES_SIMD_AVX2
#endif

namespace {

constexpr std::size_t kSimdWidth = 8;

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::invalid_argument(std::string("Dimension product overflow for ") + label);
    }
    return lhs * rhs;
}

void validate_matrix_storage(const std::vector<float>& a,
                            const std::vector<float>& b,
                            std::size_t m,
                            std::size_t n,
                            std::size_t k) {
    const std::size_t expected_a_size = checked_product(m, k, "A");
    const std::size_t expected_b_size = checked_product(k, n, "B");
    const std::size_t expected_output_size = checked_product(m, n, "output");

    if (a.size() != expected_a_size) {
        throw std::invalid_argument("A storage size does not match the declared M x K dimensions.");
    }
    if (b.size() != expected_b_size) {
        throw std::invalid_argument("B storage size does not match the declared K x N dimensions.");
    }

    (void)expected_output_size;
}

std::size_t safe_tile_end(std::size_t start, std::size_t tile_size, std::size_t limit) {
    if (tile_size == 0 || start >= limit) {
        return limit;
    }

    const std::size_t remaining = limit - start;
    if (tile_size >= remaining) {
        return limit;
    }
    return start + tile_size;
}

bool cpu_supports_avx2_fma() {
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__GNUC__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace

std::vector<float> matmul_naive(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k) {
    validate_matrix_storage(a, b, m, n, k);
    const std::size_t expected_output_size = checked_product(m, n, "output");
    std::vector<float> result(expected_output_size, 0.0f);

    // Zero-sized dimensions are valid if the storage matches the product of the dimensions.
    // When any dimension is zero, the result is empty and no arithmetic is needed.
    if (m == 0 || n == 0 || k == 0) {
        return result;
    }

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float sum = 0.0f;

            for (std::size_t inner = 0; inner < k; ++inner) {
                const std::size_t a_index = row * k + inner;
                const std::size_t b_index = inner * n + col;
                sum += a[a_index] * b[b_index];
            }

            result[row * n + col] = sum;
        }
    }

    return result;
}

std::vector<float> matmul_blocked(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t block_size) {
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be non-zero.");
    }

    validate_matrix_storage(a, b, m, n, k);
    const std::size_t expected_output_size = checked_product(m, n, "output");
    std::vector<float> result(expected_output_size, 0.0f);

    if (m == 0 || n == 0 || k == 0) {
        return result;
    }

    // Process the result in tiles, updating each output block with the contributions
    // from the matching A and B submatrices. This improves data reuse while preserving
    // the exact arithmetic of the naive reference implementation.
    for (std::size_t row = 0; row < m; row += block_size) {
        const std::size_t row_end = safe_tile_end(row, block_size, m);

        for (std::size_t col = 0; col < n; col += block_size) {
            const std::size_t col_end = safe_tile_end(col, block_size, n);

            for (std::size_t inner = 0; inner < k; inner += block_size) {
                const std::size_t inner_end = safe_tile_end(inner, block_size, k);

                for (std::size_t i = row; i < row_end; ++i) {
                    for (std::size_t j = col; j < col_end; ++j) {
                        float sum = result[i * n + j];

                        for (std::size_t p = inner; p < inner_end; ++p) {
                            sum += a[i * k + p] * b[p * n + j];
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
        }
    }

    return result;
}

ACHILLES_SIMD_AVX2
static std::vector<float> matmul_simd_avx2(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k) {
    const std::size_t expected_output_size = checked_product(m, n, "output");
    std::vector<float> result(expected_output_size, 0.0f);

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; col += kSimdWidth) {
            const std::size_t width = std::min<std::size_t>(kSimdWidth, n - col);

            if (width == kSimdWidth) {
                __m256 acc = _mm256_setzero_ps();
                for (std::size_t inner = 0; inner < k; ++inner) {
                    const std::size_t a_index = row * k + inner;
                    const std::size_t b_index = inner * n + col;
                    const __m256 a_vec = _mm256_set1_ps(a[a_index]);
                    const __m256 b_vec = _mm256_loadu_ps(&b[b_index]);
                    acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                }
                _mm256_storeu_ps(&result[row * n + col], acc);
            } else {
                for (std::size_t inner = 0; inner < k; ++inner) {
                    const float a_value = a[row * k + inner];
                    for (std::size_t j = 0; j < width; ++j) {
                        result[row * n + col + j] += a_value * b[inner * n + col + j];
                    }
                }
            }
        }
    }

    return result;
}

ACHILLES_SIMD_AVX2
static std::vector<float> matmul_blocked_simd_avx2(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t block_size) {
    const std::size_t expected_output_size = checked_product(m, n, "output");
    std::vector<float> result(expected_output_size, 0.0f);

    for (std::size_t row = 0; row < m; row = safe_tile_end(row, block_size, m)) {
        const std::size_t row_end = safe_tile_end(row, block_size, m);

        for (std::size_t inner = 0; inner < k; inner = safe_tile_end(inner, block_size, k)) {
            const std::size_t inner_end = safe_tile_end(inner, block_size, k);

            for (std::size_t col = 0; col < n; col = safe_tile_end(col, block_size, n)) {
                const std::size_t col_end = safe_tile_end(col, block_size, n);

                for (std::size_t i = row; i < row_end; ++i) {
                    const std::size_t a_row_offset = i * k;
                    std::size_t j = col;

                    while (j < col_end) {
                        if (col_end - j < kSimdWidth) {
                            break;
                        }

                        __m256 acc = _mm256_loadu_ps(&result[i * n + j]);
                        for (std::size_t p = inner; p < inner_end; ++p) {
                            const __m256 a_vec = _mm256_set1_ps(a[a_row_offset + p]);
                            const __m256 b_vec = _mm256_loadu_ps(&b[p * n + j]);
                            acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                        }
                        _mm256_storeu_ps(&result[i * n + j], acc);
                        j += kSimdWidth;
                    }

                    for (; j < col_end; ++j) {
                        float sum = result[i * n + j];
                        for (std::size_t p = inner; p < inner_end; ++p) {
                            sum += a[a_row_offset + p] * b[p * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }
            }
        }
    }

    return result;
}

std::vector<float> matmul_blocked_simd(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t block_size) {
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be non-zero.");
    }

    validate_matrix_storage(a, b, m, n, k);
    const std::size_t expected_output_size = checked_product(m, n, "output");
    std::vector<float> result(expected_output_size, 0.0f);

    if (m == 0 || n == 0 || k == 0) {
        return result;
    }

    if (cpu_supports_avx2_fma()) {
        return matmul_blocked_simd_avx2(a, b, m, n, k, block_size);
    }

    return matmul_blocked(a, b, m, n, k, block_size);
}

std::vector<float> matmul_simd(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::size_t m,
    std::size_t n,
    std::size_t k) {
    validate_matrix_storage(a, b, m, n, k);
    const std::size_t expected_output_size = checked_product(m, n, "output");
    std::vector<float> result(expected_output_size, 0.0f);

    if (m == 0 || n == 0 || k == 0) {
        return result;
    }

    // Runtime CPU detection remains the key portability check; we only call the AVX2/FMA
    // helper when the machine actually supports those instructions. The scalar fallback is
    // always available and is used otherwise.
    if (cpu_supports_avx2_fma()) {
        return matmul_simd_avx2(a, b, m, n, k);
    }

    return matmul_naive(a, b, m, n, k);
}
