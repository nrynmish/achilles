#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "achilles/matmul.hpp"

namespace {

constexpr float kTolerance = 1.0e-6f;
constexpr float kBlockedTolerance = 1.0e-5f;

bool nearly_equal(float lhs, float rhs, float tolerance = kTolerance) {
    return std::fabs(lhs - rhs) <= tolerance;
}

bool expect_vector_equal(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        const std::string& label,
                        float tolerance = kTolerance) {
    if (actual.size() != expected.size()) {
        std::cout << "[FAIL] " << label << ": size mismatch (actual=" << actual.size()
                  << ", expected=" << expected.size() << ")\n";
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!nearly_equal(actual[i], expected[i], tolerance)) {
            std::cout << "[FAIL] " << label << ": value mismatch at index " << i
                      << " (actual=" << actual[i] << ", expected=" << expected[i] << ")\n";
            return false;
        }
    }

    std::cout << "[PASS] " << label << "\n";
    return true;
}

bool test_1x1_multiplication() {
    const std::vector<float> a = {3.0f};
    const std::vector<float> b = {4.0f};
    const std::vector<float> expected = {12.0f};

    const auto actual = matmul_naive(a, b, 1, 1, 1);
    return expect_vector_equal(actual, expected, "1x1 multiplication");
}

bool test_rectangular_multiplication() {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};

    const auto actual = matmul_naive(a, b, 2, 2, 3);
    return expect_vector_equal(actual, expected, "rectangular 2x3 by 3x2 multiplication");
}

bool test_identity_matrix_multiplication() {
    const std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    const std::vector<float> b = {2.0f, 3.0f, 5.0f, 7.0f, 11.0f, 13.0f, 17.0f, 19.0f, 23.0f};
    const std::vector<float> expected = b;

    const auto actual = matmul_naive(a, b, 3, 3, 3);
    return expect_vector_equal(actual, expected, "identity matrix multiplication");
}

bool test_zero_matrices() {
    const std::vector<float> a(8, 0.0f);
    const std::vector<float> b(12, 0.0f);
    const std::vector<float> expected(6, 0.0f);

    const auto actual = matmul_naive(a, b, 2, 3, 4);
    return expect_vector_equal(actual, expected, "zero-valued matrices");
}

bool test_zero_sized_dimensions() {
    const std::vector<float> empty;
    const auto result1 = matmul_naive(empty, empty, 0, 0, 0);
    const auto result2 = matmul_naive(empty, empty, 0, 5, 0);
    const auto result3 = matmul_naive(empty, empty, 5, 0, 0);
    const auto blocked1 = matmul_blocked(empty, empty, 0, 0, 0);
    const auto blocked2 = matmul_blocked(empty, empty, 0, 5, 0);
    const auto blocked3 = matmul_blocked(empty, empty, 5, 0, 0);

    if (result1.size() != 0 || result2.size() != 0 || result3.size() != 0 ||
        blocked1.size() != 0 || blocked2.size() != 0 || blocked3.size() != 0) {
        std::cout << "[FAIL] zero-sized dimensions: unexpected non-empty result\n";
        return false;
    }

    std::cout << "[PASS] zero-sized dimensions\n";
    return true;
}

bool test_a_storage_size_mismatch_throws() {
    try {
        (void)matmul_naive({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, 3, 2, 2);
        std::cout << "[FAIL] incorrect A storage size did not throw std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] incorrect A storage size throws std::invalid_argument\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] incorrect A storage size threw unexpected exception: " << ex.what() << "\n";
        return false;
    }
}

bool test_b_storage_size_mismatch_throws() {
    try {
        (void)matmul_naive({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f}, 2, 2, 2);
        std::cout << "[FAIL] incorrect B storage size did not throw std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] incorrect B storage size throws std::invalid_argument\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] incorrect B storage size threw unexpected exception: " << ex.what() << "\n";
        return false;
    }
}

bool test_tolerance_documented() {
    // Absolute tolerance used for floating-point comparisons: 1e-6 per element.
    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const std::vector<float> expected = {40.0f, 46.0f};
    const auto actual = matmul_naive(a, b, 1, 2, 3);

    if (actual.size() != expected.size()) {
        std::cout << "[FAIL] tolerance test: result size mismatch\n";
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!nearly_equal(actual[i], expected[i], kTolerance)) {
            std::cout << "[FAIL] tolerance test: mismatch at index " << i << " (actual="
                      << actual[i] << ", expected=" << expected[i] << ")\n";
            return false;
        }
    }

    std::cout << "[PASS] floating-point tolerance check uses absolute tolerance " << kTolerance << "\n";
    return true;
}

bool test_blocked_matches_naive_rectangular() {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const auto expected = matmul_naive(a, b, 2, 2, 3);
    const auto actual = matmul_blocked(a, b, 2, 2, 3, 32);
    return expect_vector_equal(actual, expected, "blocked output matches naive for rectangular matrices");
}

bool test_blocked_matches_naive_non_multiple_of_32() {
    const std::size_t rows = 19;
    const std::size_t cols = 17;
    const std::size_t inner = 13;

    std::vector<float> a(rows * inner, 0.0f);
    std::vector<float> b(inner * cols, 0.0f);

    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>((i % 7) + 1) / 3.0f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>((i % 11) + 2) / 5.0f;
    }

    const auto expected = matmul_naive(a, b, rows, cols, inner);
    const auto actual = matmul_blocked(a, b, rows, cols, inner, 32);
    return expect_vector_equal(actual, expected, "blocked output matches naive when dimensions are not multiples of 32");
}

bool test_blocked_matches_naive_for_multiple_block_sizes() {
    const std::vector<float> a = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f,
        49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f, 64.0f
    };
    const std::vector<float> b = {
        2.0f, 0.0f, 1.0f, 3.0f,
        1.0f, 4.0f, 2.0f, 0.0f,
        3.0f, 1.0f, 0.0f, 2.0f,
        0.0f, 2.0f, 5.0f, 1.0f,
        1.0f, 3.0f, 2.0f, 4.0f,
        5.0f, 0.0f, 1.0f, 2.0f,
        4.0f, 2.0f, 3.0f, 0.0f,
        2.0f, 1.0f, 4.0f, 3.0f
    };

    const auto expected = matmul_naive(a, b, 8, 4, 8);
    bool all_ok = true;
    for (std::size_t block_size : {1u, 2u, 16u, 32u, 64u}) {
        const auto actual = matmul_blocked(a, b, 8, 4, 8, block_size);
        if (!expect_vector_equal(actual, expected, "blocked output matches naive for block size " + std::to_string(block_size))) {
            all_ok = false;
        }
    }
    return all_ok;
}

bool test_zero_block_size_throws() {
    try {
        (void)matmul_blocked({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, 2, 2, 2, 0);
        std::cout << "[FAIL] zero block size did not throw std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] zero block size throws std::invalid_argument\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] zero block size threw unexpected exception: " << ex.what() << "\n";
        return false;
    }
}

bool test_blocked_rejects_invalid_a_storage() {
    try {
        (void)matmul_blocked({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, 3, 2, 2, 32);
        std::cout << "[FAIL] invalid A storage was not rejected by matmul_blocked\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] invalid A storage is rejected by matmul_blocked\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] invalid A storage threw unexpected exception: " << ex.what() << "\n";
        return false;
    }
}

bool test_blocked_rejects_invalid_b_storage() {
    try {
        (void)matmul_blocked({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f}, 2, 2, 2, 32);
        std::cout << "[FAIL] invalid B storage was not rejected by matmul_blocked\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] invalid B storage is rejected by matmul_blocked\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] invalid B storage threw unexpected exception: " << ex.what() << "\n";
        return false;
    }
}

bool test_blocked_zero_sized_dimensions() {
    const std::vector<float> empty;
    const auto result = matmul_blocked(empty, empty, 0, 4, 0, 16);
    if (result.size() != 0) {
        std::cout << "[FAIL] blocked zero-sized dimensions produced a non-empty matrix\n";
        return false;
    }
    std::cout << "[PASS] blocked zero-sized dimensions\n";
    return true;
}

bool test_blocked_randomized_small_matrices() {
    std::mt19937 generator(12345);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    const std::size_t rows = 5;
    const std::size_t cols = 4;
    const std::size_t inner = 6;

    std::vector<float> a(rows * inner, 0.0f);
    std::vector<float> b(inner * cols, 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(generator);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = dist(generator);
    }

    const auto expected = matmul_naive(a, b, rows, cols, inner);
    const auto actual = matmul_blocked(a, b, rows, cols, inner, 2);
    return expect_vector_equal(actual, expected, "randomized small matrices match the naive reference", kBlockedTolerance);
}

bool test_blocked_simd_matches_naive_rectangular() {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const auto expected = matmul_naive(a, b, 2, 2, 3);
    const auto actual = matmul_blocked_simd(a, b, 2, 2, 3, 8);
    return expect_vector_equal(actual, expected, "blocked SIMD output matches naive for rectangular matrices", 1.0e-5f);
}

bool test_blocked_simd_matches_naive_small_matrices() {
    const std::vector<float> a = {2.0f, -1.0f, 3.0f, 0.5f, 4.0f, -2.0f};
    const std::vector<float> b = {1.0f, 2.0f, -3.0f, 0.25f, 4.0f, 5.0f};
    const auto expected = matmul_naive(a, b, 2, 2, 3);
    const auto actual = matmul_blocked_simd(a, b, 2, 2, 3, 2);
    return expect_vector_equal(actual, expected, "blocked SIMD output matches naive for small matrices", 1.0e-5f);
}

bool test_blocked_simd_matches_naive_non_multiple_of_8() {
    const std::size_t rows = 11;
    const std::size_t cols = 9;
    const std::size_t inner = 7;

    std::vector<float> a(rows * inner, 0.0f);
    std::vector<float> b(inner * cols, 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>((i % 13) + 1) / 3.0f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>((i % 9) + 2) / 5.0f;
    }

    const auto expected = matmul_naive(a, b, rows, cols, inner);
    const auto actual = matmul_blocked_simd(a, b, rows, cols, inner, 8);
    return expect_vector_equal(actual, expected, "blocked SIMD output matches naive for non-multiple-of-8 widths", 1.0e-5f);
}

bool test_blocked_simd_matches_naive_randomized_sizes() {
    std::mt19937 generator(77777);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    const std::vector<std::size_t> rows_list = {1u, 3u, 5u, 7u, 9u};
    const std::vector<std::size_t> col_list = {2u, 4u, 6u, 8u, 10u};
    const std::vector<std::size_t> inner_list = {2u, 3u, 5u, 7u, 11u};

    bool all_ok = true;
    for (std::size_t rows : rows_list) {
        for (std::size_t cols : col_list) {
            for (std::size_t inner : inner_list) {
                std::vector<float> a(rows * inner, 0.0f);
                std::vector<float> b(inner * cols, 0.0f);
                for (std::size_t i = 0; i < a.size(); ++i) {
                    a[i] = dist(generator);
                }
                for (std::size_t i = 0; i < b.size(); ++i) {
                    b[i] = dist(generator);
                }

                const auto expected = matmul_naive(a, b, rows, cols, inner);
                const auto actual = matmul_blocked_simd(a, b, rows, cols, inner, 8);
                if (!expect_vector_equal(actual, expected, "blocked SIMD output matches naive for randomized dimensions", 1.0e-5f)) {
                    all_ok = false;
                }
            }
        }
    }
    return all_ok;
}

bool test_blocked_simd_zero_size_block_throws() {
    try {
        (void)matmul_blocked_simd({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, 2, 2, 2, 0);
        std::cout << "[FAIL] zero block size did not throw std::invalid_argument for matmul_blocked_simd\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] zero block size throws std::invalid_argument for matmul_blocked_simd\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] zero block size threw unexpected exception for matmul_blocked_simd: " << ex.what() << "\n";
        return false;
    }
}

bool test_blocked_simd_zero_sized_dimensions() {
    const std::vector<float> empty;
    const auto result = matmul_blocked_simd(empty, empty, 0, 7, 0, 8);
    if (result.size() != 0) {
        std::cout << "[FAIL] blocked SIMD zero-sized dimensions produced a non-empty matrix\n";
        return false;
    }
    std::cout << "[PASS] blocked SIMD zero-sized dimensions\n";
    return true;
}

bool test_simd_matches_naive_rectangular() {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const auto expected = matmul_naive(a, b, 2, 2, 3);
    const auto actual = matmul_simd(a, b, 2, 2, 3);
    return expect_vector_equal(actual, expected, "SIMD output matches naive for rectangular matrices", 1.0e-5f);
}

bool test_simd_matches_naive_smaller_than_vector_width() {
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> b = {2.0f, 1.0f, 0.0f, 3.0f, 4.0f, 5.0f};
    const auto expected = matmul_naive(a, b, 2, 2, 3);
    const auto actual = matmul_simd(a, b, 2, 2, 3);
    return expect_vector_equal(actual, expected, "SIMD output matches naive below vector width", 1.0e-5f);
}

bool test_simd_matches_naive_non_multiple_of_8() {
    const std::size_t rows = 11;
    const std::size_t cols = 9;
    const std::size_t inner = 7;

    std::vector<float> a(rows * inner, 0.0f);
    std::vector<float> b(inner * cols, 0.0f);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>((i % 13) + 1) / 3.0f;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>((i % 9) + 2) / 5.0f;
    }

    const auto expected = matmul_naive(a, b, rows, cols, inner);
    const auto actual = matmul_simd(a, b, rows, cols, inner);
    return expect_vector_equal(actual, expected, "SIMD output matches naive for non-multiple-of-8 widths", 1.0e-5f);
}

bool test_simd_matches_naive_randomized_sizes() {
    std::mt19937 generator(54321);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    const std::vector<std::size_t> rows_list = {1u, 3u, 5u, 7u, 9u};
    const std::vector<std::size_t> col_list = {2u, 4u, 6u, 8u, 10u};
    const std::vector<std::size_t> inner_list = {2u, 3u, 5u, 7u, 11u};

    bool all_ok = true;
    for (std::size_t rows : rows_list) {
        for (std::size_t cols : col_list) {
            for (std::size_t inner : inner_list) {
                std::vector<float> a(rows * inner, 0.0f);
                std::vector<float> b(inner * cols, 0.0f);
                for (std::size_t i = 0; i < a.size(); ++i) {
                    a[i] = dist(generator);
                }
                for (std::size_t i = 0; i < b.size(); ++i) {
                    b[i] = dist(generator);
                }

                const auto expected = matmul_naive(a, b, rows, cols, inner);
                const auto actual = matmul_simd(a, b, rows, cols, inner);
                if (!expect_vector_equal(actual, expected,
                                        "SIMD output matches naive for randomized dimensions",
                                        1.0e-5f)) {
                    all_ok = false;
                }
            }
        }
    }
    return all_ok;
}

bool test_simd_zero_sized_dimensions() {
    const std::vector<float> empty;
    const auto result = matmul_simd(empty, empty, 0, 7, 0);
    if (result.size() != 0) {
        std::cout << "[FAIL] SIMD zero-sized dimensions produced a non-empty matrix\n";
        return false;
    }
    std::cout << "[PASS] SIMD zero-sized dimensions\n";
    return true;
}

bool test_simd_rejects_invalid_a_storage() {
    try {
        (void)matmul_simd({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}, 3, 2, 2);
        std::cout << "[FAIL] invalid A storage was not rejected by matmul_simd\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] invalid A storage is rejected by matmul_simd\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] invalid A storage threw unexpected exception: " << ex.what() << "\n";
        return false;
    }
}

bool test_simd_rejects_invalid_b_storage() {
    try {
        (void)matmul_simd({1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f}, 2, 2, 2);
        std::cout << "[FAIL] invalid B storage was not rejected by matmul_simd\n";
        return false;
    } catch (const std::invalid_argument&) {
        std::cout << "[PASS] invalid B storage is rejected by matmul_simd\n";
        return true;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] invalid B storage threw unexpected exception: " << ex.what() << "\n";
        return false;
    }
}

bool test_all_implementations_agree() {
    const std::vector<float> a = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f
    };
    const std::vector<float> b = {
        2.0f, 0.0f, 1.0f, 3.0f,
        1.0f, 4.0f, 2.0f, 0.0f,
        3.0f, 1.0f, 0.0f, 2.0f,
        0.0f, 2.0f, 5.0f, 1.0f
    };

    const auto naive_out = matmul_naive(a, b, 6, 4, 4);
    const auto blocked_out = matmul_blocked(a, b, 6, 4, 4, 32);
    const auto simd_out = matmul_simd(a, b, 6, 4, 4);

    const bool blocked_ok = expect_vector_equal(blocked_out, naive_out, "blocked matches naive in all-implementations agreement", 1.0e-5f);
    const bool simd_ok = expect_vector_equal(simd_out, naive_out, "SIMD matches naive in all-implementations agreement", 1.0e-5f);
    return blocked_ok && simd_ok;
}

}  // namespace

int main() {
    bool all_ok = true;

    const auto run_test = [&](const char* label, bool (*test_fn)()) {
        try {
            if (!test_fn()) {
                all_ok = false;
            }
        } catch (const std::exception& ex) {
            std::cout << "[FAIL] " << label << ": unexpected exception: " << ex.what() << "\n";
            all_ok = false;
        }
    };

    run_test("test_1x1_multiplication", test_1x1_multiplication);
    run_test("test_rectangular_multiplication", test_rectangular_multiplication);
    run_test("test_identity_matrix_multiplication", test_identity_matrix_multiplication);
    run_test("test_zero_matrices", test_zero_matrices);
    run_test("test_zero_sized_dimensions", test_zero_sized_dimensions);
    run_test("test_a_storage_size_mismatch_throws", test_a_storage_size_mismatch_throws);
    run_test("test_b_storage_size_mismatch_throws", test_b_storage_size_mismatch_throws);
    run_test("test_tolerance_documented", test_tolerance_documented);
    run_test("test_blocked_matches_naive_rectangular", test_blocked_matches_naive_rectangular);
    run_test("test_blocked_matches_naive_non_multiple_of_32", test_blocked_matches_naive_non_multiple_of_32);
    run_test("test_blocked_matches_naive_for_multiple_block_sizes", test_blocked_matches_naive_for_multiple_block_sizes);
    run_test("test_zero_block_size_throws", test_zero_block_size_throws);
    run_test("test_blocked_rejects_invalid_a_storage", test_blocked_rejects_invalid_a_storage);
    run_test("test_blocked_rejects_invalid_b_storage", test_blocked_rejects_invalid_b_storage);
    run_test("test_blocked_zero_sized_dimensions", test_blocked_zero_sized_dimensions);
    run_test("test_blocked_randomized_small_matrices", test_blocked_randomized_small_matrices);
    run_test("test_blocked_simd_matches_naive_rectangular", test_blocked_simd_matches_naive_rectangular);
    run_test("test_blocked_simd_matches_naive_small_matrices", test_blocked_simd_matches_naive_small_matrices);
    run_test("test_blocked_simd_matches_naive_non_multiple_of_8", test_blocked_simd_matches_naive_non_multiple_of_8);
    run_test("test_blocked_simd_matches_naive_randomized_sizes", test_blocked_simd_matches_naive_randomized_sizes);
    run_test("test_blocked_simd_zero_size_block_throws", test_blocked_simd_zero_size_block_throws);
    run_test("test_blocked_simd_zero_sized_dimensions", test_blocked_simd_zero_sized_dimensions);
    run_test("test_simd_matches_naive_rectangular", test_simd_matches_naive_rectangular);
    run_test("test_simd_matches_naive_smaller_than_vector_width", test_simd_matches_naive_smaller_than_vector_width);
    run_test("test_simd_matches_naive_non_multiple_of_8", test_simd_matches_naive_non_multiple_of_8);
    run_test("test_simd_matches_naive_randomized_sizes", test_simd_matches_naive_randomized_sizes);
    run_test("test_simd_zero_sized_dimensions", test_simd_zero_sized_dimensions);
    run_test("test_simd_rejects_invalid_a_storage", test_simd_rejects_invalid_a_storage);
    run_test("test_simd_rejects_invalid_b_storage", test_simd_rejects_invalid_b_storage);
    run_test("test_all_implementations_agree", test_all_implementations_agree);

    if (!all_ok) {
        std::cout << "\nOne or more tests failed.\n";
        return 1;
    }

    std::cout << "\nAll tests passed.\n";
    return 0;
}
