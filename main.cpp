#include "babai_quantize_seq.hpp"
#include "babai_quantize_par.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
#include <random>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>>
babai_quantize_seq(
    const std::vector<std::vector<double>>& W,
    const std::vector<std::vector<double>>& S,
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& T_in,
    const std::vector<int>& z,
    double lambda
);

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>>
babai_quantize_par(
    const std::vector<std::vector<double>>& W,
    const std::vector<std::vector<double>>& S,
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& T_in,
    const std::vector<int>& z,
    double lambda
);

// Benchmarking
using Clock = std::chrono::high_resolution_clock;

template <typename T>
std::vector<std::vector<T>> make_matrix(size_t rows, size_t cols, std::mt19937_64& rng, double a, double b) {
    std::uniform_real_distribution<double> dist(a, b);
    std::vector<std::vector<T>> M(rows, std::vector<T>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            M[i][j] = static_cast<T>(dist(rng));
    return M;
}
std::vector<std::vector<double>> eye(size_t n) {
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}
std::vector<int> range_int(int lo, int hi_inclusive) {
    std::vector<int> v; v.reserve(hi_inclusive - lo + 1);
    for (int x = lo; x <= hi_inclusive; ++x) v.push_back(x);
    return v;
}


int main() {
    // Matrix dimensions
    const size_t c = 512, r = 1024, n = 2048;

    // Reproducible RNG
    std::mt19937_64 rng(42);

    // Inputs for testing
    auto W = make_matrix<double>(c, r, rng, -100.0, 100.0);
    auto S = make_matrix<double>(c, r, rng, 0.1, 1.0);
    auto X = make_matrix<double>(n, c, rng, -1.0, 1.0);
    auto T = eye(c);
    auto z = range_int(-8, 7);
    double lambda = 0.01;

    // OpenMP check
    #ifndef _OPENMP
        std::cout << "OpenMP: OFF\n";
    #else
        std::cout << "OpenMP: ON, max_threads = " << omp_get_max_threads() << "\n";
        int team = 1;
        #pragma omp parallel
        {
            #pragma omp single
            team = omp_get_num_threads();  // actual threads in a parallel region
        }
        std::cout << "Parallel region threads = " << team << "\n";
    #endif

    // Sequential
    auto t0 = Clock::now();
    auto [Z_seq, Q_seq] = babai_quantize_seq(W, S, X, T, z, lambda);
    auto t1 = Clock::now();

    // Parallel
    auto t2 = Clock::now();
    auto [Z_par, Q_par] = babai_quantize_par(W, S, X, T, z, lambda);
    auto t3 = Clock::now();

    // Timing
    auto ms_seq = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto ms_par = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Sequential Babai Quantize: " << ms_seq << " ms\n";
    std::cout << "Parallel Babai Quantize:   " << ms_par << " ms\n";
}
