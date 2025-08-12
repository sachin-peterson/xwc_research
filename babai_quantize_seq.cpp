#include "babai_quantize_seq.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

// Babai Quantize, Sequential
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>>
babai_quantize_seq(
    const std::vector<std::vector<double>>& W,
    const std::vector<std::vector<double>>& S,
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& T_in,
    const std::vector<int>& z,
    double lambda
) {
    // Matrix sizing
    size_t c = W.size();
    size_t r = W[0].size();

    // H = Tᵀ (Xᵀ X + λ I) T
    auto XTX = matmul_f64(transpose(X), X);
    auto H0  = add_scalar_identity(XTX, lambda);

    // Minmimum pivot
    auto T = min_pivot(H0);
    auto H = matmul_f64(matmul_f64(transpose(T), H0), T);

    // A = cholesky(H)ᵀ
    auto L = cholesky(H);
    auto A = transpose(L);

    // W ← T⁻¹W, S ← T⁻¹S
    auto T_inv   = inv_perm(T);
    auto W_tilde = matmul_f64(T_inv, W);
    auto S_tilde = matmul_f64(T_inv, S);

    // Y, Q, Z ← AW, W, 0
    auto Y = matmul_f64(A, W_tilde);
    auto Q = W_tilde;
    std::vector<std::vector<int>> Z(c, std::vector<int>(r, 0));

    // Looping from c..1
    for (size_t j = c; j --> 0;) {
        // ω ← Y[j,:] / A[j,j]
        std::vector<double> omega(r);
        for (size_t col = 0; col < r; ++col) {
            omega[col] = Y[j][col] / A[j][j];
        }

        // ζ ← ω / S[j,:]
        std::vector<double> zeta(r);
        for (size_t col = 0; col < r; ++col) {
            zeta[col] = omega[col] / S_tilde[j][col];
        }

        // Z[j,:] = ROUND(ζ, z-grid)
        // Q[j,:] = Z[j,:] * S[j,:]
        for (size_t col = 0; col < r; ++col) {
            int zi = nearest_in_grid(zeta[col], z);
            Z[j][col] = zi;
            Q[j][col] = static_cast<double>(zi) * S_tilde[j][col];
        }

        // Y ← Y − A[:,j] Q[j,:]
        for (size_t row = 0; row < c; ++row) {
            double a_rj = A[row][j];
            if (a_rj != 0.0) {
                for (size_t col = 0; col < r; ++col) {
                    Y[row][col] -= a_rj * Q[j][col];
                }
            }
        }
    }

    // Z, Q ← TZ, TQ
    auto Z_final = matmul_i32(T, Z);
    auto Q_final = matmul_f64(T, Q);
    return {Z_final, Q_final};
}

// Minimum pivot
std::vector<std::vector<double>> min_pivot(
    const std::vector<std::vector<double>>& H
) {
    // Matrix sizing
    size_t n = H.size();

    // Working copy
    auto H_copy = H;

    // Index tracking
    std::vector<size_t> remaining; remaining.reserve(n);
    for (size_t i = 0; i < n; ++i) remaining.push_back(i);

    // Permutation list
    std::vector<size_t> perm(n, 0);

    // Permutation matrix
    std::vector<std::vector<double>> T(n, std::vector<double>(n, 0.0));

    // Looping
    for (size_t j = 0; j < n; ++j) {
        // Pivot w/ smallest diagonal entry
        size_t best_idx = 0;
        double best_val = std::numeric_limits<double>::infinity();

        for (size_t idx = 0; idx < remaining.size(); ++idx) {
            size_t row_idx = remaining[idx];
            double diag_val = std::abs(H_copy[row_idx][row_idx]);
            if (diag_val < best_val) {
                best_val = diag_val;
                best_idx = idx;
            }
        }
        size_t jp = remaining[best_idx];
        perm[j] = jp;

        // Complement update
        double pivot_val = H_copy[jp][jp];
        if (std::abs(pivot_val) > 1e-12) {
            // H(:,j')
            std::vector<double> col_jp(n);
            for (size_t i = 0; i < n; ++i) col_jp[i] = H_copy[i][jp];

            // H(j',:)
            std::vector<double> row_jp = H_copy[jp];

            // H = H - H(:,j') * H(j',:) / H(j',j')
            const double inv_pivot = 1.0 / pivot_val;
            for (size_t i = 0; i < n; ++i) {
                double ci = col_jp[i];
                if (ci != 0.0) {
                    double scale = ci * inv_pivot;
                    for (size_t k = 0; k < n; ++k) {
                        H_copy[i][k] -= scale * row_jp[k];
                    }
                }
            }
        }

        // Remove pivot from set
        remaining.erase(std::remove(remaining.begin(), remaining.end(), jp), remaining.end());
    }

    // Update T
    for (size_t j = 0; j < n; ++j) T[perm[j]][j] = 1.0;

    // Return T
    return T;
}

// Grid rounding
int nearest_in_grid(
    double val, const std::vector<int>& grid
) {
    int best = grid[0];
    double best_dist = std::abs(val - static_cast<double>(grid[0]));

    for (size_t i = 1; i < grid.size(); ++i) {
        double dist = std::abs(val - static_cast<double>(grid[i]));
        if (dist < best_dist) {best_dist = dist; best = grid[i];}
    }
    return best;
}


// HELPER METHODS

// f64 matrix multiplication
std::vector<std::vector<double>> matmul_f64(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
 ) {
    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();
    std::vector<std::vector<double>> res(m, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t p = 0; p < k; ++p) {
                res[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return res;
 }

// i32 matrix multiplication
std::vector<std::vector<int>> matmul_i32(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<int>>& B
) {
    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();

    std::vector<std::vector<int>> res(m, std::vector<int>(n, 0)); 
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t p = 0; p < k; ++p) {
                res[i][j] += static_cast<int>(std::round(A[i][p] * static_cast<double>(B[p][j])));
            }
        }
    }
    return res;
}

// matrix transpose
std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& A
) {
    size_t m = A.size();
    size_t n = A[0].size();

    std::vector<std::vector<double>> res(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            res[j][i] = A[i][j];
        }
    }
    return res;
}

// invert permutation
std::vector<std::vector<double>> inv_perm(
    const std::vector<std::vector<double>>& T
) {
    size_t n = T.size();

    std::vector<std::vector<double>> res(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (T[i][j] == 1.0) {
                res[j][i] = 1.0;
            }
        }
    }
    return res;
}

// cholesky decomposition
std::vector<std::vector<double>> cholesky(
    const std::vector<std::vector<double>>& A
) {
    // Matrix sizing
    size_t n = A.size();

    // Storing result
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    // Decomposition
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;

            // Diagonal summation
            if (j == i) {
                for (size_t k = 0; k < j; ++k) {
                    sum += pow(L[j][k], 2);
                }
                L[j][j] = sqrt(A[j][j] - sum);
            }

            // L[i,j] from L[j,j]
            else {
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
   return L;
}

// add scalar identity
std::vector<std::vector<double>> add_scalar_identity(
    const std::vector<std::vector<double>>& A,
    double lambda
) {
    size_t n = A.size();

    auto res = A;
    for (size_t i = 0; i < n; ++i) {
        res[i][i] += lambda;
    }
    return res;
}
