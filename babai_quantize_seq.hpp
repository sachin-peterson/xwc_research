#pragma once
#include <vector>
#include <utility>

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<double>>>
babai_quantize_seq(
  const std::vector<std::vector<double>>& W,
  const std::vector<std::vector<double>>& S,
  const std::vector<std::vector<double>>& X,
  const std::vector<std::vector<double>>& T,
  const std::vector<int>& z,
  double lambda
);

std::vector<std::vector<double>> min_pivot(
  const std::vector<std::vector<double>>& H
);

int nearest_in_grid(
    double val, 
    const std::vector<int>& grid
);

std::vector<std::vector<double>> matmul_f64(
  const std::vector<std::vector<double>>& A,
  const std::vector<std::vector<double>>& B
);

std::vector<std::vector<int>> matmul_i32(
  const std::vector<std::vector<double>>& A,
  const std::vector<std::vector<int>>& B
);

std::vector<std::vector<double>> transpose(
  const std::vector<std::vector<double>>& A
);

std::vector<std::vector<double>> inv_perm(
  const std::vector<std::vector<double>>& T)
  ;

std::vector<std::vector<double>> cholesky(
  const std::vector<std::vector<double>>& A
);

std::vector<std::vector<double>> add_scalar_identity(
  const std::vector<std::vector<double>>& A, double lambda
);

