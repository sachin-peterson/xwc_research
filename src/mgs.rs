use crate::utils::*;
use rayon::prelude::*;

/// QR factorization using Modified Gram-Schmit (MGS)
/// Input – A (m x n)
/// Output – R (n x n upper triangular)
/// Note: A -> Q, in place modification

/// SEQUENTIAL
pub fn mgs_seq(A: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    // Matrix sizing
    let n = A[0].len();

    // Initialize R
    let mut R = vec![vec![0.0; n]; n];

    // Outer loop
    for i in 0..n {
        // R[i][i] = 2-norm of i-th column
        R[i][i] = norm2(&A, i);

        // Normalize Q[i]
        normalize(A, i, R[i][i]);

        // Inner loop
        for j in i+1..n {
            // Project Q[j] onto Q[i]
            R[i][j] = dot_prod(&A, i, j);

            // Subtract projection
            subtract_proj(A, i, j, R[i][j]);
        }
    }
    R
}

/// PARALLEL
pub fn mgs_par(A: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    // Matrix sizing
    let n = A[0].len();

    // Initialize R
    let mut R = vec![vec![0.0; n]; n];

    // Outer loop
    for i in 0..n {
        // R[i][i] = 2-norm of i-th column
        R[i][i] = norm2(&A, i);

        // Normalize Q[i]
        normalize(A, i, R[i][i]);

        // Compute dot products in parallel
        let rij_vec: Vec<(usize, f64)> = (i+1..n)
            .into_par_iter()
            .map(|j| (j, dot_prod(A, i, j)))
            .collect();

        // Update A and R using rij values
        for (j, rij) in rij_vec {
            R[i][j] = rij;
            subtract_proj(A, i, j, rij);
        }
    }
    R
}
