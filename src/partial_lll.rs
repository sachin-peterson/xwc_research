use crate::mgs_min::{mgs_min_seq, mgs_min_par};
use crate::utils::*;
use rayon::prelude::*;

/// QRZ factorization via Partial LLL
/// Input – A (m x n), y (m x 1)
/// Output – R (n x n upper triangular), Z (n x n unimodular), y_bar (n x 1)
/// Note: A -> Q, in-place modification

/// SEQUENTIAL
pub fn plll_seq(A: &mut Vec<Vec<f64>>, y: &Vec<f64>) -> (Vec<Vec<f64>>, Vec<Vec<i32>>, Vec<f64>) {
    // Matrix sizing
    let n = A[0].len();
    
    // MGS QR factorization
    let (mut R, P) = mgs_min_seq(A);

    // Initializing
    let mut Z = P;
    let mut k = 1;
    let delta = 0.5;

    // Compute y_bar = Q^T * y
    let Qt = transpose(&A);
    let mut y_bar = matrix_vector_multiply(&Qt, &y);

    // Main loop
    while k < n {
        let rkk = R[k][k];
        let rkm1k = R[k-1][k];
        let rkm1km1 = R[k-1][k-1];

        let mut zeta = (rkm1k / rkm1km1).round();
        let alpha = rkm1k - zeta * rkm1km1;

        if delta*rkm1km1.powi(2) > rkk.powi(2) + alpha.powi(2) {
            if zeta != 0.0 {
                // rkm1k = alpha;

                for i in 0..(k - 1) {
                    R[i][k] -= zeta * R[i][k-1];
                }
                for i in 0..n {
                    Z[i][k] -= zeta as i32 * Z[i][k-1];
                }

                // Backwards reduction
                for i in (0..(k - 1)).rev() {
                    let rik = R[i][k];
                    let rii = R[i][i];
                    zeta = (rik / rii).round();

                    if zeta != 0.0 {
                        for j in 0..=i {
                            R[j][k] -= zeta * R[j][i];
                        }
                        for j in 0..n {
                            Z[j][k] -= zeta as i32 * Z[j][i];
                        }
                    }
                }
            }
            // Permute and triangularize
            for i in 0..n {
                R[i].swap(k-1, k);
                Z[i].swap(k-1, k);
            }
            givens_rotation(&mut R, &mut y_bar, k);

            if k > 1 {
                k -= 1;
            } else {
                k += 1;
            }
        } else {
            k += 1
        }
    }
    (R, Z, y_bar)
}

/// PARALLEL
pub fn plll_par(A: &mut Vec<Vec<f64>>, y: &Vec<f64>) -> (Vec<Vec<f64>>, Vec<Vec<i32>>, Vec<f64>) {
    // Matrix sizing
    let n = A[0].len();
    
    // MGS QR factorization, in parallel
    let (mut R, P) = mgs_min_par(A);

    // Initializing
    let mut Z = P;
    let mut k = 1;
    let delta = 0.5;

    // Compute y_bar = Q^T * y, in parallel
    let Qt = transpose(&A);
    let mut y_bar = matrix_vector_multiply(&Qt, &y);

    // Main loop
    while k < n {
        let rkk = R[k][k];
        let rkm1k = R[k-1][k];
        let rkm1km1 = R[k-1][k-1];

        let mut zeta = (rkm1k / rkm1km1).round();
        let alpha = rkm1k - zeta * rkm1km1;

        if delta*rkm1km1.powi(2) > rkk.powi(2) + alpha.powi(2) {
            if zeta != 0.0 {
                // rkm1k = alpha;

                for i in 0..(k - 1) {
                    R[i][k] -= zeta * R[i][k-1];
                }
                for i in 0..n {
                    Z[i][k] -= zeta as i32 * Z[i][k-1];
                }

                // Backwards reduction
                for i in (0..(k - 1)).rev() {
                    let rik = R[i][k];
                    let rii = R[i][i];
                    zeta = (rik/rii).round();

                    if zeta != 0.0 {
                        for j in 0..=i {
                            R[j][k] -= zeta * R[j][i];
                        }
                        for j in 0..n {
                            Z[j][k] -= zeta as i32 * Z[j][i];
                        }
                    }
                }
            }
            // Permute and triangularize
            R.par_iter_mut().for_each(|row| row.swap(k-1, k));
            Z.par_iter_mut().for_each(|row| row.swap(k-1, k));
            givens_rotation(&mut R, &mut y_bar, k);

            if k > 1 {
                k -= 1;
            } else {
                k += 1;
            }
        } else {
            k += 1;
        }
    }
    (R, Z, y_bar)
}
