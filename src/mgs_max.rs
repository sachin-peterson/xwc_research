use crate::utils::*;
use rayon::prelude::*;

/// QRP factorization w/ Maximum Column Pivoting
/// Input – A (m x n)
/// Output – R (n x n upper triangular), P (n x n permutation)
/// Note: A -> Q, in place modification

/// SEQUENTIAL
pub fn mgs_max_seq(A: &mut Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<i32>>) {
    // Matrix sizing
    let n = A[0].len();

    // Initializing
    let mut R = vec![vec![0.0; n]; n];
    let mut P = compute_identity(n);
    let mut s = vec![vec![0.0; n]; 2];

    // Pivot vector
    for i in 0..n {
        s[0][i] = norm2_squared(A, i);
    }

    // Outer loop
    for i in 0..n {
        // Finding pivot
        let mut l = i;
        let mut max_val = s[0][i] - s[1][i];
        for k in (i+1)..n {
            let diff = s[0][k] - s[1][k];
            if diff > max_val {
                max_val = diff;
                l = k;
            }
        }

        // Swapping
        if l != i {
            A.iter_mut().for_each(|row| row.swap(i,l));
            R.swap(i,l);
            s[0].swap(i,l);
            s[1].swap(i,l);
            P.iter_mut().for_each(|row| row.swap(i,l));
        }

        // Normalizing Q[i]
        R[i][i] = norm2(A, i);
        normalize(A, i, R[i][i]);

        // Project and update
        for j in i+1..n {
            R[i][j] = dot_prod(A, i, j);
            s[1][j] += R[i][j].powi(2);
            subtract_proj(A, i, j, R[i][j]);
        }
    }
    (R, P)
}

/// PARALLEL
pub fn mgs_max_par(A: &mut Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<i32>>) {
    // Matrix sizing
    let n = A[0].len();

    // Initializing
    let mut R = vec![vec![0.0; n]; n];
    let mut P = compute_identity(n);
    let mut s = vec![vec![0.0; n]; 2];

    // Pivot vector, parallel
    s[0].par_iter_mut()
        .enumerate()
        .for_each(|(i,val)| {
            *val = norm2_squared(A,i);
        });

    // Outer loop, sequential
    for i in 0..n {
        // Find pivot
        let (l,_) = (i..n)
            .map(|k| (k, s[0][k] - s[1][k]))
            .max_by(|a,b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        // Swapping
        if l != i {
            A.iter_mut().for_each(|row| row.swap(i, l));
            R.swap(i, l);
            s[0].swap(i, l);
            s[1].swap(i, l);
            P.iter_mut().for_each(|row| row.swap(i, l));
        }

        // Normalize Q[i]
        R[i][i] = norm2(A, i);
        normalize(A, i, R[i][i]);

        // Compute dot products in parallel
        let rij_vec: Vec<(usize, f64)> = (i+1..n)
            .into_par_iter()
            .map(|j| (j, dot_prod(A, i, j)))
            .collect();

        // Updating A and R accordingly
        for (j, rij) in rij_vec {
            R[i][j] = rij;
            s[1][j] += rij * rij;
            subtract_proj(A, i, j, rij);
        }
    }
    (R,P)
}
