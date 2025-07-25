use rayon::prelude::*;
use rand::prelude::*;


// HELPER METHODS, SEQUENTIAL

pub fn norm2(A: &Vec<Vec<f64>>, col: usize) -> f64 {
    let mut sum = 0.0;
    for row in A {
            sum += row[col] * row[col];
        }
    sum.sqrt()
}

pub fn normalize(A: &mut Vec<Vec<f64>>, col: usize, factor: f64) {
    for row in A.iter_mut() {
        row[col] /= factor;
    }
}

pub fn dot_prod(A: &Vec<Vec<f64>>, col1: usize, col2: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..A.len() {
        sum += A[i][col1] * A[i][col2];
    }
    sum
}

pub fn subtract_proj(A: &mut Vec<Vec<f64>>, col1: usize, col2: usize, factor: f64) {
    for row in A.iter_mut() {
        row[col2] -= row[col1] * factor;
    }
}

pub fn norm2_squared(A: &Vec<Vec<f64>>, col: usize) -> f64 {
    let mut sum = 0.0;
    for row in A {
        sum += row[col] * row[col];
    }
    sum
}

pub fn givens_rotation(R: &mut Vec<Vec<f64>>, y: &mut Vec<f64>, k: usize) {
    let a = R[k-1][k-1];
    let b = R[k][k-1];
    let r = (a.powi(2) + b.powi(2)).sqrt();

    if r == 0.0 {return;}

    let (c, s) = (a/r, b/r);

    for j in k-1..R[0].len() {
        let temp1 = c*R[k-1][j] + s*R[k][j];
        let temp2 = -s*R[k-1][j] + c*R[k][j];
        R[k-1][j] = temp1;
        R[k][j] = temp2;
    }

    let temp1 = c*y[k-1] + s*y[k];
    let temp2 = -s*y[k-1] + c*y[k];
    y[k-1] = temp1;
    y[k] = temp2;
}


// HELPER METHODS, PARALLEL

pub fn compute_identity(n: usize) -> Vec<Vec<i32>> {
    (0..n).into_par_iter()
        .map(|i| {
            let mut row = vec![0; n];
            row[i] = 1;
            row
    })
    .collect()
}

pub fn matrix_vector_multiply(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    a.par_iter()
        .map(|row| row.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
        .collect()
}

pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    (0..cols).into_par_iter()
        .map(|j| {
            (0..rows).map(|i| matrix[i][j]).collect()
        })
        .collect()
}

pub fn _norm2_par(A: &Vec<Vec<f64>>, col: usize) -> f64 {
    A.par_iter()
        .map(|row| row[col]*row[col])
        .sum::<f64>()
        .sqrt()
}

pub fn _normalize_par(A: &mut Vec<Vec<f64>>, col: usize, factor: f64) {
    A.par_iter_mut().for_each(|row| {
        row[col] /= factor;
    });
}

pub fn _dot_prod_par(A: &Vec<Vec<f64>>, col1: usize, col2: usize) -> f64 {
    A.par_iter()
        .map(|row| row[col1] * row[col2])
        .sum()
}

pub fn _subtract_proj_par(A: &mut Vec<Vec<f64>>, col1: usize, col2: usize, factor: f64) {
    A.par_iter_mut().for_each(|row| {
        row[col2] -= row[col1] * factor;
    });
}

pub fn _norm2_squared_par(A: &Vec<Vec<f64>>, col: usize) -> f64 {
    A.par_iter()
        .map(|row| row[col]*row[col])
        .sum::<f64>()
}


// HELPER METHODS, MAIN

// Fn to print matrix nicely
pub fn _print_matrix(mat: &Vec<Vec<f64>>) {
    for row in mat {
        for val in row {
            print!("{:8.4} ", val);
        }
        println!();
    }
}

// Fn to generate large matrix
pub fn generate_matrix(m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..m)
        .map(|_| {
            (0..n).map(|_| rng.gen::<f64>()).collect()
        })
        .collect()
}

// Fn to generate UT matrix
pub fn generate_ut_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut R = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            if i == j {
                // SET RANGE
                R[i][j] = rng.gen_range(0.0..=10.0);
            } else {
                // SET RANG
                R[i][j] = rng.gen_range(0.0..=10.0);
            }
        }
    }
    R
}

// Fn to generate a y_bar given R and true Z
pub fn generate_y_bar(R: &Vec<Vec<f64>>, z_true: &Vec<i32>) -> Vec<f64> {
    let n = R.len();
    let mut y_bar = vec![0.0; n];
    for i in 0..n {
        for j in i..n {
            y_bar[i] += R[i][j] * z_true[j] as f64;
        }
    }
    y_bar
}
