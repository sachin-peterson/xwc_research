use core::f64;

/// Schnorr-Euchner Search Algorithm
/// Input: R (n x n, upper triangular), y_bar (n x 1)
/// Output: z_ils (n x 1, integer)

pub fn se_search(R: &Vec<Vec<f64>>, y_bar: &Vec<f64>) -> Vec<i32> {
    // Matrix sizing
    let n = R.len();

    // Buffer for z
    let mut z = vec![0i32; n];

    // Optimal solution
    let mut z_ils = vec![0i32; n];

    // Parameters
    let mut c = vec![0.0; n];
    let mut delta = vec![0.0; n];

    // Initializing
    let mut i = n-1;

    // Guess beta
    let mut beta = f64::INFINITY;

    // Looping
    loop {
        // Compute c[i]
        let mut sum = 0.0;
        for j in i+1..n {
            sum += R[i][j] * z[j] as f64;
        }
        c[i] = (y_bar[i] - sum) / R[i][i];

        // z[i] and delta[i]
        z[i] = c[i].round() as i32;
        delta[i] = (c[i] - z[i] as f64).signum();


        loop {
            // Left-side for check
            let check = R[i][i].powi(2) * (z[i] as f64 - c[i]).powi(2);

            // Right-side for checl
            let mut tail_sum = 0.0;
            for j in i+1..n {
                tail_sum += R[j][j].powi(2) * (z[j] as f64 - c[j]).powi(2);
            }

            // Check w.r.t. beta
            if check > beta.powi(2) - tail_sum {
                // Optimal solution found
                if i == n-1 {
                    return z_ils;
                }

                // Otherwise
                else {
                    i += 1;
                    z[i] += delta[i] as i32;
                    delta[i] = -delta[i] - delta[i].signum();
                    continue;
                }
            }

            // Otherwise
            else if i > 0 {
                i -= 1;
                break;
            }

            // Valid point found
            else {
                // Update beta as norm
                let mut norm = 0.0;
                for k in 0..n {
                    let mut row_sum = 0.0;
                    for j in k..n {
                        row_sum += R[k][j] * z[j] as f64;
                    }
                    norm += (y_bar[k] - row_sum).powi(2);
                }
                beta = norm.sqrt();

                // Update z_ils
                z_ils.copy_from_slice(&z);

                // Update and check
                i += 1;
                if i == n-1 {
                    return z_ils;    
                }

                // Update rest
                z[i] += delta[i] as i32;
                delta[i] = -delta[i] - delta[i].signum();
                continue;
            }
        }
    }
}


/// Modified Schnorr-Euchner Search Algorithm
/// Input: R (n x n, upper triangular), y_bar (n x 1), z_approx (n x 1, integer), 
/// delta_approx (n x 1, integer), T (usize), zeta1 (bool), zeta2 (bool)
/// Output: z (n x 1, integer), delta (n x 1, integer), T (usize)

pub fn se_search_mod(R: &Vec<Vec<f64>>, y_bar: &Vec<f64>, z_approx: &Vec<i32>, delta_approx: &Vec<i32>, T: usize, zeta1: bool, zeta2: bool) -> (Vec<i32>, Vec<i32>, usize) {
    // Matrix sizing
    let n = R.len();

    // Buffer for z
    let mut z_hat = vec![0i32; n];

    // Optimal solution
    let mut z = vec![0i32; n];

    // Parameters
    let mut delta = delta_approx.clone();
    let mut c = vec![0.0; n];
    let mut T = T;

    // Initializing beta
    let mut beta = if zeta1 {
        f64::INFINITY
    } else {
        let mut norm = 0.0;
        for i in 0..n {
            let mut sum = 0.0;
            for j in i..n {
                sum += R[i][j] * z_approx[j] as f64;
            }
            norm += (y_bar[i] - sum).powi(2);
        }
        norm.sqrt()
    };

    // Initializing i
    let mut i: usize;
    let mut skip_first_compute = false;
    if zeta2 {
        z_hat.copy_from_slice(&z_approx);
        i = 1;

        // Line 26
        z_hat[i] += delta[i];
        delta[i] = -delta[i] - delta[i].signum();

        // Skip initial computation
        skip_first_compute = true;
    } else {
        i = n-1;
    }

    // Looping
    loop {
        if !skip_first_compute {
            // Compute c[i]
            let mut sum = 0.0;
            for p in i+1..n {
                sum += R[i][p] * z_hat[p] as f64;
            }
            c[i] = (y_bar[i] - sum) / R[i][i];

            // Compute z_hat[i]
            z_hat[i] = c[i].round() as i32;

            // Compute delta[i]
            delta[i] = (c[i] - z_hat[i] as f64).signum() as i32;
        } else {
            skip_first_compute = false;
        }

        // Inner loop
        loop {
            // Left-side for check
            let check = R[i][i].powi(2) * (z_hat[i] as f64 - c[i]).powi(2);

            // Right-side for check
            let mut tail_sum = 0.0;
            for p in i+1..n {
                tail_sum += R[p][p].powi(2) * (z_hat[p] as f64 - c[p]).powi(2);
            }

            // Check w.r.t. beta
            if check >= beta.powi(2) - tail_sum {
                // Optimal solution found
                if i == n-1 {
                    return (z, delta, T);
                }

                // Otherwise
                else {
                    i += 1;
                    z_hat[i] += delta[i];
                    delta[i] = -delta[i] - delta[i].signum();
                    continue;
                }
            }

            // Otherwise
            else if i > 0 {
                i -= 1;
                break;
            }

            // Valid point found
            else {
                // Update z (optimal)
                z.copy_from_slice(&z_hat);

                // Update T
                T -= 1;
                
                if T == 0 {
                    return (z, delta, T);
                }

                // Update beta as norm
                let mut norm = 0.0;
                for k in 0..n {
                    let mut sum = 0.0;
                    for j in k..n {
                        sum += R[k][j] * z_hat[j] as f64;
                    }
                    norm += (y_bar[k] - sum).powi(2)
                }
                beta = norm.sqrt();

                // Update rest
                i += 1;
                z_hat[i] += delta[i];
                delta[i] = -delta[i] - delta[i].signum();
                continue;
            }
        }
    }
}
