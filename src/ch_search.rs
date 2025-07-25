use core::f64;

/// CH Search Algorithm (constrained SE)
/// Input: R (n x n, upper triangular), y_bar (n x 1), l_bar (n x 1, lower bound), u_bar (n x 1, upper bound)
/// Output: z_hat (n x 1, integer)

pub fn ch_search(R: &Vec<Vec<f64>>, y_bar: &Vec<f64>, l_bar: &Vec<i32>, u_bar: &Vec<i32>) -> Vec<i32> {
    // Matrix sizing
    let n = R.len();
    
    // Initializing
    let mut i = n-1;
    let mut beta = f64::INFINITY;

    let mut c = vec![0.0; n];
    let mut delta = vec![0i32; n];

    let mut lb = vec![0i32; n];
    let mut ub = vec![0i32; n];

    // Buffer for z
    let mut z = vec![0i32; n];

    // Optimal solution
    let mut z_hat = vec![0i32; n];

    // Looping
    loop {
        // Compute c[i]
        let mut sum = 0.0;
        for k in i+1..n {
            sum += R[i][k] * z[k] as f64;
        }
        c[i] = (y_bar[i] - sum) / R[i][i];

        // Set z[i], lb[i], ub[i]
        z[i] = c[i].round() as i32;
        lb[i] = 0;
        ub[i] = 0;

        // Checking lower bound
        if z[i] < l_bar[i] {
            lb[i] = 1;
            z[i] = l_bar[i];
            delta[i] = 1;
        }
        // Checking upper bound
        else if z[i] > u_bar[i] {
            ub[i] = 1;
            z[i] = u_bar[i];
            delta[i] = -1;
        }
        // Else z[i] in bounds
        else {
            delta[i] = (c[i] - z[i] as f64).signum() as i32;
        }

        // Inner loop #1
        loop {
            // Pruning check
            let check = R[i][i].powi(2) * (z[i] as f64 - c[i]).powi(2);
            let mut tail_sum = 0.0;
            for k in i+1..n {
                tail_sum += R[k][k].powi(2) * (z[k] as f64 - c[k]).powi(2);
            }

            if check + tail_sum > beta.powi(2) {
                loop {
                    // Optimal solution found
                    if i == n-1 {
                        return z_hat;
                    }

                    // Otherwise
                    else {
                        i += 1;

                        // Enumeration
                        if ub[i] == 1 && lb[i] == 1 {
                            continue;
                        }

                        z[i] += delta[i];

                        if z[i] == l_bar[i] {
                            lb[i] = 1;
                            delta[i] = -delta[i] - delta[i].signum();
                        }

                        else if z[i] == u_bar[i] {
                            ub[i] = 1;
                            delta[i] = -delta[i] - delta[i].signum();
                        }

                        else if lb[i] == 1 {
                            delta[i] = 1;
                        }

                        else if ub[i] == 1 {
                            delta[i] = -1;
                        }

                        else {
                            delta[i] = -delta[i] - delta[i].signum();
                            break;
                        }
                    }
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

                // Update z_hat
                z_hat.copy_from_slice(&z);

                // Update and check
                i += 1;
                if ub[i] == 1 && lb[i] == 1 {
                    if i == n-1 {
                        return z_hat;
                    }
                    else {
                        i +=1
                    }
                }

                // Enumeration
                z[i] += delta[i];

                if z[i] == l_bar[i] {
                    lb[i] = 1;
                    delta[i] = -delta[i] - delta[i].signum();
                }

                else if z[i] == u_bar[i] {
                    ub[i] = 1;
                    delta[i] = -delta[i] - delta[i].signum();
                }

                else if lb[i] == 1 {
                    delta[i] = 1;
                }

                else if ub[i] == 1 {
                    delta[i] = -1;
                }

                else {
                    delta[i] = -delta[i] - delta[i].signum();
                    continue;
                }
            }
        }
    }
}


/// Modified CH Search Algorithm (constrained SE)
/// Input: R (n x n, upper triangular), y_bar (n x 1), l_bar (n x 1, lower bound), u_bar (n x 1, upper bound), 
/// z_approx (n x 1, integer), delta_approx (n x 1, integer), T (usize), zeta1 (bool), zeta2 (bool)
/// Output: z_hat (n x 1, integer), delta (n x 1, integer), T (usize), lb (n x 1, integer), ub (n x 1, integer)

pub fn ch_search_mod(R: &Vec<Vec<f64>>, y_bar: &Vec<f64>, l_bar: &Vec<i32>, u_bar: &Vec<i32>, z_approx: &Vec<i32>, delta_approx: &Vec<i32>, T: usize, zeta1: bool, zeta2: bool, lb_init: &Vec<i32>, ub_init: &Vec<i32>) -> (Vec<i32>, Vec<i32>, usize, Vec<i32>, Vec<i32>) {
    // Matrix sizing
    let n = R.len();

    // Optimal solution
    let mut z = vec![0i32; n];

    // Buffer for z
    let mut z_hat = vec![0i32; n];

    // Parameters
    let mut ub = ub_init.clone();
    let mut lb = lb_init.clone();
    let mut delta = delta_approx.clone();

    let mut c = vec![0.0; n];
    let mut T = T;

    // Initializaing beta
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
        skip_first_compute = true;
        i = 1;

        // Line 33
        if ub[i] == 1 && lb[i] == 1 {
            // Optimal solution found
            if i == n-1 {
                return (z, delta, T, lb, ub)
            }

            // Otherwise
            else {
                i +=1;

                // Line 32
                let mut norm = 0.0;
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in j..n {
                        sum += R[j][k] * z_hat[k] as f64;
                    }
                    norm += (y_bar[j] - sum).powi(2);
                }
                beta = norm.sqrt();
                i += 1;

                // Update z_hat
                z_hat[i] += delta[i];

                // Enumerate
                if z_hat[i] == l_bar[i] {
                    lb[i] = 1;
                    delta[i] = -delta[i] - delta[i].signum();
                }

                else if z_hat[i] == u_bar[i] {
                    ub[i] = 1;
                    delta[i] = -delta[i] - delta[i].signum();
                }

                else if lb[i] == 1 {
                    delta[i] = 1;
                }

                else if ub[i] == 1 {
                    delta[i] = -1;
                }

                else {
                    delta[i] = -delta[i] - delta[i].signum();
                }
            }
        }
    }
    else {
        i = n-1;
    }

    // Looping
    loop {
        if !skip_first_compute {
            let mut sum = 0.0;
            for p in i+1..n {
                sum += R[i][p] * z_hat[p] as f64;
            }
            c[i] = (y_bar[i] - sum) / R[i][i];

            // Compute z_hat[i]
            z_hat[i] = c[i].round() as i32;

            // Set lb[i] and ub[i]
            lb[i] = 0;
            ub[i] = 0;

            // Checking lower bound
            if z_hat[i] < l_bar[i] {
                lb[i] = 1;
                z_hat[i] = l_bar[i];
                delta[i] = 1;
            }
            // Checking upper bound
            else if z_hat[i] > u_bar[i] {
                ub[i] = 1;
                z_hat[i] = u_bar[i];
                delta[i] = -1;
            }
            // Else z[i] in bounds
            else {
                delta[i] = (c[i] - z[i] as f64).signum() as i32;
            } 
        } else {
            skip_first_compute = false;
        }

        // Inner loop
        loop {
            // Pruning check
            let check = R[i][i].powi(2) * (z_hat[i] as f64 - c[i]).powi(2);
            let mut tail_sum = 0.0;
            for k in i+1..n {
                tail_sum += R[k][k].powi(2) * (z_hat[k] as f64 - c[k]).powi(2);
            }

            if check + tail_sum >= beta.powi(2) {
                loop {
                    if i == n-1 {
                        return (z, delta, T, lb, ub);
                    }
                    else {
                        i +=1;

                        // Line 32
                        let mut norm = 0.0;
                        for j in 0..n {
                            let mut sum = 0.0;
                            for k in j..n {
                                sum += R[j][k] * z_hat[k] as f64;
                            }
                            norm += (y_bar[j] - sum).powi(2);
                        }
                        beta = norm.sqrt();
                        i += 1;

                        // Update z_hat
                        z_hat[i] += delta[i];

                        // Enumerate
                        if z_hat[i] == l_bar[i] {
                            lb[i] = 1;
                            delta[i] = -delta[i] - delta[i].signum();
                        }

                        else if z_hat[i] == u_bar[i] {
                            ub[i] = 1;
                            delta[i] = -delta[i] - delta[i].signum();
                        }

                        else if lb[i] == 1 {
                            delta[i] = 1;
                        }

                        else if ub[i] == 1 {
                            delta[i] = -1;
                        }

                        else {
                            delta[i] = -delta[i] - delta[i].signum();
                        }
                        break;
                    }
                }
            }

            // Otherwise
            else if i > 0 {
                i -= 1;
                break;
            }

            // Valid point found
            else {
                z.copy_from_slice(&z_hat);
                T -= 1;

                if T == 0 {
                    return (z, delta, T, lb, ub);
                }

                // Update beta as norm
                let mut norm = 0.0;
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in j..n {
                        sum += R[j][k] * z_hat[k] as f64;
                    }
                    norm += (y_bar[j] - sum).powi(2);
                }
                beta = norm.sqrt();
                i += 1;

                if ub[i] == 1 && lb[i] == 1 {
                    if i == n-1 {
                        return (z_hat, delta, T, lb, ub)
                    }

                    // Otherwise
                    else {
                        i +=1;

                        // Line 32
                        let mut norm = 0.0;
                        for j in 0..n {
                            let mut sum = 0.0;
                            for k in j..n {
                                sum += R[j][k] * z_hat[k] as f64;
                            }
                            norm += (y_bar[j] - sum).powi(2);
                        }
                        beta = norm.sqrt();
                        i += 1;

                        // Update z_hat
                        z_hat[i] += delta[i];

                        // Enumerate
                        if z_hat[i] == l_bar[i] {
                            lb[i] = 1;
                            delta[i] = -delta[i] - delta[i].signum();
                        }

                        else if z_hat[i] == u_bar[i] {
                            ub[i] = 1;
                            delta[i] = -delta[i] - delta[i].signum();
                        }

                        else if lb[i] == 1 {
                            delta[i] = 1;
                        }

                        else if ub[i] == 1 {
                            delta[i] = -1;
                        }

                        else {
                            delta[i] = -delta[i] - delta[i].signum();
                        }
                        continue;
                    }
                }
            }
        }
    }
}
