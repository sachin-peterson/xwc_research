#![allow(non_snake_case)]

use std::time::Instant;

mod utils; use crate::utils::{generate_matrix, generate_ut_matrix, generate_y_bar};

mod mgs; use crate::mgs::{mgs_seq, mgs_par};
mod mgs_max; use crate::mgs_max::{mgs_max_seq, mgs_max_par};
mod mgs_min; use crate::mgs_min::{mgs_min_seq, mgs_min_par};

mod partial_lll; use crate::partial_lll::{plll_seq, plll_par};
mod lll_permutation; use crate::lll_permutation::{lllp_seq, lllp_par};

mod se_search; use crate::se_search::{se_search, se_search_mod};
mod ch_search; use crate::ch_search::{ch_search, ch_search_mod};


fn main() {

    let A = generate_matrix(100, 100);
    let mut Q1 = A.clone();
    let mut Q2 = A.clone();
    let mut Q3 = A.clone();
    let mut Q4 = A.clone();
    let mut Q5 = A.clone();
    let mut Q6 = A.clone();
    let mut Q7 = A.clone();
    let mut Q8 = A.clone();
    let mut Q9 = A.clone();
    let mut Q10 = A.clone();

    // Test sequential QR
    {
        let start = Instant::now();
        mgs_seq(&mut Q1);
        let duration = start.elapsed();
        println!("Sequential MGS:");
        println!("Duration: {:?}\n", duration);
    }

    // Test parallel QR
    {
        let start = Instant::now();
        mgs_par(&mut Q2);
        let duration = start.elapsed();
        println!("Parallel MGS:");
        println!("Duration: {:?}\n", duration);
    }

    // Test sequential QRP w/ max pivot
    {
        let start = Instant::now();
        mgs_max_seq(&mut Q3);
        let duration = start.elapsed();
        println!("Sequential QRP w/ max pivot:");
        println!("Duration: {:?}\n", duration);
    }

    // Test parallel QRP w/ max pivot
    {
        let start = Instant::now();
        mgs_max_par(&mut Q4);
        let duration = start.elapsed();
        println!("Parallel QRP w/ max pivot:");
        println!("Duration: {:?}\n", duration);
    }

    // Test sequential QRP w/ min pivot
    {
        let start = Instant::now();
        mgs_min_seq(&mut Q5);
        let duration = start.elapsed();
        println!("Sequential QRP w/ min pivot:");
        println!("Duration: {:?}\n", duration);
    }

    // Test parallel QRP w/ min pivot
    {
        let start = Instant::now();
        mgs_min_par(&mut Q6);
        let duration = start.elapsed();
        println!("Parallel QRP w/ min pivot:");
        println!("Duration: {:?}\n", duration);
    }

    // Test sequential PLLL
    {
        let mut y1 = vec![0.0; 100];
        let start = Instant::now();
        plll_seq(&mut Q7, &mut y1);
        let duration = start.elapsed();
        println!("Sequential PLLL:");
        println!("Duration: {:?}\n", duration);
    }

    // Test parallel PLLL
    {
        let mut y2 = vec![0.0; 100];
        let start = Instant::now();
        plll_par(&mut Q8, &mut y2);
        let duration = start.elapsed();
        println!("Parallel PLLL:");
        println!("Duration: {:?}\n", duration);
    }

    // Test SE search
    {
        let n = 1000;
        let R = generate_ut_matrix(n);
        let z = vec![2; n];
        let y_bar = generate_y_bar(&R, &z);

        let start = Instant::now();
        se_search(&R, &y_bar);
        let duration = start.elapsed();
        println!("SE SEARCH, unmodified:");
        println!("Duration: {:?}\n", duration);
    }

    // Test modified SE
    {
        let n = 1000;
        let R = generate_ut_matrix(n);
        let z = vec![1; n];
        let y_bar = generate_y_bar(&R, &z);
        let z_approx = vec![0; n];
        let delta_approx = vec![0; n];

        let start = Instant::now();
        se_search_mod(&R, &y_bar, &z_approx, &delta_approx, 1, true, false);
        let duration = start.elapsed();
        println!("SE SEARCH, modified:");
        println!("Duration: {:?}\n", duration);
    }

    // Test CH search
    {
        let n = 1000;
        let R = generate_ut_matrix(n);
        let z = vec![1; n];
        let y_bar = generate_y_bar(&R, &z);
        let l_bar = vec![-5; n];
        let u_bar = vec![5; n];

        let start = Instant::now();
        ch_search(&R, &y_bar, &l_bar, &u_bar);
        let duration = start.elapsed();
        println!("CH SEARCH, unmodified:");
        println!("Duration: {:?}\n", duration);
    }

    // Test modified CH
    {
        let n = 1000;
        let R = generate_ut_matrix(n);
        let z = vec![1; n];
        let y_bar = generate_y_bar(&R, &z);
        let l_bar = vec![-5; n];
        let u_bar = vec![5; n];
        let z_approx = vec![0; n];
        let delta_approx = vec![0; n];
        let lb_init = vec![0; n];
        let ub_init = vec![0; n];

        let start = Instant::now();
        ch_search_mod(&R, &y_bar, &l_bar, &u_bar, &z_approx, &delta_approx, 1, true, false, &lb_init, &ub_init);
        let duration = start.elapsed();
        println!("CH SEARCH, modified:");
        println!("Duration: {:?}\n", duration);
    }

    // Test sequential LLL-P
    {
        let mut y3 = vec![0.0; 100];
        let start = Instant::now();
        lllp_seq(&mut Q9, &mut y3);
        let duration = start.elapsed();
        println!("Sequential LLL-P:");
        println!("Duration: {:?}\n", duration);
    }

    // Test parallel LLL-P
    {
        let mut y4 = vec![0.0; 100];
        let start = Instant::now();
        lllp_par(&mut Q10, &mut y4);
        let duration = start.elapsed();
        println!("Parallel LLL-P:");
        println!("Duration: {:?}\n", duration);
    }

    // Test all-swap sequential PLLL
    {

    }

    // Test all-swap parallel PLLL
    {
        
    }

    // Test all-swap sequential LLL-P
    {

    }

    // Test all-swap parallel LLL-P
    {

    }
}
