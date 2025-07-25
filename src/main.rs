#![allow(non_snake_case)]

use std::time::Instant;

mod utils; use crate::utils::{generate_matrix, generate_ut_matrix, generate_y_bar};

mod mgs; use crate::mgs::{mgs_seq, mgs_par};
mod mgs_max; use crate::mgs_max::{mgs_max_seq, mgs_max_par};
mod mgs_min; use crate::mgs_min::{mgs_min_seq, mgs_min_par};

mod partial_lll; use crate::partial_lll::{plll_seq, plll_par};

mod se_search; use crate::se_search::{se_search, se_search_mod};


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
        let mut y1 = vec![0.0; 4];
        let start = Instant::now();
        plll_seq(&mut Q7, &mut y1);
        let duration = start.elapsed();
        println!("Sequential PLLL:");
        println!("Duration: {:?}\n", duration);
    }

    // Test parallel PLLL
    {
        let mut y2 = vec![0.0; 4];
        let start = Instant::now();
        plll_par(&mut Q8, &mut y2);
        let duration = start.elapsed();
        println!("Parallel PLLL:");
        println!("Duration: {:?}\n", duration);
    }

    // Test Schnorr-Euchner
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

    // Test modified Schnorr-Euchner
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

    // 
}
