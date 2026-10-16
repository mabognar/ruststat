macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        let diff = (*a - *b).abs();
        assert!(
            diff < $eps,
            "Assertion failed: values not approximately equal.\n  Expected: {:?}\n  Actual: {:?}\n  Difference: {:?}",
            *a, *b, diff
        );
    }};
}

#[test]
fn test_distribution_equivalencies() {
    let tol = 1e-10;

    // 1. Beta(1, 1) is identical to Uniform(0, 1)
    assert_approx_eq!(ruststat::beta_pdf(0.4, 1.0, 1.0), ruststat::unif_pdf(0.4, 0.0, 1.0), tol);

    // 2. Chi-Square(2) is identical to Exponential(0.5)
    assert_approx_eq!(ruststat::chisq_pdf(3.0, 2.0), ruststat::exp_pdf(3.0, 0.5), tol);

    // 3. Gamma(1, lambda) is identical to Exponential(lambda)
    assert_approx_eq!(ruststat::gamma_cdf(2.5, 1.0, 3.5), ruststat::exp_cdf(2.5, 3.5), tol);

    // 4. Geometric is a Negative Binomial where r = 1
    assert_approx_eq!(ruststat::geo_pmf(3, 0.4), ruststat::nb_pmf(3, 1, 0.4), tol);
    assert_approx_eq!(ruststat::geo2_pmf(4, 0.4), ruststat::nb2_pmf(4, 1, 0.4), tol);

    // 5. Log-Normal is the exponentiation of a Normal distribution
    // P(LogNormal <= x) == P(Normal <= ln(x))
    let x = 5.0f64;
    assert_approx_eq!(
        ruststat::lognormal_cdf(x, 1.0, 2.0),
        ruststat::normal_cdf(x.ln(), 1.0, 2.0),
        tol
    );

    // 6. Chi-Square is a special case of the Gamma distribution
    // ChiSq(nu) == Gamma(alpha = nu/2, beta = 0.5)
    assert_approx_eq!(ruststat::chisq_cdf(4.5, 5.0), ruststat::gamma_cdf(4.5, 2.5, 0.5), tol);

    // 7. Student's t squared is an F-distribution
    // P(F <= x) = P(-sqrt(x) <= T <= sqrt(x))
    let f_val = 2.5f64;
    let t_val = f_val.sqrt();
    let prob_t_squared = ruststat::t_cdf(t_val, 15.0) - ruststat::t_cdf(-t_val, 15.0);
    assert_approx_eq!(ruststat::f_cdf(f_val, 1.0, 15.0), prob_t_squared, tol);

    // 8. Asymptotic: Student's t approaches Standard Normal as nu -> infinity
    assert_approx_eq!(
        ruststat::t_cdf(1.5, 1_000_000.0),
        ruststat::normal_cdf(1.5, 0.0, 1.0),
        1e-5
    );
}
