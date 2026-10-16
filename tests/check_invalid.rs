#[test]
fn test_invalid_parameters_yield_nan() {
    // Beta (alpha > 0, beta > 0)
    assert!(ruststat::beta_pdf(0.5, 0.0, 1.0).is_nan());
    assert!(ruststat::beta_pdf(0.5, 1.0, -1.0).is_nan());

    // Chi-Square (nu > 0)
    assert!(ruststat::chisq_pdf(2.0, 0.0).is_nan());
    assert!(ruststat::chisq_pdf(2.0, -2.5).is_nan());

    // Exponential (lambda > 0)
    assert!(ruststat::exp_pdf(1.0, 0.0).is_nan());
    assert!(ruststat::exp_pdf(1.0, -2.5).is_nan());
    assert!(ruststat::exp_per(0.5, -2.5).is_nan()); // Includes our newly fixed function

    // F-Distribution (nu1 > 0, nu2 > 0)
    assert!(ruststat::f_pdf(1.0, 0.0, 5.0).is_nan());
    assert!(ruststat::f_pdf(1.0, 5.0, -2.0).is_nan());

    // Gamma (alpha > 0, beta > 0)
    assert!(ruststat::gamma_pdf(1.0, -1.0, 2.0).is_nan());
    assert!(ruststat::gamma_pdf(1.0, 2.0, 0.0).is_nan());

    // Gumbel (beta > 0)
    assert!(ruststat::gumbel_pdf(1.0, 0.0, 0.0).is_nan());
    assert!(ruststat::gumbel_pdf(1.0, 0.0, -1.0).is_nan());

    // Logistic (s > 0)
    assert!(ruststat::logistic_pdf(1.0, 0.0, 0.0).is_nan());
    assert!(ruststat::logistic_pdf(1.0, 0.0, -1.5).is_nan());

    // Log-Normal (sigma > 0)
    assert!(ruststat::lognormal_pdf(1.0, 0.0, 0.0).is_nan());
    assert!(ruststat::lognormal_pdf(1.0, 0.0, -1.0).is_nan());

    // Normal (sigma > 0)
    assert!(ruststat::normal_pdf(0.0, 0.0, 0.0).is_nan());
    assert!(ruststat::normal_pdf(0.0, 0.0, -1.0).is_nan());

    // Pareto Type IV (sigma > 0, gamma > 0, alpha > 0)
    assert!(ruststat::pareto4_pdf(1.0, 0.0, 0.0, 1.0, 1.0).is_nan());
    assert!(ruststat::pareto4_pdf(1.0, 0.0, 1.0, -1.0, 1.0).is_nan());
    assert!(ruststat::pareto4_pdf(1.0, 0.0, 1.0, 1.0, -1.0).is_nan());

    // Student's t (nu > 0)
    assert!(ruststat::t_pdf(0.0, 0.0).is_nan());
    assert!(ruststat::t_pdf(0.0, -1.0).is_nan());

    // Uniform (a < b)
    assert!(ruststat::unif_pdf(0.5, 5.0, 1.0).is_nan()); // a > b
    assert!(ruststat::unif_pdf(0.5, 5.0, 5.0).is_nan()); // a = b

    // Binomial (n >= 1, 0 <= p <= 1)
    assert!(ruststat::bin_pmf(5, 0, 0.5).is_nan());
    assert!(ruststat::bin_pmf(5, 10, -0.1).is_nan());
    assert!(ruststat::bin_pmf(5, 10, 1.5).is_nan());

    // Geometric (0 < p <= 1)
    assert!(ruststat::geo_pmf(2, -0.1).is_nan());
    assert!(ruststat::geo_pmf(2, 1.5).is_nan());

    // Geometric 2 (0 < p <= 1)
    assert!(ruststat::geo2_pmf(2, -0.1).is_nan());
    assert!(ruststat::geo2_pmf(2, 1.5).is_nan());

    // Hypergeometric (n >= 1, N >= 1, M <= N, n <= N)
    assert!(ruststat::hg_pmf(2, 0, 10, 5).is_nan());  // Trial count < 1
    assert!(ruststat::hg_pmf(2, 5, 0, 0).is_nan());   // Population < 1
    assert!(ruststat::hg_pmf(2, 5, 10, 15).is_nan()); // Successes in pop > Population
    assert!(ruststat::hg_pmf(2, 15, 10, 5).is_nan()); // Trial count > Population

    // Negative Binomial (r >= 1, 0 < p <= 1)
    assert!(ruststat::nb_pmf(2, 0, 0.5).is_nan());
    assert!(ruststat::nb_pmf(2, 3, -0.1).is_nan());
    assert!(ruststat::nb_pmf(2, 3, 1.5).is_nan());

    // Negative Binomial 2 (r >= 1, 0 < p <= 1)
    assert!(ruststat::nb2_pmf(2, 0, 0.5).is_nan());
    assert!(ruststat::nb2_pmf(2, 3, -0.1).is_nan());
    assert!(ruststat::nb2_pmf(2, 3, 1.5).is_nan());

    // Poisson (lambda > 0)
    assert!(ruststat::pois_pmf(2, 0.0).is_nan());
    assert!(ruststat::pois_pmf(2, -1.0).is_nan());
}
