#[test]
fn test_distribution_boundaries() {
    // ========================================================
    // DISCRETE DISTRIBUTIONS (p=0, p=1, and out-of-bounds x)
    // ========================================================

    // Bernoulli
    assert_eq!(ruststat::bernoulli_pmf(2, 0.5), 0.0); // x > 1
    assert_eq!(ruststat::bernoulli_cdf(2, 0.5), 1.0); // x > 1 returns 1.0

    // Binomial
    assert_eq!(ruststat::bin_pmf(11, 10, 0.5), 0.0); // x > n
    assert_eq!(ruststat::bin_cdf(10, 10, 0.5), 1.0); // x == n returns 1.0
    assert_eq!(ruststat::bin_cdf(15, 10, 0.5), 1.0); // x > n returns 1.0
    assert_eq!(ruststat::bin_pmf(0, 10, 0.0), 1.0);  // Guaranteed failure
    assert_eq!(ruststat::bin_pmf(5, 10, 0.0), 0.0);  // Impossible success
    assert_eq!(ruststat::bin_pmf(10, 10, 1.0), 1.0); // Guaranteed success
    assert_eq!(ruststat::bin_pmf(5, 10, 1.0), 0.0);  // Impossible failure
    assert_eq!(ruststat::bin_cdf(9, 10, 1.0), 0.0);  // CDF before guaranteed success

    // Discrete Uniform
    assert_eq!(ruststat::discrete_unif_pmf(0, 1, 5), 0.0); // x < a
    assert_eq!(ruststat::discrete_unif_cdf(0, 1, 5), 0.0); // x < a
    assert_eq!(ruststat::discrete_unif_pmf(6, 1, 5), 0.0); // x > b
    assert_eq!(ruststat::discrete_unif_cdf(6, 1, 5), 1.0); // x > b

    // Geometric (failures before first success)
    assert_eq!(ruststat::geo_pmf(5, 0.0), 0.0);      // Never succeeds
    assert_eq!(ruststat::geo_pmf(0, 1.0), 1.0);      // Always succeeds on try 1 (0 failures)
    assert_eq!(ruststat::geo_pmf(5, 1.0), 0.0);

    // Geometric 2 (trial of first success)
    assert_eq!(ruststat::geo2_pmf(0, 0.5), 0.0);     // Trial count cannot be < 1
    assert_eq!(ruststat::geo2_cdf(0, 0.5), 0.0);
    assert_eq!(ruststat::geo2_pmf(5, 0.0), 0.0);     // Never succeeds
    assert_eq!(ruststat::geo2_pmf(1, 1.0), 1.0);     // Always succeeds on trial 1
    assert_eq!(ruststat::geo2_pmf(5, 1.0), 0.0);

    // Hypergeometric
    assert_eq!(ruststat::hg_pmf(11, 10, 100, 50), 0.0); // Cannot have more successes than sample size (x > n)
    assert_eq!(ruststat::hg_pmf(10, 10, 100, 5), 0.0);  // Cannot have more successes than exist in population (x > M)
    assert_eq!(ruststat::hg_pmf(0, 10, 100, 0), 1.0);   // M=0 (no successes exist)
    assert_eq!(ruststat::hg_pmf(10, 10, 100, 100), 1.0);// M=N (population is entirely successes)
    assert_eq!(ruststat::hg_cdf(15, 10, 100, 50), 1.0); // x > n returns 1.0

    // Negative Binomial (failures before r-th success)
    assert_eq!(ruststat::nb_pmf(0, 5, 1.0), 1.0);       // p=1 means exactly 0 failures to get r successes
    assert_eq!(ruststat::nb_pmf(3, 5, 1.0), 0.0);       // Impossible to fail if p=1
    assert_eq!(ruststat::nb_cdf(5, 5, 1.0), 1.0);

    // Negative Binomial 2 (trial of r-th success)
    assert_eq!(ruststat::nb2_pmf(4, 5, 0.5), 0.0);      // Cannot achieve 5 successes in 4 trials (x < r)
    assert_eq!(ruststat::nb2_cdf(4, 5, 0.5), 0.0);
    assert_eq!(ruststat::nb2_pmf(5, 5, 1.0), 1.0);      // p=1 means 5 successes take exactly 5 trials
    assert_eq!(ruststat::nb2_pmf(7, 5, 1.0), 0.0);      // Will never take > 5 trials if p=1
    assert_eq!(ruststat::nb2_cdf(5, 5, 1.0), 1.0);

    // Zero-Inflated Poisson
    assert_eq!(ruststat::zip_pmf(5, 3.0, 1.0), 0.0); // 100% chance of 0
    assert_eq!(ruststat::zip_pmf(0, 3.0, 1.0), 1.0); // 100% chance of 0
    assert_eq!(ruststat::zip_cdf(5, 3.0, 1.0), 1.0);

    // Zero-Inflated Negative Binomial
    assert_eq!(ruststat::zinb_pmf(5, 3, 0.5, 1.0), 0.0); // 100% chance of 0
    assert_eq!(ruststat::zinb_pmf(0, 3, 0.5, 1.0), 1.0); // 100% chance of 0
    assert_eq!(ruststat::zinb_cdf(5, 3, 0.5, 1.0), 1.0);

    // ========================================================
    // CONTINUOUS DISTRIBUTIONS (Support Range Limits)
    // ========================================================

    // Beta (Support: 0 < x < 1)
    assert_eq!(ruststat::beta_pdf(0.0, 2.0, 2.0), 0.0);
    assert_eq!(ruststat::beta_cdf(0.0, 2.0, 2.0), 0.0);
    assert_eq!(ruststat::beta_pdf(1.0, 2.0, 2.0), 0.0);
    assert_eq!(ruststat::beta_cdf(1.0, 2.0, 2.0), 1.0);

    // Burr Type XII (Support: x > 0)
    assert_eq!(ruststat::burr_pdf(0.0, 1.0, 2.0, 3.0), 0.0);
    assert_eq!(ruststat::burr_cdf(0.0, 1.0, 2.0, 3.0), 0.0);
    assert_eq!(ruststat::burr_pdf(-1.5, 1.0, 2.0, 3.0), 0.0);

    // Chi-Square (Support: x > 0)
    assert_eq!(ruststat::chisq_pdf(0.0, 5.0), 0.0);
    assert_eq!(ruststat::chisq_cdf(0.0, 5.0), 0.0);
    assert_eq!(ruststat::chisq_pdf(-1.5, 5.0), 0.0);

    // Exponential (Support: x > 0)
    assert_eq!(ruststat::exp_pdf(0.0, 2.0), 0.0);
    assert_eq!(ruststat::exp_cdf(0.0, 2.0), 0.0);
    assert_eq!(ruststat::exp_pdf(-1.5, 2.0), 0.0);

    // F-Distribution (Support: x > 0)
    assert_eq!(ruststat::f_pdf(0.0, 2.0, 2.0), 0.0);
    assert_eq!(ruststat::f_cdf(0.0, 2.0, 2.0), 0.0);
    assert_eq!(ruststat::f_pdf(-1.5, 2.0, 2.0), 0.0);

    // Fréchet (Support: x > mu)
    assert_eq!(ruststat::frechet_pdf(1.0, 1.0, 2.0, 3.0), 0.0); // x == mu
    assert_eq!(ruststat::frechet_cdf(1.0, 1.0, 2.0, 3.0), 0.0); // x == mu
    assert_eq!(ruststat::frechet_pdf(0.0, 1.0, 2.0, 3.0), 0.0); // x < mu

    // Gamma (Support: x > 0)
    assert_eq!(ruststat::gamma_pdf(0.0, 2.0, 2.0), 0.0);
    assert_eq!(ruststat::gamma_cdf(0.0, 2.0, 2.0), 0.0);
    assert_eq!(ruststat::gamma_pdf(-1.5, 2.0, 2.0), 0.0);

    // Inverse Gamma (Support: x > 0)
    assert_eq!(ruststat::invgamma_pdf(0.0, 2.0, 3.0), 0.0);
    assert_eq!(ruststat::invgamma_cdf(0.0, 2.0, 3.0), 0.0);
    assert_eq!(ruststat::invgamma_pdf(-1.5, 2.0, 3.0), 0.0);

    // Inverse Gaussian (Support: x > 0)
    assert_eq!(ruststat::invgauss_pdf(0.0, 1.0, 2.0), 0.0);
    assert_eq!(ruststat::invgauss_cdf(0.0, 1.0, 2.0), 0.0);
    assert_eq!(ruststat::invgauss_pdf(-1.5, 1.0, 2.0), 0.0);

    // Log-Logistic (Support: x > 0)
    assert_eq!(ruststat::loglogistic_pdf(0.0, 1.0, 2.0), 0.0);
    assert_eq!(ruststat::loglogistic_cdf(0.0, 1.0, 2.0), 0.0);
    assert_eq!(ruststat::loglogistic_pdf(-1.5, 1.0, 2.0), 0.0);

    // Log-Normal (Support: x > 0)
    assert_eq!(ruststat::lognormal_pdf(0.0, 0.0, 1.0), 0.0);
    assert_eq!(ruststat::lognormal_cdf(0.0, 0.0, 1.0), 0.0);
    assert_eq!(ruststat::lognormal_pdf(-1.5, 0.0, 1.0), 0.0);

    // Pareto Type IV (Support: x >= mu)
    assert_eq!(ruststat::pareto4_pdf(0.0, 1.0, 2.0, 3.0, 4.0), 0.0); // x < mu
    assert_eq!(ruststat::pareto4_cdf(0.0, 1.0, 2.0, 3.0, 4.0), 0.0); // x < mu

    // Rayleigh (Support: x > 0)
    assert_eq!(ruststat::rayleigh_pdf(0.0, 2.0), 0.0);
    assert_eq!(ruststat::rayleigh_cdf(0.0, 2.0), 0.0);
    assert_eq!(ruststat::rayleigh_pdf(-1.5, 2.0), 0.0);

    // Triangular (Support: a < x < b)
    assert_eq!(ruststat::triangular_pdf(0.0, 1.0, 5.0, 3.0), 0.0); // x < a
    assert_eq!(ruststat::triangular_cdf(0.0, 1.0, 5.0, 3.0), 0.0); // x < a
    assert_eq!(ruststat::triangular_pdf(6.0, 1.0, 5.0, 3.0), 0.0); // x > b
    assert_eq!(ruststat::triangular_cdf(6.0, 1.0, 5.0, 3.0), 1.0); // x > b

    // Uniform (Support: a < x < b)
    assert_eq!(ruststat::unif_pdf(1.0, 2.0, 5.0), 0.0); // x < a
    assert_eq!(ruststat::unif_cdf(1.0, 2.0, 5.0), 0.0); // x < a
    assert_eq!(ruststat::unif_pdf(6.0, 2.0, 5.0), 0.0); // x > b
    assert_eq!(ruststat::unif_cdf(6.0, 2.0, 5.0), 1.0); // x > b

    // Weibull (Support: x > 0)
    assert_eq!(ruststat::weibull_pdf(0.0, 2.0, 1.0), 0.0);
    assert_eq!(ruststat::weibull_cdf(0.0, 2.0, 1.0), 0.0);
    assert_eq!(ruststat::weibull_pdf(-1.5, 2.0, 1.0), 0.0);

}
