use ruststat::{bernoulli_cdf, bernoulli_per, beta_cdf, beta_per, bin_cdf, bin_per,
               burr_cdf, burr_per, cauchy_cdf, cauchy_per, chisq_cdf, chisq_per,
               discrete_unif_cdf, discrete_unif_per, exp_cdf, exp_per, f_cdf, f_per,
               frechet_cdf, frechet_per, gamma_cdf, gamma_per, geo2_cdf, geo2_per,
               geo_cdf, geo_per, gumbel_cdf, gumbel_per, hg_cdf, hg_per,
               invgamma_cdf, invgamma_per, invgauss_cdf, invgauss_per,
               laplace_cdf, laplace_per, logistic_cdf, logistic_per,
               loglogistic_cdf, loglogistic_per, lognormal_cdf, lognormal_per,
               nb2_cdf, nb2_per, nb_cdf, nb_per, normal_cdf, normal_per,
               pareto4_cdf, pareto4_per, pois_cdf, pois_per, rayleigh_cdf, rayleigh_per,
               t_cdf, t_per, triangular_cdf, triangular_per, unif_cdf, unif_per,
               weibull_cdf, weibull_per, zinb_cdf, zinb_per, zip_cdf, zip_per};

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
fn test_distribution_percentile_roundtrips() {
    let test_percentiles = vec![0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999];
    let tolerance = 1e-10;

    for &p in &test_percentiles {
        // ========================================================
        // CONTINUOUS DISTRIBUTIONS (Exact round-tripping expected)
        // ========================================================

        // Test Beta(0.5, 2.0)
        let x_beta = beta_per(p, 0.5, 2.0);
        assert_approx_eq!(p, beta_cdf(x_beta, 0.5, 2.0), tolerance);

        // Test Burr(1.0, 2.0, 3.0)
        let x_burr = burr_per(p, 1.0, 2.0, 3.0);
        assert_approx_eq!(p, burr_cdf(x_burr, 1.0, 2.0, 3.0), tolerance);

        // Test Cauchy(1.0, 2.0)
        let x_cauchy = cauchy_per(p, 1.0, 2.0);
        assert_approx_eq!(p, cauchy_cdf(x_cauchy, 1.0, 2.0), tolerance);

        // Test Chi-Square(8.0)
        let x_chisq = chisq_per(p, 8.0);
        assert_approx_eq!(p, chisq_cdf(x_chisq, 8.0), tolerance);

        // Test Exponential(3.5)
        let x_exp = exp_per(p, 3.5);
        assert_approx_eq!(p, exp_cdf(x_exp, 3.5), tolerance);

        // Test F(4.0, 18.0)
        let x_f = f_per(p, 4.0, 18.0);
        assert_approx_eq!(p, f_cdf(x_f, 4.0, 18.0), tolerance);

        // Test Frechet(1.0, 2.0, 3.0)
        let x_frechet = frechet_per(p, 1.0, 2.0, 3.0);
        assert_approx_eq!(p, frechet_cdf(x_frechet, 1.0, 2.0, 3.0), tolerance);

        // Test Gamma(4.0, 2.0)
        let x_gamma = gamma_per(p, 4.0, 2.0);
        assert_approx_eq!(p, gamma_cdf(x_gamma, 4.0, 2.0), tolerance);

        // Test Gumbel(4.2, 1.8)
        let x_gumbel = gumbel_per(p, 4.2, 1.8);
        assert_approx_eq!(p, gumbel_cdf(x_gumbel, 4.2, 1.8), tolerance);

        // Test Inverse Gamma(2.0, 3.0)
        let x_invgamma = invgamma_per(p, 2.0, 3.0);
        assert_approx_eq!(p, invgamma_cdf(x_invgamma, 2.0, 3.0), tolerance);

        // Test Inverse Gaussian(1.0, 2.0)
        let x_invgauss = invgauss_per(p, 1.0, 2.0);
        assert_approx_eq!(p, invgauss_cdf(x_invgauss, 1.0, 2.0), 1e-6); // Uses Bisection, limit tolerance slightly

        // Test Laplace(5.0, 2.0)
        let x_laplace = laplace_per(p, 5.0, 2.0);
        assert_approx_eq!(p, laplace_cdf(x_laplace, 5.0, 2.0), tolerance);

        // Test Logistic(0.0, 1.0)
        let x_logistic = logistic_per(p, 0.0, 1.0);
        assert_approx_eq!(p, logistic_cdf(x_logistic, 0.0, 1.0), tolerance);

        // Test Log-Logistic(1.0, 2.0)
        let x_loglogistic = loglogistic_per(p, 1.0, 2.0);
        assert_approx_eq!(p, loglogistic_cdf(x_loglogistic, 1.0, 2.0), tolerance);

        // Test Log-Normal(0.0, 1.0)
        let x_lognormal = lognormal_per(p, 0.0, 1.0);
        assert_approx_eq!(p, lognormal_cdf(x_lognormal, 0.0, 1.0), tolerance);

        // Test Normal(100.0, 16.0)
        let x_norm = normal_per(p, 100.0, 16.0);
        assert_approx_eq!(p, normal_cdf(x_norm, 100.0, 16.0), tolerance);

        // Test Pareto4(0.0, 1.0, 2.0, 0.5)
        let x_pareto = pareto4_per(p, 0.0, 1.0, 2.0, 0.5);
        assert_approx_eq!(p, pareto4_cdf(x_pareto, 0.0, 1.0, 2.0, 0.5), tolerance);

        // Test Rayleigh(2.0)
        let x_rayleigh = rayleigh_per(p, 2.0);
        assert_approx_eq!(p, rayleigh_cdf(x_rayleigh, 2.0), tolerance);

        // Test Student's t(25.0)
        let x_t = t_per(p, 25.0);
        assert_approx_eq!(p, t_cdf(x_t, 25.0), tolerance);

        // Test Triangular(1.0, 10.0, 4.0)
        let x_triangular = triangular_per(p, 1.0, 10.0, 4.0);
        assert_approx_eq!(p, triangular_cdf(x_triangular, 1.0, 10.0, 4.0), tolerance);

        // Test Uniform(1.0, 5.0)
        let x_unif = unif_per(p, 1.0, 5.0);
        assert_approx_eq!(p, unif_cdf(x_unif, 1.0, 5.0), tolerance);

        // Test Weibull(2.0, 1.5)
        let x_weibull = weibull_per(p, 2.0, 1.5);
        assert_approx_eq!(p, weibull_cdf(x_weibull, 2.0, 1.5), tolerance);

        // ========================================================
        // DISCRETE DISTRIBUTIONS (Step functions: cdf(x) >= p)
        // ========================================================

        // Test Bernoulli(0.7)
        let x_bernoulli = bernoulli_per(p, 0.7);
        assert!(bernoulli_cdf(x_bernoulli, 0.7) >= p - tolerance);

        // Test Binomial(100, 0.9)
        let x_bin = bin_per(p, 100, 0.9);
        assert!(bin_cdf(x_bin, 100, 0.9) >= p - tolerance);

        // Test Discrete Uniform(1, 5)
        let x_discrete_unif = discrete_unif_per(p, 1, 5);
        assert!(discrete_unif_cdf(x_discrete_unif, 1, 5) >= p - tolerance);

        // Test Geometric(0.2)
        let x_geo = geo_per(p, 0.2);
        assert!(geo_cdf(x_geo, 0.2) >= p - tolerance);

        // Test Geometric2(0.2)
        let x_geo2 = geo2_per(p, 0.2);
        assert!(geo2_cdf(x_geo2, 0.2) >= p - tolerance);

        // Test Hypergeometric(20, 100, 50)
        let x_hg = hg_per(p, 20, 100, 50);
        assert!(hg_cdf(x_hg, 20, 100, 50) >= p - tolerance);

        // Test Negative Binomial(4, 0.2)
        let x_nb = nb_per(p, 4, 0.2);
        assert!(nb_cdf(x_nb, 4, 0.2) >= p - tolerance);

        // Test Negative Binomial 2(4, 0.2)
        let x_nb2 = nb2_per(p, 4, 0.2);
        assert!(nb2_cdf(x_nb2, 4, 0.2) >= p - tolerance);

        // Test Poisson(2.5)
        let x_pois = pois_per(p, 2.5);
        assert!(pois_cdf(x_pois, 2.5) >= p - tolerance);

        // Test ZIP(2.5, 0.2)
        let x_zip = zip_per(p, 2.5, 0.2);
        assert!(zip_cdf(x_zip, 2.5, 0.2) >= p - tolerance);

        // Test ZINB(4, 0.2, 0.15)
        let x_zinb = zinb_per(p, 4, 0.2, 0.15);
        assert!(zinb_cdf(x_zinb, 4, 0.2, 0.15) >= p - tolerance);
    }
}

#[test]
fn test_chisq_large_parameter_regression() {
    let nu = 500.0;
    let target_p = 0.2;

    // 1. Find the x value for the 20th percentile
    let x = chisq_per(target_p, nu);

    // Ensure the root finder didn't crash
    assert!(x > 0.0, "Root finder failed and returned {}", x);

    // 2. Feed x back into the CDF
    let calc_p = chisq_cdf(x, nu);

    // 3. Check for the round-trip match
    assert_approx_eq!(target_p, calc_p, 1e-6);
}

#[test]
fn test_gamma_large_parameter_regression() {
    let alpha = 500.0;
    let beta = 0.5;
    let target_p = 0.2;

    // 1. Find the x value for the 20th percentile
    let x = gamma_per(target_p, alpha, beta);

    // Ensure the root finder didn't crash to 0.0 or NaN
    assert!(x > 0.0, "Root finder failed and returned {}", x);

    // 2. Feed x back into the CDF to see if we get 0.2 back
    let calc_p = gamma_cdf(x, alpha, beta);

    // 3. Check for the round-trip match within a tight tolerance
    assert_approx_eq!(target_p, calc_p, 1e-6);
}

