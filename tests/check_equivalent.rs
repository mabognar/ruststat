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
    let tol = 1e-9;

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

    // 9. Weibull(1.0, lambda) is identical to Exponential(1.0 / lambda)
    // Testing Weibull(k=1.0, lambda=2.0) == Exp(lambda=0.5)
    assert_approx_eq!(ruststat::weibull_cdf(3.0, 1.0, 2.0), ruststat::exp_cdf(3.0, 0.5), tol);

    // 10. Cauchy(0.0, 1.0) is identical to Student's t(1.0)
    assert_approx_eq!(ruststat::cauchy_cdf(2.5, 0.0, 1.0), ruststat::t_cdf(2.5, 1.0), tol);

    // 11. Bernoulli(p) is identical to Binomial(1, p)
    assert_approx_eq!(ruststat::bernoulli_pmf(1, 0.3), ruststat::bin_pmf(1, 1, 0.3), tol);
    assert_approx_eq!(ruststat::bernoulli_cdf(0, 0.3), ruststat::bin_cdf(0, 1, 0.3), tol);

    // 12. Rayleigh(sigma) is identical to Weibull(k=2.0, lambda=sigma*sqrt(2))
    assert_approx_eq!(
        ruststat::rayleigh_cdf(3.0, 2.0),
        ruststat::weibull_cdf(3.0, 2.0, 2.0 * 2.0f64.sqrt()),
        tol
    );

    // 13. Inverse Gamma relation: If X ~ Gamma(alpha, beta), then 1/X ~ InvGamma(alpha, beta).
    // Therefore P(InvGamma <= x) == 1 - P(Gamma <= 1/x).
    assert_approx_eq!(
        ruststat::invgamma_cdf(2.5, 3.0, 4.0),
        1.0 - ruststat::gamma_cdf(1.0 / 2.5, 3.0, 4.0),
        tol
    );

    // 14. Inverse Weibull relation: If X ~ Weibull(k, lambda), then 1/X ~ Frechet(mu=0, sigma=1/lambda, alpha=k).
    // Testing P(Frechet <= 0.5) == 1 - P(Weibull <= 2.0)
    assert_approx_eq!(
        ruststat::frechet_cdf(0.5, 0.0, 1.0 / 2.0, 3.0),
        1.0 - ruststat::weibull_cdf(1.0 / 0.5, 3.0, 2.0),
        tol
    );

    // 15. ZIP(lambda, 0) is identical to Pois(lambda)
    assert_approx_eq!(ruststat::zip_pmf(2, 3.5, 0.0), ruststat::pois_pmf(2, 3.5), tol);
    assert_approx_eq!(ruststat::zip_cdf(4, 3.5, 0.0), ruststat::pois_cdf(4, 3.5), tol);

    // 16. ZINB(r, p, 0) is identical to NB(r, p)
    assert_approx_eq!(ruststat::zinb_pmf(2, 4, 0.4, 0.0), ruststat::nb_pmf(2, 4, 0.4), tol);
    assert_approx_eq!(ruststat::zinb_cdf(4, 4, 0.4, 0.0), ruststat::nb_cdf(4, 4, 0.4), tol);
}
