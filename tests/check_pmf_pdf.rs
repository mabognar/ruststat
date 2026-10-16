use ruststat::*;

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

const TOL: f64 = 1e-8;

#[test]
fn test_pdfs() {
    
    // 1. Beta(alpha=2.0, beta=5.0) at x = 0.3
    // f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b) = 0.3^1 * 0.7^4 / (1/30) = 2.1609
    let expected_beta = 2.1609000000;
    assert_approx_eq!(expected_beta, beta_pdf(0.3, 2.0, 5.0), TOL);

    // 2. Chi-Square(nu=4.0) at x = 2.5
    // f(x) = (x^(k/2 - 1) * e^(-x/2)) / (2^(k/2) * Gamma(k/2)) = (2.5 * e^-1.25) / 4
    let expected_chisq = 0.1790654980;
    assert_approx_eq!(expected_chisq, chisq_pdf(2.5, 4.0), TOL);

    // 3. Exponential(lambda=2.5) at x = 1.0
    // f(x) = lambda * e^(-lambda * x) = 2.5 * e^-2.5
    let expected_exp = 0.2052124966;
    assert_approx_eq!(expected_exp, exp_pdf(1.0, 2.5), TOL);

    // 4. F-Distribution(nu1=2.0, nu2=2.0) at x = 1.0
    // Simplifies nicely to 0.25
    let expected_f = 0.2500000000;
    assert_approx_eq!(expected_f, f_pdf(1.0, 2.0, 2.0), TOL);

    // 5. Gamma(alpha=3.0, beta=2.0) at x = 1.0
    // f(x) = (beta^alpha * x^(alpha-1) * e^(-beta*x)) / Gamma(alpha) = (8 * 1 * e^-2) / 2
    let expected_gamma = 0.5413411329;
    assert_approx_eq!(expected_gamma, gamma_pdf(1.0, 3.0, 2.0), TOL);

    // 6. Gumbel(mu=1.0, beta=2.0) at x = 3.0
    // z = (3-1)/2 = 1.0. f(x) = (1/beta) * e^-(z + e^-z)
    let expected_gumbel = 0.1273231913;
    assert_approx_eq!(expected_gumbel, gumbel_pdf(3.0, 1.0, 2.0), TOL);

    // 7. Logistic(mu=2.0, s=1.5) at x = 2.0
    // z = 0. f(x) = e^-0 / (s * (1+e^-0)^2) = 1 / (1.5 * 4) = 1/6
    let expected_logistic = 0.1666666667;
    assert_approx_eq!(expected_logistic, logistic_pdf(2.0, 2.0, 1.5), TOL);

    // 8. Log-Normal(mu=0.0, sigma=1.0) at x = 1.0
    // f(1.0) = 1 / sqrt(2*pi)
    let expected_lognormal = 0.3989422804;
    assert_approx_eq!(expected_lognormal, lognormal_pdf(1.0, 0.0, 1.0), TOL);

    // 9. Normal(mu=10.0, sigma=2.0) at x = 10.0
    // f(10.0) = 1 / (sigma * sqrt(2*pi)) = 1 / (2 * sqrt(2*pi))
    let expected_normal = 0.1994711402;
    assert_approx_eq!(expected_normal, normal_pdf(10.0, 10.0, 2.0), TOL);

    // 10. Pareto Type IV(mu=0.0, sigma=1.0, gamma=1.0, alpha=3.0) at x = 1.0
    // f(1.0) = (3 / (1*1)) * (1)^0 * (1 + 1)^-4 = 3 / 16 = 0.1875
    let expected_pareto = 0.1875000000;
    assert_approx_eq!(expected_pareto, pareto4_pdf(1.0, 0.0, 1.0, 1.0, 3.0), TOL);

    // 11. Student's t(nu=3.0) at x = 0.0
    // f(0.0) = Gamma(2) / (sqrt(3*pi) * Gamma(1.5)) = 1 / (0.5 * pi * sqrt(3))
    let expected_t = 0.3675525969;
    assert_approx_eq!(expected_t, t_pdf(0.0, 3.0), TOL);

    // 12. Uniform(a=2.0, b=7.0) at x = 4.0
    // f(x) = 1 / (7 - 2) = 1/5 = 0.2
    let expected_unif = 0.2000000000;
    assert_approx_eq!(expected_unif, unif_pdf(4.0, 2.0, 7.0), TOL);
}

#[test]
fn test_pmfs() {

    // 1. Binomial(n=10, p=0.4) at x = 4
    // P(X=4) = (10 choose 4) * 0.4^4 * 0.6^6 = 210 * 0.0256 * 0.046656 = 0.250822656
    let expected_bin = 0.2508226560;
    assert_approx_eq!(expected_bin, bin_pmf(4, 10, 0.4), TOL);

    // 2. Geometric(p=0.3) at x = 2 (failures before first success)
    // P(X=2) = (1-p)^2 * p = 0.7^2 * 0.3 = 0.147
    let expected_geo = 0.1470000000;
    assert_approx_eq!(expected_geo, geo_pmf(2, 0.3), TOL);

    // 3. Geometric2(p=0.3) at x = 3 (trial number of first success)
    // P(X=3) = (1-p)^2 * p = 0.147
    let expected_geo2 = 0.1470000000;
    assert_approx_eq!(expected_geo2, geo2_pmf(3, 0.3), TOL);

    // 4. Hypergeometric(n=5, N=20, M=8) at x = 2
    // P(X=2) = (8 choose 2) * (12 choose 3) / (20 choose 5) = 28 * 220 / 15504 = 6160 / 15504
    let expected_hg = 0.3973168215;
    assert_approx_eq!(expected_hg, hg_pmf(2, 5, 20, 8), TOL);

    // 5. Negative Binomial(r=3, p=0.4) at x = 2 (failures before r-th success)
    // P(X=2) = (4 choose 2) * p^3 * (1-p)^2 = 6 * 0.064 * 0.36 = 0.13824
    let expected_nb = 0.1382400000;
    assert_approx_eq!(expected_nb, nb_pmf(2, 3, 0.4), TOL);

    // 6. Negative Binomial 2(r=3, p=0.4) at x = 5 (trial of r-th success)
    // P(X=5) = (4 choose 2) * p^3 * (1-p)^2 = 0.13824
    let expected_nb2 = 0.1382400000;
    assert_approx_eq!(expected_nb2, nb2_pmf(5, 3, 0.4), TOL);

    // 7. Poisson(lambda=3.0) at x = 2
    // P(X=2) = (e^-3 * 3^2) / 2! = 4.5 * e^-3
    let expected_pois = 0.2240418077;
    assert_approx_eq!(expected_pois, pois_pmf(2, 3.0), TOL);
}