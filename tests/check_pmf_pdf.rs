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

const TOL: f64 = 1e-10;

#[test]
fn test_pdfs() {

    // Beta(alpha=2.0, beta=5.0) at x = 0.3
    // f(x) = x^(a-1) * (1-x)^(b-1) / B(a,b) = 0.3^1 * 0.7^4 / (1/30) = 2.1609
    let expected_beta = 2.160900000000;
    assert_approx_eq!(expected_beta, beta_pdf(0.3, 2.0, 5.0), TOL);

    // Burr Type XII(lambda=1.0, c=2.0, k=3.0) at x = 1.0
    // f(1.0) = (6/1) * (1)^1 * (1+1)^(-4) = 6 / 16 = 3/8
    let expected_burr = 0.375000000000;
    assert_approx_eq!(expected_burr, burr_pdf(1.0, 1.0, 2.0, 3.0), TOL);

    // Cauchy(x0=1.0, gamma=2.0) at x = 3.0
    // z = (3-1)/2 = 1.0. f(3.0) = 1 / (pi * 2 * (1 + 1^2)) = 1 / (4 * pi)
    let expected_cauchy = 0.079577471546;
    assert_approx_eq!(expected_cauchy, cauchy_pdf(3.0, 1.0, 2.0), TOL);

    // Chi-Square(nu=4.0) at x = 2.5
    // f(x) = (x^(k/2 - 1) * e^(-x/2)) / (2^(k/2) * Gamma(k/2)) = (2.5 * e^-1.25) / 4
    let expected_chisq = 0.179065498038;
    assert_approx_eq!(expected_chisq, chisq_pdf(2.5, 4.0), TOL);

    // Exponential(lambda=2.5) at x = 1.0
    // f(x) = lambda * e^(-lambda * x) = 2.5 * e^-2.5
    let expected_exp = 0.205212496560;
    assert_approx_eq!(expected_exp, exp_pdf(1.0, 2.5), TOL);

    // F-Distribution(nu1=2.0, nu2=2.0) at x = 1.0
    // Simplifies nicely to 0.25
    let expected_f = 0.250000000000;
    assert_approx_eq!(expected_f, f_pdf(1.0, 2.0, 2.0), TOL);

    // Frechet(mu=1.0, sigma=2.0, alpha=3.0) at x = 2.0
    // z = (2.0 - 1.0) / 2.0 = 0.5
    // f(2.0) = (3 / 2) * (0.5)^(-4) * exp(-(0.5)^-3) = 1.5 * 16 * exp(-8) = 24 * exp(-8)
    let expected_frechet = 0.008051103070;
    assert_approx_eq!(expected_frechet, frechet_pdf(2.0, 1.0, 2.0, 3.0), TOL);

    // Gamma(alpha=3.0, beta=2.0) at x = 1.0
    // f(x) = (beta^alpha * x^(alpha-1) * e^(-beta*x)) / Gamma(alpha) = (8 * 1 * e^-2) / 2
    let expected_gamma = 0.541341132946;
    assert_approx_eq!(expected_gamma, gamma_pdf(1.0, 3.0, 2.0), TOL);

    // Gumbel(mu=1.0, beta=2.0) at x = 3.0
    // z = (3-1)/2 = 1.0. f(x) = (1/beta) * e^-(z + e^-z)
    let expected_gumbel = 0.127323190022;
    assert_approx_eq!(expected_gumbel, gumbel_pdf(3.0, 1.0, 2.0), TOL);

    // Inverse Gamma(alpha=2.0, beta=3.0) at x = 1.0
    // f(1.0) = (3^2 / Gamma(2)) * (1)^-3 * exp(-3) = 9 * exp(-3)
    let expected_invgamma = 0.448083615311;
    assert_approx_eq!(expected_invgamma, invgamma_pdf(1.0, 2.0, 3.0), TOL);

    // Inverse Gaussian(mu=1.0, lambda=2.0) at x = 1.0
    // f(1.0) = sqrt(2 / 2pi) * exp( -2(0)/2 ) = sqrt(1 / pi)
    let expected_invgauss = 0.564189583548;
    assert_approx_eq!(expected_invgauss, invgauss_pdf(1.0, 1.0, 2.0), TOL);

    // Laplace(mu=0.0, b=2.0) at x = 1.0
    // f(1.0) = (1/4) * e^(-1/2)
    let expected_laplace = 0.151632664928;
    assert_approx_eq!(expected_laplace, laplace_pdf(1.0, 0.0, 2.0), TOL);

    // Logistic(mu=2.0, s=1.5) at x = 2.0
    // z = 0. f(x) = e^-0 / (s * (1+e^-0)^2) = 1 / (1.5 * 4) = 1/6
    let expected_logistic = 0.166666666667;
    assert_approx_eq!(expected_logistic, logistic_pdf(2.0, 2.0, 1.5), TOL);

    // Log-Logistic(alpha=2.0, beta=3.0) at x = 2.0
    // f(2.0) = (3/2) * (1)^2 / (1+1)^2 = 1.5 / 4 = 0.375
    let expected_loglogistic = 0.375000000000;
    assert_approx_eq!(expected_loglogistic, loglogistic_pdf(2.0, 2.0, 3.0), TOL);

    // Log-Normal(mu=0.0, sigma=1.0) at x = 1.0
    // f(1.0) = 1 / sqrt(2*pi)
    let expected_lognormal = 0.398942280401;
    assert_approx_eq!(expected_lognormal, lognormal_pdf(1.0, 0.0, 1.0), TOL);

    // Normal(mu=10.0, sigma=2.0) at x = 10.0
    // f(10.0) = 1 / (sigma * sqrt(2*pi)) = 1 / (2 * sqrt(2*pi))
    let expected_normal = 0.199471140201;
    assert_approx_eq!(expected_normal, normal_pdf(10.0, 10.0, 2.0), TOL);

    // Pareto Type IV(mu=0.0, sigma=1.0, gamma=1.0, alpha=3.0) at x = 1.0
    // f(1.0) = (3 / (1*1)) * (1)^0 * (1 + 1)^-4 = 3 / 16 = 0.1875
    let expected_pareto = 0.187500000000;
    assert_approx_eq!(expected_pareto, pareto4_pdf(1.0, 0.0, 1.0, 1.0, 3.0), TOL);

    // Rayleigh(sigma=2.0) at x = 2.0
    // f(2.0) = (2/4) * exp(-4/8) = 0.5 * exp(-0.5)
    let expected_rayleigh = 0.303265329856;
    assert_approx_eq!(expected_rayleigh, rayleigh_pdf(2.0, 2.0), TOL);

    // Student's t(nu=3.0) at x = 0.0
    // f(0.0) = Gamma(2) / (sqrt(3*pi) * Gamma(1.5)) = 1 / (0.5 * pi * sqrt(3))
    let expected_t = 0.367552596948;
    assert_approx_eq!(expected_t, t_pdf(0.0, 3.0), TOL);

    // Triangular(a=0.0, b=10.0, c=3.0) at x = 5.0
    // f(5.0) = 2(10-5) / ((10-0)(10-3)) = 10 / 70 = 1/7
    let expected_triangular = 0.142857142857;
    assert_approx_eq!(expected_triangular, triangular_pdf(5.0, 0.0, 10.0, 3.0), TOL);

    // Uniform(a=2.0, b=7.0) at x = 4.0
    // f(x) = 1 / (7 - 2) = 1/5 = 0.2
    let expected_unif = 0.200000000000;
    assert_approx_eq!(expected_unif, unif_pdf(4.0, 2.0, 7.0), TOL);

    // Weibull(k=2.0, lambda=3.0) at x = 1.5
    // f(1.5) = (2/3) * (1.5/3)^1 * e^-(1.5/3)^2 = (1/3) * e^-0.25
    let expected_weibull = 0.259600261024;
    assert_approx_eq!(expected_weibull, weibull_pdf(1.5, 2.0, 3.0), TOL);

}

#[test]
fn test_pmfs() {

    // Bernoulli(p=0.7) at x = 1
    // f(1) = 0.7
    let expected_bernoulli = 0.700000000000;
    assert_approx_eq!(expected_bernoulli, bernoulli_pmf(1, 0.7), TOL);

    // Binomial(n=10, p=0.4) at x = 4
    // P(X=4) = (10 choose 4) * 0.4^4 * 0.6^6 = 210 * 0.0256 * 0.046656 = 0.250822656
    let expected_bin = 0.250822656000;
    assert_approx_eq!(expected_bin, bin_pmf(4, 10, 0.4), TOL);

    // Discrete Uniform(a=1, b=5) at x = 3
    // f(3) = 1/5 = 0.2
    let expected_discrete_unif = 0.200000000000;
    assert_approx_eq!(expected_discrete_unif, discrete_unif_pmf(3, 1, 5), TOL);

    // Geometric(p=0.3) at x = 2 (failures before first success)
    // P(X=2) = (1-p)^2 * p = 0.7^2 * 0.3 = 0.147
    let expected_geo = 0.147000000000;
    assert_approx_eq!(expected_geo, geo_pmf(2, 0.3), TOL);

    // Geometric2(p=0.3) at x = 3 (trial number of first success)
    // P(X=3) = (1-p)^2 * p = 0.147
    let expected_geo2 = 0.147000000000;
    assert_approx_eq!(expected_geo2, geo2_pmf(3, 0.3), TOL);

    // Hypergeometric(n=5, N=20, M=8) at x = 2
    // P(X=2) = (8 choose 2) * (12 choose 3) / (20 choose 5) = 28 * 220 / 15504 = 6160 / 15504
    let expected_hg = 0.397316821465;
    assert_approx_eq!(expected_hg, hg_pmf(2, 5, 20, 8), TOL);

    // Negative Binomial(r=3, p=0.4) at x = 2 (failures before r-th success)
    // P(X=2) = (4 choose 2) * p^3 * (1-p)^2 = 6 * 0.064 * 0.36 = 0.13824
    let expected_nb = 0.138240000000;
    assert_approx_eq!(expected_nb, nb_pmf(2, 3, 0.4), TOL);

    // Negative Binomial 2(r=3, p=0.4) at x = 5 (trial of r-th success)
    // P(X=5) = (4 choose 2) * p^3 * (1-p)^2 = 0.13824
    let expected_nb2 = 0.138240000000;
    assert_approx_eq!(expected_nb2, nb2_pmf(5, 3, 0.4), TOL);

    // Poisson(lambda=3.0) at x = 2
    // P(X=2) = (e^-3 * 3^2) / 2! = 4.5 * e^-3
    let expected_pois = 0.224041807655;
    assert_approx_eq!(expected_pois, pois_pmf(2, 3.0), TOL);

    // ZIP(lambda=3.0, pi=0.2) at x = 0
    // P(X=0) = pi + (1-pi)*exp(-lambda) = 0.2 + 0.8 * exp(-3.0)
    let expected_zip = 0.239829654694;
    assert_approx_eq!(expected_zip, zip_pmf(0, 3.0, 0.2), TOL);

    // ZINB(r=3, p=0.4, pi=0.2) at x = 2
    // P(X=2) = (1-pi) * NB_PMF(x=2) = 0.8 * (4 choose 2) * 0.4^3 * 0.6^2 = 0.8 * 6 * 0.064 * 0.36 = 0.110592
    let expected_zinb = 0.110592000000;
    assert_approx_eq!(expected_zinb, zinb_pmf(2, 3, 0.4, 0.2), TOL);
}