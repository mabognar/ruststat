use std::f64::consts::PI;
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

const TOL: f64 = 1e-6;

#[test]
fn test_gamma_function() {
    // Gamma(n) = (n-1)! for integer n
    assert_approx_eq!(gamma(5.0), 24.0, TOL);
    assert_approx_eq!(gamma(6.0), 120.0, TOL);

    // Gamma(0.5) = sqrt(pi)
    let sqrt_pi = PI.sqrt();
    assert_approx_eq!(gamma(0.5), sqrt_pi, TOL);

    // ln(Gamma(x)) should match the log of Gamma(x)
    assert_approx_eq!(gamma_ln(5.0), 24.0_f64.ln(), TOL);
}

#[test]
fn test_factorial() {
    // fact_i computes standard factorials
    assert_eq!(fact_u(1), 1);
    assert_eq!(fact_u(5), 120);
    assert_eq!(fact_u(10), 3_628_800);

    // fact_f generalizes factorial: x! = x * Gamma(x)
    // 5! = 120
    assert_approx_eq!(fact_f(5.0), 120.0, TOL);
    // 0.5! = 0.5 * Gamma(0.5) = 0.5 * sqrt(pi)
    let half_fact = 0.5 * PI.sqrt();
    assert_approx_eq!(fact_f(0.5), half_fact, TOL);
}

#[test]
fn test_combination() {
    // 10 choose 5 = 252
    assert_approx_eq!(comb(10, 5), 252.0, TOL);
    // 50 choose 2 = (50 * 49) / 2 = 1225
    assert_approx_eq!(comb(50, 2), 1225.0, TOL);
    // n choose 0 = 1, n choose n = 1
    assert_approx_eq!(comb(20, 0), 1.0, TOL);
    assert_approx_eq!(comb(20, 20), 1.0, TOL);
}

#[test]
fn test_beta_fn() {
    // B(1, 1) = 1
    assert_approx_eq!(beta_fn(1.0, 1.0), 1.0, TOL);

    // B(2, 2) = Gamma(2)*Gamma(2) / Gamma(4) = (1 * 1) / 6 = 1/6
    assert_approx_eq!(beta_fn(2.0, 2.0), 1.0 / 6.0, TOL);

    // B(a, b) should equal B(b, a) (Symmetry)
    assert_approx_eq!(beta_fn(3.5, 7.2), beta_fn(7.2, 3.5), TOL);
}

#[test]
fn test_incomplete_beta_fn() {
    // I_x(1, 1) = x
    assert_approx_eq!(betai(0.25, 1.0, 1.0), 0.25, TOL);
    assert_approx_eq!(betai(0.75, 1.0, 1.0), 0.75, TOL);

    // I_x(a, b) at x=0 is 0, at x=1 is 1
    assert_approx_eq!(betai(0.0, 2.0, 5.0), 0.0, TOL);
    assert_approx_eq!(betai(1.0, 2.0, 5.0), 1.0, TOL);

    // I_{0.5}(a, a) = 0.5 due to symmetry
    assert_approx_eq!(betai(0.5, 3.0, 3.0), 0.5, TOL);
}

#[test]
fn test_incomplete_gamma_fn() {
    // gammp(x, a) is the regularized lower incomplete gamma function.
    // P(x, 1) = 1 - e^(-x)
    let x = 1.0;
    let expected = 1.0 - std::f64::consts::E.powf(-x);
    assert_approx_eq!(gammp(x, 1.0), expected, TOL);

    // Q(x, a) is the complement: Q(x, a) = 1 - P(x, a)
    // Therefore, gammq(x, a) + gammp(x, a) = 1.0
    assert_approx_eq!(gammp(2.5, 3.0) + gammq(2.5, 3.0), 1.0, TOL);
}

#[test]
fn test_error_fn() {
    // erf(0) = 0
    assert_approx_eq!(erf(0.0), 0.0, TOL);

    // erf(x) for a known value: erf(1) ≈ 0.84270079
    assert_approx_eq!(erf(1.0), 0.8427007929497148, TOL);

    // erf(-x) = -erf(x) (Odd function)
    assert_approx_eq!(erf(-1.0), -erf(1.0), TOL);

    // erf(x) + erfc(x) = 1.0
    assert_approx_eq!(erf(0.5) + erfc(0.5), 1.0, TOL);
}

#[test]
fn test_error_fn_inverse() {
    // Test round-tripping: erf_inv(erf(x)) == x
    let x = 0.5;
    let p = erf(x);
    assert_approx_eq!(erf_inv(p), x, TOL);

    // erf_inv(0) = 0
    assert_approx_eq!(erf_inv(0.0), 0.0, TOL);

    // Test round-tripping for erfc
    let p_comp = erfc(x);
    assert_approx_eq!(erfc_inv(p_comp), x, TOL);
}

