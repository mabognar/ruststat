#![doc = include_str!("../README.md")]

// ==========================================
// ==========================================
// ruststat
// ==========================================
// ==========================================

//! Utilities for working with many common random variables.
//! - probability mass function (pmf), probability density function (pdf)
//! - cumulative distribution function (cdf)
//! - percentiles (inverse cdf)
//! - random number generation
//! - mean, variance
//!
//! Distributions:
//! - Continuous: beta, chi-square, exponential, F, gamma, normal, log-normal, Pareto (1 through 4),
//! Student's t, continuous uniform
//! - Discrete: binomial, geometric, hypergeometric, negative binomial, poisson
//!
//! # Quick Start
//! ```
//! use ruststat::*;
//! // X~N(mu=0,sigma=1.0), find 97.5th percentile
//! println!("normal percentile: {}", normal_per(0.975, 0.0, 1.0));
//! // X~Bin(n=10,p=0.7), compute P(X=4)
//! println!("binomial probability: {}", bin_pmf(4, 10, 0.7));
//! ```
//! For convenience, functions can also be accessed via `Structs`.
//! ```
//! use ruststat::*;
//! // X~Beta(alpha=0.5,beta=2.0)
//! let mut mybeta = BetaDist{alpha:0.5, beta:2.0};
//! // 30th percentile
//! println!("percentile: {}", mybeta.per(0.3));
//! // P(X <= 0.4)
//! println!("cdf: {}", mybeta.cdf(0.4));
//! // Random draw
//! println!("random draw: {}", mybeta.ran());
//! // Variance
//! println!("variance: {}", mybeta.var());
//! ```

// use std::collections::btree_set::Iter;
use std::f64::consts::PI;
use rand::{random};

const GAMMA: f64 = 0.577215664901532860606512090082;


/// Computes sample mean of elements in a slice
///
/// # Example
/// ```
/// use ruststat::sample_mean;
/// let x = vec![1.5, 2.0, 3.5];
/// println!("Sample mean: {}", sample_mean(&x));
/// ```
pub fn sample_mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = x.iter().sum();
    sum / (x.len() as f64)
}

/// Computes sample variance of elements in a slice
///
/// # Example
/// ```
/// use ruststat::sample_var;
/// let x = vec![1.5, 2.0, 3.5];
/// println!("Sample variance: {}", sample_var(&x));
/// ```
pub fn sample_var(x: &[f64]) -> f64 {
    let n = x.len();

    // Variance requires at least 2 data points (n-1 degrees of freedom)
    if n < 2 {
        return f64::NAN;
    }

    let xbar = sample_mean(x);
    let ss: f64 = x.iter().map(|xx| (xx - xbar).powi(2)).sum();

    ss / ((n - 1) as f64)
}

/// Computes sample standard deviation of elements in a slice
///
/// # Example
/// ```
/// use ruststat::sample_sd;
/// let x = vec![1.5, 2.0, 3.5];
/// println!("Sample SD: {}", sample_sd(&x));
/// ```
pub fn sample_sd(x: &[f64]) -> f64 {
    // If x has < 2 elements, sample_var returns NAN, and NAN.sqrt() is safely NAN.
    sample_var(x).sqrt()
}


// ==========================================
// ==========================================
// ==========================================
// Continuous RVs
// ==========================================
// ==========================================
// ==========================================


// ==========================================
// ==========================================
// Beta Distribution
// ==========================================
// ==========================================

/// Struct for the beta distribution `X ~ Beta(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`.
/// ```
/// use ruststat::BetaDist;
/// let mut mybeta = BetaDist{alpha:0.5, beta:2.0};
/// println!("Probability density function f(0.7): {}", mybeta.pdf(0.7));
/// println!("Cumulative distribution function P(X<=0.7): {}", mybeta.cdf(0.7));
/// println!("99th percentile: {}", mybeta.per(0.99));
/// println!("Random draw: {}", mybeta.ran());
/// println!("Random vector: {:?}", mybeta.ranvec(5));
/// println!("Mean: {}", mybeta.mean());
/// println!("Variance: {}", mybeta.var());
/// println!("Standard deviation: {}", mybeta.sd());
/// ```
pub struct BetaDist {
    pub alpha: f64,
    pub beta: f64,
}
impl BetaDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        beta_pdf(x, self.alpha, self.beta)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        beta_cdf(x, self.alpha, self.beta)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        beta_per(x,self.alpha, self.beta)
    }
    pub fn ran(&mut self) -> f64 {
         beta_ran(self.alpha, self.beta)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
         beta_ranvec(n, self.alpha, self.beta)
    }
    pub fn mean(&mut self) -> f64 {
         self.alpha / (self.alpha + self.beta)
    }
    pub fn var(&mut self) -> f64 {
         self.alpha * self.beta /
            ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0))
    }
    pub fn sd(&mut self) -> f64 {
         self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Beta(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`.
/// ```
/// use ruststat::beta_pdf;
/// println!("Probability density function f(0.7): {}", beta_pdf(0.7, 0.5, 2.0));
/// ```
pub fn beta_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function beta_pdf");
        return f64::NAN;
    }
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    ((alpha - 1.0)*x.ln() + (beta - 1.0)*(1.0 - x).ln() - beta_fn_ln(alpha, beta)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Beta(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`.
/// ```
/// use ruststat::beta_cdf;
/// println!("P(X <= 0.7): {}", beta_cdf(0.7, 0.5, 2.0));
/// ```
pub fn beta_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function beta_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    betai(x, alpha, beta)
}


/// Computes a percentile for `X ~ Beta(alpha,beta)`
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::beta_per;
/// println!("Percentile: {}", beta_per(0.8, 0.5, 2.0));
/// ```
pub fn beta_per(q: f64, alpha: f64, beta: f64) -> f64 {
    betai_inv(q, alpha, beta)

}


/// Random draw from `X ~ Beta(alpha,beta)` distribution.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`. Use
/// ```
/// use ruststat::beta_ran;
/// println!("Random draw: {}", beta_ran(0.5, 2.0));
/// ```
pub fn beta_ran(alpha: f64, beta: f64) -> f64 {
    let x = gamma_ran(alpha, 1.0);
    let y = gamma_ran(beta, 1.0);
    x / (x+y)
}


/// Save random draws from `X ~ Beta(alpha, beta)` distribution into a `Vec`
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`. Use
/// ```
/// use ruststat::beta_ranvec;
/// println!("Random Vec: {:?}", beta_ranvec(10, 0.5, 2.0));
/// ```
pub fn beta_ranvec(n: u64, alpha: f64, beta: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(beta_ran(alpha, beta));
    }
    xvec
}


// ==========================================
// ==========================================
// ChiSquare Distribution
// ==========================================
// ==========================================

/// Struct for the chi-square distribution `X ~ ChiSq(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. Use.
/// ```
/// use ruststat::ChiSqDist;
/// let mut mychisq = ChiSqDist{nu:8.0};
/// println!("Probability density function f(4.5): {}", mychisq.pdf(4.5));
/// println!("Cumulative distribution function P(X<=4.5): {}", mychisq.cdf(4.5));
/// println!("99th percentile: {}", mychisq.per(0.99));
/// println!("Random draw: {}", mychisq.ran());
/// println!("Random vector: {:?}", mychisq.ranvec(5));
/// println!("Mean: {}", mychisq.mean());
/// println!("Variance: {}", mychisq.var());
/// println!("Standard deviation: {}", mychisq.sd());
/// ```
pub struct ChiSqDist {
    pub nu: f64,
}
impl ChiSqDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        chisq_pdf(x, self.nu)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        chisq_cdf(x, self.nu)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        chisq_per(x, self.nu)
    }
    pub fn ran(&mut self) -> f64 {
        chisq_ran(self.nu)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        chisq_ranvec(n, self.nu)
    }
    pub fn mean(&mut self) -> f64 {
        self.nu
    }
    pub fn var(&mut self) -> f64 {
        2.0 * self.nu
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ ChiSq(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`.
/// ```
/// use ruststat::chisq_pdf;
/// println!("Probability density function f(5.2): {}", chisq_pdf(5.2, 8.0));
/// ```
pub fn chisq_pdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function chisq_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gamma_pdf(x, nu/2.0,1.0/2.0)
}


/// Computes cumulative distribution function (cdf) for `X ~ ChiSq(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`.
/// ```
/// use ruststat::chisq_cdf;
/// println!("P(X<=5.2): {}", chisq_cdf(5.2, 8.0));
/// ```
pub fn chisq_cdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function chisq_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gamma_cdf(x, nu / 2.0, 0.5)
}


/// Computes a percentile for `X ~ ChiSq(nu)`
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::chisq_per;
/// println!("Percentile: {}", chisq_per(0.8, 8.0));
/// ```
pub fn chisq_per(p: f64, nu: f64) -> f64 {
    gamma_per(p, nu/2.0, 1.0/2.0)
}


/// Random draw from `X ~ ChiSq(nu)` distribution.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. Use
/// ```
/// use ruststat::chisq_ran;
/// println!("Random draw: {}", chisq_ran(8.0));
/// ```
pub fn chisq_ran(nu: f64) -> f64 {
    gamma_ran(nu/2.0, 1.0/2.0)
}


/// Save random draws from `X ~ ChiSq(nu)` into a `Vec`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. Use
/// ```
/// use ruststat::chisq_ranvec;
/// println!("Random Vec: {:?}", chisq_ranvec(10, 8.0));
/// ```
pub fn chisq_ranvec(n: u64, nu: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(chisq_ran(nu));
    }
    xvec
}


// ==========================================
// ==========================================
// Exponential Distribution
// ==========================================
// ==========================================

/// Struct for the exponential distribution `X ~ Exp(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=3.5)`. Use.
/// ```
/// use ruststat::ExpDist;
/// let mut myexp = ExpDist{lambda:3.5};
/// println!("Probability density function f(1.2): {}", myexp.pdf(1.2));
/// println!("Cumulative distribution function P(X<=1.2): {}", myexp.cdf(1.2));
/// println!("99th percentile: {}", myexp.per(0.99));
/// println!("Random draw: {}", myexp.ran());
/// println!("Random vector: {:?}", myexp.ranvec(5));
/// println!("Mean: {}", myexp.mean());
/// println!("Variance: {}", myexp.var());
/// println!("Standard deviation: {}", myexp.sd());
/// ```
pub struct ExpDist {
    pub lambda: f64,
}
impl ExpDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        exp_pdf(x, self.lambda)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        exp_cdf(x, self.lambda)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        exp_per(x, self.lambda)
    }
    pub fn ran(&mut self) -> f64 {
        exp_ran(self.lambda)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        exp_ranvec(n, self.lambda)
    }
    pub fn mean(&mut self) -> f64 {
        1.0 / self.lambda
    }
    pub fn var(&mut self) -> f64 {
        1.0 / self.lambda.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Exp(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`.
/// ```
/// use ruststat::exp_pdf;
/// println!("Probability density function f(0.2): {}", exp_pdf(0.2, 8.0));
/// ```
pub fn exp_pdf(x: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function exp_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    lambda * (-lambda*x).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Exp(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`.
/// ```
/// use ruststat::exp_cdf;
/// println!("P(X<=0.8): {}", exp_cdf(0.8, 8.0));
/// ```
pub fn exp_cdf(x: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function exp_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }

    // Exact closed-form solution
    1.0 - (-lambda * x).exp()
}

/// Computes a percentile for `X ~ Exp(lambda)`
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::exp_per;
/// println!("Percentile: {}", exp_per(0.8, 8.0));
/// ```
pub fn exp_per(p: f64, lambda: f64) -> f64 {
    -(1.0-p).ln() / lambda
}


/// Random draw from `X ~ Exp(lambda)` distribution.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`. Use
/// ```
/// use ruststat::exp_ran;
/// println!("Random draw: {}", exp_ran(8.0));
/// ```
pub fn exp_ran(lambda: f64) -> f64 {
    let u: f64;
    u = random();
    // u = rand::thread_rng().gen();
    -(1.0-u).ln()/lambda
}


/// Save random draws from `X ~ Exp(lambda)` into a `Vec`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`. Use
/// ```
/// use ruststat::exp_ranvec;
/// println!("Random vector: {:?}", exp_ranvec(10, 8.0));
/// ```
pub fn exp_ranvec(n: u64, lambda: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(exp_ran(lambda));
    }
    xvec
}


// ==========================================
// ==========================================
// F Distribution
// ==========================================
// ==========================================

/// Struct for the F distribution `X ~ F(nu1, nu2)`.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use.
/// ```
/// use ruststat::FDist;
/// let mut myf = FDist{nu1:4.0, nu2:18.0};
/// println!("Probability density function f(2.5): {}", myf.pdf(2.5));
/// println!("Cumulative distribution function P(X<=2.5): {}", myf.cdf(2.5));
/// println!("99th percentile: {}", myf.per(0.99));
/// println!("Random draw: {}", myf.ran());
/// println!("Random vector: {:?}", myf.ranvec(5));
/// println!("Mean: {}", myf.mean());
/// println!("Variance: {}", myf.var());
/// println!("Standard deviation: {}", myf.sd());
/// ```
pub struct FDist {
    pub nu1: f64,
    pub nu2: f64,
}
impl FDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        f_pdf(x, self.nu1, self.nu2)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        f_cdf(x, self.nu1, self.nu2)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        f_per(x, self.nu1, self.nu2)
    }
    pub fn ran(&mut self) -> f64 {
        f_ran(self.nu1, self.nu2)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        f_ranvec(n, self.nu1, self.nu2)
    }
    pub fn mean(&mut self) -> f64 {
        self.nu2 / (self.nu2-2.0)
    }
    pub fn var(&mut self) -> f64 {
        2.0 * self.nu2.powi(2) * (self.nu1 + self.nu2 - 2.0) /
            (self.nu1 * (self.nu2 - 2.0).powi(2) * (self.nu2 - 4.0))
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ F(nu1, nu2)`.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use.
/// ```
/// use ruststat::f_pdf;
/// println!("Probability density function f(2.5): {}", f_pdf(2.5, 4.0, 18.0));
/// ```
pub fn f_pdf(x: f64, nu1: f64, nu2: f64) -> f64 {
    if nu1 <= 0.0 || nu2 <= 0.0 {
        println!("NAN produced. Error in function f_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (0.5*(nu1*(nu1*x).ln() + nu2*nu2.ln() - (nu1+nu2)*(nu1*x + nu2).ln())
        - x.ln() - beta_fn_ln(nu1/2.0, nu2/2.0)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ F(nu1, nu2)`.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use.
/// ```
/// use ruststat::f_cdf;
/// println!("P(X<=2.5): {}", f_cdf(2.5, 4.0, 18.0));
/// ```
pub fn f_cdf(x: f64, nu1: f64, nu2: f64) -> f64 {
    if nu1 <= 0.0 || nu2 <= 0.0 {
        println!("NAN produced. Error in function f_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    1.0 - betai(nu2 / (nu2 + nu1 * x), nu2 / 2.0, nu1 / 2.0)
}


/// Computes a percentile for `X ~ F(nu1, nu2)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::f_per;
/// println!("Percentile: {}", f_per(0.8, 4.0, 18.0));
/// ```
pub fn f_per(p: f64, nu1: f64, nu2: f64) -> f64 {
    nu2/nu1 * (1.0 / beta_per(1.0-p,nu2/2.0,nu1/2.0) - 1.0)
}


/// Random draw from `X ~ F(nu1, nu2)` distribution.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use
/// ```
/// use ruststat::f_ran;
/// println!("Random draw: {}", f_ran(4.0, 18.0));
/// ```
pub fn f_ran(nu1: f64, nu2: f64) -> f64 {
    let x = beta_ran(nu1/2.0, nu2/2.0);
    nu2 * x / (nu1 * (1.0-x))
}

/// Save random draws from `X ~ F(nu1, nu2)` distribution into a `Vec`
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(alpha=4.0, beta=18.0)`. Use
/// ```
/// use ruststat::f_ranvec;
/// println!("Random Vec: {:?}", f_ranvec(10, 4.0, 18.0));
/// ```
pub fn f_ranvec(n: u64, nu1: f64, nu2: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(f_ran(nu1, nu2));
    }
    xvec
}


// ==========================================
// ==========================================
// Gamma Distribution
// ==========================================
// ==========================================

/// Struct for the gamma distribution `X ~ Gamma(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use.
/// ```
/// use ruststat::GammaDist;
/// let mut mygamma = GammaDist{alpha:4.0, beta:2.7};
/// println!("Probability density function f(3.9): {}", mygamma.pdf(3.9));
/// println!("Cumulative distribution function P(X<=3.9): {}", mygamma.cdf(3.9));
/// println!("99th percentile: {}", mygamma.per(0.99));
/// println!("Random draw: {}", mygamma.ran());
/// println!("Random vector: {:?}", mygamma.ranvec(5));
/// println!("Mean: {}", mygamma.mean());
/// println!("Variance: {}", mygamma.var());
/// println!("Standard deviation: {}", mygamma.sd());
/// ```
pub struct GammaDist {
    pub alpha: f64,
    pub beta: f64,
}
impl GammaDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        gamma_pdf(x, self.alpha, self.beta)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        gamma_cdf(x, self.alpha, self.beta)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        gamma_per(x, self.alpha, self.beta)
    }
    pub fn ran(&mut self) -> f64 {
        gamma_ran(self.alpha, self.beta)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        gamma_ranvec(n, self.alpha, self.beta)
    }
    pub fn mean(&mut self) -> f64 {
        self.alpha / self.beta
    }
    pub fn var(&mut self) -> f64 {
        self.alpha / self.beta.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Gamma(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use.
/// ```
/// use ruststat::gamma_pdf;
/// println!("Probability density function f(3.9): {}", gamma_pdf(3.9, 4.0, 2.7));
pub fn gamma_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function gamma_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (alpha*beta.ln() - gamma_ln(alpha) + (alpha-1.0)*x.ln()  - beta*x).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Gamma(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use.
/// ```
/// use ruststat::gamma_cdf;
/// println!("P(X<=3.9): {}", gamma_cdf(3.9, 4.0, 2.7));
/// ```
pub fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function gamma_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }

    // Route directly through the protected gammp function
    gammp(x * beta, alpha)
}


/// Computes a percentile for `X ~ Gamma(alpha, beta)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::gamma_per;
/// println!("Percentile: {}", gamma_per(0.8, 4.0, 2.7));
/// ```
pub fn gamma_per(p: f64, alpha: f64, beta: f64) -> f64 {
    gammai_inv(p, alpha) / beta
}


/// Random draw from `X ~ Gamma(alpha, beta)` distribution.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use
/// ```
/// use ruststat::gamma_ran;
/// println!("Random draw: {}", gamma_ran(4.0, 2.7));
/// ```
pub fn gamma_ran(alpha: f64, beta: f64) -> f64 {

    let (d, c, mut x, mut v, mut u): (f64, f64, f64, f64, f64);
    d = alpha - 1.0/3.0;
    c = 1.0/(9.0*d).sqrt();

    loop {
        x = normal_ran(0.0, 1.0);
        v = 1.0 + c * x;
        while v <= 0.0 {
            x = normal_ran(0.0, 1.0);
            v = 1.0 + c * x;
            break;
        }

        v = v.powi(3);
        u = random();
        // u = rand::thread_rng().gen();

        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v / beta;
        }
        if u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln()) {
            return d * v / beta;
        }
    }
}


/// Save random draws from `X ~ Gamma(alpha, beta)` distribution into a `Vec`
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use
/// ```
/// use ruststat::gamma_ranvec;
/// println!("Random Vec: {:?}", gamma_ranvec(10, 4.0, 2.7));
/// ```
pub fn gamma_ranvec(n: u64, alpha: f64, beta: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(gamma_ran(alpha, beta));
    }
    xvec
}


// ==========================================
// ==========================================
// Gumbel Distribution
// ==========================================
// ==========================================

/// Struct for the Gumbel distribution `X ~ Gumbel(mu, beta)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=4.2, beta=1.8)`. Use.
/// ```
/// use ruststat::GumbelDist;
/// let mut mygum = GumbelDist{mu:4.2, beta:1.8};
/// println!("Probability density function f(3.9): {}", mygum.pdf(3.9));
/// println!("Cumulative distribution function P(X<=3.9): {}", mygum.cdf(3.9));
/// println!("99th percentile: {}", mygum.per(0.99));
/// println!("Random draw: {}", mygum.ran());
/// println!("Random vector: {:?}", mygum.ranvec(5));
/// println!("Mean: {}", mygum.mean());
/// println!("Variance: {}", mygum.var());
/// println!("Standard deviation: {}", mygum.sd());
/// ```
pub struct GumbelDist {
    pub mu: f64,    //location
    pub beta: f64,  //scale
}
impl GumbelDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        gumbel_pdf(x, self.mu, self.beta)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        gumbel_cdf(x, self.mu, self.beta)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        gumbel_per(x, self.mu, self.beta)
    }
    pub fn ran(&mut self) -> f64 {
        gumbel_ran(self.mu, self.beta)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        gumbel_ranvec(n, self.mu, self.beta)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu + self.beta * GAMMA
    }
    pub fn var(&mut self) -> f64 {
        PI.powi(2)/6.0 * self.beta.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Gumbel(mu, beta)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use.
/// ```
/// use ruststat::gumbel_pdf;
/// println!("Probability density function f(3.9): {}", gumbel_pdf(3.9, 5.5, 2.0));
pub fn gumbel_pdf(x: f64, mu: f64, beta: f64) -> f64 {
    if beta <= 0.0 {
        println!("NAN produced. Error in function gumbel_pdf");
        return f64::NAN;
    }
    1.0 / beta * (-((x-mu)/beta + (-(x-mu)/beta).exp())).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Gumbel(mu, beta)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use.
/// ```
/// use ruststat::gumbel_cdf;
/// println!("P(X<=3.9): {}", gumbel_cdf(3.9, 5.5, 2.0));
/// ```
pub fn gumbel_cdf(x: f64, mu: f64, beta: f64) -> f64 {
    if beta <= 0.0 {
        println!("NAN produced. Error in function gumbel_pdf");
        return f64::NAN;
    }

    (-(-(x-mu)/beta).exp()).exp()
}


/// Computes a percentile for `X ~ Gumbel(mu, beta)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::gumbel_per;
/// println!("Percentile: {}", gumbel_per(0.8, 5.5, 2.0));
/// ```
pub fn gumbel_per(p: f64, mu: f64, beta: f64) -> f64 {
    mu - beta * (-p.ln()).ln()
}


/// Random draw from `X ~ Gumbel(mu, beta)` distribution.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use
/// ```
/// use ruststat::gumbel_ran;
/// println!("Random draw: {:?}", gumbel_ran(5.5, 2.0));
/// ```
pub fn gumbel_ran(mu: f64, beta: f64) -> f64 {
    mu - beta * (-unif_ran(0.0,1.0).ln()).ln()
}


/// Random Vector taken from `X ~ Gumbel(mu, beta)` distribution.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use
/// ```
/// use ruststat::gumbel_ranvec;
/// println!("Random Vec: {:?}", gumbel_ranvec(10, 5.5, 2.0));
/// ```
pub fn gumbel_ranvec(n: u64, mu: f64, beta: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(mu - beta * (-unif_ran(0.0,1.0).ln()).ln());
    }
    xvec
}


// ==========================================
// ==========================================
// LogNormal Distribution
// ==========================================
// ==========================================

/// Struct for the log-normal distribution `X ~ LogNormal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ LogNormal(mu=0.0, sigma=1.0)`. Use
/// ```
/// use ruststat::LogNormalDist;
/// let mut mylogn = LogNormalDist{mu:0.0, sigma:1.0};
/// println!("Probability density function f(5.2): {}", mylogn.pdf(5.2));
/// println!("Cumulative distribution function P(X<=5.2): {}", mylogn.cdf(5.2));
/// println!("99th percentile: {}", mylogn.per(0.99));
/// println!("Random draw: {}", mylogn.ran());
/// println!("Random vector: {:?}", mylogn.ranvec(5));
/// println!("Mean: {}", mylogn.mean());
/// println!("Variance: {}", mylogn.var());
/// println!("Standard deviation: {}", mylogn.sd());
/// ```
pub struct LogNormalDist {
    pub mu: f64,
    pub sigma: f64,
}
impl LogNormalDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        lognormal_pdf(x, self.mu, self.sigma)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        lognormal_cdf(x, self.mu, self.sigma)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        lognormal_per(x,self.mu, self.sigma)
    }
    pub fn ran(&mut self) -> f64 {
        lognormal_ran(self.mu, self.sigma)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        lognormal_ranvec(n, self.mu, self.sigma)
    }
    pub fn mean(&mut self) -> f64 {
        (self.mu + 0.5 * self.sigma.powi(2)).exp()
    }
    pub fn var(&mut self) -> f64 {
        (self.sigma.powi(2).exp() - 1.0) *
            (2.0 * self.mu + self.sigma.powi(2)).exp()
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ LogNormal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ LogNormal(mu=0.0, sigma=1.0)`. Use.
/// ```
/// use ruststat::lognormal_pdf;
/// println!("Probability density function f(2.1): {}", lognormal_pdf(2.0, 0.0, 1.0));
pub fn lognormal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function lognormal_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (-(x.ln()-mu).powi(2) / (2.0*sigma.powi(2)) - x.ln() - sigma.ln() -
        0.5*(2.0*PI).ln()).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ LogNormal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ LogNormal(mu=0.0, sigma=1.0)`. Use.
/// ```
/// use ruststat::lognormal_cdf;
/// println!("P(X<=2.1): {}", lognormal_cdf(2.0, 0.0, 1.0));
/// ```
pub fn lognormal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function lognormal_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }

    0.5 * (1.0 + erf((x.ln() - mu)/(sigma*2.0f64.sqrt())))
}


/// Computes a percentile for `X ~ LogNormal(mu, sigma)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ LogNormal(mu=0.0, sigma=1.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::lognormal_per;
/// println!("Percentile: {}", lognormal_per(0.8, 0.0, 1.0));
/// ```
pub fn lognormal_per(p: f64, mu: f64, sigma: f64) -> f64 {
    (2f64.sqrt() * sigma * erf_inv(2.0*p-1.0) + mu).exp()
}


/// Random draw from `X ~ LogNormal(mu, sig)` distribution.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ LogNormal(mu=0.0, sigma=1.0)`. Use
/// ```
/// use ruststat::lognormal_ran;
/// println!("Random draw: {}", lognormal_ran(0.0, 1.0));
/// ```
pub fn lognormal_ran(mu: f64, sigma: f64) -> f64 {
    normal_ran(mu, sigma).exp()
}


/// Save random draws from `X ~ LogNormal(mu, sigma)` distribution into a `Vec`
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ LogNormal(mu=0.0, sigma=1.0)`. Use
/// ```
/// use ruststat::lognormal_ranvec;
/// println!("Random Vec: {:?}", lognormal_ranvec(10, 0.5, 2.0));
/// ```
pub fn lognormal_ranvec(n: u64, mu: f64, sigma: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(lognormal_ran(mu, sigma));
    }
    xvec
}


// ==========================================
// ==========================================
// Normal Distribution
// ==========================================
// ==========================================

/// Struct for the normal distribution `X ~ Normal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use
/// ```
/// use ruststat::NormalDist;
/// let mut mynormal = NormalDist{mu:100.0, sigma:16.0};
/// println!("Probability density function f(110.0): {}", mynormal.pdf(110.0));
/// println!("Cumulative distribution function P(X<=110.0): {}", mynormal.cdf(110.0));
/// println!("99th percentile: {}", mynormal.per(0.99));
/// println!("Random draw: {}", mynormal.ran());
/// println!("Random vector: {:?}", mynormal.ranvec(5));
/// println!("Mean: {}", mynormal.mean());
/// println!("Variance: {}", mynormal.var());
/// println!("Standard deviation: {}", mynormal.sd());
/// ```
pub struct NormalDist {
    pub mu: f64,
    pub sigma: f64,
}
impl NormalDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        normal_pdf(x, self.mu, self.sigma)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        normal_cdf(x, self.mu, self.sigma)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        normal_per(x,self.mu, self.sigma)
    }
    pub fn ran(&mut self) -> f64 {
        normal_ran(self.mu, self.sigma)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        normal_ranvec(n,self.mu, self.sigma)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu
    }
    pub fn var(&mut self) -> f64 {
        self.sigma * self.sigma
    }
    pub fn sd(&mut self) -> f64 {
        self.sigma
    }
}


/// Computes probability density function (pdf) for `X ~ Normal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use.
/// ```
/// use ruststat::normal_pdf;
/// println!("Probability density function f(110.0): {}", normal_pdf(110.0, 100.0, 16.0));
pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function normal_pdf");
        return f64::NAN;
    }
    (-0.5*(2.0*PI*sigma).ln() -0.5*((x-mu)/sigma).powi(2)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Normal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use.
/// ```
/// use ruststat::normal_cdf;
/// println!("P(X<=110.0): {}", normal_cdf(110.0, 100.0, 16.0));
/// ```
pub fn normal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function normal_cdf");
        return f64::NAN;
    }

    0.5 * (1.0 + erf(((x-mu)/sigma) / 2.0f64.sqrt()))
}


/// Computes a percentile for `X ~ Normal(mu, sigma)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::normal_per;
/// println!("Percentile: {}", normal_per(0.8, 100.0, 16.0));
/// ```
pub fn normal_per(p: f64, mu: f64, sigma: f64) -> f64 {
    mu + sigma * 2.0f64.sqrt() * erf_inv(2.0*p-1.0)
}


/// Random draw from `X ~ Normal(mu, sigma)` distribution.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use
/// ```
/// use ruststat::normal_ran;
/// println!("Random draw: {}", normal_ran(100.0, 16.0));
/// ```
pub fn normal_ran(mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("Bad argument to normal_ran");
        return f64::NAN;
    }

    // Use the standard random() shortcut directly
    let u1: f64 = random();
    let u2: f64 = random();

    // The Box-Muller transform
    mu + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}


/// Save random draws from `X ~ Normal(mu, sigma)` distribution into a `Vec`
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use
/// ```
/// use ruststat::normal_ranvec;
/// println!("Random Vec of normals: {:?}", normal_ranvec(10, 100.0, 16.0));
/// ```
pub fn normal_ranvec(n: u64, mu: f64, sigma: f64) -> Vec<f64> {
    if n == 0 || sigma <= 0.0 {
        println!("Bad argument to normal_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }

    // Pre-allocate the exact amount of memory needed
    let mut zvecthr: Vec<f64> = Vec::with_capacity(n as usize);

    // Calculate how many loop iterations we need (generating 2 at a time)
    let niter = (n + 1) / 2;

    for _ in 0..niter {
        // Use the standard random() shortcut directly
        let u1: f64 = random();
        let u2: f64 = random();

        // Calculate the shared components of the Box-Muller transform once
        let r = sigma * (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        // Generate the pair
        let z1 = mu + r * theta.cos();
        let z2 = mu + r * theta.sin();

        zvecthr.push(z1);
        zvecthr.push(z2);
    }

    // If 'n' was an odd number, we generated exactly 1 too many. Pop it off.
    if n % 2 != 0 {
        zvecthr.pop();
    }

    zvecthr
}



// ==========================================
// ==========================================
// Pareto (Type I) Distribution
// ==========================================
// ==========================================

/// Struct for the Pareto (type I) distribution `X ~ Pareto1(sigma, alpha)`.
///
/// # Parameters
/// - `sigma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= sigma`
/// # Example
/// Suppose `X ~ Pareto1(sigma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::Pareto1Dist;
/// let mut mydist = Pareto1Dist{sigma:2.0, alpha:0.5};
/// println!("Probability density function f(2.5): {}", mydist.pdf(2.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(2.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto1Dist {
    pub sigma: f64,
    pub alpha: f64,
}
impl Pareto1Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n, self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn mean(&mut self) -> f64 {
        self.sigma + self.sigma * gamma(self.alpha-1.0) *
            gamma(1.0 + 1.0) / gamma(self.alpha)
    }
    pub fn var(&mut self) -> f64 {
        self.sigma.powi(2) * gamma(self.alpha - 1.0 * 2.0) *
            gamma(1.0 + 1.0 * 2.0) / gamma(self.alpha) - self.mean().powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


// ==========================================
// ==========================================
// Pareto (Type II) Distribution
// ==========================================
// ==========================================

/// Struct for the Pareto (type II) distribution `X ~ Pareto2(mu, sigma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto2(mu=1.0, sigma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::Pareto2Dist;
/// let mut mydist = Pareto2Dist{mu: 1.0, sigma:2.0, alpha:0.5};
/// println!("Probability density function f(2.5): {}", mydist.pdf(2.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(2.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto2Dist {
    pub mu: f64,
    pub sigma: f64,
    pub alpha: f64,
}
impl Pareto2Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n,self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu + self.sigma * gamma(self.alpha-1.0) *
            gamma(1.0 + 1.0) / gamma(self.alpha)
    }
    pub fn var(&mut self) -> f64 {
        self.sigma.powi(2) * gamma(self.alpha - 1.0 * 2.0) *
            gamma(1.0 + 1.0 * 2.0) / gamma(self.alpha) - self.mean().powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


// ==========================================
// ==========================================
// Pareto (Type III) Distribution
// ==========================================
// ==========================================

/// Struct for the Pareto (type III) distribution `X ~ Pareto3(mu, sigma, gamma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto3(mu=0.0, sigma=1.0, gamma=2.0)`. Use
/// ```
/// use ruststat::Pareto3Dist;
/// let mut mydist = Pareto3Dist{mu:0.0, sigma:1.0, gamma:2.0};
/// println!("Probability density function f(1.5): {}", mydist.pdf(1.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(1.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto3Dist {
    pub mu: f64,
    pub sigma: f64,
    pub gamma: f64,
}
impl Pareto3Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n,self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu + self.sigma * gamma(1.0-self.gamma) *
            gamma(1.0 + self.gamma) / gamma(1.0)
    }
    pub fn var(&mut self) -> f64 {
        self.sigma.powi(2) * gamma(1.0 - self.gamma * 2.0) *
            gamma(1.0 + self.gamma * 2.0) / gamma(1.0) - self.mean().powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


// ==========================================
// ==========================================
// Pareto (Type IV) Distribution
// ==========================================
// ==========================================

/// Struct for the Pareto (type IV) distribution `X ~ Pareto4(mu, sigma, gamma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::Pareto4Dist;
/// let mut mydist = Pareto4Dist{mu:0.0, sigma:1.0, gamma:2.0, alpha:0.5};
/// println!("Probability density function f(1.5): {}", mydist.pdf(1.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(1.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto4Dist {
    pub mu: f64,
    pub sigma: f64,
    pub gamma: f64,
    pub alpha: f64,
}
impl Pareto4Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n,self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu + self.sigma * gamma(self.alpha-self.gamma) *
            gamma(1.0 + self.gamma) / gamma(self.alpha)
    }
    pub fn var(&mut self) -> f64 {
        self.sigma.powi(2) * gamma(self.alpha - self.gamma * 2.0) *
            gamma(1.0 + self.gamma * 2.0) / gamma(self.alpha) - self.mean().powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Pareto(mu, sigma, gamma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use.
/// ```
/// use ruststat::pareto4_pdf;
/// println!("Probability density function f(1.5): {}", pareto4_pdf(1.5, 0.0, 1.0, 2.0, 0.5));
pub fn pareto4_pdf(x: f64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || gamma <= 0.0 || alpha <= 0.0 {
        println!("NAN produced. Error in function pareto_pdf");
        return f64::NAN;
    }
    if x < mu {
        return 0.0;
    }
    (alpha/(gamma*sigma))*((x-mu)/sigma).powf(1.0/gamma - 1.0)*
        (1.0 + ((x-mu)/sigma).powf(1.0/gamma)).powf(-alpha-1.0)
}


/// Computes cumulative distribution function (cdf) for `X ~ Pareto(mu, sigma, gamma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use.
/// ```
/// use ruststat::pareto4_cdf;
/// println!("P(X<=1.5): {}", pareto4_cdf(1.5, 0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_cdf(x: f64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || gamma <= 0.0 || alpha <= 0.0 {
        println!("NAN produced. Error in function pareto_cdf");
        return f64::NAN;
    }
    if x < mu {
        return 0.0;
    }

    1.0 - (1.0 + ((x-mu)/sigma).powf(1.0/gamma)).powf(-alpha)
}


/// Computes a percentile for `X ~ Pareto(mu, sigma, gamma, alpha)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::pareto4_per;
/// println!("Percentile: {}", pareto4_per(0.8, 0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_per(p: f64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    mu + sigma * ((1.0-p).powf(-1.0/alpha) - 1.0).powf(gamma)
}


/// Random draw from `X ~ Pareto(mu, sigma, gamma, alpha)` distribution.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::pareto4_ran;
/// println!("Random draw: {}", pareto4_ran(0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_ran(mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    mu + sigma * ((1.0-unif_ran(0.0, 1.0)).powf(-1.0/alpha) - 1.0).powf(gamma)
}


/// Save random draws from `X ~ Pareto(mu, sigma, gamma, alpha)` distribution into a `Vec`
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::pareto4_ranvec;
/// println!("Random Vec: {:?}", pareto4_ranvec(10, 0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_ranvec(n: u64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(pareto4_ran(mu, sigma, gamma, alpha));
    }
    xvec
}


// ==========================================
// ==========================================
// t Distribution
// ==========================================
// ==========================================

/// Struct for Student's t distribution `X ~ t(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25)`. Use
/// ```
/// use ruststat::TDist;
/// let mut myt = TDist{nu:25.0};
/// println!("Probability density function f(2.2): {}", myt.pdf(2.2));
/// println!("Cumulative distribution function P(X<=2.2): {}", myt.cdf(2.2));
/// println!("99th percentile: {}", myt.per(0.99));
/// println!("Random draw: {}", myt.ran());
/// println!("Random vector: {:?}", myt.ranvec(5));
/// println!("Mean: {}", myt.mean());
/// println!("Variance: {}", myt.var());
/// println!("Standard deviation: {}", myt.sd());
/// ```
pub struct TDist {
    pub nu: f64,
}
impl TDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        t_pdf(x, self.nu)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        t_cdf(x, self.nu)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        t_per(p,self.nu)
    }
    pub fn ran(&mut self) -> f64 {
        t_ran(self.nu)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        t_ranvec(n, self.nu)
    }
    pub fn mean(&mut self) -> f64 {
        0.0
    }
    pub fn var(&mut self) -> f64 {
        if self.nu > 2.0 {
            self.nu / (self.nu - 2.0)
        } else {
            f64::NAN
        }
    }
    pub fn sd(&mut self) -> f64 {
        if self.nu > 2.0 {
            self.var().sqrt()
        } else {
            f64::NAN
        }
    }
}


/// Computes probability density function (pdf) for `X ~ t(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. Use.
/// ```
/// use ruststat::t_pdf;
/// println!("Probability density function f(2.2): {}", t_pdf(2.2, 25.0));
pub fn t_pdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function t_pdf");
        return f64::NAN;
    }
    (gamma_ln((nu+1.0)/2.0) - ((nu+1.0)/2.0)*(1.0 + x.powi(2)/nu).ln() -
        0.5*(nu * PI).ln() - gamma_ln(nu/2.0)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ t(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ nu(nu=25.0)`. Use.
/// ```
/// use ruststat::t_cdf;
/// println!("P(X<=2.2): {}", t_cdf(2.2, 25.0));
/// ```
pub fn t_cdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function t_cdf");
        return f64::NAN;
    }

    if x <= 0.0 {
        0.5 * betai(nu / (nu + x * x), nu / 2.0, 0.5)
    } else {
        1.0 - 0.5 * betai(nu / (nu + x * x), nu / 2.0, 0.5)
    }
}


/// Computes a percentile for `X ~ t(nu)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::t_per;
/// println!("Percentile: {}", t_per(0.8, 25.0));
/// ```
pub fn t_per(p: f64, nu: f64) -> f64 {
    let (a, x): (f64, f64);
    if p <= 0.5 {
        a = betai_inv(2.0 * p, nu / 2.0, 1.0 / 2.0);
        x = (nu / a - nu).sqrt();
        -x
    } else {
        a = betai_inv(2.0 * (1.0 - p), nu / 2.0, 1.0 / 2.0);
        x = (nu / a - nu).sqrt();
        x
    }
}


/// Random draw from `X ~ t(nu)` distribution.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. Use
/// ```
/// use ruststat::t_ran;
/// println!("Random draw: {}", t_ran(25.0));
/// ```
pub fn t_ran(nu: f64) -> f64 {
    normal_ran(0.0, 1.0) *
        (nu / gamma_ran(nu/2.0, 1.0/2.0)).sqrt()
}


/// Save random draws from `X ~ t(nu)` distribution into a `Vec`
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. Use
/// ```
/// use ruststat::t_ranvec;
/// println!("Random Vec: {:?}", t_ranvec(10, 25.0));
/// ```
pub fn t_ranvec(n: u64, nu: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(t_ran(nu));
    }
    xvec
}


// ==========================================
// ==========================================
// Continuous Uniform Distribution
// ==========================================
// ==========================================

/// Struct for continuous uniform distribution `X ~ Unif(a,b)`.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use
/// ```
/// use ruststat::UnifDist;
/// let mut myunif = UnifDist{a:0.0, b:1.0};
/// println!("Probability density function f(0.4): {}", myunif.pdf(0.4));
/// println!("Cumulative distribution function P(X<=0.4): {}", myunif.cdf(0.4));
/// println!("99th percentile: {}", myunif.per(0.99));
/// println!("Random draw: {}", myunif.ran());
/// println!("Random vector: {:?}", myunif.ranvec(5));
/// println!("Mean: {}", myunif.mean());
/// println!("Variance: {}", myunif.var());
/// println!("Standard deviation: {}", myunif.sd());
/// ```
pub struct UnifDist {
    pub a: f64,
    pub b: f64,
}
impl UnifDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        unif_pdf(x, self.a, self.b)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        unif_cdf(x, self.a, self.b)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        unif_per(x,self.a, self.b)
    }
    pub fn ran(&mut self) -> f64 {
        unif_ran(self.a, self.b)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        unif_ranvec(n, self.a, self.b)
    }
    pub fn mean(&mut self) -> f64 {
        (self.a + self.b) / 2.0
    }
    pub fn var(&mut self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Unif(a,b)`.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use.
/// ```
/// use ruststat::unif_pdf;
/// println!("Probability density function f(0.4): {}", unif_pdf(0.4, 0.0, 1.0));
pub fn unif_pdf(x: f64, a: f64, b: f64) -> f64 {
    if a >= b {
        println!("NAN produced. Error in function unif_pdf");
        return f64::NAN;
    }
    if x < a || x > b {
        return 0.0;
    }
    1.0 / (b - a)
}


/// Computes cumulative distribution function (cdf) for `X ~ Unif(a,b)`.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use.
/// ```
/// use ruststat::unif_cdf;
/// println!("P(X<=0.4): {}", unif_cdf(0.4, 0.0, 1.0));
/// ```
pub fn unif_cdf(x: f64, a: f64, b: f64) -> f64 {
    if a >= b {
        println!("NAN produced. Error in function unif_cdf");
        return f64::NAN;
    }
    if x < a {
        return 0.0;
    }
    if x > b {
        return 1.0;
    }

    (x - a) / (b - a)
}


/// Computes a percentile for `X ~ Unif(a,b)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::unif_per;
/// println!("Percentile: {}", unif_per(0.8, 0.0, 1.0));
/// ```
pub fn unif_per(p: f64, a: f64, b: f64) -> f64 {
    if a >= b || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function unif_per");
        return f64::NAN;
    }

    a + p * (b - a)
}

/// Random draw from `X ~ Unif(a,b)` distribution.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use
/// ```
/// use ruststat::unif_ran;
/// println!("Random draw: {}", unif_ran(0.0, 1.0));
/// ```
pub fn unif_ran(a: f64, b: f64) -> f64 {
    let u: f64;
    u = random();
    // u = rand::thread_rng().gen();
    a + u * (b-a)
}


/// Save random draws from `X ~ Unif(a,b)` distribution into a `Vec`
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use
/// ```
/// use ruststat::unif_ranvec;
/// println!("Random Vec: {:?}", unif_ranvec(10, 0.0, 1.0));
/// ```
pub fn unif_ranvec(n: u64, a: f64, b: f64) -> Vec<f64> {
    // Tell Rust exactly how much memory we need upfront
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);

    for _ in 0..n {
        xvec.push(unif_ran(a, b)); // Now it just pushes without ever resizing
    }
    xvec
}


// ==========================================
// ==========================================
// ==========================================
// Discrete RVs
// ==========================================
// ==========================================
// ==========================================


// ==========================================
// ==========================================
// Binomial Distribution
// ==========================================
// ==========================================

/// Struct for the binomial distribution `X ~ Bin(n,p)`.
///
/// # Parameters
/// - `n` = number of trials (`n = 1,2,...`)
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...,n`
/// # Example
/// Suppose `X ~ Bin(n=100,p=0.9)`. Use
/// ```
/// use ruststat::BinDist;
/// let mut mybin = BinDist{n:100, p:0.9};
/// println!("P(X=80): {}", mybin.pmf(80));
/// println!("P(X<=80): {}", mybin.cdf(80));
/// println!("99th percentile: {}", mybin.per(0.99));
/// println!("Random draw: {}", mybin.ran());
/// println!("Random vector: {:?}", mybin.ranvec(5));
/// println!("Mean: {}", mybin.mean());
/// println!("Variance: {}", mybin.var());
/// println!("Standard deviation: {}", mybin.sd());
/// ```
pub struct BinDist {
    pub n: u64,
    pub p: f64,
}
impl BinDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        bin_pmf(x, self.n, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        bin_cdf(x, self.n, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        bin_per(q, self.n, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        bin_ran(self.n, self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        bin_ranvec(n, self.n, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        (self.n as f64) * self.p
    }
    pub fn var(&mut self) -> f64 {
        (self.n as f64) * self.p * (1.0-self.p)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ Bin(n,p)`
///
/// # Parameters
/// - `n` = number of trials (`n = 1,2,...`)
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...,n`
/// # Example
/// Suppose `X ~ Bin(n=10,p=0.6)`. To compute `P(X = 7)`, use
/// ```
/// use ruststat::bin_pmf;
/// println!("P(X=x): {}", bin_pmf(7, 10, 0.6));
/// ```
pub fn bin_pmf(x: u64, n: u64, p: f64) -> f64 {
    if n <= 0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function bin_pmf");
        return f64::NAN;
    }
    if x > n {
        return 0.0;
    }

    // --- The Edge Case Patch ---
    if p == 0.0 {
        return if x == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if x == n { 1.0 } else { 0.0 };
    }
    // ---------------------------

    (factln_i(n) - factln_i(x) - factln_i(n-x) + x as f64 * p.ln() + (n-x) as f64 * (1.0-p).ln()).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ Bin(n,p)`
///
/// # Parameters
/// - `n` = number of trials (`n = 1,2,...`)
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...,n`
/// # Example
/// Suppose `X ~ Bin(n=10,p=0.6)`. To compute `P(X <= 7)`, use
/// ```
/// use ruststat::bin_cdf;
/// println!("P(X<=x): {}", bin_cdf(7, 10, 0.6));
/// ```
pub fn bin_cdf(x: u64, n: u64, p: f64) -> f64 {
    if n <= 0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function bin_cdf");
        return f64::NAN;
    }
    if x > n {
        return 1.0;
    }

    let mut sum = 0.0;
    for i in 0..=x {
        sum += bin_pmf(i, n, p);
    }
    if sum < 1.0 {
        sum
    } else {
        1.0
    }
}


/// Computes a percentile for `X ~ Bin(n,p)`
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `n` = number of trials (`n = 1,2,...`)
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...,n`
/// # Example
/// Suppose `X ~ Bin(n=10,p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::bin_per;
/// println!("Percentile: {}", bin_per(0.8, 10, 0.6));
/// ```
pub fn bin_per(q: f64, n: u64, p: f64) -> u64 {
    let mut x = 0;

    if n <= 0 || p < 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function bin_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 0;
    }
    if q == 1.0 {
        return n;
    }

    while bin_cdf(x, n, p) < q {
        x += 1;
    }
    x
}


/// Random draw from `X ~ Bin(n,p)` distribution
/// Sheldon Ross, Simulation, (2003)
/// # Parameters
/// - `n` = number of trials (`n = 1,2,...`)
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...,n`
/// # Example
/// Suppose `X ~ Bin(n=10,p=0.6)`. Use
/// ```
/// use ruststat::bin_ran;
/// println!("Random draw: {}", bin_ran(10, 0.6));
/// ```
pub fn bin_ran(n: u64, p: f64) -> u64 {

    if n <= 0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function bin_ran");
        return f64::NAN as u64;
    }
    if p == 0.0 {
        return 0;
    }
    if p == 1.0 {
        return n;
    }

    let u : f64;
    u = unif_ran(0.0, 1.0);
    let c = p / (1.0 - p);
    let mut i = 0;
    let mut pr = (1.0 - p).powi(n as i32);
    let mut f = pr;
    loop {
        if u < f {
            return i;
        }
        pr = c * ((n - i) as f64 / (i + 1) as f64) * pr;
        f = f + pr;
        i = i + 1;
    }
}

/// Save random draws from `X ~ Bin(n,p)` distribution into a `Vec`
/// # Parameters
/// - `n` = number of trials (`n = 1,2,...`)
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Example
/// Suppose `X ~ Bin(n=10,p=0.6)`. Use
/// ```
/// use ruststat::bin_ranvec;
/// println!("Random Vec: {:?}", bin_ranvec(10, 10, 0.6));
/// ```
pub fn bin_ranvec(nn: u64, n: u64, p: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(bin_ran(n,p));
    }
    xvec
}


// ==========================================
// ==========================================
// Geometric Distribution
// ==========================================
// ==========================================

/// Struct for the geometric distribution `X ~ Geo(p)` where
/// `X =` the number of failures prior to observing the first success.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Geo(p=0.2)`. Use
/// ```
/// use ruststat::GeoDist;
/// let mut mygeo = GeoDist{p:0.2};
/// println!("probability mass function: {}", mygeo.pmf(8));
/// println!("cumulative distribution function: {}", mygeo.cdf(8));
/// println!("percentile: {}", mygeo.per(0.99));
/// println!("random: {}", mygeo.ran());
/// println!("Random vector: {:?}", mygeo.ranvec(5));
/// println!("mean: {}", mygeo.mean());
/// println!("variance: {}", mygeo.var());
/// println!("standard deviation: {}", mygeo.sd());
/// ```
pub struct GeoDist {
    pub p: f64,
}
impl GeoDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        geo_pmf(x, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        geo_cdf(x, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        geo_per(q, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        geo_ran(self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        geo_ranvec(n, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        (1.0 - self.p) / self.p
    }
    pub fn var(&mut self) -> f64 {
        (1.0 - self.p) / self.p.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ Geo(p)`
/// where `X =` the number of failures prior to the first success.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To compute `P(X = 3)`, use
/// ```
/// use ruststat::geo_pmf;
/// println!("P(X=x): {}", geo_pmf(3, 0.6));
/// ```
pub fn geo_pmf(x: u64, p: f64) -> f64 {
    if p < 0.0 || p > 1.0 { // Note: changed p <= 0.0 to p < 0.0 to allow the 0.0 check
        println!("NAN produced. Error in function geo_pmf");
        return f64::NAN;
    }

    // --- The Edge Case Fast Paths ---
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return if x == 0 { 1.0 } else { 0.0 };
    }
    // --------------------------------

    (1.0-p).powi(x as i32) * p
}


/// Computes cumulative distribution function (cdf) for `X ~ Geo(p)`
/// where `X =` the number of failures prior to the first success.
///
/// # Parameters
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To compute `P(X <= 3)`, use
/// ```
/// use ruststat::geo_cdf;
/// println!("P(X<=x): {}", geo_cdf(3, 0.6));
/// ```
pub fn geo_cdf(x: u64, p: f64) -> f64 {
    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function geo_pmf");
        return f64::NAN;
    }

    let mut sum = 0.0;
    for i in 0..=x {
        sum += geo_pmf(i, p);
    }

    sum
}


/// Computes a percentile for `X ~ Geo(p)`
/// where `X =` the number of failures prior to the first success.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::geo_per;
/// println!("Percentile: {}", geo_per(0.8, 0.6));
/// ```
pub fn geo_per(q: f64, p: f64) -> u64 {
    let mut x = 0;

    if p <= 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function geo1_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 0;
    }
    if q == 1.0 {
        return u64::MAX;
    }

    while geo_cdf(x, p) < q {
        x += 1;
    }

    x
}


/// Random draw from `X ~ Geo(p)` distribution
/// where `X =` the number of failures prior to the first success.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. Use
/// ```
/// use ruststat::geo_ran;
/// println!("Random draw: {}", geo_ran(0.6));
/// ```
pub fn geo_ran(p: f64) -> u64 {

    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function geo1_ran");
        return f64::NAN as u64;
    }

    let u = unif_ran(0.0, 1.0);

    (u.ln() / (1.0-p).ln()).floor() as u64
}


/// Save random draws from `X ~ Geo(p)` distribution into a `Vec`
/// # Parameters
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. Use
/// ```
/// use ruststat::geo_ranvec;
/// println!("Random Vec: {:?}", geo_ranvec(10, 0.6));
/// ```
pub fn geo_ranvec(nn: u64, p: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(geo_ran(p));
    }
    xvec
}


// ==========================================
// ==========================================
// Geometric Distribution 2
// ==========================================
// ==========================================

/// Struct for the geometric distribution `X ~ Geo(p)` where
/// `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.2)`. Use
/// ```
/// use ruststat::Geo2Dist;
/// let mut mygeo = Geo2Dist{p:0.2};
/// println!("probability mass function: {}", mygeo.pmf(8));
/// println!("cumulative distribution function: {}", mygeo.cdf(8));
/// println!("percentile: {}", mygeo.per(0.99));
/// println!("random: {}", mygeo.ran());
/// println!("Random vector: {:?}", mygeo.ranvec(5));
/// println!("mean: {}", mygeo.mean());
/// println!("variance: {}", mygeo.var());
/// println!("standard deviation: {}", mygeo.sd());
/// ```
pub struct Geo2Dist {
    pub p: f64,
}
impl Geo2Dist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        geo2_pmf(x, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        geo2_cdf(x, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        geo2_per(q, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        geo2_ran(self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        geo2_ranvec(n, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        1.0 / self.p
    }
    pub fn var(&mut self) -> f64 {
        (1.0 - self.p) / self.p.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ Geo(p)`
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To compute `P(X = 3)`, use
/// ```
/// use ruststat::geo2_pmf;
/// println!("P(X=x): {}", geo2_pmf(3, 0.6));
/// ```
pub fn geo2_pmf(x: u64, p: f64) -> f64 {
    if p < 0.0 || p > 1.0 { // Changed to p < 0.0
        println!("NAN produced. Error in function geo2_pmf");
        return f64::NAN;
    }
    if x < 1 {
        return 0.0;
    }

    // --- The Edge Case Fast Paths ---
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return if x == 1 { 1.0 } else { 0.0 };
    }
    // --------------------------------

    (1.0-p).powi((x-1) as i32) * p
}

/// Computes cumulative distribution function (cdf) for `X ~ Geo(p)`
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To compute `P(X <= 3)`, use
/// ```
/// use ruststat::geo2_cdf;
/// println!("P(X<=x): {}", geo2_cdf(3, 0.6));
/// ```
pub fn geo2_cdf(x: u64, p: f64) -> f64 {
    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function geo2_cdf");
        return f64::NAN;
    }
    if x < 1 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 1..=x {
        sum += geo2_pmf(i, p);
    }

    sum
}


/// Computes a percentile for `X ~ Geo(p)`
/// where `X =` the trial on which the first success is observed.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::geo2_per;
/// println!("Percentile: {}", geo2_per(0.8, 0.6));
/// ```
pub fn geo2_per(q: f64, p: f64) -> u64 {
    let mut x = 1;

    if p <= 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function geo2_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 1;
    }
    if q == 1.0 {
        return u64::MAX;
    }

    while geo2_cdf(x, p) < q {
        x += 1;
    }

    x
}


/// Random draw from `X ~ Geo(p)` distribution
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. Use
/// ```
/// use ruststat::geo2_ran;
/// println!("Random draw: {}", geo2_ran(0.6));
/// ```
pub fn geo2_ran(p: f64) -> u64 {

    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function geo2_ran");
        return f64::NAN as u64;
    }

    let u = unif_ran(0.0, 1.0);

    (u.ln() / (1.0-p).ln()).floor() as u64 + 1
}

/// Save random draws from `X ~ Geo(p)` distribution into a `Vec`
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. Use
/// ```
/// use ruststat::geo2_ranvec;
/// println!("Random Vec: {:?}", geo2_ranvec(10, 0.6));
/// ```
pub fn geo2_ranvec(nn: u64, p: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(geo2_ran(p));
    }

    xvec
}


// ==========================================
// ==========================================
// HyperGeometric Distribution
// ==========================================
// ==========================================

/// Struct for the hypergeometric distribution `X ~ HG(n,N,M)`.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. Use
/// ```
/// use ruststat::HGDist;
/// let mut myhg = HGDist{n:20, N:100, M:50};
/// println!("P(X=6): {}", myhg.pmf(6));
/// println!("P(X<=6): {}", myhg.cdf(6));
/// println!("99th percentile: {}", myhg.per(0.99));
/// println!("Random draw: {}", myhg.ran());
/// println!("Random vector: {:?}", myhg.ranvec(5));
/// println!("Mean: {}", myhg.mean());
/// println!("Variance: {}", myhg.var());
/// println!("Standard deviation: {}", myhg.sd());
/// ```
#[allow(non_snake_case)]
pub struct HGDist {
    pub n: u64,
    pub N: u64,
    pub M: u64,
}
impl HGDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        hg_pmf(x, self.n, self.N, self.M)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        hg_cdf(x, self.n, self.N, self.M)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        hg_per(q, self.n, self.N, self.M)
    }
    pub fn ran(&mut self) -> u64 {
        hg_ran(self.n, self.N, self.M)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        hg_ranvec(n, self.n, self.N, self.M)
    }
    pub fn mean(&mut self) -> f64 {
        (self.n as f64) * (self.M as f64) / (self.N as f64)
    }
    pub fn var(&mut self) -> f64 {
        (self.n as f64) *
            (self.M as f64) / (self.N as f64) *
            (1.0 - (self.M as f64) / (self.N as f64)) *
            (self.N as f64 - self.n as f64) / (self.N as f64 - 1.0)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ HG(n,N,M)`
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20,N=100,M=50)`. To compute `P(X = 7)`, use
/// ```
/// use ruststat::hg_pmf;
/// println!("P(X=x): {}", hg_pmf(7, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_pmf(x: u64, n: u64, N: u64, M: u64) -> f64 {
    if  n < 1 || N < 1 || M > N || n > N { // Removed M < 1 to allow 0 successes in pop
        println!("NAN produced. Error in function hg_pmf");
        return f64::NAN;
    }
    if x > n || x > M || (n-x) > (N-M) {
        return 0.0;
    }

    // --- The Deterministic Fast Paths ---
    if M == 0 {
        // No successes exist in the population
        return if x == 0 { 1.0 } else { 0.0 };
    }
    if M == N {
        // Entire population is a success
        return if x == n { 1.0 } else { 0.0 };
    }
    // ------------------------------------

    (fact_ln_u(M) - fact_ln_u(x) - fact_ln_u(M-x) +
        fact_ln_u(N-M) - fact_ln_u(n-x) - fact_ln_u(N-M-n+x) -
        fact_ln_u(N) + fact_ln_u(n) + fact_ln_u(N-n)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ HG(n,N,M)`
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20,N=100,M=50)`. To compute `P(X <= 7)`, use
/// ```
/// use ruststat::hg_cdf;
/// println!("P(X<=x): {}", hg_cdf(7, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_cdf(x: u64, n: u64, N: u64, M: u64) -> f64 {
    if  n < 1 || N < 1 || M < 1 || M > N || n > N {
        println!("NAN produced. Error in function hg_cdf");
        return f64::NAN;
    }
    if x > M || (n-x) > (N-M) {
        return 0.0;
    }
    if x > n {
        return 1.0;
    }

    let mut sum = 0.0;
    for i in 0..=x {
        sum += hg_pmf(i, n, N, M);
    }

    sum
}


/// Computes a percentile for `X ~ HG(n,N,M)`
/// where `X =` the number of successes.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::hg_per;
/// println!("Percentile: {}", hg_per(0.8, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_per(q: f64, n: u64, N: u64, M: u64) -> u64 {
    let mut x = 0;

    if  n < 1 || N < 1 || M < 1 ||
        M > N || n > N ||
        q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function hg_per");
        return f64::NAN as u64;
    }

    while hg_cdf(x, n, N, M) < q {
        x += 1;
    }

    x
}


/// Random draw from `X ~ HG(n, N, M)` distribution
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. Use
/// ```
/// use ruststat::hg_ran;
/// println!("Random draw: {}", hg_ran(20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_ran(n: u64, N: u64, M: u64) -> u64 {
    if n < 1 || N < 1 || M > N || n > N {
        println!("NAN produced. Error in function hg_ran");
        return f64::NAN as u64;
    }

    // 1. Generate a uniform random probability
    let u = unif_ran(0.0, 1.0);

    // 2. Start at the lowest possible number of successes
    // (If we draw more items than there are failures, we MUST get some successes)
    let mut x = if n > N - M { n - (N - M) } else { 0 };

    // 3. Walk up the CDF until we cross our random threshold
    while hg_cdf(x, n, N, M) < u {
        x += 1;
    }

    x
}

/// Save random draws from `X ~ HG(n,N,M)` distribution into a `Vec`
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. Use
/// ```
/// use ruststat::hg_ranvec;
/// println!("Random Vec: {:?}", hg_ranvec(10, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_ranvec(nn: u64, n: u64, N: u64, M: u64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(hg_ran(n,N,M));
    }
    xvec
}


// ==========================================
// ==========================================
// Negative Binomial Distribution
// ==========================================
// ==========================================

/// Struct for the negative binomial distribution `X ~ NB(r,p)` where
/// `X =` the number of failures prior to observing the `r`th success.
///
/// # Parameters
/// - `r = 1,2,3,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=4,p=0.2)`. Use
/// ```
/// use ruststat::NBDist;
/// let mut mynb = NBDist{r:4, p:0.2};
/// println!("P(X=6): {}", mynb.pmf(6));
/// println!("P(X<=6): {}", mynb.cdf(6));
/// println!("99th percentile: {}", mynb.per(0.99));
/// println!("Random draw: {}", mynb.ran());
/// println!("Random vector: {:?}", mynb.ranvec(5));
/// println!("Mean: {}", mynb.mean());
/// println!("Variance: {}", mynb.var());
/// println!("Standard deviation: {}", mynb.sd());
/// ```
pub struct NBDist {
    pub r: u64,
    pub p: f64,
}
impl NBDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        nb_pmf(x, self.r, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        nb_cdf(x, self.r, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        nb_per(q, self.r, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        nb_ran(self.r, self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        nb_ranvec(n, self.r, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        (self.r as f64) * (1.0 - self.p) / self.p
    }
    pub fn var(&mut self) -> f64 {
        (self.r as f64) * (1.0 - self.p) / self.p.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ NB(r,p)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=2,p=0.6)`. To compute `P(X = 3)`, use
/// ```
/// use ruststat::nb_pmf;
/// println!("P(X=x): {}", nb_pmf(3, 2, 0.6));
/// ```
pub fn nb_pmf(x: u64, r: u64, p: f64) -> f64 {
    if r < 1 || p <= 0.0 || p > 1.0  {
        println!("NAN produced. Error in function nb_pmf");
        return f64::NAN;
    }

    // --- The Edge Case Patch ---
    if p == 1.0 {
        return if x == 0 { 1.0 } else { 0.0 };
    }
    // ---------------------------

    (fact_ln_u(x+r-1) - fact_ln_u(r-1) - fact_ln_u(x) + (r as f64)*p.ln() +
        (x as f64)*(1.0-p).ln()).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ NB(r,p)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. To compute `P(X <= 3)`, use
/// ```
/// use ruststat::nb_cdf;
/// println!("P(X<=x): {}", nb_cdf(3, 2, 0.6));
/// ```
pub fn nb_cdf(x: u64, r: u64, p: f64) -> f64 {
    if r < 1 || p <= 0.0 || p > 1.0  {
        println!("NAN produced. Error in function nb_cdf");
        return f64::NAN;
    }
    let mut sum = 0.0;
    for i in 0..=x {
        sum += nb_pmf(i, r, p);
    }
    sum
}


/// Computes a percentile for `X ~ NB(r,p)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::nb_per;
/// println!("Percentile: {}", nb_per(0.8, 2, 0.6));
/// ```
pub fn nb_per(q: f64, r: u64, p: f64) -> u64 {
    let mut x = 0;

    if p <= 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function nb_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 0;
    }
    if q == 1.0 {
        return u64::MAX;
    }

    while nb_cdf(x, r, p) < q {
        x += 1;
    }
    x
}


/// Random draw from `X ~ NB(r, p)` distribution
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. Use
/// ```
/// use ruststat::nb_ran;
/// println!("Random draw: {}", nb_ran(2, 0.6));
/// ```
pub fn nb_ran(r: u64, p: f64) -> u64 {

    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function nb_ran");
        return f64::NAN as u64;
    }

    let mut geo_vec = Vec::new();
    for _ in 0..r {
        geo_vec.push(geo_ran(p));
    }
    geo_vec.iter().sum()
}


/// Save random draws from `X ~ NB(r,p)` distribution into a `Vec`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. Use
/// ```
/// use ruststat::nb_ranvec;
/// println!("Random Vec: {:?}", nb_ranvec(10, 2, 0.6));
/// ```
pub fn nb_ranvec(nn: u64, r: u64, p: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(nb_ran(r,p));
    }
    xvec
}


// ==========================================
// ==========================================
// Negative Binomial Distribution 2
// ==========================================
// ==========================================

/// Struct for the negative binomial distribution `X ~ NB(r,p)` where
/// `X =` the trial on which the `r`th success is observed.
///
/// # Parameters
/// - `r = r,r+1,r+2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = r,r+1,r+2,...`
/// # Example
/// Suppose `X ~ NB(r=4,p=0.2)`. Use
/// ```
/// use ruststat::NB2Dist;
/// let mut mynb = NB2Dist{r:4, p:0.2};
/// println!("P(X=6): {}", mynb.pmf(6));
/// println!("P(X<=6): {}", mynb.cdf(6));
/// println!("99th percentile: {}", mynb.per(0.99));
/// println!("Random draw: {}", mynb.ran());
/// println!("Random vector: {:?}", mynb.ranvec(5));
/// println!("Mean: {}", mynb.mean());
/// println!("Variance: {}", mynb.var());
/// println!("Standard deviation: {}", mynb.sd());
/// ```
pub struct NB2Dist {
    pub r: u64,
    pub p: f64,
}
impl NB2Dist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        nb2_pmf(x, self.r, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        nb2_cdf(x, self.r, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        nb2_per(q, self.r, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        nb2_ran(self.r, self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        nb2_ranvec(n, self.r, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        (self.r as f64) / self.p
    }
    pub fn var(&mut self) -> f64 {
        (self.r as f64) * (1.0 - self.p) / self.p.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ NB(r,p)`
/// where `X =` the trial on which the `r`th success is observed.
///
/// # Parameters
/// - `r = r,r+1,r+2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = r,r+1,r+2,...`
/// # Example
/// Suppose `X ~ NB(r=2,p=0.6)`. To compute `P(X = 3)`, use
/// ```
/// use ruststat::nb2_pmf;
/// println!("P(X=x): {}", nb2_pmf(3, 2, 0.6));
/// ```
pub fn nb2_pmf(x: u64, r: u64, p: f64) -> f64 {
    if  r < 1 || p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function nb2_pmf");
        return f64::NAN;
    }
    if x < r {
        return 0.0;
    }

    // --- The Edge Case Patch ---
    if p == 1.0 {
        return if x == r { 1.0 } else { 0.0 };
    }
    // ---------------------------

    (fact_ln_u(x-1) - fact_ln_u(r-1) - fact_ln_u(x-r) + (r as f64)*p.ln() +
        ((x-r) as f64)*(1.0-p).ln()).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ NB(r,p)`
/// where `X =` the trial on which the `r`th success is observed.
///
/// # Parameters
/// - `r = r,r+1,r+2,...`
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = r,r+1,r+2,...`
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. To compute `P(X <= 3)`, use
/// ```
/// use ruststat::nb2_cdf;
/// println!("P(X<=x): {}", nb2_cdf(3, 2, 0.6));
/// ```
pub fn nb2_cdf(x: u64, r: u64, p: f64) -> f64 {
    if  r < 1 || p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function nb2_cdf");
        return f64::NAN;
    }
    if x < r {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in r..=x {
        sum += nb2_pmf(i, r, p);
    }
    sum
}


/// Computes a percentile for `X ~ NB(r,p)`
/// where `X =` the trial on which the `r`th success is observed.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `r = r,r+1,r+2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = r,r+1,r+2,...`
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::nb2_per;
/// println!("Percentile: {}", nb2_per(0.8, 2, 0.6));
/// ```
pub fn nb2_per(q: f64, r: u64, p: f64) -> u64 {

    if  r < 1 || p <= 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function nb2_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return r;
    }
    if q == 1.0 {
        return f64::INFINITY as u64;
    }

    let mut x = r;
    while nb2_cdf(x, r, p) < q {
        x += 1;
    }
    x
}


/// Random draw from `X ~ NB(r, p)` distribution
/// where `X =` the trial on which the `r`th success is observed.
///
/// # Parameters
/// - `r = r,r+1,r+2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. Use
/// ```
/// use ruststat::nb2_ran;
/// println!("Random draw: {}", nb2_ran(2, 0.6));
/// ```
pub fn nb2_ran(r: u64, p: f64) -> u64 {

    if  r < 1 || p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function nb2_ran");
        return f64::NAN as u64;
    }

    let mut geo2_vec = Vec::new();
    for _ in 0..r {
        geo2_vec.push(geo2_ran(p));
    }
    geo2_vec.iter().sum()
}


/// Save random draws from `X ~ NB(r,p)` distribution into a `Vec`
/// where `X =` the trial on which the `r`th success is observed.
///
/// # Parameters
/// - `r = r,r+1,r+2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. Use
/// ```
/// use ruststat::nb2_ranvec;
/// println!("Random Vec: {:?}", nb2_ranvec(10, 2, 0.6));
/// ```
pub fn nb2_ranvec(nn: u64, r: u64, p: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(nb2_ran(r,p));
    }
    xvec
}


// ==========================================
// ==========================================
// Poisson Distribution
// ==========================================
// ==========================================

/// Struct for the Poisson distribution `X ~ Pois(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. Use
/// ```
/// use ruststat::PoisDist;
/// let mut mypois = PoisDist{lambda:2.5};
/// println!("P(X=6): {}", mypois.pmf(6));
/// println!("P(X<=6): {}", mypois.cdf(6));
/// println!("99th percentile: {}", mypois.per(0.99));
/// println!("Random draw: {}", mypois.ran());
/// println!("Random vector: {:?}", mypois.ranvec(5));
/// println!("Mean: {}", mypois.mean());
/// println!("Variance: {}", mypois.var());
/// println!("Standard deviation: {}", mypois.sd());
/// ```
pub struct PoisDist {
    pub lambda: f64,
}
impl PoisDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        pois_pmf(x, self.lambda)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        pois_cdf(x, self.lambda)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        pois_per(q, self.lambda)
    }
    pub fn ran(&mut self) -> u64 {
        pois_ran(self.lambda)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        pois_ranvec(n, self.lambda)
    }
    pub fn mean(&mut self) -> f64 {
        self.lambda
    }
    pub fn var(&mut self) -> f64 {
        self.lambda
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ Pois(lambda).`
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. To compute `P(X = 4)`, use
/// ```
/// use ruststat::pois_pmf;
/// println!("P(X=x): {}", pois_pmf(4, 2.5));
/// ```
pub fn pois_pmf(x: u64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function pois_pmf");
        return f64::NAN;
    }
    (-lambda + (x as f64)*lambda.ln() - fact_ln_u(x)).exp()
    // return (-lambda).exp() * lambda.powi(x) / (fact_i(x) as f64);
}


/// Computes cumulative distribution function (cdf) for `X ~ Pois(lambda).`
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. To compute `P(X <= 4)`, use
/// ```
/// use ruststat::pois_cdf;
/// println!("P(X<=x): {}", pois_cdf(4, 2.5));
/// ```
pub fn pois_cdf(x: u64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function pois_cdf");
        return f64::NAN;
    }
    gammq(lambda, (x+1) as f64)
}


/// Computes a percentile for `X ~ Pois(lambda)`.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::pois_per;
/// println!("Percentile: {}", pois_per(2.5, 0.8));
/// ```
pub fn pois_per(q: f64, lambda: f64) -> u64 {
    let mut x = 0;

    if lambda <= 0.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function pois_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 0;
    }
    if q == 1.0 {
        return u64::MAX;
    }

    while pois_cdf(x, lambda) < q {
        x += 1;
    }
    x
}


/// Random draw from `X ~ Pois(r, p)` distribution.
/// Sheldon Ross, Simulation, 2003
///
/// # Parameters
/// - `lambda > 0`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. Use
/// ```
/// use ruststat::pois_ran;
/// println!("Random draw: {}", pois_ran(2.5));
/// ```
pub fn pois_ran(lambda: f64) -> u64 {

    if lambda <= 0.0 {
        println!("NAN produced. Error in function pois_ran");
        return f64::NAN as u64;
    }

    let u = unif_ran(0.0, 1.0);
    let mut i = 0;
    let mut p = (-lambda).exp();
    let mut f = p;
    loop {
        if u < f {
            return i;
        }
        p = lambda*p / (i+1) as f64;
        f = f + p;
        i = i + 1;
    }
}


/// Save random draws from `X ~ Pois(lambda)` distribution into a `Vec`
///
/// # Parameters
/// - `lambda > 0`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. Use
/// ```
/// use ruststat::pois_ranvec;
/// println!("Random Vec: {:?}", pois_ranvec(10, 2.5));
/// ```
pub fn pois_ranvec(nn: u64, lambda: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(pois_ran(lambda));
    }
    xvec
}



// ===========================================================================
// ===========================================================================
// Special Functions
// ===========================================================================
// ===========================================================================

/// Incomplete beta function.
///
/// # Parameters
/// - `a > 0`
/// - `b > 0`
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`.
/// ```
/// use ruststat::betai;
/// println!("Incomplete beta function: {}", betai(0.7, 0.5, 2.0));
/// ```
pub fn betai(x: f64, a: f64, b: f64) -> f64 {

    let bt: f64;

    if x < 0.0 || x > 1.0 {
        println!("Bad x in function betai");
        return f64::NAN;
    }
    if x == 0.0 || x == 1.0 {
        bt=0.0;
    }
    else {
        bt=(gamma_ln(a+b)-gamma_ln(a)-gamma_ln(b) +
            a*x.ln() +
            b*(1.0-x).ln()).exp();
    }
    if x < ((a + 1.0) / (a + b + 2.0)) {
        bt * betacf(x, a, b) / a
    } else {
        1.0 - bt * betacf(1.0 - x, b, a) / b
    }
}


fn betacf(x: f64, a: f64, b: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    let fpmin = 1.0e-30;

    let mut m2: u64;
    let mut aa: f64;
    let mut c: f64;
    let mut d: f64;
    let mut del: f64;
    let mut h: f64;
    let qab: f64;
    let qam: f64;
    let qap: f64;

    qab=a+b;
    qap=a+1.0;
    qam=a-1.0;
    c=1.0;
    d=1.0-qab*x/qap;
    if d.abs() < fpmin {
        d=fpmin;
    }
    d=1.0/d;
    h=d;
    for m in 1..=imax {
        m2=2*m;
        aa=(m as f64)*(b-m as f64)*x/((qam+m2 as f64)*(a+m2 as f64));
        d=1.0+aa*d;
        if d.abs() < fpmin {
            d=fpmin;
        }
        c=1.0+aa/c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d=1.0/d;
        h *= d*c;
        aa = -(a+m as f64)*(qab+m as f64)*x/((a+m2 as f64)*(qap+m2 as f64));
        d=1.0+aa*d;
        if d.abs() < fpmin {
            d=fpmin;
        }
        c=1.0+aa/c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d=1.0/d;
        del=d*c;
        h *= del;
        if (del-1.0).abs() < eps {
            break;
        }
    }
    // if (m > MAXIT) nrerror("a or b too big, or MAXIT too small in betacf");
    h
}


/// Log gamma function
/// - `x > 0`
pub fn gamma_ln(x: f64) -> f64 {
    let cof = vec![57.1562356658629235,-59.5979603554754912,
                   14.1360979747417471,-0.491913816097620199,0.339946499848118887e-4,
                   0.465236289270485756e-4,-0.983744753048795646e-4,0.158088703224912494e-3,
                   -0.210264441724104883e-3,0.217439618115212643e-3,-0.164318106536763890e-3,
                   0.844182239838527433e-4,-0.261908384015814087e-4,0.368991826595316234e-5];

    if x <= 0.0 {
        println!("Bad argument in gamma_ln; returning f64::NAN");
        return f64::NAN;
        // panic!("bad ard in gamma_ln");
    }

    let mut y= x;
    let z = x;

    let mut tmp = z + 5.24218750000000000;
    tmp = (z + 0.5) * tmp.ln() - tmp;
    let mut ser = 0.999999999999997092;
    for j in 0..14 {
        ser += cof[j] / { y += 1.0; y};
    }
    tmp + (2.5066282746310005 * ser / z).ln()
}


/// Gamma function
/// - `x > 0`
pub fn gamma(x: f64) -> f64 {
    gamma_ln(x).exp()
}


/// Factorial (`u64`).
/// - `x = 1,2,3,...`
pub fn fact_i(x: u64) -> u64 {
    if x > 1 {
        x * fact_i(x - 1)
    } else {
        1
    }
}


/// Log factorial (`u64`).
/// - `x = 1,2,3,...`
pub fn factln_i(x: u64) -> f64 {
    if x > 1 {
        (x as f64).ln() + factln_i(x - 1)
    } else {
        0.0
    }
}


/// Log factorial (`u64`).
/// - `x = 1,2,3,...`
pub fn fact_ln_u(x: u64) -> f64 {
    if x > 1 {
        (x as f64).ln() + fact_ln_u(x - 1)
    } else {
        0.0
    }
}


/// Generalized factorial (`f64`).
/// - `x > 0`
pub fn fact_f(x: f64) -> f64 {
    if x > 1.0 {
        x * fact_f(x - 1.0)
    } else {
        x * gamma(x)
    }
}


/// Log Generalized factorial (`f64`).
/// - `x > 0`
pub fn factln_f(x: f64) -> f64 {
    if x > 1.0 {
        x.ln() + fact_f(x - 1.0).ln()
    } else {
        x + gamma_ln(x)
    }
}


/// Combination
/// - `n = 0,1,2,...`
/// - `x = 0,1,2,...`
/// - `n >= x`
pub fn comb(n: u64, x: u64) -> f64 {
    (0.5 + (factln_i(n) - factln_i(x) - factln_i(n-x)).exp()).floor()
}


/// Log-Combination
/// - `n = 0,1,2,...`
/// - `x = 0,1,2,...`
/// - `n >= x`
pub fn combln(n: u64, x: u64) -> f64 {
    factln_i(n) - factln_i(x) - factln_i(n-x)
}


/// Computes beta function.
/// - `a > 0`
/// - `b > 0`
pub fn beta_fn(a: f64, b:f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a+b)).exp()
}


/// Computes log beta function.
/// - `a > 0`
/// - `b > 0`
pub fn beta_fn_ln(a: f64, b:f64) -> f64 {
    gamma_ln(a) + gamma_ln(b) - gamma_ln(a+b)
}


/// Incomplete gamma function, series representation
pub fn gser(x: f64, a: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    // let fpmin = 1.0e-30;
    let gln = gamma_ln(a);

    if x <= 0.0 {
        if x < 0.0 {
            println!("x less than 0 in routine gser");
        }
        return 0.0;
    }
    else {
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = 1.0 / a;

        for _ in 1..imax {
            ap = ap + 1.0;
            del = del * x / ap;
            sum = sum + del;
            if del.abs() < sum.abs() * eps {
                return sum * (-x + a * x.ln() - gln).exp();
            }
        }
        println!("a too large, ITMAX too small in routine gser");
    }
    0.0
}


/// Incomplete gamma function, continued fraction representation
pub fn gcf(x: f64, a: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    let fpmin = 1.0e-30;

    let gln = gamma_ln(a);
    let mut b= x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d= 1.0 / b;
    let mut h= d;

    let mut an: f64;
    let mut del: f64;

    for i in 1..=imax {
        an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (del-1.0).abs() < eps {
            break;
        }
    }
    // if (i > imax) {
    //     println!("a too large, ITMAX too small in gcf");
    // }

    (-x + a * x.ln() - gln).exp() * h
}

pub fn gcf_ln(x: f64, a: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    let fpmin = 1.0e-30;

    let gln = gamma_ln(a);
    let mut b= x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d= 1.0 / b;
    let mut h= d;

    let mut an: f64;
    let mut del: f64;

    for i in 1..=imax {
        an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (del-1.0).abs() < eps {
            break;
        }
    }
    // if (i > imax) {
    //     println!("a too large, ITMAX too small in gcf");
    // }

    (-x + a * x.ln() - gln) + h.ln()
}


/// Incomplete gamma function `P(x,a)`
// pub fn gammp(x: f64, a: f64) -> f64 {
//     if x < 0.0 || a <= 0.0 {
//         println ! ("Invalid arguments in routine gammp");
//     }
//     if x < (a + 1.0) {
//         gser(x, a)
//     } else {
//         1.0 - gcf(x, a)
//     }
// }
pub fn gammp(x: f64, a: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        println!("Invalid arguments in routine gammp");
        return f64::NAN;
    }
    if x == 0.0 {
        0.0
    } else if a >= 100.0 {
        return 1.0 - gammapprox(x, a); // <-- The crucial correction!
    } else if x < (a + 1.0) {
        return gser(x, a);
    } else {
        return 1.0 - gcf(x, a);
    }
}


/// Complementary incomplete gamma function `Q(x,a)`
pub fn gammq(x: f64, a: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        println!("Invalid arguments in routine gammq");
    }
    if x == 0.0 {
        1.0
    } else if a >= 100.0 {
        gammapprox(x, a)
    } else if x < (a + 1.0) {
        1.0 - gser(x, a)
    } else {
        gcf(x, a)
    }
}


/// Incomplete gamma function, quadrature `P(x,a)`
pub fn gammapprox(x: f64, a: f64) -> f64 {

    const Y_VEC: [f64; 18] = [
        0.0021695375159141994, 0.011413521097787704,0.027972308950302116,
        0.051727015600492421, 0.082502225484340941, 0.12007019910960293,
        0.16415283300752470, 0.21442376986779355, 0.27051082840644336,
        0.33199876341447887, 0.39843234186401943, 0.46931971407375483,
        0.54413605556657973, 0.62232745288031077, 0.70331500465597174,
        0.78649910768313447, 0.87126389619061517, 0.95698180152629142];
    const W_VEC: [f64; 18] = [
        0.0055657196642445571, 0.012915947284065419, 0.020181515297735382,
        0.027298621498568734, 0.034213810770299537,0.040875750923643261,
        0.047235083490265582, 0.053244713977759692,0.058860144245324798,
        0.064039797355015485, 0.068745323835736408,0.072941885005653087,
        0.076598410645870640, 0.079687828912071670,0.082187266704339706,
        0.084078218979661945, 0.085346685739338721,0.085983275670394821];

    let (xu,mut t,mut sum,ans,a1,lna1,sqrta1): (f64,f64,f64,f64,f64,f64,f64);
    a1 = a-1.0;
    lna1 = a1.ln();
    sqrta1 = a1.sqrt();
    if x > a1 {
        xu = f64::max(a1 + 11.5*sqrta1, x + 6.0*sqrta1);
    }
    else {
        xu = f64::max(0.0,f64::min(a1 - 7.5*sqrta1, x - 5.0*sqrta1));
    }
    sum = 0.0;
    for j in 0..18 { //Gauss-Legendre.
        t = x + (xu-x)*Y_VEC[j];
        sum += W_VEC[j]*(-(t-a1)+a1*(t.ln()-lna1)).exp();
    }
    ans = sum*(xu-x)*(a1*(a1.ln()-1.0)-gamma_ln(a)).exp();

    if ans >= 0.0 {
        ans
    } else {
        1.0 + ans
    }
}



/// Error function (gammp implementation) `Erf(x)`
pub fn erf2(x: f64) -> f64 {
    if x < 0.0 {
        -gammp(x * x, 0.5)
    } else {
        gammp(x * x, 0.5)
    }
}



/// Inverse of incomplete beta function
/// - `0 < p < 1`
/// - `alpha > 0`
/// - `beta > 0`
pub fn betai_inv(p: f64, a:f64, b: f64) -> f64 {
    let eps = 1.0e-8;
    let (pp,mut t,mut u,mut err,mut x,al,h,w,afac,a1,b1,lna,lnb):
        (f64,f64,f64,f64,f64,f64,f64,f64,f64,f64,f64,f64,f64);
    a1=a-1.0;
    b1=b-1.0;

    if p <= 0.0 {
        return 0.;
    }
    else if p >= 1.0 {
        return 1.0;
    }
    else if a >= 1.0 && b >= 1.0 {
        if p < 0.5 {
            pp = p
        }
        else {
            pp = 1.0 - p;
        }
        t = (-2.0*pp.ln()).sqrt();
        x = (2.30753 + t*0.27061) / (1.0 + t*(0.99229 + t*0.04481)) - t;
        if p < 0.5 {
            x = -x;
        }
        al = (x.powi(2)-3.0)/6.0;
        h = 2.0 / (1.0/(2.0*a-1.0)+1.0/(2.0*b-1.0));
        w = (x*(al+h).sqrt()/h)-(1.0/(2.0*b-1.0)-1.0/(2.0*a-1.0))*(al+5.0/6.0-2.0/(3.0*h));
        x = a/(a+b*(2.0*w).exp());
    }
    else {
        lna = (a/(a+b)).ln();
        lnb = (b/(a+b)).ln();
        t = (a*lna).exp()/a;
        u = (b*lnb).exp()/b;
        w = t + u;
        if p < t/w  {
            x = (a*w*p).powf(1.0/a);
        }
        else {
            x = 1.0 - (b*w*(1.-p)).powf(1.0/b);
        }
    }

    afac = -gamma_ln(a)-gamma_ln(b)+gamma_ln(a+b);
    for j in 0..10 {
        if x == 0.0 || x == 1.0 {
            return x;
        }
        err = betai(x, a, b) - p;
        t = (a1 * x.ln() + b1 * (1.0 - x).ln() + afac).exp();
        u = err / t;
        t = u / (1.0 - 0.5 * f64::min(1.0, u * (a1 / x - b1 / (1.0 - x))));
        x -= t;
        // x -= (t = u / (1. - 0.5 * MIN(1., u * (a1 / x - b1 / (1. - x)))));
        if x <= 0.0 {
            x = 0.5 * (x + t);
        }
        if x >= 1.0 {
            x = 0.5 * (x + t + 1.0);
        }
        if t.abs() < eps*x && j>0 {
            break;
        }
    }

    x
}


/// Inverse of incomplete gamma function
/// - `0 < p < 1`
/// - `alpha > 0`
pub fn gammai_inv(p: f64, alpha:f64) -> f64 {
    let (mut x,mut err,mut t,mut u,pp,lna1,afac,a1): (f64,f64,f64,f64,f64,f64,f64,f64);
    a1 = alpha-1.0;

    let eps = 1.0e-8; // Accuracy is the square of eps.
    let gln = gamma_ln(alpha);

    if alpha <= 0.0 {
        println!("a must be pos in invgammap");
    }
    if p >= 1.0 {
        return f64::max(100.0,alpha + 100.0*alpha.sqrt());
    }
    if p <= 0.0 {
        return 0.0;
    }
    lna1 = a1.ln();
    afac = (a1*(lna1-1.0)-gln).exp();
    if alpha > 1.0 {
        if p < 0.5 {
            pp = p;
        }
        else {
            pp = 1.0 - p;
        }
        t = (-2.0*pp.ln()).sqrt();
        x = (2.30753+t*0.27061) /
            (1.0+t*(0.99229+t*0.04481)) - t;
        if p < 0.5 {
            x = -x;
        }
        // x = f64::max(1.0e-3,alpha*(1.-1./(9.*alpha)-x/(3.*alpha.sqrt()).powi(3)));
        // The correct Wilson-Hilferty calculation:
        x = f64::max(1.0e-3, alpha * (1.0 - 1.0 / (9.0 * alpha) - x / (3.0 * alpha.sqrt())).powi(3));    }
    else {
        t = 1.0 - alpha*(0.253+alpha*0.12);
        if p < t {
            x = (p/t).powf(1.0/alpha);
        }
        else {
            x = 1.0-(1.0-(p-t)/(1.0-t)).ln();
        }
    }

    for _ in 0..12 {
        if x <= 0.0 {
            return 0.0;
        }
        err = gammp(x,alpha) - p;
        if alpha > 1.0 {
            t = afac*(-(x-a1)+a1*(x.ln()-lna1)).exp();
        }
        else {
            t = (-x+a1*x.ln()-gln).exp();
        }
        u = err/t;
        t = u/(1.0-0.5*f64::min(1.0,u*((alpha-1.0)/x - 1.0)));
        x -= t; //Halley’s method.
        if x <= 0.0 {
            x = 0.5*(x + t);
        } //Halve old value if x tries to go negative.
        if t.abs() < eps *x {
            break;
        }
    }

    x
}


/// Chebyshev coefficients
fn erfc_cheb(z: f64) -> f64 {

    const NCOEF: usize = 28;
    const COEF: [f64; 28] = [
        -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
        -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5,
        -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
        6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
        9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13,
        3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15,
        -1.523e-15, -9.4e-17, 1.21e-16,-2.8e-17 ];

    let (mut d, mut dd, t, ty, mut tmp) : (f64, f64, f64, f64, f64);
    d = 0.0;
    dd = 0.0;

    assert!(z >= 0f64, "erfccheb requires nonnegative argument");
    t = 2.0 / (2.0 + z);
    ty = 4.0 * t - 2.0;
    for j in (1..NCOEF-1).rev() {
        tmp = d;
        d = ty * d - dd + COEF[j];
        dd = tmp;
    }
    t * (-z.powi(2) + 0.5 * (COEF[0] + ty * d) - dd).exp()
}


/// Error function (Chebyshev implementation)
pub fn erf(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 - erfc_cheb(x)
    } else {
        erfc_cheb(-x) - 1.0
    }
}


/// Complementary error function
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        erfc_cheb(x)
    } else {
        2.0 - erfc_cheb(-x)
    }
}


/// Inverse of complementary error function
pub fn erfc_inv(p: f64) -> f64 {

    let (pp, t, mut x): (f64, f64, f64);

    // Return arbitrary large pos or neg value
    if p >= 2.0 {
        return -100.0;
    }
    else if p <= 0.0 {
        return 100.0;
    }

    if p < 1.0 {
        pp = p
    }
    else {
        pp = 2.0 - p;
    }

    t = (-2.0 * (pp / 2.0).ln()).sqrt();
    x = -std::f64::consts::FRAC_1_SQRT_2 * ((2.30753 + t * 0.27061) /
        (1f64 + t * (0.99229 + t * 0.04481)) - t);
    for _ in 0..2 {
        let err = erfc(x) - pp;
        x += err / (std::f64::consts::FRAC_2_SQRT_PI * (-x.powi(2)).exp() - x * err);
    }
    if p < 1.0 {
        x
    } else {
        -x
    }
}


/// Inverse of error function
pub fn erf_inv(p: f64) -> f64 {
    erfc_inv(1.0 - p)
}



#[cfg(test)]
mod tests {
    use super::*;

    // A handy macro to assert that two floats are within a specific tolerance
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

    #[test]
    fn test_distribution_percentile_roundtrips() {
        let test_percentiles = vec![0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
        let tolerance = 1e-5;

        for &p in &test_percentiles {
            // Test Beta(0.5, 2.0)
            let x_beta = beta_per(p, 0.5, 2.0);
            let p_beta = beta_cdf(x_beta, 0.5, 2.0);
            assert_approx_eq!(p, p_beta, tolerance);

            // Test Normal(100.0, 16.0)
            let x_norm = normal_per(p, 100.0, 16.0);
            let p_norm = normal_cdf(x_norm, 100.0, 16.0);
            assert_approx_eq!(p, p_norm, tolerance);

            // Test Exponential(3.5)
            let x_exp = exp_per(p, 3.5);
            let p_exp = exp_cdf(x_exp, 3.5);
            assert_approx_eq!(p, p_exp, tolerance);
        }
    }

    #[test]
    fn test_chisq_large_parameter_regression() {
        // nu = 500 means alpha = 250, which easily clears our >= 100.0 safeguard
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
    fn test_normal_ran_statistical_properties() {
        let mu = 100.0;
        let sigma = 16.0;
        let n = 100_000; // 100,000 draws to ensure statistical stability

        // 1. Generate a massive array of random draws
        // (This will run incredibly fast now that you pre-allocate the Vec!)
        let draws = normal_ranvec(n, mu, sigma);

        // 2. Calculate the sample statistics
        let calc_mean = sample_mean(&draws);
        let calc_var = sample_var(&draws);

        // 3. Define the theoretical variance
        let theoretical_var = sigma.powi(2); // 16^2 = 256.0

        // 4. Assert that the sample stats match the theoretical parameters
        // Note: Because it's random, we use a slightly wider tolerance than the CDF tests.
        // With n = 100,000, the sample mean should easily be within 0.5 of mu,
        // and variance within 3.0 of theoretical_var.
        assert_approx_eq!(mu, calc_mean, 0.5);
        assert_approx_eq!(theoretical_var, calc_var, 3.0);
    }

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
    fn test_beta_correctness() {
        // Beta(1, 1) is mathematically identical to a Uniform(0, 1) distribution
        assert_approx_eq!(beta_pdf(0.5, 1.0, 1.0), 1.0, TOL);
        assert_approx_eq!(beta_cdf(0.5, 1.0, 1.0), 0.5, TOL);
        assert_approx_eq!(beta_per(0.5, 1.0, 1.0), 0.5, TOL);
    }

    #[test]
    fn test_normal_correctness() {
        // Standard Normal N(0, 1) at x = 0 should be 1 / sqrt(2 * pi)
        let one_over_sqrt_2pi = 0.3989422804014327;
        assert_approx_eq!(normal_pdf(0.0, 0.0, 1.0), one_over_sqrt_2pi, TOL);

        // CDF at the mean is exactly 50%
        assert_approx_eq!(normal_cdf(0.0, 0.0, 1.0), 0.5, TOL);
        assert_approx_eq!(normal_per(0.5, 0.0, 1.0), 0.0, TOL);

        // Empirical rule: ~68.27% of data falls within 1 standard deviation
        let cdf_1 = normal_cdf(1.0, 0.0, 1.0);
        let cdf_neg1 = normal_cdf(-1.0, 0.0, 1.0);
        assert_approx_eq!(cdf_1 - cdf_neg1, 0.682689492, TOL);
    }

    #[test]
    fn test_exponential_correctness() {
        // Exp(lambda = 2.0) at x = 1.0
        // pdf(x) = lambda * e^(-lambda * x)
        // cdf(x) = 1 - e^(-lambda * x)
        let lambda = 2.0;
        let x = 1.0;
        let expected_pdf = 2.0 * std::f64::consts::E.powf(-2.0);
        let expected_cdf = 1.0 - std::f64::consts::E.powf(-2.0);

        assert_approx_eq!(exp_pdf(x, lambda), expected_pdf, TOL);
        assert_approx_eq!(exp_cdf(x, lambda), expected_cdf, TOL);
    }

    #[test]
    fn test_uniform_correctness() {
        // Unif(1.5, 4.5)
        // Width = 3.0, so the constant PDF is 1/3
        assert_approx_eq!(unif_pdf(2.0, 1.5, 4.5), 0.333333333333, TOL);
        assert_approx_eq!(unif_cdf(3.0, 1.5, 4.5), 0.5, TOL);
        assert_approx_eq!(unif_per(0.5, 1.5, 4.5), 3.0, TOL);
    }

    #[test]
    fn test_binomial_correctness() {
        // Bin(n=10, p=0.5)
        // P(X=5) = (10 choose 5) * 0.5^10 = 252 / 1024 = 0.24609375
        assert_approx_eq!(bin_pmf(5, 10, 0.5), 0.24609375, TOL);

        // Bin(n=100, p=0.9)
        // P(X=100) requires all successes, which is exactly 0.9^100
        assert_approx_eq!(bin_pmf(100, 100, 0.9), 0.9_f64.powi(100), TOL);
        assert_approx_eq!(bin_cdf(100, 100, 0.9), 1.0, TOL);
    }

    #[test]
    fn test_poisson_correctness() {
        // Pois(lambda = 2.0)
        // P(X=3) = e^(-2) * 2^3 / 3! = 0.135335 * 8 / 6
        assert_approx_eq!(pois_pmf(3, 2.0), 0.180447044315, TOL);

        // P(X=0) = e^(-lambda)
        assert_approx_eq!(pois_pmf(0, 2.0), std::f64::consts::E.powf(-2.0), TOL);
    }

    #[test]
    fn test_geometric_correctness() {
        // Geo(p = 0.05) - Number of failures before first success
        // P(X=0 failures) = you succeeded on the first try = p
        assert_approx_eq!(geo_pmf(0, 0.05), 0.05, TOL);

        // P(X=1 failure) = (1-p) * p = 0.95 * 0.05
        assert_approx_eq!(geo_pmf(1, 0.05), 0.0475, TOL);

        // Geo2(p = 0.05) - Trial number of first success
        // P(X=1st trial) = p
        assert_approx_eq!(geo2_pmf(1, 0.05), 0.05, TOL);
        assert_approx_eq!(geo2_pmf(2, 0.05), 0.0475, TOL);
    }

    #[test]
    fn test_chisq_correctness() {
        // ChiSq(nu = 2.0) is mathematically identical to an Exponential(lambda = 0.5)
        let x = 2.0;
        let expected_pdf = 0.5 * std::f64::consts::E.powf(-1.0);
        assert_approx_eq!(chisq_pdf(x, 2.0), expected_pdf, TOL);
    }

    #[test]
    fn test_hypergeometric_correctness() {
        // HG(n=20, N=100, M=50)
        // Drawing 20 items from a population of 100 where 50 are successes.
        // P(X=20) = exactly matching all 20 draws to the 50 successes.
        // (50 choose 20) / (100 choose 20)
        let expected_prob = 0.000000017402868; // Calculated via standard statistical tables
        assert_approx_eq!(hg_pmf(20, 20, 100, 50), expected_prob, TOL);
    }

    #[test]
    fn test_gamma_correctness() {
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
    fn test_factorial_correctness() {
        // fact_i computes standard factorials
        assert_eq!(fact_i(1), 1);
        assert_eq!(fact_i(5), 120);
        assert_eq!(fact_i(10), 3_628_800);

        // fact_f generalizes factorial: x! = x * Gamma(x)
        // 5! = 120
        assert_approx_eq!(fact_f(5.0), 120.0, TOL);
        // 0.5! = 0.5 * Gamma(0.5) = 0.5 * sqrt(pi)
        let half_fact = 0.5 * PI.sqrt();
        assert_approx_eq!(fact_f(0.5), half_fact, TOL);
    }

    #[test]
    fn test_combinations_correctness() {
        // 10 choose 5 = 252
        assert_approx_eq!(comb(10, 5), 252.0, TOL);
        // 50 choose 2 = (50 * 49) / 2 = 1225
        assert_approx_eq!(comb(50, 2), 1225.0, TOL);
        // n choose 0 = 1, n choose n = 1
        assert_approx_eq!(comb(20, 0), 1.0, TOL);
        assert_approx_eq!(comb(20, 20), 1.0, TOL);
    }

    #[test]
    fn test_beta_function_correctness() {
        // B(1, 1) = 1
        assert_approx_eq!(beta_fn(1.0, 1.0), 1.0, TOL);

        // B(2, 2) = Gamma(2)*Gamma(2) / Gamma(4) = (1 * 1) / 6 = 1/6
        assert_approx_eq!(beta_fn(2.0, 2.0), 1.0 / 6.0, TOL);

        // B(a, b) should equal B(b, a) (Symmetry)
        assert_approx_eq!(beta_fn(3.5, 7.2), beta_fn(7.2, 3.5), TOL);
    }

    #[test]
    fn test_incomplete_beta_correctness() {
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
    fn test_incomplete_gamma_correctness() {
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
    fn test_error_function_correctness() {
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
    fn test_error_function_inverses() {
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

    #[test]
    pub fn it_works() {
        // let result = add(2, 2);
        // assert_eq!(result, 4);
        println!("Testing each distribution function...");

        println!("\nBeta");
        println!("pdf: {}", beta_pdf(0.7, 0.5, 1.5));
        println!("cdf: {}", beta_cdf(0.7, 0.5, 1.5));
        println!("per: {}", beta_per(0.1, 0.5, 1.5));
        println!("ran: {}", beta_ran(0.5, 1.5));
        println!("ran_vec: {:?}", beta_ranvec(3, 0.5, 1.5));
        let mut mybeta = BetaDist { alpha: 0.5, beta: 1.5 };
        println!("pdf: {}", mybeta.pdf(0.7));
        println!("cdf: {}", mybeta.cdf(0.7));
        println!("per: {}", mybeta.per(0.1));
        println!("ran: {}", mybeta.ran());
        println!("ranvec: {:?}", mybeta.ranvec(5));
        println!("mean: {}", mybeta.mean());
        println!("var: {}", mybeta.var());
        println!("sd: {}", mybeta.sd());
        println!("---------------------");

        println!("\nChi-Square");
        println!("pdf: {}", chisq_pdf(0.7, 1.5));
        println!("cdf: {}", chisq_cdf(0.1, 5000.0));
        println!("per: {}", chisq_per(0.1, 5000.0));
        println!("ran: {}", chisq_ran(1.5));
        println!("ran_vec: {:?}", chisq_ranvec(3, 1.5));
        let mut mychisq = ChiSqDist { nu: 1.5 };
        println!("pdf: {}", mychisq.pdf(0.7));
        println!("cdf: {}", mychisq.cdf(0.7));
        println!("per: {}", mychisq.per(0.1));
        println!("ran: {}", mychisq.ran());
        println!("ranvec: {:?}", mychisq.ranvec(5));
        println!("mean: {}", mychisq.mean());
        println!("var: {}", mychisq.var());
        println!("sd: {}", mychisq.sd());
        println!("---------------------");

        println!("\nExp");
        println!("pdf: {}", exp_pdf(0.7, 1.5));
        println!("cdf: {}", exp_cdf(0.7, 1.5));
        println!("per: {}", exp_per(0.1, 1.5));
        println!("ran: {}", exp_ran(1.5));
        println!("ran_vec: {:?}", exp_ranvec(3, 1.5));
        let mut myexp = ExpDist { lambda: 1.5 };
        println!("pdf: {}", myexp.pdf(0.7));
        println!("cdf: {}", myexp.cdf(0.7));
        println!("per: {}", myexp.per(0.1));
        println!("ran: {}", myexp.ran());
        println!("ranvec: {:?}", myexp.ranvec(5));
        println!("mean: {}", myexp.mean());
        println!("var: {}", myexp.var());
        println!("sd: {}", myexp.sd());
        println!("---------------------");

        println!("\nF Dist");
        println!("pdf: {}", f_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", f_cdf(0.7, 1.5, 4.5));
        println!("per: {}", f_per(0.1, 1.5, 4.5));
        println!("ran: {}", f_ran(1.5, 4.5));
        println!("ran_vec: {:?}", f_ranvec(3, 1.5, 4.5));
        let mut mydist = FDist { nu1: 1.5, nu2: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.1));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nGamma Dist");
        println!("pdf: {}", gamma_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", gamma_cdf(0.7, 1.5, 4.5));
        println!("per: {}", gamma_per(0.2, 500.0, 0.5));
        println!("ran: {}", gamma_ran(1.5, 4.5));
        println!("ran_vec: {:?}", gamma_ranvec(3, 1.5, 4.5));
        let mut mydist = GammaDist { alpha: 1.5, beta: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nLog-Normal");
        println!("pdf: {}", lognormal_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", lognormal_cdf(0.7, 1.5, 4.5));
        println!("per: {}", lognormal_per(0.1, 1.5, 4.5));
        println!("ran: {}", lognormal_ran(1.5, 4.5));
        println!("ran_vec: {:?}", lognormal_ranvec(3, 1.5, 5.5));
        let mut mydist = LogNormalDist { mu: 1.5, sigma: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nNormal");
        println!("pdf: {}", normal_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", normal_cdf(0.7, 1.5, 4.5));
        println!("per: {}", normal_per(0.1, 1.5, 4.5));
        println!("ran: {}", normal_ran(1.5, 4.5));
        println!("ran_vec: {:?}", normal_ranvec(3, 1.5, 4.5));
        let mut mydist = NormalDist { mu: 1.5, sigma: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nt Dist");
        println!("pdf: {}", t_pdf(0.7, 4.5));
        println!("cdf: {}", t_cdf(0.7, 4.5));
        println!("per: {}", t_per(0.1, 4.5));
        println!("ran: {}", t_ran(4.5));
        println!("ran_vec: {:?}", t_ranvec(3, 4.5));
        let mut mydist = TDist { nu: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nUnif");
        println!("pdf: {}", unif_pdf(1.7, 1.5, 4.5));
        println!("cdf: {}", unif_cdf(1.7, 1.5, 4.5));
        println!("per: {}", unif_per(0.1, 1.5, 4.5));
        println!("ran: {}", unif_ran(1.5, 4.5));
        println!("ran_vec: {:?}", unif_ranvec(3, 1.5, 4.5));
        let mut mydist = UnifDist { a: 1.5, b: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nBinomial");
        println!("bin_pmf: {}", bin_pmf(99, 100, 0.9));
        println!("bin_cdf: {}", bin_cdf(99, 100, 0.9));
        println!("bin_per: {}", bin_per(1.0, 100, 0.9));
        println!("bin_ran: {}", bin_ran(100, 0.9));
        println!("ran_vec: {:?}", bin_ranvec(3, 100, 0.9));
        let mut mybin = BinDist { n: 100, p: 0.9 };
        println!("bin pmf: {}", mybin.pmf(80));
        println!("bin cdf: {}", mybin.cdf(80));
        println!("bin per: {}", mybin.per(0.99));
        println!("bin ran: {}", mybin.ran());
        println!("ranvec: {:?}", mybin.ranvec(5));
        println!("bin mean: {}", mybin.mean());
        println!("bin var: {}", mybin.var());
        println!("bin sd: {}", mybin.sd());

        println!("\nGeo");
        println!("geo_pmf: {}", geo_pmf(0, 0.05));
        println!("geo_cdf: {}", geo_cdf(0, 0.05));
        println!("geo_per: {}", geo_per(1.0, 0.05));
        println!("geo_ran: {}", geo_ran(0.05));
        println!("ran_vec: {:?}", geo_ranvec(3, 0.05));
        let mut mygeo = GeoDist { p: 0.05 };
        println!("geo pmf: {}", mygeo.pmf(2));
        println!("geo cdf: {}", mygeo.cdf(2));
        println!("geo per: {}", mygeo.per(0.99));
        println!("geo ran: {}", mygeo.ran());
        println!("ranvec: {:?}", mygeo.ranvec(5));
        println!("geo mean: {}", mygeo.mean());
        println!("geo var: {}", mygeo.var());
        println!("geo sd: {}", mygeo.sd());


        println!("\nGeo2");
        println!("geo2_pmf: {}", geo2_pmf(1, 0.05));
        println!("geo2_cdf: {}", geo2_cdf(1, 0.05));
        println!("geo2_per: {}", geo2_per(0.99, 0.05));
        println!("geo2_ran: {}", geo2_ran(0.05));
        println!("ran_vec: {:?}", geo2_ranvec(3, 0.05));
        let mut mygeo2 = Geo2Dist { p: 0.05 };
        println!("geo2 pmf: {}", mygeo2.pmf(2));
        println!("geo2 cdf: {}", mygeo2.cdf(2));
        println!("geo2 per: {}", mygeo2.per(0.99));
        println!("geo2 ran: {}", mygeo2.ran());
        println!("ranvec: {:?}", mygeo2.ranvec(5));
        println!("geo2 mean: {}", mygeo2.mean());
        println!("geo2 var: {}", mygeo2.var());
        println!("geo2 sd: {}", mygeo2.sd());


        println!("\nHG");
        println!("hg_pmf: {}", hg_pmf(80, 100, 1000, 900));
        println!("hg_cdf: {}", hg_cdf(80, 100, 1000, 900));
        println!("hg_per: {}", hg_per(0.26, 100, 1000, 900));
        println!("hg_ran: {}", hg_ran(100, 1000, 900));
        println!("ran_vec: {:?}", hg_ranvec(3, 100, 1000, 900));
        let mut myhg = HGDist { n: 100, N: 1000, M: 900 };
        println!("hg pmf: {}", myhg.pmf(80));
        println!("hg cdf: {}", myhg.cdf(80));
        println!("hg per: {}", myhg.per(0.26));
        println!("hg ran: {}", myhg.ran());
        println!("ranvec: {:?}", myhg.ranvec(5));
        println!("hg mean: {}", myhg.mean());
        println!("hg var: {}", myhg.var());
        println!("hg sd: {}", myhg.sd());


        println!("\nNB");
        println!("NB_pmf: {}", nb_pmf(10, 5, 0.1));
        println!("NB_cdf: {}", nb_cdf(10, 5, 0.1));
        println!("NB_per: {}", nb_per(0.0, 5, 0.1));
        println!("NB_ran: {}", nb_ran(5, 0.1));
        println!("ran_vec: {:?}", nb_ranvec(3, 5, 0.1));
        let mut mynb = NBDist { r: 5, p: 0.1 };
        println!("NB pmf: {}", mynb.pmf(2));
        println!("NB cdf: {}", mynb.cdf(2));
        println!("NB per: {}", mynb.per(0.99));
        println!("NB ran: {}", mynb.ran());
        println!("ranvec: {:?}", mynb.ranvec(5));
        println!("NB mean: {}", mynb.mean());
        println!("NB var: {}", mynb.var());
        println!("NB sd: {}", mynb.sd());


        println!("\nNB2");
        println!("NB2_pmf: {}", nb2_pmf(10, 5, 0.1));
        println!("NB2_cdf: {}", nb2_cdf(10, 5, 0.1));
        println!("NB2_per: {}", nb2_per(1.0, 5, 0.1));
        println!("NB2_ran: {}", nb2_ran(5, 0.1));
        println!("ran_vec: {:?}", nb2_ranvec(3, 5, 0.1));
        let mut mynb2 = NB2Dist { r: 5, p: 0.1 };
        println!("NB2 pmf: {}", mynb2.pmf(2));
        println!("NB2 cdf: {}", mynb2.cdf(2));
        println!("NB2 per: {}", mynb2.per(0.99));
        println!("NB2 ran: {}", mynb2.ran());
        println!("ranvec: {:?}", mynb2.ranvec(5));
        println!("NB2 mean: {}", mynb2.mean());
        println!("NB2 var: {}", mynb2.var());
        println!("NB2 sd: {}", mynb2.sd());


        println!("\nPois");
        println!("pois_pmf: {}", pois_pmf(3, 2.0));
        println!("pois_cdf: {}", pois_cdf(3, 2.0));
        println!("pois_per: {}", pois_per(1.0, 2.0));
        println!("pois_ran: {}", pois_ran(2.0));
        println!("ran_vec: {:?}", pois_ranvec(3, 2.0));
        let mut mypois = PoisDist { lambda: 2.0 };
        println!("pois pmf: {}", mypois.pmf(3));
        println!("pois cdf: {}", mypois.cdf(3));
        println!("pois per: {}", mypois.per(0.26));
        println!("pois ran: {}", mypois.ran());
        println!("ranvec: {:?}", mypois.ranvec(5));
        println!("pois mean: {}", mypois.mean());
        println!("pois var: {}", mypois.var());
        println!("pois sd: {}", mypois.sd());

        use crate::erf;
        println!("\nerf: {}", erf(1.0));
        use crate::erf2;
        println!("erf2: {}", erf2(1.0));
    }
}



