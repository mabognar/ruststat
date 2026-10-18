// ==========================================
// ==========================================
// Weibull Distribution
// ==========================================
// ==========================================

use rand::random;
use crate::gamma;

/// Struct for the Weibull distribution `X ~ Weibull(k, lambda)`.
/// # Parameters
/// - Shape: `k > 0`
/// - Scale: `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Weibull(k=2.0, lambda=1.5)`. Use
/// ```
/// use ruststat::WeibullDist;
/// let mut myweib = WeibullDist{k:2.0, lambda:1.5};
/// println!("Probability density function f(1.2): {}", myweib.pdf(1.2));
/// println!("Cumulative distribution function P(X<=1.2): {}", myweib.cdf(1.2));
/// println!("99th percentile: {}", myweib.per(0.99));
/// println!("Random draw: {}", myweib.ran());
/// println!("Random vector: {:?}", myweib.ranvec(5));
/// println!("Mean: {}", myweib.mean());
/// println!("Variance: {}", myweib.var());
/// println!("Standard deviation: {}", myweib.sd());
/// ```
pub struct WeibullDist {
    pub k: f64,
    pub lambda: f64,
}

impl WeibullDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        weibull_pdf(x, self.k, self.lambda)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        weibull_cdf(x, self.k, self.lambda)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        weibull_per(p, self.k, self.lambda)
    }
    pub fn ran(&mut self) -> f64 {
        weibull_ran(self.k, self.lambda)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        weibull_ranvec(n, self.k, self.lambda)
    }
    pub fn mean(&mut self) -> f64 {
        self.lambda * gamma(1.0 + 1.0 / self.k)
    }
    pub fn var(&mut self) -> f64 {
        self.lambda.powi(2) * (gamma(1.0 + 2.0 / self.k) - gamma(1.0 + 1.0 / self.k).powi(2))
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ Weibull(k, lambda)`.
/// # Parameters
/// - Shape: `k > 0`
/// - Scale: `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Weibull(k=2.0, lambda=1.5)`. Use.
/// ```
/// use ruststat::weibull_pdf;
/// println!("Probability density function f(1.2): {}", weibull_pdf(1.2, 2.0, 1.5));
/// ```
pub fn weibull_pdf(x: f64, k: f64, lambda: f64) -> f64 {
    if k <= 0.0 || lambda <= 0.0 {
        println!("NAN produced. Error in function weibull_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (k / lambda) * (x / lambda).powf(k - 1.0) * (-(x / lambda).powf(k)).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ Weibull(k, lambda)`.
/// # Parameters
/// - Shape: `k > 0`
/// - Scale: `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Weibull(k=2.0, lambda=1.5)`. Use.
/// ```
/// use ruststat::weibull_cdf;
/// println!("P(X<=1.2): {}", weibull_cdf(1.2, 2.0, 1.5));
/// ```
pub fn weibull_cdf(x: f64, k: f64, lambda: f64) -> f64 {
    if k <= 0.0 || lambda <= 0.0 {
        println!("NAN produced. Error in function weibull_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    1.0 - (-(x / lambda).powf(k)).exp()
}

/// Computes a percentile for `X ~ Weibull(k, lambda)`.
/// # Note
/// Determines the value of `x` such that `P(X <= x) = p`.
/// # Parameters
/// - Shape: `k > 0`
/// - Scale: `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Weibull(k=2.0, lambda=1.5)`. To find the 80th percentile, use `p=0.80` and
/// ```
/// use ruststat::weibull_per;
/// println!("Percentile: {}", weibull_per(0.8, 2.0, 1.5));
/// ```
pub fn weibull_per(p: f64, k: f64, lambda: f64) -> f64 {
    if k <= 0.0 || lambda <= 0.0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function weibull_per");
        return f64::NAN;
    }
    lambda * (-(1.0 - p).ln()).powf(1.0 / k)
}

/// Random draw from `X ~ Weibull(k, lambda)` distribution.
/// # Parameters
/// - Shape: `k > 0`
/// - Scale: `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Weibull(k=2.0, lambda=1.5)`. Use
/// ```
/// use ruststat::weibull_ran;
/// println!("Random draw: {}", weibull_ran(2.0, 1.5));
/// ```
pub fn weibull_ran(k: f64, lambda: f64) -> f64 {
    if k <= 0.0 || lambda <= 0.0 {
        println!("Bad argument to weibull_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    lambda * (-(1.0 - u).ln()).powf(1.0 / k)
}

/// Save random draws from `X ~ Weibull(k, lambda)` distribution into a `Vec`
/// # Parameters
/// - Shape: `k > 0`
/// - Scale: `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Weibull(k=2.0, lambda=1.5)`. Use
/// ```
/// use ruststat::weibull_ranvec;
/// println!("Random Vec: {:?}", weibull_ranvec(10, 2.0, 1.5));
/// ```
pub fn weibull_ranvec(n: u64, k: f64, lambda: f64) -> Vec<f64> {
    if k <= 0.0 || lambda <= 0.0 {
        println!("Bad argument to weibull_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        xvec.push(lambda * (-(1.0 - u).ln()).powf(1.0 / k));
    }
    xvec
}
