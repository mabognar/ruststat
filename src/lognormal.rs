// ==========================================
// ==========================================
// LogNormal Distribution
// ==========================================
// ==========================================

use std::f64::consts::PI;
use crate::{erf, erf_inv, normal_ran};

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
