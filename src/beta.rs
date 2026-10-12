// ==========================================
// ==========================================
// Beta Distribution
// ==========================================
// ==========================================

use crate::{beta_fn_ln, betai, betai_inv, gamma_ran};

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
