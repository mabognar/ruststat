// ==========================================
// ==========================================
// Log-Logistic Distribution
// ==========================================
// ==========================================

use std::f64::consts::PI;
use rand::random;

/// Struct for the Log-Logistic distribution `X ~ LogLogistic(alpha, beta)`.
/// # Parameters
/// - Scale: `alpha > 0`
/// - Shape: `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ LogLogistic(alpha=1.0, beta=2.0)`. Use
/// ```
/// use ruststat::LogLogisticDist;
/// let mut myll = LogLogisticDist{alpha:1.0, beta:2.0};
/// println!("Probability density function f(1.5): {}", myll.pdf(1.5));
/// println!("Cumulative distribution function P(X<=1.5): {}", myll.cdf(1.5));
/// println!("99th percentile: {}", myll.per(0.99));
/// println!("Random draw: {}", myll.ran());
/// println!("Random vector: {:?}", myll.ranvec(5));
/// println!("Mean: {}", myll.mean());
/// println!("Variance: {}", myll.var());
/// println!("Standard deviation: {}", myll.sd());
/// ```
pub struct LogLogisticDist {
    pub alpha: f64,
    pub beta: f64,
}

impl LogLogisticDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        loglogistic_pdf(x, self.alpha, self.beta)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        loglogistic_cdf(x, self.alpha, self.beta)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        loglogistic_per(p, self.alpha, self.beta)
    }
    pub fn ran(&mut self) -> f64 {
        loglogistic_ran(self.alpha, self.beta)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        loglogistic_ranvec(n, self.alpha, self.beta)
    }
    pub fn mean(&mut self) -> f64 {
        if self.beta > 1.0 {
            self.alpha * PI / self.beta / (PI / self.beta).sin()
        } else {
            f64::NAN
        }
    }
    pub fn var(&mut self) -> f64 {
        if self.beta > 2.0 {
            let b = PI / self.beta;
            self.alpha.powi(2) * (2.0 * b / (2.0 * b).sin() - (b / b.sin()).powi(2))
        } else {
            f64::NAN
        }
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ LogLogistic(alpha, beta)`.
pub fn loglogistic_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function loglogistic_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (beta / alpha) * (x / alpha).powf(beta - 1.0) / (1.0 + (x / alpha).powf(beta)).powi(2)
}

/// Computes cumulative distribution function (cdf) for `X ~ LogLogistic(alpha, beta)`.
pub fn loglogistic_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function loglogistic_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 + (x / alpha).powf(-beta))
}

/// Computes a percentile for `X ~ LogLogistic(alpha, beta)`.
pub fn loglogistic_per(p: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function loglogistic_per");
        return f64::NAN;
    }
    alpha * (p / (1.0 - p)).powf(1.0 / beta)
}

/// Random draw from `X ~ LogLogistic(alpha, beta)` distribution.
pub fn loglogistic_ran(alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("Bad argument to loglogistic_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    alpha * (u / (1.0 - u)).powf(1.0 / beta)
}

/// Save random draws from `X ~ LogLogistic(alpha, beta)` distribution into a `Vec`
pub fn loglogistic_ranvec(n: u64, alpha: f64, beta: f64) -> Vec<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("Bad argument to loglogistic_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        xvec.push(alpha * (u / (1.0 - u)).powf(1.0 / beta));
    }
    xvec
}