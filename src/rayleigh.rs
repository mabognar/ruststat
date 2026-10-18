// ==========================================
// ==========================================
// Rayleigh Distribution
// ==========================================
// ==========================================

use std::f64::consts::PI;
use rand::random;

/// Struct for the Rayleigh distribution `X ~ Rayleigh(sigma)`.
/// # Parameters
/// - Scale: `sigma > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Rayleigh(sigma=2.0)`. Use
/// ```
/// use ruststat::RayleighDist;
/// let mut myray = RayleighDist{sigma:2.0};
/// println!("Probability density function f(1.5): {}", myray.pdf(1.5));
/// println!("Cumulative distribution function P(X<=1.5): {}", myray.cdf(1.5));
/// println!("99th percentile: {}", myray.per(0.99));
/// println!("Random draw: {}", myray.ran());
/// println!("Random vector: {:?}", myray.ranvec(5));
/// println!("Mean: {}", myray.mean());
/// println!("Variance: {}", myray.var());
/// println!("Standard deviation: {}", myray.sd());
/// ```
pub struct RayleighDist {
    pub sigma: f64,
}

impl RayleighDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        rayleigh_pdf(x, self.sigma)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        rayleigh_cdf(x, self.sigma)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        rayleigh_per(p, self.sigma)
    }
    pub fn ran(&mut self) -> f64 {
        rayleigh_ran(self.sigma)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        rayleigh_ranvec(n, self.sigma)
    }
    pub fn mean(&mut self) -> f64 {
        self.sigma * (PI / 2.0).sqrt()
    }
    pub fn var(&mut self) -> f64 {
        (4.0 - PI) / 2.0 * self.sigma.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ Rayleigh(sigma)`.
pub fn rayleigh_pdf(x: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function rayleigh_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (x / sigma.powi(2)) * (-(x.powi(2)) / (2.0 * sigma.powi(2))).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ Rayleigh(sigma)`.
pub fn rayleigh_cdf(x: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function rayleigh_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    1.0 - (-(x.powi(2)) / (2.0 * sigma.powi(2))).exp()
}

/// Computes a percentile for `X ~ Rayleigh(sigma)`.
pub fn rayleigh_per(p: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function rayleigh_per");
        return f64::NAN;
    }
    sigma * (-2.0 * (1.0 - p).ln()).sqrt()
}

/// Random draw from `X ~ Rayleigh(sigma)` distribution.
pub fn rayleigh_ran(sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("Bad argument to rayleigh_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    sigma * (-2.0 * (1.0 - u).ln()).sqrt()
}

/// Save random draws from `X ~ Rayleigh(sigma)` distribution into a `Vec`
pub fn rayleigh_ranvec(n: u64, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        println!("Bad argument to rayleigh_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        xvec.push(sigma * (-2.0 * (1.0 - u).ln()).sqrt());
    }
    xvec
}