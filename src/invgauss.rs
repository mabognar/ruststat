// ==========================================
// ==========================================
// Inverse Gaussian (Wald) Distribution
// ==========================================
// ==========================================

use std::f64::consts::PI;
use rand::random;
use crate::{normal_cdf, normal_ran};

/// Struct for the Inverse Gaussian distribution `X ~ InvGauss(mu, lambda)`.
/// # Parameters
/// - Mean: `mu > 0`
/// - Shape: `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ InvGauss(mu=1.0, lambda=2.0)`. Use
/// ```
/// use ruststat::InvGaussDist;
/// let mut myig = InvGaussDist{mu:1.0, lambda:2.0};
/// println!("Probability density function f(1.5): {}", myig.pdf(1.5));
/// println!("Cumulative distribution function P(X<=1.5): {}", myig.cdf(1.5));
/// println!("99th percentile: {}", myig.per(0.99));
/// println!("Random draw: {}", myig.ran());
/// println!("Random vector: {:?}", myig.ranvec(5));
/// println!("Mean: {}", myig.mean());
/// println!("Variance: {}", myig.var());
/// println!("Standard deviation: {}", myig.sd());
/// ```
pub struct InvGaussDist {
    pub mu: f64,
    pub lambda: f64,
}

impl InvGaussDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        invgauss_pdf(x, self.mu, self.lambda)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        invgauss_cdf(x, self.mu, self.lambda)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        invgauss_per(p, self.mu, self.lambda)
    }
    pub fn ran(&mut self) -> f64 {
        invgauss_ran(self.mu, self.lambda)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        invgauss_ranvec(n, self.mu, self.lambda)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu
    }
    pub fn var(&mut self) -> f64 {
        self.mu.powi(3) / self.lambda
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ InvGauss(mu, lambda)`.
pub fn invgauss_pdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if mu <= 0.0 || lambda <= 0.0 {
        println!("NAN produced. Error in function invgauss_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (lambda / (2.0 * PI * x.powi(3))).sqrt() * (-(lambda * (x - mu).powi(2)) / (2.0 * mu.powi(2) * x)).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ InvGauss(mu, lambda)`.
pub fn invgauss_cdf(x: f64, mu: f64, lambda: f64) -> f64 {
    if mu <= 0.0 || lambda <= 0.0 {
        println!("NAN produced. Error in function invgauss_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z1 = (lambda / x).sqrt() * (x / mu - 1.0);
    let z2 = -(lambda / x).sqrt() * (x / mu + 1.0);

    // Guard against exponent explosion on extreme distributions where the 2nd term drops to 0.
    let exp_term = if (2.0 * lambda / mu) > 500.0 {
        0.0
    } else {
        (2.0 * lambda / mu).exp() * normal_cdf(z2, 0.0, 1.0)
    };

    normal_cdf(z1, 0.0, 1.0) + exp_term
}

/// Computes a percentile for `X ~ InvGauss(mu, lambda)`.
pub fn invgauss_per(p: f64, mu: f64, lambda: f64) -> f64 {
    if mu <= 0.0 || lambda <= 0.0 || p <= 0.0 || p >= 1.0 {
        println!("NAN produced. Error in function invgauss_per");
        return f64::NAN;
    }
    // Bisection Solver
    let mut low = 0.0;
    let mut high = mu * 10.0;
    while invgauss_cdf(high, mu, lambda) < p {
        high *= 2.0;
    }
    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        if invgauss_cdf(mid, mu, lambda) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    (low + high) / 2.0
}

/// Random draw from `X ~ InvGauss(mu, lambda)` distribution using the Michael-Schucany-Haas algorithm.
pub fn invgauss_ran(mu: f64, lambda: f64) -> f64 {
    if mu <= 0.0 || lambda <= 0.0 {
        println!("Bad argument to invgauss_ran");
        return f64::NAN;
    }
    let v = normal_ran(0.0, 1.0);
    let y = v * v;
    let x1 = mu + (mu * mu * y) / (2.0 * lambda)
        - (mu / (2.0 * lambda)) * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();
    let u: f64 = random();
    if u <= (mu / (mu + x1)) {
        x1
    } else {
        (mu * mu) / x1
    }
}

/// Save random draws from `X ~ InvGauss(mu, lambda)` distribution into a `Vec`
pub fn invgauss_ranvec(n: u64, mu: f64, lambda: f64) -> Vec<f64> {
    if mu <= 0.0 || lambda <= 0.0 {
        println!("Bad argument to invgauss_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        xvec.push(invgauss_ran(mu, lambda));
    }
    xvec
}