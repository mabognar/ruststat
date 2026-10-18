// ==========================================
// ==========================================
// Zero-Inflated Poisson Distribution
// ==========================================
// ==========================================

use rand::random;
use crate::{pois_cdf, pois_pmf, pois_ran};

/// Struct for the Zero-Inflated Poisson distribution `X ~ ZIP(lambda, pi)`.
/// # Parameters
/// - Poisson rate: `lambda > 0`
/// - Probability of excess zeros: `pi` (`0 <= pi <= 1`)
/// # Support
/// - `x = 0, 1, 2, ...`
/// # Example
/// Suppose `X ~ ZIP(lambda=2.5, pi=0.2)`. Use
/// ```
/// use ruststat::ZipDist;
/// let mut myzip = ZipDist{lambda:2.5, pi:0.2};
/// println!("P(X=0): {}", myzip.pmf(0));
/// println!("P(X<=2): {}", myzip.cdf(2));
/// println!("99th percentile: {}", myzip.per(0.99));
/// println!("Random draw: {}", myzip.ran());
/// println!("Random vector: {:?}", myzip.ranvec(5));
/// println!("Mean: {}", myzip.mean());
/// println!("Variance: {}", myzip.var());
/// println!("Standard deviation: {}", myzip.sd());
/// ```
pub struct ZipDist {
    pub lambda: f64,
    pub pi: f64,
}

impl ZipDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        zip_pmf(x, self.lambda, self.pi)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        zip_cdf(x, self.lambda, self.pi)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        zip_per(q, self.lambda, self.pi)
    }
    pub fn ran(&mut self) -> u64 {
        zip_ran(self.lambda, self.pi)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        zip_ranvec(n, self.lambda, self.pi)
    }
    pub fn mean(&mut self) -> f64 {
        (1.0 - self.pi) * self.lambda
    }
    pub fn var(&mut self) -> f64 {
        (1.0 - self.pi) * self.lambda * (1.0 + self.pi * self.lambda)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability mass function (pmf) for `X ~ ZIP(lambda, pi)`.
pub fn zip_pmf(x: u64, lambda: f64, pi: f64) -> f64 {
    if lambda <= 0.0 || pi < 0.0 || pi > 1.0 {
        println!("NAN produced. Error in function zip_pmf");
        return f64::NAN;
    }
    if x == 0 {
        pi + (1.0 - pi) * pois_pmf(0, lambda)
    } else {
        (1.0 - pi) * pois_pmf(x, lambda)
    }
}

/// Computes cumulative distribution function (cdf) for `X ~ ZIP(lambda, pi)`.
pub fn zip_cdf(x: u64, lambda: f64, pi: f64) -> f64 {
    if lambda <= 0.0 || pi < 0.0 || pi > 1.0 {
        println!("NAN produced. Error in function zip_cdf");
        return f64::NAN;
    }
    pi + (1.0 - pi) * pois_cdf(x, lambda)
}

/// Computes a percentile for `X ~ ZIP(lambda, pi)`.
pub fn zip_per(q: f64, lambda: f64, pi: f64) -> u64 {
    if lambda <= 0.0 || pi < 0.0 || pi > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function zip_per");
        return f64::NAN as u64;
    }
    if q <= zip_cdf(0, lambda, pi) {
        return 0;
    }
    if q == 1.0 {
        return u64::MAX;
    }
    let mut x = 0;
    while zip_cdf(x, lambda, pi) < q {
        x += 1;
    }
    x
}

/// Random draw from `X ~ ZIP(lambda, pi)` distribution.
pub fn zip_ran(lambda: f64, pi: f64) -> u64 {
    if lambda <= 0.0 || pi < 0.0 || pi > 1.0 {
        println!("NAN produced. Error in function zip_ran");
        return f64::NAN as u64;
    }
    let u: f64 = random();
    if u <= pi {
        0
    } else {
        pois_ran(lambda)
    }
}

/// Save random draws from `X ~ ZIP(lambda, pi)` distribution into a `Vec`
pub fn zip_ranvec(nn: u64, lambda: f64, pi: f64) -> Vec<u64> {
    if lambda <= 0.0 || pi < 0.0 || pi > 1.0 {
        println!("Bad argument to zip_ranvec");
        return if nn == 0 { vec![] } else { vec![f64::NAN as u64] };
    }
    let mut xvec: Vec<u64> = Vec::with_capacity(nn as usize);
    for _ in 0..nn {
        let u: f64 = random();
        if u <= pi {
            xvec.push(0);
        } else {
            xvec.push(pois_ran(lambda));
        }
    }
    xvec
}