// ==========================================
// ==========================================
// Zero-Inflated Negative Binomial
// ==========================================
// ==========================================

use rand::random;
use crate::{nb_cdf, nb_pmf, nb_ran};

/// Struct for the Zero-Inflated Negative Binomial distribution `X ~ ZINB(r, p, pi)`.
/// # Parameters
/// - Target successes: `r = 1, 2, 3, ...`
/// - Probability of success: `p` (`0 < p <= 1`)
/// - Probability of excess zeros: `pi` (`0 <= pi <= 1`)
/// # Support
/// - `x = 0, 1, 2, ...`
/// # Example
/// Suppose `X ~ ZINB(r=4, p=0.2, pi=0.15)`. Use
/// ```
/// use ruststat::ZinbDist;
/// let mut myzinb = ZinbDist{r:4, p:0.2, pi:0.15};
/// println!("P(X=0): {}", myzinb.pmf(0));
/// println!("P(X<=2): {}", myzinb.cdf(2));
/// println!("99th percentile: {}", myzinb.per(0.99));
/// println!("Random draw: {}", myzinb.ran());
/// println!("Random vector: {:?}", myzinb.ranvec(5));
/// println!("Mean: {}", myzinb.mean());
/// println!("Variance: {}", myzinb.var());
/// println!("Standard deviation: {}", myzinb.sd());
/// ```
pub struct ZinbDist {
    pub r: u64,
    pub p: f64,
    pub pi: f64,
}

impl ZinbDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        zinb_pmf(x, self.r, self.p, self.pi)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        zinb_cdf(x, self.r, self.p, self.pi)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        zinb_per(q, self.r, self.p, self.pi)
    }
    pub fn ran(&mut self) -> u64 {
        zinb_ran(self.r, self.p, self.pi)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        zinb_ranvec(n, self.r, self.p, self.pi)
    }
    pub fn mean(&mut self) -> f64 {
        (1.0 - self.pi) * (self.r as f64) * (1.0 - self.p) / self.p
    }
    pub fn var(&mut self) -> f64 {
        let mean_nb = (self.r as f64) * (1.0 - self.p) / self.p;
        let var_nb = mean_nb / self.p;
        (1.0 - self.pi) * var_nb + self.pi * (1.0 - self.pi) * mean_nb.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability mass function (pmf) for `X ~ ZINB(r, p, pi)`.
pub fn zinb_pmf(x: u64, r: u64, p: f64, pi: f64) -> f64 {
    if r < 1 || p <= 0.0 || p > 1.0 || pi < 0.0 || pi > 1.0 {
        println!("NAN produced. Error in function zinb_pmf");
        return f64::NAN;
    }
    if x == 0 {
        pi + (1.0 - pi) * nb_pmf(0, r, p)
    } else {
        (1.0 - pi) * nb_pmf(x, r, p)
    }
}

/// Computes cumulative distribution function (cdf) for `X ~ ZINB(r, p, pi)`.
pub fn zinb_cdf(x: u64, r: u64, p: f64, pi: f64) -> f64 {
    if r < 1 || p <= 0.0 || p > 1.0 || pi < 0.0 || pi > 1.0 {
        println!("NAN produced. Error in function zinb_cdf");
        return f64::NAN;
    }
    pi + (1.0 - pi) * nb_cdf(x, r, p)
}

/// Computes a percentile for `X ~ ZINB(r, p, pi)`.
pub fn zinb_per(q: f64, r: u64, p: f64, pi: f64) -> u64 {
    if r < 1 || p <= 0.0 || p > 1.0 || pi < 0.0 || pi > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function zinb_per");
        return f64::NAN as u64;
    }
    if q <= zinb_cdf(0, r, p, pi) {
        return 0;
    }
    if q == 1.0 {
        return u64::MAX;
    }
    let mut x = 0;
    while zinb_cdf(x, r, p, pi) < q {
        x += 1;
    }
    x
}

/// Random draw from `X ~ ZINB(r, p, pi)` distribution.
pub fn zinb_ran(r: u64, p: f64, pi: f64) -> u64 {
    if r < 1 || p <= 0.0 || p > 1.0 || pi < 0.0 || pi > 1.0 {
        println!("NAN produced. Error in function zinb_ran");
        return f64::NAN as u64;
    }
    let u: f64 = random();
    if u <= pi {
        0
    } else {
        nb_ran(r, p)
    }
}

/// Save random draws from `X ~ ZINB(r, p, pi)` distribution into a `Vec`
pub fn zinb_ranvec(nn: u64, r: u64, p: f64, pi: f64) -> Vec<u64> {
    if r < 1 || p <= 0.0 || p > 1.0 || pi < 0.0 || pi > 1.0 {
        println!("Bad argument to zinb_ranvec");
        return if nn == 0 { vec![] } else { vec![f64::NAN as u64] };
    }
    let mut xvec: Vec<u64> = Vec::with_capacity(nn as usize);
    for _ in 0..nn {
        let u: f64 = random();
        if u <= pi {
            xvec.push(0);
        } else {
            xvec.push(nb_ran(r, p));
        }
    }
    xvec
}