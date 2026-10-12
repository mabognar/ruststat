// ==========================================
// ==========================================
// Negative Binomial Distribution 2
// ==========================================
// ==========================================

use crate::{betai, factln_u};
use crate::geo2::geo2_ran;

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

    (factln_u(x-1) - factln_u(r-1) - factln_u(x-r) + (r as f64)*p.ln() +
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
    if r < 1 || p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function nb2_cdf");
        return f64::NAN;
    }
    if x < r { return 0.0; }
    if p == 1.0 { return 1.0; }

    betai(p, r as f64, (x - r + 1) as f64)
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
    if r < 1 || p <= 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function nb2_per");
        return f64::NAN as u64;
    }

    if q == 0.0 { return r; }
    if q == 1.0 { return u64::MAX; } // Safely replaced f64::INFINITY as u64

    let mut x = r;
    let mut sum = nb2_pmf(x, r, p);

    // Accumulate the PMF
    while sum < q {
        x += 1;
        sum += nb2_pmf(x, r, p);
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
