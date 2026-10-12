// ==========================================
// ==========================================
// Binomial Distribution
// ==========================================
// ==========================================

use crate::{betai, factln_u};
use crate::unif::unif_ran;

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

    if p == 0.0 {
        return if x == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if x == n { 1.0 } else { 0.0 };
    }

    (factln_u(n) - factln_u(x) - factln_u(n-x) + x as f64 * p.ln() + (n-x) as f64 * (1.0-p).ln()).exp()
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
    if x >= n { return 1.0; }
    if p == 0.0 { return 1.0; }
    if p == 1.0 { return 0.0; } // Caught by x >= n if it was a guaranteed success

    // P(X <= x) = I_{1-p}(n-x, x+1)
    betai(1.0 - p, (n - x) as f64, (x + 1) as f64)
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
    if n <= 0 || p < 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function bin_per");
        return f64::NAN as u64;
    }
    if q == 0.0 { return 0; }
    if q == 1.0 { return n; }

    let mut x = 0;
    let mut sum = bin_pmf(0, n, p);

    // Accumulate the PMF to avoid recalculating the CDF from zero
    while sum < q && x < n {
        x += 1;
        sum += bin_pmf(x, n, p);
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
