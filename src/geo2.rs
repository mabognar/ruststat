// ==========================================
// ==========================================
// Geometric Distribution 2
// ==========================================
// ==========================================

use crate::unif::unif_ran;

/// Struct for the geometric distribution `X ~ Geo(p)` where
/// `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.2)`. Use
/// ```
/// use ruststat::Geo2Dist;
/// let mut mygeo = Geo2Dist{p:0.2};
/// println!("probability mass function: {}", mygeo.pmf(8));
/// println!("cumulative distribution function: {}", mygeo.cdf(8));
/// println!("percentile: {}", mygeo.per(0.99));
/// println!("random: {}", mygeo.ran());
/// println!("Random vector: {:?}", mygeo.ranvec(5));
/// println!("mean: {}", mygeo.mean());
/// println!("variance: {}", mygeo.var());
/// println!("standard deviation: {}", mygeo.sd());
/// ```
pub struct Geo2Dist {
    pub p: f64,
}
impl Geo2Dist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        geo2_pmf(x, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        geo2_cdf(x, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        geo2_per(q, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        geo2_ran(self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        geo2_ranvec(n, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        1.0 / self.p
    }
    pub fn var(&mut self) -> f64 {
        (1.0 - self.p) / self.p.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ Geo(p)`
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To compute `P(X = 3)`, use
/// ```
/// use ruststat::geo2_pmf;
/// println!("P(X=x): {}", geo2_pmf(3, 0.6));
/// ```
pub fn geo2_pmf(x: u64, p: f64) -> f64 {
    if p < 0.0 || p > 1.0 { // Changed to p < 0.0
        println!("NAN produced. Error in function geo2_pmf");
        return f64::NAN;
    }
    if x < 1 {
        return 0.0;
    }

    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return if x == 1 { 1.0 } else { 0.0 };
    }

    (1.0-p).powi((x-1) as i32) * p
}

/// Computes cumulative distribution function (cdf) for `X ~ Geo(p)`
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To compute `P(X <= 3)`, use
/// ```
/// use ruststat::geo2_cdf;
/// println!("P(X<=x): {}", geo2_cdf(3, 0.6));
/// ```
pub fn geo2_cdf(x: u64, p: f64) -> f64 {
    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function geo2_cdf");
        return f64::NAN;
    }
    if x < 1 { return 0.0; }

    1.0 - (1.0 - p).powi(x as i32)
}

/// Computes a percentile for `X ~ Geo(p)`
/// where `X =` the trial on which the first success is observed.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 1,2,3,...`
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::geo2_per;
/// println!("Percentile: {}", geo2_per(0.8, 0.6));
/// ```
pub fn geo2_per(q: f64, p: f64) -> u64 {
    let mut x = 1;

    if p <= 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function geo2_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 1;
    }
    if q == 1.0 {
        return u64::MAX;
    }

    while geo2_cdf(x, p) < q {
        x += 1;
    }

    x
}


/// Random draw from `X ~ Geo(p)` distribution
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. Use
/// ```
/// use ruststat::geo2_ran;
/// println!("Random draw: {}", geo2_ran(0.6));
/// ```
pub fn geo2_ran(p: f64) -> u64 {

    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function geo2_ran");
        return f64::NAN as u64;
    }

    let u = unif_ran(0.0, 1.0);

    (u.ln() / (1.0-p).ln()).floor() as u64 + 1
}

/// Save random draws from `X ~ Geo(p)` distribution into a `Vec`
/// where `X =` the trial on which the first success is observed.
///
/// # Parameters
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ Geo(p=0.6)`. Use
/// ```
/// use ruststat::geo2_ranvec;
/// println!("Random Vec: {:?}", geo2_ranvec(10, 0.6));
/// ```
pub fn geo2_ranvec(nn: u64, p: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(geo2_ran(p));
    }

    xvec
}
