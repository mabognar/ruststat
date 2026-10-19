use crate::{betai, factln_u};
use crate::geo::geo_ran;

/// Struct for the negative binomial distribution `X ~ NB(r,p)` where
/// `X =` the number of failures prior to observing the `r`th success.
///
/// # Parameters
/// - `r = 1,2,3,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=4,p=0.2)`. Use
/// ```
/// use ruststat::NBDist;
/// let mut mynb = NBDist{r:4, p:0.2};
/// println!("P(X=6): {}", mynb.pmf(6));
/// println!("P(X<=6): {}", mynb.cdf(6));
/// println!("99th percentile: {}", mynb.per(0.99));
/// println!("Random draw: {}", mynb.ran());
/// println!("Random vector: {:?}", mynb.ranvec(5));
/// println!("Mean: {}", mynb.mean());
/// println!("Variance: {}", mynb.var());
/// println!("Standard deviation: {}", mynb.sd());
/// ```
pub struct NBDist {
    pub r: u64,
    pub p: f64,
}
impl NBDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        nb_pmf(x, self.r, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        nb_cdf(x, self.r, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        nb_per(q, self.r, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        nb_ran(self.r, self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        nb_ranvec(n, self.r, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        (self.r as f64) * (1.0 - self.p) / self.p
    }
    pub fn var(&mut self) -> f64 {
        (self.r as f64) * (1.0 - self.p) / self.p.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ NB(r,p)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=2,p=0.6)`. To compute `P(X = 3)`, use
/// ```
/// use ruststat::nb_pmf;
/// println!("P(X=x): {}", nb_pmf(3, 2, 0.6));
/// ```
pub fn nb_pmf(x: u64, r: u64, p: f64) -> f64 {
    if r < 1 || p <= 0.0 || p > 1.0  {
        println!("NAN produced. Error in function nb_pmf");
        return f64::NAN;
    }

    if p == 1.0 {
        return if x == 0 { 1.0 } else { 0.0 };
    }

    (factln_u(x+r-1) - factln_u(r-1) - factln_u(x) + (r as f64)*p.ln() +
        (x as f64)*(1.0-p).ln()).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ NB(r,p)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 <= p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. To compute `P(X <= 3)`, use
/// ```
/// use ruststat::nb_cdf;
/// println!("P(X<=x): {}", nb_cdf(3, 2, 0.6));
/// ```
pub fn nb_cdf(x: u64, r: u64, p: f64) -> f64 {
    if r < 1 || p <= 0.0 || p > 1.0  {
        println!("NAN produced. Error in function nb_cdf");
        return f64::NAN;
    }
    if p == 1.0 { return 1.0; }

    // P(X <= x) = I_p(r, x+1)
    betai(p, r as f64, (x + 1) as f64)
}

/// Computes a percentile for `X ~ NB(r,p)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::nb_per;
/// println!("Percentile: {}", nb_per(0.8, 2, 0.6));
/// ```
pub fn nb_per(q: f64, r: u64, p: f64) -> u64 {
    if p <= 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function nb_per");
        return f64::NAN as u64;
    }

    if q == 0.0 { return 0; }
    if q == 1.0 { return u64::MAX; }

    let mut x = 0;
    let mut sum = nb_pmf(x, r, p);

    while sum < q {
        x += 1;
        sum += nb_pmf(x, r, p);
    }

    x
}

/// Random draw from `X ~ NB(r, p)` distribution
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. Use
/// ```
/// use ruststat::nb_ran;
/// println!("Random draw: {}", nb_ran(2, 0.6));
/// ```
pub fn nb_ran(r: u64, p: f64) -> u64 {

    if p <= 0.0 || p > 1.0 {
        println!("NAN produced. Error in function nb_ran");
        return f64::NAN as u64;
    }

    let mut geo_vec = Vec::new();
    for _ in 0..r {
        geo_vec.push(geo_ran(p));
    }
    geo_vec.iter().sum()
}


/// Save random draws from `X ~ NB(r,p)` distribution into a `Vec`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - `r = 1,2,...`
/// - `p` = probability of success (`0 < p <= 1`)
/// # Example
/// Suppose `X ~ NB(r=2, p=0.6)`. Use
/// ```
/// use ruststat::nb_ranvec;
/// println!("Random Vec: {:?}", nb_ranvec(10, 2, 0.6));
/// ```
pub fn nb_ranvec(nn: u64, r: u64, p: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(nb_ran(r,p));
    }
    xvec
}
