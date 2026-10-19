use crate::{factln_u, gammq};
use crate::unif::unif_ran;

/// Struct for the Poisson distribution `X ~ Pois(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. Use
/// ```
/// use ruststat::PoisDist;
/// let mut mypois = PoisDist{lambda:2.5};
/// println!("P(X=6): {}", mypois.pmf(6));
/// println!("P(X<=6): {}", mypois.cdf(6));
/// println!("99th percentile: {}", mypois.per(0.99));
/// println!("Random draw: {}", mypois.ran());
/// println!("Random vector: {:?}", mypois.ranvec(5));
/// println!("Mean: {}", mypois.mean());
/// println!("Variance: {}", mypois.var());
/// println!("Standard deviation: {}", mypois.sd());
/// ```
pub struct PoisDist {
    pub lambda: f64,
}
impl PoisDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        pois_pmf(x, self.lambda)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        pois_cdf(x, self.lambda)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        pois_per(q, self.lambda)
    }
    pub fn ran(&mut self) -> u64 {
        pois_ran(self.lambda)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        pois_ranvec(n, self.lambda)
    }
    pub fn mean(&mut self) -> f64 {
        self.lambda
    }
    pub fn var(&mut self) -> f64 {
        self.lambda
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ Pois(lambda).`
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. To compute `P(X = 4)`, use
/// ```
/// use ruststat::pois_pmf;
/// println!("P(X=x): {}", pois_pmf(4, 2.5));
/// ```
pub fn pois_pmf(x: u64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function pois_pmf");
        return f64::NAN;
    }
    (-lambda + (x as f64)*lambda.ln() - factln_u(x)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Pois(lambda).`
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. To compute `P(X <= 4)`, use
/// ```
/// use ruststat::pois_cdf;
/// println!("P(X<=x): {}", pois_cdf(4, 2.5));
/// ```
pub fn pois_cdf(x: u64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function pois_cdf");
        return f64::NAN;
    }
    gammq(lambda, (x+1) as f64)
}


/// Computes a percentile for `X ~ Pois(lambda)`.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x = 0,1,2,...`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::pois_per;
/// println!("Percentile: {}", pois_per(2.5, 0.8));
/// ```
pub fn pois_per(q: f64, lambda: f64) -> u64 {
    let mut x = 0;

    if lambda <= 0.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function pois_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 0;
    }
    if q == 1.0 {
        return u64::MAX;
    }

    while pois_cdf(x, lambda) < q {
        x += 1;
    }
    x
}


/// Random draw from `X ~ Pois(r, p)` distribution.
/// Sheldon Ross, Simulation, 2003
///
/// # Parameters
/// - `lambda > 0`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. Use
/// ```
/// use ruststat::pois_ran;
/// println!("Random draw: {}", pois_ran(2.5));
/// ```
pub fn pois_ran(lambda: f64) -> u64 {

    if lambda <= 0.0 {
        println!("NAN produced. Error in function pois_ran");
        return f64::NAN as u64;
    }

    let u = unif_ran(0.0, 1.0);
    let mut i = 0;
    let mut p = (-lambda).exp();
    let mut f = p;
    loop {
        if u < f {
            return i;
        }
        p = lambda*p / (i+1) as f64;
        f = f + p;
        i = i + 1;
    }
}


/// Save random draws from `X ~ Pois(lambda)` distribution into a `Vec`
///
/// # Parameters
/// - `lambda > 0`
/// # Example
/// Suppose `X ~ Pois(lambda=2.5)`. Use
/// ```
/// use ruststat::pois_ranvec;
/// println!("Random Vec: {:?}", pois_ranvec(10, 2.5));
/// ```
pub fn pois_ranvec(nn: u64, lambda: f64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(pois_ran(lambda));
    }
    xvec
}
