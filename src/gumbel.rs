use std::f64::consts::PI;
use crate::{unif_ran};

const GAMMA: f64 = 0.577215664901532860606512090082;

/// Struct for the Gumbel distribution `X ~ Gumbel(mu, beta)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=4.2, beta=1.8)`. Use.
/// ```
/// use ruststat::GumbelDist;
/// let mut mygum = GumbelDist{mu:4.2, beta:1.8};
/// println!("Probability density function f(3.9): {}", mygum.pdf(3.9));
/// println!("Cumulative distribution function P(X<=3.9): {}", mygum.cdf(3.9));
/// println!("99th percentile: {}", mygum.per(0.99));
/// println!("Random draw: {}", mygum.ran());
/// println!("Random vector: {:?}", mygum.ranvec(5));
/// println!("Mean: {}", mygum.mean());
/// println!("Variance: {}", mygum.var());
/// println!("Standard deviation: {}", mygum.sd());
/// ```
pub struct GumbelDist {
    pub mu: f64,    //location
    pub beta: f64,  //scale
}
impl GumbelDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        gumbel_pdf(x, self.mu, self.beta)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        gumbel_cdf(x, self.mu, self.beta)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        gumbel_per(x, self.mu, self.beta)
    }
    pub fn ran(&mut self) -> f64 {
        gumbel_ran(self.mu, self.beta)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        gumbel_ranvec(n, self.mu, self.beta)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu + self.beta * GAMMA
    }
    pub fn var(&mut self) -> f64 {
        PI.powi(2)/6.0 * self.beta.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Gumbel(mu, beta)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use.
/// ```
/// use ruststat::gumbel_pdf;
/// println!("Probability density function f(3.9): {}", gumbel_pdf(3.9, 5.5, 2.0));
pub fn gumbel_pdf(x: f64, mu: f64, beta: f64) -> f64 {
    if beta <= 0.0 {
        println!("NAN produced. Error in function gumbel_pdf");
        return f64::NAN;
    }
    1.0 / beta * (-((x-mu)/beta + (-(x-mu)/beta).exp())).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Gumbel(mu, beta)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use.
/// ```
/// use ruststat::gumbel_cdf;
/// println!("P(X<=3.9): {}", gumbel_cdf(3.9, 5.5, 2.0));
/// ```
pub fn gumbel_cdf(x: f64, mu: f64, beta: f64) -> f64 {
    if beta <= 0.0 {
        println!("NAN produced. Error in function gumbel_pdf");
        return f64::NAN;
    }

    (-(-(x-mu)/beta).exp()).exp()
}


/// Computes a percentile for `X ~ Gumbel(mu, beta)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::gumbel_per;
/// println!("Percentile: {}", gumbel_per(0.8, 5.5, 2.0));
/// ```
pub fn gumbel_per(p: f64, mu: f64, beta: f64) -> f64 {
    mu - beta * (-p.ln()).ln()
}


/// Random draw from `X ~ Gumbel(mu, beta)` distribution.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use
/// ```
/// use ruststat::gumbel_ran;
/// println!("Random draw: {:?}", gumbel_ran(5.5, 2.0));
/// ```
pub fn gumbel_ran(mu: f64, beta: f64) -> f64 {
    mu - beta * (-unif_ran(0.0,1.0).ln()).ln()
}


/// Random Vector taken from `X ~ Gumbel(mu, beta)` distribution.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `beta > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Gumbel(mu=5.5, beta=2.0)`. Use
/// ```
/// use ruststat::gumbel_ranvec;
/// println!("Random Vec: {:?}", gumbel_ranvec(10, 5.5, 2.0));
/// ```
pub fn gumbel_ranvec(n: u64, mu: f64, beta: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(mu - beta * (-unif_ran(0.0,1.0).ln()).ln());
    }
    xvec
}
