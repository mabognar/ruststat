use std::f64::consts::PI;
use rand::random;
use crate::{erf, erf_inv};

/// Struct for the normal distribution `X ~ Normal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use
/// ```
/// use ruststat::NormalDist;
/// let mut mynormal = NormalDist{mu:100.0, sigma:16.0};
/// println!("Probability density function f(110.0): {}", mynormal.pdf(110.0));
/// println!("Cumulative distribution function P(X<=110.0): {}", mynormal.cdf(110.0));
/// println!("99th percentile: {}", mynormal.per(0.99));
/// println!("Random draw: {}", mynormal.ran());
/// println!("Random vector: {:?}", mynormal.ranvec(5));
/// println!("Mean: {}", mynormal.mean());
/// println!("Variance: {}", mynormal.var());
/// println!("Standard deviation: {}", mynormal.sd());
/// ```
pub struct NormalDist {
    pub mu: f64,
    pub sigma: f64,
}
impl NormalDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        normal_pdf(x, self.mu, self.sigma)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        normal_cdf(x, self.mu, self.sigma)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        normal_per(x,self.mu, self.sigma)
    }
    pub fn ran(&mut self) -> f64 {
        normal_ran(self.mu, self.sigma)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        normal_ranvec(n,self.mu, self.sigma)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu
    }
    pub fn var(&mut self) -> f64 {
        self.sigma * self.sigma
    }
    pub fn sd(&mut self) -> f64 {
        self.sigma
    }
}


/// Computes probability density function (pdf) for `X ~ Normal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use.
/// ```
/// use ruststat::normal_pdf;
/// println!("Probability density function f(110.0): {}", normal_pdf(110.0, 100.0, 16.0));
pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function normal_pdf");
        return f64::NAN;
    }

    (-0.5 * (2.0 * PI * sigma.powi(2)).ln() - 0.5 * ((x - mu) / sigma).powi(2)).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ Normal(mu, sigma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use.
/// ```
/// use ruststat::normal_cdf;
/// println!("P(X<=110.0): {}", normal_cdf(110.0, 100.0, 16.0));
/// ```
pub fn normal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("NAN produced. Error in function normal_cdf");
        return f64::NAN;
    }

    0.5 * (1.0 + erf(((x-mu)/sigma) / 2.0f64.sqrt()))
}


/// Computes a percentile for `X ~ Normal(mu, sigma)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::normal_per;
/// println!("Percentile: {}", normal_per(0.8, 100.0, 16.0));
/// ```
pub fn normal_per(p: f64, mu: f64, sigma: f64) -> f64 {
    mu + sigma * 2.0f64.sqrt() * erf_inv(2.0*p-1.0)
}


/// Random draw from `X ~ Normal(mu, sigma)` distribution.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use
/// ```
/// use ruststat::normal_ran;
/// println!("Random draw: {}", normal_ran(100.0, 16.0));
/// ```
pub fn normal_ran(mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        println!("Bad argument to normal_ran");
        return f64::NAN;
    }

    let u1: f64 = random();
    let u2: f64 = random();

    // Box-Muller transform
    mu + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}


/// Save random draws from `X ~ Normal(mu, sigma)` distribution into a `Vec`
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Normal(mu=100.0, sigma=16.0)`. Use
/// ```
/// use ruststat::normal_ranvec;
/// println!("Random Vec of normals: {:?}", normal_ranvec(10, 100.0, 16.0));
/// ```
pub fn normal_ranvec(n: u64, mu: f64, sigma: f64) -> Vec<f64> {
    if n == 0 || sigma <= 0.0 {
        println!("Bad argument to normal_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }

    let mut zvecthr: Vec<f64> = Vec::with_capacity(n as usize);

    // calculate how many loop iterations we need (generating 2 at a time)
    let niter = (n + 1) / 2;

    for _ in 0..niter {
        let u1: f64 = random();
        let u2: f64 = random();

        // shared components of Box-Muller transform
        let r = sigma * (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        // generate pair
        let z1 = mu + r * theta.cos();
        let z2 = mu + r * theta.sin();

        zvecthr.push(z1);
        zvecthr.push(z2);
    }

    // if 'n' was odd number, pop off last one (un-needed)
    if n % 2 != 0 {
        zvecthr.pop();
    }

    zvecthr
}
