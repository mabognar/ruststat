use std::f64::consts::PI;
use rand::random;

/// Struct for the Cauchy distribution `X ~ Cauchy(x0, gamma)`.
/// # Parameters
/// - Location: `-infinity < x0 < infinity`
/// - Scale: `gamma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Cauchy(x0=0.0, gamma=1.0)`. Use
/// ```
/// use ruststat::CauchyDist;
/// let mut mycauchy = CauchyDist{x0:0.0, gamma:1.0};
/// println!("Probability density function f(2.5): {}", mycauchy.pdf(2.5));
/// println!("Cumulative distribution function P(X<=2.5): {}", mycauchy.cdf(2.5));
/// println!("99th percentile: {}", mycauchy.per(0.99));
/// println!("Random draw: {}", mycauchy.ran());
/// println!("Random vector: {:?}", mycauchy.ranvec(5));
/// println!("Mean: {}", mycauchy.mean());
/// println!("Variance: {}", mycauchy.var());
/// println!("Standard deviation: {}", mycauchy.sd());
/// ```
pub struct CauchyDist {
    pub x0: f64,
    pub gamma: f64,
}

impl CauchyDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        cauchy_pdf(x, self.x0, self.gamma)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        cauchy_cdf(x, self.x0, self.gamma)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        cauchy_per(p, self.x0, self.gamma)
    }
    pub fn ran(&mut self) -> f64 {
        cauchy_ran(self.x0, self.gamma)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        cauchy_ranvec(n, self.x0, self.gamma)
    }
    pub fn mean(&mut self) -> f64 {
        f64::NAN // Cauchy mean is undefined
    }
    pub fn var(&mut self) -> f64 {
        f64::NAN // Cauchy variance is undefined
    }
    pub fn sd(&mut self) -> f64 {
        f64::NAN
    }
}

/// Computes probability density function (pdf) for `X ~ Cauchy(x0, gamma)`.
/// # Parameters
/// - Location: `-infinity < x0 < infinity`
/// - Scale: `gamma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Cauchy(x0=0.0, gamma=1.0)`. Use.
/// ```
/// use ruststat::cauchy_pdf;
/// println!("Probability density function f(2.5): {}", cauchy_pdf(2.5, 0.0, 1.0));
/// ```
pub fn cauchy_pdf(x: f64, x0: f64, gamma: f64) -> f64 {
    if gamma <= 0.0 {
        println!("NAN produced. Error in function cauchy_pdf");
        return f64::NAN;
    }
    1.0 / (PI * gamma * (1.0 + ((x - x0) / gamma).powi(2)))
}

/// Computes cumulative distribution function (cdf) for `X ~ Cauchy(x0, gamma)`.
/// # Parameters
/// - Location: `-infinity < x0 < infinity`
/// - Scale: `gamma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Cauchy(x0=0.0, gamma=1.0)`. Use.
/// ```
/// use ruststat::cauchy_cdf;
/// println!("P(X<=2.5): {}", cauchy_cdf(2.5, 0.0, 1.0));
/// ```
pub fn cauchy_cdf(x: f64, x0: f64, gamma: f64) -> f64 {
    if gamma <= 0.0 {
        println!("NAN produced. Error in function cauchy_cdf");
        return f64::NAN;
    }
    0.5 + (1.0 / PI) * ((x - x0) / gamma).atan()
}

/// Computes a percentile for `X ~ Cauchy(x0, gamma)`.
/// # Note
/// Determines the value of `x` such that `P(X <= x) = p`.
/// # Parameters
/// - Location: `-infinity < x0 < infinity`
/// - Scale: `gamma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Cauchy(x0=0.0, gamma=1.0)`. To find the 80th percentile, use `p=0.80` and
/// ```
/// use ruststat::cauchy_per;
/// println!("Percentile: {}", cauchy_per(0.8, 0.0, 1.0));
/// ```
pub fn cauchy_per(p: f64, x0: f64, gamma: f64) -> f64 {
    if gamma <= 0.0 || p <= 0.0 || p >= 1.0 {
        println!("NAN produced. Error in function cauchy_per");
        return f64::NAN;
    }
    x0 + gamma * (PI * (p - 0.5)).tan()
}

/// Random draw from `X ~ Cauchy(x0, gamma)` distribution.
/// # Parameters
/// - Location: `-infinity < x0 < infinity`
/// - Scale: `gamma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Cauchy(x0=0.0, gamma=1.0)`. Use
/// ```
/// use ruststat::cauchy_ran;
/// println!("Random draw: {}", cauchy_ran(0.0, 1.0));
/// ```
pub fn cauchy_ran(x0: f64, gamma: f64) -> f64 {
    if gamma <= 0.0 {
        println!("Bad argument to cauchy_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    x0 + gamma * (PI * (u - 0.5)).tan()
}

/// Save random draws from `X ~ Cauchy(x0, gamma)` distribution into a `Vec`
/// # Parameters
/// - Location: `-infinity < x0 < infinity`
/// - Scale: `gamma > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Cauchy(x0=0.0, gamma=1.0)`. Use
/// ```
/// use ruststat::cauchy_ranvec;
/// println!("Random Vec: {:?}", cauchy_ranvec(10, 0.0, 1.0));
/// ```
pub fn cauchy_ranvec(n: u64, x0: f64, gamma: f64) -> Vec<f64> {
    if gamma <= 0.0 {
        println!("Bad argument to cauchy_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        xvec.push(x0 + gamma * (PI * (u - 0.5)).tan());
    }
    xvec
}
