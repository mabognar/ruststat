use std::f64::consts::PI;
use rand::random;

/// Struct for the logistic distribution `X ~ Logistic(mu, s)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `s > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Logistic(mu=0.0, s=1.0)`. Use
/// ```
/// use ruststat::LogisticDist;
/// let mut mylogis = LogisticDist{mu:0.0, s:1.0};
/// println!("Probability density function f(2.5): {}", mylogis.pdf(2.5));
/// println!("Cumulative distribution function P(X<=2.5): {}", mylogis.cdf(2.5));
/// println!("99th percentile: {}", mylogis.per(0.99));
/// println!("Random draw: {}", mylogis.ran());
/// println!("Random vector: {:?}", mylogis.ranvec(5));
/// println!("Mean: {}", mylogis.mean());
/// println!("Variance: {}", mylogis.var());
/// println!("Standard deviation: {}", mylogis.sd());
/// ```
pub struct LogisticDist {
    pub mu: f64,
    pub s: f64,
}
impl LogisticDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        logistic_pdf(x, self.mu, self.s)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        logistic_cdf(x, self.mu, self.s)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        logistic_per(p, self.mu, self.s)
    }
    pub fn ran(&mut self) -> f64 {
        logistic_ran(self.mu, self.s)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        logistic_ranvec(n, self.mu, self.s)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu
    }
    pub fn var(&mut self) -> f64 {
        (self.s.powi(2) * PI.powi(2)) / 3.0
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ Logistic(mu, s)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `s > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Logistic(mu=0.0, s=1.0)`. Use.
/// ```
/// use ruststat::logistic_pdf;
/// println!("Probability density function f(2.5): {}", logistic_pdf(2.5, 0.0, 1.0));
/// ```
pub fn logistic_pdf(x: f64, mu: f64, s: f64) -> f64 {
    if s <= 0.0 {
        println!("NAN produced. Error in function logistic_pdf");
        return f64::NAN;
    }
    let z = (x - mu) / s;
    (-z).exp() / (s * (1.0 + (-z).exp()).powi(2))
}

/// Computes cumulative distribution function (cdf) for `X ~ Logistic(mu, s)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `s > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Logistic(mu=0.0, s=1.0)`. Use.
/// ```
/// use ruststat::logistic_cdf;
/// println!("P(X<=2.5): {}", logistic_cdf(2.5, 0.0, 1.0));
/// ```
pub fn logistic_cdf(x: f64, mu: f64, s: f64) -> f64 {
    if s <= 0.0 {
        println!("NAN produced. Error in function logistic_cdf");
        return f64::NAN;
    }
    let z = (x - mu) / s;
    1.0 / (1.0 + (-z).exp())
}

/// Computes a percentile for `X ~ Logistic(mu, s)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = p`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `s > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Logistic(mu=0.0, s=1.0)`. To find the 80th percentile, use `p=0.80` and
/// ```
/// use ruststat::logistic_per;
/// println!("Percentile: {}", logistic_per(0.8, 0.0, 1.0));
/// ```
pub fn logistic_per(p: f64, mu: f64, s: f64) -> f64 {
    if s <= 0.0 || p <= 0.0 || p >= 1.0 {
        println!("NAN produced. Error in function logistic_per");
        return f64::NAN;
    }
    mu + s * (p / (1.0 - p)).ln()
}

/// Random draw from `X ~ Logistic(mu, s)` distribution.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `s > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Logistic(mu=0.0, s=1.0)`. Use
/// ```
/// use ruststat::logistic_ran;
/// println!("Random draw: {}", logistic_ran(0.0, 1.0));
/// ```
pub fn logistic_ran(mu: f64, s: f64) -> f64 {
    if s <= 0.0 {
        println!("Bad argument to logistic_ran");
        return f64::NAN;
    }

    let u: f64 = random();
    mu + s * (u / (1.0 - u)).ln()
}

/// Save random draws from `X ~ Logistic(mu, s)` distribution into a `Vec`
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `s > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Logistic(mu=0.0, s=1.0)`. Use
/// ```
/// use ruststat::logistic_ranvec;
/// println!("Random Vec: {:?}", logistic_ranvec(10, 0.0, 1.0));
/// ```
pub fn logistic_ranvec(n: u64, mu: f64, s: f64) -> Vec<f64> {
    if s <= 0.0 {
        println!("Bad argument to logistic_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }

    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        xvec.push(mu + s * (u / (1.0 - u)).ln());
    }
    xvec
}
