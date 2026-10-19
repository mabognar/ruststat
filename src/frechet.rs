use rand::random;
use crate::gamma;

/// Struct for the Fréchet distribution `X ~ Frechet(mu, sigma, alpha)`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `sigma > 0`
/// - Shape: `alpha > 0`
/// # Support
/// - `x > mu`
/// # Example
/// Suppose `X ~ Frechet(mu=0.0, sigma=1.0, alpha=3.0)`. Use
/// ```
/// use ruststat::FrechetDist;
/// let mut myfrechet = FrechetDist{mu:0.0, sigma:1.0, alpha:3.0};
/// println!("Probability density function f(1.5): {}", myfrechet.pdf(1.5));
/// println!("Cumulative distribution function P(X<=1.5): {}", myfrechet.cdf(1.5));
/// println!("99th percentile: {}", myfrechet.per(0.99));
/// println!("Random draw: {}", myfrechet.ran());
/// println!("Random vector: {:?}", myfrechet.ranvec(5));
/// println!("Mean: {}", myfrechet.mean());
/// println!("Variance: {}", myfrechet.var());
/// println!("Standard deviation: {}", myfrechet.sd());
/// ```
pub struct FrechetDist {
    pub mu: f64,
    pub sigma: f64,
    pub alpha: f64,
}

impl FrechetDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        frechet_pdf(x, self.mu, self.sigma, self.alpha)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        frechet_cdf(x, self.mu, self.sigma, self.alpha)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        frechet_per(p, self.mu, self.sigma, self.alpha)
    }
    pub fn ran(&mut self) -> f64 {
        frechet_ran(self.mu, self.sigma, self.alpha)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        frechet_ranvec(n, self.mu, self.sigma, self.alpha)
    }
    pub fn mean(&mut self) -> f64 {
        if self.alpha > 1.0 {
            self.mu + self.sigma * gamma(1.0 - 1.0 / self.alpha)
        } else {
            f64::NAN
        }
    }
    pub fn var(&mut self) -> f64 {
        if self.alpha > 2.0 {
            self.sigma.powi(2) * (gamma(1.0 - 2.0 / self.alpha) - gamma(1.0 - 1.0 / self.alpha).powi(2))
        } else {
            f64::NAN
        }
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ Frechet(mu, sigma, alpha)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `sigma > 0`
/// - Shape: `alpha > 0`
///
/// # Support
/// - `x > mu`
///
/// # Example
/// Suppose `X ~ Frechet(mu=0.0, sigma=1.0, alpha=3.0)`. Use
/// ```
/// use ruststat::frechet_pdf;
/// println!("Probability density function f(1.5): {}", frechet_pdf(1.5, 0.0, 1.0, 3.0));
/// ```
pub fn frechet_pdf(x: f64, mu: f64, sigma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || alpha <= 0.0 {
        println!("NAN produced. Error in function frechet_pdf");
        return f64::NAN;
    }
    if x <= mu {
        return 0.0;
    }
    let z = (x - mu) / sigma;
    (alpha / sigma) * z.powf(-alpha - 1.0) * (-(z.powf(-alpha))).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ Frechet(mu, sigma, alpha)`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `sigma > 0`
/// - Shape: `alpha > 0`
///
/// # Support
/// - `x > mu`
///
/// # Example
/// Suppose `X ~ Frechet(mu=0.0, sigma=1.0, alpha=3.0)`. Use
/// ```
/// use ruststat::frechet_cdf;
/// println!("P(X<=1.5): {}", frechet_cdf(1.5, 0.0, 1.0, 3.0));
/// ```
pub fn frechet_cdf(x: f64, mu: f64, sigma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || alpha <= 0.0 {
        println!("NAN produced. Error in function frechet_cdf");
        return f64::NAN;
    }
    if x <= mu {
        return 0.0;
    }
    let z = (x - mu) / sigma;
    (-(z.powf(-alpha))).exp()
}

/// Computes a percentile for `X ~ Frechet(mu, sigma, alpha)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = p`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `sigma > 0`
/// - Shape: `alpha > 0`
///
/// # Support
/// - `x > mu`
///
/// # Example
/// Suppose `X ~ Frechet(mu=0.0, sigma=1.0, alpha=3.0)`. To find the 80th percentile, use `p=0.80` and
/// ```
/// use ruststat::frechet_per;
/// println!("Percentile: {}", frechet_per(0.8, 0.0, 1.0, 3.0));
/// ```
pub fn frechet_per(p: f64, mu: f64, sigma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || alpha <= 0.0 || p <= 0.0 || p >= 1.0 {
        println!("NAN produced. Error in function frechet_per");
        return f64::NAN;
    }
    mu + sigma * (-p.ln()).powf(-1.0 / alpha)
}

/// Random draw from `X ~ Frechet(mu, sigma, alpha)` distribution.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `sigma > 0`
/// - Shape: `alpha > 0`
///
/// # Support
/// - `x > mu`
///
/// # Example
/// Suppose `X ~ Frechet(mu=0.0, sigma=1.0, alpha=3.0)`. Use
/// ```
/// use ruststat::frechet_ran;
/// println!("Random draw: {}", frechet_ran(0.0, 1.0, 3.0));
/// ```
pub fn frechet_ran(mu: f64, sigma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || alpha <= 0.0 {
        println!("Bad argument to frechet_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    mu + sigma * (-u.ln()).powf(-1.0 / alpha)
}

/// Save random draws from `X ~ Frechet(mu, sigma, alpha)` distribution into a `Vec`.
///
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `sigma > 0`
/// - Shape: `alpha > 0`
///
/// # Support
/// - `x > mu`
///
/// # Example
/// Suppose `X ~ Frechet(mu=0.0, sigma=1.0, alpha=3.0)`. Use
/// ```
/// use ruststat::frechet_ranvec;
/// println!("Random Vec: {:?}", frechet_ranvec(10, 0.0, 1.0, 3.0));
/// ```
pub fn frechet_ranvec(n: u64, mu: f64, sigma: f64, alpha: f64) -> Vec<f64> {
    if sigma <= 0.0 || alpha <= 0.0 {
        println!("Bad argument to frechet_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        xvec.push(mu + sigma * (-u.ln()).powf(-1.0 / alpha));
    }
    xvec
}