// ==========================================
// ==========================================
// Gamma Distribution
// ==========================================
// ==========================================

use rand::random;
use crate::{gamma_ln, gammai_inv, gammp, normal_ran};

/// Struct for the gamma distribution `X ~ Gamma(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use.
/// ```
/// use ruststat::GammaDist;
/// let mut mygamma = GammaDist{alpha:4.0, beta:2.7};
/// println!("Probability density function f(3.9): {}", mygamma.pdf(3.9));
/// println!("Cumulative distribution function P(X<=3.9): {}", mygamma.cdf(3.9));
/// println!("99th percentile: {}", mygamma.per(0.99));
/// println!("Random draw: {}", mygamma.ran());
/// println!("Random vector: {:?}", mygamma.ranvec(5));
/// println!("Mean: {}", mygamma.mean());
/// println!("Variance: {}", mygamma.var());
/// println!("Standard deviation: {}", mygamma.sd());
/// ```
pub struct GammaDist {
    pub alpha: f64,
    pub beta: f64,
}
impl GammaDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        gamma_pdf(x, self.alpha, self.beta)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        gamma_cdf(x, self.alpha, self.beta)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        gamma_per(x, self.alpha, self.beta)
    }
    pub fn ran(&mut self) -> f64 {
        gamma_ran(self.alpha, self.beta)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        gamma_ranvec(n, self.alpha, self.beta)
    }
    pub fn mean(&mut self) -> f64 {
        self.alpha / self.beta
    }
    pub fn var(&mut self) -> f64 {
        self.alpha / self.beta.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Gamma(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use.
/// ```
/// use ruststat::gamma_pdf;
/// println!("Probability density function f(3.9): {}", gamma_pdf(3.9, 4.0, 2.7));
pub fn gamma_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function gamma_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (alpha*beta.ln() - gamma_ln(alpha) + (alpha-1.0)*x.ln()  - beta*x).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Gamma(alpha, beta)`.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use.
/// ```
/// use ruststat::gamma_cdf;
/// println!("P(X<=3.9): {}", gamma_cdf(3.9, 4.0, 2.7));
/// ```
pub fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function gamma_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }

    gammp(x * beta, alpha)
}


/// Computes a percentile for `X ~ Gamma(alpha, beta)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::gamma_per;
/// println!("Percentile: {}", gamma_per(0.8, 4.0, 2.7));
/// ```
pub fn gamma_per(p: f64, alpha: f64, beta: f64) -> f64 {
    gammai_inv(p, alpha) / beta
}


/// Random draw from `X ~ Gamma(alpha, beta)` distribution.
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use
/// ```
/// use ruststat::gamma_ran;
/// println!("Random draw: {}", gamma_ran(4.0, 2.7));
/// ```
pub fn gamma_ran(alpha: f64, beta: f64) -> f64 {

    let (d, c, mut x, mut v, mut u): (f64, f64, f64, f64, f64);
    d = alpha - 1.0/3.0;
    c = 1.0/(9.0*d).sqrt();

    loop {
        x = normal_ran(0.0, 1.0);
        v = 1.0 + c * x;
        while v <= 0.0 {
            x = normal_ran(0.0, 1.0);
            v = 1.0 + c * x;
            break;
        }

        v = v.powi(3);
        u = random();
        // u = rand::thread_rng().gen();

        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v / beta;
        }
        if u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln()) {
            return d * v / beta;
        }
    }
}


/// Save random draws from `X ~ Gamma(alpha, beta)` distribution into a `Vec`
///
/// # Parameters
/// - `alpha > 0`
/// - `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Gamma(alpha=4.0, beta=2.7)`. Use
/// ```
/// use ruststat::gamma_ranvec;
/// println!("Random Vec: {:?}", gamma_ranvec(10, 4.0, 2.7));
/// ```
pub fn gamma_ranvec(n: u64, alpha: f64, beta: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(gamma_ran(alpha, beta));
    }
    xvec
}
