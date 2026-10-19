use crate::{gamma_ln, gammq, gammai_inv, gamma_ran};

/// Struct for the Inverse Gamma distribution `X ~ InvGamma(alpha, beta)`.
/// # Parameters
/// - Shape: `alpha > 0`
/// - Scale: `beta > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ InvGamma(alpha=2.0, beta=3.0)`. Use
/// ```
/// use ruststat::InvGammaDist;
/// let mut myig = InvGammaDist{alpha:2.0, beta:3.0};
/// println!("Probability density function f(1.5): {}", myig.pdf(1.5));
/// println!("Cumulative distribution function P(X<=1.5): {}", myig.cdf(1.5));
/// println!("99th percentile: {}", myig.per(0.99));
/// println!("Random draw: {}", myig.ran());
/// println!("Random vector: {:?}", myig.ranvec(5));
/// println!("Mean: {}", myig.mean());
/// println!("Variance: {}", myig.var());
/// println!("Standard deviation: {}", myig.sd());
/// ```
pub struct InvGammaDist {
    pub alpha: f64,
    pub beta: f64,
}

impl InvGammaDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        invgamma_pdf(x, self.alpha, self.beta)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        invgamma_cdf(x, self.alpha, self.beta)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        invgamma_per(p, self.alpha, self.beta)
    }
    pub fn ran(&mut self) -> f64 {
        invgamma_ran(self.alpha, self.beta)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        invgamma_ranvec(n, self.alpha, self.beta)
    }
    pub fn mean(&mut self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::NAN
        }
    }
    pub fn var(&mut self) -> f64 {
        if self.alpha > 2.0 {
            self.beta.powi(2) / ((self.alpha - 1.0).powi(2) * (self.alpha - 2.0))
        } else {
            f64::NAN
        }
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ InvGamma(alpha, beta)`.
///
/// # Parameters
/// - Shape: `alpha > 0`
/// - Scale: `beta > 0`
///
/// # Support
/// - `x > 0`
///
/// # Example
/// Suppose `X ~ InvGamma(alpha=2.0, beta=3.0)`. Use
/// ```
/// use ruststat::invgamma_pdf;
/// println!("Probability density function f(1.5): {}", invgamma_pdf(1.5, 2.0, 3.0));
/// ```
pub fn invgamma_pdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function invgamma_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (alpha * beta.ln() - gamma_ln(alpha) - (alpha + 1.0) * x.ln() - beta / x).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ InvGamma(alpha, beta)`.
///
/// # Parameters
/// - Shape: `alpha > 0`
/// - Scale: `beta > 0`
///
/// # Support
/// - `x > 0`
///
/// # Example
/// Suppose `X ~ InvGamma(alpha=2.0, beta=3.0)`. Use
/// ```
/// use ruststat::invgamma_cdf;
/// println!("P(X<=1.5): {}", invgamma_cdf(1.5, 2.0, 3.0));
/// ```
pub fn invgamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("NAN produced. Error in function invgamma_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gammq(beta / x, alpha)
}

/// Computes a percentile for `X ~ InvGamma(alpha, beta)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = p`.
///
/// # Parameters
/// - Shape: `alpha > 0`
/// - Scale: `beta > 0`
///
/// # Support
/// - `x > 0`
///
/// # Example
/// Suppose `X ~ InvGamma(alpha=2.0, beta=3.0)`. To find the 80th percentile, use `p=0.80` and
/// ```
/// use ruststat::invgamma_per;
/// println!("Percentile: {}", invgamma_per(0.8, 2.0, 3.0));
/// ```
pub fn invgamma_per(p: f64, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 || p <= 0.0 || p >= 1.0 {
        println!("NAN produced. Error in function invgamma_per");
        return f64::NAN;
    }
    beta / gammai_inv(1.0 - p, alpha)
}

/// Random draw from `X ~ InvGamma(alpha, beta)` distribution.
///
/// # Parameters
/// - Shape: `alpha > 0`
/// - Scale: `beta > 0`
///
/// # Support
/// - `x > 0`
///
/// # Example
/// Suppose `X ~ InvGamma(alpha=2.0, beta=3.0)`. Use
/// ```
/// use ruststat::invgamma_ran;
/// println!("Random draw: {}", invgamma_ran(2.0, 3.0));
/// ```
pub fn invgamma_ran(alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("Bad argument to invgamma_ran");
        return f64::NAN;
    }
    1.0 / gamma_ran(alpha, beta)
}

/// Save random draws from `X ~ InvGamma(alpha, beta)` distribution into a `Vec`.
///
/// # Parameters
/// - Shape: `alpha > 0`
/// - Scale: `beta > 0`
///
/// # Support
/// - `x > 0`
///
/// # Example
/// Suppose `X ~ InvGamma(alpha=2.0, beta=3.0)`. Use
/// ```
/// use ruststat::invgamma_ranvec;
/// println!("Random Vec: {:?}", invgamma_ranvec(10, 2.0, 3.0));
/// ```
pub fn invgamma_ranvec(n: u64, alpha: f64, beta: f64) -> Vec<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        println!("Bad argument to invgamma_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        xvec.push(1.0 / gamma_ran(alpha, beta));
    }
    xvec
}