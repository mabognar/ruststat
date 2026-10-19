use crate::gamma::{gamma_pdf, gamma_cdf, gamma_per, gamma_ran};

/// Struct for the chi-square distribution `X ~ ChiSq(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. Use.
/// ```
/// use ruststat::ChiSqDist;
/// let mut mychisq = ChiSqDist{nu:8.0};
/// println!("Probability density function f(4.5): {}", mychisq.pdf(4.5));
/// println!("Cumulative distribution function P(X<=4.5): {}", mychisq.cdf(4.5));
/// println!("99th percentile: {}", mychisq.per(0.99));
/// println!("Random draw: {}", mychisq.ran());
/// println!("Random vector: {:?}", mychisq.ranvec(5));
/// println!("Mean: {}", mychisq.mean());
/// println!("Variance: {}", mychisq.var());
/// println!("Standard deviation: {}", mychisq.sd());
/// ```
pub struct ChiSqDist {
    pub nu: f64,
}
impl ChiSqDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        chisq_pdf(x, self.nu)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        chisq_cdf(x, self.nu)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        chisq_per(x, self.nu)
    }
    pub fn ran(&mut self) -> f64 {
        chisq_ran(self.nu)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        chisq_ranvec(n, self.nu)
    }
    pub fn mean(&mut self) -> f64 {
        self.nu
    }
    pub fn var(&mut self) -> f64 {
        2.0 * self.nu
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ ChiSq(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`.
/// ```
/// use ruststat::chisq_pdf;
/// println!("Probability density function f(5.2): {}", chisq_pdf(5.2, 8.0));
/// ```
pub fn chisq_pdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function chisq_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gamma_pdf(x, nu/2.0,1.0/2.0)
}


/// Computes cumulative distribution function (cdf) for `X ~ ChiSq(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`.
/// ```
/// use ruststat::chisq_cdf;
/// println!("P(X<=5.2): {}", chisq_cdf(5.2, 8.0));
/// ```
pub fn chisq_cdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function chisq_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gamma_cdf(x, nu / 2.0, 0.5)
}


/// Computes a percentile for `X ~ ChiSq(nu)`
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::chisq_per;
/// println!("Percentile: {}", chisq_per(0.8, 8.0));
/// ```
pub fn chisq_per(p: f64, nu: f64) -> f64 {
    gamma_per(p, nu/2.0, 1.0/2.0)
}


/// Random draw from `X ~ ChiSq(nu)` distribution.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. Use
/// ```
/// use ruststat::chisq_ran;
/// println!("Random draw: {}", chisq_ran(8.0));
/// ```
pub fn chisq_ran(nu: f64) -> f64 {
    gamma_ran(nu/2.0, 1.0/2.0)
}


/// Save random draws from `X ~ ChiSq(nu)` into a `Vec`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ ChiSq(nu=8.0)`. Use
/// ```
/// use ruststat::chisq_ranvec;
/// println!("Random Vec: {:?}", chisq_ranvec(10, 8.0));
/// ```
pub fn chisq_ranvec(n: u64, nu: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(chisq_ran(nu));
    }
    xvec
}
