// ==========================================
// ==========================================
// t Distribution
// ==========================================
// ==========================================

use std::f64::consts::PI;
use crate::{betai, betai_inv, gamma_ln};
use crate::gamma::gamma_ran;
use crate::normal::normal_ran;

/// Struct for Student's t distribution `X ~ t(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25)`. Use
/// ```
/// use ruststat::TDist;
/// let mut myt = TDist{nu:25.0};
/// println!("Probability density function f(2.2): {}", myt.pdf(2.2));
/// println!("Cumulative distribution function P(X<=2.2): {}", myt.cdf(2.2));
/// println!("99th percentile: {}", myt.per(0.99));
/// println!("Random draw: {}", myt.ran());
/// println!("Random vector: {:?}", myt.ranvec(5));
/// println!("Mean: {}", myt.mean());
/// println!("Variance: {}", myt.var());
/// println!("Standard deviation: {}", myt.sd());
/// ```
pub struct TDist {
    pub nu: f64,
}
impl TDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        t_pdf(x, self.nu)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        t_cdf(x, self.nu)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        t_per(p,self.nu)
    }
    pub fn ran(&mut self) -> f64 {
        t_ran(self.nu)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        t_ranvec(n, self.nu)
    }
    pub fn mean(&mut self) -> f64 {
        0.0
    }
    pub fn var(&mut self) -> f64 {
        if self.nu > 2.0 {
            self.nu / (self.nu - 2.0)
        } else {
            f64::NAN
        }
    }
    pub fn sd(&mut self) -> f64 {
        if self.nu > 2.0 {
            self.var().sqrt()
        } else {
            f64::NAN
        }
    }
}


/// Computes probability density function (pdf) for `X ~ t(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. Use.
/// ```
/// use ruststat::t_pdf;
/// println!("Probability density function f(2.2): {}", t_pdf(2.2, 25.0));
pub fn t_pdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function t_pdf");
        return f64::NAN;
    }
    (gamma_ln((nu+1.0)/2.0) - ((nu+1.0)/2.0)*(1.0 + x.powi(2)/nu).ln() -
        0.5*(nu * PI).ln() - gamma_ln(nu/2.0)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ t(nu)`.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ nu(nu=25.0)`. Use.
/// ```
/// use ruststat::t_cdf;
/// println!("P(X<=2.2): {}", t_cdf(2.2, 25.0));
/// ```
pub fn t_cdf(x: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        println!("NAN produced. Error in function t_cdf");
        return f64::NAN;
    }

    if x <= 0.0 {
        0.5 * betai(nu / (nu + x * x), nu / 2.0, 0.5)
    } else {
        1.0 - 0.5 * betai(nu / (nu + x * x), nu / 2.0, 0.5)
    }
}


/// Computes a percentile for `X ~ t(nu)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::t_per;
/// println!("Percentile: {}", t_per(0.8, 25.0));
/// ```
pub fn t_per(p: f64, nu: f64) -> f64 {
    let (a, x): (f64, f64);
    if p <= 0.5 {
        a = betai_inv(2.0 * p, nu / 2.0, 1.0 / 2.0);
        x = (nu / a - nu).sqrt();
        -x
    } else {
        a = betai_inv(2.0 * (1.0 - p), nu / 2.0, 1.0 / 2.0);
        x = (nu / a - nu).sqrt();
        x
    }
}


/// Random draw from `X ~ t(nu)` distribution.
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. Use
/// ```
/// use ruststat::t_ran;
/// println!("Random draw: {}", t_ran(25.0));
/// ```
pub fn t_ran(nu: f64) -> f64 {
    normal_ran(0.0, 1.0) *
        (nu / gamma_ran(nu/2.0, 1.0/2.0)).sqrt()
}


/// Save random draws from `X ~ t(nu)` distribution into a `Vec`
///
/// # Parameters
/// - `nu > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ t(nu=25.0)`. Use
/// ```
/// use ruststat::t_ranvec;
/// println!("Random Vec: {:?}", t_ranvec(10, 25.0));
/// ```
pub fn t_ranvec(n: u64, nu: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(t_ran(nu));
    }
    xvec
}
