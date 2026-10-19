use crate::{beta_fn_ln, betai};
use crate::beta::{beta_per, beta_ran};

/// Struct for the F distribution `X ~ F(nu1, nu2)`.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use.
/// ```
/// use ruststat::FDist;
/// let mut myf = FDist{nu1:4.0, nu2:18.0};
/// println!("Probability density function f(2.5): {}", myf.pdf(2.5));
/// println!("Cumulative distribution function P(X<=2.5): {}", myf.cdf(2.5));
/// println!("99th percentile: {}", myf.per(0.99));
/// println!("Random draw: {}", myf.ran());
/// println!("Random vector: {:?}", myf.ranvec(5));
/// println!("Mean: {}", myf.mean());
/// println!("Variance: {}", myf.var());
/// println!("Standard deviation: {}", myf.sd());
/// ```
pub struct FDist {
    pub nu1: f64,
    pub nu2: f64,
}
impl FDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        f_pdf(x, self.nu1, self.nu2)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        f_cdf(x, self.nu1, self.nu2)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        f_per(x, self.nu1, self.nu2)
    }
    pub fn ran(&mut self) -> f64 {
        f_ran(self.nu1, self.nu2)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        f_ranvec(n, self.nu1, self.nu2)
    }
    pub fn mean(&mut self) -> f64 {
        self.nu2 / (self.nu2-2.0)
    }
    pub fn var(&mut self) -> f64 {
        2.0 * self.nu2.powi(2) * (self.nu1 + self.nu2 - 2.0) /
            (self.nu1 * (self.nu2 - 2.0).powi(2) * (self.nu2 - 4.0))
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ F(nu1, nu2)`.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use.
/// ```
/// use ruststat::f_pdf;
/// println!("Probability density function f(2.5): {}", f_pdf(2.5, 4.0, 18.0));
/// ```
pub fn f_pdf(x: f64, nu1: f64, nu2: f64) -> f64 {
    if nu1 <= 0.0 || nu2 <= 0.0 {
        println!("NAN produced. Error in function f_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (0.5*(nu1*(nu1*x).ln() + nu2*nu2.ln() - (nu1+nu2)*(nu1*x + nu2).ln())
        - x.ln() - beta_fn_ln(nu1/2.0, nu2/2.0)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ F(nu1, nu2)`.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use.
/// ```
/// use ruststat::f_cdf;
/// println!("P(X<=2.5): {}", f_cdf(2.5, 4.0, 18.0));
/// ```
pub fn f_cdf(x: f64, nu1: f64, nu2: f64) -> f64 {
    if nu1 <= 0.0 || nu2 <= 0.0 {
        println!("NAN produced. Error in function f_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    1.0 - betai(nu2 / (nu2 + nu1 * x), nu2 / 2.0, nu1 / 2.0)
}


/// Computes a percentile for `X ~ F(nu1, nu2)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::f_per;
/// println!("Percentile: {}", f_per(0.8, 4.0, 18.0));
/// ```
pub fn f_per(p: f64, nu1: f64, nu2: f64) -> f64 {
    nu2/nu1 * (1.0 / beta_per(1.0-p,nu2/2.0,nu1/2.0) - 1.0)
}


/// Random draw from `X ~ F(nu1, nu2)` distribution.
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(nu1=4.0, nu2=18.0)`. Use
/// ```
/// use ruststat::f_ran;
/// println!("Random draw: {}", f_ran(4.0, 18.0));
/// ```
pub fn f_ran(nu1: f64, nu2: f64) -> f64 {
    let x = beta_ran(nu1/2.0, nu2/2.0);
    nu2 * x / (nu1 * (1.0-x))
}

/// Save random draws from `X ~ F(nu1, nu2)` distribution into a `Vec`
///
/// # Parameters
/// - `nu1 > 0`
/// - `nu2 > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ F(alpha=4.0, beta=18.0)`. Use
/// ```
/// use ruststat::f_ranvec;
/// println!("Random Vec: {:?}", f_ranvec(10, 4.0, 18.0));
/// ```
pub fn f_ranvec(n: u64, nu1: f64, nu2: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(f_ran(nu1, nu2));
    }
    xvec
}
