// ==========================================
// ==========================================
// Exponential Distribution
// ==========================================
// ==========================================

use rand::random;

/// Struct for the exponential distribution `X ~ Exp(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=3.5)`. Use.
/// ```
/// use ruststat::ExpDist;
/// let mut myexp = ExpDist{lambda:3.5};
/// println!("Probability density function f(1.2): {}", myexp.pdf(1.2));
/// println!("Cumulative distribution function P(X<=1.2): {}", myexp.cdf(1.2));
/// println!("99th percentile: {}", myexp.per(0.99));
/// println!("Random draw: {}", myexp.ran());
/// println!("Random vector: {:?}", myexp.ranvec(5));
/// println!("Mean: {}", myexp.mean());
/// println!("Variance: {}", myexp.var());
/// println!("Standard deviation: {}", myexp.sd());
/// ```
pub struct ExpDist {
    pub lambda: f64,
}
impl ExpDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        exp_pdf(x, self.lambda)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        exp_cdf(x, self.lambda)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        exp_per(x, self.lambda)
    }
    pub fn ran(&mut self) -> f64 {
        exp_ran(self.lambda)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        exp_ranvec(n, self.lambda)
    }
    pub fn mean(&mut self) -> f64 {
        1.0 / self.lambda
    }
    pub fn var(&mut self) -> f64 {
        1.0 / self.lambda.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Exp(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`.
/// ```
/// use ruststat::exp_pdf;
/// println!("Probability density function f(0.2): {}", exp_pdf(0.2, 8.0));
/// ```
pub fn exp_pdf(x: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function exp_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    lambda * (-lambda*x).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ Exp(lambda)`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`.
/// ```
/// use ruststat::exp_cdf;
/// println!("P(X<=0.8): {}", exp_cdf(0.8, 8.0));
/// ```
pub fn exp_cdf(x: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        println!("NAN produced. Error in function exp_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }

    // Exact closed-form solution
    1.0 - (-lambda * x).exp()
}

/// Computes a percentile for `X ~ Exp(lambda)`
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::exp_per;
/// println!("Percentile: {}", exp_per(0.8, 8.0));
/// ```
pub fn exp_per(p: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function exp_per");
        return f64::NAN;
    }

    -(1.0-p).ln() / lambda
}

/// Random draw from `X ~ Exp(lambda)` distribution.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`. Use
/// ```
/// use ruststat::exp_ran;
/// println!("Random draw: {}", exp_ran(8.0));
/// ```
pub fn exp_ran(lambda: f64) -> f64 {
    let u: f64;
    u = random();
    -(1.0-u).ln()/lambda
}


/// Save random draws from `X ~ Exp(lambda)` into a `Vec`.
///
/// # Parameters
/// - `lambda > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Exp(lambda=8.0)`. Use
/// ```
/// use ruststat::exp_ranvec;
/// println!("Random vector: {:?}", exp_ranvec(10, 8.0));
/// ```
pub fn exp_ranvec(n: u64, lambda: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(exp_ran(lambda));
    }
    xvec
}
