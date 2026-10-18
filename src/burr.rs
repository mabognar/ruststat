// ==========================================
// ==========================================
// Burr (Type XII) Distribution
// ==========================================
// ==========================================

use rand::random;
use crate::beta_fn;

/// Struct for the Burr (Type XII) distribution `X ~ Burr(s, c, k)`.
/// # Parameters
/// - Scale: `s > 0`
/// - Shapes: `c > 0`, `k > 0`
/// # Support
/// - `x > 0`
/// # Example
/// Suppose `X ~ Burr(s=1.0, c=2.0, k=3.0)`. Use
/// ```
/// use ruststat::BurrDist;
/// let mut myburr = BurrDist{s:1.0, c:2.0, k:3.0};
/// println!("Probability density function f(1.5): {}", myburr.pdf(1.5));
/// println!("Cumulative distribution function P(X<=1.5): {}", myburr.cdf(1.5));
/// println!("99th percentile: {}", myburr.per(0.99));
/// println!("Random draw: {}", myburr.ran());
/// println!("Random vector: {:?}", myburr.ranvec(5));
/// println!("Mean: {}", myburr.mean());
/// println!("Variance: {}", myburr.var());
/// println!("Standard deviation: {}", myburr.sd());
/// ```
pub struct BurrDist {
    pub s: f64,
    pub c: f64,
    pub k: f64,
}

impl BurrDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        burr_pdf(x, self.s, self.c, self.k)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        burr_cdf(x, self.s, self.c, self.k)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        burr_per(p, self.s, self.c, self.k)
    }
    pub fn ran(&mut self) -> f64 {
        burr_ran(self.s, self.c, self.k)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        burr_ranvec(n, self.s, self.c, self.k)
    }
    pub fn mean(&mut self) -> f64 {
        if self.k > 1.0 / self.c {
            self.s * self.k * beta_fn(1.0 + 1.0 / self.c, self.k - 1.0 / self.c)
        } else {
            f64::NAN
        }
    }
    pub fn var(&mut self) -> f64 {
        if self.k > 2.0 / self.c {
            let m1 = self.mean();
            let m2 = self.s.powi(2) * self.k * beta_fn(1.0 + 2.0 / self.c, self.k - 2.0 / self.c);
            m2 - m1.powi(2)
        } else {
            f64::NAN
        }
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ Burr(s, c, k)`.
/// # Parameters
/// - Scale: `s > 0`
/// - Shapes: `c > 0`, `k > 0`
/// # Support
/// - `x > 0`
pub fn burr_pdf(x: f64, s: f64, c: f64, k: f64) -> f64 {
    if s <= 0.0 || c <= 0.0 || k <= 0.0 {
        println!("NAN produced. Error in function burr_pdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    (c * k / s) * (x / s).powf(c - 1.0) * (1.0 + (x / s).powf(c)).powf(-(k + 1.0))
}

/// Computes cumulative distribution function (cdf) for `X ~ Burr(s, c, k)`.
pub fn burr_cdf(x: f64, s: f64, c: f64, k: f64) -> f64 {
    if s <= 0.0 || c <= 0.0 || k <= 0.0 {
        println!("NAN produced. Error in function burr_cdf");
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    1.0 - (1.0 + (x / s).powf(c)).powf(-k)
}

/// Computes a percentile for `X ~ Burr(s, c, k)`.
pub fn burr_per(p: f64, s: f64, c: f64, k: f64) -> f64 {
    if s <= 0.0 || c <= 0.0 || k <= 0.0 || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function burr_per");
        return f64::NAN;
    }
    s * ((1.0 - p).powf(-1.0 / k) - 1.0).powf(1.0 / c)
}

/// Random draw from `X ~ Burr(s, c, k)` distribution.
pub fn burr_ran(s: f64, c: f64, k: f64) -> f64 {
    if s <= 0.0 || c <= 0.0 || k <= 0.0 {
        println!("Bad argument to burr_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    s * ((1.0 - u).powf(-1.0 / k) - 1.0).powf(1.0 / c)
}

/// Save random draws from `X ~ Burr(s, c, k)` distribution into a `Vec`
pub fn burr_ranvec(n: u64, s: f64, c: f64, k: f64) -> Vec<f64> {
    if s <= 0.0 || c <= 0.0 || k <= 0.0 {
        println!("Bad argument to burr_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        xvec.push(s * ((1.0 - u).powf(-1.0 / k) - 1.0).powf(1.0 / c));
    }
    xvec
}