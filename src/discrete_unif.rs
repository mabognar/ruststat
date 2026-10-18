// ==========================================
// ==========================================
// Discrete Uniform Distribution
// ==========================================
// ==========================================

use rand::random;

/// Struct for the Discrete Uniform distribution `X ~ DiscreteUnif(a, b)`.
/// # Parameters
/// - Lower limit: `a` (inclusive)
/// - Upper limit: `b` (inclusive, `a <= b`)
/// # Support
/// - `x = a, a+1, ..., b`
/// # Example
/// Suppose `X ~ DiscreteUnif(a=1, b=5)`. Use
/// ```
/// use ruststat::DiscreteUnifDist;
/// let mut mydunif = DiscreteUnifDist{a:1, b:5};
/// println!("Probability mass function P(X=3): {}", mydunif.pmf(3));
/// println!("Cumulative distribution function P(X<=3): {}", mydunif.cdf(3));
/// println!("99th percentile: {}", mydunif.per(0.99));
/// println!("Random draw: {}", mydunif.ran());
/// println!("Random vector: {:?}", mydunif.ranvec(5));
/// println!("Mean: {}", mydunif.mean());
/// println!("Variance: {}", mydunif.var());
/// println!("Standard deviation: {}", mydunif.sd());
/// ```
pub struct DiscreteUnifDist {
    pub a: i64,
    pub b: i64,
}

impl DiscreteUnifDist {
    pub fn pmf(&mut self, x: i64) -> f64 {
        discrete_unif_pmf(x, self.a, self.b)
    }
    pub fn cdf(&mut self, x: i64) -> f64 {
        discrete_unif_cdf(x, self.a, self.b)
    }
    pub fn per(&mut self, p: f64) -> i64 {
        discrete_unif_per(p, self.a, self.b)
    }
    pub fn ran(&mut self) -> i64 {
        discrete_unif_ran(self.a, self.b)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<i64> {
        discrete_unif_ranvec(n, self.a, self.b)
    }
    pub fn mean(&mut self) -> f64 {
        (self.a as f64 + self.b as f64) / 2.0
    }
    pub fn var(&mut self) -> f64 {
        let n = (self.b - self.a + 1) as f64;
        (n.powi(2) - 1.0) / 12.0
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability mass function (pmf) for `X ~ DiscreteUnif(a, b)`.
/// # Parameters
/// - Lower limit: `a` (inclusive)
/// - Upper limit: `b` (inclusive, `a <= b`)
/// # Support
/// - `x = a, a+1, ..., b`
/// # Example
/// Suppose `X ~ DiscreteUnif(a=1, b=5)`. Use.
/// ```
/// use ruststat::discrete_unif_pmf;
/// println!("Probability mass function P(X=3): {}", discrete_unif_pmf(3, 1, 5));
/// ```
pub fn discrete_unif_pmf(x: i64, a: i64, b: i64) -> f64 {
    if a > b {
        println!("NAN produced. Error in function discrete_unif_pmf");
        return f64::NAN;
    }
    if x < a || x > b {
        return 0.0;
    }
    1.0 / ((b - a + 1) as f64)
}

/// Computes cumulative distribution function (cdf) for `X ~ DiscreteUnif(a, b)`.
/// # Parameters
/// - Lower limit: `a` (inclusive)
/// - Upper limit: `b` (inclusive, `a <= b`)
/// # Support
/// - `x = a, a+1, ..., b`
/// # Example
/// Suppose `X ~ DiscreteUnif(a=1, b=5)`. Use.
/// ```
/// use ruststat::discrete_unif_cdf;
/// println!("P(X<=3): {}", discrete_unif_cdf(3, 1, 5));
/// ```
pub fn discrete_unif_cdf(x: i64, a: i64, b: i64) -> f64 {
    if a > b {
        println!("NAN produced. Error in function discrete_unif_cdf");
        return f64::NAN;
    }
    if x < a {
        return 0.0;
    }
    if x >= b {
        return 1.0;
    }
    ((x - a + 1) as f64) / ((b - a + 1) as f64)
}

/// Computes a percentile for `X ~ DiscreteUnif(a, b)`.
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - Lower limit: `a` (inclusive)
/// - Upper limit: `b` (inclusive, `a <= b`)
/// # Support
/// - `x = a, a+1, ..., b`
/// # Example
/// Suppose `X ~ DiscreteUnif(a=1, b=5)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::discrete_unif_per;
/// println!("Percentile: {}", discrete_unif_per(0.8, 1, 5));
/// ```
pub fn discrete_unif_per(q: f64, a: i64, b: i64) -> i64 {
    if a > b || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function discrete_unif_per");
        // Because we return i64, returning a sentinel or panicking is standard. f64::NAN cast to int is normally 0.
        return f64::NAN as i64;
    }
    if q == 0.0 {
        return a;
    }
    if q == 1.0 {
        return b;
    }
    let n = (b - a + 1) as f64;
    let mut x = a + (q * n).ceil() as i64 - 1;
    if x < a { x = a; }
    if x > b { x = b; }
    x
}

/// Random draw from `X ~ DiscreteUnif(a, b)` distribution.
/// # Parameters
/// - Lower limit: `a` (inclusive)
/// - Upper limit: `b` (inclusive, `a <= b`)
/// # Support
/// - `x = a, a+1, ..., b`
/// # Example
/// Suppose `X ~ DiscreteUnif(a=1, b=5)`. Use
/// ```
/// use ruststat::discrete_unif_ran;
/// println!("Random draw: {}", discrete_unif_ran(1, 5));
/// ```
pub fn discrete_unif_ran(a: i64, b: i64) -> i64 {
    if a > b {
        println!("Bad argument to discrete_unif_ran");
        return f64::NAN as i64;
    }
    let u: f64 = random();
    let n = (b - a + 1) as f64;
    let mut val = a + (u * n).floor() as i64;
    if val > b { val = b; } // Safety limit in case random() gives exactly 1.0
    val
}

/// Save random draws from `X ~ DiscreteUnif(a, b)` distribution into a `Vec`
/// # Parameters
/// - Lower limit: `a` (inclusive)
/// - Upper limit: `b` (inclusive, `a <= b`)
/// # Support
/// - `x = a, a+1, ..., b`
/// # Example
/// Suppose `X ~ DiscreteUnif(a=1, b=5)`. Use
/// ```
/// use ruststat::discrete_unif_ranvec;
/// println!("Random Vec: {:?}", discrete_unif_ranvec(10, 1, 5));
/// ```
pub fn discrete_unif_ranvec(n: u64, a: i64, b: i64) -> Vec<i64> {
    if a > b {
        println!("Bad argument to discrete_unif_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN as i64] };
    }
    let mut xvec: Vec<i64> = Vec::with_capacity(n as usize);
    let num_vals = (b - a + 1) as f64;
    for _ in 0..n {
        let u: f64 = random();
        let mut val = a + (u * num_vals).floor() as i64;
        if val > b { val = b; }
        xvec.push(val);
    }
    xvec
}