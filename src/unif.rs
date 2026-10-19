use rand::random;

/// Struct for continuous uniform distribution `X ~ Unif(a,b)`.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use
/// ```
/// use ruststat::UnifDist;
/// let mut myunif = UnifDist{a:0.0, b:1.0};
/// println!("Probability density function f(0.4): {}", myunif.pdf(0.4));
/// println!("Cumulative distribution function P(X<=0.4): {}", myunif.cdf(0.4));
/// println!("99th percentile: {}", myunif.per(0.99));
/// println!("Random draw: {}", myunif.ran());
/// println!("Random vector: {:?}", myunif.ranvec(5));
/// println!("Mean: {}", myunif.mean());
/// println!("Variance: {}", myunif.var());
/// println!("Standard deviation: {}", myunif.sd());
/// ```
pub struct UnifDist {
    pub a: f64,
    pub b: f64,
}
impl UnifDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        unif_pdf(x, self.a, self.b)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        unif_cdf(x, self.a, self.b)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        unif_per(x,self.a, self.b)
    }
    pub fn ran(&mut self) -> f64 {
        unif_ran(self.a, self.b)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        unif_ranvec(n, self.a, self.b)
    }
    pub fn mean(&mut self) -> f64 {
        (self.a + self.b) / 2.0
    }
    pub fn var(&mut self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Unif(a,b)`.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use.
/// ```
/// use ruststat::unif_pdf;
/// println!("Probability density function f(0.4): {}", unif_pdf(0.4, 0.0, 1.0));
pub fn unif_pdf(x: f64, a: f64, b: f64) -> f64 {
    if a >= b {
        println!("NAN produced. Error in function unif_pdf");
        return f64::NAN;
    }
    if x < a || x > b {
        return 0.0;
    }
    1.0 / (b - a)
}


/// Computes cumulative distribution function (cdf) for `X ~ Unif(a,b)`.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use.
/// ```
/// use ruststat::unif_cdf;
/// println!("P(X<=0.4): {}", unif_cdf(0.4, 0.0, 1.0));
/// ```
pub fn unif_cdf(x: f64, a: f64, b: f64) -> f64 {
    if a >= b {
        println!("NAN produced. Error in function unif_cdf");
        return f64::NAN;
    }
    if x < a {
        return 0.0;
    }
    if x > b {
        return 1.0;
    }

    (x - a) / (b - a)
}


/// Computes a percentile for `X ~ Unif(a,b)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::unif_per;
/// println!("Percentile: {}", unif_per(0.8, 0.0, 1.0));
/// ```
pub fn unif_per(p: f64, a: f64, b: f64) -> f64 {
    if a >= b || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function unif_per");
        return f64::NAN;
    }

    a + p * (b - a)
}

/// Random draw from `X ~ Unif(a,b)` distribution.
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use
/// ```
/// use ruststat::unif_ran;
/// println!("Random draw: {}", unif_ran(0.0, 1.0));
/// ```
pub fn unif_ran(a: f64, b: f64) -> f64 {
    let u: f64;
    u = random();
    a + u * (b-a)
}


/// Save random draws from `X ~ Unif(a,b)` distribution into a `Vec`
///
/// # Parameters
/// - `-infinity < a < infinity`
/// - `-infinity < b < infinity`
/// - `a < b`
/// # Support
/// - `a < x < b`
/// # Example
/// Suppose `X ~ Unif(a=0.0, b=1.0)`. Use
/// ```
/// use ruststat::unif_ranvec;
/// println!("Random Vec: {:?}", unif_ranvec(10, 0.0, 1.0));
/// ```
pub fn unif_ranvec(n: u64, a: f64, b: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);

    for _ in 0..n {
        xvec.push(unif_ran(a, b));
    }
    xvec
}
