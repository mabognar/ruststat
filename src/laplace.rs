use rand::random;

/// Struct for the Laplace distribution `X ~ Laplace(mu, b)`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `b > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Laplace(mu=0.0, b=2.0)`. Use
/// ```
/// use ruststat::LaplaceDist;
/// let mut mylaplace = LaplaceDist{mu:0.0, b:2.0};
/// println!("Probability density function f(1.5): {}", mylaplace.pdf(1.5));
/// println!("Cumulative distribution function P(X<=1.5): {}", mylaplace.cdf(1.5));
/// println!("99th percentile: {}", mylaplace.per(0.99));
/// println!("Random draw: {}", mylaplace.ran());
/// println!("Random vector: {:?}", mylaplace.ranvec(5));
/// println!("Mean: {}", mylaplace.mean());
/// println!("Variance: {}", mylaplace.var());
/// println!("Standard deviation: {}", mylaplace.sd());
/// ```
pub struct LaplaceDist {
    pub mu: f64,
    pub b: f64,
}

impl LaplaceDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        laplace_pdf(x, self.mu, self.b)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        laplace_cdf(x, self.mu, self.b)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        laplace_per(p, self.mu, self.b)
    }
    pub fn ran(&mut self) -> f64 {
        laplace_ran(self.mu, self.b)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        laplace_ranvec(n, self.mu, self.b)
    }
    pub fn mean(&mut self) -> f64 {
        self.mu
    }
    pub fn var(&mut self) -> f64 {
        2.0 * self.b.powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ Laplace(mu, b)`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `b > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Laplace(mu=0.0, b=2.0)`. Use.
/// ```
/// use ruststat::laplace_pdf;
/// println!("Probability density function f(1.5): {}", laplace_pdf(1.5, 0.0, 2.0));
/// ```
pub fn laplace_pdf(x: f64, mu: f64, b: f64) -> f64 {
    if b <= 0.0 {
        println!("NAN produced. Error in function laplace_pdf");
        return f64::NAN;
    }
    1.0 / (2.0 * b) * (-(x - mu).abs() / b).exp()
}

/// Computes cumulative distribution function (cdf) for `X ~ Laplace(mu, b)`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `b > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Laplace(mu=0.0, b=2.0)`. Use.
/// ```
/// use ruststat::laplace_cdf;
/// println!("P(X<=1.5): {}", laplace_cdf(1.5, 0.0, 2.0));
/// ```
pub fn laplace_cdf(x: f64, mu: f64, b: f64) -> f64 {
    if b <= 0.0 {
        println!("NAN produced. Error in function laplace_cdf");
        return f64::NAN;
    }
    if x < mu {
        0.5 * ((x - mu) / b).exp()
    } else {
        1.0 - 0.5 * (-(x - mu) / b).exp()
    }
}

/// Computes a percentile for `X ~ Laplace(mu, b)`.
/// # Note
/// Determines the value of `x` such that `P(X <= x) = p`.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `b > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Laplace(mu=0.0, b=2.0)`. To find the 80th percentile, use `p=0.80` and
/// ```
/// use ruststat::laplace_per;
/// println!("Percentile: {}", laplace_per(0.8, 0.0, 2.0));
/// ```
pub fn laplace_per(p: f64, mu: f64, b: f64) -> f64 {
    if b <= 0.0 || p <= 0.0 || p >= 1.0 {
        println!("NAN produced. Error in function laplace_per");
        return f64::NAN;
    }
    if p < 0.5 {
        mu + b * (2.0 * p).ln()
    } else {
        mu - b * (2.0 * (1.0 - p)).ln()
    }
}

/// Random draw from `X ~ Laplace(mu, b)` distribution.
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `b > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Laplace(mu=0.0, b=2.0)`. Use
/// ```
/// use ruststat::laplace_ran;
/// println!("Random draw: {}", laplace_ran(0.0, 2.0));
/// ```
pub fn laplace_ran(mu: f64, b: f64) -> f64 {
    if b <= 0.0 {
        println!("Bad argument to laplace_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    if u < 0.5 {
        mu + b * (2.0 * u).ln()
    } else {
        mu - b * (2.0 * (1.0 - u)).ln()
    }
}

/// Save random draws from `X ~ Laplace(mu, b)` distribution into a `Vec`
/// # Parameters
/// - Location: `-infinity < mu < infinity`
/// - Scale: `b > 0`
/// # Support
/// - `-infinity < x < infinity`
/// # Example
/// Suppose `X ~ Laplace(mu=0.0, b=2.0)`. Use
/// ```
/// use ruststat::laplace_ranvec;
/// println!("Random Vec: {:?}", laplace_ranvec(10, 0.0, 2.0));
/// ```
pub fn laplace_ranvec(n: u64, mu: f64, b: f64) -> Vec<f64> {
    if b <= 0.0 {
        println!("Bad argument to laplace_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        if u < 0.5 {
            xvec.push(mu + b * (2.0 * u).ln());
        } else {
            xvec.push(mu - b * (2.0 * (1.0 - u)).ln());
        }
    }
    xvec
}