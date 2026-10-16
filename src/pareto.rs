// ==========================================
// ==========================================
// Pareto (Type I) Distribution
// ==========================================
// ==========================================

use crate::{gamma, unif_ran};

/// Struct for the Pareto (type I) distribution `X ~ Pareto1(sigma, alpha)`.
///
/// # Parameters
/// - `sigma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= sigma`
/// # Example
/// Suppose `X ~ Pareto1(sigma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::Pareto1Dist;
/// let mut mydist = Pareto1Dist{sigma:2.0, alpha:0.5};
/// println!("Probability density function f(2.5): {}", mydist.pdf(2.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(2.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto1Dist {
    pub sigma: f64,
    pub alpha: f64,
}
impl Pareto1Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n, self.sigma, self.sigma, 1.0, self.alpha)
    }
    pub fn mean(&mut self) -> f64 {
        if self.alpha <= 1.0 { return f64::NAN; }
        self.sigma + self.sigma * gamma(self.alpha - 1.0) * gamma(2.0) / gamma(self.alpha)
    }
    pub fn var(&mut self) -> f64 {
        if self.alpha <= 2.0 { return f64::NAN; }
        self.sigma.powi(2) * gamma(self.alpha - 2.0) * gamma(3.0) / gamma(self.alpha) - (self.mean() - self.sigma).powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


// ==========================================
// ==========================================
// Pareto (Type II) Distribution
// ==========================================
// ==========================================

/// Struct for the Pareto (type II) distribution `X ~ Pareto2(mu, sigma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto2(mu=1.0, sigma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::Pareto2Dist;
/// let mut mydist = Pareto2Dist{mu: 1.0, sigma:2.0, alpha:0.5};
/// println!("Probability density function f(2.5): {}", mydist.pdf(2.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(2.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto2Dist {
    pub mu: f64,
    pub sigma: f64,
    pub alpha: f64,
}
impl Pareto2Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n,self.mu, self.sigma, 1.0, self.alpha)
    }
    pub fn mean(&mut self) -> f64 {
        if self.alpha <= 1.0 { return f64::NAN; }
        self.mu + self.sigma * gamma(self.alpha - 1.0) * gamma(2.0) / gamma(self.alpha)
    }
    pub fn var(&mut self) -> f64 {
        if self.alpha <= 2.0 { return f64::NAN; }
        self.sigma.powi(2) * gamma(self.alpha - 2.0) * gamma(3.0) / gamma(self.alpha) - (self.mean() - self.mu).powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


// ==========================================
// ==========================================
// Pareto (Type III) Distribution
// ==========================================
// ==========================================

/// Struct for the Pareto (type III) distribution `X ~ Pareto3(mu, sigma, gamma)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto3(mu=0.0, sigma=1.0, gamma=2.0)`. Use
/// ```
/// use ruststat::Pareto3Dist;
/// let mut mydist = Pareto3Dist{mu:0.0, sigma:1.0, gamma:2.0};
/// println!("Probability density function f(1.5): {}", mydist.pdf(1.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(1.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto3Dist {
    pub mu: f64,
    pub sigma: f64,
    pub gamma: f64,
}
impl Pareto3Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n,self.mu, self.sigma, self.gamma, 1.0)
    }
    pub fn mean(&mut self) -> f64 {
        if self.gamma >= 1.0 { return f64::NAN; }
        self.mu + self.sigma * gamma(1.0 - self.gamma) * gamma(1.0 + self.gamma) / gamma(1.0)
    }
    pub fn var(&mut self) -> f64 {
        if self.gamma >= 0.5 { return f64::NAN; }
        self.sigma.powi(2) * gamma(1.0 - self.gamma * 2.0) * gamma(1.0 + self.gamma * 2.0) / gamma(1.0) - (self.mean() - self.mu).powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


// ==========================================
// ==========================================
// Pareto (Type IV) Distribution
// ==========================================
// ==========================================

/// Struct for the Pareto (type IV) distribution `X ~ Pareto4(mu, sigma, gamma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::Pareto4Dist;
/// let mut mydist = Pareto4Dist{mu:0.0, sigma:1.0, gamma:2.0, alpha:0.5};
/// println!("Probability density function f(1.5): {}", mydist.pdf(1.5));
/// println!("Cumulative distribution function P(X<=110.0): {}", mydist.cdf(1.5));
/// println!("99th percentile: {}", mydist.per(0.99));
/// println!("Random draw: {}", mydist.ran());
/// println!("Random vector: {:?}", mydist.ranvec(5));
/// println!("Mean: {}", mydist.mean());
/// println!("Variance: {}", mydist.var());
/// println!("Standard deviation: {}", mydist.sd());
/// ```
pub struct Pareto4Dist {
    pub mu: f64,
    pub sigma: f64,
    pub gamma: f64,
    pub alpha: f64,
}
impl Pareto4Dist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        pareto4_pdf(x, self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        pareto4_cdf(x, self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn per(&mut self, x: f64) -> f64 {
        pareto4_per(x,self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn ran(&mut self) -> f64 {
        pareto4_ran(self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        pareto4_ranvec(n,self.mu, self.sigma, self.gamma, self.alpha)
    }
    pub fn mean(&mut self) -> f64 {
        if self.alpha <= self.gamma { return f64::NAN; }
        self.mu + self.sigma * gamma(self.alpha - self.gamma) * gamma(1.0 + self.gamma) / gamma(self.alpha)
    }
    pub fn var(&mut self) -> f64 {
        if self.alpha <= self.gamma * 2.0 { return f64::NAN; }
        self.sigma.powi(2) * gamma(self.alpha - self.gamma * 2.0) * gamma(1.0 + self.gamma * 2.0) / gamma(self.alpha) - (self.mean() - self.mu).powi(2)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability density function (pdf) for `X ~ Pareto(mu, sigma, gamma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use.
/// ```
/// use ruststat::pareto4_pdf;
/// println!("Probability density function f(1.5): {}", pareto4_pdf(1.5, 0.0, 1.0, 2.0, 0.5));
pub fn pareto4_pdf(x: f64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || gamma <= 0.0 || alpha <= 0.0 {
        println!("NAN produced. Error in function pareto_pdf");
        return f64::NAN;
    }
    if x < mu {
        return 0.0;
    }
    (alpha/(gamma*sigma))*((x-mu)/sigma).powf(1.0/gamma - 1.0)*
        (1.0 + ((x-mu)/sigma).powf(1.0/gamma)).powf(-alpha-1.0)
}


/// Computes cumulative distribution function (cdf) for `X ~ Pareto(mu, sigma, gamma, alpha)`.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use.
/// ```
/// use ruststat::pareto4_cdf;
/// println!("P(X<=1.5): {}", pareto4_cdf(1.5, 0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_cdf(x: f64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    if sigma <= 0.0 || gamma <= 0.0 || alpha <= 0.0 {
        println!("NAN produced. Error in function pareto_cdf");
        return f64::NAN;
    }
    if x < mu {
        return 0.0;
    }

    1.0 - (1.0 + ((x-mu)/sigma).powf(1.0/gamma)).powf(-alpha)
}


/// Computes a percentile for `X ~ Pareto(mu, sigma, gamma, alpha)`.
///
/// # Note
/// Determines the value of `x` such that `P(X <= x) = q`.
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::pareto4_per;
/// println!("Percentile: {}", pareto4_per(0.8, 0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_per(p: f64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    mu + sigma * ((1.0-p).powf(-1.0/alpha) - 1.0).powf(gamma)
}


/// Random draw from `X ~ Pareto(mu, sigma, gamma, alpha)` distribution.
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::pareto4_ran;
/// println!("Random draw: {}", pareto4_ran(0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_ran(mu: f64, sigma: f64, gamma: f64, alpha: f64) -> f64 {
    mu + sigma * ((1.0-unif_ran(0.0, 1.0)).powf(-1.0/alpha) - 1.0).powf(gamma)
}


/// Save random draws from `X ~ Pareto(mu, sigma, gamma, alpha)` distribution into a `Vec`
///
/// # Parameters
/// - `-infinity < mu < infinity`
/// - `sigma > 0`
/// - `gamma > 0`
/// - `alpha > 0`
/// # Support
/// - `x >= mu`
/// # Example
/// Suppose `X ~ Pareto4(mu=0.0, sigma=1.0, gamma=2.0, alpha=0.5)`. Use
/// ```
/// use ruststat::pareto4_ranvec;
/// println!("Random Vec: {:?}", pareto4_ranvec(10, 0.0, 1.0, 2.0, 0.5));
/// ```
pub fn pareto4_ranvec(n: u64, mu: f64, sigma: f64, gamma: f64, alpha: f64) -> Vec<f64> {
    let mut xvec: Vec<f64> = Vec::new();
    for _ in 0..n {
        xvec.push(pareto4_ran(mu, sigma, gamma, alpha));
    }
    xvec
}
