use crate::{factln_u};
use crate::unif::unif_ran;

/// Struct for the hypergeometric distribution `X ~ HG(n,N,M)`.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. Use
/// ```
/// use ruststat::HGDist;
/// let mut myhg = HGDist{n:20, N:100, M:50};
/// println!("P(X=6): {}", myhg.pmf(6));
/// println!("P(X<=6): {}", myhg.cdf(6));
/// println!("99th percentile: {}", myhg.per(0.99));
/// println!("Random draw: {}", myhg.ran());
/// println!("Random vector: {:?}", myhg.ranvec(5));
/// println!("Mean: {}", myhg.mean());
/// println!("Variance: {}", myhg.var());
/// println!("Standard deviation: {}", myhg.sd());
/// ```
#[allow(non_snake_case)]
pub struct HGDist {
    pub n: u64,
    pub N: u64,
    pub M: u64,
}
impl HGDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        hg_pmf(x, self.n, self.N, self.M)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        hg_cdf(x, self.n, self.N, self.M)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        hg_per(q, self.n, self.N, self.M)
    }
    pub fn ran(&mut self) -> u64 {
        hg_ran(self.n, self.N, self.M)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        hg_ranvec(n, self.n, self.N, self.M)
    }
    pub fn mean(&mut self) -> f64 {
        (self.n as f64) * (self.M as f64) / (self.N as f64)
    }
    pub fn var(&mut self) -> f64 {
        (self.n as f64) *
            (self.M as f64) / (self.N as f64) *
            (1.0 - (self.M as f64) / (self.N as f64)) *
            (self.N as f64 - self.n as f64) / (self.N as f64 - 1.0)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ HG(n,N,M)`
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20,N=100,M=50)`. To compute `P(X = 7)`, use
/// ```
/// use ruststat::hg_pmf;
/// println!("P(X=x): {}", hg_pmf(7, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_pmf(x: u64, n: u64, N: u64, M: u64) -> f64 {
    if  n < 1 || N < 1 || M > N || n > N {
        println!("NAN produced. Error in function hg_pmf");
        return f64::NAN;
    }
    if x > n || x > M || (n-x) > (N-M) {
        return 0.0;
    }

    if M == 0 {
        return if x == 0 { 1.0 } else { 0.0 };
    }
    if M == N {
        return if x == n { 1.0 } else { 0.0 };
    }

    (factln_u(M) - factln_u(x) - factln_u(M-x) +
        factln_u(N-M) - factln_u(n-x) - factln_u(N-M-n+x) -
        factln_u(N) + factln_u(n) + factln_u(N-n)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ HG(n,N,M)`
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20,N=100,M=50)`. To compute `P(X <= 7)`, use
/// ```
/// use ruststat::hg_cdf;
/// println!("P(X<=x): {}", hg_cdf(7, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_cdf(x: u64, n: u64, N: u64, M: u64) -> f64 {
    if n < 1 || N < 1 || M > N || n > N {
        println!("NAN produced. Error in function hg_cdf");
        return f64::NAN;
    }

    if x >= n || x >= M {
        return 1.0;
    }

    if (n - x) > (N - M) {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..=x {
        sum += hg_pmf(i, n, N, M);
    }
    sum
}


/// Computes a percentile for `X ~ HG(n,N,M)`
/// where `X =` the number of successes.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,n`
/// - `x <= M`
/// - `n-x <= N-M`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::hg_per;
/// println!("Percentile: {}", hg_per(0.8, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_per(q: f64, n: u64, N: u64, M: u64) -> u64 {
    if n < 1 || N < 1 || M > N || n > N || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function hg_per");
        return f64::NAN as u64;
    }

    let min_x = if n > N - M { n - (N - M) } else { 0 };
    let max_x = n.min(M);

    if q == 0.0 { return min_x; }
    if q == 1.0 { return max_x; }

    let mut x = min_x;
    let mut sum = hg_pmf(x, n, N, M);

    while sum < q && x < max_x {
        x += 1;
        sum += hg_pmf(x, n, N, M);
    }

    x
}

/// Random draw from `X ~ HG(n, N, M)` distribution
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. Use
/// ```
/// use ruststat::hg_ran;
/// println!("Random draw: {}", hg_ran(20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_ran(n: u64, N: u64, M: u64) -> u64 {
    if n < 1 || N < 1 || M > N || n > N {
        println!("NAN produced. Error in function hg_ran");
        return f64::NAN as u64;
    }

    let u = unif_ran(0.0, 1.0);

    let mut x = if n > N - M { n - (N - M) } else { 0 };

    while hg_cdf(x, n, N, M) < u {
        x += 1;
    }

    x
}

/// Save random draws from `X ~ HG(n,N,M)` distribution into a `Vec`
/// where `X =` the number of successes.
///
/// # Parameters
/// - Sample size: `n = 1,2,3,...,N`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Example
/// Suppose `X ~ HG(n=20, N=100, M=50)`. Use
/// ```
/// use ruststat::hg_ranvec;
/// println!("Random Vec: {:?}", hg_ranvec(10, 20, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn hg_ranvec(nn: u64, n: u64, N: u64, M: u64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(hg_ran(n,N,M));
    }
    xvec
}
