use crate::factln_u;
use crate::unif::unif_ran;

/// Struct for the negative hypergeometric distribution `X ~ NHG(r,N,M)` where
/// `X =` the number of failures prior to observing the `r`th success.
///
/// # Parameters
/// - Target number of successes: `r = 1,2,3,...,M`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,N-M`
/// # Example
/// Suppose `X ~ NHG(r=10, N=100, M=50)`. Use
/// ```
/// use ruststat::NHGDist;
/// let mut mynhg = NHGDist{r:10, N:100, M:50};
/// println!("P(X=6): {}", mynhg.pmf(6));
/// println!("P(X<=6): {}", mynhg.cdf(6));
/// println!("99th percentile: {}", mynhg.per(0.99));
/// println!("Random draw: {}", mynhg.ran());
/// println!("Random vector: {:?}", mynhg.ranvec(5));
/// println!("Mean: {}", mynhg.mean());
/// println!("Variance: {}", mynhg.var());
/// println!("Standard deviation: {}", mynhg.sd());
/// ```
#[allow(non_snake_case)]
pub struct NHGDist {
    pub r: u64,
    pub N: u64,
    pub M: u64,
}

impl NHGDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        nhg_pmf(x, self.r, self.N, self.M)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        nhg_cdf(x, self.r, self.N, self.M)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        nhg_per(q, self.r, self.N, self.M)
    }
    pub fn ran(&mut self) -> u64 {
        nhg_ran(self.r, self.N, self.M)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        nhg_ranvec(n, self.r, self.N, self.M)
    }
    pub fn mean(&mut self) -> f64 {
        (self.r as f64) * (self.N as f64 - self.M as f64) / (self.M as f64 + 1.0)
    }
    pub fn var(&mut self) -> f64 {
        let r_f = self.r as f64;
        let n_f = self.N as f64;
        let m_f = self.M as f64;

        r_f * (n_f - m_f) * (n_f + 1.0) * (m_f + 1.0 - r_f) /
            ((m_f + 1.0).powi(2) * (m_f + 2.0))
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}


/// Computes probability mass function (pmf) for `X ~ NHG(r,N,M)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - Target number of successes: `r = 1,2,3,...,M`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,N-M`
/// # Example
/// Suppose `X ~ NHG(r=10,N=100,M=50)`. To compute `P(X = 7)`, use
/// ```
/// use ruststat::nhg_pmf;
/// println!("P(X=x): {}", nhg_pmf(7, 10, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn nhg_pmf(x: u64, r: u64, N: u64, M: u64) -> f64 {
    if r < 1 || N < 1 || M > N || r > M {
        println!("NAN produced. Error in function nhg_pmf");
        return f64::NAN;
    }
    if x > (N - M) {
        return 0.0;
    }
    if M == N {
        return if x == 0 { 1.0 } else { 0.0 };
    }

    (factln_u(x+r-1) - factln_u(x) - factln_u(r-1) +
        factln_u(N-x-r) - factln_u(M-r) - factln_u(N-M-x) -
        factln_u(N) + factln_u(M) + factln_u(N-M)).exp()
}


/// Computes cumulative distribution function (cdf) for `X ~ NHG(r,N,M)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - Target number of successes: `r = 1,2,3,...,M`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,N-M`
/// # Example
/// Suppose `X ~ NHG(r=10,N=100,M=50)`. To compute `P(X <= 7)`, use
/// ```
/// use ruststat::nhg_cdf;
/// println!("P(X<=x): {}", nhg_cdf(7, 10, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn nhg_cdf(x: u64, r: u64, N: u64, M: u64) -> f64 {
    if r < 1 || N < 1 || M > N || r > M {
        println!("NAN produced. Error in function nhg_cdf");
        return f64::NAN;
    }

    if x >= (N - M) {
        return 1.0;
    }

    let mut sum = 0.0;
    for i in 0..=x {
        sum += nhg_pmf(i, r, N, M);
    }
    sum
}


/// Computes a percentile for `X ~ NHG(r,N,M)`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - Target number of successes: `r = 1,2,3,...,M`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Support
/// - `x = 0,1,2,...,N-M`
/// # Example
/// Suppose `X ~ NHG(r=10, N=100, M=50)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::nhg_per;
/// println!("Percentile: {}", nhg_per(0.8, 10, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn nhg_per(q: f64, r: u64, N: u64, M: u64) -> u64 {
    if r < 1 || N < 1 || M > N || r > M || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function nhg_per");
        return f64::NAN as u64;
    }

    let max_x = N - M;
    if q == 0.0 { return 0; }
    if q == 1.0 { return max_x; }

    let mut x = 0;
    let mut sum = nhg_pmf(x, r, N, M);

    while sum < q && x < max_x {
        x += 1;
        sum += nhg_pmf(x, r, N, M);
    }

    x
}

/// Random draw from `X ~ NHG(r, N, M)` distribution
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - Target number of successes: `r = 1,2,3,...,M`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Example
/// Suppose `X ~ NHG(r=10, N=100, M=50)`. Use
/// ```
/// use ruststat::nhg_ran;
/// println!("Random draw: {}", nhg_ran(10, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn nhg_ran(r: u64, N: u64, M: u64) -> u64 {
    if r < 1 || N < 1 || M > N || r > M {
        println!("NAN produced. Error in function nhg_ran");
        return f64::NAN as u64;
    }

    let u = unif_ran(0.0, 1.0);
    let mut x = 0;

    while nhg_cdf(x, r, N, M) < u && x < (N - M) {
        x += 1;
    }

    x
}

/// Save random draws from `X ~ NHG(r,N,M)` distribution into a `Vec`
/// where `X =` the number of failures prior to the `r`th success.
///
/// # Parameters
/// - Target number of successes: `r = 1,2,3,...,M`
/// - Population size: `N = 1,2,3,...`
/// - Number of successes in population: `M = 1,2,3,...,N`
/// # Example
/// Suppose `X ~ NHG(r=10, N=100, M=50)`. Use
/// ```
/// use ruststat::nhg_ranvec;
/// println!("Random Vec: {:?}", nhg_ranvec(10, 10, 100, 50));
/// ```
#[allow(non_snake_case)]
pub fn nhg_ranvec(nn: u64, r: u64, N: u64, M: u64) -> Vec<u64> {
    let mut xvec: Vec<u64> = Vec::new();
    for _ in 0..nn {
        xvec.push(nhg_ran(r,N,M));
    }
    xvec
}