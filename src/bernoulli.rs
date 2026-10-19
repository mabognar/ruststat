use rand::random;

/// Struct for the Bernoulli distribution `X ~ Bernoulli(p)`.
/// # Parameters
/// - Probability of success: `p` (`0 <= p <= 1`)
/// # Support
/// - `x = 0, 1`
/// # Example
/// Suppose `X ~ Bernoulli(p=0.6)`. Use
/// ```
/// use ruststat::BernoulliDist;
/// let mut mybern = BernoulliDist{p:0.6};
/// println!("Probability mass function P(X=1): {}", mybern.pmf(1));
/// println!("Cumulative distribution function P(X<=0): {}", mybern.cdf(0));
/// println!("99th percentile: {}", mybern.per(0.99));
/// println!("Random draw: {}", mybern.ran());
/// println!("Random vector: {:?}", mybern.ranvec(5));
/// println!("Mean: {}", mybern.mean());
/// println!("Variance: {}", mybern.var());
/// println!("Standard deviation: {}", mybern.sd());
/// ```
pub struct BernoulliDist {
    pub p: f64,
}

impl BernoulliDist {
    pub fn pmf(&mut self, x: u64) -> f64 {
        bernoulli_pmf(x, self.p)
    }
    pub fn cdf(&mut self, x: u64) -> f64 {
        bernoulli_cdf(x, self.p)
    }
    pub fn per(&mut self, q: f64) -> u64 {
        bernoulli_per(q, self.p)
    }
    pub fn ran(&mut self) -> u64 {
        bernoulli_ran(self.p)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<u64> {
        bernoulli_ranvec(n, self.p)
    }
    pub fn mean(&mut self) -> f64 {
        self.p
    }
    pub fn var(&mut self) -> f64 {
        self.p * (1.0 - self.p)
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability mass function (pmf) for `X ~ Bernoulli(p)`.
/// # Parameters
/// - Probability of success: `p` (`0 <= p <= 1`)
/// # Support
/// - `x = 0, 1`
/// # Example
/// Suppose `X ~ Bernoulli(p=0.6)`. Use.
/// ```
/// use ruststat::bernoulli_pmf;
/// println!("Probability mass function P(X=1): {}", bernoulli_pmf(1, 0.6));
/// ```
pub fn bernoulli_pmf(x: u64, p: f64) -> f64 {
    if p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function bernoulli_pmf");
        return f64::NAN;
    }
    if x == 1 {
        p
    } else if x == 0 {
        1.0 - p
    } else {
        0.0
    }
}

/// Computes cumulative distribution function (cdf) for `X ~ Bernoulli(p)`.
/// # Parameters
/// - Probability of success: `p` (`0 <= p <= 1`)
/// # Support
/// - `x = 0, 1`
/// # Example
/// Suppose `X ~ Bernoulli(p=0.6)`. Use.
/// ```
/// use ruststat::bernoulli_cdf;
/// println!("P(X<=0): {}", bernoulli_cdf(0, 0.6));
/// ```
pub fn bernoulli_cdf(x: u64, p: f64) -> f64 {
    if p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function bernoulli_cdf");
        return f64::NAN;
    }
    if x == 0 {
        1.0 - p
    } else {
        1.0
    }
}

/// Computes a percentile for `X ~ Bernoulli(p)`.
/// # Note
/// Determines the smallest (integer) value of `x` such that `P(X <= x) >= q`.
/// # Parameters
/// - Probability of success: `p` (`0 <= p <= 1`)
/// # Support
/// - `x = 0, 1`
/// # Example
/// Suppose `X ~ Bernoulli(p=0.6)`. To find the 80th percentile, use `q=0.80` and
/// ```
/// use ruststat::bernoulli_per;
/// println!("Percentile: {}", bernoulli_per(0.8, 0.6));
/// ```
pub fn bernoulli_per(q: f64, p: f64) -> u64 {
    if p < 0.0 || p > 1.0 || q < 0.0 || q > 1.0 {
        println!("NAN produced. Error in function bernoulli_per");
        return f64::NAN as u64;
    }
    if q == 0.0 {
        return 0;
    }
    if q <= 1.0 - p {
        0
    } else {
        1
    }
}

/// Random draw from `X ~ Bernoulli(p)` distribution.
/// # Parameters
/// - Probability of success: `p` (`0 <= p <= 1`)
/// # Support
/// - `x = 0, 1`
/// # Example
/// Suppose `X ~ Bernoulli(p=0.6)`. Use
/// ```
/// use ruststat::bernoulli_ran;
/// println!("Random draw: {}", bernoulli_ran(0.6));
/// ```
pub fn bernoulli_ran(p: f64) -> u64 {
    if p < 0.0 || p > 1.0 {
        println!("Bad argument to bernoulli_ran");
        return f64::NAN as u64;
    }
    let u: f64 = random();
    if u <= p {
        1
    } else {
        0
    }
}

/// Save random draws from `X ~ Bernoulli(p)` distribution into a `Vec`
/// # Parameters
/// - Probability of success: `p` (`0 <= p <= 1`)
/// # Support
/// - `x = 0, 1`
/// # Example
/// Suppose `X ~ Bernoulli(p=0.6)`. Use
/// ```
/// use ruststat::bernoulli_ranvec;
/// println!("Random Vec: {:?}", bernoulli_ranvec(10, 0.6));
/// ```
pub fn bernoulli_ranvec(n: u64, p: f64) -> Vec<u64> {
    if p < 0.0 || p > 1.0 {
        println!("Bad argument to bernoulli_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN as u64] };
    }
    let mut xvec: Vec<u64> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let u: f64 = random();
        if u <= p {
            xvec.push(1);
        } else {
            xvec.push(0);
        }
    }
    xvec
}