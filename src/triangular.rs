// ==========================================
// ==========================================
// Triangular Distribution
// ==========================================
// ==========================================

use rand::random;

/// Struct for the Triangular distribution `X ~ Triangular(a, b, c)`.
/// # Parameters
/// - Lower limit: `a`
/// - Upper limit: `b` (`a < b`)
/// - Mode: `c` (`a <= c <= b`)
/// # Support
/// - `a <= x <= b`
/// # Example
/// Suppose `X ~ Triangular(a=0.0, b=10.0, c=3.0)`. Use
/// ```
/// use ruststat::TriangularDist;
/// let mut mytri = TriangularDist{a:0.0, b:10.0, c:3.0};
/// println!("Probability density function f(4.5): {}", mytri.pdf(4.5));
/// println!("Cumulative distribution function P(X<=4.5): {}", mytri.cdf(4.5));
/// println!("99th percentile: {}", mytri.per(0.99));
/// println!("Random draw: {}", mytri.ran());
/// println!("Random vector: {:?}", mytri.ranvec(5));
/// println!("Mean: {}", mytri.mean());
/// println!("Variance: {}", mytri.var());
/// println!("Standard deviation: {}", mytri.sd());
/// ```
pub struct TriangularDist {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl TriangularDist {
    pub fn pdf(&mut self, x: f64) -> f64 {
        triangular_pdf(x, self.a, self.b, self.c)
    }
    pub fn cdf(&mut self, x: f64) -> f64 {
        triangular_cdf(x, self.a, self.b, self.c)
    }
    pub fn per(&mut self, p: f64) -> f64 {
        triangular_per(p, self.a, self.b, self.c)
    }
    pub fn ran(&mut self) -> f64 {
        triangular_ran(self.a, self.b, self.c)
    }
    pub fn ranvec(&mut self, n: u64) -> Vec<f64> {
        triangular_ranvec(n, self.a, self.b, self.c)
    }
    pub fn mean(&mut self) -> f64 {
        (self.a + self.b + self.c) / 3.0
    }
    pub fn var(&mut self) -> f64 {
        (self.a.powi(2) + self.b.powi(2) + self.c.powi(2) - self.a * self.b - self.a * self.c - self.b * self.c) / 18.0
    }
    pub fn sd(&mut self) -> f64 {
        self.var().sqrt()
    }
}

/// Computes probability density function (pdf) for `X ~ Triangular(a, b, c)`.
/// # Parameters
/// - Lower limit: `a`
/// - Upper limit: `b` (`a < b`)
/// - Mode: `c` (`a <= c <= b`)
/// # Support
/// - `a <= x <= b`
/// # Example
/// Suppose `X ~ Triangular(a=0.0, b=10.0, c=3.0)`. Use.
/// ```
/// use ruststat::triangular_pdf;
/// println!("Probability density function f(4.5): {}", triangular_pdf(4.5, 0.0, 10.0, 3.0));
/// ```
pub fn triangular_pdf(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if a >= b || c < a || c > b {
        println!("NAN produced. Error in function triangular_pdf");
        return f64::NAN;
    }
    if x < a || x > b {
        return 0.0;
    }
    if x < c {
        2.0 * (x - a) / ((b - a) * (c - a))
    } else if x > c {
        2.0 * (b - x) / ((b - a) * (b - c))
    } else {
        2.0 / (b - a)
    }
}

/// Computes cumulative distribution function (cdf) for `X ~ Triangular(a, b, c)`.
/// # Parameters
/// - Lower limit: `a`
/// - Upper limit: `b` (`a < b`)
/// - Mode: `c` (`a <= c <= b`)
/// # Support
/// - `a <= x <= b`
/// # Example
/// Suppose `X ~ Triangular(a=0.0, b=10.0, c=3.0)`. Use.
/// ```
/// use ruststat::triangular_cdf;
/// println!("P(X<=4.5): {}", triangular_cdf(4.5, 0.0, 10.0, 3.0));
/// ```
pub fn triangular_cdf(x: f64, a: f64, b: f64, c: f64) -> f64 {
    if a >= b || c < a || c > b {
        println!("NAN produced. Error in function triangular_cdf");
        return f64::NAN;
    }
    if x <= a {
        return 0.0;
    }
    if x >= b {
        return 1.0;
    }
    if x <= c {
        (x - a).powi(2) / ((b - a) * (c - a))
    } else {
        1.0 - (b - x).powi(2) / ((b - a) * (b - c))
    }
}

/// Computes a percentile for `X ~ Triangular(a, b, c)`.
/// # Note
/// Determines the value of `x` such that `P(X <= x) = p`.
/// # Parameters
/// - Lower limit: `a`
/// - Upper limit: `b` (`a < b`)
/// - Mode: `c` (`a <= c <= b`)
/// # Support
/// - `a <= x <= b`
/// # Example
/// Suppose `X ~ Triangular(a=0.0, b=10.0, c=3.0)`. To find the 80th percentile, use `p=0.80` and
/// ```
/// use ruststat::triangular_per;
/// println!("Percentile: {}", triangular_per(0.8, 0.0, 10.0, 3.0));
/// ```
pub fn triangular_per(p: f64, a: f64, b: f64, c: f64) -> f64 {
    if a >= b || c < a || c > b || p < 0.0 || p > 1.0 {
        println!("NAN produced. Error in function triangular_per");
        return f64::NAN;
    }
    if p == 0.0 { return a; }
    if p == 1.0 { return b; }

    let fc = (c - a) / (b - a);
    if p < fc {
        a + (p * (b - a) * (c - a)).sqrt()
    } else {
        b - ((1.0 - p) * (b - a) * (b - c)).sqrt()
    }
}

/// Random draw from `X ~ Triangular(a, b, c)` distribution.
/// # Parameters
/// - Lower limit: `a`
/// - Upper limit: `b` (`a < b`)
/// - Mode: `c` (`a <= c <= b`)
/// # Support
/// - `a <= x <= b`
/// # Example
/// Suppose `X ~ Triangular(a=0.0, b=10.0, c=3.0)`. Use
/// ```
/// use ruststat::triangular_ran;
/// println!("Random draw: {}", triangular_ran(0.0, 10.0, 3.0));
/// ```
pub fn triangular_ran(a: f64, b: f64, c: f64) -> f64 {
    if a >= b || c < a || c > b {
        println!("Bad argument to triangular_ran");
        return f64::NAN;
    }
    let u: f64 = random();
    let fc = (c - a) / (b - a);
    if u < fc {
        a + (u * (b - a) * (c - a)).sqrt()
    } else {
        b - ((1.0 - u) * (b - a) * (b - c)).sqrt()
    }
}

/// Save random draws from `X ~ Triangular(a, b, c)` distribution into a `Vec`
/// # Parameters
/// - Lower limit: `a`
/// - Upper limit: `b` (`a < b`)
/// - Mode: `c` (`a <= c <= b`)
/// # Support
/// - `a <= x <= b`
/// # Example
/// Suppose `X ~ Triangular(a=0.0, b=10.0, c=3.0)`. Use
/// ```
/// use ruststat::triangular_ranvec;
/// println!("Random Vec: {:?}", triangular_ranvec(10, 0.0, 10.0, 3.0));
/// ```
pub fn triangular_ranvec(n: u64, a: f64, b: f64, c: f64) -> Vec<f64> {
    if a >= b || c < a || c > b {
        println!("Bad argument to triangular_ranvec");
        return if n == 0 { vec![] } else { vec![f64::NAN] };
    }
    let mut xvec: Vec<f64> = Vec::with_capacity(n as usize);
    let fc = (c - a) / (b - a);
    for _ in 0..n {
        let u: f64 = random();
        if u < fc {
            xvec.push(a + (u * (b - a) * (c - a)).sqrt());
        } else {
            xvec.push(b - ((1.0 - u) * (b - a) * (b - c)).sqrt());
        }
    }
    xvec
}