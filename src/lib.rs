#![doc = include_str!("../README.md")]

//! Utilities for working with many common random variables.
//! - probability mass function (pmf), probability density function (pdf)
//! - cumulative distribution function (cdf)
//! - percentiles (inverse cdf)
//! - random number generation
//! - mean, variance
//!
//! Distributions:
//! - Continuous: beta, chi-square, exponential, F, gamma, gumbel, logistic, normal, log-normal, Pareto (1 through 4),
//! Student's t, continuous uniform
//! - Discrete: binomial, geometric, hypergeometric, negative binomial, poisson
//!
//! # Quick Start
//! ```
//! use ruststat::*;
//! // X~N(mu=0,sigma=1.0), find 97.5th percentile
//! println!("normal percentile: {}", normal_per(0.975, 0.0, 1.0));
//! // X~Bin(n=10,p=0.7), compute P(X=4)
//! println!("binomial probability: {}", bin_pmf(4, 10, 0.7));
//! ```
//! For convenience, functions can also be accessed via `Structs`.
//! ```
//! use ruststat::*;
//! // X~Beta(alpha=0.5,beta=2.0)
//! let mut mybeta = BetaDist{alpha:0.5, beta:2.0};
//! // 30th percentile
//! println!("percentile: {}", mybeta.per(0.3));
//! // P(X <= 0.4)
//! println!("cdf: {}", mybeta.cdf(0.4));
//! // Random draw
//! println!("random draw: {}", mybeta.ran());
//! // Variance
//! println!("variance: {}", mybeta.var());
//! ```

pub mod beta;
pub mod chisq;
pub mod exp;
pub mod f;
pub mod gamma;
pub mod gumbel;
pub mod logistic;
pub mod lognormal;
pub mod normal;
pub mod pareto;
pub mod t;
pub mod unif;
pub mod bin;
pub mod geo;
pub mod geo2;
pub mod hg;
pub mod nb;
pub mod nb2;
pub mod pois;

pub use beta::*;
pub use chisq::*;
pub use exp::*;
pub use f::*;
pub use gamma::*;
pub use gumbel::*;
pub use logistic::*;
pub use lognormal::*;
pub use normal::*;
pub use pareto::*;
pub use t::*;
pub use unif::*;
pub use bin::*;
pub use geo::*;
pub use geo2::*;
pub use hg::*;
pub use nb::*;
pub use nb2::*;
pub use pois::*;

// use std::collections::btree_set::Iter;
use std::f64::consts::PI;
// use rand::{random};
use crate::beta::{beta_per, beta_ran};
use crate::gamma::gamma_ran;
use crate::geo2::geo2_ran;
use crate::geo::geo_ran;
use crate::normal::normal_ran;
use crate::unif::unif_ran;

// const GAMMA: f64 = 0.577215664901532860606512090082;


/// Computes sample mean of elements in a slice
///
/// # Example
/// ```
/// use ruststat::sample_mean;
/// let x = vec![1.5, 2.0, 3.5];
/// println!("Sample mean: {}", sample_mean(&x));
/// ```
pub fn sample_mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = x.iter().sum();
    sum / (x.len() as f64)
}

/// Computes sample variance of elements in a slice
///
/// # Example
/// ```
/// use ruststat::sample_var;
/// let x = vec![1.5, 2.0, 3.5];
/// println!("Sample variance: {}", sample_var(&x));
/// ```
pub fn sample_var(x: &[f64]) -> f64 {
    let n = x.len();

    if n < 2 {
        return f64::NAN;
    }

    let xbar = sample_mean(x);
    let ss: f64 = x.iter().map(|xx| (xx - xbar).powi(2)).sum();

    ss / ((n - 1) as f64)
}

/// Computes sample standard deviation of elements in a slice
///
/// # Example
/// ```
/// use ruststat::sample_sd;
/// let x = vec![1.5, 2.0, 3.5];
/// println!("Sample SD: {}", sample_sd(&x));
/// ```
pub fn sample_sd(x: &[f64]) -> f64 {
    sample_var(x).sqrt()
}


// ===========================================================================
// ===========================================================================
// Special Functions
// ===========================================================================
// ===========================================================================

/// Incomplete beta function.
///
/// # Parameters
/// - `a > 0`
/// - `b > 0`
/// - `0 < x < 1`
/// # Example
/// Suppose `X ~ Beta(alpha=0.5, beta=2.0)`.
/// ```
/// use ruststat::betai;
/// println!("Incomplete beta function: {}", betai(0.7, 0.5, 2.0));
/// ```
pub fn betai(x: f64, a: f64, b: f64) -> f64 {

    let bt: f64;

    if x < 0.0 || x > 1.0 {
        println!("Bad x in function betai");
        return f64::NAN;
    }
    if x == 0.0 || x == 1.0 {
        bt=0.0;
    }
    else {
        bt=(gamma_ln(a+b)-gamma_ln(a)-gamma_ln(b) +
            a*x.ln() +
            b*(1.0-x).ln()).exp();
    }
    if x < ((a + 1.0) / (a + b + 2.0)) {
        bt * betacf(x, a, b) / a
    } else {
        1.0 - bt * betacf(1.0 - x, b, a) / b
    }
}


/// Evaluates the continued fraction for the incomplete beta function.
///
/// # Note
/// This is an internal helper function used by `betai` to compute the incomplete
/// beta function using Lentz's method for evaluating continued fractions.
///
/// # Parameters
/// - `a > 0`
/// - `b > 0`
/// - `0 < x < 1`
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    let fpmin = 1.0e-30;

    let mut m2: u64;
    let mut aa: f64;
    let mut c: f64;
    let mut d: f64;
    let mut del: f64;
    let mut h: f64;
    let qab: f64;
    let qam: f64;
    let qap: f64;

    qab=a+b;
    qap=a+1.0;
    qam=a-1.0;
    c=1.0;
    d=1.0-qab*x/qap;
    if d.abs() < fpmin {
        d=fpmin;
    }
    d=1.0/d;
    h=d;
    for m in 1..=imax {
        m2=2*m;
        aa=(m as f64)*(b-m as f64)*x/((qam+m2 as f64)*(a+m2 as f64));
        d=1.0+aa*d;
        if d.abs() < fpmin {
            d=fpmin;
        }
        c=1.0+aa/c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d=1.0/d;
        h *= d*c;
        aa = -(a+m as f64)*(qab+m as f64)*x/((a+m2 as f64)*(qap+m2 as f64));
        d=1.0+aa*d;
        if d.abs() < fpmin {
            d=fpmin;
        }
        c=1.0+aa/c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d=1.0/d;
        del=d*c;
        h *= del;
        if (del-1.0).abs() < eps {
            break;
        }
    }
    // if (m > MAXIT) nrerror("a or b too big, or MAXIT too small in betacf");
    h
}


/// Log gamma function
/// - `x > 0`
pub fn gamma_ln(x: f64) -> f64 {
    let cof = vec![57.1562356658629235,-59.5979603554754912,
                   14.1360979747417471,-0.491913816097620199,0.339946499848118887e-4,
                   0.465236289270485756e-4,-0.983744753048795646e-4,0.158088703224912494e-3,
                   -0.210264441724104883e-3,0.217439618115212643e-3,-0.164318106536763890e-3,
                   0.844182239838527433e-4,-0.261908384015814087e-4,0.368991826595316234e-5];

    if x <= 0.0 {
        println!("Bad argument in gamma_ln; returning f64::NAN");
        return f64::NAN;
        // panic!("bad ard in gamma_ln");
    }

    let mut y= x;
    let z = x;

    let mut tmp = z + 5.24218750000000000;
    tmp = (z + 0.5) * tmp.ln() - tmp;
    let mut ser = 0.999999999999997092;
    for j in 0..14 {
        ser += cof[j] / { y += 1.0; y};
    }
    tmp + (2.5066282746310005 * ser / z).ln()
}


/// Gamma function
/// - `x > 0`
pub fn gamma(x: f64) -> f64 {
    gamma_ln(x).exp()
}

/// Factorial (`u64`).
/// Note: Will overflow `u64` capacity at x = 21. Iterative to prevent stack overflow.
pub fn fact_i(x: u64) -> u64 {
    let mut res = 1;
    for i in 2..=x {
        res *= i;
    }
    res
}

/// Log factorial (`u64`).
/// - `x = 1,2,3,...`
pub fn factln_u(x: u64) -> f64 {
    gamma_ln(x as f64 + 1.0)
}


/// Generalized factorial (`f64`).
/// - `x > 0`
pub fn fact_f(x: f64) -> f64 {
    gamma(x + 1.0)
}


/// Log Generalized factorial (`f64`).
/// - `x > 0`
pub fn factln_f(x: f64) -> f64 {
    gamma_ln(x + 1.0)
}


/// Combination
/// - `n = 0,1,2,...`
/// - `x = 0,1,2,...`
/// - `n >= x`
pub fn comb(n: u64, x: u64) -> f64 {
    (0.5 + (factln_u(n) - factln_u(x) - factln_u(n-x)).exp()).floor()
}


/// Log-Combination
/// - `n = 0,1,2,...`
/// - `x = 0,1,2,...`
/// - `n >= x`
pub fn combln(n: u64, x: u64) -> f64 {
    factln_u(n) - factln_u(x) - factln_u(n-x)
}


/// Computes beta function.
/// - `a > 0`
/// - `b > 0`
pub fn beta_fn(a: f64, b:f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a+b)).exp()
}


/// Computes log beta function.
/// - `a > 0`
/// - `b > 0`
pub fn beta_fn_ln(a: f64, b:f64) -> f64 {
    gamma_ln(a) + gamma_ln(b) - gamma_ln(a+b)
}


/// Incomplete gamma function, series representation
pub fn gser(x: f64, a: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    // let fpmin = 1.0e-30;
    let gln = gamma_ln(a);

    if x <= 0.0 {
        if x < 0.0 {
            println!("x less than 0 in routine gser");
        }
        return 0.0;
    }
    else {
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = 1.0 / a;

        for _ in 1..imax {
            ap = ap + 1.0;
            del = del * x / ap;
            sum = sum + del;
            if del.abs() < sum.abs() * eps {
                return sum * (-x + a * x.ln() - gln).exp();
            }
        }
        println!("a too large, ITMAX too small in routine gser");
    }
    0.0
}


/// Incomplete gamma function, continued fraction representation
pub fn gcf(x: f64, a: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    let fpmin = 1.0e-30;

    let gln = gamma_ln(a);
    let mut b= x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d= 1.0 / b;
    let mut h= d;

    let mut an: f64;
    let mut del: f64;

    for i in 1..=imax {
        an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (del-1.0).abs() < eps {
            break;
        }
    }

    (-x + a * x.ln() - gln).exp() * h
}

pub fn gcf_ln(x: f64, a: f64) -> f64 {
    let imax = 100;
    let eps = 3.0e-7;
    let fpmin = 1.0e-30;

    let gln = gamma_ln(a);
    let mut b= x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d= 1.0 / b;
    let mut h= d;

    let mut an: f64;
    let mut del: f64;

    for i in 1..=imax {
        an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c=fpmin;
        }
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (del-1.0).abs() < eps {
            break;
        }
    }

    (-x + a * x.ln() - gln) + h.ln()
}


/// Incomplete gamma function `P(x,a)`
pub fn gammp(x: f64, a: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        println!("Invalid arguments in routine gammp");
        return f64::NAN;
    }
    if x == 0.0 {
        0.0
    } else if a >= 100.0 {
        return 1.0 - gammapprox(x, a); // <-- The crucial correction!
    } else if x < (a + 1.0) {
        return gser(x, a);
    } else {
        return 1.0 - gcf(x, a);
    }
}


/// Complementary incomplete gamma function `Q(x,a)`
pub fn gammq(x: f64, a: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        println!("Invalid arguments in routine gammq");
    }
    if x == 0.0 {
        1.0
    } else if a >= 100.0 {
        gammapprox(x, a)
    } else if x < (a + 1.0) {
        1.0 - gser(x, a)
    } else {
        gcf(x, a)
    }
}


/// Incomplete gamma function, quadrature `P(x,a)`
pub fn gammapprox(x: f64, a: f64) -> f64 {

    const Y_VEC: [f64; 18] = [
        0.0021695375159141994, 0.011413521097787704,0.027972308950302116,
        0.051727015600492421, 0.082502225484340941, 0.12007019910960293,
        0.16415283300752470, 0.21442376986779355, 0.27051082840644336,
        0.33199876341447887, 0.39843234186401943, 0.46931971407375483,
        0.54413605556657973, 0.62232745288031077, 0.70331500465597174,
        0.78649910768313447, 0.87126389619061517, 0.95698180152629142];
    const W_VEC: [f64; 18] = [
        0.0055657196642445571, 0.012915947284065419, 0.020181515297735382,
        0.027298621498568734, 0.034213810770299537,0.040875750923643261,
        0.047235083490265582, 0.053244713977759692,0.058860144245324798,
        0.064039797355015485, 0.068745323835736408,0.072941885005653087,
        0.076598410645870640, 0.079687828912071670,0.082187266704339706,
        0.084078218979661945, 0.085346685739338721,0.085983275670394821];

    let (xu,mut t,mut sum,ans,a1,lna1,sqrta1): (f64,f64,f64,f64,f64,f64,f64);
    a1 = a-1.0;
    lna1 = a1.ln();
    sqrta1 = a1.sqrt();
    if x > a1 {
        xu = f64::max(a1 + 11.5*sqrta1, x + 6.0*sqrta1);
    }
    else {
        xu = f64::max(0.0,f64::min(a1 - 7.5*sqrta1, x - 5.0*sqrta1));
    }
    sum = 0.0;
    for j in 0..18 { //Gauss-Legendre.
        t = x + (xu-x)*Y_VEC[j];
        sum += W_VEC[j]*(-(t-a1)+a1*(t.ln()-lna1)).exp();
    }
    ans = sum*(xu-x)*(a1*(a1.ln()-1.0)-gamma_ln(a)).exp();

    if ans >= 0.0 {
        ans
    } else {
        1.0 + ans
    }
}


/// Error function (gammp implementation) `Erf(x)`
pub fn erf2(x: f64) -> f64 {
    if x < 0.0 {
        -gammp(x * x, 0.5)
    } else {
        gammp(x * x, 0.5)
    }
}


/// Inverse of incomplete beta function
/// - `0 < p < 1`
/// - `alpha > 0`
/// - `beta > 0`
pub fn betai_inv(p: f64, a:f64, b: f64) -> f64 {
    let eps = 1.0e-8;
    let (pp,mut t,mut u,mut err,mut x,al,h,w,afac,a1,b1,lna,lnb):
        (f64,f64,f64,f64,f64,f64,f64,f64,f64,f64,f64,f64,f64);
    a1=a-1.0;
    b1=b-1.0;

    if p <= 0.0 {
        return 0.;
    }
    else if p >= 1.0 {
        return 1.0;
    }
    else if a >= 1.0 && b >= 1.0 {
        if p < 0.5 {
            pp = p
        }
        else {
            pp = 1.0 - p;
        }
        t = (-2.0*pp.ln()).sqrt();
        x = (2.30753 + t*0.27061) / (1.0 + t*(0.99229 + t*0.04481)) - t;
        if p < 0.5 {
            x = -x;
        }
        al = (x.powi(2)-3.0)/6.0;
        h = 2.0 / (1.0/(2.0*a-1.0)+1.0/(2.0*b-1.0));
        w = (x*(al+h).sqrt()/h)-(1.0/(2.0*b-1.0)-1.0/(2.0*a-1.0))*(al+5.0/6.0-2.0/(3.0*h));
        x = a/(a+b*(2.0*w).exp());
    }
    else {
        lna = (a/(a+b)).ln();
        lnb = (b/(a+b)).ln();
        t = (a*lna).exp()/a;
        u = (b*lnb).exp()/b;
        w = t + u;
        if p < t/w  {
            x = (a*w*p).powf(1.0/a);
        }
        else {
            x = 1.0 - (b*w*(1.-p)).powf(1.0/b);
        }
    }

    afac = -gamma_ln(a)-gamma_ln(b)+gamma_ln(a+b);
    for j in 0..10 {
        if x == 0.0 || x == 1.0 {
            return x;
        }
        err = betai(x, a, b) - p;
        t = (a1 * x.ln() + b1 * (1.0 - x).ln() + afac).exp();
        u = err / t;
        t = u / (1.0 - 0.5 * f64::min(1.0, u * (a1 / x - b1 / (1.0 - x))));
        x -= t;
        if x <= 0.0 {
            x = 0.5 * (x + t);
        }
        if x >= 1.0 {
            x = 0.5 * (x + t + 1.0);
        }
        if t.abs() < eps*x && j>0 {
            break;
        }
    }

    x
}


/// Inverse of incomplete gamma function
/// - `0 < p < 1`
/// - `alpha > 0`
pub fn gammai_inv(p: f64, alpha:f64) -> f64 {
    let (mut x,mut err,mut t,mut u,pp,lna1,afac,a1): (f64,f64,f64,f64,f64,f64,f64,f64);
    a1 = alpha-1.0;

    let eps = 1.0e-8; // Accuracy is the square of eps.
    let gln = gamma_ln(alpha);

    if alpha <= 0.0 {
        println!("a must be pos in invgammap");
    }
    if p >= 1.0 {
        return f64::max(100.0,alpha + 100.0*alpha.sqrt());
    }
    if p <= 0.0 {
        return 0.0;
    }
    lna1 = a1.ln();
    afac = (a1*(lna1-1.0)-gln).exp();
    if alpha > 1.0 {
        if p < 0.5 {
            pp = p;
        }
        else {
            pp = 1.0 - p;
        }
        t = (-2.0*pp.ln()).sqrt();
        x = (2.30753+t*0.27061) /
            (1.0+t*(0.99229+t*0.04481)) - t;
        if p < 0.5 {
            x = -x;
        }
        // Wilson-Hilferty calculation:
        x = f64::max(1.0e-3, alpha * (1.0 - 1.0 / (9.0 * alpha) - x / (3.0 * alpha.sqrt())).powi(3));    }
    else {
        t = 1.0 - alpha*(0.253+alpha*0.12);
        if p < t {
            x = (p/t).powf(1.0/alpha);
        }
        else {
            x = 1.0-(1.0-(p-t)/(1.0-t)).ln();
        }
    }

    for _ in 0..12 {
        if x <= 0.0 {
            return 0.0;
        }
        err = gammp(x,alpha) - p;
        if alpha > 1.0 {
            t = afac*(-(x-a1)+a1*(x.ln()-lna1)).exp();
        }
        else {
            t = (-x+a1*x.ln()-gln).exp();
        }
        u = err/t;
        t = u/(1.0-0.5*f64::min(1.0,u*((alpha-1.0)/x - 1.0)));
        x -= t; //Halley’s method.
        if x <= 0.0 {
            x = 0.5*(x + t);
        } // halve old value if x tries to go negative.
        if t.abs() < eps *x {
            break;
        }
    }

    x
}


/// Chebyshev coefficients
fn erfc_cheb(z: f64) -> f64 {

    const NCOEF: usize = 28;
    const COEF: [f64; 28] = [
        -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
        -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5,
        -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
        6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
        9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13,
        3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15,
        -1.523e-15, -9.4e-17, 1.21e-16,-2.8e-17 ];

    let (mut d, mut dd, t, ty, mut tmp) : (f64, f64, f64, f64, f64);
    d = 0.0;
    dd = 0.0;

    assert!(z >= 0f64, "erfccheb requires nonnegative argument");
    t = 2.0 / (2.0 + z);
    ty = 4.0 * t - 2.0;
    for j in (1..NCOEF-1).rev() {
        tmp = d;
        d = ty * d - dd + COEF[j];
        dd = tmp;
    }
    t * (-z.powi(2) + 0.5 * (COEF[0] + ty * d) - dd).exp()
}


/// Error function (Chebyshev implementation)
pub fn erf(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 - erfc_cheb(x)
    } else {
        erfc_cheb(-x) - 1.0
    }
}


/// Complementary error function
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        erfc_cheb(x)
    } else {
        2.0 - erfc_cheb(-x)
    }
}


/// Inverse of complementary error function
pub fn erfc_inv(p: f64) -> f64 {

    let (pp, t, mut x): (f64, f64, f64);

    // return arbitrary large pos or neg value
    if p >= 2.0 {
        return -100.0;
    }
    else if p <= 0.0 {
        return 100.0;
    }

    if p < 1.0 {
        pp = p
    }
    else {
        pp = 2.0 - p;
    }

    t = (-2.0 * (pp / 2.0).ln()).sqrt();
    x = -std::f64::consts::FRAC_1_SQRT_2 * ((2.30753 + t * 0.27061) /
        (1f64 + t * (0.99229 + t * 0.04481)) - t);
    for _ in 0..2 {
        let err = erfc(x) - pp;
        x += err / (std::f64::consts::FRAC_2_SQRT_PI * (-x.powi(2)).exp() - x * err);
    }
    if p < 1.0 {
        x
    } else {
        -x
    }
}


/// Inverse of error function
pub fn erf_inv(p: f64) -> f64 {
    erfc_inv(1.0 - p)
}



#[cfg(test)]
mod tests {
    use crate::beta::{beta_cdf, beta_pdf, beta_ranvec, BetaDist};
    use crate::bin::{bin_cdf, bin_per, bin_pmf, bin_ran, bin_ranvec, BinDist};
    use crate::chisq::{chisq_cdf, chisq_pdf, chisq_per, chisq_ran, chisq_ranvec, ChiSqDist};
    use crate::exp::{exp_cdf, exp_pdf, exp_per, exp_ran, exp_ranvec, ExpDist};
    use crate::f::{f_cdf, f_pdf, f_per, f_ran, f_ranvec, FDist};
    use crate::gamma::{gamma_cdf, gamma_pdf, gamma_per, gamma_ranvec, GammaDist};
    use crate::geo::{geo_cdf, geo_per, geo_pmf, geo_ranvec, GeoDist};
    use crate::geo2::{geo2_cdf, geo2_per, geo2_pmf, geo2_ranvec, Geo2Dist};
    use crate::hg::{hg_cdf, hg_per, hg_pmf, hg_ran, hg_ranvec, HGDist};
    use crate::lognormal::{lognormal_cdf, lognormal_pdf, lognormal_per, lognormal_ran, lognormal_ranvec, LogNormalDist};
    use crate::nb::{nb_cdf, nb_per, nb_pmf, nb_ran, nb_ranvec, NBDist};
    use crate::nb2::{nb2_cdf, nb2_per, nb2_pmf, nb2_ran, nb2_ranvec, NB2Dist};
    use crate::normal::{normal_cdf, normal_pdf, normal_per, normal_ranvec, NormalDist};
    use crate::pois::{pois_cdf, pois_per, pois_pmf, pois_ran, pois_ranvec, PoisDist};
    use crate::t::{t_cdf, t_pdf, t_per, t_ran, t_ranvec, TDist};
    use crate::unif::{unif_cdf, unif_pdf, unif_per, unif_ranvec, UnifDist};
    use super::*;

    // macro to assert that two floats are within a specific tolerance
    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr, $eps:expr) => {{
            let (a, b) = (&$a, &$b);
            let diff = (*a - *b).abs();
            assert!(
                diff < $eps,
                "Assertion failed: values not approximately equal.\n  Expected: {:?}\n  Actual: {:?}\n  Difference: {:?}",
                *a, *b, diff
            );
        }};
    }

    #[test]
    fn test_gamma_large_parameter_regression() {
        let alpha = 500.0;
        let beta = 0.5;
        let target_p = 0.2;

        // 1. Find the x value for the 20th percentile
        let x = gamma_per(target_p, alpha, beta);

        // Ensure the root finder didn't crash to 0.0 or NaN
        assert!(x > 0.0, "Root finder failed and returned {}", x);

        // 2. Feed x back into the CDF to see if we get 0.2 back
        let calc_p = gamma_cdf(x, alpha, beta);

        // 3. Check for the round-trip match within a tight tolerance
        assert_approx_eq!(target_p, calc_p, 1e-6);
    }

    #[test]
    fn test_distribution_percentile_roundtrips() {
        let test_percentiles = vec![0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
        let tolerance = 1e-5;

        for &p in &test_percentiles {
            // Test Beta(0.5, 2.0)
            let x_beta = beta_per(p, 0.5, 2.0);
            let p_beta = beta_cdf(x_beta, 0.5, 2.0);
            assert_approx_eq!(p, p_beta, tolerance);

            // Test Normal(100.0, 16.0)
            let x_norm = normal_per(p, 100.0, 16.0);
            let p_norm = normal_cdf(x_norm, 100.0, 16.0);
            assert_approx_eq!(p, p_norm, tolerance);

            // Test Exponential(3.5)
            let x_exp = exp_per(p, 3.5);
            let p_exp = exp_cdf(x_exp, 3.5);
            assert_approx_eq!(p, p_exp, tolerance);
        }
    }

    #[test]
    fn test_chisq_large_parameter_regression() {
        // nu = 500 means alpha = 250, which easily clears our >= 100.0 safeguard
        let nu = 500.0;
        let target_p = 0.2;

        // 1. Find the x value for the 20th percentile
        let x = chisq_per(target_p, nu);

        // Ensure the root finder didn't crash
        assert!(x > 0.0, "Root finder failed and returned {}", x);

        // 2. Feed x back into the CDF
        let calc_p = chisq_cdf(x, nu);

        // 3. Check for the round-trip match
        assert_approx_eq!(target_p, calc_p, 1e-6);
    }

    #[test]
    fn test_normal_ran_statistical_properties() {
        let mu = 100.0;
        let sigma = 16.0;
        let n = 100_000; // 100,000 draws to ensure statistical stability

        // 1. Generate a massive array of random draws
        // (This will run incredibly fast now that you pre-allocate the Vec!)
        let draws = normal_ranvec(n, mu, sigma);

        // 2. Calculate the sample statistics
        let calc_mean = sample_mean(&draws);
        let calc_var = sample_var(&draws);

        // 3. Define the theoretical variance
        let theoretical_var = sigma.powi(2); // 16^2 = 256.0

        // 4. Assert that the sample stats match the theoretical parameters
        // Note: Because it's random, we use a slightly wider tolerance than the CDF tests.
        // With n = 100,000, the sample mean should easily be within 0.5 of mu,
        // and variance within 3.0 of theoretical_var.
        assert_approx_eq!(mu, calc_mean, 0.5);
        assert_approx_eq!(theoretical_var, calc_var, 3.0);
    }

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr, $eps:expr) => {{
            let (a, b) = (&$a, &$b);
            let diff = (*a - *b).abs();
            assert!(
                diff < $eps,
                "Assertion failed: values not approximately equal.\n  Expected: {:?}\n  Actual: {:?}\n  Difference: {:?}",
                *a, *b, diff
            );
        }};
    }

    const TOL: f64 = 1e-6;

    #[test]
    fn test_beta_correctness() {
        // Beta(1, 1) is mathematically identical to a Uniform(0, 1) distribution
        assert_approx_eq!(beta_pdf(0.5, 1.0, 1.0), 1.0, TOL);
        assert_approx_eq!(beta_cdf(0.5, 1.0, 1.0), 0.5, TOL);
        assert_approx_eq!(beta_per(0.5, 1.0, 1.0), 0.5, TOL);
    }

    #[test]
    fn test_normal_correctness() {
        // Standard Normal N(0, 1) at x = 0 should be 1 / sqrt(2 * pi)
        let one_over_sqrt_2pi = 0.3989422804014327;
        assert_approx_eq!(normal_pdf(0.0, 0.0, 1.0), one_over_sqrt_2pi, TOL);

        // CDF at the mean is exactly 50%
        assert_approx_eq!(normal_cdf(0.0, 0.0, 1.0), 0.5, TOL);
        assert_approx_eq!(normal_per(0.5, 0.0, 1.0), 0.0, TOL);

        // Empirical rule: ~68.27% of data falls within 1 standard deviation
        let cdf_1 = normal_cdf(1.0, 0.0, 1.0);
        let cdf_neg1 = normal_cdf(-1.0, 0.0, 1.0);
        assert_approx_eq!(cdf_1 - cdf_neg1, 0.682689492, TOL);
    }

    #[test]
    fn test_exponential_correctness() {
        // Exp(lambda = 2.0) at x = 1.0
        // pdf(x) = lambda * e^(-lambda * x)
        // cdf(x) = 1 - e^(-lambda * x)
        let lambda = 2.0;
        let x = 1.0;
        let expected_pdf = 2.0 * std::f64::consts::E.powf(-2.0);
        let expected_cdf = 1.0 - std::f64::consts::E.powf(-2.0);

        assert_approx_eq!(exp_pdf(x, lambda), expected_pdf, TOL);
        assert_approx_eq!(exp_cdf(x, lambda), expected_cdf, TOL);
    }

    #[test]
    fn test_uniform_correctness() {
        // Unif(1.5, 4.5)
        // Width = 3.0, so the constant PDF is 1/3
        assert_approx_eq!(unif_pdf(2.0, 1.5, 4.5), 0.333333333333, TOL);
        assert_approx_eq!(unif_cdf(3.0, 1.5, 4.5), 0.5, TOL);
        assert_approx_eq!(unif_per(0.5, 1.5, 4.5), 3.0, TOL);
    }

    #[test]
    fn test_binomial_correctness() {
        // Bin(n=10, p=0.5)
        // P(X=5) = (10 choose 5) * 0.5^10 = 252 / 1024 = 0.24609375
        assert_approx_eq!(bin_pmf(5, 10, 0.5), 0.24609375, TOL);

        // Bin(n=100, p=0.9)
        // P(X=100) requires all successes, which is exactly 0.9^100
        assert_approx_eq!(bin_pmf(100, 100, 0.9), 0.9_f64.powi(100), TOL);
        assert_approx_eq!(bin_cdf(100, 100, 0.9), 1.0, TOL);
    }

    #[test]
    fn test_poisson_correctness() {
        // Pois(lambda = 2.0)
        // P(X=3) = e^(-2) * 2^3 / 3! = 0.135335 * 8 / 6
        assert_approx_eq!(pois_pmf(3, 2.0), 0.180447044315, TOL);

        // P(X=0) = e^(-lambda)
        assert_approx_eq!(pois_pmf(0, 2.0), std::f64::consts::E.powf(-2.0), TOL);
    }

    #[test]
    fn test_geometric_correctness() {
        // Geo(p = 0.05) - Number of failures before first success
        // P(X=0 failures) = you succeeded on the first try = p
        assert_approx_eq!(geo_pmf(0, 0.05), 0.05, TOL);

        // P(X=1 failure) = (1-p) * p = 0.95 * 0.05
        assert_approx_eq!(geo_pmf(1, 0.05), 0.0475, TOL);

        // Geo2(p = 0.05) - Trial number of first success
        // P(X=1st trial) = p
        assert_approx_eq!(geo2_pmf(1, 0.05), 0.05, TOL);
        assert_approx_eq!(geo2_pmf(2, 0.05), 0.0475, TOL);
    }

    #[test]
    fn test_chisq_correctness() {
        // ChiSq(nu = 2.0) is mathematically identical to an Exponential(lambda = 0.5)
        let x = 2.0;
        let expected_pdf = 0.5 * std::f64::consts::E.powf(-1.0);
        assert_approx_eq!(chisq_pdf(x, 2.0), expected_pdf, TOL);
    }

    #[test]
    fn test_hypergeometric_correctness() {
        // HG(n=20, N=100, M=50)
        // Drawing 20 items from a population of 100 where 50 are successes.
        // P(X=20) = exactly matching all 20 draws to the 50 successes.
        // (50 choose 20) / (100 choose 20)
        let expected_prob = 0.000000017402868; // Calculated via standard statistical tables
        assert_approx_eq!(hg_pmf(20, 20, 100, 50), expected_prob, TOL);
    }

    #[test]
    fn test_gamma_correctness() {
        // Gamma(n) = (n-1)! for integer n
        assert_approx_eq!(gamma(5.0), 24.0, TOL);
        assert_approx_eq!(gamma(6.0), 120.0, TOL);

        // Gamma(0.5) = sqrt(pi)
        let sqrt_pi = PI.sqrt();
        assert_approx_eq!(gamma(0.5), sqrt_pi, TOL);

        // ln(Gamma(x)) should match the log of Gamma(x)
        assert_approx_eq!(gamma_ln(5.0), 24.0_f64.ln(), TOL);
    }

    #[test]
    fn test_factorial_correctness() {
        // fact_i computes standard factorials
        assert_eq!(fact_i(1), 1);
        assert_eq!(fact_i(5), 120);
        assert_eq!(fact_i(10), 3_628_800);

        // fact_f generalizes factorial: x! = x * Gamma(x)
        // 5! = 120
        assert_approx_eq!(fact_f(5.0), 120.0, TOL);
        // 0.5! = 0.5 * Gamma(0.5) = 0.5 * sqrt(pi)
        let half_fact = 0.5 * PI.sqrt();
        assert_approx_eq!(fact_f(0.5), half_fact, TOL);
    }

    #[test]
    fn test_combinations_correctness() {
        // 10 choose 5 = 252
        assert_approx_eq!(comb(10, 5), 252.0, TOL);
        // 50 choose 2 = (50 * 49) / 2 = 1225
        assert_approx_eq!(comb(50, 2), 1225.0, TOL);
        // n choose 0 = 1, n choose n = 1
        assert_approx_eq!(comb(20, 0), 1.0, TOL);
        assert_approx_eq!(comb(20, 20), 1.0, TOL);
    }

    #[test]
    fn test_beta_function_correctness() {
        // B(1, 1) = 1
        assert_approx_eq!(beta_fn(1.0, 1.0), 1.0, TOL);

        // B(2, 2) = Gamma(2)*Gamma(2) / Gamma(4) = (1 * 1) / 6 = 1/6
        assert_approx_eq!(beta_fn(2.0, 2.0), 1.0 / 6.0, TOL);

        // B(a, b) should equal B(b, a) (Symmetry)
        assert_approx_eq!(beta_fn(3.5, 7.2), beta_fn(7.2, 3.5), TOL);
    }

    #[test]
    fn test_incomplete_beta_correctness() {
        // I_x(1, 1) = x
        assert_approx_eq!(betai(0.25, 1.0, 1.0), 0.25, TOL);
        assert_approx_eq!(betai(0.75, 1.0, 1.0), 0.75, TOL);

        // I_x(a, b) at x=0 is 0, at x=1 is 1
        assert_approx_eq!(betai(0.0, 2.0, 5.0), 0.0, TOL);
        assert_approx_eq!(betai(1.0, 2.0, 5.0), 1.0, TOL);

        // I_{0.5}(a, a) = 0.5 due to symmetry
        assert_approx_eq!(betai(0.5, 3.0, 3.0), 0.5, TOL);
    }

    #[test]
    fn test_incomplete_gamma_correctness() {
        // gammp(x, a) is the regularized lower incomplete gamma function.
        // P(x, 1) = 1 - e^(-x)
        let x = 1.0;
        let expected = 1.0 - std::f64::consts::E.powf(-x);
        assert_approx_eq!(gammp(x, 1.0), expected, TOL);

        // Q(x, a) is the complement: Q(x, a) = 1 - P(x, a)
        // Therefore, gammq(x, a) + gammp(x, a) = 1.0
        assert_approx_eq!(gammp(2.5, 3.0) + gammq(2.5, 3.0), 1.0, TOL);
    }

    #[test]
    fn test_error_function_correctness() {
        // erf(0) = 0
        assert_approx_eq!(erf(0.0), 0.0, TOL);

        // erf(x) for a known value: erf(1) ≈ 0.84270079
        assert_approx_eq!(erf(1.0), 0.8427007929497148, TOL);

        // erf(-x) = -erf(x) (Odd function)
        assert_approx_eq!(erf(-1.0), -erf(1.0), TOL);

        // erf(x) + erfc(x) = 1.0
        assert_approx_eq!(erf(0.5) + erfc(0.5), 1.0, TOL);
    }

    #[test]
    fn test_error_function_inverses() {
        // Test round-tripping: erf_inv(erf(x)) == x
        let x = 0.5;
        let p = erf(x);
        assert_approx_eq!(erf_inv(p), x, TOL);

        // erf_inv(0) = 0
        assert_approx_eq!(erf_inv(0.0), 0.0, TOL);

        // Test round-tripping for erfc
        let p_comp = erfc(x);
        assert_approx_eq!(erfc_inv(p_comp), x, TOL);
    }

    #[test]
    pub fn it_works() {
        // let result = add(2, 2);
        // assert_eq!(result, 4);
        println!("Testing each distribution function...");

        println!("\nBeta");
        println!("pdf: {}", beta_pdf(0.7, 0.5, 1.5));
        println!("cdf: {}", beta_cdf(0.7, 0.5, 1.5));
        println!("per: {}", beta_per(0.1, 0.5, 1.5));
        println!("ran: {}", beta_ran(0.5, 1.5));
        println!("ran_vec: {:?}", beta_ranvec(3, 0.5, 1.5));
        let mut mybeta = BetaDist { alpha: 0.5, beta: 1.5 };
        println!("pdf: {}", mybeta.pdf(0.7));
        println!("cdf: {}", mybeta.cdf(0.7));
        println!("per: {}", mybeta.per(0.1));
        println!("ran: {}", mybeta.ran());
        println!("ranvec: {:?}", mybeta.ranvec(5));
        println!("mean: {}", mybeta.mean());
        println!("var: {}", mybeta.var());
        println!("sd: {}", mybeta.sd());
        println!("---------------------");

        println!("\nChi-Square");
        println!("pdf: {}", chisq_pdf(0.7, 1.5));
        println!("cdf: {}", chisq_cdf(0.1, 5000.0));
        println!("per: {}", chisq_per(0.1, 5000.0));
        println!("ran: {}", chisq_ran(1.5));
        println!("ran_vec: {:?}", chisq_ranvec(3, 1.5));
        let mut mychisq = ChiSqDist { nu: 1.5 };
        println!("pdf: {}", mychisq.pdf(0.7));
        println!("cdf: {}", mychisq.cdf(0.7));
        println!("per: {}", mychisq.per(0.1));
        println!("ran: {}", mychisq.ran());
        println!("ranvec: {:?}", mychisq.ranvec(5));
        println!("mean: {}", mychisq.mean());
        println!("var: {}", mychisq.var());
        println!("sd: {}", mychisq.sd());
        println!("---------------------");

        println!("\nExp");
        println!("pdf: {}", exp_pdf(0.7, 1.5));
        println!("cdf: {}", exp_cdf(0.7, 1.5));
        println!("per: {}", exp_per(0.1, 1.5));
        println!("ran: {}", exp_ran(1.5));
        println!("ran_vec: {:?}", exp_ranvec(3, 1.5));
        let mut myexp = ExpDist { lambda: 1.5 };
        println!("pdf: {}", myexp.pdf(0.7));
        println!("cdf: {}", myexp.cdf(0.7));
        println!("per: {}", myexp.per(0.1));
        println!("ran: {}", myexp.ran());
        println!("ranvec: {:?}", myexp.ranvec(5));
        println!("mean: {}", myexp.mean());
        println!("var: {}", myexp.var());
        println!("sd: {}", myexp.sd());
        println!("---------------------");

        println!("\nF Dist");
        println!("pdf: {}", f_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", f_cdf(0.7, 1.5, 4.5));
        println!("per: {}", f_per(0.1, 1.5, 4.5));
        println!("ran: {}", f_ran(1.5, 4.5));
        println!("ran_vec: {:?}", f_ranvec(3, 1.5, 4.5));
        let mut mydist = FDist { nu1: 1.5, nu2: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.1));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nGamma Dist");
        println!("pdf: {}", gamma_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", gamma_cdf(0.7, 1.5, 4.5));
        println!("per: {}", gamma_per(0.2, 500.0, 0.5));
        println!("ran: {}", gamma_ran(1.5, 4.5));
        println!("ran_vec: {:?}", gamma_ranvec(3, 1.5, 4.5));
        let mut mydist = GammaDist { alpha: 1.5, beta: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nLog-Normal");
        println!("pdf: {}", lognormal_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", lognormal_cdf(0.7, 1.5, 4.5));
        println!("per: {}", lognormal_per(0.1, 1.5, 4.5));
        println!("ran: {}", lognormal_ran(1.5, 4.5));
        println!("ran_vec: {:?}", lognormal_ranvec(3, 1.5, 5.5));
        let mut mydist = LogNormalDist { mu: 1.5, sigma: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nNormal");
        println!("pdf: {}", normal_pdf(0.7, 1.5, 4.5));
        println!("cdf: {}", normal_cdf(0.7, 1.5, 4.5));
        println!("per: {}", normal_per(0.1, 1.5, 4.5));
        println!("ran: {}", normal_ran(1.5, 4.5));
        println!("ran_vec: {:?}", normal_ranvec(3, 1.5, 4.5));
        let mut mydist = NormalDist { mu: 1.5, sigma: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nt Dist");
        println!("pdf: {}", t_pdf(0.7, 4.5));
        println!("cdf: {}", t_cdf(0.7, 4.5));
        println!("per: {}", t_per(0.1, 4.5));
        println!("ran: {}", t_ran(4.5));
        println!("ran_vec: {:?}", t_ranvec(3, 4.5));
        let mut mydist = TDist { nu: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nUnif");
        println!("pdf: {}", unif_pdf(1.7, 1.5, 4.5));
        println!("cdf: {}", unif_cdf(1.7, 1.5, 4.5));
        println!("per: {}", unif_per(0.1, 1.5, 4.5));
        println!("ran: {}", unif_ran(1.5, 4.5));
        println!("ran_vec: {:?}", unif_ranvec(3, 1.5, 4.5));
        let mut mydist = UnifDist { a: 1.5, b: 4.5 };
        println!("pdf: {}", mydist.pdf(0.7));
        println!("cdf: {}", mydist.cdf(0.7));
        println!("per: {}", mydist.per(0.99));
        println!("ran: {}", mydist.ran());
        println!("ranvec: {:?}", mydist.ranvec(5));
        println!("mean: {}", mydist.mean());
        println!("var: {}", mydist.var());
        println!("sd: {}", mydist.sd());
        println!("---------------------");

        println!("\nBinomial");
        println!("bin_pmf: {}", bin_pmf(99, 100, 0.9));
        println!("bin_cdf: {}", bin_cdf(99, 100, 0.9));
        println!("bin_per: {}", bin_per(1.0, 100, 0.9));
        println!("bin_ran: {}", bin_ran(100, 0.9));
        println!("ran_vec: {:?}", bin_ranvec(3, 100, 0.9));
        let mut mybin = BinDist { n: 100, p: 0.9 };
        println!("bin pmf: {}", mybin.pmf(80));
        println!("bin cdf: {}", mybin.cdf(80));
        println!("bin per: {}", mybin.per(0.99));
        println!("bin ran: {}", mybin.ran());
        println!("ranvec: {:?}", mybin.ranvec(5));
        println!("bin mean: {}", mybin.mean());
        println!("bin var: {}", mybin.var());
        println!("bin sd: {}", mybin.sd());

        println!("\nGeo");
        println!("geo_pmf: {}", geo_pmf(0, 0.05));
        println!("geo_cdf: {}", geo_cdf(0, 0.05));
        println!("geo_per: {}", geo_per(1.0, 0.05));
        println!("geo_ran: {}", geo_ran(0.05));
        println!("ran_vec: {:?}", geo_ranvec(3, 0.05));
        let mut mygeo = GeoDist { p: 0.05 };
        println!("geo pmf: {}", mygeo.pmf(2));
        println!("geo cdf: {}", mygeo.cdf(2));
        println!("geo per: {}", mygeo.per(0.99));
        println!("geo ran: {}", mygeo.ran());
        println!("ranvec: {:?}", mygeo.ranvec(5));
        println!("geo mean: {}", mygeo.mean());
        println!("geo var: {}", mygeo.var());
        println!("geo sd: {}", mygeo.sd());


        println!("\nGeo2");
        println!("geo2_pmf: {}", geo2_pmf(1, 0.05));
        println!("geo2_cdf: {}", geo2_cdf(1, 0.05));
        println!("geo2_per: {}", geo2_per(0.99, 0.05));
        println!("geo2_ran: {}", geo2_ran(0.05));
        println!("ran_vec: {:?}", geo2_ranvec(3, 0.05));
        let mut mygeo2 = Geo2Dist { p: 0.05 };
        println!("geo2 pmf: {}", mygeo2.pmf(2));
        println!("geo2 cdf: {}", mygeo2.cdf(2));
        println!("geo2 per: {}", mygeo2.per(0.99));
        println!("geo2 ran: {}", mygeo2.ran());
        println!("ranvec: {:?}", mygeo2.ranvec(5));
        println!("geo2 mean: {}", mygeo2.mean());
        println!("geo2 var: {}", mygeo2.var());
        println!("geo2 sd: {}", mygeo2.sd());


        println!("\nHG");
        println!("hg_pmf: {}", hg_pmf(80, 100, 1000, 900));
        println!("hg_cdf: {}", hg_cdf(80, 100, 1000, 900));
        println!("hg_per: {}", hg_per(0.26, 100, 1000, 900));
        println!("hg_ran: {}", hg_ran(100, 1000, 900));
        println!("ran_vec: {:?}", hg_ranvec(3, 100, 1000, 900));
        let mut myhg = HGDist { n: 100, N: 1000, M: 900 };
        println!("hg pmf: {}", myhg.pmf(80));
        println!("hg cdf: {}", myhg.cdf(80));
        println!("hg per: {}", myhg.per(0.26));
        println!("hg ran: {}", myhg.ran());
        println!("ranvec: {:?}", myhg.ranvec(5));
        println!("hg mean: {}", myhg.mean());
        println!("hg var: {}", myhg.var());
        println!("hg sd: {}", myhg.sd());


        println!("\nNB");
        println!("NB_pmf: {}", nb_pmf(10, 5, 0.1));
        println!("NB_cdf: {}", nb_cdf(10, 5, 0.1));
        println!("NB_per: {}", nb_per(0.0, 5, 0.1));
        println!("NB_ran: {}", nb_ran(5, 0.1));
        println!("ran_vec: {:?}", nb_ranvec(3, 5, 0.1));
        let mut mynb = NBDist { r: 5, p: 0.1 };
        println!("NB pmf: {}", mynb.pmf(2));
        println!("NB cdf: {}", mynb.cdf(2));
        println!("NB per: {}", mynb.per(0.99));
        println!("NB ran: {}", mynb.ran());
        println!("ranvec: {:?}", mynb.ranvec(5));
        println!("NB mean: {}", mynb.mean());
        println!("NB var: {}", mynb.var());
        println!("NB sd: {}", mynb.sd());


        println!("\nNB2");
        println!("NB2_pmf: {}", nb2_pmf(10, 5, 0.1));
        println!("NB2_cdf: {}", nb2_cdf(10, 5, 0.1));
        println!("NB2_per: {}", nb2_per(1.0, 5, 0.1));
        println!("NB2_ran: {}", nb2_ran(5, 0.1));
        println!("ran_vec: {:?}", nb2_ranvec(3, 5, 0.1));
        let mut mynb2 = NB2Dist { r: 5, p: 0.1 };
        println!("NB2 pmf: {}", mynb2.pmf(2));
        println!("NB2 cdf: {}", mynb2.cdf(2));
        println!("NB2 per: {}", mynb2.per(0.99));
        println!("NB2 ran: {}", mynb2.ran());
        println!("ranvec: {:?}", mynb2.ranvec(5));
        println!("NB2 mean: {}", mynb2.mean());
        println!("NB2 var: {}", mynb2.var());
        println!("NB2 sd: {}", mynb2.sd());


        println!("\nPois");
        println!("pois_pmf: {}", pois_pmf(3, 2.0));
        println!("pois_cdf: {}", pois_cdf(3, 2.0));
        println!("pois_per: {}", pois_per(1.0, 2.0));
        println!("pois_ran: {}", pois_ran(2.0));
        println!("ran_vec: {:?}", pois_ranvec(3, 2.0));
        let mut mypois = PoisDist { lambda: 2.0 };
        println!("pois pmf: {}", mypois.pmf(3));
        println!("pois cdf: {}", mypois.cdf(3));
        println!("pois per: {}", mypois.per(0.26));
        println!("pois ran: {}", mypois.ran());
        println!("ranvec: {:?}", mypois.ranvec(5));
        println!("pois mean: {}", mypois.mean());
        println!("pois var: {}", mypois.var());
        println!("pois sd: {}", mypois.sd());

        use crate::erf;
        println!("\nerf: {}", erf(1.0));
        use crate::erf2;
        println!("erf2: {}", erf2(1.0));
    }
}



