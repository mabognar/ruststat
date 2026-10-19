#![doc = include_str!("../README.md")]

// continuous
mod beta;
mod burr;
mod cauchy;
mod chisq;
mod exp;
mod f;
mod frechet;
mod gamma;
mod gumbel;
mod invgamma;
mod invgauss;
mod laplace;
mod logistic;
mod loglogistic;
mod lognormal;
mod normal;
mod pareto;
mod rayleigh;
mod t;
mod triangular;
mod unif;
mod weibull;

// discrete
mod bernoulli;
mod bin;
mod discrete_unif;
mod geo;
mod geo2;
mod hg;
mod nb;
mod nb2;
mod pois;
mod zinb;
mod zip;

// export Continuous
pub use beta::*;
pub use burr::*;
pub use cauchy::*;
pub use chisq::*;
pub use exp::*;
pub use f::*;
pub use frechet::*;
pub use gamma::*;
pub use gumbel::*;
pub use invgamma::*;
pub use invgauss::*;
pub use laplace::*;
pub use logistic::*;
pub use loglogistic::*;
pub use lognormal::*;
pub use normal::*;
pub use pareto::*;
pub use rayleigh::*;
pub use t::*;
pub use triangular::*;
pub use unif::*;
pub use weibull::*;

// export Discrete
pub use bernoulli::*;
pub use bin::*;
pub use discrete_unif::*;
pub use geo::*;
pub use geo2::*;
pub use hg::*;
pub use nb::*;
pub use nb2::*;
pub use pois::*;
pub use zinb::*;
pub use zip::*;



/////////////////////////////////
// FACTORIALS AND COMBINATIONS //
/////////////////////////////////

/// Computes the factorial of a given unsigned integer `x!`.
///
/// # Note
/// Uses an iterative approach. Will overflow `u64` capacity if `x > 20`.
///
/// # Parameters
/// - `x = 0, 1, 2, ...`
///
/// # Example
/// ```
/// use ruststat::fact_u;
/// println!("5! is: {}", fact_u(5));
/// ```
pub fn fact_u(x: u64) -> u64 {
    let mut res = 1;
    for i in 2..=x {
        res *= i;
    }
    res
}


/// Computes the log-factorial of a given unsigned integer `x!`.
///
/// # Note
/// Uses an iterative approach. Will overflow `u64` capacity if `x > 20`.
///
/// # Parameters
/// - `x = 0, 1, 2, ...`
///
/// # Example
/// ```
/// use ruststat::factln_u;
/// println!("5! is: {}", factln_u(5));
/// ```
pub fn factln_u(x: u64) -> f64 {
    gamma_ln(x as f64 + 1.0)
}


/// Computes the generalized factorial (i.e. Gamma function) for floating-point numbers
///
/// # Parameters
/// - `x > 0.0`
///
/// # Example
/// ```
/// use ruststat::fact_f;
/// println!("Generalized factorial of 4.5: {}", fact_f(4.5));
/// ```
pub fn fact_f(x: f64) -> f64 {
    gamma(x + 1.0)
}


/// Computes the generalized log-factorial for floating-point numbers using the Gamma function
///
/// # Parameters
/// - `x > 0.0`
///
/// # Example
/// ```
/// use ruststat::factln_f;
/// println!("Generalized factorial of 4.5: {}", factln_f(4.5));
/// ```
pub fn factln_f(x: f64) -> f64 {
    gamma_ln(x + 1.0)
}


/// Computes the number of combinations "n choose x" `C(n, x)`.
///
/// # Parameters
/// - `n = 0, 1, 2, ...`
/// - `x = 0, 1, 2, ...`
/// - `n >= x`
///
/// # Example
/// ```
/// use ruststat::comb;
/// println!("10 choose 3: {}", comb(10, 3));
/// ```
pub fn comb(n: u64, x: u64) -> f64 {
    (0.5 + (factln_u(n) - factln_u(x) - factln_u(n-x)).exp()).floor()
}


/// Computes the natural logarithm of the combinations function `ln(C(n, x))`.
///
/// # Parameters
/// - `n = 0, 1, 2, ...`
/// - `x = 0, 1, 2, ...`
/// - `n >= x`
///
/// # Example
/// ```
/// use ruststat::combln;
/// println!("Log 10 choose 3: {}", combln(10, 3));
/// ```
pub fn combln(n: u64, x: u64) -> f64 {
    factln_u(n) - factln_u(x) - factln_u(n-x)
}



/////////////////////
// GAMMA FUNCTIONS //
/////////////////////

/// Computes the Gamma function `Γ(x)`, an extension of the factorial function.
///
/// # Parameters
/// - `x > 0.0`
///
/// # Example
/// ```
/// use ruststat::gamma;
/// println!("Gamma(5.0): {}", gamma(5.0)); // Evaluates to 4! = 24.0
/// ```
pub fn gamma(x: f64) -> f64 {
    gamma_ln(x).exp()
}

/// Computes the natural logarithm of the Gamma function `ln(Γ(x))`.
///
/// # Parameters
/// - `x > 0.0`
///
/// # Example
/// ```
/// use ruststat::gamma_ln;
/// println!("Log Gamma(5.0): {}", gamma_ln(5.0));
/// ```
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

/// Computes the inverse of the lower incomplete gamma function.
///
/// # Parameters
/// - `p`: Probability (`0.0 <= p <= 1.0`)
/// - `alpha > 0.0`
///
/// # Example
/// ```
/// use ruststat::gammai_inv;
/// println!("Inverse incomplete gamma: {}", gammai_inv(0.5, 2.0));
/// ```
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


/// Computes the lower incomplete gamma function `P(a, x)`.
///
/// # Parameters
/// - `x >= 0.0`
/// - `a > 0.0`
///
/// # Example
/// ```
/// use ruststat::gammp;
/// println!("Lower incomplete gamma P(1.5, 2.0): {}", gammp(1.5, 2.0));
/// ```
pub fn gammp(x: f64, a: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        println!("Invalid arguments in routine gammp");
        return f64::NAN;
    }
    if x == 0.0 {
        0.0
    } else if a >= 100.0 {
        return 1.0 - gammapprox(x, a);
    } else if x < (a + 1.0) {
        return gser(x, a);
    } else {
        return 1.0 - gcf(x, a);
    }
}


/// Computes the upper (complementary) incomplete gamma function `Q(a, x) = 1 - P(a, x)`.
///
/// # Parameters
/// - `x >= 0.0`
/// - `a > 0.0`
///
/// # Example
/// ```
/// use ruststat::gammq;
/// println!("Complementary incomplete gamma Q(1.5, 2.0): {}", gammq(1.5, 2.0));
/// ```
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


/// Computes the incomplete gamma function using a series representation.
///
/// # Note
/// This is typically used as a helper function when `x < a + 1.0` to ensure convergence.
/// For general probability evaluation, consider using `gammp`.
///
/// # Parameters
/// - `x >= 0.0`
/// - `a > 0.0`
///
/// # Example
/// ```
/// use ruststat::gser;
/// println!("Incomplete gamma (series): {}", gser(1.5, 2.0));
/// ```
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


/// Computes the complementary (upper) incomplete gamma function using a continued fraction representation.
///
/// # Note
/// This is typically used as a helper function when `x >= a + 1.0` to ensure rapid convergence.
/// For general probability evaluation, consider using `gammq`.
///
/// # Parameters
/// - `x >= 0.0`
/// - `a > 0.0`
///
/// # Example
/// ```
/// use ruststat::gcf;
/// println!("Complementary incomplete gamma (continued fraction): {}", gcf(1.5, 2.0));
/// ```
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


/// Computes the natural logarithm of the complementary (upper) incomplete gamma function using a continued fraction representation.
///
/// # Note
/// This is the logarithmic version of `gcf`. It is typically used as a helper function
/// for numerical stability when evaluating log-probabilities or working with extreme values
/// that might otherwise cause floating-point underflow or overflow.
///
/// # Parameters
/// - `x >= 0.0`
/// - `a > 0.0`
///
/// # Example
/// ```
/// use ruststat::gcf_ln;
/// println!("Log complementary incomplete gamma (continued fraction): {}", gcf_ln(1.5, 2.0));
/// ```
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



////////////////////
// BETA FUNCTIONS //
////////////////////

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
    h
}


/// Computes the Beta function `B(a, b)`.
///
/// # Parameters
/// - `a > 0.0`
/// - `b > 0.0`
///
/// # Example
/// ```
/// use ruststat::beta_fn;
/// println!("Beta function B(2.0, 3.0): {}", beta_fn(2.0, 3.0));
/// ```
pub fn beta_fn(a: f64, b:f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a+b)).exp()
}


/// Computes the natural logarithm of the Beta function `ln(B(a, b))`.
///
/// # Parameters
/// - `a > 0.0`
/// - `b > 0.0`
///
/// # Example
/// ```
/// use ruststat::beta_fn_ln;
/// println!("Log Beta function: {}", beta_fn_ln(2.0, 3.0));
/// ```
pub fn beta_fn_ln(a: f64, b:f64) -> f64 {
    gamma_ln(a) + gamma_ln(b) - gamma_ln(a+b)
}


/// Computes the inverse of the incomplete beta function.
///
/// # Parameters
/// - `p`: Probability (`0.0 <= p <= 1.0`)
/// - `a > 0.0`
/// - `b > 0.0`
///
/// # Example
/// ```
/// use ruststat::betai_inv;
/// println!("Inverse incomplete beta: {}", betai_inv(0.5, 2.0, 3.0));
/// ```
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



/////////////////////
// ERROR FUNCTIONS //
/////////////////////

/// Computes the error function `erf(x)` using a Chebyshev approximation.
///
/// # Parameters
/// - `-infinity < x < infinity`
///
/// # Example
/// ```
/// use ruststat::erf;
/// println!("erf(1.0): {}", erf(1.0));
/// ```
pub fn erf(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 - erfc_cheb(x)
    } else {
        erfc_cheb(-x) - 1.0
    }
}


/// Computes the inverse of the error function.
///
/// # Parameters
/// - `p`: Probability (`0.0 < p < 1.0`)
///
/// # Example
/// ```
/// use ruststat::erf_inv;
/// println!("Inverse erf(0.5): {}", erf_inv(0.5));
/// ```

pub fn erf_inv(p: f64) -> f64 {
    erfc_inv(1.0 - p)
}


/// Error function (gammp implementation) `Erf(x)`
pub fn erf2(x: f64) -> f64 {
    if x < 0.0 {
        -gammp(x * x, 0.5)
    } else {
        gammp(x * x, 0.5)
    }
}


/// Computes the complementary error function `erfc(x) = 1 - erf(x)`.
///
/// # Parameters
/// - `-infinity < x < infinity`
///
/// # Example
/// ```
/// use ruststat::erfc;
/// println!("erfc(1.0): {}", erfc(1.0));
/// ```
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        erfc_cheb(x)
    } else {
        2.0 - erfc_cheb(-x)
    }
}


/// Computes the inverse of the complementary error function.
///
/// # Parameters
/// - `p`: Probability (`0.0 < p < 2.0`)
///
/// # Example
/// ```
/// use ruststat::erfc_inv;
/// println!("Inverse erfc(0.5): {}", erfc_inv(0.5));
/// ```
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



//////////////////////////
// Sample mean, sd, var //
//////////////////////////

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


