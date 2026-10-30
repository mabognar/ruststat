# ruststat - Tools for working with probability distributions

## Version 0.4.2

- Added Negative Hypergeometric distribution (2 parameterizations)

## Features

- Probability mass function (pmf), probability density function (pdf)
- Cumulative distribution function (cdf)
- Quantiles/Percentiles (inverse cdf)
- Random number generation
- Mean, variance, and standard deviation

## Distributions

#### Continuous
- Beta
- Burr (Type XII)
- Cauchy
- Chi-square
- Continuous Uniform
- Exponential
- F
- Fréchet
- Gamma
- Gumbel
- Inverse Gamma
- Inverse Gaussian (Wald)
- Laplace (Double Exponential)
- Log-Logistic (Fisk)
- Logistic
- Log-Normal
- Normal
- Pareto (Types I, II, III, IV)
- Rayleigh
- Student's t
- Triangular
- Weibull

#### Discrete
- Bernoulli
- Binomial
- Discrete Uniform
- Geometric (2 parameterizations)
- Hypergeometric
- Negative Binomial (2 parameterizations)
- Negative Hypergeometric (2 parameterizations)
- Poisson
- Zero-Inflated Negative Binomial (ZINB)
- Zero-Inflated Poisson (ZIP)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ruststat = "0.4.2"
```

## Quick Start
```rust
use ruststat::*;
// X~N(mu=0,sigma=1.0), find 97.5th percentile
println!("normal percentile: {}", normal_per(0.975, 0.0, 1.0));
// X~Bin(n=10,p=0.7), compute P(X=4)
println!("binomial probability: {}", bin_pmf(4, 10, 0.7));
```

For convenience, functions can also be accessed via `Structs`.
```rust
use ruststat::*;
// X~Beta(alpha=0.5,beta=2.0)
let mut mybeta = BetaDist{alpha:0.5, beta:2.0};
// 30th percentile
println!("percentile: {}", mybeta.per(0.3));
// P(X <= 0.4)
println!("cdf: {}", mybeta.cdf(0.4));
// Random draw
println!("random draw: {}", mybeta.ran());
// Variance
println!("variance: {}", mybeta.var());
```


