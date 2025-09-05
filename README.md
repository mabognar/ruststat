# ruststat - Tools for working with many common probability distributions

## Features

- Probability mass function (pmf), probability density function (pdf)
- Cumulative distribution function (cdf)
- Percentiles (inverse cdf)
- Random number generation
- Mean, variance

## Distributions

- Beta
- Chi-square
- Exponential
- F
- Gamma
- Normal
- Log-normal
- Pareto (1 thru 4)
- Student's t
- Continuous uniform
- Binomial
- Geometric (2 parameterizations)
- Hypergeometric
- Negative binomial (2 parameterizations)
- Poisson

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
ruststat = "0.1.5"  # Replace with the latest version
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


