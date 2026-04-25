# Regression-Based Conditional Independence (RegressionCI) Test

The RegressionCI test is a parametric likelihood-ratio test that uses regression models to assess conditional independence, with native support for mixed continuous and discrete data. It is the regression-family CI test exposed by the `tigramite` library (Runge et al., 2019) and is well-suited to settings where a parametric model class (linear, logistic) is plausible for each variable.

## Mathematical Formulation

For a hypothesis $X \perp Y \mid Z$, RegressionCI compares two nested regression models for the response $Y$:

```{math}
\mathcal{M}_0:\; Y = f(Z) + \epsilon_0, \qquad \mathcal{M}_1:\; Y = g(Z, X) + \epsilon_1
```

The link function $f, g$ is chosen by the response type: linear regression for continuous responses, logistic regression for discrete responses. Conditional independence is tested by the likelihood-ratio statistic comparing the two fits:

```{math}
\Lambda = -2 \left[ \ell(\mathcal{M}_0) - \ell(\mathcal{M}_1) \right]
```

where $\ell(\cdot)$ is the maximized log-likelihood. Under the null $X \perp Y \mid Z$ and standard regularity conditions, $\Lambda$ is asymptotically $\chi^2$-distributed with degrees of freedom equal to the number of parameters added in $\mathcal{M}_1$ over $\mathcal{M}_0$ (Wilks, 1938).

## Assumptions

- **Correctly specified model class**: Linear or logistic links must be a reasonable approximation of the conditional mean / log-odds.
- **Mixed data support**: Each variable's type (continuous vs. discrete) must be declared via the `data_type` argument so the appropriate link is selected per regression.
- **Asymptotic regime**: P-values rely on the chi-square approximation, which requires a sufficiently large sample for stable behaviour.

## Code Example

```python
import numpy as np
from citk.tests import RegressionCI

# Continuous chain: X -> Z -> Y
n = 400
X = np.random.randn(n)
Z = 0.8 * X + 0.5 * np.random.randn(n)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test (continuous variables by default)
regci_test = RegressionCI(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = regci_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = regci_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.regression_tests.RegressionCI`.

## References

Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances, 5*(11), eaau4996.

Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *The Annals of Mathematical Statistics, 9*(1), 60-62.
