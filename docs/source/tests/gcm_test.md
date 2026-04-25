# Generalised Covariance Measure (GCM) Test

The Generalised Covariance Measure is a general-purpose conditional independence test built around regression residualisation (Shah & Peters, 2020). It regresses each of $X$ and $Y$ on the conditioning set $Z$ using flexible machine-learning predictors, then tests whether the resulting residuals are uncorrelated. By plugging in modern regressors, GCM detects non-linear dependence while preserving exact asymptotic-normal calibration of the test statistic.

The `citk` implementation uses random forest regression by default (via the `pycomets` library) for both nuisance regressions.

## Mathematical Formulation

Let $\hat{f}(z) \approx \mathbb{E}[X \mid Z = z]$ and $\hat{g}(z) \approx \mathbb{E}[Y \mid Z = z]$ denote the nuisance regression estimates. The residuals are

```{math}
r_{X,i} = X_i - \hat{f}(Z_i), \qquad r_{Y,i} = Y_i - \hat{g}(Z_i)
```

and the GCM test statistic is the studentised average of their elementwise product:

```{math}
T_{\mathrm{GCM}} = \frac{\sqrt{n}\, \overline{R}}{\hat{\sigma}_R}, \qquad R_i = r_{X,i} \cdot r_{Y,i}, \quad \overline{R} = \frac{1}{n}\sum_i R_i
```

where $\hat{\sigma}_R$ is the empirical standard deviation of $R$. Under the null $X \perp Y \mid Z$ and a mild rate condition on the nuisance estimates, $T_{\mathrm{GCM}} \xrightarrow{d} \mathcal{N}(0, 1)$, so two-sided p-values are computed from the standard normal (Shah & Peters, 2020).

## Assumptions

- **Consistent nuisance regression**: The nuisance product rate must satisfy $\| \hat{f} - f^* \|_2 \cdot \| \hat{g} - g^* \|_2 = o_P(n^{-1/2})$ for the asymptotic normality to hold; flexible learners like random forests typically meet this in low to moderate $\dim(Z)$.
- **Continuous responses**: Default settings target continuous $X$ and $Y$.
- **Sample size**: Studentised normal calibration requires an adequate sample size for stable variance estimation.

## Code Example

```python
import numpy as np
from citk.tests import GCM

# Non-linear chain: X -> Z -> Y
n = 400
X = np.random.randn(n)
Z = np.sin(X) + 0.2 * np.random.randn(n)
Y = Z**2 + 0.2 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test (uses pycomets random forest regression by default)
gcm_test = GCM(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = gcm_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = gcm_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.ml_based_tests.GCM`.

## References

Shah, R. D., & Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. *The Annals of Statistics, 48*(3), 1514-1538.
