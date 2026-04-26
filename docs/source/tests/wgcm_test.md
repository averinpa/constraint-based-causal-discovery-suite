# Weighted Generalised Covariance Measure (WGCM) Test

The Weighted GCM is an extension of :doc:`/tests/gcm_test` that improves power against alternatives where the conditional dependence is **localised** in $Z$ — that is, where $X$ and $Y$ are dependent only over a sub-region of the conditioning space (Scheidegger et al., 2022). It adds a sample-splitting step and a learned weighting function that emphasises samples in the dependence region.

The `citk` implementation uses random forest regression by default (via the `pycomets` library) for both the nuisance regressions and the weighting.

## Mathematical Formulation

WGCM splits the data into two folds. On the first fold it learns the nuisance regressions $\hat{f}, \hat{g}$ as in GCM and, in addition, a weighting function $\hat{w}(z)$ that targets regions where the residual product covaries with $z$. On the second fold it computes the weighted residual product

```{math}
R_i = \hat{w}(Z_i) \cdot \bigl(X_i - \hat{f}(Z_i)\bigr) \cdot \bigl(Y_i - \hat{g}(Z_i)\bigr)
```

and forms the studentised mean test statistic

```{math}
T_{\mathrm{WGCM}} = \frac{\sqrt{n_2}\, \overline{R}}{\hat{\sigma}_R}
```

where $n_2$ is the size of the second fold. Under the null $X \perp Y \mid Z$, the same rate condition as GCM gives $T_{\mathrm{WGCM}} \xrightarrow{d} \mathcal{N}(0, 1)$ (Scheidegger et al., 2022). When the dependence is localised, the weights amplify the relevant region and the test gains power against the unweighted GCM.

## Assumptions

- **Consistent nuisance regression**: Same product-rate requirement as GCM on $\hat{f}$ and $\hat{g}$.
- **Useful weight learning**: The weighting function provides power gains only when the dependence has localisable structure; on globally constant alternatives, WGCM reduces to GCM with extra variance from sample splitting.
- **Variable types**: Random forest nuisance regressions handle continuous, discrete, or mixed $X$, $Y$, and $Z$ natively; no separate type declaration is required.

## Code Example

```python
import numpy as np
from citk.tests import WGCM

# Non-linear chain with localised dependence on Z
n = 500
X = np.random.randn(n)
Z = X + 0.5 * np.random.randn(n)
Y = (Z > 0) * Z**2 + 0.3 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test (uses pycomets random forest regression and weighting)
wgcm_test = WGCM(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = wgcm_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = wgcm_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.ml_based_tests.WGCM`.

## References

Scheidegger, C., Hörrmann, J., & Bühlmann, P. (2022). The weighted generalised covariance measure. *Journal of Machine Learning Research, 23*(273), 1-68.

Shah, R. D., & Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. *The Annals of Statistics, 48*(3), 1514-1538.
