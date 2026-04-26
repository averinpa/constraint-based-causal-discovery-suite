# Projected Covariance Measure (PCM) Test

The Projected Covariance Measure is an assumption-lean conditional independence test that addresses the principal weakness of :doc:`/tests/gcm_test`: GCM has trivial power when the predictor of interest is **weakly identified** by the nuisance regressions (Lundborg et al., 2024). PCM constructs a data-driven projection of $Y$ onto a direction along which dependence with $X$ given $Z$ can still be detected, restoring power in regimes where GCM is uninformative.

The `citk` implementation uses sample-splitting with random forest regression by default (via the `pycomets` library).

## Mathematical Formulation

PCM splits the data into two folds. On the first fold it learns a **projection function** $\hat{h}(X, Z)$ — typically the regression of $Y$ on $(X, Z)$ — that captures the conditional contribution of $X$ given $Z$. On the second fold it residualises both $\hat{h}(X, Z)$ and $Y$ against $Z$ via nuisance regressions $\hat{m}_h$ and $\hat{m}_Y$:

```{math}
\tilde{h}_i = \hat{h}(X_i, Z_i) - \hat{m}_h(Z_i), \qquad \tilde{Y}_i = Y_i - \hat{m}_Y(Z_i)
```

and forms the studentised covariance test statistic

```{math}
T_{\mathrm{PCM}} = \frac{\sqrt{n_2}\, \overline{R}}{\hat{\sigma}_R}, \qquad R_i = \tilde{h}_i \cdot \tilde{Y}_i
```

Under the null $X \perp Y \mid Z$, $T_{\mathrm{PCM}} \xrightarrow{d} \mathcal{N}(0, 1)$ under nuisance rate conditions analogous to GCM (Lundborg et al., 2024). Crucially, the validity of the test does not depend on the projection $\hat{h}$ being a good predictor of $Y$ — only on the residualisation step being consistent — so PCM remains assumption-lean.

## Assumptions

- **Consistent residualisation**: $\hat{m}_h$ and $\hat{m}_Y$ must converge fast enough for studentised normal calibration; random forests with sample splitting typically suffice.
- **Variable types**: Random forest nuisance regressions handle continuous, discrete, or mixed $X$, $Y$, and $Z$ natively; no separate type declaration is required.
- **Sample size**: Both folds need to be large enough for the projection step on the first fold and the test statistic on the second fold.

## Code Example

```python
import numpy as np
from citk.tests import PCM

# Non-linear chain: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = np.sin(X) + 0.2 * np.random.randn(n)
Y = Z**2 + 0.2 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test (uses pycomets random forest regression with sample splitting)
pcm_test = PCM(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = pcm_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = pcm_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.ml_based_tests.PCM`.

## References

Lundborg, A. R., Kim, I., Shah, R. D., & Samworth, R. J. (2024). The projected covariance measure for assumption-lean variable significance testing. *The Annals of Statistics*, to appear.

Shah, R. D., & Peters, J. (2020). The hardness of conditional independence testing and the generalised covariance measure. *The Annals of Statistics, 48*(3), 1514-1538.
