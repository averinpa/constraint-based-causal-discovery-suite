# Projected Covariance Measure (PCM) Test

The Projected Covariance Measure of Lundborg et al. (2024) is an assumption-lean test for the model-free null of conditional mean independence — that the conditional mean of $Y$ given $(X, Z)$ does not depend on $X$. It addresses the principal weakness of :doc:`/tests/gcm_test` (Shah & Peters, 2020): GCM has reduced power when $X$ is involved in complex interactions or when the predictor of interest is weakly identified by the nuisance regressions. Lundborg et al. (2024) prove that a spline-regression instance of their procedure attains the minimax optimal rate in this nonparametric testing problem.

The `citk` implementation uses sample-splitting with random forest regression by default (via the `pycomets` library).

**Intuition.** Rather than testing zero residual covariance directly, PCM first uses one half of the data to estimate a *projection* of $Y$ on $(X, Z)$ — typically the regression $\hat{h}(X, Z) \approx \mathbb{E}[Y \mid X, Z]$ — and then on the other half tests the expected conditional covariance between this projection and $Y$, after adjusting both for $Z$ (Lundborg et al., 2024). The procedure inherits robust Type I error control from the orthogonality of the residualisation step and gains power from the data-driven projection (Lundborg et al., 2024).

## Mathematical Formulation

PCM splits the data into two folds. On the first fold it learns a projection function $\hat{h}(X, Z)$, typically $\hat{h}(X, Z) \approx \mathbb{E}[Y \mid X, Z]$ (Lundborg et al., 2024). On the second fold it residualises both $\hat{h}(X, Z)$ and $Y$ against $Z$ via nuisance regressions $\hat{m}_h$ and $\hat{m}_Y$:

```{math}
\tilde{h}_i = \hat{h}(X_i, Z_i) - \hat{m}_h(Z_i), \qquad \tilde{Y}_i = Y_i - \hat{m}_Y(Z_i)
```

and forms the studentised covariance test statistic (Lundborg et al., 2024):

```{math}
T_{\mathrm{PCM}} = \frac{\sqrt{n_2}\, \overline{R}}{\hat{\sigma}_R}, \qquad R_i = \tilde{h}_i \cdot \tilde{Y}_i
```

Under the null, $T_{\mathrm{PCM}} \xrightarrow{d} \mathcal{N}(0, 1)$ under nuisance rate conditions analogous to GCM's (Lundborg et al., 2024). Crucially, validity of the test does not require $\hat{h}$ to be a good predictor of $Y$ — only that the residualisation step is consistent — so PCM remains assumption-lean (Lundborg et al., 2024).

## Assumptions

- **Conditional mean independence null.** PCM tests $\mathbb{E}[Y \mid X, Z] = \mathbb{E}[Y \mid Z]$ (i.e. the conditional mean of $Y$ does not depend on $X$ given $Z$), not full conditional independence (Lundborg et al., 2024).
- **Consistent residualisation.** $\hat{m}_h$ and $\hat{m}_Y$ must converge fast enough for studentised normal calibration; flexible learners with sample splitting typically suffice (Lundborg et al., 2024).
- **Variable types.** Random forest nuisance regressions handle continuous, discrete, or mixed $X$, $Y$, and $Z$ natively; no separate type declaration is required (Shah & Peters, 2020; Lundborg et al., 2024).
- **Sample size.** Both folds need to be large enough for the projection step on the first fold and the test statistic on the second fold (Lundborg et al., 2024).
- **Optimality.** A spline-regression version achieves the minimax optimal rate for this nonparametric testing problem (Lundborg et al., 2024).

## v0.1.0 implementation notes

The pycomets backend, regressor (`RandomForestRegressor`), the projection estimator, and the sample-splitting fold count are **not surfaced as constructor kwargs in v0.1.0**. Future minor versions may add explicit kwargs additively. **Empty conditioning set** is handled by substituting a constant column $Z = 0$, as in :doc:`/tests/gcm_test`.

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
