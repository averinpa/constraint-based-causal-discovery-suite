# Multivariate Mixed kNN-CMI (mCMIkNN) Test

mCMIkNN is the kNN-based non-parametric CI test for mixed discrete-continuous data of Hügle et al. (2023). It uses a kNN conditional mutual information estimator as the test statistic and a kNN-based local permutation scheme for p-values; Hügle et al. (2023) prove statistical validity and power, including consistency in constraint-based causal discovery, and report state-of-the-art accuracy at low sample sizes.

`citk` vendors the upstream `indeptests` package under `citk/_vendor/indeptests/` so the test is available out of the box (see `citk/_vendor/NOTICE.md` for the upstream source URL and the vendored revision SHA).

**Intuition.** Like CMIknn (Runge, 2018), mCMIkNN treats CMI as a non-parametric measure of conditional dependence (Cover & Thomas, 2006) and calibrates with local permutations; mCMIkNN's distinguishing choice is a discrete-aware rank transform that preserves ties for categorical variables, providing direct mixed-type support without one-hot encoding (Hügle et al., 2023).

## Mathematical Formulation

mCMIkNN estimates the conditional mutual information

```{math}
I(X; Y \mid Z) = \int p(x, y, z) \log \frac{p(x, y \mid z)}{p(x \mid z)\, p(y \mid z)} \, dx \, dy \, dz
```

from the $k$-th nearest-neighbour structure of the data, adapting the Kraskov-style estimator (Kraskov et al., 2004) so ties in discrete coordinates are handled correctly while continuous coordinates retain density-based behaviour (Hügle et al., 2023). P-values are obtained via the kNN local-permutation procedure of Hügle et al. (2023), which extends the local-permutation scheme of Runge (2018) to the mixed-type setting. The exact algorithmic choices (neighbourhood metric, tie-breaking, and permutation strategy) follow the vendored `mCMIkNN` reference implementation.

## Assumptions

- **Vendored implementation.** No additional installation is required; `citk` ships the upstream `indeptests` package under `citk/_vendor/` (Hügle et al., 2023).
- **Constructor parameters.** Per-test parameters (`kcmi`, `kperm`, `Mperm`, `subsample`, `transform`) can be passed via `test_kwargs` and are forwarded to the underlying `mCMIkNN(...)` constructor; these correspond to estimator and permutation knobs documented in Hügle et al. (2023).
- **Sample size.** kNN-based density estimation needs adequate sample size to be reliable (Kraskov et al., 2004); Hügle et al. (2023) report particular advantages over alternatives in low-sample regimes.
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

## Code Example

```python
import numpy as np
from citk.tests import MCMIknn

# Mixed chain: continuous X -> binary Z -> continuous Y
n = 400
X = np.random.randn(n)
Z = (X > 0).astype(int)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test (uses the vendored mCMIkNN implementation)
mcmiknn_test = MCMIknn(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = mcmiknn_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = mcmiknn_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.nearest_neighbor_tests.MCMIknn`.

## References

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Hügle, J., Hagedorn, C., & Schlosser, R. (2023). A kNN-based non-parametric conditional independence test for mixed data and application in causal discovery. *Proceedings of ECML PKDD 2023*.

Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E, 69*(6), 066138.

Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS 2018)*, 938-947.
