# Multivariate Mixed kNN-CMI (mCMIkNN) Test

mCMIkNN is a mixed-data extension of the $k$-nearest-neighbour conditional mutual information test (Hügle et al., 2023). The upstream implementation is distributed as a standalone research codebase rather than via PyPI; `citk` vendors the relevant `indeptests` package under `citk/_vendor/indeptests/` so the test is available out of the box, with no additional installation step.

## Mathematical Formulation

Like :doc:`/tests/cmiknn_test` and :doc:`/tests/cmiknn_mixed_test`, mCMIkNN estimates the conditional mutual information

```{math}
I(X; Y \mid Z) = \int p(x, y, z) \log \frac{p(x, y \mid z)}{p(x \mid z)\, p(y \mid z)} \, dx \, dy \, dz
```

from the $k$-th nearest-neighbour structure of the data. The mixed-data variant adapts the Kraskov-style estimator (Kraskov et al., 2004) so that ties in discrete coordinates are handled correctly and continuous coordinates retain density-based behaviour. P-values are obtained via a local-permutation procedure analogous to the one used in CMIknn (Runge, 2018).

The exact algorithmic choices (neighbourhood metric, tie-breaking, and permutation strategy) follow the vendored `mCMIkNN` reference implementation; see `citk/_vendor/NOTICE.md` for the upstream source URL and the vendored revision SHA.

## Assumptions

- **Vendored implementation**: No additional installation is required; `citk` ships the upstream `indeptests` package under `citk/_vendor/`.
- **Constructor parameters**: Per-test parameters (`kcmi`, `kperm`, `Mperm`, `subsample`, `transform`) can be passed via `test_kwargs` and are forwarded to the underlying `mCMIkNN(...)` constructor.
- **Sample size**: $k$NN density estimation needs adequate sample size to be reliable.

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

Hügle, J., Hagedorn, C., & Uflacker, M. (2023). A kNN-based non-parametric conditional independence test for mixed data and application in causal discovery. *Proceedings of ECML PKDD 2023*.

Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E, 69*(6), 066138.

Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS 2018)*, 938-947.
