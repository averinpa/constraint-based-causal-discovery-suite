# Multivariate Mixed kNN-CMI (mCMIkNN) Test

mCMIkNN is a mixed-data extension of the $k$-nearest-neighbour conditional mutual information test, distributed as a standalone research codebase rather than via PyPI. The wrapper in `citk` lazily loads the implementation from a local checkout of the `mCMIkNN` repository, so this test is only available when that source tree is present on disk.

## Mathematical Formulation

Like :doc:`/tests/cmiknn_test` and :doc:`/tests/cmiknn_mixed_test`, mCMIkNN estimates the conditional mutual information

```{math}
I(X; Y \mid Z) = \int p(x, y, z) \log \frac{p(x, y \mid z)}{p(x \mid z)\, p(y \mid z)} \, dx \, dy \, dz
```

from the $k$-th nearest-neighbour structure of the data. The mixed-data variant adapts the Kraskov-style estimator (Kraskov et al., 2004) so that ties in discrete coordinates are handled correctly and continuous coordinates retain density-based behaviour. P-values are obtained via a local-permutation procedure analogous to the one used in CMIknn (Runge, 2018).

The exact algorithmic choices (neighbourhood metric, tie-breaking, and permutation strategy) follow the upstream `mCMIkNN` reference implementation; see the upstream repository for the canonical specification.

## Assumptions

- **Local source available**: The wrapper raises a clear `ImportError` if the local `mCMIkNN` repository is not present on disk; clone or build it first.
- **Variable type declarations**: Mixed-data behaviour requires the per-variable type marker exposed via `test_kwargs` (forwarded to the underlying implementation).
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

# Initialize the test (requires local mCMIkNN repo to be present)
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

Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E, 69*(6), 066138.

Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS 2018)*, 938-947.
