# Mixed-Data Nearest-Neighbor CMI (CMIknnMixed) Test

CMIknnMixed extends the kNN-based conditional mutual information test to data sets that contain both continuous and discrete variables. It uses a mixed-type CMI estimator that handles ties correctly when discrete values repeat, while still detecting general non-linear dependencies in the continuous coordinates (Mesner & Shalizi, 2021). It is part of the `tigramite` library and is preferred over the continuous-only :doc:`/tests/cmiknn_test` when any of $X$, $Y$, or $Z$ is discrete.

## Mathematical Formulation

The estimator decomposes the joint space into discrete and continuous coordinates. For mixed data, naive $k$NN distance counting collapses on tied discrete values; the mixed-data variant of the Kraskov estimator (Mesner & Shalizi, 2021) uses an adaptive neighbourhood that respects the discrete-coordinate equivalence classes. The conditional mutual information

```{math}
I(X; Y \mid Z) = \int p(x, y, z) \log \frac{p(x, y \mid z)}{p(x \mid z)\, p(y \mid z)} \, dx \, dy \, dz
```

is then estimated via differences of digamma evaluations on the resulting neighbourhood counts, generalising the continuous Kraskov estimator (Kraskov et al., 2004) to the mixed setting.

P-values are computed via the same local-permutation scheme as :doc:`/tests/cmiknn_test`, with $Y$ values shuffled within $Z$-neighbourhoods (Runge, 2018).

## Assumptions

- **Variable type declarations**: Each variable must be tagged as continuous or discrete via the `data_type` array so the estimator selects the correct neighbourhood logic.
- **Discrete cardinality**: Many discrete variables with very high cardinality reduce to the continuous case; very low cardinality combined with small sample size yields noisy estimates.
- **Sample size**: As with all $k$NN-based estimators, accuracy improves with sample size.

## Code Example

```python
import numpy as np
from citk.tests import CMIknnMixed

# Mixed chain: continuous X -> binary Z -> continuous Y
n = 400
X = np.random.randn(n)
Z = (X > 0).astype(int)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Declare per-variable types: 0 = continuous, 1 = discrete
data_type = np.array([[0, 0, 1]])

# Initialize the test
cmiknn_mixed_test = CMIknnMixed(data, data_type=data_type)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = cmiknn_mixed_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = cmiknn_mixed_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.nearest_neighbor_tests.CMIknnMixed`.

## References

Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E, 69*(6), 066138.

Mesner, O. C., & Shalizi, C. R. (2021). Conditional mutual information estimation for mixed, discrete and continuous data. *IEEE Transactions on Information Theory, 67*(1), 464-484.

Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS 2018)*, 938-947.
