# Mixed-Data Nearest-Neighbor CMI (CMIknnMixed) Test

CMIknnMixed extends the kNN-based CMI test of Runge (2018) to data sets that contain both continuous and discrete variables. The implementation follows the variant of Popescu et al. (2024) that builds on the mixed-type Kraskov-style estimator of Mesner & Shalizi (2021), without one-hot encoding the categorical coordinates.

**Intuition.** Naive $k$NN distance counting collapses on tied discrete values, biasing the CMI estimate (Mesner & Shalizi, 2021). Mesner & Shalizi (2021) introduce an adaptive neighbourhood that respects discrete-coordinate equivalence classes; Popescu et al. (2024) study and refine this estimator for mixed CI testing in tigramite, retaining the $Z$-local permutation calibration of Runge (2018).

## Mathematical Formulation

The conditional mutual information

```{math}
I(X; Y \mid Z) = \int p(x, y, z) \log \frac{p(x, y \mid z)}{p(x \mid z)\, p(y \mid z)} \, dx \, dy \, dz
```

equals zero if and only if $X \perp Y \mid Z$ (Cover & Thomas, 2006). The mixed-data variant of the Kraskov estimator (Mesner & Shalizi, 2021) decomposes the joint space into discrete and continuous coordinates and uses an adaptive neighbourhood that handles ties in the discrete coordinates correctly; Popescu et al. (2024) propose the variant adopted here that does not treat categorical variables as numeric. P-values are obtained via the same local-permutation scheme as :doc:`/tests/cmiknn_test`, with $Y$ shuffled within $Z$-neighbourhoods (Runge, 2018; Popescu et al., 2024).

## Assumptions

- **Per-variable type tags.** Each variable must be tagged continuous or discrete via the `data_type` array so the estimator selects the correct neighbourhood logic (Popescu et al., 2024).
- **Discrete cardinality.** Categorical variables with very high cardinality reduce to the continuous case; very low cardinality combined with small sample size yields noisy estimates (Mesner & Shalizi, 2021; Popescu et al., 2024).
- **Sample size.** Like all $k$NN-based estimators, accuracy improves with sample size (Runge, 2018; Popescu et al., 2024).
- **No analytical null distribution.** No general analytical results for the finite-sample or asymptotic distribution of CMI under $H_0$ are known, so calibration relies on local permutation (Popescu et al., 2024).
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

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

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Mesner, O. C., & Shalizi, C. R. (2021). Conditional mutual information estimation for mixed, discrete and continuous data. *IEEE Transactions on Information Theory, 67*(1), 464-484.

Popescu, O.-I., Gerhardus, A., & Runge, J. (2024). Non-parametric conditional independence testing for mixed continuous-categorical variables: a novel method and numerical evaluation. *Proceedings of AAAI 2024*.

Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS 2018)*, 938-947.
