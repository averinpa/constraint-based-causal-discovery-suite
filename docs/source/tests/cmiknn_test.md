# Nearest-Neighbor Conditional Mutual Information (CMIknn) Test

CMIknn is the non-parametric CI test of Runge (2018), based on a $k$-nearest-neighbour estimator of conditional mutual information (CMI) combined with a local-permutation procedure for p-values. Runge (2018) shows that the test reliably generates the null distribution and has higher power than kernel-based tests in lower dimensions and similar power in higher dimensions for smooth nonlinear dependencies.

**Intuition.** Conditional independence is equivalent to zero conditional mutual information, $X \perp Y \mid Z \iff I(X; Y \mid Z) = 0$ (Cover & Thomas, 2006). The CMI is estimated directly from $k$-th-nearest-neighbour distances in the joint space (Kraskov et al., 2004; Runge, 2018), and a $Z$-local permutation generates an empirical null that respects the conditional structure (Runge, 2018).

## Mathematical Formulation

The conditional mutual information is

```{math}
I(X; Y \mid Z) = \int p(x, y, z) \log \frac{p(x, y \mid z)}{p(x \mid z)\, p(y \mid z)} \, dx \, dy \, dz
```

and equals zero exactly when $X \perp Y \mid Z$ (Cover & Thomas, 2006). CMIknn estimates $I(X; Y \mid Z)$ via the Kraskov-style $k$NN entropy estimator (Kraskov et al., 2004) generalised to the conditional setting (Runge, 2018). P-values are computed via a **local permutation test**: for each sample $i$, the $Y_i$ value is shuffled with a $Y_j$ from a sample $j$ whose $Z_j$ falls in a $k$-nearest-neighbour neighbourhood of $Z_i$, preserving the marginal $Z$-conditional distribution while breaking any $X$-$Y$ link (Runge, 2018).

## Assumptions

- **Sample size.** $k$NN-based density estimation requires moderate sample size for reliable behaviour; Runge (2018) reports good calibration in experiments from $n = 50$ up to $n = 2{,}000$ across dimensions up to $10$.
- **Choice of $k$.** Two integers govern the procedure: $k_{\mathrm{CMI}}$ for the CMI estimator and $k_{\mathrm{perm}}$ for the local-permutation neighbourhood (Runge, 2018). Larger $k$ smooths the estimator at the cost of bias.
- **Continuous variables.** Vanilla CMIknn is designed for continuous data; for mixed types use :doc:`/tests/cmiknn_mixed_test` or :doc:`/tests/mcmiknn_test` (Popescu et al., 2024; Hügle et al., 2023).
- **Cost.** Runtime scales as $O(B n \log n)$ for $B$ permutations; CMIknn is faster than RFF kernel tests for smaller $n$ but its runtime grows more sharply with $n$ and dimensionality (Runge, 2018).
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

## Code Example

```python
import numpy as np
from citk.tests import CMIknn

# Non-linear chain: X -> Z -> Y
n = 300
X = np.random.randn(n)
Z = np.tanh(X) + 0.2 * np.random.randn(n)
Y = Z**2 + 0.2 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test
cmiknn_test = CMIknn(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = cmiknn_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = cmiknn_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.nearest_neighbor_tests.CMIknn`.

## References

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Hügle, J., Hagedorn, C., & Schlosser, R. (2023). A kNN-based non-parametric conditional independence test for mixed data and application in causal discovery. *Proceedings of ECML PKDD 2023*.

Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E, 69*(6), 066138.

Popescu, O.-I., Gerhardus, A., & Runge, J. (2024). Non-parametric conditional independence testing for mixed continuous-categorical variables: a novel method and numerical evaluation. *Proceedings of AAAI 2024*.

Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS 2018)*, 938-947.
