# Dummy-Encoded Fisher's Z (DummyFisherZ) Test

DummyFisherZ is an adapter that one-hot encodes discrete variables and runs pairwise :doc:`/tests/fisher_z_test` calls across the encoded columns, then aggregates the per-pair p-values via Fisher's combined probability test (Cover & Thomas, 2006). It is a practical option when discrete variables have moderate cardinality and a linear-on-the-encoded-space dependence structure is plausible (Hotelling, 1953).

**Intuition.** A categorical variable can be represented in a partial-correlation framework by replacing it with $K-1$ indicator columns; conditional independence in the original space then implies (and, under linearity, is implied by) zero partial correlation between the indicator blocks given the conditioning indicator block (Anderson, 2003). Combining the resulting per-indicator p-values via a chi-square-on-log-p aggregation gives a single CI verdict.

## Mathematical Formulation

For each variable $V$ in the input data:

- If $V$ is continuous, it contributes a single column to the expanded design matrix.
- If $V$ is categorical with $K$ levels, it contributes $K - 1$ one-hot indicator columns (drop-first to keep the design matrix full-rank) (Anderson, 2003).

For a triple $(X, Y, Z)$, the adapter runs :doc:`/tests/fisher_z_test` on every $(x_\text{col}, y_\text{col})$ pair drawn from the expanded $X$ and $Y$ blocks, conditioning on the union of expanded $Z$ columns (excluding the test pair itself to avoid degeneracy) (Hotelling, 1953; Anderson, 2003). This yields a set of p-values $\{p_1, \ldots, p_m\}$.

The aggregated p-value uses the standard chi-square-on-log-p combination:

```{math}
T = -2 \sum_{i=1}^{m} \ln p_i \;\sim\; \chi^2_{2m} \quad \text{under the null}
```

provided the input p-values are independent and uniform on $[0, 1]$ under $H_0$ (Cover & Thomas, 2006). A single p-value is then obtained from the upper tail of the $\chi^2_{2m}$ distribution.

## Assumptions

- **Linear partial correlation captures dependence in the encoded space.** As with :doc:`/tests/fisher_z_test`, the test has low power against alternatives that one-hot encoding cannot linearise (Anderson, 2003; Baba et al., 2004).
- **Discrete cardinalities are moderate.** Each additional level adds an indicator column, so high-cardinality discrete variables blow up the per-pair count $m$ and reduce power per pair (Anderson, 2003).
- **Independence of per-pair p-values is approximate.** The chi-square-on-log-p combination assumes independent uniform inputs under $H_0$ (Cover & Thomas, 2006); per-pair p-values from overlapping conditioning sets are not strictly independent, so the combined p-value is approximate.
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

## Code Example

```python
import numpy as np
from citk.tests import DummyFisherZ

# Discrete chain: X -> Z -> Y
n = 600
X = np.random.randint(0, 3, size=n)
Z = (X + np.random.randint(0, 2, size=n)) % 3
Y = (Z + np.random.randint(0, 2, size=n)) % 3
data = np.vstack([X, Y, Z]).T

# Initialize the test
dummy_fisherz_test = DummyFisherZ(data)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = dummy_fisherz_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = dummy_fisherz_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.adapter_tests.DummyFisherZ`.

## References

Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis* (3rd ed.). Wiley-Interscience.

Baba, K., Shibata, R., & Sibuya, M. (2004). Partial correlation and conditional correlation as measures of conditional independence. *Australian & New Zealand Journal of Statistics, 46*(4), 657-664.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

Hotelling, H. (1953). New light on the correlation coefficient and its transforms. *Journal of the Royal Statistical Society: Series B (Methodological), 15*(2), 193-225.
