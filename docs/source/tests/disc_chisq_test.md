# Discretised Chi-Squared (DiscChiSq) Test

DiscChiSq is an adapter that applies equal-frequency discretisation to continuous variables before running the standard :doc:`/tests/chi_sq_test`. It provides a practical baseline for applying classical contingency-table tests to continuous or mixed data, at the cost of the information loss inherent in binning.

## Mathematical Formulation

For each continuous column $V$ of the input data, the adapter produces a discrete version

```{math}
\widetilde{V}_i = \mathrm{quantile\_bin}(V_i; b)
```

where `quantile_bin` assigns each observation to one of $b$ equal-frequency bins (i.e., bins defined by the empirical $1/b, 2/b, \ldots, (b-1)/b$ quantiles). Columns that are already categorical are left unchanged. The Pearson chi-squared statistic is then computed on the binned data:

```{math}
\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}
```

where $O_i$ and $E_i$ are the observed and expected cell counts in the binned contingency table under the null $X \perp Y \mid Z$. Under the null and standard regularity conditions, the statistic is asymptotically $\chi^2$-distributed with degrees of freedom matching the binned cardinalities (see :doc:`/tests/chi_sq_test`).

## Assumptions

- **Discretisation preserves dependence**: The chosen number of bins must be coarse enough to keep cells populated, but fine enough that the underlying dependence is still detectable after binning.
- **Independent observations**: Same as Chi-Squared.
- **Adequate cell counts**: As a rule of thumb, the test may be unreliable if more than 20% of cells have an expected frequency below 5 (Cochran, 1954).

## Code Example

```python
import numpy as np
from citk.tests import DiscChiSq

# Continuous chain: X -> Z -> Y
n = 500
X = np.random.randn(n)
Z = 0.8 * X + 0.5 * np.random.randn(n)
Y = 0.8 * Z + 0.5 * np.random.randn(n)
data = np.vstack([X, Y, Z]).T

# Initialize the test with 4 equal-frequency bins per continuous variable
disc_chisq_test = DiscChiSq(data, n_bins=4)

# Test for conditional independence of X and Y given Z
# Expected: p-value is large (cannot reject H0 of independence)
p_value_conditional = disc_chisq_test(0, 1, [2])
print(f"P-value for X _||_ Y | Z: {p_value_conditional:.4f}")

# Test for unconditional independence of X and Y
# Expected: p-value is small (reject H0 of independence)
p_value_unconditional = disc_chisq_test(0, 1)
print(f"P-value for X _||_ Y: {p_value_unconditional:.4f}")
```

## API Reference

For a full list of parameters, see the API documentation: :class:`citk.tests.adapter_tests.DiscChiSq`.

## References

Cochran, W. G. (1954). Some methods for strengthening the common $\chi^2$ tests. *Biometrics, 10*(4), 417-451.

Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. *Philosophical Magazine, 50*(302), 157-175.
