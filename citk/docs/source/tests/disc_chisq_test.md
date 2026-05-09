# Discretised Chi-Squared (DiscChiSq) Test

DiscChiSq is an adapter that applies equal-frequency discretisation to continuous variables before running the standard :doc:`/tests/chi_sq_test` (Agresti, 2013). It provides a practical baseline for applying classical contingency-table CI tests to continuous or mixed data, at the cost of the information loss inherent in binning.

**Intuition.** Once continuous variables are mapped to discrete bins, CI testing reduces to verifying the factorisation of a finite joint probability table — the setting Pearson's $\chi^2$ test was designed for (Agresti, 2013). The trade-off is that the null hypothesis being tested is no longer conditional independence in the continuous space, but rather independence between discretised representations of the variables.

## Mathematical Formulation

For each continuous column $V$ of the input data, the adapter produces a discrete version

```{math}
\widetilde{V}_i = \mathrm{quantile\_bin}(V_i; b)
```

where `quantile_bin` assigns each observation to one of $b$ equal-frequency bins (i.e., bins defined by the empirical $1/b, 2/b, \ldots, (b-1)/b$ quantiles). Columns that are already categorical are left unchanged. The Pearson chi-squared statistic is then computed on the binned data (Agresti, 2013):

```{math}
\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}
```

where $O_i$ and $E_i$ are the observed and expected cell counts in the binned contingency table under the null $X \perp Y \mid Z$ (Agresti, 2013). Under the null and standard regularity conditions, the statistic is asymptotically $\chi^2$-distributed with degrees of freedom matching the binned cardinalities (Agresti, 2013); see :doc:`/tests/chi_sq_test`.

## Assumptions

- **Discretisation preserves dependence.** The number of bins must be coarse enough to keep cells populated, but fine enough that the underlying dependence is still detectable after binning (Agresti, 2013).
- **Independent observations.** Same multinomial sampling assumption as Pearson's $\chi^2$ (Agresti, 2013).
- **Adequate cell counts.** Cochran (1954) recommends that the test may be unreliable if more than 20% of cells have an expected frequency below 5 or any cell has expected frequency below 1; the same rule applies after binning (Agresti, 2013).
- **Approximate CI procedure.** Discretisation tests independence between binned representations rather than the underlying continuous CI null; CI verdicts should be treated as approximate (Agresti, 2013).
- **Dtype validation is opt-in.** Passing data outside the declared dtype produces undefined results; call `Test.validate_data(data)` to check. citk does not enforce ``supported_dtypes`` at construction.

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

Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.

Cochran, W. G. (1954). Some methods for strengthening the common $\chi^2$ tests. *Biometrics, 10*(4), 417-451.
